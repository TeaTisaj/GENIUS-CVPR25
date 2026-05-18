"""
Feature extraction using BiomedCLIP (ViT-B/16, 512-dim).

Produces the same .pt embedding files as the CLIP-SF extraction scripts
so Stage 1 (RQ training) can consume them unchanged.

Usage:
    torchrun --nproc_per_node=<N> biomedclip_feature_extraction.py \
        --config_path config_biomedclip_cand.yaml \
        --genir_dir /path/to/GENIUS-CVPR25 \
        --mbeir_data_dir /path/to/mbeir_data
"""

import argparse
import logging
import os
import gc

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
from torch.cuda.amp import autocast
from omegaconf import OmegaConf
from tqdm import tqdm

import common.dist_utils as dist_utils
from data.mbeir_dataset import (
    MBEIRMainDataset,
    MBEIRMainCollator,
    MBEIRCandidatePoolDataset,
    MBEIRCandidatePoolCollator,
    Mode,
)
from models.uniir_clip import utils
from models.biomedclip.biomedclip_nofusion import BiomedCLIPNoFusion

logger = logging.getLogger()


def generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config):
    genir_dir = config.genir_dir
    mbeir_data_dir = config.mbeir_data_dir
    save_embed_path = os.path.join(config.genir_dir, config.model.emb_save_path)

    data_config = config.data_config
    query_instruct_path = data_config.query_instruct_path
    cand_pool_dir = data_config.cand_pool_dir_name
    image_size = tuple(map(int, data_config.image_size.split(",")))

    splits = []
    for split_name in ["train", "val", "test"]:
        split_dir_name = getattr(data_config, f"{split_name}_dir_name", None)
        embed_dataset_config = getattr(config.embed_config, f"{split_name}_datasets_config", None)
        if embed_dataset_config and embed_dataset_config.enable_embed:
            splits.append((
                split_name,
                split_dir_name,
                embed_dataset_config.datasets_name,
                embed_dataset_config.correspond_cand_pools_name,
            ))

    embed_cand_pool_config = config.embed_config.cand_pools_config
    if embed_cand_pool_config and embed_cand_pool_config.enable_embed:
        cand_pool_name_list = embed_cand_pool_config.cand_pools_name_to_embed
        splits.append((
            "cand_pool",
            data_config.cand_pool_dir_name,
            [None] * len(cand_pool_name_list),
            cand_pool_name_list,
        ))

    for split_name, split_dir, dataset_name_list, cand_pool_name_list in splits:
        for dataset_name, cand_pool_name in zip(dataset_name_list, cand_pool_name_list):
            dataset_split_dict = {'img': [], 'text': [], 'img_mask': [], 'text_mask': [], 'id_to_index': {}}

            if split_name == "cand_pool":
                cand_pool_name = cand_pool_name.lower()
                cand_pool_file = f"mbeir_{cand_pool_name}_{split_name}.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file)
                print_config = dist_utils.is_main_process()
                dataset = MBEIRCandidatePoolDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    cand_pool_data_path=cand_pool_data_path,
                    img_preprocess_fn=img_preprocess_fn,
                    print_config=print_config,
                )
                collator = MBEIRCandidatePoolCollator(tokenizer=tokenizer, image_size=image_size)
            else:
                dataset_name = dataset_name.lower()
                query_data_path = os.path.join(split_dir, f"mbeir_{dataset_name}_{split_name}.jsonl")
                cand_pool_name = cand_pool_name.lower()
                cand_pool_data_path = os.path.join(cand_pool_dir, f"mbeir_{cand_pool_name}_cand_pool.jsonl")
                print_config = dist_utils.is_main_process()
                dataset = MBEIRMainDataset(
                    mbeir_data_dir=mbeir_data_dir,
                    query_data_path=query_data_path,
                    cand_pool_path=cand_pool_data_path,
                    query_instruct_path=query_instruct_path,
                    img_preprocess_fn=img_preprocess_fn,
                    mode=Mode.EVAL,
                    enable_query_instruct=data_config.enable_query_instruct,
                    shuffle_cand=data_config.shuffle_cand,
                    print_config=print_config,
                )
                collator = MBEIRMainCollator(tokenizer=tokenizer, image_size=image_size, mode=Mode.EVAL)

            sampler = DistributedSampler(
                dataset,
                num_replicas=dist_utils.get_world_size(),
                rank=dist_utils.get_rank(),
                shuffle=False,
            )
            data_loader = DataLoader(
                dataset,
                batch_size=config.dataloader_config.batch_size,
                num_workers=config.dataloader_config.num_workers,
                pin_memory=False,
                sampler=sampler,
                collate_fn=collator,
                drop_last=False,
            )
            if dist.is_initialized():
                dist.barrier()

            index = 0
            for batch in tqdm(data_loader):
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(config.dist_config.gpu_id, non_blocking=True)

                with autocast(enabled=config.embed_config.use_fp16):
                    img_emb, txt_emb = model.module.encode_multimodal_input(
                        batch["image_batched"], batch["txt_batched"]
                    )

                did_list = batch.get("did_list")
                txt_mask_batched = batch["txt_mask_batched"]
                image_mask_batched = batch["image_mask_batched"]

                dist.barrier()
                if utils.get_world_size() > 1:
                    did_list = torch.cat(utils.GatherLayer.apply(torch.LongTensor(did_list).to(img_emb.device)), dim=0)
                    txt_emb = torch.cat(utils.GatherLayer.apply(txt_emb), dim=0)
                    img_emb = torch.cat(utils.GatherLayer.apply(img_emb), dim=0)
                    txt_mask_batched = torch.cat(utils.GatherLayer.apply(txt_mask_batched), dim=0)
                    image_mask_batched = torch.cat(utils.GatherLayer.apply(image_mask_batched), dim=0)

                dist.barrier()
                if utils.is_main_process():
                    for j, did in enumerate(did_list):
                        if utils.get_world_size() > 1:
                            did = did.item()
                        if did not in dataset_split_dict['id_to_index']:
                            dataset_split_dict['id_to_index'][did] = index
                            dataset_split_dict['img'].append(img_emb[j].detach().cpu())
                            dataset_split_dict['text'].append(txt_emb[j].detach().cpu())
                            dataset_split_dict['img_mask'].append(image_mask_batched[j].detach().cpu())
                            dataset_split_dict['text_mask'].append(txt_mask_batched[j].detach().cpu())
                            index += 1

            dist.barrier()
            if utils.is_main_process():
                dataset_split_dict['img'] = torch.stack(dataset_split_dict['img'], dim=0)
                dataset_split_dict['text'] = torch.stack(dataset_split_dict['text'], dim=0)
                dataset_split_dict['img_mask'] = torch.stack(dataset_split_dict['img_mask'], dim=0)
                dataset_split_dict['text_mask'] = torch.stack(dataset_split_dict['text_mask'], dim=0)
                os.makedirs(save_embed_path, exist_ok=True)
                file_name = f"{split_name}_{cand_pool_name}_IT_dict.pt"
                torch.save(dataset_split_dict, os.path.join(save_embed_path, file_name))
                print(f"Saved {file_name}")

            del dataset_split_dict, dataset, collator, data_loader
            gc.collect()
            torch.cuda.empty_cache()


def main(config):
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    cudnn.benchmark = True

    model = BiomedCLIPNoFusion(config=config)
    model.float()
    model.eval()
    model = model.to(config.dist_config.gpu_id)

    if config.dist_config.distributed_mode:
        model = DDP(model, device_ids=[config.dist_config.gpu_id])

    model_without_ddp = model.module if config.dist_config.distributed_mode else model
    img_preprocess_fn = model_without_ddp.get_img_preprocess_fn()
    tokenizer = model_without_ddp.get_tokenizer()

    generate_embeds_for_config(model, img_preprocess_fn, tokenizer, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config_biomedclip_cand.yaml")
    parser.add_argument("--genir_dir", type=str, required=True)
    parser.add_argument("--mbeir_data_dir", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.genir_dir = args.genir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    args.dist_url = config.dist_config.dist_url
    utils.init_distributed_mode(args)
    config.dist_config.gpu_id = args.gpu
    config.dist_config.distributed_mode = args.distributed

    main(config)

    if config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()

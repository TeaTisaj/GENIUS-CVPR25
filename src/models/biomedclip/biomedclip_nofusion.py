"""
BiomedCLIP encoder wrapper with the same interface as CLIPNoFusion.

Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
Encoder: ViT-B/16 → 512-dim embeddings
Tokenizer: PubMedBERT, context_length=256

Requires: open_clip_torch
  pip install open_clip_torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


MODEL_HF_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CONTEXT_LENGTH = 256


class BiomedCLIPNoFusion(nn.Module):
    """
    Frozen BiomedCLIP encoder that returns separate image and text embeddings.

    Interface matches CLIPNoFusion so it can be used as a drop-in replacement
    for the GENIUS feature extraction and RQ training pipelines.

    encode_multimodal_input(img_tensor, txt_tensor) → (img_emb, txt_emb)
      both of shape [B, 512], unnormalized.
    """

    def __init__(self, config=None):
        super().__init__()
        self.model, self.img_preprocess_fn = open_clip.create_model_from_pretrained(MODEL_HF_ID)
        self._tokenizer = open_clip.get_tokenizer(MODEL_HF_ID)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def get_img_preprocess_fn(self):
        return self.img_preprocess_fn

    def get_tokenizer(self):
        _tokenizer = self._tokenizer

        def tokenizer_wrapper(txt):
            result = _tokenizer(txt, context_length=CONTEXT_LENGTH)
            # open_clip HF tokenizers may return a dict; extract input_ids
            if isinstance(result, dict):
                return result["input_ids"]
            return result

        return tokenizer_wrapper

    @torch.no_grad()
    def encode_image(self, image_tensor):
        return self.model.encode_image(image_tensor)

    @torch.no_grad()
    def encode_text(self, text_tokens):
        if hasattr(text_tokens, "input_ids"):
            text_tokens = text_tokens.input_ids
        return self.model.encode_text(text_tokens)

    def encode_multimodal_input(self, img_tensor, txt_tensor):
        """Return (img_emb, txt_emb) — same signature as CLIPNoFusion."""
        img_emb = self.encode_image(img_tensor)
        txt_emb = self.encode_text(txt_tensor)
        return img_emb, txt_emb

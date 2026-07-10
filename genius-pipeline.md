the pipeline should be:

1. Download the full M-BEIR data, including the images. The images are available here: https://huggingface.co/datasets/TIGER-Lab/M-BEIR/tree/main, and please follow the official downloading instructions here: https://huggingface.co/datasets/TIGER-Lab/M-BEIR#downloading-the-m-beir-dataset

2. Use the Stage 0 checkpoint to extract the CLIP-SF embeddings.

3. Use the Stage 1 checkpoint, rq_clip_large.pth, to construct the RQ codes.

4. Then train the Stage 2 GR model and evaluate it. Here, you can first evaluate COCO, FashionIQ

also While the full data is being downloaded/prepared, you can first run Stage 2 with the existing COCO + FashionIQ embeddings as a small local sanity check.



you could try using the provided RQ model, i.e., rq_clip_large.pth, to train the GR model.

remember the report provides the candidate pool and the download links. You can download it from https://huggingface.co/datasets/TIGER-Lab/M-BEIR/tree/main/cand_pool.
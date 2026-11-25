# FVD

Fréchet Video Distance between generated and ground-truth test-split videos, as reported in the paper.

## Protocol

| Step | What is done |
|---|---|
| Videos | the 2,048 test-split segments of [`eval_ids_duet_2048.json`](../src/data/eval_ids/eval_ids_duet_2048.json) (from the `duet` test split of the training dataset), generated with [`config/inference_fvd.yaml`](../config/inference_fvd.yaml). Each output `video_N_side_by_side.mp4` is 1280×704 with four 640×352 quadrants: 0 = GT player 1, 1 = generated player 1, 2 = GT player 2, 3 = generated player 2 |
| Clip | one contiguous 16-frame window per video, its start drawn from `np.random.default_rng(42)`, so every model is scored on the same windows |
| Resize | each quadrant resized to 224×224 (bilinear) and scaled to [-1, 1] |
| Features | the StyleGAN-V TorchScript port of the TF-Hub I3D ([`flateon/FVD-I3D-torchscript`](https://huggingface.co/flateon/FVD-I3D-torchscript), downloaded on first use), 400-d logits |
| FVD | Fréchet distance with both players pooled: quadrants 0+2 (4,096 GT clips) vs 1+3 (4,096 generated clips) |

## Run

```bash
# 1. Test-split generations (needs the duet dataset, see "Download training datasets" in the main README)
CUDA_VISIBLE_DEVICES=0 python src/inference.py --config-name inference_fvd \
    experiment_name=solaris device.eval_num_samples=2048

# 2. Features and FVD (PyTorch; see fvd/requirements.txt)
pip install -r fvd/requirements.txt
bash fvd/run_fvd.sh output/solaris/duet_2048
```

`run_fvd.sh` saves the features of each quadrant to `<video dir>/fvd_features/q{0,1,2,3}.npz` and skips quadrants already done. To compare different models, keep the same `--seed` and number of videos so that all models are scored on the same windows.

| File | Purpose |
|---|---|
| `compute_features.py` | I3D features of one quadrant of every video in a folder (`--sample_mode random_window --num_frames 16 --seed 42` for the paper) |
| `compute_fvd.py` | Fréchet distance between two sets of feature files; several files per side are concatenated |
| `run_fvd.sh` | both steps for one model with the paper's settings |

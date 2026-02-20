# VPT Dataset Download and Preparation

This directory contains scripts to download and prepare the Video PreTraining (VPT) dataset for training and evaluation.

## Step 1: Download the Dataset

Use `download_vpt_data.py` to download episodes from OpenAI's blob storage.

### Usage

```bash
conda activate solaris

python download_vpt_data.py --save_dir YOUR_DATASETS_DIR
```

### Arguments

- `--save_dir` (required): Directory where the dataset will be saved. The script creates the `vpt/` subdirectory and downloads the data there.
- `--num_workers` (default: 32): Number of parallel download workers
- `--num_index_workers` (default: 4): Number of parallel workers for processing index files
- `--max_samples` (optional): Limit the number of samples to download (useful for testing)

### How it works

1. The script reads index files from the `vpt_indices/` directory (e.g., `all_7xx_Apr_6.json`, `all_10xx_Jun_29.json`)
2. Each index file contains:
   - `basedir`: Base URL for the dataset (e.g., `https://openaipublic.blob.core.windows.net/minecraft-rl/`)
   - `relpaths`: List of relative paths to video files
3. The script extracts version numbers from filenames (e.g., `all_7xx_Apr_6.json` → version `7`)
4. Episodes are downloaded to versioned subdirectories (e.g., `v7/`, `v10/`)
5. For each episode, both the video file (`.mp4`) and the actions file (`.jsonl`) are downloaded

## Step 2: Prepare Test/Train Splits

After downloading the dataset, use `prepare_test_train_splits.py` to organize episodes into train and test directories.

### Usage

```bash
python prepare_test_train_splits.py YOUR_DATASETS_DIR/vpt/
```

### Arguments

- `dataset_path` (required): Path to the dataset directory (same as `--save_dir` from download step plus `vpt/`)

### How it works

1. The script reads `episodes_info_test.json` and `episodes_info_train.json` from the script directory
2. Each JSON file contains a list of episodes with:
   - `video_path`: Relative path to the video file
   - `actions_path`: Relative path to the actions file
   - `episode_id`: Unique episode identifier
   - `length`: Episode length
3. Episodes are moved to `test/` or `train/` subdirectories, preserving the version subfolder structure
4. Episode info files are copied to their respective directories (as `episodes_info.json`)
5. Any remaining files not in the train/test splits are moved to `excluded/`

### Output structure

This will reorganize the dataset into:

```
vpt/
├── test/
│   ├── v6/
│   │   └── ...
│   ├── v7/
│   │   └── ...
│   └── episodes_info.json
├── train/
│   ├── v6/
│   │   └── ...
│   ├── v7/
│   │   └── ...
│   └── episodes_info.json
└── excluded/
    └── ...
```

import argparse
import json
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm


def launch_jobs(worker_args, num_workers, worker_fn):
    with tqdm(total=len(worker_args)) as pbar:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker_fn, *args) for args in worker_args]
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)


def download_worker(
    basedir,
    relpath,
    save_dir,
):
    if relpath.endswith(".mp4"):
        relpath = relpath[:-4]
    os.makedirs(save_dir, exist_ok=True)
    file_exts = []
    file_exts.append(".jsonl")
    file_exts.append(".mp4")
    for file_ext in file_exts:
        cloud_file_path = os.path.join(basedir, relpath + file_ext)
        dest_file_path = os.path.join(save_dir, os.path.basename(cloud_file_path))
        if os.path.exists(dest_file_path):
            continue
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                response = requests.get(cloud_file_path, stream=True)
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            tmp_file.write(chunk)
                else:
                    print(
                        f"Failed to download: {cloud_file_path} (status code {response.status_code})"
                    )
                    continue
            # Basic verification: check that the file is non-empty.
            if os.path.getsize(tmp_file.name) == 0:
                print(f"Downloaded file is empty: {cloud_file_path}")
                os.remove(tmp_file.name)
                continue
            # Move the temp file to the destination
            shutil.move(tmp_file.name, dest_file_path)
            # print(f"Downloaded {cloud_file_path} to {dest_file_path}")
        except Exception as e:
            print(f"Error downloading {cloud_file_path}: {e}")


def extract_version_from_filename(filename):
    """Extract version number from filename like 'all_10xx_Jun_29.json' -> '10'"""
    match = re.search(r'_(\d+)xx_', filename)
    if match:
        return match.group(1)
    return None


def download_data(
    index_file,
    save_dir,
    num_workers=8,
    max_samples=None,
):
    with open(index_file, "r") as f:
        index_data = json.load(f)
    basedir = index_data["basedir"]
    relpaths = index_data["relpaths"]
    if max_samples is not None:
        relpaths = relpaths[:max_samples]

    worker_args = [
        (
            basedir,
            relpath,
            save_dir,
        )
        for relpath in relpaths
    ]
    launch_jobs(worker_args, num_workers, download_worker)


def process_index_file(
    index_file,
    base_save_dir,
    num_workers=8,
    max_samples=None,
):
    """Process a single index file and download to versioned subdirectory"""
    filename = os.path.basename(index_file)
    version = extract_version_from_filename(filename)
    if version is None:
        print(f"Warning: Could not extract version from {filename}, skipping")
        return
    
    versioned_save_dir = os.path.join(base_save_dir, f"v{version}")
    download_data(index_file, versioned_save_dir, num_workers, max_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument(
        "--num_index_workers",
        type=int,
        default=4,
        help="Number of parallel workers for processing index files",
    )
    args = parser.parse_args()
    
    
    # Process all index files in vpt_indices directory
    indices_dir = Path(__file__).parent / args.vpt_indices_dir
    
    index_files = list(indices_dir.glob("*.json"))
    if not index_files:
        print(f"No JSON files found in {indices_dir}")
        exit(1)
    
    print(f"Found {len(index_files)} index files to process")
    for idx_file in index_files:
        version = extract_version_from_filename(idx_file.name)
        if version:
            print(f"  - {idx_file.name} -> v{version}")
        else:
            print(f"  - {idx_file.name} -> (no version)")
    
    # Process index files in parallel
    worker_args = [
        (
            str(index_file),
            args.save_dir,
            args.num_workers,
            args.max_samples,
        )
        for index_file in index_files
    ]
    launch_jobs(worker_args, args.num_index_workers, process_index_file)

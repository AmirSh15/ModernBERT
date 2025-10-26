#!/usr/bin/env python3
"""
Script to download datasets from S3 buckets.
Downloads two datasets:
1. 25 songs per sub-genre dataset
2. 100 songs per sub-genre dataset
"""

import boto3
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# S3 Configuration
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Dataset paths in S3
DATASETS = {
    "clip_dataset": {
        "s3_prefix": "",
        "local_dir": "./data/embeddings",
    }
}

# Number of parallel downloads
MAX_WORKERS = 10


def setup_s3_client():
    """Initialize and return S3 client."""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    return s3_client


def list_s3_objects(s3_client, bucket, prefix):
    """List all objects in S3 bucket with given prefix."""
    objects = []
    paginator = s3_client.get_paginator('list_objects_v2')
    
    print(f"Listing objects in s3://{bucket}/{prefix}")
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                # Skip directories (objects ending with /)
                if not obj['Key'].endswith('/'):
                    objects.append(obj)
    
    return objects


def download_file(s3_client, bucket, s3_key, local_path):
    """Download a single file from S3."""
    try:
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the file
        s3_client.download_file(bucket, s3_key, local_path)
        return True, s3_key, None
    except Exception as e:
        return False, s3_key, str(e)


def download_dataset(s3_client, bucket, s3_prefix, local_dir):
    """Download all files from S3 prefix to local directory."""
    # List all objects
    objects = list_s3_objects(s3_client, bucket, s3_prefix)
    
    if not objects:
        print(f"No objects found in s3://{bucket}/{s3_prefix}")
        return
    
    print(f"Found {len(objects)} files to download")
    total_size = sum(obj['Size'] for obj in objects)
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    
    # Prepare download tasks
    download_tasks = []
    for obj in objects:
        s3_key = obj['Key']
        # Remove the prefix to get relative path
        relative_path = s3_key[len(s3_prefix):]
        local_path = os.path.join(local_dir, relative_path)
        
        # Skip if file already exists with same size
        if os.path.exists(local_path) and os.path.getsize(local_path) == obj['Size']:
            print(f"Skipping {s3_key} (already exists)")
            continue
        
        download_tasks.append((s3_key, local_path))
    
    if not download_tasks:
        print("All files already downloaded!")
        return
    
    print(f"Downloading {len(download_tasks)} files...")
    
    # Download files in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_file, s3_client, bucket, s3_key, local_path): (s3_key, local_path)
            for s3_key, local_path in download_tasks
        }
        
        with tqdm(total=len(download_tasks), desc="Downloading") as pbar:
            for future in as_completed(futures):
                success, s3_key, error = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"\nFailed to download {s3_key}: {error}")
                pbar.update(1)
    
    print(f"\nDownload complete: {successful} successful, {failed} failed")


def main():
    """Main function to download all datasets."""
    print("=" * 80)
    print("S3 Dataset Downloader")
    print("=" * 80)
    
    # Initialize S3 client
    s3_client = setup_s3_client()
    
    # Download each dataset
    for dataset_name, config in DATASETS.items():
        print(f"\n{'=' * 80}")
        print(f"Downloading dataset: {dataset_name}")
        print(f"S3 Path: s3://{BUCKET_NAME}/{config['s3_prefix']}")
        print(f"Local Path: {config['local_dir']}")
        print(f"{'=' * 80}\n")
        
        download_dataset(
            s3_client=s3_client,
            bucket=BUCKET_NAME,
            s3_prefix=config['s3_prefix'],
            local_dir=config['local_dir']
        )
    
    print(f"\n{'=' * 80}")
    print("All downloads complete!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

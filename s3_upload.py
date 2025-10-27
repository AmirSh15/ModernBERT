#!/usr/bin/env python3
"""
Script to upload checkpoint files to S3 buckets.
Uploads model checkpoint to S3 with the same path structure.
"""

import boto3
import os
from pathlib import Path
from tqdm import tqdm
import dotenv
import sys

# Load environment variables from .env file
dotenv.load_dotenv()

# S3 Configuration
ACCESS_KEY = os.getenv("ACCESS_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Default checkpoint path
DEFAULT_CHECKPOINT_PATH = "outputs/encoder_training/checkpoints/encoder-epoch=270-val_loss=0.0242.ckpt"


def setup_s3_client():
    """Initialize and return S3 client."""
    s3_client = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )
    return s3_client


def get_s3_object_info(s3_client, bucket, s3_key):
    """Get S3 object info (size, etag) if it exists."""
    try:
        response = s3_client.head_object(Bucket=bucket, Key=s3_key)
        return {
            'size': response['ContentLength'],
            'etag': response['ETag']
        }
    except s3_client.exceptions.ClientError as e:
        if e.response['Error']['Code'] == '404':
            return None
        raise


def upload_checkpoint(s3_client, bucket, local_path, s3_key=None):
    """Upload a checkpoint file to S3."""
    local_file = Path(local_path)
    
    if not local_file.exists():
        print(f"Error: File not found: {local_path}")
        return False
    
    if not local_file.is_file():
        print(f"Error: Path is not a file: {local_path}")
        return False
    
    # If s3_key not provided, use the same path structure
    if s3_key is None:
        s3_key = str(local_file)
        # Remove leading ./ if present
        if s3_key.startswith('./'):
            s3_key = s3_key[2:]
    
    file_size = local_file.stat().st_size
    print(f"Local file: {local_path}")
    print(f"File size: {file_size / (1024**2):.2f} MB")
    print(f"S3 destination: s3://{bucket}/{s3_key}")
    
    # Check if file already exists in S3 with same size
    s3_info = get_s3_object_info(s3_client, bucket, s3_key)
    
    if s3_info and s3_info['size'] == file_size:
        print("File already exists in S3 with same size. Skipping upload.")
        return True
    
    try:
        print("\nUploading checkpoint...")
        
        # Upload with progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Uploading") as pbar:
            def callback(bytes_transferred):
                pbar.update(bytes_transferred)
            
            s3_client.upload_file(
                str(local_file),
                bucket,
                s3_key,
                Callback=callback
            )
        
        print("\n✓ Upload successful!")
        print(f"S3 location: s3://{bucket}/{s3_key}")
        return True
        
    except Exception as e:
        print(f"\n✗ Upload failed: {str(e)}")
        return False


def main():
    """Main function to upload checkpoint."""
    print("=" * 80)
    print("S3 Checkpoint Uploader")
    print("=" * 80)
    
    # Get checkpoint path from command line or use default
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = DEFAULT_CHECKPOINT_PATH
    
    # Optional: S3 key override
    s3_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    print(f"\n")
    
    # Initialize S3 client
    s3_client = setup_s3_client()
    
    # Upload checkpoint
    success = upload_checkpoint(
        s3_client=s3_client,
        bucket=BUCKET_NAME,
        local_path=checkpoint_path,
        s3_key=s3_key
    )
    
    print(f"\n{'=' * 80}")
    if success:
        print("Upload complete!")
    else:
        print("Upload failed!")
        sys.exit(1)
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

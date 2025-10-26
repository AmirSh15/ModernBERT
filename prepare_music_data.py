"""
Utility script to prepare your music caption data for training.
This script helps you split your CSV files and validate the data format.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_csv(csv_path: str) -> bool:
    """
    Validate that CSV has the required columns and correct format.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        if 'caption_text' not in df.columns:
            logger.error(f"Missing 'caption_text' column in {csv_path}")
            return False
        
        if 'vector' not in df.columns:
            logger.error(f"Missing 'vector' column in {csv_path}")
            return False
        
        # Check for missing values
        missing_text = df['caption_text'].isna().sum()
        missing_vector = df['vector'].isna().sum()
        
        if missing_text > 0:
            logger.warning(f"Found {missing_text} missing caption_text values")
        
        if missing_vector > 0:
            logger.warning(f"Found {missing_vector} missing vector values")
        
        # Try to parse a few vectors
        valid_vectors = 0
        vector_dims = None
        
        for idx, row in df.head(10).iterrows():
            try:
                if pd.isna(row['vector']) or pd.isna(row['caption_text']):
                    continue
                
                vector_str = str(row['vector']).strip()
                if vector_str.startswith('[') and vector_str.endswith(']'):
                    vector_str = vector_str[1:-1]
                
                vector = np.array([float(x.strip()) for x in vector_str.split(',')])
                
                if vector_dims is None:
                    vector_dims = len(vector)
                elif len(vector) != vector_dims:
                    logger.error(f"Inconsistent vector dimensions at row {idx}: expected {vector_dims}, got {len(vector)}")
                    return False
                
                valid_vectors += 1
                
            except Exception as e:
                logger.error(f"Failed to parse vector at row {idx}: {e}")
                return False
        
        logger.info(f"✓ Valid CSV: {csv_path}")
        logger.info(f"  - Total rows: {len(df)}")
        logger.info(f"  - Valid samples checked: {valid_vectors}")
        logger.info(f"  - Vector dimension: {vector_dims}")
        logger.info(f"  - Missing caption_text: {missing_text}")
        logger.info(f"  - Missing vectors: {missing_vector}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {e}")
        return False


def split_csv_files(
    input_files: List[str],
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.15,
    test_ratio: float = 0.05,
    seed: int = 42,
):
    """
    Combine multiple CSV files and split into train/val/test sets.
    
    Args:
        input_files: List of input CSV file paths
        output_dir: Directory to save split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed
    """
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total}")
    
    # Load and combine all CSV files
    logger.info(f"Loading {len(input_files)} CSV file(s)...")
    dfs = []
    
    for csv_file in input_files:
        logger.info(f"  Reading {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Keep only required columns
        if 'caption_text' in df.columns and 'vector' in df.columns:
            df = df[['caption_text', 'vector']]
            # Remove rows with missing values
            df = df.dropna(subset=['caption_text', 'vector'])
            dfs.append(df)
        else:
            logger.warning(f"  Skipping {csv_file}: missing required columns")
    
    if not dfs:
        raise ValueError("No valid CSV files found")
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined total: {len(combined_df)} samples")
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split points
    n_samples = len(combined_df)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split
    train_df = combined_df[:n_train]
    val_df = combined_df[n_train:n_train + n_val]
    test_df = combined_df[n_train + n_val:]
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"\n✓ Data split completed:")
    logger.info(f"  - Train: {len(train_df)} samples -> {train_path}")
    logger.info(f"  - Val: {len(val_df)} samples -> {val_path}")
    logger.info(f"  - Test: {len(test_df)} samples -> {test_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare music caption data for training')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate CSV file format')
    validate_parser.add_argument('--input', type=str, required=True, help='Input CSV file to validate')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split CSV files into train/val/test')
    split_parser.add_argument('--input', type=str, nargs='+', required=True, 
                             help='Input CSV file(s) to combine and split')
    split_parser.add_argument('--output_dir', type=str, required=True, 
                             help='Output directory')
    split_parser.add_argument('--train_ratio', type=float, default=0.8, 
                             help='Train ratio (default: 0.8)')
    split_parser.add_argument('--val_ratio', type=float, default=0.15, 
                             help='Validation ratio (default: 0.15)')
    split_parser.add_argument('--test_ratio', type=float, default=0.05, 
                             help='Test ratio (default: 0.05)')
    split_parser.add_argument('--seed', type=int, default=42, 
                             help='Random seed')
    
    args = parser.parse_args()
    
    if args.command == 'validate':
        validate_csv(args.input)
    
    elif args.command == 'split':
        split_csv_files(
            input_files=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

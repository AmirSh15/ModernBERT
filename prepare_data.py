"""
Utility script to prepare training data for the embedding encoder.
This script helps convert your data into the required CSV format.
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Union
import json


def create_csv_from_arrays(
    sentences: List[str],
    vectors: Union[List[List[float]], np.ndarray],
    output_path: str,
    format: str = 'columns',
):
    """
    Create a CSV file from sentences and vectors.
    
    Args:
        sentences: List of sentences
        vectors: List or array of embedding vectors
        output_path: Path to save the CSV file
        format: 'columns' for separate dimension columns, 'single' for one vector column
    """
    vectors = np.array(vectors)
    
    if len(sentences) != len(vectors):
        raise ValueError(f"Mismatch: {len(sentences)} sentences but {len(vectors)} vectors")
    
    if format == 'columns':
        # Create separate columns for each dimension
        df_data = {'sentence': sentences}
        
        for i in range(vectors.shape[1]):
            df_data[f'dim_{i}'] = vectors[:, i]
        
        df = pd.DataFrame(df_data)
    
    elif format == 'single':
        # Create a single column with array representation
        vector_strings = [','.join(map(str, vec)) for vec in vectors]
        df = pd.DataFrame({
            'sentence': sentences,
            'vector': vector_strings,
        })
    
    else:
        raise ValueError(f"Unknown format: {format}. Use 'columns' or 'single'")
    
    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Created {output_path}")
    print(f"  - Number of samples: {len(sentences)}")
    print(f"  - Vector dimension: {vectors.shape[1]}")
    print(f"  - Format: {format}")


def split_data(
    input_csv: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split a single CSV file into train/val/test sets.
    
    Args:
        input_csv: Path to input CSV file
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
    
    # Load data
    df = pd.read_csv(input_csv)
    
    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Calculate split points
    n_samples = len(df)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split
    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = output_dir / 'train.csv'
    val_path = output_dir / 'val.csv'
    test_path = output_dir / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Split data into:")
    print(f"  - Train: {len(train_df)} samples -> {train_path}")
    print(f"  - Val: {len(val_df)} samples -> {val_path}")
    print(f"  - Test: {len(test_df)} samples -> {test_path}")


def create_example_data(output_dir: str, num_samples: int = 100, vector_dim: int = 768):
    """
    Create example/dummy data for testing.
    
    Args:
        output_dir: Directory to save example files
        num_samples: Number of samples to generate
        vector_dim: Dimension of embedding vectors
    """
    np.random.seed(42)
    
    # Generate random sentences
    sentence_templates = [
        "This is example sentence number {}.",
        "Sample text {} for training purposes.",
        "The quick brown fox jumps over the lazy dog {}.",
        "Machine learning example {}.",
        "Natural language processing sample {}.",
        "Deep learning training data {}.",
        "Neural network example {}.",
        "Artificial intelligence sample text {}.",
    ]
    
    sentences = [
        sentence_templates[i % len(sentence_templates)].format(i)
        for i in range(num_samples)
    ]
    
    # Generate random vectors
    vectors = np.random.randn(num_samples, vector_dim).astype(np.float32)
    # Normalize to unit length (common for embeddings)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Create train/val split
    n_train = int(num_samples * 0.8)
    
    train_sentences = sentences[:n_train]
    train_vectors = vectors[:n_train]
    
    val_sentences = sentences[n_train:]
    val_vectors = vectors[n_train:]
    
    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    create_csv_from_arrays(
        train_sentences,
        train_vectors,
        output_dir / 'example_train.csv',
        format='columns' if vector_dim <= 50 else 'single',
    )
    
    create_csv_from_arrays(
        val_sentences,
        val_vectors,
        output_dir / 'example_val.csv',
        format='columns' if vector_dim <= 50 else 'single',
    )
    
    print(f"\nExample data created in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for embedding encoder training')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split CSV into train/val/test')
    split_parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    split_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    split_parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    split_parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    split_parser.add_argument('--test_ratio', type=float, default=0.1, help='Test ratio')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Example command
    example_parser = subparsers.add_parser('example', help='Create example data')
    example_parser.add_argument('--output_dir', type=str, default='data/embeddings',
                               help='Output directory')
    example_parser.add_argument('--num_samples', type=int, default=100,
                               help='Number of samples')
    example_parser.add_argument('--vector_dim', type=int, default=768,
                               help='Vector dimension')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_data(
            input_csv=args.input,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    
    elif args.command == 'example':
        create_example_data(
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            vector_dim=args.vector_dim,
        )
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

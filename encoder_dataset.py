import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingDataset(Dataset):
    """
    Dataset for loading sentences and their corresponding target embedding vectors from CSV files.
    
    CSV file format should be:
    - 'caption_text' column: The input text
    - 'vector' column: The target embedding vector (comma-separated or array string)
    
    OR (legacy format):
    - First column: 'sentence' (text)
    - Remaining columns: vector dimensions (e.g., 'dim_0', 'dim_1', ..., 'dim_767')
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_length: int = 512,
        vector_dim: Optional[int] = None,
        text_column: str = 'caption_text',
        vector_column: str = 'vector',
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            split: Dataset split ('train', 'val', 'test')
            max_length: Maximum sequence length (not used for now, but kept for compatibility)
            vector_dim: Expected dimension of target vectors (auto-detected if None)
            text_column: Name of the column containing text (default: 'caption_text')
            vector_column: Name of the column containing vectors (default: 'vector')
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.vector_dim = vector_dim
        self.text_column = text_column
        self.vector_column = vector_column
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        if len(self.data) > 0:
            logger.info(f"Vector dimension: {self.data[0][1].shape[0]}")
    
    def _load_data(self) -> List[Tuple[str, torch.Tensor]]:
        """Load data from CSV files."""
        data = []
        
        # Look for CSV files matching the split
        csv_files = list(self.data_dir.glob(f'*{self.split}*.csv'))
        
        if not csv_files:
            # If no split-specific files, look for any CSV files
            csv_files = list(self.data_dir.glob('*.csv'))
            logger.warning(f"No files found for split '{self.split}', loading all CSV files")
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {self.data_dir}")
        
        logger.info(f"Loading data from {len(csv_files)} CSV file(s)...")
        
        for csv_file in csv_files:
            logger.info(f"Reading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            
            # Check for text column (try caption_text first, then fall back to sentence)
            text_col = None
            if self.text_column in df.columns:
                text_col = self.text_column
            elif 'sentence' in df.columns:
                text_col = 'sentence'
                logger.info(f"Using 'sentence' column instead of '{self.text_column}'")
            else:
                raise ValueError(f"CSV file {csv_file} must have a '{self.text_column}' or 'sentence' column")
            
            # Check for vector column
            if self.vector_column in df.columns:
                # Vector is in a single column (comma-separated or JSON array)
                data.extend(self._parse_vector_column(df, text_col))
            elif 'vector' in df.columns:
                # Fallback to 'vector' column
                logger.info(f"Using 'vector' column instead of '{self.vector_column}'")
                data.extend(self._parse_vector_column(df, text_col))
            else:
                # Vector dimensions are in separate columns
                data.extend(self._parse_dimension_columns(df, text_col))
        
        return data
    
    def _parse_vector_column(self, df: pd.DataFrame, text_col: str) -> List[Tuple[str, torch.Tensor]]:
        """Parse data where vectors are in a single column."""
        data = []
        
        for idx, row in df.iterrows():
            try:
                # Get text
                text = row[text_col]
                
                # Skip if text is missing or NaN
                if pd.isna(text) or (isinstance(text, str) and len(text.strip()) == 0):
                    logger.warning(f"Skipping row {idx}: missing or empty text")
                    continue
                
                text = str(text).strip()
                
                # Get vector
                vector_str = row[self.vector_column if self.vector_column in df.columns else 'vector']
                
                # Skip if vector is missing or NaN
                if pd.isna(vector_str):
                    logger.warning(f"Skipping row {idx}: missing vector")
                    continue
                
                # Parse the vector
                if isinstance(vector_str, str):
                    # Remove brackets and split by comma
                    vector_str = vector_str.strip()
                    # Handle both [1,2,3] and 1,2,3 formats
                    if vector_str.startswith('[') and vector_str.endswith(']'):
                        vector_str = vector_str[1:-1]
                    vector = np.array([float(x.strip()) for x in vector_str.split(',')])
                else:
                    # Assume it's already a numeric type or list
                    vector = np.array(vector_str)
                
                # Check vector dimension
                if self.vector_dim is None:
                    self.vector_dim = len(vector)
                elif len(vector) != self.vector_dim:
                    logger.warning(f"Skipping row {idx}: vector dimension mismatch "
                                 f"(expected {self.vector_dim}, got {len(vector)})")
                    continue
                
                vector_tensor = torch.tensor(vector, dtype=torch.float32)
                data.append((text, vector_tensor))
                
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to parsing error: {e}")
                continue
        
        return data
    
    def _parse_dimension_columns(self, df: pd.DataFrame, text_col: str) -> List[Tuple[str, torch.Tensor]]:
        """Parse data where vector dimensions are in separate columns."""
        data = []
        
        # Find dimension columns (all columns except text column)
        dim_columns = [col for col in df.columns if col != text_col]
        
        if not dim_columns:
            raise ValueError("No vector dimension columns found in CSV")
        
        # Check vector dimension
        if self.vector_dim is None:
            self.vector_dim = len(dim_columns)
        elif len(dim_columns) != self.vector_dim:
            raise ValueError(f"Expected {self.vector_dim} dimensions, but found {len(dim_columns)}")
        
        for idx, row in df.iterrows():
            try:
                text = str(row[text_col]).strip()
                
                # Skip if text is missing or empty
                if pd.isna(text) or len(text) == 0:
                    logger.warning(f"Skipping row {idx}: missing or empty text")
                    continue
                
                vector = np.array([float(row[col]) for col in dim_columns])
                vector_tensor = torch.tensor(vector, dtype=torch.float32)
                data.append((text, vector_tensor))
                
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to parsing error: {e}")
                continue
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        """
        Returns:
            sentence: The input sentence (string)
            target_vector: The target embedding vector (tensor)
        """
        return self.data[idx]


def collate_fn(batch: List[Tuple[str, torch.Tensor]]) -> Tuple[List[str], torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Args:
        batch: List of (sentence, target_vector) tuples
        
    Returns:
        sentences: List of sentences
        target_vectors: Batched tensor of target vectors
    """
    sentences, target_vectors = zip(*batch)
    target_vectors = torch.stack(target_vectors)
    
    return list(sentences), target_vectors


if __name__ == '__main__':
    # Test the dataset
    import argparse
    
    parser = argparse.ArgumentParser(description='Test EmbeddingDataset')
    parser.add_argument('--data_dir', type=str, default='data/embeddings',
                      help='Directory containing CSV files')
    parser.add_argument('--split', type=str, default='train',
                      help='Dataset split to load')
    
    args = parser.parse_args()
    
    # Create dataset
    dataset = EmbeddingDataset(
        data_dir=args.data_dir,
        split=args.split,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Test a few samples
        for i in range(min(3, len(dataset))):
            sentence, vector = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Sentence: {sentence[:100]}...")
            print(f"  Vector shape: {vector.shape}")
            print(f"  Vector (first 5 dims): {vector[:5]}")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(loader))
        sentences, vectors = batch
        
        print(f"\nBatch test:")
        print(f"  Number of sentences: {len(sentences)}")
        print(f"  Vectors shape: {vectors.shape}")

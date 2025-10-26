import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
from transformers import AutoTokenizer, DataCollatorWithPadding

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
        model_id: str = "answerdotai/ModernBERT-base",
    ):
        """
        Args:
            data_dir: Directory containing CSV files
            split: Dataset split ('train', 'val', 'test')
            max_length: Maximum sequence length for tokenization
            vector_dim: Expected dimension of target vectors (auto-detected if None)
            text_column: Name of the column containing text (default: 'caption_text')
            vector_column: Name of the column containing vectors (default: 'vector')
            model_id: HuggingFace model ID for tokenizer
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.vector_dim = vector_dim
        self.text_column = text_column
        self.vector_column = vector_column
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Track missing vectors
        self.missing_vectors = []
        
        # Load data
        self.data = self._load_data()
        
        logger.info(f"Loaded {len(self.data)} samples for {split} split")
        if len(self.data) > 0:
            logger.info(f"Vector dimension: {self.data[0][1].shape[0]}")
        
        # Log missing vectors summary
        self._log_missing_vectors()
    
    def _load_data(self) -> List[Tuple[str, torch.Tensor]]:
        """Load data from CSV files."""
        data = []
        
        # Look for CSV files matching the split
        csv_files = list(self.data_dir.glob(f'*{self.split}*.csv'))

        if not csv_files and self.split == 'train':
            # If no split-specific files, look for any CSV files
            csv_files = list(self.data_dir.glob('*.csv'))
            logger.warning(f"No files found for split '{self.split}', loading all CSV files")
        
        if not csv_files:
            logger.error(f"No CSV files found in {self.data_dir} for split '{self.split}'")
            return data  # Return empty data if no files found
        
        logger.info(f"Loading data from {len(csv_files)} CSV file(s)...")
        
        for csv_file in csv_files:
            logger.info(f"Reading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            
            # Store current length to track which missing vectors came from this file
            current_missing_count = len(self.missing_vectors)
            
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
            
            # Update file name for missing vectors from this file
            for i in range(current_missing_count, len(self.missing_vectors)):
                self.missing_vectors[i]['file'] = csv_file.name
        
        return data

    def _parse_vector_column(self, df: pd.DataFrame, text_col: str, verbose: bool=False) -> List[Tuple[str, torch.Tensor]]:
        """Parse data where vectors are in a single column."""
        data = []
        
        for idx, row in df.iterrows():
            try:
                # Get text
                text = row[text_col]
                
                # Skip if text is missing or NaN
                if pd.isna(text) or (isinstance(text, str) and len(text.strip()) == 0):
                    if verbose:
                        logger.warning(f"Skipping row {idx}: missing or empty text")
                    continue
                
                text = str(text).strip()
                
                # Get vector
                vector_str = row[self.vector_column if self.vector_column in df.columns else 'vector']
                
                # Skip if vector is missing or NaN
                if pd.isna(vector_str):
                    if verbose:
                        logger.warning(f"Skipping row {idx}: missing vector")
                    self.missing_vectors.append({
                        'row_index': idx,
                        'text': text,
                        'reason': 'missing vector (NaN)',
                        'file': None  # Will be set in _load_data
                    })
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
                    if verbose:
                        logger.warning(f"Skipping row {idx}: vector dimension mismatch "
                                       f"(expected {self.vector_dim}, got {len(vector)})")
                    self.missing_vectors.append({
                        'row_index': idx,
                        'text': text,
                        'reason': f'vector dimension mismatch (expected {self.vector_dim}, got {len(vector)})',
                        'file': None  # Will be set in _load_data
                    })
                    continue
                
                vector_tensor = torch.tensor(vector, dtype=torch.float32)
                data.append((text, vector_tensor))
                
            except Exception as e:
                if verbose:
                    logger.warning(f"Skipping row {idx} due to parsing error: {e}")
                self.missing_vectors.append({
                    'row_index': idx,
                    'text': text if 'text' in locals() else 'N/A',
                    'reason': f'parsing error: {str(e)}',
                    'file': None  # Will be set in _load_data
                })
                continue
        
        return data
    
    def _log_missing_vectors(self):
        """Log summary of missing vectors."""
        if not self.missing_vectors:
            logger.info("✓ No missing vectors found - all rows have valid vectors!")
            return
        
        # Group by file and count total rows per file
        files_dict = {}
        file_total_rows = {}
        
        for mv in self.missing_vectors:
            file_name = mv['file'] if mv['file'] else 'Unknown'
            if file_name not in files_dict:
                files_dict[file_name] = []
            files_dict[file_name].append(mv)
        
        # Read each file to get total row count
        total_original_rows = 0
        for file_name in files_dict.keys():
            if file_name != 'Unknown':
                try:
                    file_path = self.data_dir / file_name
                    df = pd.read_csv(file_path)
                    file_total_rows[file_name] = len(df)
                    total_original_rows += len(df)
                except:
                    file_total_rows[file_name] = None
        
        logger.warning(f"\nMISSING VECTORS SUMMARY:")
        logger.warning(f"Total rows with missing vectors: {len(self.missing_vectors)} out of {total_original_rows} ({len(self.missing_vectors) / total_original_rows * 100:.2f}%)")
        
        # Log concise summary for each file
        for file_name, missing_list in files_dict.items():
            missing_count = len(missing_list)
            if file_name in file_total_rows and file_total_rows[file_name]:
                total = file_total_rows[file_name]
                percentage = (missing_count / total) * 100
                logger.warning(f"  {file_name}: {missing_count}/{total} rows ({percentage:.2f}%) have missing vectors")
            else:
                logger.warning(f"  {file_name}: {missing_count} rows have missing vectors")
    
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
                
                # Check for NaN values in vector
                if np.isnan(vector).any():
                    logger.warning(f"Skipping row {idx}: vector contains NaN values")
                    self.missing_vectors.append({
                        'row_index': idx,
                        'text': text,
                        'reason': 'vector contains NaN values',
                        'file': None  # Will be set in _load_data
                    })
                    continue
                
                vector_tensor = torch.tensor(vector, dtype=torch.float32)
                data.append((text, vector_tensor))
                
            except Exception as e:
                logger.warning(f"Skipping row {idx} due to parsing error: {e}")
                self.missing_vectors.append({
                    'row_index': idx,
                    'text': text if 'text' in locals() else 'N/A',
                    'reason': f'parsing error: {str(e)}',
                    'file': None  # Will be set in _load_data
                })
                continue
        
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            A dictionary containing:
                - 'input_ids': Tokenized input IDs (tensor)
                - 'attention_mask': Attention mask (tensor)
                - 'target_vector': The target embedding vector (tensor)
        """
        text, target_vector = self.data[idx]
        
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        
        # Remove batch dimension since DataLoader will add it back
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'target_vector': target_vector,
        }


class EmbeddingCollator:
    """
    Collator class for EmbeddingDataset that handles padding and batching.
    This class is picklable, unlike nested functions.
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        """
        Args:
            tokenizer: The tokenizer to use for padding
        """
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors='pt')
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader with padding.
        
        Args:
            batch: List of dictionaries with 'input_ids', 'attention_mask', and 'target_vector'
            
        Returns:
            A dictionary containing:
                - 'input_ids': Padded tensor of shape (batch_size, max_seq_len)
                - 'attention_mask': Padded tensor of shape (batch_size, max_seq_len)
                - 'target_vectors': Stacked tensor of shape (batch_size, vector_dim)
        """
        # Extract target vectors
        target_vectors = [item['target_vector'] for item in batch]
        
        # Prepare the batch for the data collator (only input_ids and attention_mask)
        collator_input = [
            {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask']
            }
            for item in batch
        ]
        
        # Use the data collator to pad
        padded_batch = self.data_collator(collator_input)
        
        # Add target vectors to the batch
        padded_batch['target_vectors'] = torch.stack(target_vectors)
        
        return padded_batch


def get_collate_fn(tokenizer: AutoTokenizer):
    """
    Create a collate function with padding using DataCollatorWithPadding.
    
    Args:
        tokenizer: The tokenizer to use for padding
        
    Returns:
        An EmbeddingCollator instance that can be used with DataLoader
    """
    return EmbeddingCollator(tokenizer)


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
            item = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Input IDs shape: {item['input_ids'].shape}")
            print(f"  Attention mask shape: {item['attention_mask'].shape}")
            print(f"  Target vector shape: {item['target_vector'].shape}")
            print(f"  Target vector (first 5 dims): {item['target_vector'][:5]}")
        
        # Test DataLoader
        from torch.utils.data import DataLoader
        
        # Get the collate function with the dataset's tokenizer
        collate_fn = get_collate_fn(dataset.tokenizer)
        
        loader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            collate_fn=collate_fn,
        )
        
        batch = next(iter(loader))
        
        print(f"\nBatch test:")
        print(f"  Input IDs shape: {batch['input_ids'].shape}")
        print(f"  Attention mask shape: {batch['attention_mask'].shape}")
        print(f"  Target vectors shape: {batch['target_vectors'].shape}")
        print(f"  Max sequence length in batch: {batch['input_ids'].shape[1]}")

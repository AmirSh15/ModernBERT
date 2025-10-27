#!/usr/bin/env python3
"""
Inference script for the trained encoder model.
Loads a checkpoint and performs inference on text inputs.
"""

import torch
import argparse
from pathlib import Path
from typing import List, Union
import numpy as np

from train_encoder import EmbeddingTrainingModule


class EncoderInference:
    """
    Inference wrapper for the trained encoder model.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = None,
    ):
        """
        Initialize the inference model.
        
        Args:
            checkpoint_path: Path to the trained checkpoint (.ckpt file)
            device: Device to run inference on ('cuda', 'cpu', or None for auto-detect)
        """
        self.checkpoint_path = checkpoint_path
        
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Load the model from checkpoint
        print(f"Loading checkpoint from: {checkpoint_path}")
        self.model = EmbeddingTrainingModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )
        
        # Set to evaluation mode
        self.model.eval()
        self.model.to(self.device)
        
        # Get tokenizer from the model
        self.tokenizer = self.model.encoder.tokenizer
        
        print("Model loaded successfully!")
        print(f"Model configuration:")
        print(f"  - Model ID: {self.model.hparams.model_id}")
        print(f"  - Use projection: {self.model.hparams.use_projection}")
        if self.model.hparams.use_projection:
            print(f"  - Projection hidden dim: {self.model.hparams.projection_hidden_dim}")
            print(f"  - Output dim: {self.model.hparams.output_dim}")
        
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        max_length: int = 512,
        normalize: bool = False,
        return_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text string or list of text strings
            batch_size: Batch size for processing multiple texts
            max_length: Maximum sequence length for tokenization
            normalize: Whether to L2-normalize the embeddings
            return_numpy: If True, return numpy array; if False, return torch tensor
            
        Returns:
            Embeddings as numpy array or torch tensor
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
            single_input = True
        else:
            single_input = False
        
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Move to device
            input_ids = encoded_inputs['input_ids'].to(self.device)
            attention_mask = encoded_inputs['attention_mask'].to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model.encoder.get_embeddings(input_ids, attention_mask)
                
                # Normalize if requested
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                all_embeddings.append(embeddings.cpu())
        
        # Concatenate all batches
        embeddings = torch.cat(all_embeddings, dim=0)
        
        # Return single embedding if single input
        if single_input:
            embeddings = embeddings[0]
        
        # Convert to numpy if requested
        if return_numpy:
            embeddings = embeddings.numpy()
        
        return embeddings
    
    def compute_similarity(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]],
        metric: str = 'cosine'
    ) -> Union[float, np.ndarray]:
        """
        Compute similarity between two texts or two lists of texts.
        
        Args:
            text1: First text or list of texts
            text2: Second text or list of texts
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score(s)
        """
        # Get embeddings
        emb1 = self.encode(text1, normalize=(metric == 'cosine'), return_numpy=True)
        emb2 = self.encode(text2, normalize=(metric == 'cosine'), return_numpy=True)
        
        # Ensure both are 2D
        if emb1.ndim == 1:
            emb1 = emb1.reshape(1, -1)
        if emb2.ndim == 1:
            emb2 = emb2.reshape(1, -1)
        
        if metric == 'cosine':
            # Cosine similarity
            similarity = np.sum(emb1 * emb2, axis=1)
        elif metric == 'euclidean':
            # Negative euclidean distance (so higher is more similar)
            similarity = -np.linalg.norm(emb1 - emb2, axis=1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Return scalar if single pair
        if len(similarity) == 1:
            return float(similarity[0])
        
        return similarity


def main():
    """Main inference function with example usage."""
    parser = argparse.ArgumentParser(description='Run inference with trained encoder')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/encoder_training/checkpoints/encoder-epoch=270-val_loss=0.0242.ckpt',
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--text',
        type=str,
        nargs='+',
        default=None,
        help='Text(s) to encode'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare similarity between pairs of texts'
    )
    
    args = parser.parse_args()
    
    # Initialize inference model
    inferencer = EncoderInference(
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # If no text provided, run example
    if args.text is None:
        print("\n" + "="*80)
        print("Running example inference...")
        print("="*80)
        
        # Example 1: Single text encoding
        print("\n--- Example 1: Single text encoding ---")
        text = "A beautiful sunset over the ocean with orange and pink colors"
        embedding = inferencer.encode(text)
        print(f"Text: {text}")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 5 dims): {embedding[:5]}")
        
        # Example 2: Batch encoding
        print("\n--- Example 2: Batch encoding ---")
        texts = [
            "A cat sitting on a windowsill",
            "A dog playing in the park",
            "A bird flying in the sky",
            "Classical music with piano and strings",
            "Rock music with electric guitar"
        ]
        embeddings = inferencer.encode(texts)
        print(f"Number of texts: {len(texts)}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Example 3: Similarity comparison
        print("\n--- Example 3: Similarity comparison ---")
        text1 = "A cat sitting on a windowsill"
        text2 = "A dog playing in the park"
        text3 = "A feline resting by the window"
        
        sim_1_2 = inferencer.compute_similarity(text1, text2)
        sim_1_3 = inferencer.compute_similarity(text1, text3)
        
        print(f"Text 1: {text1}")
        print(f"Text 2: {text2}")
        print(f"Text 3: {text3}")
        print(f"\nCosine similarity (Text 1 vs Text 2): {sim_1_2:.4f}")
        print(f"Cosine similarity (Text 1 vs Text 3): {sim_1_3:.4f}")
        print("(Text 1 and Text 3 should be more similar as they describe similar scenes)")
        
        # Example 4: Batch similarity
        print("\n--- Example 4: Batch similarity comparison ---")
        queries = [
            "Rock music with drums",
            "Classical orchestral music"
        ]
        documents = [
            "Heavy metal rock band playing",
            "Symphony orchestra performance",
            "Jazz music with saxophone",
            "Electronic dance music"
        ]
        
        print("Queries:")
        for i, q in enumerate(queries):
            print(f"  Q{i+1}: {q}")
        
        print("\nDocuments:")
        for i, d in enumerate(documents):
            print(f"  D{i+1}: {d}")
        
        print("\nSimilarity matrix:")
        for i, query in enumerate(queries):
            similarities = []
            for doc in documents:
                sim = inferencer.compute_similarity(query, doc)
                similarities.append(sim)
            print(f"  Q{i+1}: {' '.join([f'{s:.4f}' for s in similarities])}")
    
    else:
        # User provided text(s)
        print("\n" + "="*80)
        print("Encoding provided text(s)...")
        print("="*80)
        
        if args.compare and len(args.text) >= 2:
            # Compare pairs
            print("\n--- Similarity Comparison ---")
            for i in range(0, len(args.text) - 1, 2):
                if i + 1 < len(args.text):
                    text1 = args.text[i]
                    text2 = args.text[i + 1]
                    sim = inferencer.compute_similarity(text1, text2)
                    print(f"\nText 1: {text1}")
                    print(f"Text 2: {text2}")
                    print(f"Cosine similarity: {sim:.4f}")
        else:
            # Just encode
            embeddings = inferencer.encode(args.text)
            print(f"\nEncoded {len(args.text)} text(s)")
            print(f"Embeddings shape: {embeddings.shape}")
            
            for i, text in enumerate(args.text):
                print(f"\nText {i+1}: {text}")
                if embeddings.ndim == 1:
                    print(f"Embedding (first 5 dims): {embeddings[:5]}")
                else:
                    print(f"Embedding (first 5 dims): {embeddings[i, :5]}")
    
    print("\n" + "="*80)
    print("Inference complete!")
    print("="*80)


if __name__ == "__main__":
    main()

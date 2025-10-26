"""
Inference script for testing the trained embedding encoder.
"""

import torch
from transformers import AutoTokenizer, AutoModel
import argparse
import numpy as np
from pathlib import Path


class TrainedEncoder:
    """Wrapper for the trained encoder model."""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: Path to the trained model directory
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        
        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def encode(self, text: str, normalize: bool = False) -> np.ndarray:
        """
        Encode a single text into an embedding vector.
        
        Args:
            text: Input text
            normalize: Whether to normalize the embedding to unit length
            
        Returns:
            Embedding vector as numpy array
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
            
            if normalize:
                cls_embedding = cls_embedding / np.linalg.norm(cls_embedding)
            
            return cls_embedding
    
    def encode_batch(self, texts: list, normalize: bool = False, batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple texts into embedding vectors.
        
        Args:
            texts: List of input texts
            normalize: Whether to normalize embeddings to unit length
            batch_size: Batch size for processing
            
        Returns:
            Array of embedding vectors
        """
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                if normalize:
                    batch_embeddings = batch_embeddings / np.linalg.norm(
                        batch_embeddings, axis=1, keepdims=True
                    )
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def similarity(self, text1: str, text2: str, metric: str = 'cosine') -> float:
        """
        Compute similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        emb1 = self.encode(text1, normalize=(metric == 'cosine'))
        emb2 = self.encode(text2, normalize=(metric == 'cosine'))
        
        if metric == 'cosine':
            return float(np.dot(emb1, emb2))
        elif metric == 'euclidean':
            return float(-np.linalg.norm(emb1 - emb2))
        else:
            raise ValueError(f"Unknown metric: {metric}")


def main():
    parser = argparse.ArgumentParser(description='Test the trained encoder')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model directory')
    parser.add_argument('--text', type=str, default=None,
                       help='Single text to encode')
    parser.add_argument('--texts', type=str, nargs='+', default=None,
                       help='Multiple texts to encode')
    parser.add_argument('--compare', type=str, nargs=2, default=None,
                       help='Two texts to compare similarity')
    parser.add_argument('--normalize', action='store_true',
                       help='Normalize embeddings to unit length')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load encoder
    encoder = TrainedEncoder(args.model_path, device=args.device)
    
    # Single text encoding
    if args.text:
        print(f"\nEncoding: '{args.text}'")
        embedding = encoder.encode(args.text, normalize=args.normalize)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding (first 10 dims): {embedding[:10]}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Batch encoding
    elif args.texts:
        print(f"\nEncoding {len(args.texts)} texts...")
        embeddings = encoder.encode_batch(args.texts, normalize=args.normalize)
        print(f"Embeddings shape: {embeddings.shape}")
        for i, (text, emb) in enumerate(zip(args.texts, embeddings)):
            print(f"\nText {i + 1}: '{text}'")
            print(f"  First 10 dims: {emb[:10]}")
            print(f"  Norm: {np.linalg.norm(emb):.4f}")
    
    # Compare two texts
    elif args.compare:
        text1, text2 = args.compare
        print(f"\nComparing:")
        print(f"  Text 1: '{text1}'")
        print(f"  Text 2: '{text2}'")
        
        cos_sim = encoder.similarity(text1, text2, metric='cosine')
        euc_dist = encoder.similarity(text1, text2, metric='euclidean')
        
        print(f"\nCosine similarity: {cos_sim:.4f}")
        print(f"Euclidean distance: {-euc_dist:.4f}")
    
    # Interactive mode
    else:
        print("\nInteractive mode - enter texts to encode (or 'quit' to exit)")
        while True:
            text = input("\nEnter text: ").strip()
            if text.lower() in ['quit', 'exit', 'q']:
                break
            if not text:
                continue
            
            embedding = encoder.encode(text, normalize=args.normalize)
            print(f"Embedding shape: {embedding.shape}")
            print(f"First 10 dims: {embedding[:10]}")
            print(f"Norm: {np.linalg.norm(embedding):.4f}")


if __name__ == '__main__':
    main()

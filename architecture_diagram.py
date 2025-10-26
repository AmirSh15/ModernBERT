"""
Visualize the training architecture and data flow.
Run this to see how the components fit together.
"""

def print_architecture():
    print("""
╔═══════════════════════════════════════════════════════════════════════════╗
║                    EMBEDDING ENCODER TRAINING ARCHITECTURE                 ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│                          DATA PREPARATION                                │
└─────────────────────────────────────────────────────────────────────────┘

   CSV Files (data/embeddings/)
   ├── train.csv                    ┌─────────────────────┐
   │   ├── sentence                 │  prepare_data.py    │
   │   └── vector (or dim_0..N)     │  - Split data       │
   └── val.csv                       │  - Create examples  │
       ├── sentence                  │  - Validate format  │
       └── vector (or dim_0..N)      └─────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  EmbeddingDataset    │
         │  (encoder_dataset.py)│
         │  - Parse CSV         │
         │  - Load vectors      │
         │  - Validate dims     │
         └──────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────────┐
   │                  PyTorch Lightning Trainer                   │
   │                   (train_encoder.py)                         │
   └─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   ┌─────────┐         ┌─────────┐         ┌──────────┐
   │ DataLoader│        │  Model  │         │Optimizer │
   │ (train)  │        │ Module  │         │Scheduler │
   └─────────┘         └─────────┘         └──────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
              ┌───────────────────────────┐
              │ EmbeddingTrainingModule  │
              │  ┌────────────────────┐  │
              │  │   emerge_text      │  │
              │  │   (hf.py)          │  │
              │  │  ┌──────────────┐  │  │
              │  │  │ Tokenizer    │  │  │
              │  │  │ ModernBERT   │  │  │
              │  │  │ (Transformer)│  │  │
              │  │  └──────────────┘  │  │
              │  └────────────────────┘  │
              └───────────────────────────┘
                         │
                         ▼
              ┌───────────────────┐
              │  CLS Embedding    │  [batch_size, hidden_dim]
              └───────────────────┘
                         │
                         ▼
              ┌───────────────────┐
              │    MSE Loss       │  Compare with target vectors
              │  loss = ||pred -  │
              │         target||² │
              └───────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      TRAINING LOOP                                       │
└─────────────────────────────────────────────────────────────────────────┘

   For each batch:
   
   1. Load batch (sentences, target_vectors)
          │
          ▼
   2. Tokenize sentences
          │
          ▼
   3. Forward pass through ModernBERT
          │
          ▼
   4. Extract CLS embeddings (first token)
          │
          ▼
   5. Compute MSE loss vs target_vectors
          │
          ▼
   6. Backward pass (compute gradients)
          │
          ▼
   7. Update model weights
          │
          ▼
   8. Log metrics (loss, cosine similarity)
          │
          └──→ Repeat

┌─────────────────────────────────────────────────────────────────────────┐
│                      MONITORING & LOGGING                                │
└─────────────────────────────────────────────────────────────────────────┘

   ┌──────────────┐        ┌──────────────┐       ┌──────────────┐
   │ TensorBoard  │        │ Checkpoints  │       │ Early Stop   │
   │              │        │              │       │              │
   │ - train_loss │        │ - Best model │       │ - Monitor    │
   │ - val_loss   │        │ - Last model │       │   val_loss   │
   │ - cosine_sim │        │ - Top 3      │       │ - Patience=5 │
   │ - LR         │        │              │       │              │
   └──────────────┘        └──────────────┘       └──────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT                                         │
└─────────────────────────────────────────────────────────────────────────┘

   outputs/encoder_training/
   ├── checkpoints/
   │   ├── encoder-epoch=00-val_loss=0.1234.ckpt
   │   ├── encoder-epoch=01-val_loss=0.0987.ckpt
   │   └── last.ckpt
   ├── logs/
   │   └── encoder-training/
   │       └── events.out.tfevents.*
   └── final_model/
       ├── config.json           ← HuggingFace format
       ├── pytorch_model.bin     ← Trained weights
       ├── tokenizer_config.json
       └── vocab.txt

┌─────────────────────────────────────────────────────────────────────────┐
│                          INFERENCE                                       │
└─────────────────────────────────────────────────────────────────────────┘

   New Text → Tokenizer → ModernBERT → CLS Embedding → Output
                              ↑
                   (Trained weights from final_model/)

┌─────────────────────────────────────────────────────────────────────────┐
│                     KEY HYPERPARAMETERS                                  │
└─────────────────────────────────────────────────────────────────────────┘

   Learning Rate: 2e-5 (with warmup + decay)
   Batch Size: 16
   Max Epochs: 10
   Optimizer: AdamW (β1=0.9, β2=0.999)
   Weight Decay: 0.01
   Gradient Clip: 1.0
   Precision: Mixed (16-bit)

┌─────────────────────────────────────────────────────────────────────────┐
│                        LOSS FUNCTION                                     │
└─────────────────────────────────────────────────────────────────────────┘

   MSE Loss = (1/N) Σ ||embedding_i - target_i||²
   
   Where:
   - embedding_i = CLS token from ModernBERT
   - target_i = ground truth vector from CSV
   - N = batch size
   
   Goal: Minimize the squared distance between predicted 
         and target embeddings

""")

if __name__ == '__main__':
    print_architecture()

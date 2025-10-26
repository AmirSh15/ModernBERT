# Embedding Encoder Training with PyTorch Lightning

This directory contains a complete training pipeline for fine-tuning the ModernBERT encoder to match target embeddings using MSE loss.

## Overview

The training system uses PyTorch Lightning to train the `emerge_text` encoder model. The goal is to make the CLS token embeddings match pre-computed target vectors from CSV files.

## Files Structure

```
.
├── hf.py                          # Your encoder model (emerge_text)
├── train_encoder.py               # Main training script with PyTorch Lightning
├── encoder_dataset.py             # Dataset class for loading CSV data
├── config/
│   └── train_config.yaml         # Training configuration
├── data/
│   └── embeddings/               # Your CSV data files
│       ├── example_train.csv     # Training data example
│       └── example_val.csv       # Validation data example
└── requirements-training.txt      # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements-training.txt
```

### 2. Prepare Your Data

Create CSV files with your sentences and target embeddings. Two formats are supported:

**Format 1: Separate dimension columns (recommended for smaller dimensions)**
```csv
sentence,dim_0,dim_1,dim_2,...,dim_767
"Your sentence here",0.123,-0.456,0.789,...,0.321
```

**Format 2: Single vector column (better for large dimensions)**
```csv
sentence,vector
"Your sentence here","[0.123, -0.456, 0.789, ..., 0.321]"
```

### 3. Organize Your Data

Place your CSV files in the data directory:
- Training files: Should contain `train` in filename (e.g., `my_train.csv`, `train_data.csv`)
- Validation files: Should contain `val` in filename (e.g., `my_val.csv`, `val_data.csv`)

Example structure:
```
data/embeddings/
├── train_embeddings.csv
└── val_embeddings.csv
```

### 4. Configure Training

Edit `config/train_config.yaml` to adjust:
- Model settings (model ID, freeze layers)
- Data paths
- Training hyperparameters (learning rate, batch size, epochs)
- Hardware settings (GPU, precision)
- Logging preferences

## Usage

### Basic Training

```bash
python train_encoder.py --config config/train_config.yaml
```

### Override Config with Command Line Arguments

```bash
python train_encoder.py \
    --config config/train_config.yaml \
    --data_dir data/my_embeddings \
    --output_dir outputs/my_experiment
```

### Test Dataset Loading

Before training, you can test if your data loads correctly:

```bash
python encoder_dataset.py --data_dir data/embeddings --split train
```

## Training Configuration

Key parameters in `config/train_config.yaml`:

### Model Settings
- `model_id`: HuggingFace model identifier (default: "answerdotai/ModernBERT-base")
- `freeze_layers`: Number of encoder layers to freeze (0 = train all)

### Training Hyperparameters
- `batch_size`: Batch size per device (default: 16)
- `learning_rate`: Learning rate (default: 2e-5)
- `max_epochs`: Maximum training epochs (default: 10)
- `max_steps`: Maximum training steps (default: 10000)
- `warmup_steps`: LR warmup steps (default: 500)
- `weight_decay`: Weight decay for regularization (default: 0.01)

### Hardware
- `devices`: Number of GPUs (1 for single GPU, -1 for all)
- `precision`: Training precision ("32", "16-mixed", "bf16-mixed")
- `num_workers`: Data loading workers (default: 4)

### Early Stopping
- `early_stopping_patience`: Patience for early stopping (default: 5)

## Outputs

Training produces the following outputs in the `output_dir`:

```
outputs/encoder_training/
├── checkpoints/
│   ├── encoder-epoch=00-val_loss=0.1234.ckpt
│   ├── encoder-epoch=01-val_loss=0.0987.ckpt
│   └── last.ckpt
├── logs/
│   └── encoder-training/
│       └── version_0/
│           └── events.out.tfevents.*
└── final_model/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## Monitoring Training

### TensorBoard (Default)

```bash
tensorboard --logdir outputs/encoder_training/logs
```

Then open http://localhost:6006 in your browser.

### Weights & Biases (Optional)

1. Set `use_wandb: true` in config
2. Login to wandb: `wandb login`
3. Training metrics will be logged to your W&B account

## Advanced Usage

### Multi-GPU Training

Edit `config/train_config.yaml`:
```yaml
training:
  devices: 2  # Use 2 GPUs
  # or
  devices: -1  # Use all available GPUs
```

### Gradient Accumulation

For effective larger batch sizes with limited GPU memory:
```yaml
training:
  batch_size: 8
  accumulate_grad_batches: 4  # Effective batch size = 8 * 4 = 32
```

### Freeze Base Layers

To only train the top layers:
```yaml
model:
  freeze_layers: 8  # Freeze first 8 encoder layers
```

### Mixed Precision Training

For faster training with less memory:
```yaml
training:
  precision: "16-mixed"  # or "bf16-mixed" for newer GPUs
```

## Using the Trained Model

After training, load and use your model:

```python
from transformers import AutoTokenizer, AutoModel
import torch

# Load the fine-tuned model
model_path = "outputs/encoder_training/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Use it
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
cls_embedding = outputs.last_hidden_state[0, 0, :]

print(f"Embedding shape: {cls_embedding.shape}")
```

Or use with your original `emerge_text` class:

```python
from hf import emerge_text

# Initialize with your fine-tuned model
encoder = emerge_text(model_id="outputs/encoder_training/final_model")
embedding = encoder.get_embeddings("Your text here")
```

## Metrics

During training, the following metrics are tracked:
- **train_loss**: MSE loss on training data
- **val_loss**: MSE loss on validation data
- **val_cosine_similarity**: Cosine similarity between predicted and target embeddings
- **learning_rate**: Current learning rate

## Troubleshooting

### Out of Memory (OOM)

1. Reduce batch size in config
2. Enable gradient accumulation
3. Use mixed precision training
4. Reduce number of data loading workers
5. Freeze more base layers

### Data Loading Issues

- Check CSV format matches expected structure
- Verify all vectors have the same dimension
- Check file naming contains 'train' or 'val'
- Run `python encoder_dataset.py --data_dir your/data/dir` to test

### Slow Training

- Increase `num_workers` for data loading
- Use mixed precision (`16-mixed` or `bf16-mixed`)
- Reduce validation frequency (`val_check_interval`)
- Use multiple GPUs if available

## Example Workflow

```bash
# 1. Install dependencies
pip install -r requirements-training.txt

# 2. Prepare your data
# Place CSV files in data/embeddings/

# 3. Test data loading
python encoder_dataset.py --data_dir data/embeddings --split train

# 4. Start training
python train_encoder.py --config config/train_config.yaml

# 5. Monitor training
tensorboard --logdir outputs/encoder_training/logs

# 6. Use trained model
python -c "
from hf import emerge_text
encoder = emerge_text(model_id='outputs/encoder_training/final_model')
print(encoder.get_embeddings('Test sentence'))
"
```

## License

Same as the parent ModernBERT project.

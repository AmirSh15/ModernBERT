# Complete PyTorch Lightning Training Codebase

## 📦 What I've Created

I've built a complete, production-ready training system for fine-tuning your ModernBERT encoder with MSE loss. Here's everything included:

### Core Training Files

1. **`train_encoder.py`** - Main training script
   - PyTorch Lightning module with MSE loss
   - Automatic learning rate scheduling (warmup + decay)
   - Model checkpointing and early stopping
   - TensorBoard and W&B logging support
   - Mixed precision training
   - Multi-GPU support

2. **`encoder_dataset.py`** - Dataset and data loading
   - Reads CSV files with sentences and target vectors
   - Supports two CSV formats (separate columns or single vector column)
   - Automatic train/val split detection
   - Built-in data validation

3. **`hf.py`** - Your original encoder (already existed)
   - `emerge_text` class wrapper around ModernBERT

### Configuration

4. **`config/train_config.yaml`** - Comprehensive training configuration
   - Model settings (model ID, layer freezing)
   - Hyperparameters (learning rate, batch size, epochs)
   - Hardware settings (GPU, precision, workers)
   - Logging preferences

### Utilities

5. **`prepare_data.py`** - Data preparation utilities
   - Split single CSV into train/val/test
   - Create example/dummy data for testing
   - Convert data to required format

6. **`test_encoder.py`** - Inference and testing script
   - Load and test trained models
   - Single or batch encoding
   - Compute similarity between texts
   - Interactive mode

7. **`quickstart.sh`** - One-command setup and training
   - Creates virtual environment
   - Installs dependencies
   - Creates example data if needed
   - Starts training

### Documentation

8. **`TRAINING_README.md`** - Comprehensive documentation
   - Setup instructions
   - Usage examples
   - Configuration guide
   - Troubleshooting tips

9. **`requirements-training.txt`** - Python dependencies
   - PyTorch, PyTorch Lightning
   - Transformers, pandas, numpy
   - Logging tools

### Example Data

10. **`data/embeddings/example_train.csv`** - Example training data
11. **`data/embeddings/example_val.csv`** - Example validation data

## 🚀 Quick Start

### Option 1: Use the Quick Start Script (Recommended)

```bash
./quickstart.sh
```

This will:
- Set up virtual environment
- Install dependencies
- Create example data
- Start training

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements-training.txt

# Create example data (or prepare your own)
python prepare_data.py example --output_dir data/embeddings --num_samples 1000 --vector_dim 768

# Start training
python train_encoder.py --config config/train_config.yaml

# Monitor with TensorBoard
tensorboard --logdir outputs/encoder_training/logs
```

## 📊 CSV Data Format

Your CSV files should have:

**Option A: Separate dimension columns**
```csv
sentence,dim_0,dim_1,dim_2,...,dim_767
"Your sentence",0.123,-0.456,0.789,...,0.321
```

**Option B: Single vector column**
```csv
sentence,vector
"Your sentence","[0.123, -0.456, 0.789, ..., 0.321]"
```

File naming:
- Training: Include `train` in filename (e.g., `my_train.csv`)
- Validation: Include `val` in filename (e.g., `my_val.csv`)

## 🎯 Key Features

### Training Features
- ✅ MSE loss between CLS embeddings and target vectors
- ✅ Cosine similarity metric for validation
- ✅ Linear warmup + linear decay LR scheduling
- ✅ Weight decay with proper parameter grouping
- ✅ Gradient clipping
- ✅ Mixed precision training (16-bit or bfloat16)
- ✅ Multi-GPU support
- ✅ Gradient accumulation
- ✅ Layer freezing capability

### Data Features
- ✅ Automatic CSV parsing (two formats supported)
- ✅ Train/val split detection
- ✅ Data validation and error handling
- ✅ Efficient batching

### Monitoring
- ✅ TensorBoard logging
- ✅ Weights & Biases integration
- ✅ Automatic checkpoint saving
- ✅ Early stopping
- ✅ Learning rate monitoring

## 📁 Output Structure

```
outputs/encoder_training/
├── checkpoints/
│   ├── encoder-epoch=00-val_loss=0.1234.ckpt
│   ├── encoder-epoch=01-val_loss=0.0987.ckpt
│   └── last.ckpt
├── logs/
│   └── encoder-training/
│       └── version_0/
└── final_model/
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files
```

## 🧪 Testing the Trained Model

### Command Line
```bash
# Encode single text
python test_encoder.py --model_path outputs/encoder_training/final_model \
    --text "Your test sentence"

# Encode multiple texts
python test_encoder.py --model_path outputs/encoder_training/final_model \
    --texts "First sentence" "Second sentence" "Third sentence"

# Compare similarity
python test_encoder.py --model_path outputs/encoder_training/final_model \
    --compare "Machine learning" "Deep learning"

# Interactive mode
python test_encoder.py --model_path outputs/encoder_training/final_model
```

### In Python
```python
from test_encoder import TrainedEncoder

# Load model
encoder = TrainedEncoder("outputs/encoder_training/final_model")

# Encode text
embedding = encoder.encode("Your text here")

# Batch encoding
embeddings = encoder.encode_batch(["Text 1", "Text 2", "Text 3"])

# Similarity
score = encoder.similarity("Text A", "Text B")
```

## 🔧 Common Customizations

### Change Model
```yaml
# In config/train_config.yaml
model:
  model_id: "answerdotai/ModernBERT-large"  # Use larger model
```

### Adjust Learning
```yaml
training:
  learning_rate: 1e-5  # Lower for fine-tuning
  batch_size: 32       # Larger batch
  max_epochs: 20       # More epochs
```

### Freeze Layers
```yaml
model:
  freeze_layers: 8  # Only train top layers
```

### Multi-GPU
```yaml
training:
  devices: 2  # Use 2 GPUs
```

## 📈 Monitoring Metrics

During training, you'll see:
- **train_loss**: MSE loss on training data
- **val_loss**: MSE loss on validation data
- **val_cosine_similarity**: How similar predictions are to targets (0-1)
- **lr**: Current learning rate

## 🔍 Troubleshooting

### Out of Memory
1. Reduce `batch_size` in config
2. Use `accumulate_grad_batches` for effective larger batch
3. Enable mixed precision (`16-mixed`)
4. Freeze more layers

### Slow Training
1. Increase `num_workers`
2. Use mixed precision
3. Use multiple GPUs

### Data Issues
```bash
# Test your data first
python encoder_dataset.py --data_dir data/embeddings --split train
```

## 📝 Next Steps

1. **Prepare your actual data**
   - Create CSV files with your sentences and target vectors
   - Place in `data/embeddings/` directory

2. **Adjust configuration**
   - Edit `config/train_config.yaml`
   - Set appropriate hyperparameters

3. **Train the model**
   ```bash
   python train_encoder.py --config config/train_config.yaml
   ```

4. **Monitor training**
   ```bash
   tensorboard --logdir outputs/encoder_training/logs
   ```

5. **Test the trained model**
   ```bash
   python test_encoder.py --model_path outputs/encoder_training/final_model
   ```

6. **Use in production**
   ```python
   from hf import emerge_text
   encoder = emerge_text(model_id="outputs/encoder_training/final_model")
   embedding = encoder.get_embeddings("Your text")
   ```

## 📚 Files Overview

| File | Purpose | Required? |
|------|---------|-----------|
| `train_encoder.py` | Main training script | ✅ Yes |
| `encoder_dataset.py` | Data loading | ✅ Yes |
| `config/train_config.yaml` | Configuration | ✅ Yes |
| `hf.py` | Your encoder model | ✅ Yes |
| `requirements-training.txt` | Dependencies | ✅ Yes |
| `prepare_data.py` | Data preparation utility | Optional |
| `test_encoder.py` | Testing utility | Optional |
| `quickstart.sh` | Setup automation | Optional |
| `TRAINING_README.md` | Documentation | Optional |
| `data/embeddings/*.csv` | Your training data | ✅ Yes |

## 🎓 Learning Resources

- See `TRAINING_README.md` for detailed documentation
- Check `config/train_config.yaml` for all available options
- Run scripts with `--help` for usage information

Happy training! 🚀

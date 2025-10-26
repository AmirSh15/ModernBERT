# Training Guide for Music Caption Data

## Your Data Format

Your CSV files have the following structure:
- **`caption_text`**: The input text (music descriptions/captions)
- **`vector`**: The target embedding vector (768-dimensional, comma-separated in brackets)

Example row:
```csv
caption_text,vector
"This is a smooth blues track with soulful vocals","[-0.148,-0.144,-0.179,...]"
```

## Quick Start for Your Data

### Step 1: Prepare Your Data

You have multiple CSV files in your local drive. Let's organize them:

```bash
# Option 1: Validate your CSV files first
python prepare_music_data.py validate --input /path/to/your/file.csv

# Option 2: Combine and split multiple CSV files into train/val/test
python prepare_music_data.py split \
    --input /path/to/file1.csv /path/to/file2.csv /path/to/file3.csv \
    --output_dir data/music_embeddings \
    --train_ratio 0.8 \
    --val_ratio 0.15 \
    --test_ratio 0.05
```

This will create:
- `data/music_embeddings/train.csv`
- `data/music_embeddings/val.csv`
- `data/music_embeddings/test.csv`

### Step 2: Update Configuration

Edit `config/train_config.yaml`:

```yaml
data:
  data_dir: "data/music_embeddings"  # Your data directory
  text_column: "caption_text"         # Already configured!
  vector_column: "vector"             # Already configured!
```

### Step 3: Test Data Loading

```bash
python encoder_dataset.py --data_dir data/music_embeddings --split train
```

You should see:
```
INFO:__main__:Loading data from X CSV file(s)...
INFO:__main__:Reading train.csv...
INFO:__main__:Loaded XXXX samples for train split
INFO:__main__:Vector dimension: 768
```

### Step 4: Start Training

```bash
python train_encoder.py --config config/train_config.yaml
```

Or with custom data directory:

```bash
python train_encoder.py \
    --config config/train_config.yaml \
    --data_dir data/music_embeddings \
    --output_dir outputs/music_encoder
```

### Step 5: Monitor Training

```bash
tensorboard --logdir outputs/music_encoder/logs
```

Open http://localhost:6006 in your browser.

## Important Notes for Your Data

### 1. Multiple CSV Files

If you have multiple CSV files, you can:

**Option A: Keep them separate** (recommended for large datasets)
- Name them with train/val in the filename:
  - `music_train_1.csv`, `music_train_2.csv`, ...
  - `music_val_1.csv`, `music_val_2.csv`, ...
- The dataset will automatically load all files matching the split

**Option B: Combine them first**
```bash
python prepare_music_data.py split \
    --input file1.csv file2.csv file3.csv \
    --output_dir data/music_embeddings
```

### 2. Vector Format

Your vectors are stored as: `[-0.148, -0.144, -0.179, ...]`

The dataset will automatically:
- Remove the brackets `[]`
- Split by comma
- Convert to numpy array
- Create PyTorch tensor

### 3. Missing Values

If some rows have missing `caption_text` or `vector`, they will be automatically skipped with a warning.

### 4. Data Statistics

Check your data before training:

```bash
# Count total samples
wc -l data/music_embeddings/train.csv
wc -l data/music_embeddings/val.csv

# View first few rows
head -n 5 data/music_embeddings/train.csv
```

## Recommended Configuration for Music Data

Based on your data (music captions with 768-dim vectors), here's a recommended config:

```yaml
# In config/train_config.yaml

model:
  model_id: "answerdotai/ModernBERT-base"  # 768-dim output matches your vectors
  freeze_layers: 0  # Train all layers

data:
  data_dir: "data/music_embeddings"
  text_column: "caption_text"
  vector_column: "vector"

training:
  batch_size: 32  # Adjust based on GPU memory
  learning_rate: 2e-5  # Good default for fine-tuning
  max_epochs: 10
  warmup_steps: 1000  # 10% of total steps is common
  
  # GPU settings
  devices: 1  # Single GPU
  precision: "16-mixed"  # Fast training, less memory
  
  # Monitoring
  val_check_interval: 0.5  # Validate twice per epoch
  log_every_n_steps: 100
```

## Example Workflow

```bash
# 1. Validate one of your CSV files
python prepare_music_data.py validate --input /your/data/file.csv

# 2. Combine all your CSV files and split
python prepare_music_data.py split \
    --input /your/data/*.csv \
    --output_dir data/music_embeddings

# 3. Test data loading
python encoder_dataset.py --data_dir data/music_embeddings --split train

# 4. Start training
python train_encoder.py --config config/train_config.yaml

# 5. In another terminal, monitor training
tensorboard --logdir outputs/encoder_training/logs

# 6. After training, test the model
python test_encoder.py \
    --model_path outputs/encoder_training/final_model \
    --text "This is a smooth blues track with soulful vocals"
```

## Memory Optimization

If you run into GPU memory issues with large datasets:

```yaml
training:
  batch_size: 16  # Reduce batch size
  accumulate_grad_batches: 2  # Effective batch = 16 * 2 = 32
  precision: "16-mixed"  # Use mixed precision
  num_workers: 2  # Reduce data loading workers
```

## Expected Results

For music caption embeddings:
- **Initial val_loss**: ~0.5-1.0 (depends on initial random weights vs target)
- **Target val_loss**: <0.1 (good alignment)
- **Cosine similarity**: >0.8 (high similarity between predicted and target)

## Troubleshooting

### "No CSV files found"
- Check the `data_dir` path in config
- Ensure files have `.csv` extension
- Check file permissions

### "Missing caption_text column"
- Validate your CSV with: `python prepare_music_data.py validate --input your_file.csv`
- Ensure no typos in column names

### "Vector dimension mismatch"
- All vectors must have the same dimension (768)
- Check for corrupted rows with: `python prepare_music_data.py validate --input your_file.csv`

### Training is slow
- Reduce `batch_size`
- Increase `num_workers` (but not too high, 4-8 is usually good)
- Use `precision: "16-mixed"`

## Next Steps

After training:

1. **Evaluate on test set** (if you created one)
2. **Compare predictions to targets** manually
3. **Test on new music descriptions** not in training data
4. **Deploy** the model for your music application

## Using the Trained Model

```python
from hf import emerge_text

# Load your trained model
encoder = emerge_text(model_id="outputs/encoder_training/final_model")

# Encode new music descriptions
caption = "This is a smooth blues track with soulful vocals"
embedding = encoder.get_embeddings(caption)

print(f"Embedding shape: {embedding.shape}")  # Should be [768]
print(f"Embedding: {embedding[:10]}")  # First 10 dimensions
```

Good luck with your training! 🎵

# Training Checklist

Use this checklist to ensure you have everything set up correctly before training.

## ☐ Pre-Training Checklist

### Environment Setup
- [ ] Python 3.8+ installed
- [ ] Virtual environment created (optional but recommended)
- [ ] Dependencies installed from `requirements-training.txt`
  ```bash
  pip install -r requirements-training.txt
  ```

### Data Preparation
- [ ] CSV files created with sentences and target vectors
- [ ] Files use correct format:
  - Option A: `sentence,dim_0,dim_1,...,dim_N`
  - Option B: `sentence,vector`
- [ ] Training file(s) contain "train" in filename
- [ ] Validation file(s) contain "val" in filename
- [ ] Files placed in `data/embeddings/` directory (or custom path)
- [ ] All vectors have the same dimension
- [ ] No missing values in the data
- [ ] Data tested with:
  ```bash
  python encoder_dataset.py --data_dir data/embeddings --split train
  ```

### Configuration
- [ ] Reviewed `config/train_config.yaml`
- [ ] Set correct `data_dir` path
- [ ] Adjusted `batch_size` based on GPU memory
- [ ] Set appropriate `learning_rate` (2e-5 is a good default)
- [ ] Configured `max_epochs` or `max_steps`
- [ ] Set `output_dir` for checkpoints
- [ ] Decided on precision (`16-mixed`, `bf16-mixed`, or `32`)

### GPU/Hardware
- [ ] Checked GPU availability:
  ```bash
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
  ```
- [ ] Set correct number of `devices` in config
- [ ] Ensured enough disk space for checkpoints (~2GB per checkpoint)

## ☐ During Training Checklist

### Monitoring
- [ ] TensorBoard started:
  ```bash
  tensorboard --logdir outputs/encoder_training/logs
  ```
- [ ] Training metrics being logged:
  - train_loss decreasing
  - val_loss decreasing
  - val_cosine_similarity increasing (closer to 1.0)
  - Learning rate following schedule

### Health Checks
- [ ] No NaN or Inf in losses
- [ ] GPU utilization reasonable (check with `nvidia-smi`)
- [ ] No out-of-memory errors
- [ ] Checkpoints being saved regularly
- [ ] Validation running at expected intervals

### Adjustments (if needed)
- [ ] Reduce batch_size if OOM
- [ ] Increase/decrease learning_rate if not converging
- [ ] Enable gradient accumulation for larger effective batch size
- [ ] Adjust early_stopping_patience if needed

## ☐ Post-Training Checklist

### Verify Outputs
- [ ] Final model saved in `output_dir/final_model/`
- [ ] Checkpoints saved in `output_dir/checkpoints/`
- [ ] Logs available for review
- [ ] Training completed without errors

### Test Model
- [ ] Load and test the model:
  ```bash
  python test_encoder.py --model_path outputs/encoder_training/final_model \
      --text "Test sentence"
  ```
- [ ] Embeddings have expected dimension
- [ ] Embeddings are reasonable (not all zeros, not NaN)

### Evaluation
- [ ] Check final validation loss
- [ ] Review cosine similarity scores
- [ ] Test on sample sentences similar to training data
- [ ] Test on sentences different from training data
- [ ] Compare embeddings to target vectors manually

### Deployment
- [ ] Model saved in HuggingFace format
- [ ] Can load with transformers library
- [ ] Can integrate with existing code
- [ ] Performance is acceptable for use case

## ☐ Troubleshooting Checklist

### If training is slow:
- [ ] Increase `num_workers` in config
- [ ] Use mixed precision (`16-mixed`)
- [ ] Profile with PyTorch profiler
- [ ] Check data loading isn't bottleneck

### If loss not decreasing:
- [ ] Check data is correct (no label mismatch)
- [ ] Try different learning rate (1e-5, 5e-5)
- [ ] Increase warmup steps
- [ ] Check if model is frozen accidentally
- [ ] Verify loss calculation is correct

### If out of memory:
- [ ] Reduce batch_size
- [ ] Enable gradient accumulation
- [ ] Use mixed precision
- [ ] Freeze more layers
- [ ] Use smaller model variant

### If validation loss increasing:
- [ ] Overfitting - reduce max_epochs
- [ ] Add weight_decay
- [ ] Use early stopping
- [ ] Get more training data
- [ ] Use data augmentation

## ☐ Production Checklist

### Before deployment:
- [ ] Model tested thoroughly
- [ ] Performance benchmarked
- [ ] Inference speed acceptable
- [ ] Memory usage acceptable
- [ ] Model versioned/tagged
- [ ] Documentation updated

### Integration:
- [ ] Model can be loaded in production environment
- [ ] Dependencies compatible
- [ ] API/interface defined
- [ ] Error handling implemented
- [ ] Monitoring in place

## Quick Command Reference

```bash
# Test data loading
python encoder_dataset.py --data_dir data/embeddings --split train

# Start training
python train_encoder.py --config config/train_config.yaml

# Monitor training
tensorboard --logdir outputs/encoder_training/logs

# Test model
python test_encoder.py --model_path outputs/encoder_training/final_model \
    --text "Your test text"

# Create example data
python prepare_data.py example --output_dir data/embeddings

# View architecture
python architecture_diagram.py
```

## Notes

- [ ] Training time estimate: ~______ hours/minutes
- [ ] Expected final val_loss: ~______
- [ ] Expected cosine similarity: ~______
- [ ] Any custom modifications made: ________________

---

**Last updated:** [Current date]
**Reviewed by:** [Your name]

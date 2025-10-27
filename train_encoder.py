import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch.utils.data import DataLoader
import argparse
import yaml
from pathlib import Path

from hf import emerge_text
from encoder_dataset import EmbeddingDataset, get_collate_fn
from s3_download import main as download_s3_data
from datetime import datetime
import glob


def find_latest_checkpoint(output_dir: Path) -> str:
    """
    Find the latest checkpoint in the output directory.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        Path to the latest checkpoint file, or None if no checkpoint exists
    """
    checkpoint_dir = output_dir / 'checkpoints'
    
    if not checkpoint_dir.exists():
        return None
    
    # Look for last.ckpt first (saved by ModelCheckpoint with save_last=True)
    last_ckpt = checkpoint_dir / 'last.ckpt'
    if last_ckpt.exists():
        print(f"Found last checkpoint: {last_ckpt}")
        return str(last_ckpt)
    
    # Otherwise, find the most recent checkpoint file
    checkpoint_files = list(checkpoint_dir.glob('*.ckpt'))
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time (most recent first)
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    print(f"Found latest checkpoint: {latest_checkpoint}")
    
    return str(latest_checkpoint)


class EmbeddingTrainingModule(pl.LightningModule):
    """PyTorch Lightning module for training the encoder with MSE loss."""
    
    def __init__(
        self,
        model_id: str = "answerdotai/ModernBERT-base",
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 10000,
        freeze_layers: int = 0,
        use_projection: bool = False,
        projection_hidden_dim: int = 3072,
        output_dim: int = 768,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize the model
        self.encoder = emerge_text(
            model_id=model_id,
            use_projection=use_projection,
            projection_hidden_dim=projection_hidden_dim,
            output_dim=output_dim
        )
        
        # Optionally freeze some layers, if -1 is passed, all layers are frozen
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        elif freeze_layers == -1:
            self._freeze_layers(len(self.encoder.model.layers))  # +1 for embeddings
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # For tracking
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        
    def _freeze_layers(self, num_layers: int):
        """Freeze the first num_layers of the encoder."""
        # Freeze embeddings
        for param in self.encoder.model.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze encoder layers
        for i, layer in enumerate(self.encoder.model.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
                    
    def forward(self, text):
        """Forward pass through the encoder."""
        return self.encoder.get_embeddings(text)
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_vectors = batch['target_vectors']
        
        # Get embeddings from the model
        embeddings = self.encoder.get_embeddings(input_ids, attention_mask)
        
        # Calculate MSE loss
        loss = self.criterion(embeddings, target_vectors)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        target_vectors = batch['target_vectors']
        
        # Get embeddings from the model
        embeddings = self.encoder.get_embeddings(input_ids, attention_mask)
        
        # Calculate MSE loss
        loss = self.criterion(embeddings, target_vectors)
        
        # Calculate cosine similarity as an additional metric
        cos_sim = nn.functional.cosine_similarity(embeddings, target_vectors, dim=1).mean()
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cosine_similarity', cos_sim, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Separate parameters for weight decay
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [p for n, p in self.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            }
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Linear warmup + linear decay scheduler
        def lr_lambda(current_step: int):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return max(
                0.0,
                float(self.max_steps - current_step) / float(max(1, self.max_steps - self.warmup_steps))
            )
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }


class TextEncodingCapabilityCallback(Callback):
    """
    Callback to monitor the model's text encoding capabilities during training.
    Tests on semantic similarity tasks to detect catastrophic forgetting.
    """
    
    def __init__(self, check_every_n_epochs: int = 1):
        super().__init__()
        self.check_every_n_epochs = check_every_n_epochs
        
        # High similarity pairs
        self.similar_pairs = [
            ("The cat sat on the mat", "A cat is sitting on a mat"),
            ("I love eating pizza", "Pizza is my favorite food"),
            ("The weather is sunny today", "Today the sun is shining"),
            ("She is reading a book", "A woman is looking at a book"),
            ("The dog is playing in the garden", "A dog plays outside in the yard"),
            ("He drives a red car", "His car is red and he drives it"),
            ("The baby is sleeping peacefully", "A baby sleeps quietly"),
            ("Students are studying for exams", "Pupils are preparing for their tests"),
            ("The mountain is covered with snow", "Snow covers the mountain peak"),
            ("She enjoys listening to music", "Music is something she likes to hear"),
            ("The train arrives at noon", "At 12 PM the train gets here"),
            ("Birds are flying in the sky", "Several birds fly above us"),
            ("He is cooking dinner", "A man prepares an evening meal"),
            ("The flowers smell wonderful", "These flowers have a beautiful fragrance"),
            ("Children are playing soccer", "Kids play football together"),
        ]
        
        # Low similarity pairs
        self.dissimilar_pairs = [
            ("The cat sat on the mat", "Quantum physics is fascinating"),
            ("I love eating pizza", "The stock market crashed yesterday"),
            ("Machine learning is powerful", "The ocean is very deep"),
            ("She is reading a book", "Robots are taking over manufacturing"),
            ("The dog is playing in the garden", "Mathematics requires logical thinking"),
            ("He drives a red car", "Ancient civilizations built pyramids"),
            ("The baby is sleeping peacefully", "Climate change affects polar ice caps"),
            ("Students are studying for exams", "Volcanoes erupt with molten lava"),
            ("The mountain is covered with snow", "Bacteria are microscopic organisms"),
            ("She enjoys listening to music", "The periodic table organizes elements"),
            ("The train arrives at noon", "Photosynthesis produces oxygen"),
            ("Birds are flying in the sky", "Democracy requires active participation"),
            ("He is cooking dinner", "Binary code consists of zeros and ones"),
            ("The flowers smell wonderful", "Economics studies resource allocation"),
            ("Children are playing soccer", "DNA contains genetic information"),
            ("Coffee keeps me awake", "Galaxies contain billions of stars"),
            ("The sunset is beautiful tonight", "Algorithms solve computational problems"),
            ("My phone battery is low", "Renaissance art revolutionized painting"),
            ("She won the marathon race", "Protein synthesis occurs in ribosomes"),
            ("The library is quiet today", "Nuclear fusion powers the sun"),
        ]
        
        # Paraphrase pairs (should be very similar)
        self.paraphrase_pairs = [
            ("The quick brown fox jumps over the lazy dog", 
             "A fast brown fox leaps over a lazy dog"),
            ("Scientists discovered a new species", 
             "A new species was discovered by scientists"),
            ("The company announced record profits", 
             "Record profits were announced by the company"),
            ("She completed the project ahead of schedule",
             "The project was finished early by her"),
            ("The teacher explained the concept clearly",
             "The concept was clearly explained by the teacher"),
            ("Heavy rain caused flooding in the city",
             "Flooding in the city was caused by heavy rain"),
            ("The artist painted a beautiful landscape",
             "A beautiful landscape was painted by the artist"),
            ("Researchers developed a new vaccine",
             "A new vaccine was developed by researchers"),
            ("The storm damaged several buildings",
             "Several buildings were damaged by the storm"),
            ("The chef prepared an excellent meal",
             "An excellent meal was prepared by the chef"),
            ("The committee approved the proposal unanimously",
             "The proposal was unanimously approved by the committee"),
            ("The movie captivated audiences worldwide",
             "Audiences worldwide were captivated by the movie"),
        ]
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Run text encoding capability tests at the end of validation."""
        if trainer.current_epoch % self.check_every_n_epochs != 0:
            return
        
        pl_module.eval()
        
        with torch.no_grad():
            # Test similar pairs
            similar_scores = []
            for text1, text2 in self.similar_pairs:
                emb1 = pl_module([text1])
                emb2 = pl_module([text2])
                similarity = nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
                similar_scores.append(similarity)
            
            # Test dissimilar pairs
            dissimilar_scores = []
            for text1, text2 in self.dissimilar_pairs:
                emb1 = pl_module([text1])
                emb2 = pl_module([text2])
                similarity = nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
                dissimilar_scores.append(similarity)
            
            # Test paraphrase pairs
            paraphrase_scores = []
            for text1, text2 in self.paraphrase_pairs:
                emb1 = pl_module([text1])
                emb2 = pl_module([text2])
                similarity = nn.functional.cosine_similarity(emb1, emb2, dim=1).item()
                paraphrase_scores.append(similarity)
            
            # Calculate metrics
            avg_similar = sum(similar_scores) / len(similar_scores)
            avg_dissimilar = sum(dissimilar_scores) / len(dissimilar_scores)
            avg_paraphrase = sum(paraphrase_scores) / len(paraphrase_scores)
            
            # Separation score: higher is better (similar should be high, dissimilar low)
            separation_score = avg_similar - avg_dissimilar
            
            # Log metrics
            pl_module.log('encoding_capability/similar_avg', avg_similar, on_epoch=True)
            pl_module.log('encoding_capability/dissimilar_avg', avg_dissimilar, on_epoch=True)
            pl_module.log('encoding_capability/paraphrase_avg', avg_paraphrase, on_epoch=True)
            pl_module.log('encoding_capability/separation_score', separation_score, on_epoch=True)
            

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train embedding encoder with MSE loss')
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--data_dir', type=str, default=None,
                      help='Directory containing CSV files (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                      help='Output directory for checkpoints (overrides config)')
    
    args = parser.parse_args()

    # Download data from S3
    download_s3_data()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir
    
    # Set seed for reproducibility
    pl.seed_everything(config['training'].get('seed', 42))
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize datasets
    train_dataset = EmbeddingDataset(
        data_dir=config['data']['data_dir'],
        split='train',
        max_length=config['data'].get('max_length', 512),
        text_column=config['data'].get('text_column', 'caption_text'),
        vector_column=config['data'].get('vector_column', 'vector'),
        model_id=config['model']['model_id'],
    )
    
    val_dataset = EmbeddingDataset(
        data_dir=config['data']['data_dir'],
        split='val',
        max_length=config['data'].get('max_length', 512),
        text_column=config['data'].get('text_column', 'caption_text'),
        vector_column=config['data'].get('vector_column', 'vector'),
        model_id=config['model']['model_id'],
    )
    
    # If val_dataset is empty, split train_dataset into train and val
    if len(val_dataset) == 0:
        print("Validation dataset is empty. Splitting train dataset into train and val...")
        from torch.utils.data import random_split
        
        # Calculate split sizes (e.g., 90% train, 10% val)
        val_split_ratio = config['data'].get('val_split_ratio', 0.1)
        total_size = len(train_dataset)
        val_size = int(total_size * val_split_ratio)
        train_size = total_size - val_size
        
        print(f"Splitting {total_size} samples into {train_size} train and {val_size} val samples")
        
        # Split the dataset
        train_dataset, val_dataset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(config['training'].get('seed', 42))
        )
    
    # Get the tokenizer from the original train_dataset (before potential split)
    # If train_dataset is a Subset (after random_split), get the tokenizer from the base dataset
    if hasattr(train_dataset, 'dataset'):
        tokenizer = train_dataset.dataset.tokenizer
    else:
        tokenizer = train_dataset.tokenizer
    
    # Create collate function with the tokenizer
    collate_fn = get_collate_fn(tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    ) if len(val_dataset) > 0 else None
    
    # Initialize model
    model = EmbeddingTrainingModule(
        model_id=config['model']['model_id'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_steps=config['training']['warmup_steps'],
        max_steps=config['training']['max_steps'],
        freeze_layers=config['model'].get('freeze_layers', 0),
        use_projection=config['model'].get('use_projection', False),
        projection_hidden_dim=config['model'].get('projection_hidden_dim', 3072),
        output_dim=config['model'].get('output_dim', 768),
    )
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='encoder-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training'].get('early_stopping_patience', 5),
            mode='min',
        ),
        LearningRateMonitor(logging_interval='step'),
        TextEncodingCapabilityCallback(
            check_every_n_epochs=config['training'].get('encoding_check_interval', 1)
        ),
    ]
    
    # Logger
    if config['training'].get('use_wandb', False):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = config['training'].get('run_name', f'encoder-mse') + f'-{timestamp}'
        logger = WandbLogger(
            project=config['training'].get('wandb_project', 'emerge_modernbert'),
            name=run_name,
            save_dir=output_dir,
        )
        # Log configuration file to wandb
        logger.experiment.config.update(config)
        # Save config file as artifact
        logger.experiment.save(args.config, policy='now')
    else:
        logger = TensorBoardLogger(
            save_dir=output_dir / 'logs',
            name='encoder-training',
        )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        max_steps=config['training']['max_steps'],
        accelerator='auto',
        devices=config['training'].get('devices', 0),
        precision=config['training'].get('precision', '16-mixed'),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        val_check_interval=config['training'].get('val_check_interval', 1.0 if val_loader else None),
        log_every_n_steps=config['training'].get('log_every_n_steps', 50),
    )
    
    # Check for resume
    ckpt_path = None
    if config['training'].get('resume', False):
        # Check if a specific checkpoint path is provided
        if config['training'].get('resume_checkpoint'):
            ckpt_path = config['training']['resume_checkpoint']
            if not Path(ckpt_path).exists():
                print(f"Warning: Specified checkpoint {ckpt_path} does not exist. Starting from scratch.")
                ckpt_path = None
            else:
                print(f"Resuming from specified checkpoint: {ckpt_path}")
        else:
            # Auto-detect the latest checkpoint
            ckpt_path = find_latest_checkpoint(output_dir)
            if ckpt_path:
                print(f"Resuming training from checkpoint: {ckpt_path}")
            else:
                print("No checkpoint found. Starting training from scratch.")
    else:
        print("Resume is disabled. Starting training from scratch.")
    
    # Train
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path)
    
    # Save final model
    final_model_path = output_dir / 'final_model'
    final_model_path.mkdir(parents=True, exist_ok=True)
    
    # Save tokenizer and model
    model.encoder.tokenizer.save_pretrained(final_model_path)
    model.encoder.model.save_pretrained(final_model_path)
    
    print(f"\nTraining completed! Model saved to {final_model_path}")


if __name__ == '__main__':
    main()

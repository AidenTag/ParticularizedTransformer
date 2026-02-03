import argparse
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from src.model import ListOpsTransformer, GPTConfig
from src.dataset import ListOpsDataModule

def main():
    parser = argparse.ArgumentParser(description="Train DALex Transformer on ListOps")
    
    # Data args
    parser.add_argument('--data_dir', type=str, default='data/lra_release/listops-1000', help='Path to ListOps data directory')
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    
    # Model args
    parser.add_argument('--n_layer', type=int, default=4, help='Number of layers')
    parser.add_argument('--n_head', type=int, default=8, help='Number of heads')
    parser.add_argument('--n_embd', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # DALex args
    parser.add_argument('--dalex_pressure', type=float, default=0.5, help='DALex particularity pressure')
    parser.add_argument('--disable_dalex', action='store_true', help='Disable DALex (use Standard Attention equivalent)')
    
    # Training args
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum epochs')
    parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0, help='Number of GPUs to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    pl.seed_everything(args.seed)
    
    # 1. Setup Data
    # Ensure data dir exists or warn
    if not os.path.exists(args.data_dir):
        print(f"Directory {args.data_dir} does not exist. Please download LRA ListOps dataset.")
        # Create dummy directory structure to prevent immediate crash if user just testing
        # os.makedirs(args.data_dir, exist_ok=True) 
    
    dm = ListOpsDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    # We need to setup dm to access vocab size, but typically we want to defer
    # For now, let's just instantiate validation to peek at vocab size or assume fixed
    # dm.setup() # This might fail if files don't exist
    # If using fixed vocab in dataset.py, size is fixed.
    vocab_size = 15 # 0-9 + PAD + 4 ops + parens -> roughly 16.
    # From dataset.py: PAD, (, ), MIN, MAX, MED, SUM_MOD, 0-9.
    # Total = 1 + 2 + 4 + 10 = 17.
    vocab_size = 20 # Safe margin
    
    # 2. Setup Model
    config = GPTConfig(
        vocab_size=vocab_size, 
        block_size=args.max_length,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        dalex_pressure=args.dalex_pressure,
        use_dalex=not args.disable_dalex
    )
    
    model = ListOpsTransformer(config, learning_rate=args.lr)
    
    # 3. Setup Trainer
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        mode='max',
        save_top_k=1,
        filename='dalex-listops-{epoch:02d}-{val_acc:.3f}'
    )
    
    # Detect accelerator
    if args.gpus > 0 and torch.cuda.is_available():
        accelerator = "gpu"
        devices = args.gpus
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        gradient_clip_val=1.0,
    )
    
    # 4. Train
    print(f"Starting training with DALex={config.use_dalex} (Pressure={config.dalex_pressure if config.use_dalex else 'N/A'})...")
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()

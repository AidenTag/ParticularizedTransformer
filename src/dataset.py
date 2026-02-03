import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import os
import csv

class ListOpsDataset(Dataset):
    def __init__(self, file_path, vocab=None, max_length=2048):
        self.data = []
        self.max_length = max_length
        
        # Define basic vocabulary if not provided
        if vocab is None:
            self.vocab = {
                "<PAD>": 0,
                "(": 1,
                ")": 2,
                "MIN": 3,
                "MAX": 4,
                "MED": 5,
                "SUM_MOD": 6,  # Sometimes appearing as SM in some versions
            }
            # Add digits 0-9
            for i in range(10):
                self.vocab[str(i)] = len(self.vocab)
        else:
            self.vocab = vocab
            
        self.idx_to_token = {v: k for k, v in self.vocab.items()}
        
        # Load data
        # Expects TSV: Label <tab> Sequence
        # Example: 3    ( MAX 2 3 )
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found. Dataset will be empty.")
            return

        with open(file_path, 'r') as f:
            # Check header
            first_line = f.readline()
            has_header = "Source" in first_line or "Target" in first_line
            if not has_header:
                f.seek(0)
            
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                
                # Try to determine which is source and which is label
                # Valid labels are single digits 0-9
                p0 = parts[0].strip()
                p1 = parts[1].strip()
                
                if p0.isdigit() and len(p0) == 1 and not (p1.isdigit() and len(p1) == 1):
                    # p0 is label, p1 is sequence (reverse format)
                    label = int(p0)
                    sequence = p1
                elif p1.isdigit() and len(p1) == 1:
                    # p1 is label, p0 is sequence (standard format)
                    label = int(p1)
                    sequence = p0
                else:
                    # Ambiguous or header row or corrupted, skip
                    continue
                
                # Tokenize
                # ListOps is space separated usually
                tokens = sequence.strip().split()
                token_ids = [self.vocab.get(t, self.vocab.get("<PAD>")) for t in tokens]
                
                # Truncate or Pad
                if len(token_ids) > self.max_length:
                    token_ids = token_ids[:self.max_length]
                else:
                    token_ids = token_ids + [0] * (self.max_length - len(token_ids))
                
                self.data.append({
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "label": torch.tensor(label, dtype=torch.long)
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["input_ids"], self.data[idx]["label"]

class ListOpsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, max_length=2000, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.vocab = None # Will be set after loading train

    def setup(self, stage=None):
        # We assume standard LRA split filenames
        # LRA release folder structure usually has basic_train.tsv, basic_val.tsv, basic_test.tsv
        train_path = os.path.join(self.data_dir, "basic_train.tsv")
        val_path = os.path.join(self.data_dir, "basic_val.tsv")
        test_path = os.path.join(self.data_dir, "basic_test.tsv")
        
        # Check if files exist, if not look for alternate naming
        if not os.path.exists(train_path):
             # Try alternate naming
             train_path = os.path.join(self.data_dir, "basic_listops_train.tsv")
             val_path = os.path.join(self.data_dir, "basic_listops_val.tsv")
             test_path = os.path.join(self.data_dir, "basic_listops_test.tsv")
        
        # Initialize train to get vocab (though we use fixed vocab mostly)
        self.train_dataset = ListOpsDataset(train_path, max_length=self.max_length)
        self.vocab = self.train_dataset.vocab # Share vocab
        
        self.val_dataset = ListOpsDataset(val_path, vocab=self.vocab, max_length=self.max_length)
        self.test_dataset = ListOpsDataset(test_path, vocab=self.vocab, max_length=self.max_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

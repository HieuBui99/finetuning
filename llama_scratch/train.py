import lightning as pl
import tiktoken
from pathlib import Path
from torch.utils.data import DataLoader
from data import LMDataset
from lightning_module import LitGPT

config = {
    "vocab_size": 50257, 
    "context_length": 256,
    "emb_dim": 512,     
    "n_heads": 8,  
    "n_layers": 8,        
    "hidden_dim": 1024, 
    "drop_rate": 0.0,
    "qkv_bias": False,
    "weight_decay": 0.01,
    "lr": 5e-4,
    "bs": 32,
}

data_path = Path("../data/shakespeare_sample.txt")
tokenizer = tiktoken.get_encoding("gpt2")
with open(data_path) as f:
    text = f.read()
n = len(text)
train_text = text[:int(0.8*n)]
valid_text = text[int(0.8*n):]

train_dataset = LMDataset(train_text, tokenizer, config["context_length"], stride=32)
valid_dataset = LMDataset(valid_text, tokenizer, config["context_length"], stride=32)
train_dl = DataLoader(train_dataset, batch_size=config["bs"], shuffle=True, num_workers=8)
valid_dl = DataLoader(valid_dataset, batch_size=config["bs"], shuffle=False, num_workers=8)

model = LitGPT(config)
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", precision="bf16-mixed")

trainer.fit(model, train_dl, valid_dl)
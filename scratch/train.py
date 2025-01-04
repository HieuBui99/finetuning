import lightning as pl
import tiktoken
from pathlib import Path
from torch.utils.data import DataLoader
from data import LMDataset
from lightning_module import LitGPT

config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 256,
    "n_heads": 4,
    "n_layers": 3,
    "drop_rate": 0.1,
    "qkv_bias": False,
    "weight_decay": 0.01,
    "lr": 5e-4,
    "bs": 16,
    "warmup_steps": 1000,
    "max_lr": 6e-4,
    "min_lr": 6e-5,
}

data_path = Path("../data/shakespeare_sample.txt")
tokenizer = tiktoken.get_encoding("gpt2")
with open(data_path) as f:
    text = f.read()
n = len(text)
train_text = text[:int(0.8*n)]
valid_text = text[int(0.8*n):]

train_dataset = LMDataset(train_text, tokenizer, config["context_length"])
valid_dataset = LMDataset(valid_text, tokenizer, config["context_length"])
train_dl = DataLoader(train_dataset, batch_size=config["bs"], shuffle=True, num_workers=8)
valid_dl = DataLoader(valid_dataset, batch_size=config["bs"], shuffle=False, num_workers=8)

model = LitGPT(config)
trainer = pl.Trainer(max_epochs=10)

trainer.fit(model, train_dl, valid_dl)
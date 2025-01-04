import lightning as pl
import tiktoken
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from data import LMDataset
from lightning_module import LitGPT

ckpt_path = "/home/aki/workspace/learning/finetuning/scratch/lightning_logs/version_0/checkpoints/epoch=9-step=1310.ckpt"

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

model = LitGPT.load_from_checkpoint(ckpt_path, cfg=config)
model.model.eval()
model.model.cuda()

tokenizer = tiktoken.get_encoding("gpt2")

seed_text = "Thou didst"
tokens = tokenizer.encode(seed_text, allowed_special={"<|endoftext|>"})
tokens = torch.tensor(tokens).unsqueeze(0)

generated_text = tokens.cuda()

while generated_text.size(1) < 32:
    logits = model.model(generated_text)
    next_token = torch.argmax(logits[:, -1], dim=-1).unsqueeze(1)
    print(next_token)
    generated_text = torch.cat([generated_text, next_token], dim=-1)

generated_text = generated_text.cpu().numpy()
decoded_text = tokenizer.decode(generated_text[0].tolist())
print(decoded_text)
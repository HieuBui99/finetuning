import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, drop_rate=0.1, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.output_proj = nn.Linear(d_out, d_out)
        self.n_heads = n_heads
        self.drop_rate = drop_rate

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # b, t, d_out * 3 -> b, t, d_out
        q = q.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        k = k.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        v = v.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)

        context_vec = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop_rate
        )
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, t, c)
        context_vec = self.output_proj(context_vec)
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["emb_dim"] * 4)
        self.fc2 = nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"])
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, x):
        x = self.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalAttention(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["n_heads"],
            cfg["drop_rate"],
            cfg["qkv_bias"],
        )
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.mlp = FeedForward(cfg)
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]),
                wpe=nn.Embedding(cfg["context_length"], cfg["emb_dim"]),
                blocks=nn.ModuleList(
                    [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
                ),
                ln_f=nn.LayerNorm(cfg["emb_dim"]),
            )
        )
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, x):
        b, t = x.shape
        tok_emb = self.transformer.wte(x)
        pos_emb = self.transformer.wpe(torch.arange(0, t, device=x.device))
        x = tok_emb + pos_emb

        for block in self.transformer.blocks:
            x = block(x)

        logits = self.lm_head(self.transformer.ln_f(x))

        return logits


config = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

if __name__ == "__main__":
    model = Transformer(config)
    x = torch.randint(0, 50257, (1, 1024))
    out = model(x)
    print(out.shape)

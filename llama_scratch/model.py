import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    return x_rotated.to(dtype=x.dtype)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, drop_rate=0.1, qkv_bias=False, context_length=1024):
        super().__init__()
        self.d_out = d_out
        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.output_proj = nn.Linear(d_out, d_out)
        self.n_heads = n_heads
        self.drop_rate = drop_rate
        cos, sin = precompute_rope_params(head_dim=d_in//n_heads, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # b, t, d_out * 3 -> b, t, d_out
        q = q.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        k = k.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        v = v.view(b, t, self.n_heads, c // self.n_heads).transpose(1, 2)
        
        # Apply rotary position embedding
        k = compute_rope(k, self.cos, self.sin)
        q = compute_rope(q, self.cos, self.sin)

        context_vec = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.drop_rate
        )
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, t, c)
        context_vec = self.output_proj(context_vec)
        return context_vec


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc2 = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"])
        self.fc3 = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"])
        self.silu = nn.SiLU()

    def forward(self, x):
        x_fc1 = self.fc1(x)
        x_fc2 = self.fc2(x)
        x = self.silu(x_fc1) * x_fc2
        return self.fc3(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalAttention(
            cfg["emb_dim"],
            cfg["emb_dim"],
            cfg["n_heads"],
            cfg["drop_rate"],
            cfg["qkv_bias"],
            cfg["context_length"],
        )
        self.norm1 = nn.RMSNorm(cfg["emb_dim"])
        self.mlp = FeedForward(cfg)
        self.norm2 = nn.RMSNorm(cfg["emb_dim"])

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class LLama2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(cfg["vocab_size"], cfg["emb_dim"]),
                # wpe=nn.Embedding(cfg["context_length"], cfg["emb_dim"]),
                blocks=nn.ModuleList(
                    [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
                ),
                ln_f=nn.RMSNorm(cfg["emb_dim"]),
            )
        )
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

    def forward(self, x):
        b, t = x.shape
        tok_emb = self.transformer.wte(x)
        x = tok_emb

        for block in self.transformer.blocks:
            x = block(x)
        
        logits = self.lm_head(self.transformer.ln_f(x))

        return logits



if __name__ == "__main__":
    llama_config = {
        "vocab_size": 32000, 
        "context_length": 256,
        "emb_dim": 512,     
        "n_heads": 8,  
        "n_layers": 8,        
        "hidden_dim": 1024, 
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    model = LLama2(llama_config)
    x = torch.randint(0, 32000, (1, 256))
    out = model(x)
    print(out.shape)

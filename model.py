import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
import math

@dataclass
class GPTConfig:
    context_length: int = 1024
    vocab_size: int = 50257 # 50,257 tokens in GPT-2
    n_layer: int = 12
    n_head: int = 12
    embedding_dim: int = 768
    dropout: float = 0.1
    bias: bool = False


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc_1 = nn.Linear(config.embedding_dim, config.embedding_dim * 4, bias=config.bias)
        self.act = nn.GELU()
        self.fc_2 = nn.Linear(config.embedding_dim * 4, config.embedding_dim, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.act(x)
        x = self.fc_2(x)
        x = self.dropout(x)

        return x


class CasualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.n_head == 0     # ensure embedding channel is divisible by number of heads

        # Attention Channel lifting and projection
        self.attn = nn.Linear(config.embedding_dim, config.embedding_dim * 3, bias=config.bias)
        self.proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)

        # Dropouts
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)

        # Attention Parameters
        self.n_head = config.n_head
        self.embedding_dim = config.embedding_dim
        self.dropout = config.dropout

        self.flash = False#hasattr(torch.nn.functional, 'scaled_dot_product_attention')   # check if torch version support flash attention
        if not self.flash:
            self.register_buffer('mask', torch.tril(torch.ones(config.context_length, config.context_length))
                                                    .view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, L, C = x.size()

        # Q, K, V calculation
        q, k, v = self.attn(x).split(self.embedding_dim, dim=2)
        q = q.view(B, L, self.n_head, C // self.n_head).transpose(1, 2) # (B, n_head, L, channel_head)
        k = k.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, L, self.n_head, C // self.n_head).transpose(1, 2)

        # Casual Self-Attention 
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout, is_causal=True)
        else:
            # Original formula from transformer paper
            att = (q @ k.transpose(-2, -1)) / (self.embedding_dim ** 0.5)   # Attention = Q * K^T / sqrt(d_k)
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))    # Masking upper triangular matrix
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, L, C)    # Assemble heads outputs

        # Projection
        y = self.proj(y)
        y = self.proj_dropout(y)
        
        return y


class block(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.embedding_dim)
        self.ln_2 = nn.LayerNorm(config.embedding_dim)
        self.attn = CasualAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Transformer layers
        self.transformer = nn.ModuleDict(dict(
            Wtoken = nn.Embedding(config.vocab_size, config.embedding_dim),
            Wpos = nn.Embedding(config.context_length, config.embedding_dim),
            drop = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([block(config) for _ in range(config.n_layer)]),
            ln = nn.LayerNorm(config.embedding_dim, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=config.bias)

        self.transformer.Wtoken.weight = self.lm_head.weight    # weight tying

        # Init weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, target=None):
        device = x.device
        B, L = x.size()
        assert L <= self.config.context_length, f'Cannot forward, model has fixed context length: {self.config.context_length} while giving input of length {L}.'
        pos = torch.arange(0, L, dtype=torch.long, device=device)

        # GPT forward
        # Tokenization and Positional Embedding
        tok_emb = self.transformer.Wtoken(x)    # (B, L, embedding_dim)
        pos_emb = self.transformer.Wpos(pos)    # (L, embedding_dim)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.blocks:
            x = block(x)
        x = self.transformer.ln(x)  # (B, L, embedding_dim)

        # Training phase
        if target is not None:
            logits = self.lm_head(x)    # (B, L, vocab_size)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=-1)  # (B * L, vocab_size) and (B * L)

        # Inference phase
        else:
            logits = self.lm_head(x[:, [-1], :])  # (B, 1, vocab_size), Only forward last token
            loss = None

        return logits, loss


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.embedding_dim//cfg.n_head, cfg.context_length
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take conditioning sequence of indices "idx" (in shape (B, L)) and complete the sequence max_new_tokens times,
        feeding predictions back into model each time.
        """
        for _ in range(max_new_tokens):
            # If sequence context is too long, crop it into context_length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]

            # forward the model to get logits
            logits, _ = self(idx_cond)

            # pluck the logits at the final stop and scale by desired temperature
            logits = logits[:, -1, :] / temperature

            # optionally crop the logits to only top_k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled index to running sequence and continue
            idx = torch.cat((idx,idx_next), dim=1)

        return idx            
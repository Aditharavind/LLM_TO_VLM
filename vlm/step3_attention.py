"""
STEP 3 -- Multi-Head Attention, written out longhand.
=====================================================

One sentence: every token asks a question (Q), every token advertises what it
has (K), and every token offers content (V).  Each token then takes a weighted
average of everybody's V, where the weights come from how well its Q matches
their K.

    scores  = Q @ K^T / sqrt(d_head)      "how relevant is each token to me"
    weights = softmax(scores)             "turn that into percentages (rows sum to 1)"
    out     = weights @ V                 "weighted average of the content"

Why *multi*-head?  One attention map can only express one kind of relationship.
So we split the 64-dim vector into 4 chunks of 16, run 4 independent attentions
in parallel, and concatenate.  One head can learn "look at patches of the same
colour", another "look at my neighbours", another "look at the background".
Same cost, four opinions.

This file has TWO classes:
  MultiHeadAttention  -- generic; Q comes from x, K/V come from `context`
  SelfAttention       -- context is x itself (what the ViT and the LLM use)

Cross-attention (context != x) is what the optional "resampler" bridge in
step5 uses, and it is also how Flamingo-style VLMs inject images into an LLM.

Everything is written with explicit matmuls + softmax so you can print any
intermediate tensor.  (Production code calls F.scaled_dot_product_attention,
which is the same maths, fused and faster, but hides the attention matrix.)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, context_dim=None, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0, "dim must split evenly across heads"
        self.dim = dim
        self.n_heads = n_heads
        self.d_head = dim // n_heads          # 64 / 4 = 16
        context_dim = context_dim or dim

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(context_dim, dim)
        self.v_proj = nn.Linear(context_dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, t):
        """[B, T, dim] -> [B, heads, T, d_head]   (heads move up front)"""
        B, T, _ = t.shape
        return t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

    def _merge_heads(self, t):
        """[B, heads, T, d_head] -> [B, T, dim]   (glue the heads back together)"""
        B, H, T, dh = t.shape
        return t.transpose(1, 2).contiguous().view(B, T, H * dh)

    def forward(self, x, context=None, causal=False, key_padding_mask=None,
                return_attn=False):
        """
        x        : [B, Tq, dim]        the tokens that are *asking*
        context  : [B, Tk, ctx_dim]    the tokens being *looked at* (None -> x)
        causal   : True for an LLM (a token may not look at the future)
        returns  : [B, Tq, dim]  (+ attention [B, heads, Tq, Tk] if requested)
        """
        ctx = x if context is None else context

        # 1) three linear projections -------------------------------------
        q = self._split_heads(self.q_proj(x))     # [B, H, Tq, dh]
        k = self._split_heads(self.k_proj(ctx))   # [B, H, Tk, dh]
        v = self._split_heads(self.v_proj(ctx))   # [B, H, Tk, dh]

        # 2) how much does each query match each key? ---------------------
        # the /sqrt(d_head) keeps the numbers small so softmax doesn't saturate
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)   # [B,H,Tq,Tk]

        # 3) masking ------------------------------------------------------
        if causal:
            Tq, Tk = scores.shape[-2], scores.shape[-1]
            # True = "not allowed to look here"; upper triangle = the future
            future = torch.ones(Tq, Tk, dtype=torch.bool, device=scores.device)
            future = future.triu(diagonal=1 + (Tk - Tq))
            scores = scores.masked_fill(future, float("-inf"))
        if key_padding_mask is not None:
            # key_padding_mask: [B, Tk], True where the key is padding
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        # 4) percentages --------------------------------------------------
        attn = scores.softmax(dim=-1)             # each row sums to 1.0
        attn = self.dropout(attn)

        # 5) weighted average of the values, then mix the heads -----------
        out = attn @ v                            # [B, H, Tq, dh]
        out = self.out_proj(self._merge_heads(out))   # [B, Tq, dim]

        return (out, attn) if return_attn else out


class SelfAttention(MultiHeadAttention):
    """Sugar: attention where the tokens look at each other."""

    def forward(self, x, causal=False, key_padding_mask=None, return_attn=False):
        return super().forward(x, context=None, causal=causal,
                               key_padding_mask=key_padding_mask,
                               return_attn=return_attn)


if __name__ == "__main__":
    torch.manual_seed(0)
    att = SelfAttention(dim=64, n_heads=4)
    x = torch.randn(1, 5, 64)
    out, w = att(x, return_attn=True)
    print("in ", tuple(x.shape), "-> out", tuple(out.shape))
    print("attention matrix:", tuple(w.shape), "= [batch, heads, query, key]")
    print("every row sums to 1:", torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1))))

    _, wc = att(x, causal=True, return_attn=True)
    print("\ncausal mask (head 0) -- note the zeros above the diagonal:")
    print(wc[0, 0].round(decimals=2))

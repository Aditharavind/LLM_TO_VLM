"""
STEP 4 -- Stack the blocks: this is the Vision Transformer.
===========================================================

A transformer block is only two ideas glued together:

    x = x + Attention(LayerNorm(x))     <- tokens TALK to each other
    x = x + MLP(LayerNorm(x))           <- each token THINKS on its own

Attention is the only place where information moves *between* patches.
The MLP processes every token independently.

`x = x + ...` is the residual connection: each block *edits* the token instead
of replacing it, which is what lets us stack many of them without the signal
dying.  LayerNorm-before-the-block ("pre-LN") is what modern models do.

Output of the ViT: one vector per patch, but now each of those vectors has
looked at the whole image.  Patch 0 no longer means "the top-left 16x16 pixels",
it means "the top-left region, in the context of everything else".
"""
import torch
import torch.nn as nn

from .step2_patch_embed import PatchEmbedding
from .step3_attention import SelfAttention


class MLP(nn.Module):
    """Expand 4x, apply a non-linearity, squeeze back. The 'thinking' half."""

    def __init__(self, dim, hidden_mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.GELU(),
            nn.Linear(dim * hidden_mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.0, causal=False):
        super().__init__()
        self.causal = causal
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dropout=dropout)

    def forward(self, x, key_padding_mask=None, return_attn=False):
        a = self.attn(self.norm1(x), causal=self.causal,
                      key_padding_mask=key_padding_mask, return_attn=return_attn)
        if return_attn:
            a, attn = a
        x = x + a                      # residual 1: tokens exchanged info
        x = x + self.mlp(self.norm2(x))  # residual 2: each token digested it
        return (x, attn) if return_attn else x


class VisionTransformer(nn.Module):
    """images -> [B, n_patches, vision_dim], one context-aware vector per patch."""

    def __init__(self, image_size=64, patch_size=16, in_channels=3,
                 dim=64, n_heads=4, n_layers=4, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.apply(self._init_weights)
        nn.init.trunc_normal_(self.patch_embed.pos_embed, std=0.02)
        self.grid_size = self.patch_embed.grid_size
        self.n_patches = self.patch_embed.n_patches

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, images, return_attn=False):
        x = self.patch_embed(images)          # [B, N, dim]  step 1 + step 2
        attns = []
        for blk in self.blocks:
            if return_attn:
                x, a = blk(x, return_attn=True)
                attns.append(a)               # each: [B, heads, N, N]
            else:
                x = blk(x)
        x = self.norm(x)
        return (x, attns) if return_attn else x


if __name__ == "__main__":
    vit = VisionTransformer()
    img = torch.randn(2, 3, 64, 64)
    feats, attns = vit(img, return_attn=True)
    print("image        ", tuple(img.shape))
    print("patch tokens ", tuple(feats.shape))
    print("attn per layer", tuple(attns[0].shape), f"x {len(attns)} layers")
    print("params:", sum(p.numel() for p in vit.parameters()) / 1e3, "k")

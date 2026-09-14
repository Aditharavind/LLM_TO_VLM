"""
STEP 5 -- The bridge.  THIS is the line that turns an LLM into a VLM.
=====================================================================

The ViT speaks in 64-dim vectors.  The LLM's word embeddings are 128-dim.
They are different languages of different widths.  The bridge (also called
"connector", "projector" or "adapter") translates.

Two flavours, both provided here:

 A) MLPProjector      (LLaVA)
        16 patch vectors  ->  16 LLM-shaped vectors.  One patch = one token.
        Dead simple: Linear -> GELU -> Linear. Literally 2 layers.
        Keeps all spatial detail, but costs you 16 tokens of context.

 B) AttentionResampler (Flamingo / BLIP-2 Q-Former)
        16 patch vectors  ->  n_query (say 8) LLM-shaped vectors.
        We keep `n_query` LEARNED query vectors. They cross-attend to the
        patches and pull out what matters.  This is the "multi-head attention"
        version of the bridge, and it fixes a real problem: a 336x336 image at
        patch 14 is 576 patches = 576 tokens of your context window gone.

Both return [B, n_image_tokens, llm_dim] -- from the LLM's point of view these
are indistinguishable from word embeddings.  That is the entire trick.
"""
import torch
import torch.nn as nn

from .step3_attention import MultiHeadAttention


class MLPProjector(nn.Module):
    """One patch in, one LLM token out. (LLaVA-1.5 is exactly this.)"""

    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, patch_feats):
        # [B, N, vision_dim] -> [B, N, llm_dim]
        return self.net(patch_feats)


class AttentionResampler(nn.Module):
    """
    Squeeze N patches into a fixed, smaller number of tokens with cross-attention.

    The learned queries are the interesting bit: they are parameters, not data.
    Query #3 might learn to mean "tell me about colour", query #5 "tell me
    where the object is".  They ask the image the same questions every time.
    """

    def __init__(self, vision_dim, llm_dim, n_query=8, n_heads=4, query_init_std=0.02):
        super().__init__()
        # NOTE: this init std matters more than it looks. The queries are the
        # only thing distinguishing one output token from another, and nothing
        # in the loss pushes them apart. Start them all near zero and they can
        # stay effectively identical -- "query collapse". demos/demo_06 measures it.
        self.query = nn.Parameter(torch.randn(1, n_query, llm_dim) * query_init_std)
        self.norm_ctx = nn.LayerNorm(vision_dim)
        self.cross_attn = MultiHeadAttention(llm_dim, n_heads, context_dim=vision_dim)
        self.norm_out = nn.LayerNorm(llm_dim)
        self.ffn = nn.Sequential(
            nn.Linear(llm_dim, llm_dim * 4), nn.GELU(), nn.Linear(llm_dim * 4, llm_dim)
        )

    def forward(self, patch_feats, return_attn=False):
        B = patch_feats.shape[0]
        q = self.query.expand(B, -1, -1)             # [B, n_query, llm_dim]
        ctx = self.norm_ctx(patch_feats)             # [B, N, vision_dim]
        out = self.cross_attn(q, context=ctx, return_attn=return_attn)
        if return_attn:
            out, attn = out                          # attn: [B, heads, n_query, N]
        x = q + out                                  # residual
        x = x + self.ffn(self.norm_out(x))
        return (x, attn) if return_attn else x


def build_connector(cfg):
    if cfg.connector == "mlp":
        return MLPProjector(cfg.vision_dim, cfg.llm_dim)
    if cfg.connector == "resampler":
        return AttentionResampler(cfg.vision_dim, cfg.llm_dim,
                                  cfg.n_query_tokens, cfg.vision_heads,
                                  cfg.query_init_std)
    raise ValueError(f"unknown connector: {cfg.connector}")


if __name__ == "__main__":
    feats = torch.randn(2, 16, 64)      # what the ViT produced
    print("vit output        ", tuple(feats.shape))
    print("after MLPProjector", tuple(MLPProjector(64, 128)(feats).shape),
          " <- 16 patches -> 16 llm tokens")
    r = AttentionResampler(64, 128, n_query=8)
    out, a = r(feats, return_attn=True)
    print("after Resampler   ", tuple(out.shape), " <- 16 patches -> 8 llm tokens")
    print("cross-attn map    ", tuple(a.shape), "= [B, heads, queries, patches]")

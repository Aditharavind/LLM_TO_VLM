"""
STEP 7 -- Glue it together.  An LLM is now a VLM.
=================================================

The whole model in one picture:

     image ──▶ patchify ──▶ ViT ──▶ connector ──┐
                                                 ├──▶ [img][img]...[<bos>][a][red]... ──▶ LLM ──▶ next word
     caption ─────────────▶ token embedding ────┘
                                                 ^
                                         ONE sequence. The LLM cannot tell
                                         which vectors came from pixels.

The sequence the LLM actually sees:

    position:   0    1   ...  M-1    M      M+1   M+2   M+3 ...
    content:  [img][img] ... [img] [<bos>]  [a]  [red] [circle] ...
              \_____ from the image _____/  \______ from the text _______/

Loss: plain next-token cross-entropy, but ONLY on the text positions -- there
is no "correct next token" for an image patch, and padding is ignored too.

Which parts are trained is a choice (see `freeze_vision`):
  - real VLMs usually freeze the ViT and the LLM at first and train ONLY the
    connector (cheap, a few hours) -- "feature alignment"
  - then unfreeze the LLM for instruction tuning
  - here everything is tiny, so by default we train it all end to end
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .step4_vit import VisionTransformer
from .step5_projector import build_connector
from .step6_tiny_llm import TinyLLM
from .tokenizer import WordTokenizer


class VLM(nn.Module):
    def __init__(self, cfg: Config, tokenizer: WordTokenizer = None):
        super().__init__()
        self.cfg = cfg
        self.tok = tokenizer or WordTokenizer()

        self.vision = VisionTransformer(
            cfg.image_size, cfg.patch_size, cfg.in_channels,
            cfg.vision_dim, cfg.vision_heads, cfg.vision_layers, cfg.dropout)

        self.connector = build_connector(cfg)          # <-- the bridge

        self.llm = TinyLLM(len(self.tok), cfg.llm_dim, cfg.llm_heads,
                           cfg.llm_layers,
                           max_seq_len=cfg.n_image_tokens + cfg.max_text_len,
                           dropout=cfg.dropout)

        self.n_image_tokens = cfg.n_image_tokens

        # Purely for demos/demo_05_ablations.py: set this to a LongTensor to
        # re-order the image tokens before they reach the LLM. Leaving it None
        # is normal operation.
        self.shuffle_image_tokens = None

    # ------------------------------------------------------------------ #
    def encode_image(self, images, return_attn=False):
        """[B,3,H,W] -> [B, n_image_tokens, llm_dim].  Pixels become 'words'."""
        feats = self.vision(images)                    # [B, N, vision_dim]
        out = self.connector(feats, return_attn=True) if (
            return_attn and hasattr(self.connector, "query")) else self.connector(feats)
        return out

    def build_sequence(self, images, text_ids):
        """
        Concatenate image tokens and text token embeddings into one sequence.
        This is the actual 'conversion' step -- three lines of tensor surgery.
        """
        img_tokens = self.encode_image(images)          # [B, M, llm_dim]
        if self.shuffle_image_tokens is not None:       # ablation hook
            img_tokens = img_tokens[:, self.shuffle_image_tokens]
        txt_tokens = self.llm.embed_tokens(text_ids)    # [B, T, llm_dim]
        return torch.cat([img_tokens, txt_tokens], dim=1), img_tokens.shape[1]

    # ------------------------------------------------------------------ #
    def forward(self, images, text_ids, return_attn=False):
        seq, M = self.build_sequence(images, text_ids)   # [B, M+T, llm_dim]
        out = self.llm(inputs_embeds=seq, return_attn=return_attn)
        return out

    def loss(self, images, text_ids):
        """Next-token cross-entropy over the caption only."""
        logits = self(images, text_ids)                  # [B, M+T, vocab]
        M = self.n_image_tokens
        T = text_ids.shape[1]

        # position M+j-1 is asked to predict text token j.
        # We skip j=0 (<bos> is given, not predicted).
        pred = logits[:, M:M + T - 1, :]                 # predicts text[:, 1:]
        target = text_ids[:, 1:]                         # [B, T-1]
        return F.cross_entropy(
            pred.reshape(-1, pred.size(-1)),
            target.reshape(-1),
            ignore_index=self.tok.pad_id,                # don't learn from <pad>
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(self, images, max_new_tokens=10, return_attn=False):
        """
        Greedy decoding.  Start from <bos>, feed image + text-so-far, take the
        argmax, append, repeat.  (Recomputes the whole sequence each step --
        slow but obvious. Real code caches the keys/values.)
        """
        self.eval()
        B = images.shape[0]
        device = images.device
        ids = torch.full((B, 1), self.tok.bos_id, dtype=torch.long, device=device)
        img_attn = None

        for _ in range(max_new_tokens):
            out = self(images, ids, return_attn=return_attn)
            if return_attn:
                logits, attns = out
                # attention of the LAST text position back onto the image tokens,
                # averaged over heads, from the final layer:
                # [B, heads, T_total, T_total] -> [B, n_image_tokens]
                img_attn = attns[-1][:, :, -1, :self.n_image_tokens].mean(1)
            else:
                logits = out
            nxt = logits[:, -1, :].argmax(-1, keepdim=True)     # [B, 1]
            ids = torch.cat([ids, nxt], dim=1)
            if (nxt == self.tok.eos_id).all():
                break

        texts = [self.tok.decode(row) for row in ids]
        return (texts, ids, img_attn) if return_attn else (texts, ids)

    # ------------------------------------------------------------------ #
    def freeze_vision(self, freeze=True):
        """Mimic stage-1 VLM training: only the connector learns."""
        for p in self.vision.parameters():
            p.requires_grad = not freeze

    def freeze_llm(self, freeze=True):
        for p in self.llm.parameters():
            p.requires_grad = not freeze

    def param_report(self):
        def n(m):
            return sum(p.numel() for p in m.parameters())
        return {
            "vision (ViT)": n(self.vision),
            "connector":    n(self.connector),
            "llm":          n(self.llm),
            "TOTAL":        n(self),
        }


if __name__ == "__main__":
    cfg = Config()
    m = VLM(cfg)
    imgs = torch.randn(2, 3, 64, 64)
    ids = torch.randint(3, 17, (2, 9))
    print("logits:", tuple(m(imgs, ids).shape))
    print("loss  :", float(m.loss(imgs, ids)))
    for k, v in m.param_report().items():
        print(f"  {k:14s} {v/1e3:8.1f} k")

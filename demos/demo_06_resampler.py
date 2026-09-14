"""
DEMO 6 -- The other kind of bridge: cross-attention.
====================================================
Run:  python train.py --connector resampler --out checkpoints/vlm_resampler.pt
      python demos/demo_06_resampler.py

The MLP connector is "one patch -> one token".  Simple, but a real image at
336x336 with patch 14 is 576 patches = 576 tokens of context burned per image.

The resampler is the alternative.  We keep `n_query` LEARNED vectors -- they are
parameters, they do not depend on the image at all.  They CROSS-ATTEND to the
patches:

        queries  (learned, n_query x llm_dim)   <- ask the questions
        keys/values come from the PATCHES       <- the image answers

Output: exactly n_query tokens, no matter how many patches went in.

This figure shows, for each learned query, which patches it pulled from.
Different queries specialise -- which is, again, the multi-head-attention idea:
ask several different questions at once.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from vlm.config import Config, pick_device
from vlm.data import caption_for, render, to_tensor
from vlm.step7_vlm import VLM
from vlm.viz import clean, draw_patch_grid, img_np, save

CKPT = "checkpoints/vlm_resampler.pt"
if not os.path.exists(CKPT):
    sys.exit("run:  python train.py --connector resampler "
             "--out checkpoints/vlm_resampler.pt")

blob = torch.load(CKPT, map_location="cpu")
cfg = Config(**blob["cfg"])
assert cfg.connector == "resampler"
device = pick_device()
model = VLM(cfg); model.load_state_dict(blob["model"]); model.eval().to(device)
G, Q = cfg.grid_size, cfg.n_query_tokens

print(f"  {cfg.n_patches} patches  ->  {cfg.n_image_tokens} tokens "
      f"({100*(1-cfg.n_image_tokens/cfg.n_patches):.0f}% fewer tokens for the LLM)")

cases = [("red", "circle", ("top", "left")),
         ("blue", "square", ("bottom", "right"))]

fig, axes = plt.subplots(len(cases), Q + 1,
                         figsize=(1.85 * (Q + 1), 2.75 * len(cases)))
fig.subplots_adjust(hspace=0.45)
collapse = []
for ci, (color, shape, pos) in enumerate(cases):
    img = render(color, shape, pos, cfg.image_size, seed=200 + ci)
    x = to_tensor(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feats = model.vision(x)                               # [1, N, vision_dim]
        _, attn = model.connector(feats, return_attn=True)    # [1, heads, Q, N]
        said = model.generate(x, max_new_tokens=cfg.max_text_len - 1)[0][0]
    a = attn[0].mean(0).cpu().numpy()                         # [Q, N] mean over heads

    # Do the 8 learned queries actually ask DIFFERENT questions?  Measure it
    # instead of eyeballing the colours: L1 distance between every pair of rows.
    d = np.abs(a[:, None, :] - a[None, :, :]).sum(-1)
    collapse.append(d.max())

    ax = clean(axes[ci, 0], "the image" if ci == 0 else "", 10)
    ax.imshow(img_np(x[0].cpu()))
    draw_patch_grid(ax, cfg.image_size, cfg.patch_size, color="#888", lw=.7,
                    numbers=False)
    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])

    for q in range(Q):
        ax = clean(axes[ci, q + 1], f"query {q}" if ci == 0 else "", 10)
        h = a[q].reshape(G, G)
        ax.imshow(h, cmap="inferno")
        ax.set_xlabel(f"max {h.max():.2f}", fontsize=7)
    w = said.split()
    axes[ci, 0].set_xlabel("said: " + " ".join(w[:3]) + "\n" + " ".join(w[3:]),
                           fontsize=7)

fig.suptitle("cross-attention bridge: what does each LEARNED query pull out "
             f"of the image?\n{cfg.n_patches} patches -> {Q} tokens   "
             "(the queries are parameters -- they ask the same questions of every image)",
             fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.86])
save(fig, "06_resampler_queries.png")

worst = max(collapse)
print(f"\n  largest difference between any two query attention rows: {worst:.4f}")
print(f"  (0.0 would mean all {Q} queries are asking the IDENTICAL question)")
if worst < 0.2:
    print("""
  -> QUERY COLLAPSE. The queries all learned the same thing, so these 8 tokens
     carry about one token's worth of information. Nothing in the loss pushes
     the queries apart, and they were initialised almost on top of each other
     (query_init_std = %.2f), so they never separated.
     Try:  python train.py --connector resampler --query-init-std 1.0 \\
                 --epochs 40 --lr 5e-4 --out checkpoints/vlm_resampler.pt
     and run this demo again.""" % cfg.query_init_std)
else:
    print(f"""
  -> The queries DID separate (init std was {cfg.query_init_std}). Each one is
     pulling from a different part of the image, which is the whole point of a
     resampler: n_query genuinely different questions, asked of every image.""")

print("""
  Compare with demo 1: there, token k WAS patch k -- a fixed, spatial mapping.
  Here there is no such mapping. Query 3 means whatever it learned to mean, and
  it may gather from all over the image. You trade interpretability and detail
  for a much shorter sequence.
""")

"""
DEMO 3 -- "What does a patch look at?"  (self-attention inside the ViT)
=======================================================================
Run:  python demos/demo_03_vit_attention.py

We pick one patch (the one sitting on the object), then plot, for every layer
and every head, how much that patch attends to each of the 16 patches.

Each little 4x4 square IS the image's patch grid.  Bright = "I am looking here".

What to point out:
  * different HEADS in the same layer look at different things -- that is the
    entire reason for MULTI-head attention
  * early layers tend to look locally, later layers pull in the object
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from vlm.config import Config, pick_device
from vlm.data import render, to_tensor
from vlm.step7_vlm import VLM
from vlm.viz import clean, draw_patch_grid, img_np, save

CKPT = "checkpoints/vlm.pt"
device = pick_device()

cfg = Config()
model = VLM(cfg)
if os.path.exists(CKPT):
    sd = torch.load(CKPT, map_location="cpu")
    if sd["cfg"]["connector"] == cfg.connector:
        model.load_state_dict(sd["model"])
        print("loaded trained weights")
else:
    print("no checkpoint found -- showing an UNTRAINED ViT "
          "(run `python train.py` first for the interesting version)")
model.eval().to(device)

color, shape, pos = "red", "square", ("bottom", "right")
img = render(color, shape, pos, cfg.image_size, seed=7)
x = to_tensor(img).unsqueeze(0).to(device)

with torch.no_grad():
    _, attns = model.vision(x, return_attn=True)      # list of [1, H, 16, 16]

G = cfg.grid_size
# the query patch: the one that contains the object (bottom-right quadrant)
query = 10
L, H = len(attns), cfg.vision_heads
# share one colour scale across every panel so heads are actually comparable
vmax = max(float(a[0, :, query].max()) for a in attns)
uniform = 1.0 / cfg.n_patches
print(f"uniform attention would be 1/{cfg.n_patches} = {uniform:.3f};  max seen = {vmax:.3f}")

fig, axes = plt.subplots(L, H + 1, figsize=(2.0 * (H + 1), 2.0 * L))
for li in range(L):
    ax = clean(axes[li, 0], "the image\n(query patch outlined)" if li == 0 else "")
    ax.imshow(img_np(x[0].cpu()))
    draw_patch_grid(ax, cfg.image_size, cfg.patch_size, color="#666", lw=.7,
                    numbers=False)
    r, c = query // G, query % G
    ax.add_patch(plt.Rectangle((c * cfg.patch_size - .5, r * cfg.patch_size - .5),
                               cfg.patch_size, cfg.patch_size, fill=False,
                               edgecolor="orange", lw=3))
    ax.set_ylabel(f"layer {li}", fontsize=11)
    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])

    for h in range(H):
        a = attns[li][0, h, query].reshape(G, G).cpu().numpy()
        ax = clean(axes[li, h + 1], f"head {h}" if li == 0 else "")
        ax.imshow(a, cmap="Reds", vmin=0, vmax=vmax)   # darker red = more attention
        ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                                   edgecolor="cyan", lw=2))
fig.suptitle(f"ViT self-attention: where does patch {query} look?\n"
             f"each 4x4 square = the patch grid   |   DARK RED = high attention "
             f"(uniform would be {uniform:.2f} everywhere)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.93])
save(fig, "03_vit_attention.png")

# ---- second view: the actual numbers ---------------------------------------
# With only 16 patches we can just PRINT the attention weights. No colour
# illusion, no ambiguity: each cell is the fraction of attention patch `query`
# sends to that patch, and the 16 cells sum to 1.
fig, axes = plt.subplots(1, L + 1, figsize=(3.1 * (L + 1), 3.4))
clean(axes[0], f"image\nquery = patch {query} (orange)")
axes[0].imshow(img_np(x[0].cpu()))
draw_patch_grid(axes[0], cfg.image_size, cfg.patch_size, color="#666", lw=.8)
r, c = query // G, query % G
axes[0].add_patch(plt.Rectangle((c * cfg.patch_size - .5, r * cfg.patch_size - .5),
                                cfg.patch_size, cfg.patch_size, fill=False,
                                edgecolor="orange", lw=3))
for li in range(L):
    a = attns[li][0, :, query].mean(0).reshape(G, G).cpu().numpy()
    ax = clean(axes[li + 1], f"layer {li}  (mean over {H} heads)\nrow sums to 1.00")
    # colour is stretched between this panel's own min and max, so faint
    # differences are visible; trust the NUMBERS, not the shade
    ax.imshow(a, cmap="Reds", vmin=a.min(), vmax=a.max())
    for i in range(G):
        for j in range(G):
            ax.text(j, i, f"{a[i, j]:.02f}".lstrip("0"), ha="center", va="center",
                    fontsize=9,
                    color="white" if a[i, j] > a.min() + 0.6 * (a.max() - a.min()) else "#333")
    ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False,
                               edgecolor="orange", lw=2.5))
fig.suptitle(f"the same attention as plain numbers  "
             f"(uniform = {uniform:.2f} in every cell; bigger = looked at more)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.88])
save(fig, "03b_vit_attention_numbers.png")

print()
print("  NOTE for teaching: on this toy task the ViT's attention stays close to")
print("  uniform (~0.06).  That is honest and worth explaining -- with only 16")
print("  patches and one object, a patch does not need to hunt for information.")
print("  Sharp, interpretable attention shows up in demo 4, where the LANGUAGE")
print("  model has to go and find the object in order to name it.")

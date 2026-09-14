"""
DEMO 1 -- "How does an image become a sequence?"
================================================
Run:  python demos/demo_01_patching.py

Produces assets/01_patching.png, which walks through, left to right:
  the image -> the patch grid -> the patches pulled apart -> the flat sequence
  -> the raw numbers -> after the linear layer -> plus position embeddings.

The key thing to point out to a student: between panel 1 and panel 4 NOTHING
is learned and no information is lost.  We are only re-arranging pixels.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
import torch

from vlm.config import Config
from vlm.data import render, to_tensor
from vlm.step1_patchify import patchify, unpatchify
from vlm.step2_patch_embed import PatchEmbedding
from vlm.viz import clean, draw_patch_grid, img_np, save

cfg = Config()
torch.manual_seed(0)

img = render("blue", "triangle", ("top", "right"), cfg.image_size, seed=3)
x = to_tensor(img).unsqueeze(0)                        # [1, 3, 64, 64]
P, G = cfg.patch_size, cfg.grid_size

patches = patchify(x, P)                               # [1, 16, 768]
print(f"image   {tuple(x.shape)}  ->  patches {tuple(patches.shape)}")
print("reversible:", torch.allclose(unpatchify(patches, P, G), x))

pe = PatchEmbedding(cfg.image_size, P, 3, cfg.vision_dim)
tokens, parts = pe(x, return_parts=True)

fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 0.42, 1.0],
                      hspace=0.42, wspace=0.25)

# ---- 1. the raw image ------------------------------------------------------
ax = clean(fig.add_subplot(gs[0, 0]), "1. the image\n64 x 64 x 3 = 12,288 numbers", 11)
ax.imshow(img_np(x[0]))

# ---- 2. the cut lines ------------------------------------------------------
ax = clean(fig.add_subplot(gs[0, 1]),
           f"2. cut into {G}x{G} patches of {P}x{P}\n(read left->right, top->bottom)", 11)
ax.imshow(img_np(x[0]))
draw_patch_grid(ax, cfg.image_size, P, color="#222", lw=1.4)
# the reading-order arrow
ax.annotate("", xy=(60, 60), xytext=(4, 4),
            arrowprops=dict(arrowstyle="->", color="orange", lw=2,
                            connectionstyle="arc3,rad=0.12"))

# ---- 3. patches pulled apart ----------------------------------------------
ax = clean(fig.add_subplot(gs[0, 2]), "3. pull them apart\n16 independent tiles", 11)
gap = 4
canvas = np.full((G * (P + gap), G * (P + gap), 3), 255, np.uint8)
im = img_np(x[0])
for idx in range(G * G):
    r, c = idx // G, idx % G
    tile = im[r * P:(r + 1) * P, c * P:(c + 1) * P]
    canvas[r * (P + gap):r * (P + gap) + P, c * (P + gap):c * (P + gap) + P] = tile
ax.imshow(canvas)
for idx in range(G * G):
    r, c = idx // G, idx % G
    ax.text(c * (P + gap) + 2, r * (P + gap) + 9, str(idx), color="orange",
            fontsize=9, fontweight="bold")

# ---- 4. the sequence -------------------------------------------------------
ax = clean(fig.add_subplot(gs[1, :]),
           "4. lay the 16 patches out in a LINE  --  this IS the sequence the ViT reads", 12)
strip = np.full((P + 2, G * G * (P + 2), 3), 255, np.uint8)
for idx in range(G * G):
    r, c = idx // G, idx % G
    strip[1:P + 1, idx * (P + 2) + 1: idx * (P + 2) + 1 + P] = \
        im[r * P:(r + 1) * P, c * P:(c + 1) * P]
ax.imshow(strip)
ax.set_xlabel("patch 0 ............................ patch 15", fontsize=9)

# ---- explainer box in the free top-right cell -----------------------------
ax = clean(fig.add_subplot(gs[0, 3]))
ax.text(0.0, 0.98,
        "why patches?\n\n"
        "a transformer reads a LIST\nof vectors, not a grid of\npixels.\n\n"
        "one pixel per token would\nbe 64*64 = 4096 tokens, and\nattention costs O(n^2).\n\n"
        f"one PATCH per token is only\n{cfg.n_patches} tokens -- {(64*64)//cfg.n_patches}x cheaper.\n\n"
        "the patch size is the dial\nbetween detail and cost.",
        va="top", ha="left", fontsize=10.5, family="monospace", color="#223")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# ---- 5. raw numbers --------------------------------------------------------
ax = clean(fig.add_subplot(gs[2, 0]),
           "5. flatten each patch\n[16 patches, 768 raw pixel values]", 11)
ax.imshow(patches[0].detach().numpy(), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_ylabel("patch index", fontsize=9)

# ---- 6. after the linear layer --------------------------------------------
ax = clean(fig.add_subplot(gs[2, 1]),
           "6. Linear(768 -> 64)\n[16, 64]  now they are FEATURES", 11)
ax.imshow(parts["embedded"][0].detach().numpy(), aspect="auto", cmap="RdBu_r")

# ---- 7. position embeddings ------------------------------------------------
ax = clean(fig.add_subplot(gs[2, 2]),
           "7. learned position embedding\none row per patch SLOT", 11)
ax.imshow(parts["pos_embed"][0].detach().numpy(), aspect="auto", cmap="RdBu_r")

# ---- 8. the tokens ---------------------------------------------------------
ax = clean(fig.add_subplot(gs[2, 3]),
           "8. sum = the 16 vision tokens\n'what' + 'where', ready for attention", 11)
ax.imshow(tokens[0].detach().numpy(), aspect="auto", cmap="RdBu_r")

fig.suptitle("STEP 1+2 :  an image becomes a sequence of vectors  "
             "(panels 1-5 move numbers around; 6-8 learn)",
             fontsize=13, y=0.99)
save(fig, "01_patching.png")

# ---- a second, simpler figure: what ONE patch actually is ------------------
fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
pick = 2
r, c = pick // G, pick % G
clean(axes[0], f"the whole image\n(patch {pick} highlighted)")
axes[0].imshow(img_np(x[0]))
axes[0].add_patch(plt.Rectangle((c * P - .5, r * P - .5), P, P, fill=False,
                                edgecolor="orange", lw=3))
clean(axes[1], f"patch {pick} alone\n{P} x {P} x 3 pixels")
axes[1].imshow(im[r * P:(r + 1) * P, c * P:(c + 1) * P])
clean(axes[2], f"flattened: {3*P*P} numbers")
axes[2].plot(patches[0, pick].detach().numpy(), lw=.7, color="#3355cc")
axes[2].set_xticks([]); axes[2].set_yticks([])
clean(axes[3], f"after Linear: {cfg.vision_dim} numbers\n= one 'visual word'")
axes[3].bar(range(cfg.vision_dim), parts["embedded"][0, pick].detach().numpy(),
            color="#cc5533", width=1.0)
axes[3].set_xticks([]); axes[3].set_yticks([])
fig.suptitle(f"zoom in on a single patch: {3*P*P} pixels -> one {cfg.vision_dim}-dim vector",
             fontsize=12)
save(fig, "01b_single_patch.png")

"""
DEMO 5 -- "Prove that each piece actually does something."
==========================================================
Run:  python train.py  &&  python demos/demo_05_ablations.py

Five experiments on the SAME trained model.  Nothing is retrained; we break one
thing at a time and watch the caption change.  Experiment 3 has a genuinely
surprising result -- do not skip it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import torch

from vlm.config import Config, pick_device
from vlm.data import caption_for, render, to_tensor
from vlm.step1_patchify import patchify, unpatchify
from vlm.step7_vlm import VLM
from vlm.viz import clean, draw_patch_grid, img_np, save

CKPT = "checkpoints/vlm.pt"
if not os.path.exists(CKPT):
    sys.exit("run `python train.py` first")

blob = torch.load(CKPT, map_location="cpu")
cfg = Config(**blob["cfg"])
device = pick_device()
model = VLM(cfg); model.load_state_dict(blob["model"]); model.eval().to(device)
P, G = cfg.patch_size, cfg.grid_size
torch.manual_seed(0)


def say(img_tensor):
    texts, _ = model.generate(img_tensor.unsqueeze(0).to(device),
                              max_new_tokens=cfg.max_text_len - 1)
    return texts[0]


color, shape, pos = "red", "circle", ("top", "left")
truth = caption_for(color, shape, pos)
x = to_tensor(render(color, shape, pos, cfg.image_size, seed=42))
results = []

# ---- 1. control -----------------------------------------------------------
results.append(("1. the real image\n(control)", x, say(x), "should be correct", "same"))

# ---- 2. physically move the object ----------------------------------------
# swap the WHOLE top-left quadrant (patches 0,1,4,5) with the bottom-right one
# (10,11,14,15).  The object is carried along with its tiles.
perm = list(range(cfg.n_patches))
for a, b in [(0, 10), (1, 11), (4, 14), (5, 15)]:
    perm[a], perm[b] = perm[b], perm[a]
p = patchify(x.unsqueeze(0), P)
x_moved = unpatchify(p[:, perm], P, G)[0]
results.append(("2. top-left and bottom-right\nquadrants swapped", x_moved,
                say(x_moved), "location should FOLLOW the tiles", "diff"))

# ---- 3. kill the ViT's position embeddings --------------------------------
saved = model.vision.patch_embed.pos_embed.data.clone()
model.vision.patch_embed.pos_embed.data.zero_()
results.append(("3. ViT position embeddings\nzeroed out", x, say(x),
                "surprise: still correct!", "same"))
model.vision.patch_embed.pos_embed.data.copy_(saved)

# ---- 4. destroy the ORDER of the image tokens -----------------------------
model.shuffle_image_tokens = torch.randperm(cfg.n_image_tokens)
results.append(("4. image tokens shuffled\nbefore the LLM", x, say(x),
                "NOW the location breaks", "diff"))
model.shuffle_image_tokens = None

# ---- 5. no image at all ---------------------------------------------------
blank = torch.zeros_like(x)
results.append(("5. blank image\n(nothing to see)", blank, say(blank),
                "it makes something up", "diff"))

# ---- report ---------------------------------------------------------------
print(f"\n  ground truth: {truth}\n")
fig, axes = plt.subplots(1, len(results), figsize=(3.5 * len(results), 4.6))
for ax, (title, im, out, note, expect) in zip(axes, results):
    same = out.strip() == truth
    # green = "behaved the way the lesson predicts", not "caption is correct"
    ok = same if expect == "same" else not same
    clean(ax, title, 11)
    ax.imshow(img_np(im))
    draw_patch_grid(ax, cfg.image_size, P, color="#888", lw=.7, numbers=False)
    ax.set_xlabel(f'"{out}"\n{note}', fontsize=9.5,
                  color="#177245" if ok else "#b02020")
    print(f"  {title.splitlines()[0]:<32} -> \"{out}\"")

fig.suptitle("break one piece at a time and watch the caption change\n"
             f'ground truth: "{truth}"   |   green = behaved as the lesson predicts',
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.86])
save(fig, "05_ablations.png")

print(f"""
  How to read this with a student
  -------------------------------
  #2  The object never "moved" in any semantic sense -- we only swapped tiles.
      The caption follows the tiles, so the model really is consuming PATCHES.

  #3  THE INTERESTING ONE.  Zeroing the ViT's position embeddings does NOT
      destroy the location.  Why?  Because the image tokens are handed to the
      LLM in patch order, and the LLM adds its OWN position embedding over the
      whole sequence.  Position 0 = patch 0 = top-left, and the LLM learned
      that.  Location information had a second, unblocked route.

  #4  Which is exactly why this one works: shuffle the ORDER of the image
      tokens and both routes are gone, so the location word finally breaks
      while colour and shape survive.
      Lesson: an ablation that "does nothing" usually means you missed a path.

  #5  With no visual evidence the model still emits a confident, grammatical
      sentence, because the LLM half has a strong prior over what captions look
      like.  That is what hallucination IS -- architecture, not a bug.
""")

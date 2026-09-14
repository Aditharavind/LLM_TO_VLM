"""
DEMO 4 -- "Which patch is the model looking at when it says 'red'?"
===================================================================
Run:  python train.py  &&  python demos/demo_04_where_it_looks.py

This is the payoff demo.

The LLM generates the caption one word at a time.  Because the image tokens sit
at the FRONT of the same sequence, every text position can attend back to them.
So for each word it produces we can read off exactly how much attention went to
each of the 16 image tokens -- and each image token corresponds to a known
square of the picture.

Result: a heatmap over the image, per generated word.  Expect the location
words ("top", "left") to light up the patch the object is actually in.
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

CKPT = "checkpoints/vlm.pt"
if not os.path.exists(CKPT):
    sys.exit("run `python train.py` first -- this demo needs a trained model")

blob = torch.load(CKPT, map_location="cpu")
cfg = Config(**{k: v for k, v in blob["cfg"].items()})
device = pick_device()
model = VLM(cfg)
model.load_state_dict(blob["model"])
model.eval().to(device)
tok = model.tok
M = model.n_image_tokens
G = cfg.grid_size

# four different images so we can see the attention MOVE with the object
cases = [("red", "circle", ("top", "left")),
         ("blue", "square", ("bottom", "right")),
         ("green", "triangle", ("top", "right")),
         ("yellow", "circle", ("bottom", "left"))]

for ci, (color, shape, pos) in enumerate(cases):
    img = render(color, shape, pos, cfg.image_size, seed=100 + ci)
    x = to_tensor(img).unsqueeze(0).to(device)

    # 1. let the model write its own caption
    texts, ids = model.generate(x, max_new_tokens=cfg.max_text_len - 1)
    said = texts[0]

    # 2. replay that caption through the model and grab the attention
    with torch.no_grad():
        _, attns = model(x, ids, return_attn=True)   # each [1, heads, M+T, M+T]

    # attention from every text position back onto the image block,
    # averaged over heads and over the last two layers (less noisy)
    layers = attns[-2:]
    a = torch.stack([l[0, :, :, :M].mean(0) for l in layers]).mean(0)  # [M+T, M]
    a = a.cpu().numpy()

    words = [tok.itos[i] for i in ids[0].tolist()]
    # position M+j holds word j and is the position that PREDICTS word j+1
    steps = [(j, words[j], words[j + 1] if j + 1 < len(words) else "-")
             for j in range(len(words) - 1)]

    # silhouette of the object, so every heatmap has a visual reference of
    # WHERE the thing actually is
    rgb = img_np(x[0].cpu()).astype(np.float32)
    obj_mask = (np.abs(rgb - rgb[0, 0]).sum(-1) > 30).astype(float)

    n = len(steps)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.05 * (n + 1), 2.9))
    clean(axes[0], f"input image\ntruth: {caption_for(color, shape, pos)}", 9)
    axes[0].imshow(img_np(x[0].cpu()))
    draw_patch_grid(axes[0], cfg.image_size, cfg.patch_size, color="#888", lw=.7,
                    numbers=False)

    for k, (j, cur, nxt) in enumerate(steps):
        heat = a[M + j].reshape(G, G)
        up = np.kron(heat, np.ones((cfg.patch_size, cfg.patch_size)))
        ax = clean(axes[k + 1], f"after '{cur}'\n-> says '{nxt}'", 9)
        ax.imshow(up, cmap="inferno")
        ax.contour(obj_mask, levels=[0.5], colors="white", linewidths=1.6)
        draw_patch_grid(ax, cfg.image_size, cfg.patch_size, color="#ffffff55",
                        lw=.6, numbers=False)
        best = int(heat.argmax())
        ax.add_patch(plt.Rectangle(((best % G) * cfg.patch_size - .5,
                                    (best // G) * cfg.patch_size - .5),
                                   cfg.patch_size, cfg.patch_size, fill=False,
                                   edgecolor="lime", lw=2))
    fig.suptitle(f"where the LANGUAGE model looks while writing:  \"{said}\"      "
                 "(white outline = the real object, green box = most-attended patch)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    save(fig, f"04_where_it_looks_{ci}.png")
    print(f"  case {ci}: said '{said}'")

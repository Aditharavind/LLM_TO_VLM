"""Small plotting helpers shared by the demos. Nothing conceptual lives here."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import to_pil

ASSETS = "assets"


def img_np(x):
    """tensor [3,H,W] -> HxWx3 uint8 array for imshow."""
    return np.asarray(to_pil(x))


def draw_patch_grid(ax, image_size, patch_size, color="k", lw=1.2, numbers=True,
                    fontsize=8):
    """Overlay the patch boundaries (and patch indices) on an imshow axis."""
    g = image_size // patch_size
    for i in range(1, g):
        ax.axhline(i * patch_size - 0.5, color=color, lw=lw)
        ax.axvline(i * patch_size - 0.5, color=color, lw=lw)
    if numbers:
        for idx in range(g * g):
            r, c = idx // g, idx % g
            ax.text(c * patch_size + 2, r * patch_size + 9, str(idx),
                    color=color, fontsize=fontsize, fontweight="bold")


def clean(ax, title=None, fontsize=10):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=fontsize)
    return ax


def save(fig, name):
    import os
    os.makedirs(ASSETS, exist_ok=True)
    path = os.path.join(ASSETS, name)
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {path}")
    return path

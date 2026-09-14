"""
A toy dataset we can *see*.

Each sample is a 64x64 image containing ONE colored shape placed in ONE of the
four quadrants, plus the matching caption:

        "a red circle in the top left"

Why this dataset?  Because the caption cannot be guessed from text alone -- the
model is forced to actually read the image to get colour, shape and position
right.  That makes it obvious when the vision path is working.
"""
import random
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from .tokenizer import WordTokenizer

COLORS = {
    "red":    (220, 50, 50),
    "green":  (60, 180, 75),
    "blue":   (60, 100, 220),
    "yellow": (240, 200, 40),
}
SHAPES = ["circle", "square", "triangle"]
POSITIONS = {            # (row, col) of the quadrant
    ("top", "left"):     (0, 0),
    ("top", "right"):    (0, 1),
    ("bottom", "left"):  (1, 0),
    ("bottom", "right"): (1, 1),
}
BACKGROUND = (245, 245, 248)


def render(color: str, shape: str, pos: Tuple[str, str], image_size: int = 64,
           jitter: bool = True, seed: int = None) -> Image.Image:
    """Draw one shape of one colour inside one quadrant. Returns a PIL image."""
    rng = random.Random(seed)
    img = Image.new("RGB", (image_size, image_size), BACKGROUND)
    draw = ImageDraw.Draw(img)

    half = image_size // 2
    row, col = POSITIONS[pos]
    # centre of the chosen quadrant, with a little wobble so the model can't
    # memorise exact pixel coordinates
    cx = col * half + half // 2
    cy = row * half + half // 2
    if jitter:
        cx += rng.randint(-3, 3)
        cy += rng.randint(-3, 3)
    r = rng.randint(9, 12) if jitter else 10
    rgb = COLORS[color]

    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL image -> float tensor [3, H, W] scaled to roughly [-1, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0      # [H, W, 3] in 0..1
    arr = (arr - 0.5) / 0.5                              # centre it
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # [3, H, W]


def to_pil(x: torch.Tensor) -> Image.Image:
    """Inverse of to_tensor, for plotting."""
    arr = (x.detach().cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5).clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def caption_for(color: str, shape: str, pos: Tuple[str, str]) -> str:
    return f"a {color} {shape} in the {pos[0]} {pos[1]}"


def all_combinations() -> List[Tuple[str, str, Tuple[str, str]]]:
    """4 colours x 3 shapes x 4 quadrants = 48 distinct captions."""
    return [(c, s, p) for c in COLORS for s in SHAPES for p in POSITIONS]


class ShapesDataset(Dataset):
    """Generates (image, token_ids) pairs on the fly."""

    def __init__(self, n_samples: int = 2000, image_size: int = 64,
                 max_text_len: int = 16, seed: int = 0):
        self.n_samples = n_samples
        self.image_size = image_size
        self.max_text_len = max_text_len
        self.tok = WordTokenizer()
        self.combos = all_combinations()
        self.seed = seed

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        rng = random.Random(self.seed * 1_000_003 + idx)
        color, shape, pos = rng.choice(self.combos)
        img = render(color, shape, pos, self.image_size, jitter=True,
                     seed=rng.randint(0, 2**31 - 1))

        ids = self.tok.encode(caption_for(color, shape, pos))      # <bos> ... <eos>
        assert len(ids) <= self.max_text_len, f"caption too long: {ids}"
        ids = self.tok.pad_to(ids, self.max_text_len)
        return to_tensor(img), torch.tensor(ids, dtype=torch.long)

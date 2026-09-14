"""
STEP 1 -- Cut the image into patches.
=====================================

A transformer cannot read a grid of pixels; it reads a *sequence* of vectors.
So the very first thing a VLM does is chop the image into square tiles
("patches") and lay them out in a line, reading order: left->right, top->bottom.

    64x64 image, patch_size 16
        -> a 4x4 grid of patches
        -> a sequence of 16 patches
        -> each patch is 16*16*3 = 768 numbers

Nothing is learned here.  This is pure re-arranging: not a single number
changes value, they are only *moved*.  Run demos/demo_01_patching.py to watch
it happen.
"""
import torch


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    [B, C, H, W]  ->  [B, N, C*patch*patch]

    N = (H/patch) * (W/patch), in raster order (row by row).
    """
    B, C, H, W = images.shape
    p = patch_size
    assert H % p == 0 and W % p == 0, "image size must be divisible by patch size"

    # .unfold(dim, size, step) slides a window along one dimension.
    # Do it along H, then along W, and we have every tile.
    x = images.unfold(2, p, p).unfold(3, p, p)   # [B, C, gh, gw, p, p]
    gh, gw = x.shape[2], x.shape[3]

    # Put the grid dims next to each other so we can flatten them into "N",
    # and put the pixel dims last so we can flatten them into one long vector.
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # [B, gh, gw, C, p, p]
    x = x.reshape(B, gh * gw, C * p * p)          # [B, N, C*p*p]
    return x


def unpatchify(patches: torch.Tensor, patch_size: int, grid_size: int,
               channels: int = 3) -> torch.Tensor:
    """The exact inverse of patchify -- handy to prove we lost nothing."""
    B, N, D = patches.shape
    p, g = patch_size, grid_size
    x = patches.reshape(B, g, g, channels, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()   # [B, C, g, p, g, p]
    return x.reshape(B, channels, g * p, g * p)


def patch_index_to_rowcol(idx: int, grid_size: int):
    """Patch #6 in a 4x4 grid lives at row 1, col 2.  Useful for plotting."""
    return idx // grid_size, idx % grid_size


if __name__ == "__main__":
    x = torch.randn(2, 3, 64, 64)
    p = patchify(x, 16)
    print("image   ", tuple(x.shape))
    print("patches ", tuple(p.shape), " <- 16 patches of 768 numbers each")
    back = unpatchify(p, 16, 4)
    print("perfectly reversible:", torch.allclose(x, back))

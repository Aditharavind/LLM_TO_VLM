"""
STEP 2 -- Turn each patch into a vector the transformer can use.
================================================================

After step 1 every patch is 768 raw pixel numbers.  Two things are missing:

 1. those numbers are raw pixels, not *features*  -> fix with one Linear layer
 2. the sequence has no idea where each patch came from, because attention is
    permutation-invariant (shuffle the patches and it cannot tell)
                                                   -> fix with position embeddings

So:   patch pixels --Linear--> patch embedding  +  position embedding = token

That "+ position embedding" is the whole reason the model knows the difference
between "top left" and "bottom right".

Bonus: the same operation can be written as a Conv2d with kernel=stride=patch.
`PatchEmbedding.forward_conv` does exactly that and we assert the two agree.
"""
import torch
import torch.nn as nn

from .step1_patchify import patchify


class PatchEmbedding(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=3, dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size ** 2
        self.patch_dim = in_channels * patch_size * patch_size   # 768

        # 768 raw pixel values -> a 64-dim feature vector
        self.proj = nn.Linear(self.patch_dim, dim)

        # One learned vector per patch slot ("this is slot 0", "this is slot 1"...)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor, return_parts: bool = False):
        """[B, 3, 64, 64] -> [B, 16, dim]"""
        patches = patchify(images, self.patch_size)   # [B, N, 768]   just moving
        embedded = self.proj(patches)                 # [B, N, dim]   learned
        tokens = embedded + self.pos_embed            # [B, N, dim]   + "where"
        if return_parts:
            return tokens, {"patches": patches, "embedded": embedded,
                            "pos_embed": self.pos_embed}
        return tokens

    @torch.no_grad()
    def forward_conv(self, images: torch.Tensor) -> torch.Tensor:
        """
        The same maths, expressed as a convolution -- this is how real ViTs are
        coded.  A conv with kernel_size == stride == patch_size looks at each
        patch exactly once and never overlaps, so it *is* "patchify + linear".
        """
        w = self.proj.weight.view(-1, 3, self.patch_size, self.patch_size)
        out = torch.nn.functional.conv2d(images, w, self.proj.bias,
                                         stride=self.patch_size)   # [B, dim, g, g]
        return out.flatten(2).transpose(1, 2) + self.pos_embed     # [B, N, dim]


if __name__ == "__main__":
    pe = PatchEmbedding()
    x = torch.randn(1, 3, 64, 64)
    a = pe(x)
    b = pe.forward_conv(x)
    print("tokens:", tuple(a.shape))
    print("linear-on-patches == conv2d :", torch.allclose(a, b, atol=1e-5))

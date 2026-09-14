"""All the knobs in one place, so nothing is hidden in a function signature."""
from dataclasses import dataclass


@dataclass
class Config:
    # ---------------- image side ----------------
    image_size: int = 64      # we use tiny 64x64 images so everything runs fast
    patch_size: int = 16      # -> (64/16)^2 = 16 patches per image
    in_channels: int = 3      # RGB

    vision_dim: int = 64      # width of the ViT
    vision_heads: int = 4     # multi-head attention: 4 heads of size 64/4 = 16
    vision_layers: int = 4    # how many transformer blocks in the ViT

    # ---------------- language side ----------------
    llm_dim: int = 128        # width of the LLM  (note: != vision_dim on purpose!)
    llm_heads: int = 4
    llm_layers: int = 4
    max_text_len: int = 16    # longest caption (in tokens)

    # ---------------- the bridge ----------------
    # "mlp"       -> one image patch becomes one LLM token   (LLaVA style)
    # "resampler" -> N patches get squeezed into n_query tokens via cross-attention
    #                (Flamingo / BLIP-2 "Q-Former" style)
    connector: str = "mlp"
    n_query_tokens: int = 8   # only used when connector == "resampler"
    # How spread out the learned queries start. Too small and all the queries
    # stay near-identical forever ("query collapse") -- see demos/demo_06.
    query_init_std: float = 0.02

    dropout: float = 0.0

    @property
    def grid_size(self) -> int:
        """How many patches along one side, e.g. 64/16 = 4."""
        assert self.image_size % self.patch_size == 0, "image must divide into whole patches"
        return self.image_size // self.patch_size

    @property
    def n_patches(self) -> int:
        """Total patches per image = grid_size ** 2."""
        return self.grid_size ** 2

    @property
    def n_image_tokens(self) -> int:
        """How many tokens the LLM will actually receive from the image."""
        return self.n_query_tokens if self.connector == "resampler" else self.n_patches


def pick_device(prefer: str = "auto") -> str:
    """
    'auto' -> use the GPU if it exists AND is actually usable, else CPU.
    (This model is ~1M parameters; CPU trains it in well under a minute.)
    """
    import torch
    if prefer != "auto":
        return prefer
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")   # some GPUs are visible but busy
            return "cuda"
        except Exception:
            pass
    return "cpu"

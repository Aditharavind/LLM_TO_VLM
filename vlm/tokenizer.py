"""
The world's smallest tokenizer.

Real LLMs use BPE (~50k sub-word pieces).  We use whole words, because the
point of this repo is the *image -> token* path, not the text path.
"""
from typing import List

# Special tokens first, so their ids are easy to remember.
SPECIALS = ["<pad>", "<bos>", "<eos>"]
WORDS = [
    "a", "in", "the",
    "red", "green", "blue", "yellow",          # colors
    "circle", "square", "triangle",            # shapes
    "top", "bottom", "left", "right",          # positions
]


class WordTokenizer:
    def __init__(self):
        self.itos: List[str] = SPECIALS + WORDS
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, add_bos=True, add_eos=True) -> List[int]:
        ids = [self.stoi[w] for w in text.split()]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special=True) -> str:
        out = []
        for i in ids:
            i = int(i)
            tok = self.itos[i]
            if skip_special and tok in SPECIALS:
                continue
            out.append(tok)
        return " ".join(out)

    def pad_to(self, ids: List[int], length: int) -> List[int]:
        return ids + [self.pad_id] * (length - len(ids))

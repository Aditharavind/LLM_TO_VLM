"""
DEMO 2 -- "Follow one image through the whole model, shape by shape."
=====================================================================
Run:  python demos/demo_02_shape_journey.py

No pictures, just printed tensor shapes.  This is the demo to walk through
on a whiteboard: every arrow is one line of code in vlm/.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from vlm.config import Config
from vlm.data import caption_for, render, to_tensor
from vlm.step1_patchify import patchify
from vlm.step7_vlm import VLM

W = 78


def rule(ch="-"):
    print(ch * W)


def line(what, tensor, note=""):
    shape = "x".join(str(s) for s in tuple(tensor.shape))
    print(f"  {what:<34} [{shape:<16}]  {note}")


def header(t):
    print()
    rule("=")
    print(f"  {t}")
    rule("=")


cfg = Config()
torch.manual_seed(0)
model = VLM(cfg)
tok = model.tok

color, shape, pos = "red", "circle", ("top", "left")
img = render(color, shape, pos, cfg.image_size, seed=1)
x = to_tensor(img).unsqueeze(0)
caption = caption_for(color, shape, pos)
ids = torch.tensor([tok.encode(caption)])

print()
print(f"  caption : '{caption}'")
print(f"  tokens  : {tok.encode(caption)}  ->  {[tok.itos[i] for i in tok.encode(caption)]}")

header("THE IMAGE SIDE")
line("input image", x, "3 colour channels, 64x64 pixels")

p = patchify(x, cfg.patch_size)
line("after patchify", p, f"{cfg.n_patches} patches x (3*{cfg.patch_size}*{cfg.patch_size}) pixels")
print(f"  {'':<34} {'':<18}  ^ pure re-arranging, nothing learned")

emb = model.vision.patch_embed(x)
line("after patch embedding", emb, "Linear + position embedding")

feats = model.vision(x)
line(f"after the ViT ({cfg.vision_layers} blocks)", feats,
     "same shape, but every patch")
print(f"  {'':<34} {'':<18}  has now SEEN the whole image")

img_tokens = model.connector(feats)
line(f"after the connector ({cfg.connector})", img_tokens,
     f"{cfg.vision_dim} -> {cfg.llm_dim}: now LLM-shaped")

header("THE TEXT SIDE")
line("token ids", ids, "just integers")
txt_tokens = model.llm.embed_tokens(ids)
line("after the embedding table", txt_tokens, f"each id -> a {cfg.llm_dim}-dim vector")

header("THE JOIN  <-- this is the whole trick")
seq, M = model.build_sequence(x, ids)
line("torch.cat([image, text])", seq,
     f"{M} image tokens + {ids.shape[1]} text tokens")
print()
print("  what the LLM sees at each position:")
labels = [f"IMG{i}" for i in range(M)] + [tok.itos[i] for i in ids[0].tolist()]
for start in range(0, len(labels), 8):
    chunk = labels[start:start + 8]
    print("    pos " + " ".join(f"{start+i:>7}" for i in range(len(chunk))))
    print("        " + " ".join(f"{c:>7}" for c in chunk))
print()
print("  the LLM has NO idea the first block came from pixels.")
print("  to it they are just vectors, exactly like word vectors.")

header("THE OUTPUT")
logits = model.llm(inputs_embeds=seq)
line("logits", logits, f"a score for each of the {len(tok)} words,")
print(f"  {'':<34} {'':<18}  at every position")
loss = model.loss(x, ids)
print(f"\n  loss = {loss.item():.3f}   (untrained; ln({len(tok)}) = 2.833 is 'no idea')")

header("WHO HAS THE PARAMETERS?")
rep = model.param_report()
for k, v in rep.items():
    bar = "#" * int(40 * v / rep["TOTAL"])
    print(f"  {k:<14} {v:>9,d}  {bar}")
print()
print("  note how tiny the connector is.  In real VLMs it is often the ONLY")
print("  part trained in stage 1 -- the ViT and the LLM are frozen.")
rule()

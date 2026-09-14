# Turning an LLM into a VLM — from scratch, small enough to read

A teaching repo. It builds a working vision-language model out of a Vision
Transformer, an MLP, and multi-head attention — about 1M parameters total, trains
to 100% on a toy captioning task **in 36 seconds on a CPU**, and prints/plots what
is happening at every step.

The one idea the whole repo exists to deliver:

> A language model only ever eats **vectors**. So to make it see, you don't change
> the language model at all — you turn an image into vectors that *look like* word
> vectors to it, and paste them on the front of the sequence.

That's it. Everything else is detail.

---

## Quick start

```bash
pip install -r requirements.txt

python demos/demo_01_patching.py      # how an image becomes a sequence  (no training needed)
python demos/demo_02_shape_journey.py # follow one image through, shape by shape
python train.py                       # ~36s on CPU -> 100% exact-match captions
python demos/demo_03_vit_attention.py # what does a patch look at?
python demos/demo_04_where_it_looks.py# where the LLM looks while writing each word
python demos/demo_05_ablations.py     # break one piece at a time and watch it fail
```

Also open **`explainer.html`** in a browser — an interactive version of the same
seven steps, good for teaching at a screen. Live copy:


**She can drop in her own photo** (or drag it, or paste it) and watch that exact
image move through every stage: it gets centre-cropped and resized the way a real
image preprocessor does, then cut into patches, embedded, attended over, projected
and joined to her own caption — which she can type herself.

Everything on the page is live and recomputed from the image on screen:

- **resolution** 64² / 96² / 128² and **patch size** 8 / 16 / 32 — patch sizes that
  don't divide the image, or that would blow the sequence length past 64, are
  disabled and say why. The prose numbers ("64×64 would be 4,096 tokens") rewrite
  themselves to match.
- **explode** the grid into separate tiles and hover to link a tile to its place in
  the sequence
- **click any patch** to see where each of the four attention heads sends it — the
  attention is really computed (Q·Kᵀ/√d, softmax), on toy weights
- **toggle position embeddings off** and watch identical patches become
  indistinguishable
- **switch the bridge** between MLP projector and cross-attention resampler
- **click any token** in the final sequence to see its causal window

The image is read in the browser tab and never uploaded anywhere.

Or run everything in one go:

```bash
python run_all.py     # every demo + both trainings, ~4 minutes
```

---

## The seven steps

Read `vlm/` in this order. Each file runs standalone (`python -m vlm.step3_attention`)
and prints a self-check.

| file | what it does | shape after |
|---|---|---|
| `step1_patchify.py` | cut the image into 16 tiles, lay them in a line | `[16, 768]` |
| `step2_patch_embed.py` | `Linear(768→64)` + learned position embedding | `[16, 64]` |
| `step3_attention.py` | multi-head attention, written out longhand | — |
| `step4_vit.py` | stack 4 blocks → the Vision Transformer | `[16, 64]` |
| `step5_projector.py` | **the bridge**: 64-dim → 128-dim | `[16, 128]` |
| `step6_tiny_llm.py` | a small causal LLM that accepts `inputs_embeds` | — |
| `step7_vlm.py` | `torch.cat([image_tokens, text_tokens])` | `[16+9, 128]` |

### Step 1 is not learned

`patchify` has zero parameters and loses nothing — `unpatchify` reconstructs the
image exactly, and the repo asserts it. Pixels are only *moved*. Students often
assume something clever happens here; it doesn't.

Why patches at all? One pixel per token would be 4096 tokens and attention is
O(n²). One patch per token is 16. **Patch size is the dial between detail and cost.**

### Step 2 is where "top left" comes from

Attention is permutation-invariant — shuffle the patches and it genuinely cannot
tell. The learned position embedding is the only thing that says *where* a patch
was. (`step2` also shows the `Conv2d(kernel=stride=patch)` formulation real ViTs
use, and asserts it is numerically identical.)

### Step 3: why *multi*-head

One attention map can express exactly one kind of relationship. Split the 64-dim
vector into 4 chunks of 16, run four attentions in parallel, concatenate. Same
cost, four opinions. The code uses explicit `matmul` + `softmax` rather than
`F.scaled_dot_product_attention` so you can print the attention matrix.

### Step 5 is the actual conversion

Two bridges are implemented:

- **`MLPProjector`** (LLaVA) — `Linear → GELU → Linear`, one patch → one token.
  Keeps all detail; costs one context token per patch.
- **`AttentionResampler`** (Flamingo / BLIP-2 Q-Former) — a few *learned* query
  vectors cross-attend to the patches. 16 patches → 8 tokens, regardless of how
  many patches went in.

The connector is ~25k parameters against the LLM's ~800k. In real VLM training it
is often the **only** thing trained in stage 1 — try `python train.py --stage1`,
which freezes the ViT and the LLM.

### Step 7 is three lines

```python
img_tokens = connector(vit(images))        # [B, 16, 128]
txt_tokens = llm.embed_tokens(text_ids)    # [B,  9, 128]
seq = torch.cat([img_tokens, txt_tokens], dim=1)
logits = llm(inputs_embeds=seq)
```

`inputs_embeds` is the door. Every real LLM exposes it. The model has no idea the
first 16 vectors came from pixels.

---

## The dataset is deliberately un-guessable

64×64 images, one coloured shape in one quadrant, caption
`"a red circle in the top left"`. 4 colours × 3 shapes × 4 quadrants = 48 captions.
You cannot produce the right caption from text priors alone, so when accuracy hits
100% you know the vision path is doing real work.

Watch epoch 1 during training: the model says *the same caption for every image*.
It learns the caption template first and only later learns to look. That is worth
pausing on.

---

## Three results worth teaching

These are measured by the demos in this repo, not claims.

**1. An ablation that "does nothing" usually means you missed a path.**
Zeroing the ViT's position embeddings does **not** break the location word. The
image tokens reach the LLM in patch order, and the LLM adds its *own* position
embedding over the sequence — so location had a second, unblocked route. Shuffle
the image-token order (`demo_05`, experiment 4) and the location finally breaks
while colour and shape survive.

**2. Hallucination is architectural.**
Feed the trained model a blank image and it still emits a confident, grammatical
caption (`demo_05`, experiment 5). The language half has a strong prior over what
captions look like. Nothing is broken; this is what the architecture does.

**3. Learned queries collapse if you let them — and one number fixes it.**
With the default `query_init_std=0.02`, all 8 resampler queries train to
*essentially the same* attention pattern: max L1 distance between any two query
rows is **0.015** — eight tokens carrying one token's worth of information.
Caption accuracy stalls around 60%.

Nothing in the loss pushes the queries apart, and they start almost on top of each
other. Spread them at init and both numbers move together:

| `--query-init-std` | query separation | caption accuracy |
|---|---|---|
| 0.02 (default) | 0.015 — collapsed | ~62% |
| 1.0 | 1.79 — genuinely different | **100%** |

```bash
python train.py --connector resampler --query-init-std 1.0 \
    --epochs 40 --lr 5e-4 --out checkpoints/vlm_resampler.pt
python demos/demo_06_resampler.py     # measures the separation for you
```

This is a good lesson in its own right: the architecture was never the problem,
and the demo tells you which it was because it *measures* the thing instead of
showing a colourful picture and hoping.

---

## Honest notes

- **The ViT's own attention is close to uniform** on this task (~0.06 against a
  0.0625 uniform baseline). With 16 patches and one object, a patch doesn't need
  to hunt. `demo_03` prints the numbers rather than hiding them behind a colormap.
  The *sharp*, interpretable attention is in `demo_04`, where the language model
  has to find the object in order to name it.
- **The resampler needs `--query-init-std 1.0` to match the MLP projector** here
  (see result 3). At the default init it collapses and scores ~62%. Fixed, it
  reaches 100% with half the image tokens — but token *k* no longer corresponds to
  any one place in the image, so you lose the ability to paint attention back onto
  the picture the way `demo_04` does. That is the real trade.
- Everything is tiny and on purpose: 4 layers, 64/128 dims, word-level tokenizer,
  no KV cache. Generation recomputes the whole sequence each step because that is
  easier to read.

---

## Files

```
vlm/
  config.py            every knob in one place
  tokenizer.py         17-word vocabulary
  data.py              the toy shapes dataset
  step1..step7         the model, in reading order
  viz.py               plotting helpers
demos/                 six runnable demos, each saves a figure to assets/
train.py               training loop + loss curve
explainer.html         interactive browser walkthrough
```

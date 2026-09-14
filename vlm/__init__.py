"""
A tiny, readable Vision-Language Model (VLM), built step by step.

The whole idea in one line:
    an LLM only eats *vectors*.  So if we can turn an image into vectors that
    "look like" word-vectors to the LLM, the LLM can talk about the image.

Read the files in this order:
    step1_patchify.py    image  -> patches
    step2_patch_embed.py patches-> vectors (+ position info)
    step3_attention.py   multi-head attention, written out by hand
    step4_vit.py         stack of attention blocks = the Vision Transformer
    step5_projector.py   the "bridge": vision vectors -> LLM-shaped vectors
    step6_tiny_llm.py    a small causal LLM (the thing we are upgrading)
    step7_vlm.py         glue it all together
"""

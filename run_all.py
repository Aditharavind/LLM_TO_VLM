"""
Run the whole lesson end to end:  python run_all.py

  1. the two demos that need no training
  2. train the MLP-connector VLM      (~40s on CPU)
  3. the demos that need a trained model
  4. train the resampler variant      (~2 min) and its demo

Every figure lands in assets/.
"""
import subprocess, sys, time

STEPS = [
    ("image -> sequence (no training needed)", [sys.executable, "demos/demo_01_patching.py"]),
    ("shape-by-shape walkthrough",             [sys.executable, "demos/demo_02_shape_journey.py"]),
    ("train the VLM",                          [sys.executable, "train.py"]),
    ("what does a patch look at?",             [sys.executable, "demos/demo_03_vit_attention.py"]),
    ("where the LLM looks per word",           [sys.executable, "demos/demo_04_where_it_looks.py"]),
    ("break it on purpose",                    [sys.executable, "demos/demo_05_ablations.py"]),
    ("train the cross-attention variant",      [sys.executable, "train.py", "--connector", "resampler",
                                                "--epochs", "40", "--lr", "5e-4",
                                                "--query-init-std", "1.0",
                                                "--out", "checkpoints/vlm_resampler.pt"]),
    ("the learned queries",                    [sys.executable, "demos/demo_06_resampler.py"]),
]

for i, (label, cmd) in enumerate(STEPS, 1):
    print(f"\n{'='*72}\n  [{i}/{len(STEPS)}]  {label}\n{'='*72}")
    t = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"failed: {' '.join(cmd)}")
    print(f"  ({time.time()-t:.0f}s)")

print("\nall done -- open assets/ for the figures, explainer.html for the browser version")

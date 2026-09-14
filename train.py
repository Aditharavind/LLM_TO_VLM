"""
Train the VLM on the toy shapes dataset.

    python train.py                       # default: MLP connector, everything trainable
    python train.py --connector resampler # cross-attention bridge instead
    python train.py --stage1              # freeze ViT + LLM, train ONLY the connector

Watch the loss fall from ~2.83 ("no idea", = ln(vocab)) towards ~0.
Then look at the sample captions it prints -- that is the model reading pixels.
"""
import argparse, os, sys, time

import torch
from torch.utils.data import DataLoader

from vlm.config import Config, pick_device
from vlm.data import ShapesDataset, all_combinations, caption_for, render, to_tensor
from vlm.step7_vlm import VLM


def evaluate(model, device, n=8, verbose=True):
    """Caption n unseen images and count exact matches."""
    model.eval()
    combos = all_combinations()
    rng = torch.Generator().manual_seed(1234)
    picks = torch.randperm(len(combos), generator=rng)[:n].tolist()

    imgs, wanted = [], []
    for i in picks:
        c, s, p = combos[i]
        imgs.append(to_tensor(render(c, s, p, model.cfg.image_size, seed=9000 + i)))
        wanted.append(caption_for(c, s, p))
    imgs = torch.stack(imgs).to(device)

    preds, _ = model.generate(imgs, max_new_tokens=model.cfg.max_text_len - 1)
    correct = sum(p.strip() == w for p, w in zip(preds, wanted))
    if verbose:
        for p, w in zip(preds, wanted):
            mark = "OK  " if p.strip() == w else "MISS"
            print(f"    [{mark}] said: {p:<38} | truth: {w}")
    model.train()
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connector", default="mlp", choices=["mlp", "resampler"])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--stage1", action="store_true",
                    help="freeze ViT and LLM, train only the connector")
    ap.add_argument("--out", default="checkpoints/vlm.pt")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--query-init-std", type=float, default=0.02,
                    help="resampler only: how far apart the learned queries start")
    args = ap.parse_args()

    device = pick_device(args.device)
    cfg = Config(connector=args.connector, query_init_std=args.query_init_std)
    ds = ShapesDataset(args.n_samples, cfg.image_size, cfg.max_text_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = VLM(cfg, ds.tok).to(device)
    if args.stage1:
        model.freeze_vision(True)
        model.freeze_llm(True)
        print("stage-1 mode: ViT and LLM are FROZEN, only the connector learns")

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    print(f"device={device}  connector={cfg.connector}  "
          f"image tokens={cfg.n_image_tokens}")
    for k, v in model.param_report().items():
        print(f"  {k:<14} {v:>9,d}")
    print(f"  trainable      {sum(p.numel() for p in trainable):>9,d}\n")

    history = []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        total, nb = 0.0, 0
        for imgs, ids in dl:
            imgs, ids = imgs.to(device), ids.to(device)
            loss = model.loss(imgs, ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            total += loss.item(); nb += 1
            history.append(loss.item())
        acc = evaluate(model, device, n=8, verbose=(ep == args.epochs or ep == 1))
        print(f"epoch {ep:>2}/{args.epochs}  loss {total/nb:.4f}  "
              f"caption-exact-match {acc*100:5.1f}%   ({time.time()-t0:.0f}s)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                "history": history}, args.out)
    print(f"\nsaved -> {args.out}")

    # loss curve
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.plot(history, lw=.6, alpha=.35, color="#3355cc")
        k = max(1, len(history) // 100)
        sm = np.convolve(history, np.ones(k) / k, mode="valid")
        ax.plot(sm, lw=2, color="#cc3322", label="smoothed")
        ax.axhline(np.log(len(ds.tok)), ls="--", c="gray",
                   label=f"ln(vocab)={np.log(len(ds.tok)):.2f} = guessing")
        ax.set_xlabel("training step"); ax.set_ylabel("next-token loss")
        ax.set_title(f"the moment the LLM learns to see  ({cfg.connector} connector)")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        os.makedirs("assets", exist_ok=True)
        out_png = f"assets/02_loss_{cfg.connector}.png"
        fig.savefig(out_png, dpi=130, bbox_inches="tight", facecolor="white")
        print(f"saved -> {out_png}")
    except Exception as e:
        print("(skipped loss plot:", e, ")")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train the discrete diffusion denoising model.

Trains indefinitely until Ctrl+C. Saves checkpoint every --save-every epochs.

Usage:
    python train.py --data data/expressions.txt --max-length 100
    python train.py --data data/expressions.txt --batch-size 512 --num-workers 8 --save-every 5
"""

import argparse
import itertools
import random
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tokenizer import encode_batch, PAD_ID
from diffusion import corrupt_batch, sample_timestep, NUM_TIMESTEPS
from model import DenoisingTransformer


def load_expressions(path: Path, max_exprs: Optional[int] = None) -> list[str]:
    """Load expressions from file, one per line."""
    exprs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                exprs.append(line)
                if max_exprs and len(exprs) >= max_exprs:
                    break
    return exprs


class ExprDataset(Dataset):
    """Dataset that corrupts on-the-fly for parallel loading."""

    def __init__(self, exprs: list[str], max_length: int, seed: int = 42):
        self.exprs = exprs
        self.max_length = max_length
        self.seed = seed

    def __len__(self) -> int:
        return len(self.exprs)

    def __getitem__(self, idx: int):
        # Worker-specific RNG for reproducibility with num_workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            rng = random.Random(self.seed + worker_info.id + worker_info.seed % 2**32)
        else:
            rng = random.Random(self.seed + idx)
        expr = self.exprs[idx]
        clean = encode_batch([expr], self.max_length)[0]
        t = sample_timestep(rng=rng)
        corrupted = corrupt_batch([clean], t, rng=rng)[0]
        pad_mask = [c == PAD_ID for c in clean]
        return (
            torch.tensor(clean, dtype=torch.long),
            torch.tensor(corrupted, dtype=torch.long),
            torch.tensor(t, dtype=torch.long),
            torch.tensor(pad_mask, dtype=torch.bool),
        )


def train_step(
    model: nn.Module,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    t: torch.Tensor,
    pad_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """One training step. Returns loss."""
    model.train()
    clean = clean.to(device, non_blocking=True)
    corrupted = corrupted.to(device, non_blocking=True)
    t = t.to(device, non_blocking=True)
    pad_mask = pad_mask.to(device, non_blocking=True)

    use_amp = scaler is not None
    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
        logits = model(corrupted, t, pad_mask=pad_mask)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            clean.view(-1),
            ignore_index=PAD_ID,
            reduction="mean",
        )

    optimizer.zero_grad(set_to_none=True)
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return loss.item()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train discrete diffusion model")
    parser.add_argument("--data", type=Path, default=Path("data/expressions.txt"))
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256, help="Larger for A100 (256-512)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6, help="Min LR for cosine scheduler")
    parser.add_argument("--save-every", type=int, default=5, help="Save checkpoint every N epochs (Ctrl+C to stop)")
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--max-exprs", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers for GPU throughput")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (AMP on by default for CUDA)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save", type=Path, default=Path("checkpoints/model.pt"))
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    exprs = load_expressions(args.data, args.max_exprs)
    if not exprs:
        print("No expressions loaded.", file=sys.stderr)
        return 1
    print(f"Loaded {len(exprs)} expressions", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    if use_amp:
        print("Using mixed precision (AMP)", file=sys.stderr)

    model = DenoisingTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=args.lr_min
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    dataset = ExprDataset(exprs, args.max_length, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    args.save.parent.mkdir(parents=True, exist_ok=True)
    print("Training indefinitely. Ctrl+C to stop. Checkpoints saved every --save-every epochs.", file=sys.stderr)

    try:
        for epoch in itertools.count(1):
            t0 = time.perf_counter()
            total_loss = 0.0
            n_batches = 0
            for clean, corrupted, t, pad_mask in dataloader:
                loss = train_step(
                    model, clean, corrupted, t, pad_mask, optimizer, device, scaler=scaler
                )
                total_loss += loss
                n_batches += 1
            elapsed = time.perf_counter() - t0
            avg = total_loss / max(n_batches, 1)
            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch}  loss={avg:.4f}  lr={lr:.2e}  {elapsed:.1f}s", file=sys.stderr)

            if epoch % args.save_every == 0:
                torch.save(
                    {"model": model.state_dict(), "max_length": args.max_length, "epoch": epoch},
                    args.save,
                )
                print(f"  Saved to {args.save}", file=sys.stderr)
    except KeyboardInterrupt:
        print("\nStopped by user. Saving final checkpoint...", file=sys.stderr)
        torch.save(
            {"model": model.state_dict(), "max_length": args.max_length},
            args.save,
        )
        print(f"Saved to {args.save}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

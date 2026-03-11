#!/usr/bin/env python3
"""
Train the discrete diffusion denoising model.

Usage:
    python train.py --data data/expressions.txt --max-length 50
    python train.py --data data/expressions.txt --epochs 10 --batch-size 64
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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


def create_dataloader(
    exprs: list[str],
    max_length: int,
    batch_size: int,
    shuffle: bool = True,
    rng=None,
):
    """Yield batches of (clean_ids, corrupted_ids, t, pad_mask)."""
    r = rng or random
    indices = list(range(len(exprs)))
    if shuffle:
        r.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_exprs = [exprs[j] for j in batch_idx]
        clean = encode_batch(batch_exprs, max_length)
        t = sample_timestep(rng=r)
        corrupted = corrupt_batch(clean, t, rng=r)
        pad_mask = [[ids[i] == PAD_ID for i in range(len(ids))] for ids in clean]
        yield (
            torch.tensor(clean, dtype=torch.long),
            torch.tensor(corrupted, dtype=torch.long),
            torch.full((len(batch_exprs),), t, dtype=torch.long),
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
) -> float:
    """One training step. Returns loss."""
    model.train()
    clean = clean.to(device)
    corrupted = corrupted.to(device)
    t = t.to(device)
    pad_mask = pad_mask.to(device)

    logits = model(corrupted, t, pad_mask=pad_mask)
    # Cross-entropy, ignore padding
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        clean.view(-1),
        ignore_index=PAD_ID,
        reduction="mean",
    )
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train discrete diffusion model")
    parser.add_argument("--data", type=Path, default=Path("data/expressions.txt"))
    parser.add_argument("--max-length", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-exprs", type=int, default=None, help="Limit dataset size")
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
    model = DenoisingTransformer(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_len=args.max_length,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for clean, corrupted, t, pad_mask in create_dataloader(
            exprs, args.max_length, args.batch_size, rng=rng
        ):
            loss = train_step(model, clean, corrupted, t, pad_mask, optimizer, device)
            total_loss += loss
            n_batches += 1
        avg = total_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}", file=sys.stderr)

    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "max_length": args.max_length}, args.save)
    print(f"Saved to {args.save}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

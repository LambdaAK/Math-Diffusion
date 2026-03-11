#!/usr/bin/env python3
"""
Watch an expression get corrupted across diffusion timesteps.

Usage:
    python watch_corruption.py "(x+2)*y"
    python watch_corruption.py "sin(x)+3" --steps 20
    python watch_corruption.py  # uses random expression from grammar
"""

import argparse
import random
import sys

from tokenizer import encode, decode
from diffusion import corrupt, NUM_TIMESTEPS


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch diffusion corruption over time")
    parser.add_argument("expr", nargs="?", default=None, help="Expression to corrupt (default: random from grammar)")
    parser.add_argument("--steps", type=int, default=NUM_TIMESTEPS, help="Number of timesteps")
    parser.add_argument("--samples", type=int, default=11, help="Number of snapshots to show (0–steps)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    if args.expr:
        expr = args.expr
    else:
        from grammar import generate_expr
        expr = generate_expr(max_length=30, rng=rng)
        print(f"Using random expression: {expr}\n", file=sys.stderr)

    ids = encode(expr)
    max_len = len(expr)

    # Sample timesteps to display
    if args.samples <= 0:
        timesteps = range(args.steps + 1)
    else:
        timesteps = [int(i * args.steps / (args.samples - 1)) for i in range(args.samples)]
        timesteps = sorted(set(min(t, args.steps) for t in timesteps))

    print(f"{'t':>4}  {'gamma':>6}  expression")
    print("-" * (4 + 2 + 6 + 2 + max(max_len, 20)))

    for t in timesteps:
        corrupted = corrupt(ids, t, T=args.steps, rng=rng)
        corrupted_str = decode(corrupted)
        gamma = (t / args.steps) if t <= args.steps else 1.0
        print(f"{t:>4}  {gamma:>6.2f}  {corrupted_str}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

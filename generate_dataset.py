#!/usr/bin/env python3
"""
Generate a dataset of mathematical expressions from the formal grammar.

Usage:
    python generate_dataset.py --n 10000 --output data/expressions.txt
    python generate_dataset.py --n 1000 --seed 42 --max-depth 4
"""

import argparse
import sys
from pathlib import Path

from grammar import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic mathematical expressions from formal grammar."
    )
    parser.add_argument(
        "-n", "--num",
        type=int,
        default=1000,
        help="Number of expressions to generate (default: 1000)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum nesting depth (default: 3)",
    )
    parser.add_argument(
        "--max-add-terms",
        type=int,
        default=2,
        help="Max +/- terms per expression (default: 2)",
    )
    parser.add_argument(
        "--max-mul-factors",
        type=int,
        default=2,
        help="Max */ factors per term (default: 2)",
    )
    parser.add_argument(
        "--number-max-digits",
        type=int,
        default=1,
        help="Max digits in generated numbers (default: 1)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum expression length in characters (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Allow duplicate expressions",
    )

    args = parser.parse_args()

    expressions = generate_dataset(
        n=args.num,
        max_depth=args.max_depth,
        max_add_terms=args.max_add_terms,
        max_mul_factors=args.max_mul_factors,
        number_max_digits=args.number_max_digits,
        max_length=args.max_length,
        seed=args.seed,
        deduplicate=not args.no_deduplicate,
    )

    out = args.output
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(expressions) + "\n", encoding="utf-8")
        print(f"Wrote {len(expressions)} expressions to {out}", file=sys.stderr)
    else:
        for e in expressions:
            print(e)

    return 0


if __name__ == "__main__":
    sys.exit(main())

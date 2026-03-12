# Math-Diffusion

A discrete diffusion model for generating mathematical expressions. Uses a character-level tokenizer, formal grammar for valid expressions, and a transformer-based denoising model.

## Features

- **Grammar-constrained expressions**: Variables (`x`, `y`, `z`), operators (`+`, `-`, `*`, `/`, `^`), functions (`sin`, `cos`, `tan`, `log`, `sqrt`, `exp`)
- **Clean formatting**: Redundant parentheses removed (e.g. `(y)` → `y`), spaces around operators
- **Stochastic sampling** with temperature control
- **GPU-optimized training**: DataLoader workers, mixed precision (AMP), cosine LR schedule

## Setup

```bash
pip install torch>=2.0.0
```

## Quick Start

### 1. Generate training data

```bash
python generate_dataset.py -n 500000 -o data/expressions.txt --max-length 100 --seed 42
```

### 2. Train

```bash
# Large model (512d, 12 layers) - recommended for A100
python train.py \
  --data data/expressions.txt \
  --max-length 100 \
  --batch-size 64 \
  --d-model 512 \
  --n-heads 8 \
  --n-layers 12 \
  --num-workers 8 \
  --save-every 5 \
  --save checkpoints/model_large.pt

# Small model (128d, 4 layers) - faster, less capacity
python train.py \
  --data data/expressions.txt \
  --max-length 100 \
  --batch-size 512 \
  --num-workers 8 \
  --save-every 5 \
  --save checkpoints/model.pt
```

Training runs indefinitely until Ctrl+C. Checkpoints are saved every `--save-every` epochs and on interrupt.

### 3. Sample

```bash
# Stochastic (default temperature 1.0)
python sample.py --checkpoint checkpoints/model_large.pt --num 20

# More conservative
python sample.py --checkpoint checkpoints/model_large.pt --num 20 --temperature 0.5

# Deterministic
python sample.py --checkpoint checkpoints/model_large.pt --num 20 --temperature 0
```

## Project Structure

| File | Description |
|------|-------------|
| `grammar.py` | Formal grammar, expression generation, redundant paren removal, spacing |
| `tokenizer.py` | Character-level tokenizer (vocab: operators, digits, variables, functions, space, [PAD], [MASK]) |
| `diffusion.py` | Corruption process (linear/cosine schedule, MASK + random replacement) |
| `model.py` | DenoisingTransformer (encoder-only, timestep embedding) |
| `train.py` | Training loop (DataLoader, AMP, cosine warm restarts) |
| `sample.py` | Iterative denoising from all-MASK |
| `generate_dataset.py` | CLI for generating expression datasets |
| `watch_corruption.py` | Visualize corruption across timesteps |

## Example Outputs

After training, the model generates expressions like:

```
x + y ^ 2
sin(x) + 3
sqrt(x * x + 1) / (y - 4)
log(x) + exp(y ^ 2)
```

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--max-length` | 100 | Max sequence length (chars) |
| `--batch-size` | 256 | Batch size (reduce for larger models) |
| `--d-model` | 512 | Transformer dimension |
| `--n-heads` | 8 | Attention heads |
| `--n-layers` | 12 | Transformer layers |
| `--lr` | 1e-4 | Learning rate |
| `--lr-min` | 1e-6 | Min LR for cosine scheduler |
| `--save-every` | 5 | Checkpoint interval (epochs) |

## Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

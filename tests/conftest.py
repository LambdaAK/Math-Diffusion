import sys
from pathlib import Path

# Add project root to path so tests can import tokenizer, grammar, etc.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

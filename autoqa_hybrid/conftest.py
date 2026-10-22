"""Pytest configuration for the autoqa_hybrid project."""
import sys
from pathlib import Path

HYBRID_ROOT = Path(__file__).resolve().parent
if str(HYBRID_ROOT) not in sys.path:
    sys.path.insert(0, str(HYBRID_ROOT))

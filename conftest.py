"""Root conftest.py — adds the project root to sys.path so tests can import
top-level modules (local_model, storage, etc.) regardless of where pytest is
invoked from.
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

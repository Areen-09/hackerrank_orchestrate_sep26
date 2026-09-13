"""
HackerRank Orchestrate: Buy or Wait?
Top-level entry point that forwards directly to code.main.
"""

import os
import sys

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from code.main import main

if __name__ == "__main__":
    main()

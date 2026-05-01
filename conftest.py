"""
Root-level pytest configuration.

Adds the project root directory to ``sys.path`` so that test files can use
``from src.xxx import yyy`` without needing to install the package or set
PYTHONPATH manually.
"""
import sys
import os

# Insert project root at the front of the module search path
sys.path.insert(0, os.path.dirname(__file__))


"""
Root conftest.py — Ensures repository root is automatically on sys.path
for all pytest and script invocations without requiring PYTHONPATH=.
"""

import os
import sys

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

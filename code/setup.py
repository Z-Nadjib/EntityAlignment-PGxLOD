"""Compatibility shim. Configuration lives in ``pyproject.toml``.

Enables editable installs on older tooling:  ``pip install -e .``
"""
from setuptools import setup

if __name__ == "__main__":
    setup()

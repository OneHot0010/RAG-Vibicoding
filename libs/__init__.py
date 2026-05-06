"""Compatibility package for source-tree imports before installation."""

from __future__ import annotations

from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / __name__
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))

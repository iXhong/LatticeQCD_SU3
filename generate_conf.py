"""Compatibility imports for the lattice_su3 package."""

from pathlib import Path
import sys

src_path = Path(__file__).resolve().parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from lattice_su3 import *  # noqa: E402,F403

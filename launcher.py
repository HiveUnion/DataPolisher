"""PyInstaller entry point — must be a top-level script (no relative imports)."""
import sys
from data_polisher.gui import main

sys.exit(main())

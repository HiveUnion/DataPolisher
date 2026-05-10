"""Allow `python -m data_polisher` to launch the GUI directly."""

from .gui import main


if __name__ == "__main__":
    raise SystemExit(main())

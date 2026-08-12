"""PyQt5 graphical interface for LEAF_FINCH."""


def main() -> int:
    """Launch the GUI without importing PyQt5 until it is actually needed."""
    from .main_window import main as _main

    return _main()


__all__ = ["main"]

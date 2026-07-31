import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

try:
    from leaf_finch.gui.main_window import main
except ModuleNotFoundError as exc:
    if exc.name == "PyQt5":
        raise SystemExit("PyQt5 is not installed. Run: pip install -r requirements.txt") from exc
    raise

if __name__ == "__main__":
    raise SystemExit(main())

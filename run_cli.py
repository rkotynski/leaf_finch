import os

os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from leaf_finch.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

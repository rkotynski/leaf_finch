from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .backend import list_accelerators
from .config import AppConfig
from .propagation import CancelledError
from .runner import run_simulation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LEAF_FINCH binary-DMD optimizer")
    parser.add_argument("--config", type=Path, help="JSON configuration file")
    parser.add_argument("--write-default", type=Path, help="Write a default JSON configuration and exit")
    parser.add_argument("--list-devices", action="store_true", help="List CPU/CUDA/ROCm devices and exit")
    parser.add_argument("--model", type=Path, help="Load a saved .pt optimizer model")
    parser.add_argument(
        "--restart-from-weights",
        action="store_true",
        help="Load model logits but reset Adam state, history, and epoch counter",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        for accelerator in list_accelerators():
            print(accelerator.label)
        return 0
    if args.write_default:
        AppConfig().save_json(args.write_default)
        print(args.write_default)
        return 0
    config = AppConfig.load_json(args.config) if args.config else AppConfig()
    try:
        summary = run_simulation(
            config,
            progress_callback=lambda state: print(
                f"\r{state.get('phase', '')}: {100*state.get('fraction', 0):5.1f}%",
                end="",
                flush=True,
            ),
            log_callback=lambda text: print(f"\n{text}"),
            model_checkpoint=args.model,
            resume_optimizer=not args.restart_from_weights,
        )
    except CancelledError:
        print("\nCancelled", file=sys.stderr)
        return 2
    print(f"\nResults: {summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

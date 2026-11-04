#!/usr/bin/env python3
import argparse
from pathlib import Path

from solvercell.config import load_config, debug_config
from solvercell.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.debug:
        config = debug_config(config)
    result = train(config, Path(args.output), resume=args.resume)
    return 0 if result["status"] == "complete" else 130


if __name__ == "__main__":
    raise SystemExit(main())

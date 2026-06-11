from __future__ import annotations

import argparse
import sys

from mataltmag_hybrid.config import GateError, ensure_parent, load_paths
from mataltmag_hybrid.io.cache import write_table
from mataltmag_hybrid.io.load_mataltmag import build_dataset_index


def prepare(config: str = "configs/paths.yaml") -> None:
    paths = load_paths(config)
    index = build_dataset_index(paths.path("raw_mataltmag_dir"))
    out = paths.path("dataset_index")
    ensure_parent(out)
    write_table(index, out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/paths.yaml")
    args = parser.parse_args()
    try:
        prepare(args.config)
    except GateError as exc:
        print(f"Gate blocked: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()

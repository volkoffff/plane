from __future__ import annotations

import argparse

from viewer.app import run_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance le viewer avion Panda3D."
    )
    parser.add_argument(
        "--animations",
        action="store_true",
        help="Reactive les animations de l'avion pour comparer les performances.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_viewer(
        animations_enabled=args.animations,
    )

from __future__ import annotations

import argparse

from viewer.app import ViewerMode, run_viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance le viewer avion Panda3D."
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ViewerMode],
        default=ViewerMode.INSTRUCTIONS.value,
        help="Mode de pilotage au lancement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_viewer(ViewerMode(args.mode))

from __future__ import annotations

import argparse

from viewer.app2 import PandaSceneViewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance le viewer avion Panda3D."
    )
    parser.add_argument(
        "--animations", action=argparse.BooleanOptionalAction, default=True,
        help="Active les animations (actives par defaut).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = PandaSceneViewer(animations_enabled=args.animations)
    app.run()

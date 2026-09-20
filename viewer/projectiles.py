from __future__ import annotations

import numpy as np
from panda3d.core import LineSegs, NodePath, TransparencyAttrib

from physics.projectiles import Bullet
from viewer.transforms import ned_to_panda


class ProjectileRenderer:
    def __init__(self, render: NodePath) -> None:
        self.render = render
        self.node: NodePath | None = None

    def clear(self) -> None:
        if self.node is not None:
            self.node.removeNode()
            self.node = None

    def update(self, bullets: list[Bullet]) -> None:
        self.clear()

        if not bullets:
            return

        lines = LineSegs("bullets")
        lines.setThickness(3.5)

        for bullet in bullets:
            age_ratio = np.clip(bullet.age / bullet.lifetime, 0.0, 1.0)
            alpha = 1.0 - age_ratio
            lines.setColor(1.0, 0.82, 0.18, alpha)
            lines.moveTo(*ned_to_panda(bullet.previous_position))
            lines.drawTo(*ned_to_panda(bullet.position))

        self.node = self.render.attachNewNode(lines.create())
        self.node.setTransparency(TransparencyAttrib.MAlpha)

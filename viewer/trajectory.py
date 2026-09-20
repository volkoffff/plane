from __future__ import annotations

import numpy as np
from panda3d.core import LineSegs, NodePath


class TrajectoryRenderer:
    def __init__(
        self,
        render: NodePath,
        initial_position: np.ndarray,
        sample_interval: float = 0.08,
    ) -> None:
        self.render = render
        self.sample_interval = sample_interval
        self.points: list[np.ndarray] = [initial_position.copy()]
        self.node: NodePath | None = None
        self.last_sample_time = 0.0

    def update(self, position: np.ndarray, elapsed_time: float) -> None:
        if elapsed_time - self.last_sample_time < self.sample_interval:
            return

        self.last_sample_time = elapsed_time
        self.points.append(position.copy())

        if len(self.points) < 2:
            return

        if self.node is not None:
            self.node.removeNode()

        lines = LineSegs("trajectory")
        lines.setColor(1.0, 0.55, 0.05, 1.0)
        lines.setThickness(2.5)

        first = self.points[0]
        lines.moveTo(*first)
        for point in self.points[1:]:
            lines.drawTo(*point)

        self.node = self.render.attachNewNode(lines.create())

from __future__ import annotations

import numpy as np
from panda3d.core import LPoint3f, NodePath


class ChaseCamera:
    def __init__(self, camera: NodePath) -> None:
        self.camera = camera

    def update(
        self,
        aircraft_position: np.ndarray,
        aircraft_rotation: np.ndarray,
    ) -> None:
        if self.camera is None or self.camera.isEmpty():
            return

        forward = aircraft_rotation @ np.array([0.0, 1.0, 0.0])
        world_up = np.array([0.0, 0.0, 1.0])

        level_forward = forward - np.dot(forward, world_up) * world_up
        if np.linalg.norm(level_forward) < 1e-6:
            level_forward = np.array([0.0, 1.0, 0.0])
        else:
            level_forward = level_forward / np.linalg.norm(level_forward)

        right = np.cross(level_forward, world_up)
        camera_position = (
            aircraft_position
            - 60.0 * forward
            - 0.0 * right
            + 18.0 * world_up
        )
        focal_point = aircraft_position + 70.0 * forward

        self.camera.setPos(*camera_position)
        self.camera.lookAt(LPoint3f(*focal_point))

from __future__ import annotations

import numpy as np
from panda3d.core import Filename, LPoint3f, NodePath

from model.aircraft import create_aircraft
from model.aircraft_animations import AircraftAnimationController
from physics.controls import ControlInputs
from physics.state import AircraftState
from viewer.transforms import (
    MODEL_PANDA_TO_BODY_PANDA,
    aircraft_panda_rotation,
    ned_to_panda,
    panda_matrix_from_rotation_translation,
)


class AircraftVisual:
    def __init__(
        self,
        loader,
        render: NodePath,
        animations_enabled: bool = False,
    ) -> None:
        self.loader = loader
        self.render = render
        self.animations_enabled = animations_enabled
        self.root: NodePath | None = None
        self.animations: AircraftAnimationController | None = None

    def setup(self) -> None:
        try:
            import gltf  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Le chargeur GLB Panda3D est manquant. Installe panda3d-gltf."
            ) from exc

        model_path = Filename.fromOsSpecific(str(create_aircraft()))
        aircraft_model = self.loader.loadModel(model_path)

        if aircraft_model.isEmpty():
            raise RuntimeError(f"Modele 3D introuvable : {model_path}")

        self.root = self.render.attachNewNode("aircraft-root")
        aircraft_visual = self.root.attachNewNode("aircraft-visual")
        aircraft_visual.setMat(
            panda_matrix_from_rotation_translation(
                MODEL_PANDA_TO_BODY_PANDA,
                np.zeros(3),
            )
        )

        aircraft_model.reparentTo(aircraft_visual)
        aircraft_model.setPos(0.0, 0.0, 0.0)

        bounds = aircraft_model.getTightBounds(aircraft_visual)
        if bounds is None:
            model_center = LPoint3f(0.0, 0.0, 0.0)
        else:
            model_center = (bounds[0] + bounds[1]) * 0.5

        aircraft_model.setPos(
            -model_center.x,
            -model_center.y,
            -model_center.z,
        )
        self.animations = AircraftAnimationController(
            aircraft_model,
            self.root,
            aircraft_visual.getTightBounds(self.root),
        )

    def update_pose(self, state: AircraftState) -> tuple[np.ndarray, np.ndarray]:
        if self.root is None:
            raise RuntimeError("AircraftVisual.setup() doit etre appele avant update_pose().")

        position = ned_to_panda(state.position)
        rotation = aircraft_panda_rotation(state)
        self.root.setMat(panda_matrix_from_rotation_translation(rotation, position))

        return position, rotation

    def update_animations(
        self,
        controls: ControlInputs | None,
        throttle: float,
        speed: float,
        elapsed_time: float,
    ) -> None:
        if not self.animations_enabled:
            return

        if self.animations is None:
            return

        self.animations.update(controls, throttle, speed, elapsed_time)

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from simulation.aircraft import AircraftEntity
from viewer.aircraft_visual import AircraftVisual

AircraftPose = tuple[np.ndarray, np.ndarray]


class AircraftVisualRegistry:
    def __init__(
        self,
        loader,
        render,
        animations_enabled: bool = False,
    ) -> None:
        self.loader = loader
        self.render = render
        self.animations_enabled = animations_enabled
        self.visuals: dict[int, AircraftVisual] = {}

    def setup(self, aircraft_list: Iterable[AircraftEntity]) -> None:
        for aircraft in aircraft_list:
            visual = AircraftVisual(
                self.loader,
                self.render,
                animations_enabled=self.animations_enabled,
            )
            visual.setup()
            self.visuals[aircraft.id] = visual

    def update(
        self,
        aircraft_list: Iterable[AircraftEntity],
        animation_time: float | None = None,
    ) -> dict[int, AircraftPose]:
        poses = {}
        for aircraft in aircraft_list:
            visual = self.visuals[aircraft.id]
            poses[aircraft.id] = visual.update_pose(aircraft.physics.state)
            self._update_animations(visual, aircraft, animation_time)

        return poses

    def set_animations_enabled(self, enabled: bool) -> None:
        self.animations_enabled = enabled
        for visual in self.visuals.values():
            visual.set_animations_enabled(enabled)

    def _update_animations(
        self,
        visual: AircraftVisual,
        aircraft: AircraftEntity,
        animation_time: float | None,
    ) -> None:
        physics = aircraft.physics
        elapsed_time = physics.elapsed_time
        if elapsed_time <= 0.0 and animation_time is not None:
            elapsed_time = animation_time

        visual.update_animations(
            controls=physics.last_controls,
            throttle=physics.state.throttle,
            speed=float(physics.air_data.get("speed", 0.0)),
            elapsed_time=elapsed_time,
        )

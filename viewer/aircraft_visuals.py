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
        """Keep exactly one visual per living aircraft, removing stale nodes."""
        living_aircraft = {
            aircraft.id: aircraft for aircraft in aircraft_list if aircraft.alive
        }
        for aircraft_id in self.visuals.keys() - living_aircraft.keys():
            self.visuals.pop(aircraft_id).destroy()

        for aircraft_id in living_aircraft:
            if aircraft_id in self.visuals:
                continue
            visual = AircraftVisual(
                self.loader,
                self.render,
                animations_enabled=self.animations_enabled,
            )
            visual.setup()
            self.visuals[aircraft_id] = visual

    def update(
        self,
        aircraft_list: Iterable[AircraftEntity],
        animation_time: float | None = None,
    ) -> dict[int, AircraftPose]:
        aircraft_list = tuple(aircraft_list)
        self.setup(aircraft_list)
        poses = {}
        for aircraft in aircraft_list:
            if not aircraft.alive:
                continue
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

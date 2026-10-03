from dataclasses import dataclass, field
from math import isfinite
from itertools import count
from typing import ClassVar

from physics.aircraft_physics import AircraftPhysics
from simulation.teams import Team
from simulation.weapons import Gun


@dataclass
class AircraftEntity:
    _id_counter: ClassVar[int] = count(1)

    id: int = field(init=False)
    team: Team
    physics: AircraftPhysics
    weapon: Gun = field(default_factory=Gun)
    health: float = 100
    alive: bool = True
    # Approximate body-frame half-length, half-span and half-height, in metres.
    # Independent of the GLB; configure these dimensions for each aircraft type.
    collision_half_extents: tuple[float, float, float] = (9.75, 5.45, 2.0)

    def __post_init__(self):
        if len(self.collision_half_extents) != 3 or any(
            not isfinite(value) or value <= 0.0 for value in self.collision_half_extents
        ):
            raise ValueError(
                "Collision half-extents must be three positive finite lengths"
            )
        self.id = next(self._id_counter)

    def death(self):
        self.alive = False

    def is_dead(self):
        return not self.alive

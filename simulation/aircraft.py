from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import count
from math import isfinite
from typing import ClassVar

from physics.aircraft_physics import AircraftPhysics
from simulation.teams import Team
from simulation.weapons import Gun


@dataclass
class AircraftEntity:
    _id_counter: ClassVar[Iterator[int]] = count(1)

    id: int = field(init=False)
    team: Team
    physics: AircraftPhysics
    weapon: Gun = field(default_factory=Gun)
    health: float = 100.0
    # Approximate body-frame half-length, half-span and half-height, in metres.
    # Independent of the GLB; configure these dimensions for each aircraft type.
    collision_half_extents: tuple[float, float, float] = (9.75, 5.45, 2.0)

    def __post_init__(self) -> None:
        if not isfinite(self.health) or self.health < 0.0:
            raise ValueError("Health must be finite and nonnegative")
        if len(self.collision_half_extents) != 3 or any(
            not isfinite(value) or value <= 0.0 for value in self.collision_half_extents
        ):
            raise ValueError(
                "Collision half-extents must be three positive finite lengths"
            )
        self.id = next(self._id_counter)

    @property
    def alive(self) -> bool:
        """Health is the single source of truth for the aircraft's life state."""
        return self.health > 0.0

    def take_damage(self, damage: float) -> float:
        """Apply damage and return health actually lost, including overkill."""
        if not isfinite(damage) or damage < 0.0:
            raise ValueError("Damage must be finite and nonnegative")
        previous_health = self.health
        self.health = max(0.0, self.health - damage)
        return previous_health - self.health

    def death(self) -> None:
        self.health = 0.0

    def is_dead(self) -> bool:
        return not self.alive

from dataclasses import dataclass, field
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

    def __post_init__(self):
        self.id = next(self._id_counter)

    def death(self):
        self.alive = False

    def is_dead(self):
        return not self.alive

from dataclasses import dataclass, field
from math import isfinite

from physics.projectiles import Bullet
from piloting.commands import AircraftAction
from simulation.aircraft import AircraftEntity
from simulation.combat import CombatSystem
from simulation.weapons import WeaponSystem


@dataclass
class SimulationWorld:
    aircraft: dict[int, AircraftEntity] = field(default_factory=dict)
    combat: CombatSystem = field(default_factory=CombatSystem)
    weapons: WeaponSystem = field(default_factory=WeaponSystem)
    bullets: list[Bullet] = field(default_factory=list)

    def add_aircraft(self, entity: AircraftEntity) -> None:
        self.aircraft[entity.id] = entity

    def step(self, actions: dict[int, AircraftAction], dt: float) -> None:
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("World step must be positive and finite")
        previous_positions = {
            entity.id: entity.physics.state.position.copy()
            for entity in self.aircraft.values()
            if not entity.is_dead()
        }
        # Spawn at the start-of-step pose, then advance bullets and aircraft over
        # the same interval before resolving their relative trajectories.
        self.weapons.update(self, actions, dt)
        for aircraft_id, entity in self.aircraft.items():
            if entity.is_dead():
                continue

            action = actions.get(aircraft_id)
            if action is None:
                continue

            entity.physics.step(action.flight, dt)

        self.combat.update(self, dt, previous_positions=previous_positions)

    def get_all_aircrafts(self) -> tuple[AircraftEntity, ...]:
        return tuple(self.aircraft.values())

from dataclasses import dataclass, field

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
        for aircraft_id, entity in self.aircraft.items():
            if entity.is_dead():
                continue

            action = actions.get(aircraft_id)
            if action is None:
                continue

            entity.physics.step(action.flight)

        self.weapons.update(self, actions, dt)
        self.combat.update(self, dt)

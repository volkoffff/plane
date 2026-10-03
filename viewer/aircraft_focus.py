from __future__ import annotations

from simulation.aircraft import AircraftEntity
from simulation.world import SimulationWorld


class AircraftFocusController:
    def __init__(self, world: SimulationWorld) -> None:
        self.world = world
        aircraft_ids = self.aircraft_ids
        self.current_aircraft_id = aircraft_ids[0] if aircraft_ids else None

    @property
    def aircraft_ids(self) -> list[int]:
        return [aircraft.id for aircraft in self.world.get_all_aircrafts()]

    @property
    def current_aircraft(self) -> AircraftEntity | None:
        if self.current_aircraft_id is None:
            return None

        return self.world.aircraft.get(self.current_aircraft_id)

    def previous(self) -> None:
        self._move(-1)

    def next(self) -> None:
        self._move(1)

    def _move(self, offset: int) -> None:
        aircraft_ids = self.aircraft_ids
        if not aircraft_ids:
            self.current_aircraft_id = None
            return

        if self.current_aircraft_id not in aircraft_ids:
            self.current_aircraft_id = aircraft_ids[0]
            return

        current_index = aircraft_ids.index(self.current_aircraft_id)
        self.current_aircraft_id = aircraft_ids[
            (current_index + offset) % len(aircraft_ids)
        ]

from __future__ import annotations

from physics.aircraft_physics import AircraftPhysics
from simulation.aircraft import AircraftEntity
from simulation.teams import Team
from simulation.world import SimulationWorld


def create_player_world(
    team: Team = Team.ALPHA,
) -> tuple[SimulationWorld, AircraftEntity]:
    world = SimulationWorld()
    player_aircraft = AircraftEntity(
        team=team,
        physics=AircraftPhysics(),
    )
    world.add_aircraft(player_aircraft)

    return world, player_aircraft

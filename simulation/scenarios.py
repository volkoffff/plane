from __future__ import annotations

import numpy as np

from physics.aircraft_physics import AircraftPhysics
from physics.math3d import euler_to_quaternion
from physics.state import AircraftState, initial_state
from simulation.aircraft import AircraftEntity
from simulation.teams import Team
from simulation.world import SimulationWorld


def create_aircraft_state(
    position: tuple[float, float, float],
    yaw_deg: float = 0.0,
    speed: float = 180.0,
) -> AircraftState:
    state = initial_state()
    yaw = np.deg2rad(yaw_deg)

    state.position = np.array(position, dtype=float)
    state.velocity = np.array(
        [
            speed * np.cos(yaw),
            speed * np.sin(yaw),
            0.0,
        ],
        dtype=float,
    )
    state.quaternion = euler_to_quaternion(
        roll=0.0,
        pitch=np.deg2rad(3.0),
        yaw=yaw,
    )

    return state


def create_aircraft_in_world(
    world: SimulationWorld,
    team: Team = Team.ALPHA,
    state: AircraftState | None = None,
) -> AircraftEntity:
    aircraft = AircraftEntity(
        team=team,
        physics=AircraftPhysics(
            state=state if state is not None else initial_state(),
        ),
    )
    world.add_aircraft(aircraft)
    return aircraft


def create_1_vs_1_world(
    world: SimulationWorld
) -> tuple[AircraftEntity, AircraftEntity]:
    player_a = create_aircraft_in_world(
        world,
        Team.ALPHA,
        create_aircraft_state(
            position=(0.0, -90.0, -1_000.0),
            yaw_deg=0.0,
        ),
    )
    player_b = create_aircraft_in_world(
        world,
        Team.BRAVO,
        create_aircraft_state(
            position=(320.0, 90.0, -1_000.0),
            yaw_deg=180.0,
        ),
    )

    return player_a, player_b

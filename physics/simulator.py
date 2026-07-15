from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physics.autopilot import FlightCommand
from physics.controls import flight_control_law_from_mouse
from physics.dynamics import integrate_aircraft
from physics.parameters import AircraftParameters, rafale_like_parameters
from physics.state import AircraftState, initial_state


@dataclass
class AircraftSimulation:
    state: AircraftState = field(default_factory=initial_state)
    params: AircraftParameters = field(default_factory=rafale_like_parameters)
    dt: float = 0.01
    elapsed_time: float = 0.0
    air_data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.air_data:
            self.air_data = {
                "speed": float(np.linalg.norm(self.state.velocity)),
                "alpha": 0.0,
                "beta": 0.0,
            }

    def step(self, command: FlightCommand) -> dict:
        controls = flight_control_law_from_mouse(
            self.state,
            mouse_dx=command.mouse_dx,
            mouse_dy=command.mouse_dy,
            rudder_input=command.rudder_input,
            throttle_command=command.throttle_command,
            beta=float(self.air_data.get("beta", 0.0)),
            target_yaw_rate=command.target_yaw_rate,
        )

        self.air_data = integrate_aircraft(
            self.state,
            self.params,
            controls,
            self.dt,
        )
        self.elapsed_time += self.dt

        return self.air_data

    def reset(self) -> None:
        self.state = initial_state()
        self.elapsed_time = 0.0
        self.air_data = {
            "speed": float(np.linalg.norm(self.state.velocity)),
            "alpha": 0.0,
            "beta": 0.0,
        }

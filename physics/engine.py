import numpy as np

from physics.atmosphere import air_density
from physics.parameters import AircraftParameters
from physics.state import AircraftState


def update_throttle(
    state: AircraftState,
    params: AircraftParameters,
    throttle_command: float,
    dt: float,
) -> None:
    throttle_command = np.clip(throttle_command, 0, 1.5)

    error = throttle_command - state.throttle

    state.throttle += params.throttle_response * error * dt
    state.throttle = float(np.clip(state.throttle, 0.0, 1.5))


def thrust_altitude_factor(altitude_m: float) -> float:
    rho = air_density(altitude_m)
    rho0 = 1.225
    factor = rho / rho0

    return float(np.clip(factor, 0.3, 1.0))


def thrust_magnitude(
    state,
    params,
) -> float:
    throttle = state.throttle

    if throttle <= 1.0:
        thrust = throttle * params.max_dry_thrust
    else:
        afterburner_ratio = (throttle - 1.0) / 0.5

        thrust = (
            params.max_dry_thrust
            + afterburner_ratio
            * (
                params.max_afterburner_thrust
                - params.max_dry_thrust
            )
        )

    altitude = -state.position[2]
    thrust *= thrust_altitude_factor(altitude)

    return float(thrust)

def thrust_force(
    state: AircraftState,
    params: AircraftParameters,
) -> np.ndarray:
    thrust = thrust_magnitude(state, params)
    direction = np.array([
        np.cos(state.pitch),
        0.0,
        np.sin(state.pitch),
    ])

    return thrust * direction

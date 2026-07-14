import numpy as np

from physics.engine import update_throttle
from physics.math3d import (
    quaternion_derivative,
    normalize_quaternion,
)
from physics.aerodynamics import total_forces_and_moments


def integrate_aircraft(
    state,
    params,
    controls,
    dt: float,
    wind_world: np.ndarray | None = None,
):
    update_throttle(
        state,
        params,
        controls.throttle_command,
        dt,
    )

    force_world, moment_body, air_data = total_forces_and_moments(
        state,
        params,
        controls,
        wind_world,
    )

    # Translation
    acceleration_world = force_world / params.mass

    state.velocity = state.velocity + acceleration_world * dt
    state.position = state.position + state.velocity * dt

    # Rotation
    I_omega = params.inertia_body @ state.omega_body

    omega_dot = np.linalg.solve(
        params.inertia_body,
        moment_body - np.cross(state.omega_body, I_omega),
    )

    state.omega_body = state.omega_body + omega_dot * dt

    q_dot = quaternion_derivative(
        state.quaternion,
        state.omega_body,
    )

    state.quaternion = state.quaternion + q_dot * dt
    state.quaternion = normalize_quaternion(state.quaternion)

    return air_data
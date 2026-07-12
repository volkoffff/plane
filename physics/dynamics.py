import numpy as np

from physics.aerodynamics import angle_of_attack_2d, compute_forces
from physics.atmosphere import air_density
from physics.controls import elevator_deflection
from physics.engine import update_throttle
from physics.parameters import AircraftParameters
from physics.state import AircraftState


def compute_pitch_moment(
    state: AircraftState,
    params: AircraftParameters,
    elevator_control: float,
) -> float:
    altitude = max(state.position[2], 0.0)
    rho = air_density(altitude)

    speed = np.linalg.norm(state.velocity)

    if speed < 1e-6:
        return 0.0

    alpha = angle_of_attack_2d(state)
    delta_e = elevator_deflection(elevator_control)
    dynamic_pressure = 0.5 * rho * speed**2
    pitch_rate_term = state.pitch_rate * params.mean_chord / (2.0 * speed)

    cm = (
        params.cm0
        + params.cm_alpha * alpha
        + params.cm_q * pitch_rate_term
        + params.cm_delta_e * delta_e
    )

    return float(
        dynamic_pressure
        * params.wing_area
        * params.mean_chord
        * cm
    )


def update_state(
    state: AircraftState,
    params: AircraftParameters,
    elevator_control: float,
    throttle_command: float,
    dt: float,
) -> AircraftState:
    update_throttle(state, params, throttle_command, dt)

    force = compute_forces(state, params)
    pitch_moment = compute_pitch_moment(
        state,
        params,
        elevator_control,
    )

    acceleration = force / params.mass
    state.velocity = state.velocity + acceleration * dt
    state.position = state.position + state.velocity * dt

    pitch_acceleration = pitch_moment / params.pitch_inertia
    state.pitch_rate = state.pitch_rate + pitch_acceleration * dt
    state.pitch = state.pitch + state.pitch_rate * dt

    return state

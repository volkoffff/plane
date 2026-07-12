import numpy as np

from physics.atmosphere import air_density
from physics.engine import thrust_force
from physics.parameters import AircraftParameters
from physics.state import AircraftState


def angle_of_attack_2d(state: AircraftState) -> float:
    vx = state.velocity[0]
    vz = state.velocity[2]

    flight_path_angle = np.arctan2(vz, vx)
    return float(state.pitch - flight_path_angle)


def lift_coefficient_from_alpha(alpha_rad: float) -> float:
    cl0 = 0.05
    cl_alpha = 4.5

    alpha_stall = np.deg2rad(30.0)
    alpha_abs = abs(alpha_rad)

    cl_linear = cl0 + cl_alpha * alpha_rad

    cl_max = 1.4
    cl_min = -0.8

    if alpha_abs <= alpha_stall:
        return float(np.clip(cl_linear, cl_min, cl_max))

    excess = alpha_abs - alpha_stall
    stall_decay = np.exp(-3.0 * excess)
    cl_stalled = np.clip(cl_linear, cl_min, cl_max) * stall_decay

    return float(cl_stalled)


def lift_direction_2d(state: AircraftState) -> np.ndarray:
    vx = state.velocity[0]
    vz = state.velocity[2]

    speed_xz = np.sqrt(vx**2 + vz**2)

    if speed_xz < 1e-6:
        return np.array([0.0, 0.0, 1.0])

    return np.array([
        -vz / speed_xz,
        0.0,
        vx / speed_xz,
    ])


def compute_forces(
    state: AircraftState,
    params: AircraftParameters,
) -> np.ndarray:
    g = 9.81

    altitude = max(state.position[2], 0.0)
    rho = air_density(altitude)

    velocity = state.velocity
    speed = np.linalg.norm(velocity)

    gravity_force = np.array([
        0.0,
        0.0,
        -params.mass * g,
    ])

    if speed < 1e-6:
        return gravity_force

    alpha = angle_of_attack_2d(state)
    cl = lift_coefficient_from_alpha(alpha)
    dynamic_pressure = 0.5 * rho * speed**2

    lift_magnitude = dynamic_pressure * params.wing_area * cl
    max_lift = params.max_load_factor * params.mass * g
    lift_magnitude = np.clip(
        lift_magnitude,
        -3.2 * params.mass * g,
        max_lift,
    )

    lift_force = lift_direction_2d(state) * lift_magnitude

    cd = params.drag_coefficient_zero + params.induced_drag_factor * cl**2
    drag_magnitude = dynamic_pressure * params.wing_area * cd
    drag_direction = -velocity / speed
    drag_force = drag_direction * drag_magnitude

    return gravity_force + lift_force + drag_force + thrust_force(state, params)

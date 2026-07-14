import numpy as np

from physics.atmosphere import air_density
from physics.engine import thrust_force
from physics.math3d import quaternion_to_matrix
from physics.engine import thrust_magnitude
from physics.controls import normalized_to_surface_deflections
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


def compute_air_data(
    state,
    wind_world: np.ndarray | None = None,
):
    if wind_world is None:
        wind_world = np.zeros(3)

    R_body_to_world = quaternion_to_matrix(state.quaternion)

    velocity_relative_world = state.velocity - wind_world

    velocity_body = R_body_to_world.T @ velocity_relative_world

    u = velocity_body[0]  # avant
    v = velocity_body[1]  # droite
    w = velocity_body[2]  # bas

    speed = np.linalg.norm(velocity_body)

    if speed < 1e-6:
        alpha = 0.0
        beta = 0.0
    else:
        alpha = np.arctan2(w, u)
        beta = np.arcsin(np.clip(v / speed, -1.0, 1.0))

    altitude = -state.position[2]
    rho = air_density(altitude)
    qbar = 0.5 * rho * speed**2

    return {
        "R_body_to_world": R_body_to_world,
        "velocity_body": velocity_body,
        "speed": speed,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "qbar": qbar,
        "altitude": altitude,
    }


def aerodynamic_coefficients(
    state,
    params,
    controls,
    air_data,
):
    alpha = air_data["alpha"]
    beta = air_data["beta"]
    speed = air_data["speed"]

    p, q, r = state.omega_body

    delta_e, delta_a, delta_r = normalized_to_surface_deflections(
        controls,
        params,
    )

    if speed < 1e-6:
        p_hat = 0.0
        q_hat = 0.0
        r_hat = 0.0
    else:
        p_hat = p * params.span / (2.0 * speed)
        q_hat = q * params.mean_chord / (2.0 * speed)
        r_hat = r * params.span / (2.0 * speed)

    # Portance
    CL = (
        params.cl0
        + params.cl_alpha * alpha
        + params.cl_q * q_hat
        + params.cl_delta_e * delta_e
    )

    CL = float(np.clip(CL, params.cl_min, params.cl_max))

    # Traînée
    CD = (
        params.cd0
        + params.induced_drag_factor * CL**2
        + 0.02 * beta**2
    )

    # Force latérale
    CY = (
        params.cy_beta * beta
        + params.cy_delta_r * delta_r
    )

    # Moment de roulis
    Cl = (
        params.roll_beta * beta
        + params.roll_p * p_hat
        + params.roll_r * r_hat
        + params.roll_delta_a * delta_a
    )

    # Moment de tangage
    Cm = (
        params.cm0
        + params.cm_alpha * alpha
        + params.cm_q * q_hat
        + params.cm_delta_e * delta_e
    )

    # Moment de lacet
    Cn = (
        params.cn_beta * beta
        + params.cn_p * p_hat
        + params.cn_r * r_hat
        + params.cn_delta_r * delta_r
    )

    return CL, CD, CY, Cl, Cm, Cn


def aerodynamic_forces_and_moments_body(
    state,
    params,
    controls,
    wind_world: np.ndarray | None = None,
):
    air_data = compute_air_data(state, wind_world)

    speed = air_data["speed"]
    velocity_body = air_data["velocity_body"]
    qbar = air_data["qbar"]

    if speed < 1e-6:
        return np.zeros(3), np.zeros(3), air_data

    CL, CD, CY, Cl, Cm, Cn = aerodynamic_coefficients(
        state,
        params,
        controls,
        air_data,
    )

    u = velocity_body[0]
    w = velocity_body[2]

    speed_xz = np.sqrt(u**2 + w**2)

    # Traînée : opposée à la vitesse relative
    drag_direction_body = -velocity_body / speed

    # Portance : perpendiculaire à la vitesse dans le plan x-z
    # En repère aéronautique, z_body pointe vers le bas.
    # Donc une portance positive pointe vers z négatif.
    if speed_xz < 1e-6:
        lift_direction_body = np.array([0.0, 0.0, -1.0])
    else:
        lift_direction_body = np.array([
            w / speed_xz,
            0.0,
            -u / speed_xz,
        ])

    # Force latérale
    side_direction_body = np.array([0.0, 1.0, 0.0])

    lift = qbar * params.wing_area * CL
    drag = qbar * params.wing_area * CD
    side = qbar * params.wing_area * CY

    force_aero_body = (
        lift * lift_direction_body
        + drag * drag_direction_body
        + side * side_direction_body
    )

    moment_body = qbar * params.wing_area * np.array([
        params.span * Cl,
        params.mean_chord * Cm,
        params.span * Cn,
    ])

    return force_aero_body, moment_body, air_data


def total_forces_and_moments(
    state,
    params,
    controls,
    wind_world: np.ndarray | None = None,
):
    force_aero_body, moment_body, air_data = aerodynamic_forces_and_moments_body(
        state,
        params,
        controls,
        wind_world,
    )

    thrust = thrust_magnitude(state, params)

    force_engine_body = np.array([
        thrust,
        0.0,
        0.0,
    ])

    force_body = force_aero_body + force_engine_body

    R_body_to_world = air_data["R_body_to_world"]

    force_world = R_body_to_world @ force_body

    # Monde NED : z vers le bas, donc gravité positive en z
    gravity_world = np.array([
        0.0,
        0.0,
        params.mass * 9.81,
    ])

    force_world += gravity_world

    return force_world, moment_body, air_data
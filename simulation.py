from dataclasses import dataclass
import numpy as np


# @dataclass
# class AircraftState:
#     position: np.ndarray
#     velocity: np.ndarray
#     pitch: float       # theta, en radians
#     pitch_rate: float  # q, en rad/s
#     throttle: float

@dataclass
class AircraftState3D:
    position: np.ndarray          # [x, y, z]
    velocity: np.ndarray          # [vx, vy, vz]
    roll: float                   # rad
    pitch: float                  # rad
    yaw: float                    # rad
    angular_velocity: np.ndarray  # [p, q, r]
    throttle: float               # 0.0 à 1.5

@dataclass
class SpeedControllerState:
    previous_speed: float = 0.0

@dataclass
class AircraftParameters:
    mass: float
    wing_area: float
    mean_chord: float
    pitch_inertia: float
    drag_coefficient_zero: float
    induced_drag_factor: float
    max_load_factor: float
    cm0: float
    cm_alpha: float
    cm_q: float
    cm_delta_e: float
    max_dry_thrust: float
    max_afterburner_thrust: float
    throttle_response: float


# def initial_state() -> AircraftState:
#     return AircraftState(
#         position=np.array([0.0, 0.0, 100.0]),
#         velocity=np.array([350.0, 0.0, 0.0]),
#         pitch=np.deg2rad(5.0),
#         pitch_rate=0.0,
#         throttle=0.7,
#     )

def initial_state() -> AircraftState3D:
    return AircraftState3D(
        position=np.array([0.0, 0.0, 1000.0]),
        velocity=np.array([180.0, 0.0, 0.0]),
        roll=0.0,
        pitch=np.deg2rad(3.0),
        yaw=0.0,
        angular_velocity=np.zeros(3),
        throttle=0.7,
    )

def rafale_like_parameters() -> AircraftParameters:
    return AircraftParameters(
        mass=15_000.0,
        wing_area=45.7,
        mean_chord=4.5,
        pitch_inertia=250_000.0,
        max_load_factor=9.0,
        drag_coefficient_zero=0.025,
        induced_drag_factor=0.08,
        cm0=0.0,
        cm_alpha=-0.8,
        cm_q=-12.0,
        cm_delta_e=1.0,
        
        # Deux moteurs M88, modèle simplifié
        max_dry_thrust=100_000.0,          # N, sans postcombustion
        max_afterburner_thrust=150_000.0,  # N, avec postcombustion

        # Plus grand = moteur répond plus vite
        throttle_response=2.0,
    )


def air_density(altitude_m: float) -> float:
    """
    Atmosphère simplifiée.
    Correcte seulement comme approximation basse altitude.
    """
    rho0 = 1.225
    scale_height = 8_500.0
    return rho0 * np.exp(-altitude_m / scale_height)


################
################

def rotation_matrix_from_euler(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # Rotation yaw autour de Z
    Rz = np.array([
        [cy, -sy, 0.0],
        [sy,  cy, 0.0],
        [0.0, 0.0, 1.0],
    ])

    # Rotation pitch autour de Y
    Ry = np.array([
        [ cp, 0.0, -sp],
        [0.0, 1.0, 0.0],
        [ sp, 0.0, cp],
    ])

    # Rotation roll autour de X
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr,  cr],
    ])

    return Rz @ Ry @ Rx


def wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

################
################
def body_axes(state: AircraftState3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rotation = rotation_matrix_from_euler(state.roll, state.pitch, state.yaw)
    forward = rotation @ np.array([1.0, 0.0, 0.0])
    right = rotation @ np.array([0.0, 1.0, 0.0])
    up = rotation @ np.array([0.0, 0.0, 1.0])

    return forward, right, up


def velocity_in_body_frame(state: AircraftState3D) -> np.ndarray:
    rotation = rotation_matrix_from_euler(state.roll, state.pitch, state.yaw)
    return rotation.T @ state.velocity


def angle_of_attack(state: AircraftState3D) -> float:
    horizontal_speed = np.linalg.norm(state.velocity[:2])
    flight_path_pitch = np.arctan2(state.velocity[2], horizontal_speed)
    alpha = state.pitch - flight_path_pitch
    alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi

    return float(np.clip(alpha, -np.pi / 2.0, np.pi / 2.0))


def sideslip_angle(state: AircraftState3D) -> float:
    body_velocity = velocity_in_body_frame(state)
    speed = np.linalg.norm(body_velocity)

    if speed < 1e-6:
        return 0.0

    return float(np.arcsin(np.clip(body_velocity[1] / speed, -1.0, 1.0)))


def angle_of_attack_2d(state: AircraftState3D) -> float:
    return angle_of_attack(state)


def lift_coefficient_from_alpha(alpha_rad: float) -> float:
    """
    Modèle simplifié CL(alpha).
    Ce n'est pas le vrai modèle Rafale.
    C'est un modèle plausible pour un avion delta/canard fictif inspiré du Rafale.
    """
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


def lift_direction(state: AircraftState3D) -> np.ndarray:
    velocity = state.velocity
    speed = np.linalg.norm(velocity)
    _, _, up = body_axes(state)

    if speed < 1e-6:
        return up

    drag_direction = -velocity / speed
    lift = up - np.dot(up, drag_direction) * drag_direction
    lift_norm = np.linalg.norm(lift)

    if lift_norm < 1e-6:
        return up

    return lift / lift_norm


def side_force_coefficient(beta_rad: float) -> float:
    return float(np.clip(-0.8 * beta_rad, -0.35, 0.35))


def control_deflection(control: float, max_degrees: float) -> float:
    return float(np.clip(control, -1.0, 1.0) * np.deg2rad(max_degrees))


def compute_forces(
    state: AircraftState3D,
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

    alpha = angle_of_attack(state)
    beta = sideslip_angle(state)
    cl = lift_coefficient_from_alpha(alpha)

    dynamic_pressure = 0.5 * rho * speed**2

    lift_magnitude = dynamic_pressure * params.wing_area * cl

    # Limitation de charge : Rafale autour de +9g en limite publique.
    max_lift = params.max_load_factor * params.mass * g
    lift_magnitude = np.clip(
        lift_magnitude,
        -3.2 * params.mass * g,
        max_lift,
    )

    lift_force = lift_direction(state) * lift_magnitude

    # Traînée avec polaire simplifiée :
    # CD = CD0 + k * CL²
    cd = (
        params.drag_coefficient_zero
        + params.induced_drag_factor * cl**2
    )

    drag_magnitude = dynamic_pressure * params.wing_area * cd
    drag_direction = -velocity / speed
    drag_force = drag_direction * drag_magnitude

    _, right, _ = body_axes(state)
    side_magnitude = dynamic_pressure * params.wing_area * side_force_coefficient(beta)
    side_force = right * side_magnitude

    engine_force = thrust_force(state, params)

    total_force = gravity_force + lift_force + drag_force + side_force + engine_force

    return total_force


def apply_coordinated_turn(state: AircraftState3D, dt: float) -> None:
    horizontal_speed = float(np.linalg.norm(state.velocity[:2]))
    if horizontal_speed < 20.0:
        return

    turn_rate = 2.6 * 9.81 * np.tan(-state.roll) / horizontal_speed
    turn_rate = float(np.clip(turn_rate, -0.8, 0.8))

    heading = np.arctan2(state.velocity[1], state.velocity[0])
    heading += turn_rate * dt

    state.velocity[0] = horizontal_speed * np.cos(heading)
    state.velocity[1] = horizontal_speed * np.sin(heading)


def update_state(
    state: AircraftState3D,
    params: AircraftParameters,
    elevator_control: float,
    throttle_command: float,
    dt: float,
    aileron_control: float = 0.0,
    rudder_control: float = 0.0,
) -> AircraftState3D:
    # Le moteur suit progressivement la commande
    update_throttle(state, params, throttle_command, dt)

    # On calcule forces et moment à partir de l'état courant
    force = compute_forces(state, params)
    moments = compute_moments(
        state,
        params,
        elevator_control,
        aileron_control,
        rudder_control,
    )

    # Translation
    acceleration = force / params.mass

    state.velocity = state.velocity + acceleration * dt
    apply_coordinated_turn(state, dt)
    state.position = state.position + state.velocity * dt

    inertia = np.array([
        params.pitch_inertia * 0.55,
        params.pitch_inertia,
        params.pitch_inertia * 1.25,
    ])
    angular_acceleration = moments / inertia

    state.angular_velocity = state.angular_velocity + angular_acceleration * dt
    state.angular_velocity = np.clip(
        state.angular_velocity,
        np.array([-1.4, -0.9, -0.8]),
        np.array([1.4, 0.9, 0.8]),
    )
    state.roll = state.roll + state.angular_velocity[0] * dt
    state.pitch = state.pitch + state.angular_velocity[1] * dt
    state.yaw = state.yaw + state.angular_velocity[2] * dt

    horizontal_speed = np.linalg.norm(state.velocity[:2])
    if horizontal_speed > 5.0:
        velocity_heading = np.arctan2(state.velocity[1], state.velocity[0])
        yaw_alignment_error = wrap_angle(velocity_heading - state.yaw)
        state.yaw += 4.0 * yaw_alignment_error * dt
        state.angular_velocity[2] = 0.35 * yaw_alignment_error

    state.roll = wrap_angle(state.roll)
    pitch_limit = np.deg2rad(80.0)
    if state.pitch < -pitch_limit or state.pitch > pitch_limit:
        state.angular_velocity[1] = 0.0

    state.pitch = float(np.clip(state.pitch, -pitch_limit, pitch_limit))
    state.yaw = wrap_angle(state.yaw)

    return state

def elevator_deflection(control: float) -> float:
    return control_deflection(control, 25.0)

def compute_pitch_moment(
    state: AircraftState3D,
    params: AircraftParameters,
    elevator_control: float,
) -> float:
    altitude = max(state.position[2], 0.0)
    rho = air_density(altitude)

    speed = np.linalg.norm(state.velocity)

    if speed < 1e-6:
        return 0.0

    alpha = angle_of_attack(state)
    delta_e = elevator_deflection(elevator_control)

    dynamic_pressure = 0.5 * rho * speed**2

    # Terme adimensionné de vitesse angulaire
    pitch_rate_term = state.angular_velocity[1] * params.mean_chord / (2.0 * speed)

    cm = (
        params.cm0
        + params.cm_alpha * alpha
        + params.cm_q * pitch_rate_term
        + params.cm_delta_e * delta_e
    )

    pitch_moment = (
        dynamic_pressure
        * params.wing_area
        * params.mean_chord
        * cm
    )

    return pitch_moment


def compute_moments(
    state: AircraftState3D,
    params: AircraftParameters,
    elevator_control: float,
    aileron_control: float = 0.0,
    rudder_control: float = 0.0,
) -> np.ndarray:
    pitch_moment = compute_pitch_moment(state, params, elevator_control)
    altitude = max(state.position[2], 0.0)
    rho = air_density(altitude)
    speed = np.linalg.norm(state.velocity)

    if speed < 1e-6:
        return np.array([0.0, pitch_moment, 0.0])

    beta = sideslip_angle(state)
    delta_a = control_deflection(aileron_control, 20.0)
    delta_r = control_deflection(rudder_control, 25.0)
    dynamic_pressure = 0.5 * rho * speed**2
    moment_scale = dynamic_pressure * params.wing_area * params.mean_chord

    roll_damping = -3.5 * state.angular_velocity[0] * params.mean_chord / (2.0 * speed)
    yaw_damping = -2.5 * state.angular_velocity[2] * params.mean_chord / (2.0 * speed)
    dihedral_roll = -0.25 * beta
    weathercock_yaw = -0.45 * beta
    aileron_roll = 0.45 * delta_a
    rudder_yaw = 0.28 * delta_r
    rudder_roll_coupling = 0.06 * delta_r

    roll_moment = moment_scale * (
        dihedral_roll
        + roll_damping
        + aileron_roll
        + rudder_roll_coupling
    )
    yaw_moment = moment_scale * (weathercock_yaw + yaw_damping + rudder_yaw)

    return np.array([roll_moment, pitch_moment, yaw_moment])

def alpha_controller(
    state,
    alpha_target_rad: float,
) -> float:
    alpha = angle_of_attack(state)

    kp = 1.5
    kd = 0.4

    error = alpha_target_rad - alpha

    command = kp * error - kd * state.angular_velocity[1]

    return float(np.clip(command, -1.0, 1.0))

def altitude_controller(
    state,
    altitude_target: float,
) -> float:
    altitude = state.position[2]
    vertical_speed = state.velocity[2]

    kp_h = 0.002
    kd_h = 0.01

    error_h = altitude_target - altitude

    alpha_target = (
        np.deg2rad(3.0)
        + kp_h * error_h
        - kd_h * vertical_speed
    )

    return float(np.clip(
        alpha_target,
        np.deg2rad(-5.0),
        np.deg2rad(12.0),
    ))

def update_throttle(
    state: AircraftState3D,
    params: AircraftParameters,
    throttle_command: float,
    dt: float,
) -> None:
    
    throttle_command = np.clip(throttle_command, 0, 1.5)
    
    error = throttle_command - state.throttle

    state.throttle += params.throttle_response * error * dt

    state.throttle = float(np.clip(state.throttle, 0.0, 1.5))

def thrust_magnitude(
    state: AircraftState3D,
    params: AircraftParameters,
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

    altitude = max(state.position[2], 0.0)
    thrust *= thrust_altitude_factor(altitude)

    return float(thrust)

def thrust_force(
    state: AircraftState3D,
    params: AircraftParameters,
) -> np.ndarray:
    thrust = thrust_magnitude(state, params)
    direction, _, _ = body_axes(state)

    return thrust * direction

def speed_controller_pd(
    aircraft_state: AircraftState3D,
    controller_state: SpeedControllerState,
    target_speed: float,
    dt: float,
) -> float:
    speed = np.linalg.norm(aircraft_state.velocity)

    error = target_speed - speed

    speed_derivative = (
        speed - controller_state.previous_speed
    ) / dt

    controller_state.previous_speed = speed

    base_throttle = 0.6
    kp = 0.01
    kd = 0.003

    throttle_command = (
        base_throttle
        + kp * error
        - kd * speed_derivative
    )

    return float(np.clip(throttle_command, 0.0, 1.5))

def thrust_altitude_factor(altitude_m: float) -> float:
    rho = air_density(altitude_m)
    rho0 = 1.225

    factor = rho / rho0

    return float(np.clip(factor, 0.3, 1.0))

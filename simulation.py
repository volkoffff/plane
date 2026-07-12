from dataclasses import dataclass
import numpy as np


@dataclass
class AircraftState:
    position: np.ndarray
    velocity: np.ndarray
    pitch: float       # theta, en radians
    pitch_rate: float  # q, en rad/s
    throttle: float

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
    drag_coefficient_zero: float
    induced_drag_factor: float
    cm0: float
    cm_alpha: float
    cm_q: float
    cm_delta_e: float
    max_dry_thrust: float
    max_afterburner_thrust: float
    throttle_response: float


def initial_state() -> AircraftState:
    return AircraftState(
        position=np.array([0.0, 0.0, 100.0]),
        velocity=np.array([350.0, 0.0, 0.0]),
        pitch=np.deg2rad(5.0),
        pitch_rate=0.0,
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


def angle_of_attack_2d(state: AircraftState) -> float:
    vx = state.velocity[0]
    vz = state.velocity[2]

    flight_path_angle = np.arctan2(vz, vx)
    alpha = state.pitch - flight_path_angle

    return alpha


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

    # Limitation de charge : Rafale autour de +9g en limite publique.
    max_lift = params.max_load_factor * params.mass * g
    lift_magnitude = np.clip(
        lift_magnitude,
        -3.2 * params.mass * g,
        max_lift,
    )

    lift_force = lift_direction_2d(state) * lift_magnitude

    # Traînée avec polaire simplifiée :
    # CD = CD0 + k * CL²
    cd = (
        params.drag_coefficient_zero
        + params.induced_drag_factor * cl**2
    )

    drag_magnitude = dynamic_pressure * params.wing_area * cd
    drag_direction = -velocity / speed
    drag_force = drag_direction * drag_magnitude

    engine_force = thrust_force(state, params)

    total_force = gravity_force + lift_force + drag_force + engine_force

    return total_force


def update_state(
    state: AircraftState,
    params: AircraftParameters,
    elevator_control: float,
    throttle_command: float,
    dt: float,
) -> AircraftState:
    # Le moteur suit progressivement la commande
    update_throttle(state, params, throttle_command, dt)

    # On calcule forces et moment à partir de l'état courant
    force = compute_forces(state, params)
    pitch_moment = compute_pitch_moment(
        state,
        params,
        elevator_control,
    )

    # Translation
    acceleration = force / params.mass

    state.velocity = state.velocity + acceleration * dt
    state.position = state.position + state.velocity * dt

    # rotation
    pitch_acceleration = pitch_moment / params.pitch_inertia

    state.pitch_rate = state.pitch_rate + pitch_acceleration * dt
    state.pitch = state.pitch + state.pitch_rate * dt

    return state

def elevator_deflection(control: float) -> float:
    max_deflection = np.deg2rad(25.0)
    control = np.clip(control, -1.0, 1.0)
    return control * max_deflection

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

    # Terme adimensionné de vitesse angulaire
    pitch_rate_term = state.pitch_rate * params.mean_chord / (2.0 * speed)

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

def alpha_controller(
    state,
    alpha_target_rad: float,
) -> float:
    alpha = angle_of_attack_2d(state)

    kp = 1.5
    kd = 0.4

    error = alpha_target_rad - alpha

    command = kp * error - kd * state.pitch_rate

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
    state: AircraftState,
    params: AircraftParameters,
    throttle_command: float,
    dt: float,
) -> None:
    
    throttle_command = np.clip(throttle_command, 0, 1.5)
    
    error = throttle_command - state.throttle

    state.throttle += params.throttle_response * error * dt

    state.throttle = float(np.clip(state.throttle, 0.0, 1.5))

def thrust_magnitude(
    state: AircraftState,
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

def speed_controller_pd(
    aircraft_state: AircraftState,
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
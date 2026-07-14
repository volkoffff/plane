import numpy as np

from physics.state import AircraftState, SpeedControllerState

from dataclasses import dataclass


def elevator_deflection(control: float) -> float:
    max_deflection = np.deg2rad(25.0)
    control = np.clip(control, -1.0, 1.0)
    return float(control * max_deflection)

def altitude_controller(
    state: AircraftState,
    altitude_target: float,
) -> float:
    altitude = state.position[2]
    vertical_speed = state.velocity[2]

    kp_h = 0.002
    kd_h = 0.01

    error_h = altitude_target - altitude
    alpha_target = np.deg2rad(3.0) + kp_h * error_h - kd_h * vertical_speed

    return float(np.clip(
        alpha_target,
        np.deg2rad(-5.0),
        np.deg2rad(12.0),
    ))


def speed_controller_pd(
    aircraft_state: AircraftState,
    controller_state: SpeedControllerState,
    target_speed: float,
    dt: float,
) -> float:
    speed = np.linalg.norm(aircraft_state.velocity)
    error = target_speed - speed
    speed_derivative = (speed - controller_state.previous_speed) / dt

    controller_state.previous_speed = speed

    base_throttle = 0.6
    kp = 0.01
    kd = 0.003

    throttle_command = base_throttle + kp * error - kd * speed_derivative

    return float(np.clip(throttle_command, 0.0, 1.5))

@dataclass
class ControlInputs:
    elevator: float          # -1 à 1
    aileron: float           # -1 à 1
    rudder: float            # -1 à 1
    throttle_command: float  # 0 à 1.5


def normalized_to_surface_deflections(
    controls: ControlInputs,
    params,
) -> tuple[float, float, float]:
    elevator = np.clip(controls.elevator, -1.0, 1.0)
    aileron = np.clip(controls.aileron, -1.0, 1.0)
    rudder = np.clip(controls.rudder, -1.0, 1.0)

    delta_e = elevator * params.max_elevator_deflection
    delta_a = aileron * params.max_aileron_deflection
    delta_r = rudder * params.max_rudder_deflection

    return delta_e, delta_a, delta_r


def flight_control_law_from_mouse(
    state,
    mouse_dx: float,
    mouse_dy: float,
    rudder_input: float,
    throttle_command: float,
) -> ControlInputs:
    """
    mouse_dx : -1 à 1
    mouse_dy : -1 à 1

    mouse_dx > 0 : demande de roulis à droite
    mouse_dy < 0 : souris vers le haut, demande de cabré
    """

    max_roll_rate = np.deg2rad(120.0)
    max_pitch_rate = np.deg2rad(60.0)

    target_p = mouse_dx * max_roll_rate
    target_q = -mouse_dy * max_pitch_rate

    p, q, r = state.omega_body

    kp_roll = 0.8
    kp_pitch = 1.0

    aileron = kp_roll * (target_p - p)
    elevator = kp_pitch * (target_q - q)

    # Petit amortisseur de lacet
    rudder = rudder_input - 0.2 * r

    return ControlInputs(
        elevator=float(np.clip(elevator, -1.0, 1.0)),
        aileron=float(np.clip(aileron, -1.0, 1.0)),
        rudder=float(np.clip(rudder, -1.0, 1.0)),
        throttle_command=float(np.clip(throttle_command, 0.0, 1.5)),
    )
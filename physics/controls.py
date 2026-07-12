import numpy as np

from physics.aerodynamics import angle_of_attack_2d
from physics.state import AircraftState, SpeedControllerState


def elevator_deflection(control: float) -> float:
    max_deflection = np.deg2rad(25.0)
    control = np.clip(control, -1.0, 1.0)
    return float(control * max_deflection)


def alpha_controller(
    state: AircraftState,
    alpha_target_rad: float,
) -> float:
    alpha = angle_of_attack_2d(state)

    kp = 1.5
    kd = 0.4

    error = alpha_target_rad - alpha
    command = kp * error - kd * state.pitch_rate

    return float(np.clip(command, -1.0, 1.0))


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

from __future__ import annotations

import numpy as np

from physics.math3d import quaternion_to_euler
from physics.state import AircraftState
from piloting.commands import FlightCommand


MAX_ROLL_RATE = np.deg2rad(120.0)
MAX_PITCH_RATE = np.deg2rad(60.0)
GRAVITY = 9.81


def wrap_angle(angle_rad: float) -> float:
    return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)


def coordinated_yaw_rate(
    state: AircraftState,
    bank_angle_deg: float,
) -> float:
    speed = max(float(np.linalg.norm(state.velocity)), 1.0)
    bank_angle = np.deg2rad(np.clip(bank_angle_deg, -70.0, 70.0))

    return float(GRAVITY * np.tan(bank_angle) / speed)


def attitude_command(
    state: AircraftState,
    target_roll_deg: float,
    target_pitch_deg: float,
    throttle_command: float,
    rudder_input: float = 0.0,
    coordinated_turn: bool = False,
) -> FlightCommand:
    roll, pitch, _ = quaternion_to_euler(state.quaternion)

    roll_error = wrap_angle(np.deg2rad(target_roll_deg) - roll)
    pitch_error = np.deg2rad(target_pitch_deg) - pitch

    target_p = np.clip(
        1.8 * roll_error,
        -np.deg2rad(45.0),
        np.deg2rad(45.0),
    )
    target_q = np.clip(
        1.6 * pitch_error,
        -np.deg2rad(35.0),
        np.deg2rad(35.0),
    )

    return FlightCommand(
        mouse_dx=float(target_p / MAX_ROLL_RATE),
        mouse_dy=float(-target_q / MAX_PITCH_RATE),
        rudder_input=rudder_input,
        throttle_command=throttle_command,
        target_yaw_rate=(
            coordinated_yaw_rate(state, target_roll_deg)
            if coordinated_turn
            else 0.0
        ),
    )


def pitch_rate_command(
    pitch_rate_deg_s: float,
    throttle_command: float,
) -> FlightCommand:
    return FlightCommand(
        mouse_dx=0.0,
        mouse_dy=float(-np.deg2rad(pitch_rate_deg_s) / MAX_PITCH_RATE),
        rudder_input=0.0,
        throttle_command=throttle_command,
    )

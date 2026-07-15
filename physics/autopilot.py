from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from physics.math3d import quaternion_to_euler
from physics.state import AircraftState


MAX_ROLL_RATE = np.deg2rad(120.0)
MAX_PITCH_RATE = np.deg2rad(60.0)
GRAVITY = 9.81


@dataclass(frozen=True)
class FlightCommand:
    mouse_dx: float
    mouse_dy: float
    rudder_input: float
    throttle_command: float
    target_yaw_rate: float = 0.0


@dataclass(frozen=True)
class FlightInstruction:
    name: str
    duration: float
    command: Callable[[AircraftState, float], FlightCommand]


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


def build_flight_program() -> list[FlightInstruction]:
    return [
        FlightInstruction(
            "avancer",
            4.0,
            lambda state, t: attitude_command(state, 0.0, 3.0, 0.90),
        ),
        FlightInstruction(
            "monter",
            6.0,
            lambda state, t: attitude_command(state, 0.0, 22.0, 1.15),
        ),
        FlightInstruction(
            "palier",
            3.0,
            lambda state, t: attitude_command(state, 0.0, 4.0, 0.95),
        ),
        FlightInstruction(
            "tourner",
            8.0,
            lambda state, t: attitude_command(
                state,
                42.0,
                8.0,
                1.00,
                coordinated_turn=True,
            ),
        ),
        FlightInstruction(
            "sortie virage",
            5.0,
            lambda state, t: attitude_command(state, 0.0, 3.0, 0.95),
        ),
        FlightInstruction(
            "prise vitesse",
            4.0,
            lambda state, t: attitude_command(state, 0.0, 0.0, 1.35),
        ),
        FlightInstruction(
            "looping",
            14.0,
            lambda state, t: pitch_rate_command(49.0, 1.45),
        ),
        FlightInstruction(
            "recuperation",
            8.0,
            lambda state, t: attitude_command(state, 0.0, 5.0, 1.05),
        ),
    ]

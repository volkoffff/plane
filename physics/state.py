from dataclasses import dataclass

import numpy as np


@dataclass
class AircraftState:
    position: np.ndarray
    velocity: np.ndarray
    pitch: float
    pitch_rate: float
    throttle: float


@dataclass
class SpeedControllerState:
    previous_speed: float = 0.0


def initial_state() -> AircraftState:
    return AircraftState(
        position=np.array([0.0, 0.0, 100.0]),
        velocity=np.array([150.0, 0.0, 0.0]),
        pitch=np.deg2rad(5.0),
        pitch_rate=0.0,
        throttle=0.7,
    )

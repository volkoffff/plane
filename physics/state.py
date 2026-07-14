from dataclasses import dataclass
from physics.math3d import euler_to_quaternion

import numpy as np


@dataclass
class AircraftState:
    position: np.ndarray       # monde NED : [x, y, z_down]
    velocity: np.ndarray       # monde NED : [vx, vy, vz_down]
    quaternion: np.ndarray     # orientation body -> world, [w, x, y, z]
    omega_body: np.ndarray     # [p, q, r] en rad/s dans le repère avion
    throttle: float            # 0.0 à 1.5


@dataclass
class SpeedControllerState:
    previous_speed: float = 0.0


def initial_state() -> AircraftState:
    return AircraftState(
        position=np.array([0.0, 0.0, -1000.0]),      # altitude 1000 m
        velocity=np.array([180.0, 0.0, 0.0]),        # 180 m/s vers l'avant
        quaternion=euler_to_quaternion(
            roll=0.0,
            pitch=np.deg2rad(3.0),
            yaw=0.0,
        ),
        omega_body=np.array([0.0, 0.0, 0.0]),
        throttle=0.7,
    )

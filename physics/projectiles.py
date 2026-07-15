from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from physics.math3d import quaternion_to_matrix
from physics.state import AircraftState


GRAVITY_NED = np.array([0.0, 0.0, 9.81])


@dataclass
class Bullet:
    position: np.ndarray
    velocity: np.ndarray
    previous_position: np.ndarray
    age: float = 0.0
    lifetime: float = 6.0

    @property
    def is_alive(self) -> bool:
        return self.age < self.lifetime


def create_bullet_from_aircraft(
    state: AircraftState,
    muzzle_speed: float = 850.0,
    muzzle_offset_body: np.ndarray | None = None,
    lifetime: float = 4.0,
) -> Bullet:
    if muzzle_offset_body is None:
        muzzle_offset_body = np.array([11.0, 0.0, 0.0])

    body_to_world = quaternion_to_matrix(state.quaternion)
    forward_world = body_to_world @ np.array([1.0, 0.0, 0.0])
    muzzle_position = state.position + body_to_world @ muzzle_offset_body
    muzzle_velocity = state.velocity + forward_world * muzzle_speed

    return Bullet(
        position=muzzle_position.copy(),
        velocity=muzzle_velocity.copy(),
        previous_position=muzzle_position.copy(),
        lifetime=lifetime,
    )


def integrate_bullets(
    bullets: list[Bullet],
    dt: float,
) -> list[Bullet]:
    alive_bullets = []

    for bullet in bullets:
        bullet.previous_position = bullet.position.copy()
        bullet.velocity = bullet.velocity + GRAVITY_NED * dt
        bullet.position = bullet.position + bullet.velocity * dt
        bullet.age += dt

        if bullet.is_alive:
            alive_bullets.append(bullet)

    return alive_bullets

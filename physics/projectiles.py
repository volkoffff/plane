from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

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
    owner_id: int | None = None
    team: object | None = None
    damage: float = 10.0
    last_step_duration: float = field(default=0.0, init=False)

    @property
    def is_alive(self) -> bool:
        return self.age < self.lifetime


def create_bullet_from_aircraft(
    state: AircraftState,
    muzzle_speed: float = 850.0,
    muzzle_offset_body: np.ndarray | None = None,
    lifetime: float = 4.0,
    owner_id: int | None = None,
    team: object | None = None,
    damage: float = 10.0,
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
        owner_id=owner_id,
        team=team,
        damage=damage,
    )


def integrate_bullets(
    bullets: list[Bullet],
    dt: float,
) -> list[Bullet]:
    """Advance active bullets, retaining final segments for collision resolution.

    Combat removes expired bullets after checking their last valid trajectory.
    """
    if not isfinite(dt) or dt <= 0.0:
        raise ValueError("Projectile step must be positive and finite")
    integrated_bullets = []

    for bullet in bullets:
        if not bullet.is_alive:
            continue
        bullet.last_step_duration = min(dt, bullet.lifetime - bullet.age)
        bullet.previous_position = bullet.position.copy()
        bullet.velocity = bullet.velocity + GRAVITY_NED * bullet.last_step_duration
        bullet.position = bullet.position + bullet.velocity * bullet.last_step_duration
        bullet.age += dt

        integrated_bullets.append(bullet)

    return integrated_bullets

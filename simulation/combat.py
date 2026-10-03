from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from physics.collisions import segment_box_entry
from physics.math3d import quaternion_to_matrix

if TYPE_CHECKING:
    from simulation.aircraft import AircraftEntity
    from simulation.world import SimulationWorld


@dataclass(frozen=True)
class DamageEvent:
    target_id: int
    attacker_id: int | None
    damage: float
    reason: str


@dataclass
class CombatSystem:
    """Resolve projectile hits; events contain damage applied in the current step."""

    events: list[DamageEvent] = field(default_factory=list)
    friendly_fire: bool = False

    def update(
        self,
        world: SimulationWorld,
        dt: float,
        *,
        previous_positions: Mapping[int, np.ndarray] | None = None,
    ) -> None:
        """Sweep bullets against translating, oriented aircraft boxes.

        Aircraft translation is interpolated over the step; orientation is held
        at its final value. Without previous positions, targets are stationary.
        """
        if not isfinite(dt) or dt <= 0.0:
            raise ValueError("Combat step must be positive and finite")
        self.events.clear()
        previous_positions = previous_positions or {}
        colliders = [
            (
                aircraft,
                quaternion_to_matrix(aircraft.physics.state.quaternion).T,
                previous_positions.get(aircraft.id, aircraft.physics.state.position),
            )
            for aircraft in world.aircraft.values()
            if not aircraft.is_dead()
        ]
        survivors = []
        for bullet in world.bullets:
            # A bullet expiring during this step still has a final valid segment.
            if not bullet.is_alive and bullet.last_step_duration <= 0.0:
                continue
            duration = bullet.last_step_duration or dt
            step_fraction = min(duration / dt, 1.0)
            closest_target = None
            closest_entry = float("inf")

            for aircraft, world_to_body, previous_position in colliders:
                if aircraft.is_dead() or aircraft.id == bullet.owner_id:
                    continue
                if not self.friendly_fire and bullet.team == aircraft.team:
                    continue

                # Subtract the target's motion before applying the slab query.
                position = aircraft.physics.state.position
                segment_end_position = (
                    previous_position + (position - previous_position) * step_fraction
                )
                start_body = world_to_body @ (
                    bullet.previous_position - previous_position
                )
                end_body = world_to_body @ (bullet.position - segment_end_position)
                entry = segment_box_entry(
                    start_body, end_body, aircraft.collision_half_extents
                )
                if entry is not None and entry < closest_entry:
                    closest_entry = entry
                    closest_target = aircraft

            if closest_target is not None:
                self.apply_damage(
                    closest_target,
                    bullet.damage,
                    attacker_id=bullet.owner_id,
                    reason="bullet",
                )
            elif bullet.is_alive:
                survivors.append(bullet)

        world.bullets = survivors

    def apply_damage(
        self,
        target: AircraftEntity,
        damage: float,
        attacker_id: int | None = None,
        reason: str = "unknown",
    ) -> None:
        if not isfinite(damage) or damage < 0.0:
            raise ValueError("Damage must be finite and nonnegative")
        if target.is_dead() or damage == 0.0:
            return

        applied_damage = min(damage, target.health)
        target.health = max(0.0, target.health - applied_damage)

        self.events.append(
            DamageEvent(
                target_id=target.id,
                attacker_id=attacker_id,
                damage=applied_damage,
                reason=reason,
            )
        )

        if target.health <= 0.0:
            target.death()

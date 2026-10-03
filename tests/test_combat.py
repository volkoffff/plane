from __future__ import annotations

import unittest

import numpy as np

from physics.aircraft_physics import AircraftPhysics
from physics.math3d import euler_to_quaternion
from physics.projectiles import Bullet, integrate_bullets
from physics.state import initial_state
from piloting.commands import AircraftAction, FlightCommand
from simulation.aircraft import AircraftEntity
from simulation.teams import Team
from simulation.world import SimulationWorld


def add_target(
    world: SimulationWorld,
    position=(0.0, 0.0, 0.0),
    team: Team = Team.BRAVO,
    half_extents=(1.0, 1.0, 1.0),
    yaw: float = 0.0,
) -> AircraftEntity:
    state = initial_state()
    state.position = np.array(position, dtype=float)
    state.velocity = np.zeros(3)
    state.quaternion = euler_to_quaternion(0.0, 0.0, yaw)
    target = AircraftEntity(
        team=team,
        physics=AircraftPhysics(state=state),
        collision_half_extents=half_extents,
    )
    world.add_aircraft(target)
    return target


def make_bullet(start, end, owner_id=None, damage=10.0, team=Team.ALPHA) -> Bullet:
    return Bullet(
        position=np.array(end, dtype=float),
        previous_position=np.array(start, dtype=float),
        velocity=np.zeros(3),
        owner_id=owner_id,
        team=team,
        damage=damage,
    )


class CombatTests(unittest.TestCase):
    def test_fast_bullet_hits_even_when_both_endpoints_are_outside(self) -> None:
        world = SimulationWorld()
        target = add_target(world)
        world.bullets = [make_bullet((-100, 0, 0), (100, 0, 0), owner_id=999)]
        world.combat.update(world, 0.01)
        self.assertEqual(target.health, 90.0)
        self.assertFalse(world.bullets)
        event = world.combat.events[0]
        self.assertEqual(
            (event.target_id, event.attacker_id, event.damage, event.reason),
            (target.id, 999, 10.0, "bullet"),
        )
        world.combat.update(world, 0.01)
        self.assertEqual(target.health, 90.0)
        self.assertFalse(world.combat.events)

    def test_only_first_target_is_hit_independent_of_insertion_order(self) -> None:
        world = SimulationWorld()
        far = add_target(world, position=(10, 0, 0))
        near = add_target(world)
        world.bullets = [make_bullet((-20, 0, 0), (20, 0, 0))]
        world.combat.update(world, 0.01)
        self.assertEqual(near.health, 90.0)
        self.assertEqual(far.health, 100.0)

    def test_owner_and_allies_are_excluded_by_default(self) -> None:
        world = SimulationWorld()
        owner = add_target(world, position=(-10, 0, 0), team=Team.ALPHA)
        ally = add_target(world, team=Team.ALPHA)
        enemy = add_target(world, position=(10, 0, 0))
        world.bullets = [make_bullet((-20, 0, 0), (20, 0, 0), owner_id=owner.id)]
        world.combat.update(world, 0.01)
        self.assertEqual(
            (owner.health, ally.health, enemy.health), (100.0, 100.0, 90.0)
        )

    def test_friendly_fire_can_be_enabled_without_hitting_owner(self) -> None:
        world = SimulationWorld()
        world.combat.friendly_fire = True
        owner = add_target(world, position=(-10, 0, 0), team=Team.ALPHA)
        ally = add_target(world, team=Team.ALPHA)
        world.bullets = [make_bullet((-20, 0, 0), (20, 0, 0), owner_id=owner.id)]
        world.combat.update(world, 0.01)
        self.assertEqual(owner.health, 100.0)
        self.assertEqual(ally.health, 90.0)

    def test_dead_targets_are_ignored(self) -> None:
        world = SimulationWorld()
        dead = add_target(world)
        dead.death()
        live = add_target(world, position=(10, 0, 0))
        world.bullets = [make_bullet((-20, 0, 0), (20, 0, 0))]
        world.combat.update(world, 0.01)
        self.assertEqual(dead.health, 100.0)
        self.assertEqual(live.health, 90.0)

    def test_miss_remains_in_world(self) -> None:
        world = SimulationWorld()
        target = add_target(world)
        bullet = make_bullet((-20, 2, 0), (20, 2, 0))
        world.bullets = [bullet]
        world.combat.update(world, 0.01)
        self.assertEqual(target.health, 100.0)
        self.assertEqual(world.bullets, [bullet])
        self.assertFalse(world.combat.events)

    def test_box_follows_aircraft_orientation(self) -> None:
        world = SimulationWorld()
        target = add_target(world, half_extents=(5.0, 1.0, 1.0), yaw=np.pi / 2)
        world.bullets = [make_bullet((-10, 4, 0), (10, 4, 0))]
        world.combat.update(world, 0.01)
        self.assertEqual(target.health, 90.0)

    def test_translating_target_crosses_stationary_bullet(self) -> None:
        world = SimulationWorld()
        target = add_target(world, position=(0, 10, 0))
        world.bullets = [make_bullet((0, 0, 0), (0, 0, 0))]
        world.combat.update(
            world, 0.01, previous_positions={target.id: np.array([0, -10, 0])}
        )
        self.assertEqual(target.health, 90.0)

    def test_same_velocity_does_not_create_false_hit_at_final_target_position(
        self,
    ) -> None:
        world = SimulationWorld()
        target = add_target(world, position=(10, 0, 0))
        world.bullets = [make_bullet((0, 0, 0), (20, 0, 0))]
        world.combat.update(
            world, 0.01, previous_positions={target.id: np.array([-10, 0, 0])}
        )
        self.assertEqual(target.health, 100.0)
        self.assertEqual(len(world.bullets), 1)

    def test_final_segment_can_hit_before_bullet_expires(self) -> None:
        world = SimulationWorld()
        target = add_target(world, position=(0.5, 0, 0), half_extents=(0.1, 0.1, 0.1))
        bullet = make_bullet((0, 0, 0), (0, 0, 0))
        bullet.velocity = np.array([200.0, 0.0, 0.0])
        bullet.lifetime = 0.005
        world.bullets = integrate_bullets([bullet], 0.01)
        self.assertFalse(bullet.is_alive)
        world.combat.update(world, 0.01)
        self.assertEqual(target.health, 90.0)
        self.assertFalse(world.bullets)

    def test_bullet_cannot_hit_beyond_its_lifetime(self) -> None:
        world = SimulationWorld()
        target = add_target(world, position=(1.5, 0, 0), half_extents=(0.1, 0.1, 0.1))
        bullet = make_bullet((0, 0, 0), (0, 0, 0))
        bullet.velocity = np.array([200.0, 0.0, 0.0])
        bullet.lifetime = 0.005
        world.bullets = integrate_bullets([bullet], 0.01)
        world.combat.update(world, 0.01)
        self.assertEqual(target.health, 100.0)
        self.assertFalse(world.bullets)

    def test_target_motion_stops_at_bullet_expiry_for_collision_query(self) -> None:
        world = SimulationWorld()
        target = add_target(world, half_extents=(0.1, 0.1, 0.1))
        bullet = make_bullet((0, 0, 0), (0, 0, 0))
        bullet.lifetime = 0.005
        world.bullets = integrate_bullets([bullet], 0.01)
        world.combat.update(
            world, 0.01, previous_positions={target.id: np.array([-2, 0, 0])}
        )
        self.assertEqual(target.health, 100.0)

    def test_expired_bullet_without_valid_segment_is_removed(self) -> None:
        world = SimulationWorld()
        target = add_target(world)
        bullet = make_bullet((-2, 0, 0), (2, 0, 0))
        bullet.age = bullet.lifetime
        world.bullets = [bullet]
        world.combat.update(world, 0.01)
        self.assertEqual(target.health, 100.0)
        self.assertFalse(world.bullets)

    def test_lethal_damage_records_only_health_actually_lost(self) -> None:
        world = SimulationWorld()
        target = add_target(world)
        world.combat.apply_damage(target, 250.0, attacker_id=999, reason="bullet")
        self.assertEqual(target.health, 0.0)
        self.assertTrue(target.is_dead())
        self.assertEqual(world.combat.events[0].damage, 100.0)
        world.combat.apply_damage(target, 10.0)
        self.assertEqual(len(world.combat.events), 1)

    def test_zero_damage_does_not_create_event_and_invalid_damage_is_rejected(
        self,
    ) -> None:
        world = SimulationWorld()
        target = add_target(world)
        world.combat.apply_damage(target, 0.0)
        self.assertFalse(world.combat.events)
        for damage in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                world.combat.apply_damage(target, damage)
        self.assertEqual(target.health, 100.0)

    def test_world_step_connects_gun_projectile_and_combat(self) -> None:
        world = SimulationWorld()
        owner = add_target(world, position=(0, 0, -1000), team=Team.ALPHA)
        target = add_target(world, position=(30, 0, -1000))
        flight = FlightCommand(0.0, 0.0, 0.0, 0.0)
        world.step(
            {
                owner.id: AircraftAction(flight, fire_gun=True),
                target.id: AircraftAction(flight),
            },
            0.05,
        )
        self.assertEqual(target.health, 90.0)
        self.assertFalse(world.bullets)
        self.assertEqual(world.combat.events[0].attacker_id, owner.id)

    def test_invalid_dt_does_not_mutate_world(self) -> None:
        world = SimulationWorld()
        target = add_target(world)
        before = target.physics.state.position.copy()
        for dt in (0.0, -0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                world.step({}, dt)
            with self.assertRaises(ValueError):
                world.combat.update(world, dt)
            with self.assertRaises(ValueError):
                integrate_bullets([], dt)
        np.testing.assert_array_equal(target.physics.state.position, before)

    def test_invalid_collision_dimensions_are_rejected(self) -> None:
        for extents in ((0, 1, 1), (-1, 1, 1), (1, 1), (float("nan"), 1, 1)):
            with self.assertRaises(ValueError):
                add_target(SimulationWorld(), half_extents=extents)


if __name__ == "__main__":
    unittest.main()

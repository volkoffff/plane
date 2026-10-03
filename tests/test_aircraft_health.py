from __future__ import annotations

import unittest

from physics.aircraft_physics import AircraftPhysics
from simulation.aircraft import AircraftEntity
from simulation.teams import Team


def make_aircraft(health: float = 100.0) -> AircraftEntity:
    return AircraftEntity(Team.ALPHA, AircraftPhysics(), health=health)


class AircraftHealthTests(unittest.TestCase):
    def test_initial_life_state_and_zero_health(self) -> None:
        living = make_aircraft()
        dead = make_aircraft(0.0)
        self.assertTrue(living.alive)
        self.assertFalse(living.is_dead())
        self.assertFalse(dead.alive)
        self.assertTrue(dead.is_dead())

    def test_health_change_cannot_leave_a_stale_alive_flag(self) -> None:
        aircraft = make_aircraft()
        aircraft.health = 0.0
        self.assertFalse(aircraft.alive)
        with self.assertRaises(AttributeError):
            aircraft.alive = True

    def test_damage_returns_actual_loss_and_cannot_make_health_negative(self) -> None:
        aircraft = make_aircraft(25.0)
        self.assertEqual(aircraft.take_damage(10.0), 10.0)
        self.assertEqual(aircraft.health, 15.0)
        self.assertTrue(aircraft.alive)
        self.assertEqual(aircraft.take_damage(100.0), 15.0)
        self.assertEqual(aircraft.health, 0.0)
        self.assertFalse(aircraft.alive)
        self.assertEqual(aircraft.take_damage(10.0), 0.0)

    def test_explicit_death_is_idempotent_and_sets_health_to_zero(self) -> None:
        aircraft = make_aircraft()
        aircraft.death()
        aircraft.death()
        self.assertEqual(aircraft.health, 0.0)
        self.assertTrue(aircraft.is_dead())

    def test_invalid_initial_health_is_rejected(self) -> None:
        for health in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                make_aircraft(health)

    def test_invalid_damage_does_not_change_health(self) -> None:
        aircraft = make_aircraft()
        for damage in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                aircraft.take_damage(damage)
        self.assertEqual(aircraft.health, 100.0)
        self.assertEqual(aircraft.take_damage(0.0), 0.0)


if __name__ == "__main__":
    unittest.main()

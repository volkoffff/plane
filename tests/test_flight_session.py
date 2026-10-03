from __future__ import annotations

import unittest

import numpy as np

from piloting.player import MouseInput
from simulation.scenarios import create_1_vs_1_world
from simulation.world import SimulationWorld
from viewer.session import FlightSession


def make_session() -> FlightSession:
    world = SimulationWorld()
    create_1_vs_1_world(world)
    return FlightSession(world)


class FlightSessionTests(unittest.TestCase):
    def test_same_flight_at_different_render_rates(self) -> None:
        slow, fast = make_session(), make_session()
        for session, frames, dt in ((slow, 50, 0.02), (fast, 200, 0.005)):
            session.set_key_state("roll_right", True)
            session.set_key_state("throttle_up", True)
            for _ in range(frames):
                session.advance(dt)
        for a, b in zip(slow.world.get_all_aircrafts(), fast.world.get_all_aircrafts()):
            np.testing.assert_allclose(a.physics.state.position, b.physics.state.position)
            np.testing.assert_allclose(a.physics.state.quaternion, b.physics.state.quaternion)
        self.assertAlmostEqual(slow.player_controls.throttle_command,
                               fast.player_controls.throttle_command)

    def test_switch_releases_inputs_and_preserves_each_throttle(self) -> None:
        session = make_session()
        first = session.player_controls
        session.set_key_state("throttle_up", True)
        session.set_fire_trigger(True)
        session.advance(0.02)
        throttle = first.throttle_command
        session.select(1)
        self.assertFalse(first.trigger_fire)
        self.assertFalse(any(first.key_state.values()))
        self.assertIsNot(first, session.player_controls)
        session.select(-1)
        self.assertAlmostEqual(session.player_controls.throttle_command, throttle)

    def test_pause_freezes_physics_and_releases_fire(self) -> None:
        session = make_session()
        session.set_fire_trigger(True)
        session.toggle_pause()
        self.assertEqual(session.advance(1.0), 0)
        self.assertEqual(session.elapsed_time, 0.0)
        self.assertFalse(session.player_controls.trigger_fire)
        session.toggle_pause()
        self.assertEqual(session.advance(0.01), 1)

    def test_stall_is_bounded_and_invalid_dt_rejected(self) -> None:
        session = make_session()
        self.assertEqual(session.advance(5.0), session.max_substeps)
        self.assertLess(session.accumulator, session.fixed_dt)
        for dt in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                session.advance(dt)

    def test_only_selected_aircraft_receives_mouse_and_fire(self) -> None:
        session = make_session()
        session.set_fire_trigger(True)
        session.advance(0.01, MouseInput(x=1.0))
        aircraft = session.world.get_all_aircrafts()
        self.assertGreater(aircraft[0].physics.last_controls.aileron, 0.0)
        self.assertEqual(aircraft[1].physics.last_controls.aileron, 0.0)
        self.assertEqual(len(session.world.bullets), 1)
        self.assertEqual(session.world.bullets[0].owner_id, aircraft[0].id)
        session.mouse_enabled = False
        actions = session.build_actions(MouseInput(x=1.0))
        self.assertEqual(actions[aircraft[0].id].flight.mouse_dx, 0.0)

    def test_empty_world_and_dead_aircraft(self) -> None:
        empty = FlightSession(SimulationWorld())
        empty.select(1)
        self.assertIsNone(empty.player_controls)
        self.assertEqual(empty.advance(0.01), 1)
        session = make_session()
        session.focus.current_aircraft.death()
        session.set_fire_trigger(True)
        session.advance(0.01)
        self.assertFalse(session.world.bullets)


if __name__ == "__main__":
    unittest.main()

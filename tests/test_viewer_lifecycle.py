from __future__ import annotations

import unittest

from physics.math3d import euler_to_quaternion
from simulation.scenarios import create_1_vs_1_world, create_aircraft_in_world
from simulation.world import SimulationWorld
from viewer.app2 import PandaSceneViewer
from viewer.session import FlightSession


class ViewerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = PandaSceneViewer(window_type="none")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.app.destroy()

    def setUp(self) -> None:
        self.world = SimulationWorld()
        self.first, self.second = create_1_vs_1_world(self.world)
        self.app.world = self.world
        self.app.session = FlightSession(self.world)
        self.app.aircraft_focus = self.app.session.focus
        self.app.update_view()

    def test_lethal_damage_removes_visual_and_selects_survivor(self) -> None:
        visual = self.app.aircraft_visuals.visuals[self.first.id]
        root = visual.root
        self.world.combat.apply_damage(self.first, 100.0)
        self.app.update_view()
        self.assertFalse(self.first.alive)
        self.assertTrue(root.isEmpty())
        self.assertIsNone(visual.animations)
        self.assertNotIn(self.first.id, self.app.aircraft_visuals.visuals)
        self.assertIs(self.app.aircraft_focus.current_aircraft, self.second)
        self.assertIn(f"Avion {self.second.id}", self.app.hud.status_text.getText())
        self.assertIn(self.first.id, self.world.aircraft)
        visual.destroy()

    def test_no_living_aircraft_clears_focus_crosshair_and_stale_hud(self) -> None:
        self.app.hud.crosshair_text.show()
        for aircraft in (self.first, self.second):
            self.world.combat.apply_damage(aircraft, 100.0)
        self.app.update_view()
        self.assertFalse(self.app.aircraft_visuals.visuals)
        self.assertIsNone(self.app.aircraft_focus.current_aircraft_id)
        self.assertTrue(self.app.hud.crosshair_text.isHidden())
        self.assertIn("Aucun avion vivant", self.app.hud.status_text.getText())
        self.app.session.select(-1)
        self.app.session.select(1)
        self.app.session.advance(0.02)
        self.app.update_view()

    def test_registry_handles_removal_addition_and_repeated_updates(self) -> None:
        old_root = self.app.aircraft_visuals.visuals[self.first.id].root
        self.world.aircraft.pop(self.first.id)
        new_aircraft = create_aircraft_in_world(self.world)
        self.app.update_view()
        self.assertTrue(old_root.isEmpty())
        self.assertEqual(
            set(self.app.aircraft_visuals.visuals), {self.second.id, new_aircraft.id}
        )
        new_root = self.app.aircraft_visuals.visuals[new_aircraft.id].root
        self.app.update_view()
        self.assertIs(self.app.aircraft_visuals.visuals[new_aircraft.id].root, new_root)

    def test_dead_aircraft_is_not_loaded_and_new_living_aircraft_restores_hud(
        self,
    ) -> None:
        self.first.death()
        self.second.death()
        self.app.update_view()
        dead_aircraft = create_aircraft_in_world(self.world)
        dead_aircraft.health = 0.0
        self.app.update_view()
        self.assertFalse(self.app.aircraft_visuals.visuals)
        living_aircraft = create_aircraft_in_world(self.world)
        self.app.update_view()
        self.assertEqual(set(self.app.aircraft_visuals.visuals), {living_aircraft.id})
        self.assertIs(self.app.aircraft_focus.current_aircraft, living_aircraft)
        self.assertNotIn("Aucun avion vivant", self.app.hud.status_text.getText())

    def test_bullet_kill_is_removed_on_next_view_update(self) -> None:
        for aircraft in (self.first, self.second):
            aircraft.physics.state.quaternion = euler_to_quaternion(0.0, 0.0, 0.0)
        self.second.health = 10.0
        root = self.app.aircraft_visuals.visuals[self.second.id].root
        self.app.session.set_fire_trigger(True)
        self.app.session.advance(0.08)
        self.app.update_view()
        self.assertEqual(self.second.health, 0.0)
        self.assertFalse(self.second.alive)
        self.assertTrue(root.isEmpty())
        self.assertNotIn(self.second.id, self.app.aircraft_visuals.visuals)


if __name__ == "__main__":
    unittest.main()

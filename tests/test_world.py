import unittest
from unittest.mock import patch

from physics.aircraft_physics import AircraftPhysics
from piloting.commands import AircraftAction, FlightCommand
from simulation.aircraft import AircraftEntity
from simulation.teams import Team
from simulation.world import SimulationWorld


class WorldTests(unittest.TestCase):
    def test_missing_action_retains_last_action_and_can_be_replaced(self):
        world = SimulationWorld()
        entity = AircraftEntity(Team.ALPHA, AircraftPhysics())
        world.add_aircraft(entity)
        action = AircraftAction(FlightCommand(0.5, -0.5, 0.2, 1.0), True)
        neutral = AircraftAction(FlightCommand(0.0, 0.0, 0.0, 0.0))
        with patch.object(entity.physics, "step", wraps=entity.physics.step) as step:
            world.step({entity.id: action}, 0.01)
            entity.weapon.cooldown = 0.0
            world.step({}, 0.01)
            self.assertEqual(step.call_args.args, (action.flight, 0.01))
            self.assertEqual(len(world.bullets), 2)
            world.step({entity.id: neutral}, 0.01)
            entity.weapon.cooldown = 0.0
            world.step({}, 0.01)
            self.assertEqual(step.call_args.args, (neutral.flight, 0.01))
            self.assertEqual(len(world.bullets), 2)
        self.assertEqual(entity.physics.elapsed_time, 0.04)

    def test_partial_actions_advance_all_living_aircraft(self):
        world = SimulationWorld()
        entities = [AircraftEntity(Team.ALPHA, AircraftPhysics()) for _ in range(3)]
        for entity in entities:
            world.add_aircraft(entity)
        entities[2].death()
        action = AircraftAction(FlightCommand(0.0, 0.0, 0.0, 1.0))
        world.step({entities[0].id: action}, 0.01)
        world.step({}, 0.01)
        self.assertEqual([e.physics.elapsed_time for e in entities], [0.02, 0.02, 0.0])
        self.assertEqual(entities[0].physics.last_controls.throttle_command, 1.0)
        self.assertEqual(entities[1].physics.last_controls.throttle_command, 0.0)

    def test_invalid_dt_does_not_replace_last_action(self):
        world = SimulationWorld()
        entity = AircraftEntity(Team.ALPHA, AircraftPhysics())
        world.add_aircraft(entity)
        action = AircraftAction(FlightCommand(0.0, 0.0, 0.0, 1.0))
        world.step({entity.id: action}, 0.01)
        with self.assertRaises(ValueError):
            world.step({entity.id: AircraftAction(FlightCommand(0, 0, 0, 0))}, 0)
        world.step({}, 0.01)
        self.assertEqual(entity.physics.last_controls.throttle_command, 1.0)

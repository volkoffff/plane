"""Physics package for the aircraft simulation."""

from physics.autopilot import FlightCommand, FlightInstruction
from physics.projectiles import Bullet
from physics.simulator import AircraftSimulation

__all__ = [
    "AircraftSimulation",
    "Bullet",
    "FlightCommand",
    "FlightInstruction",
]

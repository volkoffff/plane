"""Physics package for the aircraft simulation."""

from physics.autopilot import FlightCommand, FlightInstruction
from physics.simulator import AircraftSimulation

__all__ = [
    "AircraftSimulation",
    "FlightCommand",
    "FlightInstruction",
]

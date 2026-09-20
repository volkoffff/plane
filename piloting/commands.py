from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FlightCommand:
    mouse_dx: float
    mouse_dy: float
    rudder_input: float
    throttle_command: float
    target_yaw_rate: float = 0.0

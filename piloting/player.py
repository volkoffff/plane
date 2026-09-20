from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from piloting.commands import FlightCommand


@dataclass(frozen=True)
class MouseInput:
    x: float = 0.0
    y: float = 0.0


CONTROL_NAMES = (
    "pitch_down",
    "pitch_up",
    "roll_left",
    "roll_right",
    "rudder_left",
    "rudder_right",
    "throttle_down",
    "throttle_up",
)


def make_default_key_state() -> dict[str, bool]:
    return {key_name: False for key_name in CONTROL_NAMES}


@dataclass
class PlayerAircraftInputController:
    throttle_command: float
    throttle_rate: float = 0.55
    rudder_step: float = 0.55
    key_state: dict[str, bool] = field(default_factory=make_default_key_state)
    trigger_fire: bool = False

    def set_key_state(self, key_name: str, is_pressed: bool) -> None:
        self.key_state[key_name] = is_pressed

    def set_fire_trigger(self, is_pressed: bool) -> None:
        self.trigger_fire = is_pressed

    def build_flight_command(
        self,
        dt: float,
        mouse: MouseInput | None = None,
    ) -> FlightCommand:
        mouse = mouse or MouseInput()
        self._update_throttle(dt)

        keyboard_roll = self._axis("roll_right", "roll_left", amount=1.0)
        keyboard_pitch = self._axis("pitch_down", "pitch_up", amount=1.0)
        rudder_input = self._axis("rudder_left", "rudder_right", self.rudder_step)

        return FlightCommand(
            mouse_dx=float(np.clip(mouse.x + keyboard_roll, -1.0, 1.0)),
            mouse_dy=float(np.clip(mouse.y + keyboard_pitch, -1.0, 1.0)),
            rudder_input=rudder_input,
            throttle_command=self.throttle_command,
        )

    def _update_throttle(self, dt: float) -> None:
        throttle_axis = self._axis("throttle_up", "throttle_down", amount=1.0)
        self.throttle_command = float(
            np.clip(
                self.throttle_command + self.throttle_rate * dt * throttle_axis,
                0.0,
                1.5,
            )
        )

    def _axis(
        self,
        positive_key: str,
        negative_key: str,
        amount: float,
    ) -> float:
        value = 0.0
        if self.key_state.get(positive_key, False):
            value += amount
        if self.key_state.get(negative_key, False):
            value -= amount
        return float(value)

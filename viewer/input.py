from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from physics.autopilot import FlightCommand


@dataclass(frozen=True)
class MouseInput:
    x: float = 0.0
    y: float = 0.0


class EventBinder(Protocol):
    def __call__(
        self,
        event_name: str,
        method: Callable[..., None],
        extra_args: list[object],
    ) -> object:
        ...


KEY_BINDINGS = (
    ("control", "throttle_down", True),
    ("control-up", "throttle_down", False),
    ("lcontrol", "throttle_down", True),
    ("lcontrol-up", "throttle_down", False),
    ("rcontrol", "throttle_down", True),
    ("rcontrol-up", "throttle_down", False),
    ("shift", "throttle_up", True),
    ("shift-up", "throttle_up", False),
    ("lshift", "throttle_up", True),
    ("lshift-up", "throttle_up", False),
    ("rshift", "throttle_up", True),
    ("rshift-up", "throttle_up", False),
    ("q", "roll_left", True),
    ("q-up", "roll_left", False),
    ("d", "roll_right", True),
    ("d-up", "roll_right", False),
    ("z", "pitch_down", True),
    ("z-up", "pitch_down", False),
    ("s", "pitch_up", True),
    ("s-up", "pitch_up", False),
    ("a", "rudder_left", True),
    ("a-up", "rudder_left", False),
    ("e", "rudder_right", True),
    ("e-up", "rudder_right", False),
)
FIRE_BINDINGS = (
    ("mouse1", True),
    ("mouse1-up", False),
)
CONTROL_NAMES = tuple(sorted({key_name for _, key_name, _ in KEY_BINDINGS}))


def make_default_key_state() -> dict[str, bool]:
    return {key_name: False for key_name in CONTROL_NAMES}


@dataclass
class PlayerAircraftInputController:
    throttle_command: float
    throttle_rate: float = 0.55
    rudder_step: float = 0.55
    key_state: dict[str, bool] = field(default_factory=make_default_key_state)
    trigger_fire: bool = False

    def bind_events(self, accept: EventBinder) -> None:
        for event_name, key_name, is_pressed in KEY_BINDINGS:
            accept(event_name, self.set_key_state, [key_name, is_pressed])

        for event_name, is_pressed in FIRE_BINDINGS:
            accept(event_name, self.set_fire_trigger, [is_pressed])

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

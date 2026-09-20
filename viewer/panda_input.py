from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from piloting.player import MouseInput, PlayerAircraftInputController


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


def bind_player_controls(
    accept: EventBinder,
    controller: PlayerAircraftInputController,
) -> None:
    for event_name, key_name, is_pressed in KEY_BINDINGS:
        accept(event_name, controller.set_key_state, [key_name, is_pressed])

    for event_name, is_pressed in FIRE_BINDINGS:
        accept(event_name, controller.set_fire_trigger, [is_pressed])


def read_mouse_input(mouse_watcher_node) -> MouseInput:
    if mouse_watcher_node is not None and mouse_watcher_node.hasMouse():
        mouse = mouse_watcher_node.getMouse()
        return MouseInput(
            x=float(np.clip(mouse.getX(), -1.0, 1.0)),
            y=float(np.clip(-mouse.getY(), -1.0, 1.0)),
        )

    return MouseInput()

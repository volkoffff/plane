"""Simulation orchestration independent of Panda3D and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from piloting.commands import AircraftAction, FlightCommand
from piloting.player import MouseInput, PlayerAircraftInputController
from simulation.world import SimulationWorld
from viewer.aircraft_focus import AircraftFocusController


@dataclass
class FlightSession:
    world: SimulationWorld
    fixed_dt: float = 0.01
    max_substeps: int = 8
    paused: bool = False
    mouse_enabled: bool = True
    accumulator: float = field(default=0.0, init=False)
    elapsed_time: float = field(default=0.0, init=False)
    focus: AircraftFocusController = field(init=False)
    controllers: dict[int, PlayerAircraftInputController] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self) -> None:
        if not isfinite(self.fixed_dt) or self.fixed_dt <= 0 or self.max_substeps < 1:
            raise ValueError("A positive fixed step and substep limit are required")
        self.focus = AircraftFocusController(self.world)

    @property
    def player_controls(self) -> PlayerAircraftInputController | None:
        aircraft = self.focus.current_aircraft
        if aircraft is None or aircraft.is_dead():
            return None
        if aircraft.id not in self.controllers:
            self.controllers[aircraft.id] = PlayerAircraftInputController(
                throttle_command=float(aircraft.physics.state.throttle)
            )
        return self.controllers[aircraft.id]

    def release_inputs(self) -> None:
        for controller in self.controllers.values():
            controller.release_inputs()

    def synchronize_aircraft(self) -> None:
        """Discard inactive controllers and release inputs on automatic selection."""
        selection_changed = self.focus.refresh()
        living_ids = {
            aircraft.id for aircraft in self.world.get_all_aircrafts() if aircraft.alive
        }
        for aircraft_id in self.controllers.keys() - living_ids:
            self.controllers.pop(aircraft_id).release_inputs()
        if selection_changed:
            self.release_inputs()

    def select(self, offset: int) -> None:
        self.release_inputs()
        if offset < 0:
            self.focus.previous()
        else:
            self.focus.next()

    def set_key_state(self, key_name: str, pressed: bool) -> None:
        controller = self.player_controls
        if controller is not None:
            controller.set_key_state(key_name, pressed)

    def set_fire_trigger(self, pressed: bool) -> None:
        controller = self.player_controls
        if controller is not None:
            controller.set_fire_trigger(pressed)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.accumulator = 0.0
        self.release_inputs()

    def advance(self, frame_dt: float, mouse: MouseInput = MouseInput()) -> int:
        if not isfinite(frame_dt) or frame_dt < 0:
            raise ValueError("Frame duration must be finite and nonnegative")
        self.synchronize_aircraft()
        if self.paused:
            return 0
        # Bound catch-up work after a slow frame; retain the fractional step.
        self.accumulator += min(frame_dt, self.fixed_dt * self.max_substeps)
        steps = 0
        while self.accumulator + 1e-12 >= self.fixed_dt and steps < self.max_substeps:
            self.world.step(self.build_actions(mouse), self.fixed_dt)
            self.synchronize_aircraft()
            self.elapsed_time += self.fixed_dt
            self.accumulator = max(0.0, self.accumulator - self.fixed_dt)
            steps += 1
        return steps

    def build_actions(self, mouse: MouseInput) -> dict[int, AircraftAction]:
        actions = {}
        controller = self.player_controls
        for aircraft in self.world.get_all_aircrafts():
            if aircraft.is_dead():
                continue
            if aircraft.id == self.focus.current_aircraft_id and controller is not None:
                actions[aircraft.id] = controller.build_action(
                    self.fixed_dt, mouse if self.mouse_enabled else MouseInput()
                )
            else:
                throttle = self.controllers.get(aircraft.id)
                actions[aircraft.id] = AircraftAction(
                    FlightCommand(
                        0.0,
                        0.0,
                        0.0,
                        throttle.throttle_command
                        if throttle
                        else float(aircraft.physics.state.throttle),
                    )
                )
        return actions

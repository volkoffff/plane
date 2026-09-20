from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from physics.autopilot import FlightCommand, attitude_command, pitch_rate_command
from physics.state import AircraftState


@dataclass(frozen=True)
class FlightInstruction:
    name: str
    duration: float
    command: Callable[[AircraftState, float], FlightCommand]


def build_flight_program() -> list[FlightInstruction]:
    return [
        FlightInstruction(
            "avancer",
            4.0,
            lambda state, t: attitude_command(state, 0.0, 3.0, 0.90),
        ),
        FlightInstruction(
            "monter",
            6.0,
            lambda state, t: attitude_command(state, 0.0, 22.0, 1.15),
        ),
        FlightInstruction(
            "palier",
            3.0,
            lambda state, t: attitude_command(state, 0.0, 4.0, 0.95),
        ),
        FlightInstruction(
            "tourner",
            8.0,
            lambda state, t: attitude_command(
                state,
                42.0,
                8.0,
                1.00,
                coordinated_turn=True,
            ),
        ),
        FlightInstruction(
            "sortie virage",
            5.0,
            lambda state, t: attitude_command(state, 0.0, 3.0, 0.95),
        ),
        FlightInstruction(
            "prise vitesse",
            4.0,
            lambda state, t: attitude_command(state, 0.0, 0.0, 1.35),
        ),
        FlightInstruction(
            "looping",
            14.0,
            lambda state, t: pitch_rate_command(49.0, 1.45),
        ),
        FlightInstruction(
            "recuperation",
            8.0,
            lambda state, t: attitude_command(state, 0.0, 5.0, 1.05),
        ),
    ]


class FlightProgramRunner:
    def __init__(
        self,
        instructions: list[FlightInstruction] | None = None,
    ) -> None:
        self.instructions = instructions or build_flight_program()
        self.instruction_time = 0.0
        self.instruction_index = 0

    @property
    def is_finished(self) -> bool:
        return self.instruction_index >= len(self.instructions)

    @property
    def current_name(self) -> str:
        if self.is_finished:
            return "programme termine"

        return self.instructions[self.instruction_index].name

    def command(self, state: AircraftState, dt: float) -> FlightCommand | None:
        if self.is_finished:
            return None

        instruction = self.instructions[self.instruction_index]
        return instruction.command(state, self.instruction_time)

    def advance(self, dt: float) -> None:
        if self.is_finished:
            return

        instruction = self.instructions[self.instruction_index]
        self.instruction_time += dt

        if self.instruction_time + 1e-9 >= instruction.duration:
            self.instruction_index += 1
            self.instruction_time = 0.0

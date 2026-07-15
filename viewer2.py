from __future__ import annotations

import argparse
from enum import Enum

import numpy as np
from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Filename,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LMatrix4f,
    LPoint3f,
    LineSegs,
    TextNode,
    TransparencyAttrib,
    loadPrcFileData,
)

loadPrcFileData(
    "",
    "\n".join(
        [
            "window-title Plane - Panda3D",
            "show-frame-rate-meter true",
            "sync-video false",
            "framebuffer-multisample true",
            "multisamples 4",
            "textures-power-2 none",
        ]
    ),
)

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task

from model.aircraft import create_aircraft
from physics.autopilot import FlightCommand, build_flight_program
from physics.math3d import quaternion_to_euler, quaternion_to_matrix
from physics.simulator import AircraftSimulation
from physics.state import AircraftState


class ViewerMode(str, Enum):
    INSTRUCTIONS = "instructions"
    SIMULATION = "simulation"


NED_TO_PANDA = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
)
BODY_PANDA_TO_NED = NED_TO_PANDA
MODEL_PANDA_TO_BODY_PANDA = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
)


def ned_to_panda(position_ned: np.ndarray) -> np.ndarray:
    return NED_TO_PANDA @ position_ned


def aircraft_panda_rotation(state: AircraftState) -> np.ndarray:
    body_to_world_ned = quaternion_to_matrix(state.quaternion)
    return NED_TO_PANDA @ body_to_world_ned @ BODY_PANDA_TO_NED


def panda_matrix_from_rotation_translation(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> LMatrix4f:
    matrix = LMatrix4f(LMatrix4f.identMat())

    for input_axis in range(3):
        for output_axis in range(3):
            matrix.setCell(
                input_axis,
                output_axis,
                float(rotation[output_axis, input_axis]),
            )

    matrix.setCell(3, 0, float(translation[0]))
    matrix.setCell(3, 1, float(translation[1]))
    matrix.setCell(3, 2, float(translation[2]))

    return matrix


def make_ground(
    x_min: float = -6_000.0,
    x_max: float = 6_000.0,
    y_min: float = -2_000.0,
    y_max: float = 12_000.0,
) -> GeomNode:
    vertex_format = GeomVertexFormat.getV3n3c4()
    vertex_data = GeomVertexData("ground", vertex_format, Geom.UHStatic)

    vertex = GeomVertexWriter(vertex_data, "vertex")
    normal = GeomVertexWriter(vertex_data, "normal")
    color = GeomVertexWriter(vertex_data, "color")

    for point in (
        (x_min, y_min, 0.0),
        (x_max, y_min, 0.0),
        (x_max, y_max, 0.0),
        (x_min, y_max, 0.0),
    ):
        vertex.addData3f(*point)
        normal.addData3f(0.0, 0.0, 1.0)
        color.addData4f(0.58, 0.62, 0.62, 0.28)

    triangles = GeomTriangles(Geom.UHStatic)
    triangles.addVertices(0, 1, 2)
    triangles.addVertices(0, 2, 3)

    geom = Geom(vertex_data)
    geom.addPrimitive(triangles)

    node = GeomNode("ground")
    node.addGeom(geom)
    return node


def make_grid(
    x_min: int = -6_000,
    x_max: int = 6_000,
    y_min: int = -2_000,
    y_max: int = 12_000,
    step: int = 500,
) -> GeomNode:
    lines = LineSegs("grid")
    lines.setColor(0.38, 0.42, 0.42, 0.45)
    lines.setThickness(1.0)

    for x in range(x_min, x_max + step, step):
        lines.moveTo(x, y_min, 0.5)
        lines.drawTo(x, y_max, 0.5)

    for y in range(y_min, y_max + step, step):
        lines.moveTo(x_min, y, 0.5)
        lines.drawTo(x_max, y, 0.5)

    return lines.create()


def make_axes(length: float = 450.0) -> GeomNode:
    lines = LineSegs("axes")
    lines.setThickness(3.0)

    lines.setColor(0.85, 0.1, 0.1, 1.0)
    lines.moveTo(0.0, 0.0, 1.0)
    lines.drawTo(length, 0.0, 1.0)

    lines.setColor(0.1, 0.65, 0.18, 1.0)
    lines.moveTo(0.0, 0.0, 1.0)
    lines.drawTo(0.0, length, 1.0)

    lines.setColor(0.12, 0.28, 0.9, 1.0)
    lines.moveTo(0.0, 0.0, 1.0)
    lines.drawTo(0.0, 0.0, length)

    return lines.create()


class PandaFlightViewer(ShowBase):
    def __init__(
        self,
        window_type: str | None = None,
        mode: ViewerMode = ViewerMode.INSTRUCTIONS,
    ) -> None:
        if window_type is None:
            super().__init__()
        else:
            super().__init__(windowType=window_type)

        self.mode = mode
        self.disableMouse()
        self.accept("escape", self.userExit)
        self.accept("f1", self.set_mode, [ViewerMode.INSTRUCTIONS])
        self.accept("f2", self.set_mode, [ViewerMode.SIMULATION])

        self.simulation = AircraftSimulation()
        self.program = build_flight_program()
        self.manual_throttle_command = float(self.state.throttle)
        self.key_state = {
            "throttle_up": False,
            "throttle_down": False,
            "roll_left": False,
            "roll_right": False,
            "pitch_up": False,
            "pitch_down": False,
            "rudder_left": False,
            "rudder_right": False,
        }

        self.accumulator = 0.0
        self.instruction_time = 0.0
        self.instruction_index = 0

        self.trajectory_points: list[np.ndarray] = [ned_to_panda(self.state.position)]
        self.trajectory_node = None
        self.last_trajectory_time = 0.0

        self.setup_rendering()
        self.setup_scene()
        self.setup_aircraft()
        self.setup_controls()
        self.setup_hud()
        self.update_visuals()

        self.taskMgr.add(self.update_task, "update-flight")

    def set_mode(self, mode: ViewerMode) -> None:
        self.mode = mode

    @property
    def state(self) -> AircraftState:
        return self.simulation.state

    @property
    def air_data(self) -> dict:
        return self.simulation.air_data

    @property
    def elapsed_time(self) -> float:
        return self.simulation.elapsed_time

    @property
    def physics_dt(self) -> float:
        return self.simulation.dt

    def setup_controls(self) -> None:
        key_bindings = {
            "control": ("throttle_down", True),
            "control-up": ("throttle_down", False),
            "lcontrol": ("throttle_down", True),
            "lcontrol-up": ("throttle_down", False),
            "rcontrol": ("throttle_down", True),
            "rcontrol-up": ("throttle_down", False),
            "shift": ("throttle_up", True),
            "shift-up": ("throttle_up", False),
            "lshift": ("throttle_up", True),
            "lshift-up": ("throttle_up", False),
            "rshift": ("throttle_up", True),
            "rshift-up": ("throttle_up", False),
            "q": ("roll_left", True),
            "q-up": ("roll_left", False),
            "d": ("roll_right", True),
            "d-up": ("roll_right", False),
            "z": ("pitch_down", True),
            "z-up": ("pitch_down", False),
            "s": ("pitch_up", True),
            "s-up": ("pitch_up", False),
            "a": ("rudder_left", True),
            "a-up": ("rudder_left", False),
            "e": ("rudder_right", True),
            "e-up": ("rudder_right", False),
        }

        for event_name, (key_name, is_pressed) in key_bindings.items():
            self.accept(event_name, self.set_key_state, [key_name, is_pressed])

    def set_key_state(self, key_name: str, is_pressed: bool) -> None:
        self.key_state[key_name] = is_pressed

    def setup_rendering(self) -> None:
        self.setBackgroundColor(0.72, 0.84, 0.96, 1.0)

        if self.camLens is not None:
            self.camLens.setFov(62.0)
            self.camLens.setNearFar(0.5, 40_000.0)

        if self.win is not None:
            try:
                import simplepbr

                simplepbr.init()
            except ImportError:
                pass

    def setup_scene(self) -> None:
        ground = self.render.attachNewNode(make_ground())
        ground.setTransparency(TransparencyAttrib.MAlpha)
        ground.setTwoSided(True)

        self.render.attachNewNode(make_grid())
        self.render.attachNewNode(make_axes())

        ambient = AmbientLight("ambient")
        ambient.setColor((0.38, 0.40, 0.43, 1.0))
        self.render.setLight(self.render.attachNewNode(ambient))

        sun = DirectionalLight("sun")
        sun.setColor((1.0, 0.96, 0.88, 1.0))
        sun_np = self.render.attachNewNode(sun)
        sun_np.setHpr(-35.0, -45.0, 0.0)
        self.render.setLight(sun_np)

    def setup_aircraft(self) -> None:
        try:
            import gltf  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Le chargeur GLB Panda3D est manquant. Installe panda3d-gltf."
            ) from exc

        model_path = Filename.fromOsSpecific(str(create_aircraft()))
        aircraft_model = self.loader.loadModel(model_path)

        if aircraft_model.isEmpty():
            raise RuntimeError(f"Modele 3D introuvable : {model_path}")

        self.aircraft_root = self.render.attachNewNode("aircraft-root")
        aircraft_visual = self.aircraft_root.attachNewNode("aircraft-visual")
        aircraft_visual.setMat(
            panda_matrix_from_rotation_translation(
                MODEL_PANDA_TO_BODY_PANDA,
                np.zeros(3),
            )
        )

        aircraft_model.reparentTo(aircraft_visual)
        aircraft_model.setPos(0.0, 0.0, 0.0)

        bounds = aircraft_model.getTightBounds(aircraft_visual)
        if bounds is None:
            model_center = LPoint3f(0.0, 0.0, 0.0)
        else:
            model_center = (bounds[0] + bounds[1]) * 0.5

        aircraft_model.setPos(
            -model_center.x,
            -model_center.y,
            -model_center.z,
        )

    def setup_hud(self) -> None:
        self.status_text = OnscreenText(
            text="",
            parent=self.a2dTopLeft,
            pos=(0.04, -0.07),
            align=TextNode.ALeft,
            scale=0.042,
            fg=(0.02, 0.03, 0.04, 1.0),
            mayChange=True,
        )

    def update_task(self, task: Task.Task) -> int:
        frame_dt = min(globalClock.getDt(), 0.05)
        self.accumulator += frame_dt

        substeps = 0
        while self.accumulator >= self.physics_dt and substeps < 8:
            if not self.step_physics():
                self.accumulator = 0.0
                break

            self.accumulator -= self.physics_dt
            substeps += 1

        if substeps == 8:
            self.accumulator = 0.0

        self.update_visuals()
        return Task.cont

    def step_physics(self) -> bool:
        if self.mode == ViewerMode.INSTRUCTIONS:
            command = self.get_instruction_command()
            if command is None:
                return False
        else:
            command = self.get_manual_command()

        self.simulation.step(command)

        if self.mode == ViewerMode.INSTRUCTIONS:
            self.advance_instruction()

        return True

    def get_instruction_command(self) -> FlightCommand | None:
        if self.instruction_index >= len(self.program):
            return None

        instruction = self.program[self.instruction_index]
        return instruction.command(self.state, self.instruction_time)

    def advance_instruction(self) -> None:
        instruction = self.program[self.instruction_index]
        self.instruction_time += self.physics_dt

        if self.instruction_time + 1e-9 >= instruction.duration:
            self.instruction_index += 1
            self.instruction_time = 0.0

    def get_manual_command(self) -> FlightCommand:
        throttle_delta = 0.0
        if self.key_state["throttle_up"]:
            throttle_delta += 0.55 * self.physics_dt
        if self.key_state["throttle_down"]:
            throttle_delta -= 0.55 * self.physics_dt

        self.manual_throttle_command = float(
            np.clip(self.manual_throttle_command + throttle_delta, 0.0, 1.5)
        )

        rudder_input = 0.0
        if self.key_state["rudder_left"]:
            rudder_input += 0.55
        if self.key_state["rudder_right"]:
            rudder_input -= 0.55

        mouse_dx = 0.0
        mouse_dy = 0.0
        if self.mouseWatcherNode.hasMouse():
            mouse = self.mouseWatcherNode.getMouse()
            mouse_dx = float(np.clip(mouse.getX(), -1.0, 1.0))
            mouse_dy = float(np.clip(-mouse.getY(), -1.0, 1.0))

        keyboard_roll = 0.0
        if self.key_state["roll_left"]:
            keyboard_roll -= 1.0
        if self.key_state["roll_right"]:
            keyboard_roll += 1.0

        keyboard_pitch = 0.0
        if self.key_state["pitch_up"]:
            keyboard_pitch -= 1.0
        if self.key_state["pitch_down"]:
            keyboard_pitch += 1.0

        return FlightCommand(
            mouse_dx=float(np.clip(mouse_dx + keyboard_roll, -1.0, 1.0)),
            mouse_dy=float(np.clip(mouse_dy + keyboard_pitch, -1.0, 1.0)),
            rudder_input=rudder_input,
            throttle_command=self.manual_throttle_command,
        )

    def update_visuals(self) -> None:
        position = ned_to_panda(self.state.position)
        rotation = aircraft_panda_rotation(self.state)

        self.aircraft_root.setMat(
            panda_matrix_from_rotation_translation(rotation, position)
        )

        self.update_trajectory(position)
        self.update_camera(position, rotation)
        self.update_hud()

    def update_trajectory(self, position: np.ndarray) -> None:
        if self.elapsed_time - self.last_trajectory_time < 0.08:
            return

        self.last_trajectory_time = self.elapsed_time
        self.trajectory_points.append(position.copy())

        if len(self.trajectory_points) < 2:
            return

        if self.trajectory_node is not None:
            self.trajectory_node.removeNode()

        lines = LineSegs("trajectory")
        lines.setColor(1.0, 0.55, 0.05, 1.0)
        lines.setThickness(2.5)

        first = self.trajectory_points[0]
        lines.moveTo(*first)
        for point in self.trajectory_points[1:]:
            lines.drawTo(*point)

        self.trajectory_node = self.render.attachNewNode(lines.create())

    def update_camera(
        self,
        aircraft_position: np.ndarray,
        aircraft_rotation: np.ndarray,
    ) -> None:
        if self.camera is None or self.camera.isEmpty():
            return

        forward = aircraft_rotation @ np.array([0.0, 1.0, 0.0])
        world_up = np.array([0.0, 0.0, 1.0])

        level_forward = forward - np.dot(forward, world_up) * world_up
        if np.linalg.norm(level_forward) < 1e-6:
            level_forward = np.array([0.0, 1.0, 0.0])
        else:
            level_forward = level_forward / np.linalg.norm(level_forward)

        right = np.cross(level_forward, world_up)

        camera_position = (
            aircraft_position
            - 60.0 * forward
            - 0.0 * right
            + 18.0 * world_up
        )
        focal_point = aircraft_position + 70.0 * forward

        self.camera.setPos(*camera_position)
        self.camera.lookAt(LPoint3f(*focal_point))

    def update_hud(self) -> None:
        if self.mode == ViewerMode.SIMULATION:
            instruction_name = "simulation manuelle"
        elif self.instruction_index < len(self.program):
            instruction_name = self.program[self.instruction_index].name
        else:
            instruction_name = "programme termine"

        speed = float(self.air_data["speed"])
        altitude = float(-self.state.position[2])
        roll, pitch, yaw = quaternion_to_euler(self.state.quaternion)

        self.status_text.setText(
            "\n".join(
                [
                    f"Mode : {self.mode.value}",
                    f"Instruction : {instruction_name}",
                    f"t = {self.elapsed_time:5.1f} s",
                    f"h = {altitude:7.0f} m",
                    f"V = {speed:6.1f} m/s",
                    f"roll = {np.rad2deg(roll):6.1f} deg",
                    f"pitch = {np.rad2deg(pitch):6.1f} deg",
                    f"yaw = {np.rad2deg(yaw):6.1f} deg",
                    f"thr = {self.state.throttle:4.2f}",
                    f"cmd gaz = {self.manual_throttle_command:4.2f}",
                    "F1 instructions | F2 simulation",
                    "Souris ou Q/D: roulis | S/Z: pitch",
                    "Maj/Ctrl: gaz | A/E: lacet",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lance le viewer avion Panda3D."
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in ViewerMode],
        default=ViewerMode.INSTRUCTIONS.value,
        help="Mode de pilotage au lancement.",
    )
    return parser.parse_args()


def run_viewer(mode: ViewerMode = ViewerMode.INSTRUCTIONS) -> None:
    app = PandaFlightViewer(mode=mode)
    app.run()


if __name__ == "__main__":
    args = parse_args()
    run_viewer(ViewerMode(args.mode))

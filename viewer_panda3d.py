from collections import deque
from pathlib import Path

import gltf
import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    LineSegs,
    NodePath,
    TextNode,
    Vec3,
    WindowProperties,
    loadPrcFileData,
)

import aircraft
import simulation

SIM_DT = 0.01
MAX_STEPS_PER_FRAME = 5
TRAIL_MAX_POINTS = 1200
TRAIL_REFRESH_SECONDS = 0.05
GRID_SIZE = 2000
GRID_STEP = 100
AIRCRAFT_MODEL_HPR_OFFSET = (90.0, 0.0, 0.0)
DEBUG_AIRCRAFT_HPR_OFFSET = (0.0, 0.0, 0.0)
USE_DEBUG_AIRCRAFT_SHAPE = True

_PANDA3D_CONFIGURED = False


def configure_panda3d(window_type: str | None = None) -> None:
    global _PANDA3D_CONFIGURED

    if _PANDA3D_CONFIGURED:
        return

    loadPrcFileData("", "window-title Plane Panda3D Viewer")
    loadPrcFileData("", "sync-video false")
    loadPrcFileData("", "show-frame-rate-meter true")
    loadPrcFileData("", "textures-power-2 none")

    if window_type is not None:
        loadPrcFileData("", f"window-type {window_type}")

    _PANDA3D_CONFIGURED = True


def np_position_to_vec3(position: np.ndarray) -> Vec3:
    return Vec3(float(position[0]), float(position[1]), float(position[2]))


def np_direction_to_vec3(direction: np.ndarray) -> Vec3:
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return Vec3(0.0, 1.0, 0.0)

    return Vec3(
        float(direction[0] / norm),
        float(direction[1] / norm),
        float(direction[2] / norm),
    )


def tight_bounds_center(model: NodePath) -> Vec3:
    min_point, max_point = model.getTightBounds()
    if min_point is None or max_point is None:
        return Vec3(0.0, 0.0, 0.0)

    return Vec3(
        (min_point.getX() + max_point.getX()) * 0.5,
        (min_point.getY() + max_point.getY()) * 0.5,
        (min_point.getZ() + max_point.getZ()) * 0.5,
    )


def load_aircraft_model(model_path: Path) -> NodePath:
    settings = gltf.GltfSettings(skip_axis_conversion=True)
    model_root = gltf.load_model(str(model_path), settings)
    return NodePath(model_root)


def create_debug_aircraft_shape(loader) -> NodePath:
    root = NodePath("debug-aircraft")

    lines = LineSegs("debug-aircraft-lines")
    lines.setThickness(6.0)

    # Axe rouge: nez de l'avion. Panda3D oriente le +Y local avec lookAt().
    lines.setColor(1.0, 0.05, 0.05, 1.0)
    lines.moveTo(0.0, -8.0, 0.0)
    lines.drawTo(0.0, 10.0, 0.0)
    lines.drawTo(1.6, 7.0, 0.0)
    lines.moveTo(0.0, 10.0, 0.0)
    lines.drawTo(-1.6, 7.0, 0.0)

    # Axe vert: ailes droite/gauche.
    lines.setColor(0.05, 0.9, 0.15, 1.0)
    lines.moveTo(-7.0, -1.0, 0.0)
    lines.drawTo(7.0, -1.0, 0.0)

    # Axe bleu: haut de l'avion.
    lines.setColor(0.1, 0.35, 1.0, 1.0)
    lines.moveTo(0.0, -6.0, 0.0)
    lines.drawTo(0.0, -6.0, 3.5)

    root.attachNewNode(lines.create())
    return root


def create_grid(parent: NodePath) -> NodePath:
    lines = LineSegs("ground-grid")
    lines.setThickness(1.0)
    lines.setColor(0.35, 0.35, 0.35, 1.0)

    for coord in range(-GRID_SIZE, GRID_SIZE + 1, GRID_STEP):
        lines.moveTo(-GRID_SIZE, coord, 0.0)
        lines.drawTo(GRID_SIZE, coord, 0.0)
        lines.moveTo(coord, -GRID_SIZE, 0.0)
        lines.drawTo(coord, GRID_SIZE, 0.0)

    return parent.attachNewNode(lines.create())


def create_axes(parent: NodePath) -> NodePath:
    axis_length = 150.0
    lines = LineSegs("axes")
    lines.setThickness(3.0)

    lines.setColor(1.0, 0.15, 0.15, 1.0)
    lines.moveTo(0.0, 0.0, 0.0)
    lines.drawTo(axis_length, 0.0, 0.0)

    lines.setColor(0.1, 0.75, 0.25, 1.0)
    lines.moveTo(0.0, 0.0, 0.0)
    lines.drawTo(0.0, axis_length, 0.0)

    lines.setColor(0.15, 0.3, 1.0, 1.0)
    lines.moveTo(0.0, 0.0, 0.0)
    lines.drawTo(0.0, 0.0, axis_length)

    return parent.attachNewNode(lines.create())


class PlanePandaViewer(ShowBase):
    def __init__(self) -> None:
        super().__init__()

        self.disableMouse()
        self.setBackgroundColor(0.72, 0.84, 0.92, 1.0)
        if self.camLens is not None:
            self.camLens.setFov(60)
            self.camLens.setNearFar(0.1, 100000.0)

        if self.win is not None:
            window_properties = WindowProperties()
            window_properties.setSize(1280, 720)
            self.win.requestProperties(window_properties)

        self.aircraft_root = self.render.attachNewNode("aircraft-root")
        if USE_DEBUG_AIRCRAFT_SHAPE:
            self.model_path = None
            self.aircraft_model = create_debug_aircraft_shape(self.loader)
            self.aircraft_model.reparentTo(self.aircraft_root)
            self.aircraft_model.setHpr(*DEBUG_AIRCRAFT_HPR_OFFSET)
        else:
            self.model_path = aircraft.create_aircraft()
            self.aircraft_model = load_aircraft_model(self.model_path)
            self.aircraft_model.reparentTo(self.aircraft_root)

            aircraft_center = tight_bounds_center(self.aircraft_model)
            self.aircraft_model.setPos(
                -aircraft_center.getX(),
                -aircraft_center.getY(),
                -aircraft_center.getZ(),
            )
            self.aircraft_model.setHpr(*AIRCRAFT_MODEL_HPR_OFFSET)

        self.aircraft_model.setTwoSided(True)

        self.state = simulation.initial_state()
        self.aircraft_params = simulation.rafale_like_parameters()
        self.trajectory = deque(maxlen=TRAIL_MAX_POINTS)
        self.trail_node: NodePath | None = None
        self.elapsed_time = 0.0
        self.accumulator = 0.0
        self.trail_refresh_elapsed = 0.0
        self.paused = False
        self.last_elevator_control = 0.0
        self.last_aileron_control = 0.0
        self.last_rudder_control = 0.0
        self.last_throttle_command = 0.9

        self.telemetry = OnscreenText(
            text="",
            pos=(-1.28, 0.92),
            align=TextNode.ALeft,
            scale=0.045,
            fg=(0.04, 0.05, 0.06, 1.0),
            mayChange=True,
        )

        self.setup_lights()
        create_grid(self.render)
        create_axes(self.render)
        self.reset_simulation()

        self.accept("escape", self.userExit)
        self.accept("space", self.toggle_pause)
        self.accept("r", self.reset_simulation)
        self.taskMgr.add(self.update_viewer, "update-viewer")

    def setup_lights(self) -> None:
        ambient = AmbientLight("ambient-light")
        ambient.setColor((0.45, 0.45, 0.5, 1.0))
        ambient_node = self.render.attachNewNode(ambient)
        self.render.setLight(ambient_node)

        sun = DirectionalLight("sun-light")
        sun.setColor((0.9, 0.88, 0.8, 1.0))
        sun_node = self.render.attachNewNode(sun)
        sun_node.setHpr(-45.0, -35.0, 0.0)
        self.render.setLight(sun_node)

    def reset_simulation(self) -> None:
        self.state = simulation.initial_state()
        self.trajectory.clear()
        self.trajectory.append(self.state.position.copy())
        self.elapsed_time = 0.0
        self.accumulator = 0.0
        self.trail_refresh_elapsed = 0.0
        self.last_elevator_control = 0.0
        self.last_aileron_control = 0.0
        self.last_rudder_control = 0.0
        self.last_throttle_command = 0.9
        self.update_aircraft_transform()
        self.update_trail()
        self.update_telemetry()
        self.update_camera()

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def scripted_controls(self) -> tuple[float, float, float, float]:
        time = self.elapsed_time
        throttle_command = 0.9
        elevator_control = 0.0
        aileron_control = 0.0
        rudder_control = 0.0

        if 2.0 <= time < 6.0:
            elevator_control = 0.18
        elif 6.0 <= time < 10.0:
            aileron_control = -0.22
            rudder_control = 0.04
        elif 10.0 <= time < 14.0:
            aileron_control = 0.18
            rudder_control = -0.03
        elif 14.0 <= time < 18.0:
            elevator_control = -0.08

        return (
            elevator_control,
            aileron_control,
            rudder_control,
            throttle_command,
        )

    def step_simulation(self) -> None:
        (
            self.last_elevator_control,
            self.last_aileron_control,
            self.last_rudder_control,
            self.last_throttle_command,
        ) = self.scripted_controls()
        self.state = simulation.update_state(
            self.state,
            self.aircraft_params,
            self.last_elevator_control,
            self.last_throttle_command,
            SIM_DT,
            self.last_aileron_control,
            self.last_rudder_control,
        )
        self.trajectory.append(self.state.position.copy())
        self.elapsed_time += SIM_DT

    def update_aircraft_transform(self) -> None:
        position = np_position_to_vec3(self.state.position)
        forward, _, up = simulation.body_axes(self.state)

        self.aircraft_root.setPos(position)
        self.aircraft_root.lookAt(
            position + np_direction_to_vec3(forward),
            np_direction_to_vec3(up),
        )

    def update_trail(self) -> None:
        if self.trail_node is not None:
            self.trail_node.removeNode()

        lines = LineSegs("trajectory")
        lines.setThickness(2.0)
        lines.setColor(0.0, 0.55, 1.0, 1.0)

        first_point = True
        for point in self.trajectory:
            panda_point = np_position_to_vec3(point)
            if first_point:
                lines.moveTo(panda_point)
                first_point = False
            else:
                lines.drawTo(panda_point)

        self.trail_node = self.render.attachNewNode(lines.create())

    def update_telemetry(self) -> None:
        speed = float(np.linalg.norm(self.state.velocity))
        roll_degrees = float(np.rad2deg(self.state.roll))
        pitch_degrees = float(np.rad2deg(self.state.pitch))
        yaw_degrees = float(np.rad2deg(self.state.yaw))
        alpha_degrees = float(np.rad2deg(simulation.angle_of_attack(self.state)))
        beta_degrees = float(np.rad2deg(simulation.sideslip_angle(self.state)))

        self.telemetry.setText(
            f"Speed     {speed:7.1f} m/s\n"
            f"Altitude  {self.state.position[2]:7.1f} m\n"
            f"Roll      {roll_degrees:7.2f} deg\n"
            f"Pitch     {pitch_degrees:7.2f} deg\n"
            f"Yaw       {yaw_degrees:7.2f} deg\n"
            f"Alpha     {alpha_degrees:7.2f} deg\n"
            f"Beta      {beta_degrees:7.2f} deg\n"
            f"Throttle  {self.state.throttle:7.2f}\n"
            f"Elevator  {self.last_elevator_control:7.2f}\n"
            f"Aileron   {self.last_aileron_control:7.2f}\n"
            f"Rudder    {self.last_rudder_control:7.2f}"
        )

    def update_camera(self) -> None:
        if self.camera is None:
            return

        target = np_position_to_vec3(self.state.position)
        forward, _, _ = simulation.body_axes(self.state)
        aircraft_forward = np_direction_to_vec3(forward)
        world_up = Vec3(0.0, 0.0, 1.0)
        camera_position = target - aircraft_forward * 150.0 + world_up * 45.0
        look_at = target + aircraft_forward * 85.0 + world_up * 8.0

        self.camera.setPos(camera_position)
        self.camera.lookAt(look_at)

    def update_viewer(self, task):
        if not self.paused:
            frame_dt = min(ClockObject.getGlobalClock().getDt(), 0.1)
            self.accumulator += frame_dt

            steps = 0
            while self.accumulator >= SIM_DT and steps < MAX_STEPS_PER_FRAME:
                self.step_simulation()
                self.accumulator -= SIM_DT
                self.trail_refresh_elapsed += SIM_DT
                steps += 1

            if steps == MAX_STEPS_PER_FRAME:
                self.accumulator = 0.0

            self.update_aircraft_transform()

            if self.trail_refresh_elapsed >= TRAIL_REFRESH_SECONDS:
                self.update_trail()
                self.trail_refresh_elapsed = 0.0

            self.update_telemetry()

        self.update_camera()
        return task.cont


def run_viewer() -> None:
    configure_panda3d()
    viewer = PlanePandaViewer()
    viewer.run()


if __name__ == "__main__":
    run_viewer()

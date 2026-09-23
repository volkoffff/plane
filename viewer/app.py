from __future__ import annotations

from panda3d.core import loadPrcFileData

# Configuration Panda3D appliquee avant la creation de la fenetre.
loadPrcFileData(
    "",
    "window-title Plane - Panda3D\nshow-frame-rate-meter true\nsync-video false\nframebuffer-multisample true\nmultisamples 4\ntextures-power-2 none",
)

from direct.showbase.ShowBase import ShowBase
from direct.showbase.ShowBaseGlobal import globalClock
from direct.task import Task

from physics.state import AircraftState
from piloting.commands import AircraftAction
from piloting.player import MouseInput, PlayerAircraftInputController
from simulation.scenarios import create_player_world
from viewer.aircraft_visual import AircraftVisual
from viewer.camera import ChaseCamera
from viewer.hud import FlightHud
from viewer.panda_input import bind_player_controls
from viewer.panda_input import read_mouse_input as read_panda_mouse_input
from viewer.projectiles import ProjectileRenderer
from viewer.scene import SceneRenderer
from viewer.trajectory import TrajectoryRenderer
from viewer.transforms import ned_to_panda


class PandaFlightViewer(ShowBase):
    def __init__(
        self,
        window_type: str | None = None,
        animations_enabled: bool = False,
    ) -> None:
        if window_type is None:
            super().__init__()
        else:
            super().__init__(windowType=window_type)

        self.animations_enabled = animations_enabled
        self.disableMouse()
        self.accept("escape", self.userExit)

        self.world, self.player_aircraft = create_player_world()
        self.player_controls = PlayerAircraftInputController(
            throttle_command=float(self.state.throttle),
        )
        bind_player_controls(self.accept, self.player_controls)

        self.accumulator = 0.0

        self.setup_rendering()
        self.scene = SceneRenderer(self.render)
        self.scene.setup()

        self.aircraft_visual = AircraftVisual(
            self.loader,
            self.render,
            animations_enabled=self.animations_enabled,
        )
        self.aircraft_visual.setup()

        self.projectile_renderer = ProjectileRenderer(self.render)
        self.trajectory_renderer = TrajectoryRenderer(
            self.render,
            ned_to_panda(self.state.position),
        )
        self.camera_controller = ChaseCamera(self.camera)
        self.hud = FlightHud(self.aspect2d, self.a2dTopLeft)

        self.update_visuals()
        self.taskMgr.add(self.update_task, "update-flight")

    @property
    def state(self) -> AircraftState:
        return self.player_aircraft.physics.state

    @property
    def air_data(self) -> dict:
        return self.player_aircraft.physics.air_data

    @property
    def elapsed_time(self) -> float:
        return self.player_aircraft.physics.elapsed_time

    @property
    def physics_dt(self) -> float:
        return self.player_aircraft.physics.dt

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

    def update_task(self, task: Task.Task) -> int:
        frame_dt = min(globalClock.getDt(), 0.05)
        self.accumulator += frame_dt

        substeps = 0
        while self.accumulator >= self.physics_dt and substeps < 8:
            self.step_world()

            self.accumulator -= self.physics_dt
            substeps += 1

        if substeps == 8:
            self.accumulator = 0.0

        self.update_visuals()
        return Task.cont

    def get_aircraft_action(self) -> AircraftAction:
        return self.player_controls.build_action(
            self.physics_dt,
            self.read_mouse_input(),
        )

    def step_world(self) -> None:
        action = self.get_aircraft_action()

        self.world.step({self.player_aircraft.id: action}, self.physics_dt)

    def read_mouse_input(self) -> MouseInput:
        return read_panda_mouse_input(self.mouseWatcherNode)

    def update_visuals(self) -> None:
        position, rotation = self.aircraft_visual.update_pose(self.state)
        self.aircraft_visual.update_animations(
            self.player_aircraft.physics.last_controls,
            self.state.throttle,
            float(self.air_data.get("speed", 0.0)),
            self.elapsed_time,
        )

        self.projectile_renderer.update(self.world.bullets)
        self.trajectory_renderer.update(position, self.elapsed_time)
        self.camera_controller.update(position, rotation)
        self.hud.update(
            elapsed_time=self.elapsed_time,
            state=self.state,
            air_data=self.air_data,
            throttle_command=self.player_controls.throttle_command,
            bullet_count=len(self.world.bullets),
            camera=self.camera,
            cam_lens=self.camLens,
            render=self.render,
            aspect_ratio=float(self.getAspectRatio()),
            aircraft_position=position,
            aircraft_rotation=rotation,
        )


def run_viewer(
    animations_enabled: bool = False,
) -> None:
    app = PandaFlightViewer(
        animations_enabled=animations_enabled,
    )
    app.run()

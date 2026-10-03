from __future__ import annotations

from panda3d.core import ClockObject, loadPrcFileData

from simulation.scenarios import create_1_vs_1_world
from simulation.world import SimulationWorld

# Configuration Panda3D appliquee avant la creation de ShowBase.
loadPrcFileData(
    "",
    "window-title Plane - Scene Viewer\n"
    "show-frame-rate-meter true\n"
    "sync-video false\n"
    "framebuffer-multisample true\n"
    "multisamples 4\n"
    "textures-power-2 none",
)

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

from viewer.aircraft_visuals import AircraftVisualRegistry
from viewer.camera import ChaseCamera
from viewer.hud import FlightHud
from viewer.panda_input import bind_player_controls, read_mouse_input
from viewer.projectiles import ProjectileRenderer
from viewer.scene import SceneRenderer
from viewer.session import FlightSession


class PandaSceneViewer(ShowBase):
    def __init__(
        self,
        window_type: str | None = None,
        animations_enabled: bool = True,
    ) -> None:
        if window_type is None:
            super().__init__()
        else:
            super().__init__(windowType=window_type)

        # window setup
        self.disableMouse()
        self.accept("escape", self.userExit)

        # set caméra and background
        self.setup_rendering()

        # create the psysical world
        self.world = SimulationWorld()

        # implement 2 planes for fight in the world
        create_1_vs_1_world(self.world)

        self.scene = SceneRenderer(self.render)
        self.scene.setup()

        self.aircraft_visuals = AircraftVisualRegistry(
            self.loader,
            self.render,
            animations_enabled=animations_enabled,
        )
        self.aircraft_visuals.setup(self.world.get_all_aircrafts())

        self.session = FlightSession(self.world)
        self.aircraft_focus = self.session.focus
        self.camera_controller = ChaseCamera(self.camera)
        self.hud = FlightHud(self.aspect2d, self.a2dTopLeft)
        self.projectiles = ProjectileRenderer(self.render)
        bind_player_controls(self.accept, self.session)
        self.accept("f1", self.session.select, [-1])
        self.accept("f2", self.session.select, [1])
        self.accept("p", self.session.toggle_pause)
        self.accept("m", self.toggle_mouse)
        self.accept("f3", self.toggle_animations)
        self.accept("window-event", self.on_window_event)

        self.update_view(animation_time=0.0)
        self.taskMgr.add(self.update_task, "update-app2-view")

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
        dt = ClockObject.getGlobalClock().getDt()
        self.session.advance(dt, read_mouse_input(self.mouseWatcherNode))
        self.update_view(animation_time=self.session.elapsed_time)
        return Task.cont

    def on_window_event(self, window) -> None:
        if window == self.win and not window.getProperties().getForeground():
            self.session.release_inputs()
            self.session.mouse_enabled = False

    def toggle_mouse(self) -> None:
        self.session.mouse_enabled = not self.session.mouse_enabled

    def toggle_animations(self) -> None:
        self.aircraft_visuals.set_animations_enabled(
            not self.aircraft_visuals.animations_enabled
        )

    def update_view(self, animation_time: float = 0.0) -> None:
        aircraft_list = self.world.get_all_aircrafts()
        poses = self.aircraft_visuals.update(
            aircraft_list,
            animation_time=animation_time,
        )
        self.projectiles.update(self.world.bullets)

        focused_aircraft = self.aircraft_focus.current_aircraft
        if focused_aircraft is None:
            return

        pose = poses.get(focused_aircraft.id)
        if pose is None:
            return

        position, rotation = pose
        self.camera_controller.update(position, rotation)
        physics = focused_aircraft.physics
        controller = self.session.player_controls
        self.hud.update(
            elapsed_time=self.session.elapsed_time,
            state=physics.state,
            air_data=physics.air_data,
            throttle_command=controller.throttle_command if controller else physics.state.throttle,
            bullet_count=len(self.world.bullets),
            camera=self.camera, cam_lens=self.camLens, render=self.render,
            aspect_ratio=float(self.getAspectRatio()),
            aircraft_position=position, aircraft_rotation=rotation,
        )
        self.hud.status_text.appendText(
            f"\nAvion {focused_aircraft.id} | {focused_aircraft.team.name}"
            f" | vie {focused_aircraft.health:.0f}"
            f"\n{'PAUSE' if self.session.paused else 'VOL'}"
            f" | souris {'ON' if self.session.mouse_enabled else 'OFF'}"
            "\nF1/F2: avion | P: pause | M: souris | F3: animations"
        )


def run_scene_viewer() -> None:
    app = PandaSceneViewer()
    app.run()


if __name__ == "__main__":
    run_scene_viewer()

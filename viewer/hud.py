from __future__ import annotations

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import LPoint2f, LPoint3f, TextNode

from model.aircraft_animations import AircraftAnimationController
from physics.math3d import quaternion_to_euler
from physics.state import AircraftState


class FlightHud:
    def __init__(self, aspect2d, a2d_top_left) -> None:
        self.crosshair_text = OnscreenText(
            text="+",
            parent=aspect2d,
            pos=(0.0, 0.0),
            align=TextNode.ACenter,
            scale=0.075,
            fg=(1.0, 0.88, 0.18, 0.9),
            mayChange=True,
        )
        self.status_text = OnscreenText(
            text="",
            parent=a2d_top_left,
            pos=(0.04, -0.07),
            align=TextNode.ALeft,
            scale=0.042,
            fg=(0.02, 0.03, 0.04, 1.0),
            mayChange=True,
        )

    def update_crosshair(
        self,
        camera,
        cam_lens,
        render,
        aspect_ratio: float,
        aircraft_position: np.ndarray,
        aircraft_rotation: np.ndarray,
    ) -> None:
        if camera is None or camera.isEmpty() or cam_lens is None:
            self.crosshair_text.hide()
            return

        forward = aircraft_rotation @ np.array([0.0, 1.0, 0.0])
        aim_point = aircraft_position + 2_000.0 * forward
        camera_point = camera.getRelativePoint(render, LPoint3f(*aim_point))

        projected = LPoint2f()
        if not cam_lens.project(camera_point, projected):
            self.crosshair_text.hide()
            return

        self.crosshair_text.show()
        self.crosshair_text.setPos(
            float(projected.x * aspect_ratio),
            float(projected.y),
        )

    def update_status(
        self,
        mode_name: str,
        instruction_name: str,
        elapsed_time: float,
        state: AircraftState,
        air_data: dict,
        throttle_command: float,
        bullet_count: int,
    ) -> None:
        speed = float(air_data["speed"])
        altitude = float(-state.position[2])
        roll, pitch, yaw = quaternion_to_euler(state.quaternion)

        self.status_text.setText(
            "\n".join(
                [
                    f"Mode : {mode_name}",
                    f"Instruction : {instruction_name}",
                    f"t = {elapsed_time:5.1f} s",
                    f"h = {altitude:7.0f} m",
                    f"V = {speed:6.1f} m/s",
                    f"roll = {np.rad2deg(roll):6.1f} deg",
                    f"pitch = {np.rad2deg(pitch):6.1f} deg",
                    f"yaw = {np.rad2deg(yaw):6.1f} deg",
                    f"thr = {state.throttle:4.2f}",
                    f"moteur = {AircraftAnimationController.engine_stage_label(state.throttle)}",
                    f"cmd gaz = {throttle_command:4.2f}",
                    f"bullets = {bullet_count}",
                    "F1 instructions | F2 simulation",
                    "Souris ou Q/D: roulis | S/Z: pitch",
                    "Clic gauche maintenu: tir",
                    "Maj/Ctrl: gaz | A/E: lacet",
                ]
            )
        )

    def update(
        self,
        mode_name: str,
        instruction_name: str,
        elapsed_time: float,
        state: AircraftState,
        air_data: dict,
        throttle_command: float,
        bullet_count: int,
        camera,
        cam_lens,
        render,
        aspect_ratio: float,
        aircraft_position: np.ndarray,
        aircraft_rotation: np.ndarray,
    ) -> None:
        self.update_crosshair(
            camera,
            cam_lens,
            render,
            aspect_ratio,
            aircraft_position,
            aircraft_rotation,
        )
        self.update_status(
            mode_name,
            instruction_name,
            elapsed_time,
            state,
            air_data,
            throttle_command,
            bullet_count,
        )

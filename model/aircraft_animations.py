from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from panda3d.core import (
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LMatrix4f,
    LPoint3f,
    LVector3f,
    NodePath,
    TransparencyAttrib,
)


class ControlSurfaceState(Protocol):
    aileron: float
    elevator: float
    rudder: float


@dataclass(frozen=True)
class ControlSurfaceBinding:
    node_name: str
    control_name: str
    axis: str
    max_angle_deg: float
    sign: float = 1.0


CONTROL_SURFACE_BINDINGS = (
    ControlSurfaceBinding("Aile_Droite", "aileron", "z", 22.0, 1.0),
    ControlSurfaceBinding("Aile_Gauche", "aileron", "z", 22.0, -1.0),
    ControlSurfaceBinding("Canard_Droit", "elevator", "z", 18.0, 1.0),
    ControlSurfaceBinding("Canard_Gauche", "elevator", "z", 18.0, 1.0),
    ControlSurfaceBinding("Derive", "rudder", "y", 22.0, 1.0),
)

CONTROL_SURFACE_AXES = {
    "x": LVector3f(1.0, 0.0, 0.0),
    "y": LVector3f(0.0, 1.0, 0.0),
    "z": LVector3f(0.0, 0.0, 1.0),
}


class ControlSurfaceAnimator:
    def __init__(
        self,
        aircraft_model: NodePath,
        bindings: tuple[ControlSurfaceBinding, ...] = CONTROL_SURFACE_BINDINGS,
    ) -> None:
        self.surfaces = []

        for binding in bindings:
            node = aircraft_model.find(f"**/{binding.node_name}")
            if node.isEmpty():
                print(f"Surface animee introuvable : {binding.node_name}")
                continue

            self.surfaces.append((binding, node, LMatrix4f(node.getMat())))

    def update(self, controls: ControlSurfaceState | None) -> None:
        if controls is None:
            return

        for binding, node, base_mat in self.surfaces:
            value = float(np.clip(getattr(controls, binding.control_name), -1.0, 1.0))
            angle = binding.sign * binding.max_angle_deg * value
            axis = CONTROL_SURFACE_AXES.get(binding.axis)

            if axis is None:
                raise ValueError(f"Axe de surface invalide : {binding.axis}")

            node.setMat(LMatrix4f.rotateMat(angle, axis) * base_mat)


def make_exhaust_cone(name: str, segments: int = 8) -> GeomNode:
    vertex_format = GeomVertexFormat.getV3n3c4()
    vertex_data = GeomVertexData(name, vertex_format, Geom.UHStatic)

    vertex = GeomVertexWriter(vertex_data, "vertex")
    normal = GeomVertexWriter(vertex_data, "normal")
    color = GeomVertexWriter(vertex_data, "color")

    vertex.addData3f(0.0, -1.0, 0.0)
    normal.addData3f(0.0, -1.0, 0.0)
    color.addData4f(1.0, 1.0, 1.0, 1.0)

    for index in range(segments):
        angle = 2.0 * np.pi * index / segments
        x = float(np.cos(angle))
        z = float(np.sin(angle))
        vertex.addData3f(x, 0.0, z)
        normal.addData3f(x, 0.15, z)
        color.addData4f(1.0, 1.0, 1.0, 1.0)

    triangles = GeomTriangles(Geom.UHStatic)
    for index in range(segments):
        triangles.addVertices(0, index + 1, ((index + 1) % segments) + 1)

    geom = Geom(vertex_data)
    geom.addPrimitive(triangles)

    node = GeomNode(name)
    node.addGeom(geom)
    return node


class EngineExhaustEffect:
    def __init__(self, parent: NodePath, aircraft_bounds) -> None:
        self.plumes = []

        exhaust_positions, nozzle_radius, dry_length, afterburner_length = (
            self._build_exhaust_layout(aircraft_bounds)
        )
        self.nozzle_radius = nozzle_radius
        self.dry_length = dry_length
        self.afterburner_length = afterburner_length

        for index, position in enumerate(exhaust_positions):
            root = parent.attachNewNode(f"engine-exhaust-{index}")
            root.setPos(*position)

            dry = root.attachNewNode(make_exhaust_cone(f"dry-plume-{index}"))
            outer = root.attachNewNode(make_exhaust_cone(f"afterburner-outer-{index}"))
            core = root.attachNewNode(make_exhaust_cone(f"afterburner-core-{index}"))

            for node in (dry, outer, core):
                node.setTransparency(TransparencyAttrib.MAlpha)
                node.setTwoSided(True)
                node.setLightOff()
                node.setDepthWrite(False)
                node.hide()

            self.plumes.append((dry, outer, core, index * 1.7))

    @staticmethod
    def _build_exhaust_layout(
        aircraft_bounds,
    ) -> tuple[list[tuple[float, float, float]], float, float, float]:
        if aircraft_bounds is None:
            min_bound = LPoint3f(-5.0, -8.0, -1.0)
            max_bound = LPoint3f(5.0, 8.0, 2.0)
        else:
            min_bound, max_bound = aircraft_bounds

        width = max(float(max_bound.x - min_bound.x), 0.1)
        length = max(float(max_bound.y - min_bound.y), 0.1)
        height = max(float(max_bound.z - min_bound.z), 0.1)

        rear_y = float(min_bound.y - 0.02 * length)
        center_x = float((min_bound.x + max_bound.x) * 0.5)
        low_z = float(min_bound.z + 0.25 * height)

        engine_offset = min(max(width * 0.08, 0.8), 1.2)
        nozzle_radius = max(min(width, height) * 0.04, 0.12)
        dry_length = max(length * 0.12, nozzle_radius * 5.0)
        afterburner_length = max(length * 0.28, nozzle_radius * 11.0)

        positions = [
            (center_x - engine_offset, rear_y, low_z),
            (center_x + engine_offset * 0.92, rear_y, low_z),
        ]
        return positions, nozzle_radius, dry_length, afterburner_length

    @staticmethod
    def stage_label(throttle: float) -> str:
        throttle = float(np.clip(throttle, 0.0, 1.5))
        if throttle < 0.08:
            return "reacteur coupe"
        if throttle < 0.35:
            return f"ralenti {throttle * 100.0:3.0f}%"
        if throttle <= 1.0:
            return f"sec {throttle * 100.0:3.0f}%"

        afterburner = (throttle - 1.0) / 0.5
        return f"postcombustion {afterburner * 100.0:3.0f}%"

    def update(self, throttle: float, speed: float, elapsed_time: float) -> None:
        throttle = float(np.clip(throttle, 0.0, 1.5))
        speed_ratio = float(np.clip(speed / 350.0, 0.0, 1.2))
        dry_ratio = float(np.clip(throttle, 0.0, 1.0))
        afterburner_ratio = float(np.clip((throttle - 1.0) / 0.5, 0.0, 1.0))

        if throttle < 0.04:
            for dry, outer, core, _ in self.plumes:
                dry.hide()
                outer.hide()
                core.hide()
            return

        for dry, outer, core, phase_offset in self.plumes:
            flicker_phase = (
                elapsed_time
                * (
                    10.0
                    + 16.0 * dry_ratio
                    + 22.0 * afterburner_ratio
                    + 5.0 * speed_ratio
                )
                + phase_offset
            )
            flicker = float(0.92 + 0.08 * np.sin(flicker_phase))

            dry_length = (
                self.dry_length
                * (0.35 + 0.75 * dry_ratio)
                * (1.0 + 0.16 * speed_ratio)
                * flicker
            )
            dry_radius = self.nozzle_radius * (0.45 + 0.65 * dry_ratio)
            dry_alpha = (0.10 + 0.35 * dry_ratio) * (1.0 - 0.55 * afterburner_ratio)
            dry.setScale(dry_radius, dry_length, dry_radius)
            dry.setColorScale(1.0, 0.47 + 0.22 * dry_ratio, 0.08, dry_alpha)
            dry.show()

            if afterburner_ratio <= 0.01:
                outer.hide()
                core.hide()
                continue

            afterburner_length = (
                self.afterburner_length
                * (0.45 + 0.85 * afterburner_ratio)
                * (1.0 + 0.28 * speed_ratio)
                * flicker
            )
            outer_radius = self.nozzle_radius * (1.15 + 0.55 * afterburner_ratio)
            core_radius = self.nozzle_radius * (0.42 + 0.24 * afterburner_ratio)

            outer.setScale(outer_radius, afterburner_length, outer_radius)
            outer.setColorScale(
                1.0,
                0.18 + 0.22 * flicker,
                0.03,
                0.25 + 0.42 * afterburner_ratio,
            )
            outer.show()

            core.setScale(core_radius, afterburner_length * 0.72, core_radius)
            core.setColorScale(
                0.42,
                0.72 + 0.18 * afterburner_ratio,
                1.0,
                0.40 + 0.48 * afterburner_ratio,
            )
            core.show()


class AircraftAnimationController:
    def __init__(
        self,
        aircraft_model: NodePath,
        aircraft_root: NodePath,
        aircraft_bounds,
    ) -> None:
        self.control_surfaces = ControlSurfaceAnimator(aircraft_model)
        self.engine_exhaust = EngineExhaustEffect(aircraft_root, aircraft_bounds)

    @staticmethod
    def engine_stage_label(throttle: float) -> str:
        return EngineExhaustEffect.stage_label(throttle)

    def update(
        self,
        controls: ControlSurfaceState | None,
        throttle: float,
        speed: float,
        elapsed_time: float,
    ) -> None:
        self.control_surfaces.update(controls)
        self.engine_exhaust.update(throttle, speed, elapsed_time)

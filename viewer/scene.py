from __future__ import annotations

from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    LineSegs,
    NodePath,
    TransparencyAttrib,
)


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


class SceneRenderer:
    def __init__(self, render: NodePath) -> None:
        self.render = render

    def setup(self) -> None:
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

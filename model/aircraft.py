from pathlib import Path

import pyvista as pv


def create_aircraft() -> Path:
    fuselage = pv.Cylinder(
        center=(0, 0, 0),
        direction=(1, 0, 0),
        radius=0.4,
        height=6,
    )

    wings = pv.Box(
        bounds=(-0.5, 0.5, -4, 4, -0.1, 0.1)
    )

    tail = pv.Box(
        bounds=(-2.8, -1.8, -1.5, 1.5, -0.05, 0.05)
    )

    vertical_tail = pv.Box(
        bounds=(-2.8, -1.8, -0.08, 0.08, 0.35, 1.5)
    )

    model_path = Path(__file__).with_name("rafale.glb")
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")

    # return fuselage.merge(wings).merge(tail).merge(vertical_tail)
    return model_path

from pathlib import Path
from typing import Any

import numpy as np


AIRCRAFT_MODEL_FILENAME = "rafale.glb"


def aircraft_model_path(filename: str = AIRCRAFT_MODEL_FILENAME) -> Path:
    model_path = Path(__file__).with_name(filename)
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable : {model_path}")

    return model_path


def create_aircraft() -> Path:
    return aircraft_model_path()


def import_aircraft_actors(plotter: Any, model_path: str | Path):
    import pyvista as pv

    model_path = Path(model_path)
    model_center = np.array(pv.read(model_path).center, dtype=float)
    existing_actor_names = set(plotter.renderer.actors)

    plotter.import_gltf(str(model_path), set_camera=False)
    actors = [
        actor
        for name, actor in plotter.renderer.actors.items()
        if name not in existing_actor_names
    ]

    if not actors:
        raise RuntimeError(f"Aucun acteur charge depuis le modele 3D : {model_path}")

    return actors, model_center

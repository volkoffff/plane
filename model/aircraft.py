from pathlib import Path


def create_aircraft() -> Path:
    model_path = Path(__file__).with_name("rafale.glb")
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable : {model_path}")

    return model_path

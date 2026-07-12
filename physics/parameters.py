from dataclasses import dataclass


@dataclass
class AircraftParameters:
    mass: float
    wing_area: float
    mean_chord: float
    pitch_inertia: float
    drag_coefficient_zero: float
    induced_drag_factor: float
    max_load_factor: float
    cm0: float
    cm_alpha: float
    cm_q: float
    cm_delta_e: float
    max_dry_thrust: float
    max_afterburner_thrust: float
    throttle_response: float


def rafale_like_parameters() -> AircraftParameters:
    return AircraftParameters(
        mass=15_000.0,
        wing_area=45.7,
        mean_chord=4.5,
        pitch_inertia=250_000.0,
        max_load_factor=9.0,
        drag_coefficient_zero=0.025,
        induced_drag_factor=0.08,
        cm0=0.0,
        cm_alpha=-0.8,
        cm_q=-12.0,
        cm_delta_e=1.0,
        max_dry_thrust=100_000.0,
        max_afterburner_thrust=150_000.0,
        throttle_response=2.0,
    )

from dataclasses import dataclass
import numpy as np


@dataclass
class AircraftParameters:
    mass: float
    wing_area: float
    span: float
    mean_chord: float
    inertia_body: np.ndarray

    max_dry_thrust: float
    max_afterburner_thrust: float
    throttle_response: float

    max_elevator_deflection: float
    max_aileron_deflection: float
    max_rudder_deflection: float

    cl0: float
    cl_alpha: float
    cl_q: float
    cl_delta_e: float
    cl_max: float
    cl_min: float

    cd0: float
    induced_drag_factor: float

    cy_beta: float
    cy_delta_r: float

    roll_beta: float
    roll_p: float
    roll_r: float
    roll_delta_a: float

    cm0: float
    cm_alpha: float
    cm_q: float
    cm_delta_e: float

    cn_beta: float
    cn_p: float
    cn_r: float
    cn_delta_r: float


def rafale_like_parameters() -> AircraftParameters:
    mass = 15_000.0
    wing_area = 45.7
    span = 10.9
    mean_chord = wing_area / span

    inertia_body = np.diag([
        120_000.0,  # Ixx : roulis
        280_000.0,  # Iyy : tangage
        350_000.0,  # Izz : lacet
    ])

    return AircraftParameters(
        mass=mass,
        wing_area=wing_area,
        span=span,
        mean_chord=mean_chord,
        inertia_body=inertia_body,

        max_dry_thrust=100_000.0,
        max_afterburner_thrust=150_000.0,
        throttle_response=2.0,

        max_elevator_deflection=np.deg2rad(25.0),
        max_aileron_deflection=np.deg2rad(25.0),
        max_rudder_deflection=np.deg2rad(25.0),

        # Portance
        cl0=0.05,
        cl_alpha=4.5,
        cl_q=3.0,
        cl_delta_e=0.4,
        cl_max=1.4,
        cl_min=-0.8,

        # Traînée
        cd0=0.025,
        induced_drag_factor=0.08,

        # Force latérale
        cy_beta=-0.8,
        cy_delta_r=0.3,

        # Moment de roulis Cl
        roll_beta=-0.08,
        roll_p=-0.5,
        roll_r=0.15,
        roll_delta_a=0.6,

        # Moment de tangage Cm
        cm0=0.0,
        cm_alpha=-0.8,
        cm_q=-12.0,
        cm_delta_e=1.2,

        # Moment de lacet Cn
        cn_beta=0.12,
        cn_p=-0.05,
        cn_r=-0.4,
        cn_delta_r=-0.25,
    )
import numpy as np


def air_density(altitude_m: float) -> float:
    rho0 = 1.225
    scale_height = 8_500.0
    return float(rho0 * np.exp(-altitude_m / scale_height))

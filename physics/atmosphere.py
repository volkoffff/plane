import numpy as np


def air_density(altitude_m: float) -> float:
    rho0 = 1.225
    scale_height = 8500.0

    altitude_m = max(altitude_m, 0.0)

    return rho0 * np.exp(-altitude_m / scale_height)
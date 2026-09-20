from __future__ import annotations

import numpy as np
from panda3d.core import LMatrix4f

from physics.math3d import quaternion_to_matrix
from physics.state import AircraftState


# Conventions de repere :
# - la physique travaille en NED : x nord/avant, y est/droite, z vers le bas ;
# - Panda3D travaille avec z vers le haut et y comme axe de profondeur.
NED_TO_PANDA = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
)
BODY_PANDA_TO_NED = NED_TO_PANDA
MODEL_PANDA_TO_BODY_PANDA = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
)


def ned_to_panda(position_ned: np.ndarray) -> np.ndarray:
    return NED_TO_PANDA @ position_ned


def aircraft_panda_rotation(state: AircraftState) -> np.ndarray:
    body_to_world_ned = quaternion_to_matrix(state.quaternion)
    return NED_TO_PANDA @ body_to_world_ned @ BODY_PANDA_TO_NED


def panda_matrix_from_rotation_translation(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> LMatrix4f:
    matrix = LMatrix4f(LMatrix4f.identMat())

    for input_axis in range(3):
        for output_axis in range(3):
            matrix.setCell(
                input_axis,
                output_axis,
                float(rotation[output_axis, input_axis]),
            )

    matrix.setCell(3, 0, float(translation[0]))
    matrix.setCell(3, 1, float(translation[1]))
    matrix.setCell(3, 2, float(translation[2]))

    return matrix

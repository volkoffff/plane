import numpy as np

def normalize_quaternion(q: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(q)

  if norm < 1e-12:
    return np.array[1.0, 0.0, 0.0, 0.0]
  
  return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    w, x, y, z = q


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    w, x, y, z = q

    return np.array([
        [
            1 - 2*(y*y + z*z),
            2*(x*y - z*w),
            2*(x*z + y*w),
        ],
        [
            2*(x*y + z*w),
            1 - 2*(x*x + z*z),
            2*(y*z - x*w),
        ],
        [
            2*(x*z - y*w),
            2*(y*z + x*w),
            1 - 2*(x*x + y*y),
        ],
    ])


def quaternion_derivative(
    quaternion: np.ndarray,
    omega_body: np.ndarray,
) -> np.ndarray:
    """
    quaternion : orientation body -> world
    omega_body : [p, q, r] dans le repère avion
    """
    omega_quat = np.array([
        0.0,
        omega_body[0],
        omega_body[1],
        omega_body[2],
    ])

    return 0.5 * quaternion_multiply(quaternion, omega_quat)


def euler_to_quaternion(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """
    Angles aéronautiques :
    roll  : rotation autour de x
    pitch : nez vers le haut si positif
    yaw   : cap
    """
    cr = np.cos(roll / 2.0)
    sr = np.sin(roll / 2.0)

    cp = np.cos(pitch / 2.0)
    sp = np.sin(pitch / 2.0)

    cy = np.cos(yaw / 2.0)
    sy = np.sin(yaw / 2.0)

    return normalize_quaternion(np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy,
    ]))
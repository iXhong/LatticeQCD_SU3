"""SU(2) and SU(3) matrix helpers."""

import numpy as np


def dagger(matrix: np.ndarray) -> np.ndarray:
    """Return the conjugate transpose.

    Inputs:
        matrix: Complex matrix.
    Outputs:
        Conjugate-transposed matrix.
    """
    return matrix.conj().T


def random_su2(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random SU(2) matrix.

    Inputs:
        rng: Optional NumPy random generator.
    Outputs:
        Complex 2x2 SU(2) matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    a0, a1, a2, a3 = rng.normal(size=4)
    norm = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
    a0, a1, a2, a3 = a0 / norm, a1 / norm, a2 / norm, a3 / norm

    return np.array(
        [
            [a0 + 1j * a3, a2 + 1j * a1],
            [-a2 + 1j * a1, a0 - 1j * a3],
        ],
        dtype=np.complex128,
    )


def embed_su2(su2_matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    """Embed SU(2) into an SU(3) subgroup.

    Inputs:
        su2_matrix: Complex 2x2 SU(2) matrix.
        i: First SU(3) row/column index.
        j: Second SU(3) row/column index.
    Outputs:
        Complex 3x3 SU(3) matrix.
    """
    if su2_matrix.shape != (2, 2):
        raise ValueError("su2_matrix must have shape (2, 2)")
    if not (0 <= i < 3 and 0 <= j < 3 and i != j):
        raise ValueError("i and j must be distinct SU(3) indices in [0, 2]")

    su3_matrix = np.eye(3, dtype=np.complex128)
    su3_matrix[np.ix_([i, j], [i, j])] = su2_matrix
    return su3_matrix


def random_su3(rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate a random SU(3) matrix.

    Inputs:
        rng: Optional NumPy random generator.
    Outputs:
        Complex 3x3 SU(3) matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    r_matrix = embed_su2(random_su2(rng), 0, 1)
    s_matrix = embed_su2(random_su2(rng), 0, 2)
    t_matrix = embed_su2(random_su2(rng), 1, 2)
    return r_matrix @ s_matrix @ t_matrix


def random_su2_near_identity(
    step_size: float, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate a random SU(2) matrix near identity.

    Inputs:
        step_size: Maximum radius of the SU(2) vector part in [0, 1].
        rng: Optional NumPy random generator.
    Outputs:
        Complex 2x2 SU(2) proposal matrix.
    """
    if step_size < 0.0 or step_size > 1.0:
        raise ValueError("step_size must be in [0, 1]")
    if rng is None:
        rng = np.random.default_rng()
    if step_size == 0.0:
        return np.eye(2, dtype=np.complex128)

    direction = rng.normal(size=3)
    direction_norm = np.linalg.norm(direction)
    while direction_norm == 0.0:
        direction = rng.normal(size=3)
        direction_norm = np.linalg.norm(direction)

    radius = step_size * rng.random()
    a1, a2, a3 = radius * direction / direction_norm
    a0 = np.sqrt(1.0 - radius * radius)

    return np.array(
        [
            [a0 + 1j * a3, a2 + 1j * a1],
            [-a2 + 1j * a1, a0 - 1j * a3],
        ],
        dtype=np.complex128,
    )


def is_su3(matrix: np.ndarray, atol: float = 1e-12) -> bool:
    """Check whether a matrix is numerically in SU(3).

    Inputs:
        matrix: Complex matrix to check.
        atol: Absolute tolerance for numerical comparisons.
    Outputs:
        True if matrix is 3x3, unitary, and has determinant one.
    """
    if matrix.shape != (3, 3):
        return False

    identity = np.eye(3, dtype=np.complex128)
    return np.allclose(matrix.conj().T @ matrix, identity, atol=atol) and np.allclose(
        np.linalg.det(matrix), 1.0 + 0.0j, atol=atol
    )

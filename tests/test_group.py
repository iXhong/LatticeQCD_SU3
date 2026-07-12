import numpy as np

from lattice_su3 import is_su3, random_su3


def test_random_su3_is_unitary_and_has_unit_determinant():
    rng = np.random.default_rng(12345)
    identity = np.eye(3, dtype=np.complex128)

    for _ in range(100):
        x = random_su3(rng)

        assert x.shape == (3, 3)
        assert np.allclose(x.conj().T @ x, identity, atol=1e-12)
        assert np.allclose(np.linalg.det(x), 1.0 + 0.0j, atol=1e-12)


def test_is_su3_accepts_generated_su3_matrices():
    rng = np.random.default_rng(2024)

    for _ in range(100):
        assert is_su3(random_su3(rng))


def test_is_su3_rejects_non_su3_matrices():
    assert not is_su3(np.eye(2, dtype=np.complex128))
    assert not is_su3(2.0 * np.eye(3, dtype=np.complex128))

import numpy as np
import pytest

from generate_conf import LatticeGeometry, is_su3, random_su3


def test_lattice_geometry_index_coordinate_conversion():
    geometry = LatticeGeometry((2, 3, 4, 5))

    assert geometry.ndim == 4
    assert geometry.volume == 120
    assert geometry.index_from_coord((0, 0, 0, 0)) == 0
    assert geometry.index_from_coord((1, 2, 3, 4)) == 119
    assert geometry.coord_from_index(0) == (0, 0, 0, 0)
    assert geometry.coord_from_index(119) == (1, 2, 3, 4)


def test_lattice_geometry_periodic_neighbors_wrap_at_boundaries():
    geometry = LatticeGeometry((2, 3))

    site = geometry.index_from_coord((0, 0))
    assert geometry.coord_from_index(geometry.forward(site, 0)) == (1, 0)
    assert geometry.coord_from_index(geometry.backward(site, 0)) == (1, 0)
    assert geometry.coord_from_index(geometry.forward(site, 1)) == (0, 1)
    assert geometry.coord_from_index(geometry.backward(site, 1)) == (0, 2)

    corner = geometry.index_from_coord((1, 2))
    assert geometry.coord_from_index(geometry.forward(corner, 0)) == (0, 2)
    assert geometry.coord_from_index(geometry.backward(corner, 0)) == (0, 2)
    assert geometry.coord_from_index(geometry.forward(corner, 1)) == (1, 0)
    assert geometry.coord_from_index(geometry.backward(corner, 1)) == (1, 1)


def test_lattice_geometry_precomputed_neighbor_tables_match_methods():
    geometry = LatticeGeometry((3, 4, 5, 6))

    for site in (0, 1, 17, geometry.volume - 1):
        for mu in range(geometry.ndim):
            assert geometry.forward(site, mu) == geometry.forward_neighbors[site, mu]
            assert geometry.backward(site, mu) == geometry.backward_neighbors[site, mu]


def test_lattice_geometry_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        LatticeGeometry(())
    with pytest.raises(ValueError):
        LatticeGeometry((4, 0, 4, 4))
    with pytest.raises(ValueError):
        LatticeGeometry((4, 4)).index_from_coord((4, 0))
    with pytest.raises(ValueError):
        LatticeGeometry((4, 4)).coord_from_index(16)


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

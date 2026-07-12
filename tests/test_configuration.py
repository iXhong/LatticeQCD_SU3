import numpy as np

from lattice_su3 import LatticeGeometry, cold_start, hot_start, is_su3


def test_hot_start_creates_random_su3_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(7)

    links = hot_start(geometry, rng)

    assert links.shape == (geometry.volume, geometry.ndim, 3, 3)
    assert links.dtype == np.complex128
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu])


def test_cold_start_creates_identity_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    identity = np.eye(3, dtype=np.complex128)

    links = cold_start(geometry)

    assert links.shape == (geometry.volume, geometry.ndim, 3, 3)
    assert links.dtype == np.complex128
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert np.allclose(links[site, mu], identity)
            assert is_su3(links[site, mu])

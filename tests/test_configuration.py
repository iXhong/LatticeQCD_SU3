import numpy as np

from lattice_su3 import (
    LatticeGeometry,
    cold_start,
    hot_start,
    is_su3,
    load_configuration,
    save_configuration,
)


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


def test_save_and_load_configuration_round_trip(tmp_path):
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    metadata = {
        "shape": geometry.shape,
        "beta": 5.7,
        "sweep": 120,
        "config_index": 3,
        "backend": "jit",
        "seed": 12345,
        "tau_int": 4.25,
        "sweeps_between_configs": 9,
    }

    path = tmp_path / "config_000003.npz"
    save_configuration(path, links, metadata)
    loaded_links, loaded_metadata = load_configuration(path)

    assert loaded_links.shape == links.shape
    assert loaded_links.dtype == np.complex128
    assert np.allclose(loaded_links, links, atol=0.0)
    assert loaded_metadata["shape"] == [2, 2, 2, 2]
    assert loaded_metadata["beta"] == 5.7
    assert loaded_metadata["sweep"] == 120
    assert loaded_metadata["config_index"] == 3
    assert loaded_metadata["backend"] == "jit"
    assert loaded_metadata["seed"] == 12345
    assert loaded_metadata["tau_int"] == 4.25
    assert loaded_metadata["sweeps_between_configs"] == 9

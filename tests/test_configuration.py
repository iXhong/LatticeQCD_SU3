import numpy as np
import pytest

from lattice_su3 import (
    LatticeGeometry,
    cold_start,
    hot_start,
    is_su3,
    latest_configuration_path,
    load_configuration,
    load_start,
    save_configuration,
)


def test_hot_start_creates_random_su3_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    rng = np.random.default_rng(7)

    links = hot_start(geometry, rng)

    assert links.shape == (geometry.volume, geometry.ndim, 3, 3)
    assert links.dtype == np.complex64
    for site in range(geometry.volume):
        for mu in range(geometry.ndim):
            assert is_su3(links[site, mu])


def test_cold_start_creates_identity_links():
    geometry = LatticeGeometry((2, 2, 2, 2))
    identity = np.eye(3, dtype=np.complex64)

    links = cold_start(geometry)

    assert links.shape == (geometry.volume, geometry.ndim, 3, 3)
    assert links.dtype == np.complex64
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
    assert loaded_links.dtype == np.complex64
    assert np.allclose(loaded_links, links, atol=0.0)
    assert loaded_metadata["shape"] == [2, 2, 2, 2]
    assert loaded_metadata["beta"] == 5.7
    assert loaded_metadata["sweep"] == 120
    assert loaded_metadata["config_index"] == 3
    assert loaded_metadata["backend"] == "jit"
    assert loaded_metadata["seed"] == 12345
    assert loaded_metadata["tau_int"] == 4.25
    assert loaded_metadata["sweeps_between_configs"] == 9


def test_load_start_returns_validated_writable_links(tmp_path):
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    path = tmp_path / "source.npz"
    save_configuration(
        path,
        links,
        {"shape": geometry.shape, "sweep": 30, "start": "hot"},
    )

    loaded_links, metadata = load_start(path, geometry)

    assert np.allclose(loaded_links, links, atol=0.0)
    assert loaded_links.dtype == np.complex64
    assert loaded_links.flags.writeable
    assert metadata["sweep"] == 30
    loaded_links[0, 0, 0, 0] = 2.0
    assert links[0, 0, 0, 0] == 1.0


def test_load_start_rejects_wrong_link_shape(tmp_path):
    source_geometry = LatticeGeometry((2, 2, 2, 2))
    target_geometry = LatticeGeometry((3, 2, 2, 2))
    path = tmp_path / "source.npz"
    save_configuration(
        path,
        cold_start(source_geometry),
        {"shape": source_geometry.shape, "sweep": 10},
    )

    with pytest.raises(ValueError, match="loaded links have shape"):
        load_start(path, target_geometry)


def test_latest_configuration_path_uses_metadata_sweep_and_chain(tmp_path):
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    early = tmp_path / "z_early.npz"
    latest = tmp_path / "a_latest.npz"
    other_chain = tmp_path / "other_chain.npz"
    save_configuration(early, links, {"chain": 0, "sweep": 10})
    save_configuration(latest, links, {"chain": 0, "sweep": 30})
    save_configuration(other_chain, links, {"chain": 1, "sweep": 40})

    assert latest_configuration_path(tmp_path, chain=0) == latest
    assert latest_configuration_path(tmp_path, chain=1) == other_chain


def test_latest_configuration_path_rejects_ambiguous_latest_sweep(tmp_path):
    geometry = LatticeGeometry((2, 2, 2, 2))
    links = cold_start(geometry)
    save_configuration(tmp_path / "first.npz", links, {"sweep": 20})
    save_configuration(tmp_path / "second.npz", links, {"sweep": 20})

    with pytest.raises(ValueError, match="multiple configurations"):
        latest_configuration_path(tmp_path)

import pytest

from lattice_su3 import LatticeGeometry


def test_lattice_geometry_index_coordinate_conversion():
    geometry = LatticeGeometry((2, 3, 4, 5))

    assert geometry.ndim == 4
    assert geometry.volume == 120
    assert geometry.index_from_coord((0, 0, 0, 0)) == 0
    assert geometry.index_from_coord((1, 2, 3, 4)) == 119
    assert geometry.coord_from_index(0) == (0, 0, 0, 0)
    assert geometry.coord_from_index(119) == (1, 2, 3, 4)


def test_lattice_geometry_periodic_neighbors_wrap_at_boundaries():
    geometry = LatticeGeometry((2, 3, 4, 5))

    site = geometry.index_from_coord((0, 0, 0, 0))
    assert geometry.coord_from_index(geometry.forward(site, 0)) == (1, 0, 0, 0)
    assert geometry.coord_from_index(geometry.backward(site, 0)) == (1, 0, 0, 0)
    assert geometry.coord_from_index(geometry.forward(site, 1)) == (0, 1, 0, 0)
    assert geometry.coord_from_index(geometry.backward(site, 1)) == (0, 2, 0, 0)
    assert geometry.coord_from_index(geometry.forward(site, 2)) == (0, 0, 1, 0)
    assert geometry.coord_from_index(geometry.backward(site, 2)) == (0, 0, 3, 0)
    assert geometry.coord_from_index(geometry.forward(site, 3)) == (0, 0, 0, 1)
    assert geometry.coord_from_index(geometry.backward(site, 3)) == (0, 0, 0, 4)

    corner = geometry.index_from_coord((1, 2, 3, 4))
    assert geometry.coord_from_index(geometry.forward(corner, 0)) == (0, 2, 3, 4)
    assert geometry.coord_from_index(geometry.backward(corner, 0)) == (0, 2, 3, 4)
    assert geometry.coord_from_index(geometry.forward(corner, 1)) == (1, 0, 3, 4)
    assert geometry.coord_from_index(geometry.backward(corner, 1)) == (1, 1, 3, 4)
    assert geometry.coord_from_index(geometry.forward(corner, 2)) == (1, 2, 0, 4)
    assert geometry.coord_from_index(geometry.backward(corner, 2)) == (1, 2, 2, 4)
    assert geometry.coord_from_index(geometry.forward(corner, 3)) == (1, 2, 3, 0)
    assert geometry.coord_from_index(geometry.backward(corner, 3)) == (1, 2, 3, 3)


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

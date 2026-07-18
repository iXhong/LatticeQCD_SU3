import numpy as np
import pytest

from lattice_su3 import block_by_chain, bootstrap_by_chain, jackknife_delete_one


def test_block_by_chain_sorts_sweeps_and_never_crosses_chains():
    values = np.asarray([[30.0], [1.0], [20.0], [4.0], [10.0], [2.0], [3.0]])
    chains = np.asarray([1, 0, 1, 0, 1, 0, 0])
    sweeps = np.asarray([30, 10, 20, 40, 10, 20, 30])

    blocked = block_by_chain(values, chains, sweeps, block_size=2)

    assert np.allclose(blocked.values[:, 0], [1.5, 3.5, 15.0], atol=1e-12)
    assert np.array_equal(blocked.chains, [0, 0, 1])
    assert np.array_equal(blocked.sweep_start, [10, 30, 10])
    assert np.array_equal(blocked.sweep_stop, [20, 40, 20])
    assert np.array_equal(blocked.dropped_per_chain, [[0, 0], [1, 1]])


def test_jackknife_delete_one_matches_explicit_means():
    blocks = np.asarray([[1.0, 10.0], [2.0, 20.0], [4.0, 40.0]])

    samples = jackknife_delete_one(blocks)

    expected = np.asarray([[3.0, 30.0], [2.5, 25.0], [1.5, 15.0]])
    assert np.allclose(samples, expected, atol=1e-12)


def test_bootstrap_by_chain_is_reproducible_and_preserves_output_shape():
    blocks = np.asarray([[1.0], [2.0], [10.0], [20.0]])
    chains = np.asarray([0, 0, 1, 1])

    first = bootstrap_by_chain(blocks, chains, 25, np.random.default_rng(17))
    second = bootstrap_by_chain(blocks, chains, 25, np.random.default_rng(17))

    assert first.shape == (25, 1)
    assert np.allclose(first, second, atol=1e-12)
    assert np.all((first[:, 0] >= 5.5) & (first[:, 0] <= 11.0))


def test_block_by_chain_rejects_duplicate_sweeps_within_chain():
    with pytest.raises(ValueError, match="duplicate sweep"):
        block_by_chain(
            np.asarray([1.0, 2.0, 3.0, 4.0]),
            np.asarray([0, 0, 1, 1]),
            np.asarray([10, 10, 10, 20]),
            block_size=1,
        )


def test_block_by_chain_requires_two_complete_blocks():
    with pytest.raises(ValueError, match="at least two complete blocks"):
        block_by_chain(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0, 0, 0]),
            np.asarray([10, 20, 30]),
            block_size=2,
        )


@pytest.mark.parametrize("block_size", [0, -1])
def test_block_by_chain_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="positive"):
        block_by_chain(
            np.asarray([1.0, 2.0]),
            np.asarray([0, 0]),
            np.asarray([10, 20]),
            block_size=block_size,
        )

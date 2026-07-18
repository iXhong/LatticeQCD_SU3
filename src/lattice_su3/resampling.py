"""Blocking and jackknife helpers for correlated ensemble measurements."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlockedSamples:
    """Store equal-size blocks formed independently within Markov chains.

    Inputs:
        values: Block means with the observable dimensions retained.
        chains: Chain identifier for every block.
        sweep_start: First sweep included in every block.
        sweep_stop: Last sweep included in every block.
        block_size: Number of configurations in every block.
        dropped_per_chain: Chain identifiers and discarded tail counts.
    Outputs:
        Immutable collection of blocked measurements and provenance.
    """

    values: np.ndarray
    chains: np.ndarray
    sweep_start: np.ndarray
    sweep_stop: np.ndarray
    block_size: int
    dropped_per_chain: np.ndarray


def block_by_chain(
    values: np.ndarray,
    chains: np.ndarray,
    sweeps: np.ndarray,
    block_size: int,
) -> BlockedSamples:
    """Form contiguous equal-size blocks without crossing chain boundaries.

    Inputs:
        values: Measurements with shape ``(n_cfg, *observable_shape)``.
        chains: Integer chain identifier for every configuration.
        sweeps: Sweep number for every configuration.
        block_size: Positive number of configurations per block.
    Outputs:
        Block means and their chain and sweep provenance.
    """
    values = np.asarray(values)
    chains = np.asarray(chains, dtype=np.int64)
    sweeps = np.asarray(sweeps, dtype=np.int64)
    if values.ndim < 1 or values.shape[0] == 0:
        raise ValueError("values must contain at least one configuration")
    if chains.ndim != 1 or sweeps.ndim != 1:
        raise ValueError("chains and sweeps must be one-dimensional")
    if len(chains) != len(values) or len(sweeps) != len(values):
        raise ValueError("values, chains, and sweeps must have equal lengths")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    block_values: list[np.ndarray] = []
    block_chains: list[int] = []
    sweep_starts: list[int] = []
    sweep_stops: list[int] = []
    dropped: list[tuple[int, int]] = []

    for chain in np.unique(chains):
        indices = np.flatnonzero(chains == chain)
        order = np.argsort(sweeps[indices], kind="stable")
        indices = indices[order]
        chain_sweeps = sweeps[indices]
        if len(np.unique(chain_sweeps)) != len(chain_sweeps):
            raise ValueError(f"chain {chain} contains duplicate sweep numbers")

        n_blocks = len(indices) // block_size
        used_count = n_blocks * block_size
        dropped.append((int(chain), len(indices) - used_count))
        for start in range(0, used_count, block_size):
            block_indices = indices[start : start + block_size]
            block_values.append(values[block_indices].mean(axis=0))
            block_chains.append(int(chain))
            sweep_starts.append(int(sweeps[block_indices[0]]))
            sweep_stops.append(int(sweeps[block_indices[-1]]))

    if len(block_values) < 2:
        raise ValueError("need at least two complete blocks for jackknife analysis")
    return BlockedSamples(
        values=np.stack(block_values),
        chains=np.asarray(block_chains, dtype=np.int64),
        sweep_start=np.asarray(sweep_starts, dtype=np.int64),
        sweep_stop=np.asarray(sweep_stops, dtype=np.int64),
        block_size=block_size,
        dropped_per_chain=np.asarray(dropped, dtype=np.int64),
    )


def jackknife_delete_one(block_values: np.ndarray) -> np.ndarray:
    """Build leave-one-block-out means from equally weighted block means.

    Inputs:
        block_values: Block means with shape ``(n_blocks, *observable_shape)``.
    Outputs:
        Jackknife samples with the same shape as ``block_values``.
    """
    block_values = np.asarray(block_values)
    if block_values.ndim < 1 or block_values.shape[0] < 2:
        raise ValueError("at least two block values are required")
    return (block_values.sum(axis=0) - block_values) / (block_values.shape[0] - 1)


def bootstrap_by_chain(
    block_values: np.ndarray,
    block_chains: np.ndarray,
    n_resamples: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Bootstrap block means independently within each Markov chain.

    Inputs:
        block_values: Equal-weight block means with leading block dimension.
        block_chains: Chain identifier for every block.
        n_resamples: Positive number of bootstrap ensemble means.
        rng: Optional NumPy random generator.
    Outputs:
        Bootstrap means with shape ``(n_resamples, *observable_shape)``.
    """
    block_values = np.asarray(block_values)
    block_chains = np.asarray(block_chains, dtype=np.int64)
    if block_values.ndim < 1 or block_values.shape[0] < 2:
        raise ValueError("at least two block values are required")
    if block_chains.shape != (block_values.shape[0],):
        raise ValueError("block_chains must match the block count")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if rng is None:
        rng = np.random.default_rng()

    chain_indices = [np.flatnonzero(block_chains == chain) for chain in np.unique(block_chains)]
    samples = np.empty(
        (n_resamples, *block_values.shape[1:]), dtype=block_values.dtype
    )
    for sample_index in range(n_resamples):
        selected = []
        for indices in chain_indices:
            selected.append(rng.choice(indices, size=len(indices), replace=True))
        samples[sample_index] = block_values[np.concatenate(selected)].mean(axis=0)
    return samples

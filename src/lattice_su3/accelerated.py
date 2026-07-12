"""Optional accelerated update kernels."""

from __future__ import annotations

import numpy as np

from lattice_su3.geometry import LatticeGeometry
from lattice_su3.update import UpdateStats

try:
    from numba import njit
except ImportError as error:  # pragma: no cover - exercised only without extra deps
    NUMBA_IMPORT_ERROR = error
    njit = None
else:
    NUMBA_IMPORT_ERROR = None


if njit is not None:

    @njit(cache=True)
    def _random_su2_entries():
        """Generate random SU(2) matrix entries.

        Inputs:
            None.
        Outputs:
            Four complex entries of an SU(2) matrix.
        """
        a0 = np.random.normal()
        a1 = np.random.normal()
        a2 = np.random.normal()
        a3 = np.random.normal()
        norm = np.sqrt(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3)
        a0 = a0 / norm
        a1 = a1 / norm
        a2 = a2 / norm
        a3 = a3 / norm
        return (
            a0 + 1j * a3,
            a2 + 1j * a1,
            -a2 + 1j * a1,
            a0 - 1j * a3,
        )

    @njit(cache=True)
    def _sample_su2_heatbath_entries(x0, x1, x2, x3, beta_over_n):
        """Sample SU(2) heatbath update entries from staple coefficients.

        Inputs:
            x0: Real identity coefficient of the effective staple.
            x1: Real first Pauli-vector coefficient of the effective staple.
            x2: Real second Pauli-vector coefficient of the effective staple.
            x3: Real third Pauli-vector coefficient of the effective staple.
            beta_over_n: Wilson beta divided by the gauge group size.
        Outputs:
            Four complex entries of an SU(2) update matrix.
        """
        staple_norm = np.sqrt(max(x0 * x0 + x1 * x1 + x2 * x2 + x3 * x3, 0.0))
        if staple_norm <= 1e-14 or beta_over_n == 0.0:
            return _random_su2_entries()

        alpha = 2.0 * beta_over_n * staple_norm
        while True:
            r1 = max(np.random.random(), np.finfo(np.float64).tiny)
            r2 = np.random.random()
            r3 = max(np.random.random(), np.finfo(np.float64).tiny)
            cos_term = np.cos(2.0 * np.pi * r2)
            lambda_squared = -(np.log(r1) + cos_term * cos_term * np.log(r3)) / (
                2.0 * alpha
            )
            acceptance_sample = np.random.random()
            if (
                lambda_squared <= 1.0
                and acceptance_sample * acceptance_sample <= 1.0 - lambda_squared
            ):
                break

        y0 = 1.0 - 2.0 * lambda_squared
        vector_norm = np.sqrt(max(1.0 - y0 * y0, 0.0))
        d0 = np.random.normal()
        d1 = np.random.normal()
        d2 = np.random.normal()
        direction_norm = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
        while direction_norm == 0.0:
            d0 = np.random.normal()
            d1 = np.random.normal()
            d2 = np.random.normal()
            direction_norm = np.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

        y1 = vector_norm * d0 / direction_norm
        y2 = vector_norm * d1 / direction_norm
        y3 = vector_norm * d2 / direction_norm

        a00 = y0 + 1j * y3
        a01 = y2 + 1j * y1
        a10 = -y2 + 1j * y1
        a11 = y0 - 1j * y3

        inv_norm = 1.0 / staple_norm
        b00 = (x0 - 1j * x3) * inv_norm
        b01 = (-x2 - 1j * x1) * inv_norm
        b10 = (x2 - 1j * x1) * inv_norm
        b11 = (x0 + 1j * x3) * inv_norm

        return (
            a00 * b00 + a01 * b10,
            a00 * b01 + a01 * b11,
            a10 * b00 + a11 * b10,
            a10 * b01 + a11 * b11,
        )

    @njit(cache=True)
    def _staple_jit(links, forward_neighbors, backward_neighbors, site, mu):
        """Compute the staple sum around one link in JIT code.

        Inputs:
            links: Gauge links U[site, direction].
            forward_neighbors: Forward neighbor table.
            backward_neighbors: Backward neighbor table.
            site: Flat site index of the link.
            mu: Direction index of the link.
        Outputs:
            Complex 3x3 staple sum.
        """
        ndim = links.shape[1]
        staple_sum = np.zeros((3, 3), dtype=np.complex128)
        site_plus_mu = forward_neighbors[site, mu]

        for nu in range(ndim):
            if nu == mu:
                continue

            site_plus_nu = forward_neighbors[site, nu]
            site_minus_nu = backward_neighbors[site, nu]
            site_plus_mu_minus_nu = backward_neighbors[site_plus_mu, nu]

            temp = np.zeros((3, 3), dtype=np.complex128)
            forward_staple = np.zeros((3, 3), dtype=np.complex128)
            backward_staple = np.zeros((3, 3), dtype=np.complex128)

            for a in range(3):
                for b in range(3):
                    value = 0.0 + 0.0j
                    for c in range(3):
                        value += links[site_plus_mu, nu, a, c] * np.conj(
                            links[site_plus_nu, mu, b, c]
                        )
                    temp[a, b] = value

            for a in range(3):
                for b in range(3):
                    value = 0.0 + 0.0j
                    for c in range(3):
                        value += temp[a, c] * np.conj(links[site, nu, b, c])
                    forward_staple[a, b] = value

            for a in range(3):
                for b in range(3):
                    value = 0.0 + 0.0j
                    for c in range(3):
                        value += np.conj(links[site_plus_mu_minus_nu, nu, c, a]) * np.conj(
                            links[site_minus_nu, mu, b, c]
                        )
                    temp[a, b] = value

            for a in range(3):
                for b in range(3):
                    value = 0.0 + 0.0j
                    for c in range(3):
                        value += temp[a, c] * links[site_minus_nu, nu, c, b]
                    backward_staple[a, b] = value

            staple_sum += forward_staple + backward_staple

        return staple_sum

    @njit(cache=True)
    def _heatbath_update_link_jit(
        links, forward_neighbors, backward_neighbors, site, mu, beta
    ):
        """Run one in-place JIT heatbath link update.

        Inputs:
            links: Gauge links U[site, direction].
            forward_neighbors: Forward neighbor table.
            backward_neighbors: Backward neighbor table.
            site: Flat site index of the link.
            mu: Direction index of the link.
            beta: Wilson gauge coupling parameter.
        Outputs:
            None.
        """
        link_matrix = links[site, mu].copy()
        staple_matrix = _staple_jit(
            links, forward_neighbors, backward_neighbors, site, mu
        )
        beta_over_n = beta / 3.0

        for pair_index in range(3):
            if pair_index == 0:
                i, j = 0, 1
            elif pair_index == 1:
                i, j = 0, 2
            else:
                i, j = 1, 2

            p = 0.0 + 0.0j
            q = 0.0 + 0.0j
            r = 0.0 + 0.0j
            s = 0.0 + 0.0j
            for c in range(3):
                p += link_matrix[i, c] * staple_matrix[c, i]
                q += link_matrix[i, c] * staple_matrix[c, j]
                r += link_matrix[j, c] * staple_matrix[c, i]
                s += link_matrix[j, c] * staple_matrix[c, j]

            x0 = 0.5 * np.real(p + s)
            x1 = 0.5 * np.imag(r + q)
            x2 = 0.5 * np.real(q - r)
            x3 = 0.5 * (np.imag(p) - np.imag(s))
            u00, u01, u10, u11 = _sample_su2_heatbath_entries(
                x0, x1, x2, x3, beta_over_n
            )

            row_i0 = link_matrix[i, 0]
            row_i1 = link_matrix[i, 1]
            row_i2 = link_matrix[i, 2]
            row_j0 = link_matrix[j, 0]
            row_j1 = link_matrix[j, 1]
            row_j2 = link_matrix[j, 2]

            link_matrix[i, 0] = u00 * row_i0 + u01 * row_j0
            link_matrix[i, 1] = u00 * row_i1 + u01 * row_j1
            link_matrix[i, 2] = u00 * row_i2 + u01 * row_j2
            link_matrix[j, 0] = u10 * row_i0 + u11 * row_j0
            link_matrix[j, 1] = u10 * row_i1 + u11 * row_j1
            link_matrix[j, 2] = u10 * row_i2 + u11 * row_j2

        links[site, mu] = link_matrix

    @njit(cache=True)
    def _heatbath_jit_sweep_kernel(
        links, forward_neighbors, backward_neighbors, beta, seed
    ):
        """Run one in-place JIT heatbath sweep over all links.

        Inputs:
            links: Gauge links U[site, direction].
            forward_neighbors: Forward neighbor table.
            backward_neighbors: Backward neighbor table.
            beta: Wilson gauge coupling parameter.
            seed: Random seed, or negative to avoid reseeding.
        Outputs:
            None.
        """
        if seed >= 0:
            np.random.seed(seed)

        volume = links.shape[0]
        ndim = links.shape[1]
        for site in range(volume):
            for mu in range(ndim):
                _heatbath_update_link_jit(
                    links, forward_neighbors, backward_neighbors, site, mu, beta
                )


def heatbath_jit_sweep(
    links: np.ndarray,
    geometry: LatticeGeometry,
    beta: float,
    seed: int | None = None,
) -> UpdateStats:
    """Run one optional numba-JIT heatbath sweep over all links.

    Inputs:
        links: Gauge links U[site, direction].
        geometry: Lattice geometry object.
        beta: Wilson gauge coupling parameter.
        seed: Optional integer seed for numba's random stream.
    Outputs:
        UpdateStats with attempted links, accepted links, and acceptance rate.
    """
    if NUMBA_IMPORT_ERROR is not None:
        raise ImportError(
            "heatbath_jit_sweep requires installing the acceleration extra"
        ) from NUMBA_IMPORT_ERROR
    if beta < 0.0:
        raise ValueError("beta must be non-negative")

    kernel_seed = -1 if seed is None else int(seed)
    _heatbath_jit_sweep_kernel(
        links,
        geometry.forward_neighbors,
        geometry.backward_neighbors,
        beta,
        kernel_seed,
    )
    attempted_links = geometry.volume * geometry.ndim
    return UpdateStats(attempted_links=attempted_links, accepted_links=attempted_links)

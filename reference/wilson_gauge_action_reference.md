# SU(3) Wilson Gauge Action: Agent Reference

## Conventions

- Four-dimensional Euclidean lattice; sites are denoted by `n`.
- Link variable: $U_\mu(n)\in SU(3)$, directed from $n$ to $n+\hat\mu$.
- Reverse link: $U_{-\mu}(n+\hat\mu)=U_\mu^\dagger(n)$.
- Bare gauge coupling: $g$.
- Inverse coupling for $SU(3)$:

$$
\beta=\frac{6}{g^2}.
$$

## Plaquette

$$
U_{\mu\nu}(n)=
U_\mu(n)U_\nu(n+\hat\mu)
U_\mu^\dagger(n+\hat\nu)U_\nu^\dagger(n).
$$

The sum over $\mu<\nu$ counts each unoriented plaquette once.

## Wilson gauge action

Equivalent forms:

$$
S_G[U]
=\frac{2}{g^2}\sum_n\sum_{\mu<\nu}
\operatorname{ReTr}[\mathbf 1-U_{\mu\nu}(n)]
$$

and

$$
S_G[U]
=\beta\sum_n\sum_{\mu<\nu}
\left(1-\frac13\operatorname{ReTr}U_{\mu\nu}(n)\right).
$$

Dropping the configuration-independent constant gives

$$
S_G[U]\doteq-\frac{\beta}{3}
\sum_n\sum_{\mu<\nu}\operatorname{ReTr}U_{\mu\nu}(n).
$$

For quenched/pure-gauge simulations, the target weight is $P[U]\propto e^{-S_G[U]}$.

## Single-link local action

For an update $U_\mu(n)\to U'_\mu(n)$, only the six plaquettes containing that link change. Define the staple sum

$$
A_\mu(n)=\sum_{i=1}^{6}P_i,
$$

where each $P_i$ is the ordered product of the other three links in one adjacent plaquette, oriented so that $U_\mu(n)P_i$ is the plaquette. Explicitly,

$$
\begin{aligned}
A_\mu(n)=\sum_{\nu\ne\mu}\Big[&
U_\nu(n+\hat\mu)U_\mu^\dagger(n+\hat\nu)U_\nu^\dagger(n)\\
&+U_\nu^\dagger(n+\hat\mu-\hat\nu)
U_\mu^\dagger(n-\hat\nu)U_\nu(n-\hat\nu)
\Big].
\end{aligned}
$$

The local contribution is

$$
S_{\mathrm{loc}}[U_\mu(n)]
=\frac{\beta}{3}\operatorname{ReTr}
[6\mathbf 1-U_\mu(n)A_\mu(n)].
$$

Since $A_\mu(n)$ does not contain the updated link,

$$
\boxed{
\Delta S=S_G[U']-S_G[U]
=-\frac{\beta}{3}\operatorname{ReTr}
[(U'_\mu(n)-U_\mu(n))A_\mu(n)]
}.
$$

For a symmetric Metropolis proposal,

$$
P_{\mathrm{acc}}=\min(1,e^{-\Delta S}).
$$

## Source locations

C. Gattringer and C. B. Lang, *Quantum Chromodynamics on the Lattice*:

- Plaquette: Sec. 2.3.2, Eq. (2.48), book p. 37.
- Wilson gauge action: Sec. 2.3.2, Eq. (2.49), book p. 38.
- $\beta=6/g^2$: Eqs. (3.4)-(3.5), book p. 44.
- Metropolis definition $\Delta S=S[U']-S[U]$: Eq. (4.19), book p. 78.
- Staple sum, local action, and local $\Delta S$: Eqs. (4.20)-(4.21), book pp. 79-80.

All page numbers above are the printed book page numbers, not PDF viewer indices.

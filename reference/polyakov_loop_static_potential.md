# Polyakov Loop and Static Quark Potential

> Source: *Quantum Chromodynamics on the Lattice* by C. Gattringer and C. B. Lang
> Sections: 3.3 (Wilson and Polyakov loops), 3.3.5 (Polyakov loop), 3.4 (Static quark potential), 4.5 (Measurement & analysis), 4.5.4 (Numerical exercises), 12.1.1 (Finite temperature pure gauge theory)

---

## 1. Polyakov Loop Definition (Section 3.3.5)

The Polyakov loop (also called the *thermal Wilson line*) is a gauge-invariant observable constructed from a closed loop of temporal link variables winding around the periodic time direction.

**Definition:**

$$
P(m) = \operatorname{tr} \left[ \prod_{j=0}^{N_T - 1} U_4(m, j) \right]
\tag{3.60}
$$

where:
- $m$ is the spatial lattice site
- $N_T$ is the total number of lattice points in the time direction
- $U_4(m, j)$ is the gauge link variable in the time direction at site $(m, j)$
- The trace guarantees gauge invariance

**Construction:** Starting from a Wilson loop whose temporal extent is maximized ($n_t = N_T$), the spatial pieces cancel due to periodic boundary conditions and opposite orientation, leaving two disconnected temporal Wilson lines at spatial positions $m$ and $n$. Taking the trace of each individually yields gauge-invariant Polyakov loops.

**Physical interpretation:** The Polyakov loop couples to a static quark at position $x$ via the current $j_\mu(z) = (0, 0, 0, 1)\,\delta(z - x)$ in Euclidean metric:
$$ O = \operatorname{tr} \, \mathcal{P} \exp\left(i \int d^4z\, j_\mu(z) A_\mu(z)\right) $$

**Order parameter:** The vacuum expectation value $\langle P(n) \rangle$ of a single Polyakov loop serves as an order parameter for the deconfinement transition in pure gluodynamics at finite temperature (discussed in Chap. 12).

---

## 2. Static Potential from Polyakov Loop Correlator

The correlation function of two Polyakov loops is related to the static quark-antiquark potential:

$$
\langle P(m) P(n)^\dagger \rangle \propto e^{-N_T a V(r)} \left[ 1 + O(e^{-N_T a \Delta E}) \right]
\tag{3.61, 4.77}
$$

where:
- $r = a |m - n|$ is the spatial separation
- $a$ is the lattice spacing
- $N_T a$ is the total temporal extent
- $V(r)$ is the static quark potential
- $\Delta E$ is the energy gap to the first excited state
- The $O(e^{-N_T a \Delta E})$ term represents excited-state contamination

**Extracting the potential (up to an irrelevant constant):**

$$
a V(r) = -\frac{1}{N_T} \ln \langle P(m) P(n)^\dagger \rangle
\tag{4.78}
$$

---

## 3. Static Quark Potential Parameterization (Section 3.4)

The static QCD potential takes the form:

$$
V(r) = A + \frac{B}{r} + \sigma r
\tag{3.62}
$$

- **$A$**: an irrelevant energy offset (normalization constant)
- **$B/r$**: Coulomb term from one-gluon exchange (dominant at short distances, weak coupling)
  - Derived from the continuum gluon action in the small-$g$ limit where self-interaction terms vanish and QCD reduces to QED-like interactions for each color component
- **$\sigma r$**: linearly rising confining term (dominant at large distances, strong coupling)
  - $\sigma$ is the **string tension**, with phenomenological value $\sigma \approx 900\ \text{MeV/fm}$
  - Arises from formation of a flux tube/string between static quarks due to gluon self-interaction

**Physical implications:** The linear term implies confinement — quarks cannot be separated indefinitely. In full QCD with dynamical quarks, string breaking occurs when the energy is sufficient to create a quark-antiquark pair from the vacuum.

---

## 4. Strong Coupling Expansion of the Wilson Loop (Section 3.4.1)

In the strong-coupling limit ($g$ large, $\beta$ small), the Wilson loop expectation value demonstrates confinement:

$$
\langle W_C \rangle = \frac{1}{Z'} \int D[U] \exp\left(\frac{\beta}{6} \sum_P [\operatorname{tr} U_P + \operatorname{tr} U_P^\dagger]\right) \operatorname{tr} \prod_{l \in C} U_l
\tag{3.63, 3.64}
$$

Expanding the Boltzmann factor in $\beta$:

$$
\exp\left(\frac{\beta}{6} \sum_P [\operatorname{tr} U_P + \operatorname{tr} U_P^\dagger]\right)
= \sum_{i,j=0}^\infty \frac{1}{i! j!} \left(\frac{\beta}{6}\right)^{i+j}
\left(\sum_P \operatorname{tr} U_P\right)^i
\left(\sum_P \operatorname{tr} U_P^\dagger\right)^j
\tag{3.66}
$$

The leading contribution requires tiling the minimal area of the Wilson loop with plaquettes, giving:

$$
\langle W_C \rangle \propto \left(\frac{\beta}{6}\right)^{rt/a^2} = e^{-(\ln(6/\beta)) \, r t / a^2}
$$

which yields a linear potential $V(r) = \sigma r$ with $\sigma = -(1/a^2) \ln(\beta/6)$.

---

## 5. Coulomb Part of the Static Potential (Section 3.4.2)

In the weak-coupling limit ($g \to 0$), the gluon field strength tensor reduces to its abelian form:

$$
F_{\mu\nu}^{(i)}(x) = \partial_\mu A_\nu^{(i)}(x) - \partial_\nu A_\mu^{(i)}(x)
\tag{3.74}
$$

The action becomes a sum of QED-type interactions for each color component, implying the presence of the $1/r$ Coulomb term in the potential (3.62).

---

## 6. Sommer Scale and Setting the Lattice Scale (Section 3.5)

The Sommer parameter $r_0$ is a characteristic length scale tied to the static potential, with physical value $r_0 \simeq 0.5\ \text{fm}$.

**Definition:** The force $F(r) = dV(r)/dr$ satisfies:

$$
F(r_0)\, r_0^2 = 1.65
\tag{3.78}
$$

For the parameterized potential:

$$
F(r) = \frac{d}{dr} \left(A + \frac{B}{r} + \sigma r\right) = -\frac{B}{r^2} + \sigma
\tag{3.79}
$$

The condition for $r_0$ becomes:

$$
F(r_0) r_0^2 = -B + \sigma r_0^2 = 1.65 \quad \Rightarrow \quad r_0 = \sqrt{\frac{1.65 + B}{\sigma}}
\tag{3.80}
$$

In lattice units:

$$
\frac{r_0}{a} = \sqrt{\frac{1.65 + B}{\sigma a^2}}
\tag{3.81}
$$

**Determining $a$:** From numerical data for $aV(an)$, extract $r_0/a$ (the number of lattice spacings corresponding to $r_0 = 0.5\ \text{fm}$), then:

$$
a = \frac{0.5\ \text{fm}}{r_0/a}
$$

---

## 7. Numerical Exercise (Section 4.5.4)

### Observable 1: Average Plaquette

The sum over plaquettes entering the Wilson gauge action for SU($N$):

$$
S_P[U] = \frac{1}{N} \sum_P \operatorname{Re} \operatorname{tr} [U_P]
\tag{4.74}
$$

Its expectation value:

$$
\langle S_P \rangle = \frac{\int D[U] \exp(\beta S_P[U]) S_P[U]}{\int D[U] \exp(\beta S_P[U])}
\tag{4.75}
$$

The action relates to $S_P$ by $S[U] = \beta (6|\Lambda| - S_P[U])$. The normalized quantity $E_P \equiv \langle S_P \rangle/(6|\Lambda|)$ ranges between 0 and 1. Its derivative relates to the variance:

$$
\frac{d\langle S_P \rangle}{d\beta} = \langle S_P^2 \rangle - \langle S_P \rangle^2
\tag{4.76}
$$

### Observable 2: Static Potential from Polyakov Loops

The correlation function of Polyakov loops yields the static potential via Eq. (4.77)$\equiv$(3.61) and (4.78):

$$
\langle P(m) P(n)^\dagger \rangle \propto e^{-N_T a V(r)},
\qquad
a V(r) = -\frac{1}{N_T} \ln \langle P(m) P(n)^\dagger \rangle
$$

**Example parameters** (left-hand plot of Fig. 4.3):
- SU(3) pure gauge theory, Wilson action
- $16^3 \times 6$ lattice at $\beta = 5.7$
- 50,000 subsequent iterations
- State-of-the-art calculations use Wilson loops with higher statistics and more sophisticated methods (e.g., Necco, [21])

**Rotational invariance:** Using non-planar Wilson loops (off-axis distance vectors), one can verify that rotational invariance is restored in the continuum limit (large $\beta$, small $a$).

---

## 8. Measurement of Polyakov Loop Correlators (Section 4.5)

Due to translational invariance (for periodic boundary conditions), one averages over all possible spatial positions to improve statistics.

**Spatial averaging:**
- For a single Polyakov loop, average over all spatial sites $n$:
  $$
  P = \frac{1}{N_S^3} \sum_m P(m)
  \tag{12.8}
  $$
- For the correlator $\langle P(m) P(n)^\dagger \rangle$, average over all combinations $m, n$ for each distance $|m - n|$
- This involves sums over many terms; rounding errors become an issue, especially when subtracting large contributions — use higher-precision arithmetic

**Autocorrelation and thinning:**
- Measurements taken from subsequent Monte Carlo configurations are correlated
- The autocorrelation time (Sect. 4.5) determines how many sweeps to discard between measurements
  - A thinned/subsampled series eliminates the need for autocorrelation corrections in the analysis
  - Alternatively, all measurements can be kept and the analysis corrected for autocorrelation, but this requires estimating the autocorrelation matrix
- Store intermediate results for careful a posteriori statistical analysis

---

## 9. Finite Temperature and Center Symmetry (Section 12.1.1)

### Polyakov Loop Correlator at Finite Temperature

At finite temperature the temporal extent $N_T$ is related to the temperature:
$$
T = \frac{1}{a N_T}
$$

The Polyakov loop correlator gives the **free energy** $F_{q\bar{q}}$ of a static quark-antiquark pair at that temperature:
$$
\langle P(m) P(n)^\dagger \rangle = e^{-a N_T F_{q\bar{q}}(a|m-n|)} = e^{-F_{q\bar{q}}(r)/T}
\tag{12.6}
$$

At large distances the correlator factorizes:
$$
\lim_{a|m-n| \to \infty} \langle P(m) P(n)^\dagger \rangle = \langle P(m) \rangle \langle P(n)^\dagger \rangle = |\langle P \rangle|^2
\tag{12.7}
$$

### Polyakov Loop as Deconfinement Order Parameter

A single Polyakov loop expectation value relates to the free energy of a single static charge:
$$
|\langle P \rangle| \sim e^{-F_q / T}
\tag{12.10}
$$

The confinement criterion:
- **Confined phase:** $|\langle P \rangle| = 0$ (free energy $F_q \to \infty$, an isolated color charge cannot exist)
- **Deconfined phase:** $|\langle P \rangle| \neq 0$ (free energy $F_q$ is finite)

For pure SU(3) gauge theory the deconfinement transition occurs at $T_c \approx 270$ MeV.

### Center Symmetry (Z$_3$)

The Polyakov loop transforms under the center symmetry Z$_3$ of SU(3):
$$
U_4(n, t_0) \to z\, U_4(n, t_0), \qquad z \in \{1, e^{2\pi i/3}, e^{-2\pi i/3}\}
\tag{12.11}
$$

Under this transformation the Polyakov loop picks up a phase:
$$
P \to z P
\tag{12.13}
$$

The gauge action is invariant, but the Polyakov loop is not. In the confined phase, averaging over all center sectors forces $\langle P \rangle = 0$:
$$
\langle P \rangle = \frac{1}{3}(1 + e^{2\pi i/3} + e^{-2\pi i/3})\, \langle P \rangle = 0
\tag{12.14}
$$

Above $T_c$ the center symmetry is spontaneously broken, and $\langle P \rangle$ acquires a non-zero value — this is the **deconfinement phase transition**.

---

## Key Formulas Summary

| Formula | Description | Eq. |
|---------|-------------|-----|
| $P(m) = \operatorname{tr} \left[\prod_{j=0}^{N_T-1} U_4(m,j)\right]$ | Polyakov loop | (3.60) |
| $\langle P(m)P(n)^\dagger \rangle \propto e^{-N_T a V(r)}$ | Polyakov correlator → potential | (3.61) |
| $a V(r) = -\frac{1}{N_T} \ln \langle P(m)P(n)^\dagger \rangle$ | Extract potential | (4.78) |
| $V(r) = A + B/r + \sigma r$ | Static potential parameterization | (3.62) |
| $F(r_0) r_0^2 = 1.65$ | Sommer scale definition | (3.78) |
| $F(r) = -B/r^2 + \sigma$ | Force from parameterized potential | (3.79) |
| $r_0/a = \sqrt{(1.65+B)/(\sigma a^2)}$ | Sommer scale in lattice units | (3.81) |

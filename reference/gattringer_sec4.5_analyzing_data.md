# 4.5 Analyzing the data

Source: Christof Gattringer and Christian B. Lang, *Quantum Chromodynamics on the Lattice*, Sec. 4.5–4.5.2 (pp. 93–96).

The statistical analysis of the measured observables is the important final step of a Monte Carlo simulation. This analysis should also provide one with the information how many updating sweeps have to be discarded before configurations in equilibrium are produced and how many sweeps are necessary between two measurements. The final product of the statistical analysis is the average value which one quotes for an observable and an estimate for the corresponding statistical error.

## 4.5.1 Statistical analysis for uncorrelated data

We assume that we have computed the values $(x_1, x_2, \ldots, x_N)$ of some observable for a Markov sequence of Monte Carlo-generated configurations in equilibrium. Each of the values of the sample corresponds to a random variable $X_i$. All these variables have the same expectation value and variance:

$$
\langle X_i \rangle = \langle X \rangle, \quad \sigma^2_{X_i} = \langle (X_i - \langle X \rangle)^2 \rangle = \sigma^2_X. \tag{4.53}
$$

Candidates for unbiased estimators for these values are

$$
\bar X = \frac{1}{N} \sum_{i=1}^N X_i, \qquad
\hat\sigma^2_X = \frac{1}{N-1} \sum_{i=1}^N \bigl( X_i - \bar X \bigr)^2. \tag{4.54}
$$

If the $X_i$ are uncorrelated one finds for $i \neq j$,

$$
\langle X_i X_j \rangle = \langle X_i \rangle \langle X_j \rangle = \langle X \rangle^2, \tag{4.55}
$$

and the variance $\hat\sigma^2_X$ allows one to determine the statistical error of $\bar X$. To see this first note that the sample mean value $\bar X$ is an estimator for the correct mean value: $\langle \bar X \rangle = \langle X \rangle$. It is, however, itself a random variable, since its value may change from one set of $N$ configurations to another set. The variance of that estimator is

$$
\begin{aligned}
\sigma^2_{\bar X} &= \Bigl\langle \bigl( \bar X - \langle X \rangle \bigr)^2 \Bigr\rangle
= \frac{1}{N^2} \Bigl\langle \Bigl( \sum_{i=1}^N (X_i - \langle X \rangle) \Bigr)^2 \Bigr\rangle \\
&= \frac{1}{N^2} \sum_{i,j=1}^N \langle (X_i - \langle X \rangle)(X_j - \langle X \rangle) \rangle
= \frac{1}{N} \bigl( \langle X^2 \rangle - \langle X \rangle^2 \bigr) + \frac{1}{N^2} \sum_{i \neq j} \langle X_i X_j \rangle.
\end{aligned} \tag{4.56}
$$

For uncorrelated $X_i$ the contributions from $i \neq j$ factorize due to (4.55) and

$$
\sigma^2_{\bar X} = \frac{1}{N} \sigma^2_X. \tag{4.57}
$$

This is the well-known result for uncorrelated measurements. Thus, for the observable based on $N$ measurements, the statistical error, i.e., the standard deviation (s.d.), is $\sigma_{\bar X}$. The value $\sigma_X$ on the right-hand side of (4.57) is approximated using $\hat\sigma_X$ from (4.54). For the case of $N$ uncorrelated measurements one quotes the final result as

$$
\bar X \pm \sigma \quad \text{with} \quad \sigma = \frac{\hat\sigma_X}{\sqrt N}. \tag{4.58}
$$

The important message of this equation is that the statistical error decreases like $1/\sqrt N$ with the number $N$ of uncorrelated configurations.

## 4.5.2 Autocorrelation

Since in our case the data sample is the result of a (computer-)time series in our Monte Carlo simulation there is high chance that the observables are in fact correlated. This so-called autocorrelation leads to a nonvanishing autocorrelation function, which we define as

$$
C_X(X_i, X_{i+t}) = \langle (X_i - \langle X_i \rangle)(X_{i+t} - \langle X_{i+t} \rangle) \rangle
= \langle X_i X_{i+t} \rangle - \langle X_i \rangle \langle X_{i+t} \rangle. \tag{4.59}
$$

For a Markov chain in equilibrium the autocorrelation function depends only on the (computer time) separation $t$ and we write

$$
C_X(t) = C_X(X_i, X_{i+t}). \tag{4.60}
$$

Note that $C_X(0) = \sigma^2_X$. In a typical situation the normalized correlation function $\Gamma_X$ exhibits exponential behavior asymptotically for large $t$:

$$
\Gamma_X(t) \equiv \frac{C_X(t)}{C_X(0)} \sim \exp\!\Bigl( -\frac{t}{\tau_{X,\text{exp}}} \Bigr), \tag{4.61}
$$

and one calls $\tau_{X,\text{exp}}$ the exponential autocorrelation time for $X$. The complete expression for $\Gamma_X(t)$ involves a sum over several such terms. In (4.61) we consider only the asymptotically leading term with the largest autocorrelation time. This number provides information on how strongly subsequent measurements are correlated. The exponential autocorrelation time $\tau_{\text{exp}}$ is the supremum of the values $\tau_{X,\text{exp}}$ for all possible observables $X$:

$$
\tau_{\text{exp}} = \sup_X \tau_{X,\text{exp}}. \tag{4.62}
$$

Autocorrelations lead to systematic errors which are $O(\exp(-t/\tau_{\text{exp}}))$ if the computer time between subsequent measurements is $t$.

For correlated random variables $X_i$ the terms with $i = j$ in the second line of (4.56) do not vanish and one can continue this equation to obtain for the correlated case

$$
\begin{aligned}
\sigma^2_{\hat X}
&= \frac{1}{N^2} \sum_{i,j=1}^N C_X(|i-j|)
= \frac{1}{N^2} \sum_{t=-(N-1)}^{N-1} \sum_{k=1}^{N-|t|} C_X(|t|) \\
&= \frac{1}{N^2} \sum_{t=-N}^N (N - |t|)\, C_X(|t|)
= \frac{C_X(0)}{N^2} \sum_{t=-N}^N (N - |t|)\, \Gamma_X(|t|) \\
&\approx \frac{\sigma^2_X}{N} + 2\,\frac{\sigma^2_X}{N} \sum_{t=1}^N \Gamma_X(|t|)
\equiv \frac{\sigma^2_X}{N}\, 2\,\tau_{X,\text{int}},
\end{aligned} \tag{4.63}
$$

where we have introduced the integrated autocorrelation time

$$
\tau_{X,\text{int}} = \frac12 + \sum_{t=1}^N \Gamma_X(t). \tag{4.64}
$$

This definition is motivated by the observation that for exponential behavior

$$
\tau_{X,\text{int}} = \frac12 + \sum_{t=1}^N \Gamma_X(|t|)
\approx \int_0^\infty dt\; e^{-t/\tau} = \tau \quad (\text{for large }\tau). \tag{4.65}
$$

In the last step of (4.63) we have neglected the factor $1 - |t|/N$ which is justified for large enough $N$ due to the exponential suppression of $\Gamma_X(|t|)$.

Computing $\tau_{X,\text{int}}$ in a realistic situation one has to cut off sum (4.64) at a value of $t$ where the values of $\Gamma(t)$ become unreliable. Usually one then assumes exponential behavior for the part not explicitly taken into account in the sum. Still, the determination of $\tau_{\text{exp}}$ or even $\tau_{\text{int}}$ is a delicate business. Usually one needs at least $1000\,\tau$ data values for estimates of $\tau$ itself. In order to judge whether the measured autocorrelation time is reliable, one therefore should start with small size lattices and high statistics and work oneself up to larger sizes, carefully checking the behavior and reliability of $C(t)$.

The variance $\sigma^2_{\bar X}$ computed in this way is larger than the variance computed from (4.57), which assumes an uncorrelated sample. The number of effectively independent data out of $N$ values is therefore

$$
N_{\text{indep}} = \frac{N}{2\,\tau_{X,\text{int}}} \tag{4.66}
$$

or

$$
\sigma^2_{\bar X,\text{corrected}} = 2\,\tau_{X,\text{int}}\;\sigma^2_{\bar X}. \tag{4.67}
$$

For equilibration from some start configuration one should discard at least $20\,\tau$, for good statistical accuracy maybe $1000\,\tau$ configurations. When producing data with $1\%$ errors one typically needs $> 10\,000\,\tau$ values. For more detailed discussions, cf. [18–20].

Summing up our results we find that for the correlated case the result one quotes is given by

$$
\bar X \pm \sigma \quad \text{with} \quad \sigma = \sqrt{\frac{1}{N}\,2\,\tau_{X,\text{int}}}\,\hat\sigma_X. \tag{4.68}
$$

Finally let us briefly mention the issue of critical slowing down. The autocorrelation time depends on the updating algorithm but also on the parameters of the lattice system. For lattice field systems one expects that

$$
\tau_{X,\text{int}} \sim (\xi_X)^z, \qquad \tau_{\text{exp}} \sim \xi^z, \tag{4.69}
$$

where $\xi_X$ is the correlation length for the observable $X$ and $\xi$ the longest correlation length within the system. The correlation length is defined from the exponential decay of correlation functions between local observables measured at different points on the lattice, i.e., $\langle X(x)X(y) \rangle \sim e^{-|x-y|/\xi_X}$ for large $|x-y|$. The dynamical critical exponent $z \ge 0$ depends on the updating algorithm. At critical points $\xi$ approaches infinity, however, on finite lattices of linear size $L$ one has $\xi \le L$. Thus, near a critical point, the computational effort grows like a power of the extension of the lattice:

$$
\text{numerical cost} \propto L^z. \tag{4.70}
$$

This behavior is called critical slowing down. For first-order phase transitions, where the system may tunnel between different phases, the autocorrelation time grows like $\exp(c L^{D-1})$ for a $D$-dimensional lattice.

We summarize: From the data one has to get an estimate of the autocorrelation time. This provides (a) information on the number of update sweeps to be discarded between measurements and (b) a correction factor to the statistical error derived naively as for statistically independent data.

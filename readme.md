# Pricing American Options

American options can be exercised at any time up to maturity, so their price depends on finding the best time to exercise. Unlike a European option, there is no single terminal payoff to discount. You have to solve an optimal stopping problem, and the exercise boundary is part of the answer.

This project compares several ways of solving that problem:

| Method | Status |
|---|---|
| Deep optimal stopping (Becker, Cheridito & Jentzen, 2020) | Implemented, [`notebooks/optimal_stopping.ipynb`](notebooks/optimal_stopping.ipynb) |
| Longstaff–Schwartz least-squares Monte Carlo (AMS 514) | Used as a benchmark in the notebook |
| Physics-informed neural networks (AMS 516) | Planned |
| Multi-agent reinforcement learning | Planned, to build on the optimal stopping notebook |

## Deep optimal stopping

The notebook implements [*Deep optimal stopping*](https://arxiv.org/abs/1804.05394) (Becker, Cheridito & Jentzen, JMLR 2019) and stays close to the paper.

**Idea.** Any stopping time can be written as a sequence of 0/1 decisions, one per exercise date *n* (Theorem 1). Each decision is learned by its own neural network, working backwards from maturity:

- At maturity the decision is fixed: *f*<sub>N</sub> ≡ 1, so the option is always exercised.
- For *n* = *N*−1, …, 1, the network is trained to maximise the expected reward

  E[ g(n, Xₙ) · F<sup>θ</sup>(Xₙ) + g(τₙ₊₁, X<sub>τₙ₊₁</sub>) · (1 − F<sup>θ</sup>(Xₙ)) ]

  Here g is the discounted payoff, F<sup>θ</sup> is the network's soft stopping probability, and τₙ₊₁ is the stopping time already learned for later dates.
- The hard decision is *f* = 1{F<sup>θ</sup> ≥ ½}, which gives τₙ = n·fₙ + τₙ₊₁·(1 − fₙ).
- At *n* = 0 the decision is a constant, set by comparing the immediate payoff with the estimated continuation value (Remark 6).

**Implementation details that follow the paper**

- Network input Yₙ = (Xₙ, g(n, Xₙ))
- Depth 3, with *d* + 40 units per hidden layer, ReLU activations and a sigmoid output
- Xavier initialisation, batch normalisation and Adam
- **Lower bound** (Section 3.1): the learned policy evaluated on fresh paths not used in training
- **Upper bound** (Section 3.2): the dual martingale bound, using nested continuation simulations
- **Point estimate and 95% confidence interval** (Section 3.3)

The solver takes any one-step simulator `step(x, n)` and payoff `payoff(x)`, so it applies to other Markov models and to multi-asset payoffs, not just the example below.

### Example: American put under Black–Scholes

Parameters: S₀ = 100, K = 100, r = 5%, σ = 20%, T = 1 year, 100 exercise dates.

| Method | Price |
|---|---|
| Deep OS lower bound | 6.056 |
| Deep OS upper bound | 6.269 |
| Deep OS 95% CI | [6.043, 6.312] |
| Binomial tree, 100 exercise dates | 6.084 |
| Binomial tree, American | 6.090 |

The confidence interval contains the binomial reference, and the lower bound is within 0.5% of it. The upper bound is loose because each continuation value is estimated from only 256 nested paths. Raising `n_inner` tightens it, at the cost of runtime. On a laptop CPU, training takes about 2.5 minutes and the bounds take about 1.5 minutes.

The notebook also plots:

- the learned exercise boundary S\*(t)
- the soft stopping probability at several dates
- sample paths coloured by early exercise
- the distribution of exercise times

### Path simulators

`UnivariateEulerMethods` simulates price paths under:

- geometric Brownian motion
- GBM with Poisson (Merton-style) jumps
- GBM with self-exciting Hawkes jumps
- GBM with CIR stochastic variance (correlated with the price)

The deep optimal stopping example uses risk-neutral GBM. Pricing under the other models needs the model's full state as the input *X*. For example, the variance process for stochastic volatility, or the jump intensity for Hawkes, because the price alone isn't Markov.

## Getting started

```bash
git clone <this repo>
cd pricing_american_options
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib torch jupyter
jupyter notebook notebooks/optimal_stopping.ipynb
```

Tested with Python 3.14, PyTorch 2.14 and NumPy 2.5. Run the cells from top to bottom. The pricing cell trains the networks, and the final cell compares the results with the benchmarks and draws the plots.

## References

- S. Becker, P. Cheridito, A. Jentzen. *Deep optimal stopping.* Journal of Machine Learning Research 20 (2019). [arXiv:1804.05394](https://arxiv.org/abs/1804.05394)
- F. Longstaff, E. Schwartz. *Valuing American options by simulation: a simple least-squares approach.* Review of Financial Studies 14 (2001).
- L.C.G. Rogers. *Monte Carlo valuation of American options.* Mathematical Finance 12 (2002).
- M. Haugh, L. Kogan. *Pricing American options: a duality approach.* Operations Research 52 (2004).

---
layout: default
title: Risk Metrics Derivation
parent: Research
nav_order: 5
math: true
---

# Risk Metrics: Mathematical Foundations

This document provides rigorous mathematical derivations for Value at Risk (VaR), Conditional Value at Risk (CVaR), tail risk measures, and backtesting procedures.

## 1. Value at Risk (VaR)

### 1.1 Definition

**Definition 1 (Value at Risk)**

*For a loss random variable $L$ and confidence level $\alpha \in (0, 1)$, Value at Risk is defined as:*

$$
\text{VaR}_\alpha(L) = \inf\{x \in \mathbb{R} : \mathbb{P}(L \leq x) \geq \alpha\} = F_L^{-1}(\alpha) \tag{1}
$$

*Equivalently, VaR is the $\alpha$-quantile of the loss distribution.*

**Interpretation**: With probability $\alpha$, losses will not exceed $\text{VaR}_\alpha$.

**Example**: $\text{VaR}_{0.99} = \$50\text{M}$ means there is a 1% chance of losing more than $50M.

### 1.2 Properties

**Theorem 1 (VaR Properties)**

*VaR satisfies:*

1. *Monotonicity*: If $L_1 \leq L_2$ a.s., then $\text{VaR}_\alpha(L_1) \leq \text{VaR}_\alpha(L_2)$
2. *Translation invariance*: $\text{VaR}_\alpha(L + c) = \text{VaR}_\alpha(L) + c$
3. *Positive homogeneity*: $\text{VaR}_\alpha(\lambda L) = \lambda \text{VaR}_\alpha(L)$ for $\lambda > 0$

**Theorem 2 (VaR is NOT Subadditive)**

*VaR does not generally satisfy:*

$$
\text{VaR}_\alpha(L_1 + L_2) \leq \text{VaR}_\alpha(L_1) + \text{VaR}_\alpha(L_2) \tag{2}
$$

<details>
<summary><strong>Counterexample</strong></summary>

Consider two binary losses with $\alpha = 0.95$:
- $L_1 = \begin{cases} 0 & \text{w.p. } 0.96 \\ 100 & \text{w.p. } 0.04 \end{cases}$
- $L_2 = \begin{cases} 0 & \text{w.p. } 0.96 \\ 100 & \text{w.p. } 0.04 \end{cases}$

Assume $L_1, L_2$ are independent.

Individual VaRs:
- $\text{VaR}_{0.95}(L_1) = \text{VaR}_{0.95}(L_2) = 0$

Sum distribution:
- $\mathbb{P}(L_1 + L_2 = 0) = 0.96^2 = 0.9216$
- $\mathbb{P}(L_1 + L_2 = 100) = 2 \times 0.96 \times 0.04 = 0.0768$
- $\mathbb{P}(L_1 + L_2 = 200) = 0.04^2 = 0.0016$

Therefore:
- $\mathbb{P}(L_1 + L_2 \leq 0) = 0.9216 < 0.95$
- $\text{VaR}_{0.95}(L_1 + L_2) = 100 > 0 + 0 = \text{VaR}_{0.95}(L_1) + \text{VaR}_{0.95}(L_2)$

$\square$
</details>

### 1.3 Estimation Methods

**Definition 2 (Historical VaR)**

*Given $n$ historical losses $\{L_1, \ldots, L_n\}$, let $L_{(k)}$ be the $k$-th order statistic. Then:*

$$
\widehat{\text{VaR}}_\alpha = L_{(\lceil n\alpha \rceil)} \tag{3}
$$

**Definition 3 (Parametric VaR)**

*Assuming $L \sim \mathcal{N}(\mu, \sigma^2)$:*

$$
\text{VaR}_\alpha = \mu + \sigma \Phi^{-1}(\alpha) \tag{4}
$$

*where $\Phi^{-1}$ is the standard normal quantile function.*

**For $\alpha = 0.99$**: $\Phi^{-1}(0.99) \approx 2.326$

## 2. Conditional Value at Risk (CVaR)

### 2.1 Definition

**Definition 4 (Conditional VaR / Expected Shortfall)**

$$
\text{CVaR}_\alpha(L) = \mathbb{E}[L \mid L \geq \text{VaR}_\alpha(L)] \tag{5}
$$

*Alternative representation:*

$$
\text{CVaR}_\alpha(L) = \frac{1}{1-\alpha}\int_\alpha^1 \text{VaR}_u(L) \, du \tag{6}
$$

### 2.2 Coherent Risk Measure

**Theorem 3 (CVaR Coherence)**

*CVaR is a coherent risk measure, satisfying:*

1. *Monotonicity*: $L_1 \leq L_2$ a.s. $\Rightarrow$ $\text{CVaR}_\alpha(L_1) \leq \text{CVaR}_\alpha(L_2)$
2. *Translation invariance*: $\text{CVaR}_\alpha(L + c) = \text{CVaR}_\alpha(L) + c$
3. *Positive homogeneity*: $\text{CVaR}_\alpha(\lambda L) = \lambda \text{CVaR}_\alpha(L)$ for $\lambda > 0$
4. **Subadditivity**: $\text{CVaR}_\alpha(L_1 + L_2) \leq \text{CVaR}_\alpha(L_1) + \text{CVaR}_\alpha(L_2)$

<details>
<summary><strong>Proof of Subadditivity</strong></summary>

Using the representation:
$$
\text{CVaR}_\alpha(L) = \min_{\xi \in \mathbb{R}}\left\{\xi + \frac{1}{1-\alpha}\mathbb{E}[(L - \xi)^+]\right\}
$$

Let $\xi_1^*, \xi_2^*$ be the optimizers for $L_1, L_2$ respectively.

For the sum:
$$
\text{CVaR}_\alpha(L_1 + L_2) \leq (\xi_1^* + \xi_2^*) + \frac{1}{1-\alpha}\mathbb{E}[(L_1 + L_2 - \xi_1^* - \xi_2^*)^+]
$$

Since $(a + b)^+ \leq a^+ + b^+$:
$$
\leq \xi_1^* + \frac{1}{1-\alpha}\mathbb{E}[(L_1 - \xi_1^*)^+] + \xi_2^* + \frac{1}{1-\alpha}\mathbb{E}[(L_2 - \xi_2^*)^+]
$$

$$
= \text{CVaR}_\alpha(L_1) + \text{CVaR}_\alpha(L_2)
$$

$\square$
</details>

### 2.3 Closed-Form for Normal Distribution

**Lemma 2.1 (Normal CVaR)**

*For $L \sim \mathcal{N}(\mu, \sigma^2)$:*

$$
\text{CVaR}_\alpha(L) = \mu + \sigma \frac{\phi(\Phi^{-1}(\alpha))}{1 - \alpha} \tag{7}
$$

*where $\phi$ is the standard normal PDF.*

<details>
<summary><strong>Proof</strong></summary>

For $Z = (L - \mu)/\sigma \sim \mathcal{N}(0, 1)$:
$$
\text{CVaR}_\alpha(L) = \mu + \sigma \mathbb{E}[Z \mid Z \geq \Phi^{-1}(\alpha)]
$$

Let $z_\alpha = \Phi^{-1}(\alpha)$. Then:
$$
\mathbb{E}[Z \mid Z \geq z_\alpha] = \frac{\int_{z_\alpha}^\infty z\phi(z)dz}{\mathbb{P}(Z \geq z_\alpha)}
$$

The numerator:
$$
\int_{z_\alpha}^\infty z\phi(z)dz = \int_{z_\alpha}^\infty z \frac{1}{\sqrt{2\pi}}e^{-z^2/2}dz = \frac{1}{\sqrt{2\pi}}e^{-z_\alpha^2/2} = \phi(z_\alpha)
$$

The denominator is $1 - \alpha$.

Therefore:
$$
\text{CVaR}_\alpha(L) = \mu + \sigma \frac{\phi(\Phi^{-1}(\alpha))}{1 - \alpha}
$$

$\square$
</details>

**Example**: For $\mu = 10\text{M}$, $\sigma = 5\text{M}$, $\alpha = 0.99$:
- $\Phi^{-1}(0.99) = 2.326$
- $\phi(2.326) = 0.0267$
- $\text{CVaR}_{0.99} = 10 + 5 \times \frac{0.0267}{0.01} = 10 + 13.35 = 23.35\text{M}$

### 2.4 Monte Carlo Estimation

**Algorithm 1: CVaR Estimation**

```
Input: N samples {L_1, ..., L_N}, confidence level α
Output: CVaR estimate

1. Sort samples: L_(1) ≤ L_(2) ≤ ... ≤ L_(N)
2. k = ceil(N * α)
3. CVaR = mean(L_(k), L_(k+1), ..., L_(N))
```

**Theorem 4 (Estimation Convergence)**

*The MC estimator $\widehat{\text{CVaR}}_\alpha$ satisfies:*

$$
\sqrt{N}(\widehat{\text{CVaR}}_\alpha - \text{CVaR}_\alpha) \xrightarrow{d} \mathcal{N}(0, \sigma_{\text{CVaR}}^2) \tag{8}
$$

*where:*

$$
\sigma_{\text{CVaR}}^2 = \frac{1}{(1-\alpha)^2}\text{Var}(L \cdot \mathbf{1}_{L \geq \text{VaR}_\alpha}) \tag{9}
$$

## 3. Tail Risk Measures

### 3.1 Tail Conditional Expectation

**Definition 5 (TCE)**

$$
\text{TCE}_\alpha(L) = \mathbb{E}[L \mid L > \text{VaR}_\alpha(L)] \tag{10}
$$

*Note: TCE and CVaR differ when the distribution has atoms at VaR.*

### 3.2 Expected Tail Loss

**Definition 6 (ETL)**

$$
\text{ETL}_\alpha(L) = \frac{1}{1-\alpha}\mathbb{E}[L \cdot \mathbf{1}_{L \geq \text{VaR}_\alpha(L)}] \tag{11}
$$

### 3.3 Relationship

**Lemma 3.1 (Equivalence for Continuous Distributions)**

*If $F_L$ is continuous:*

$$
\text{CVaR}_\alpha(L) = \text{TCE}_\alpha(L) = (1-\alpha)^{-1} \text{ETL}_\alpha(L) \cdot (1-\alpha) = \text{ETL}_\alpha(L) \tag{12}
$$

## 4. Sample Size Requirements

### 4.1 VaR Confidence Intervals

**Theorem 5 (Binomial Confidence for VaR)**

*The empirical quantile $\hat{q}_\alpha$ has distribution:*

$$
\mathbb{P}(\hat{q}_\alpha \leq x) = \sum_{k=0}^{\lfloor n\alpha \rfloor} \binom{n}{k} F(x)^k (1-F(x))^{n-k} \tag{13}
$$

**Lemma 4.1 (Sample Size for VaR Precision)**

*To estimate $\text{VaR}_\alpha$ with relative precision $\epsilon$ at confidence $1-\delta$:*

$$
N \geq \frac{z_{1-\delta/2}^2 \alpha(1-\alpha)}{\epsilon^2 f(\text{VaR}_\alpha)^2} \tag{14}
$$

*where $f$ is the PDF of $L$.*

**Example**: For $\alpha = 0.99$, $\epsilon = 0.05$, $\delta = 0.05$:
- Need $N \approx 10,000$ samples for 5% precision at 95% confidence

### 4.2 CVaR Confidence Intervals

**Theorem 6 (CVaR Standard Error)**

*For i.i.d. samples:*

$$
\text{SE}(\widehat{\text{CVaR}}_\alpha) = \frac{\sigma_{\text{tail}}}{\sqrt{N(1-\alpha)}} \tag{15}
$$

*where $\sigma_{\text{tail}} = \sqrt{\text{Var}(L \mid L \geq \text{VaR}_\alpha)}$.*

### 4.3 Effective Sample Size

**Definition 7 (Tail Sample Count)**

*The effective number of samples in the tail:*

$$
N_{\text{eff}} = N(1 - \alpha) \tag{16}
$$

**Minimum Requirements**:

| Confidence | Tail Probability | Min $N$ for 30 tail samples |
|------------|------------------|----------------------------|
| 95% | 5% | 600 |
| 99% | 1% | 3,000 |
| 99.9% | 0.1% | 30,000 |

## 5. Backtesting

### 5.1 Kupiec Test

**Definition 8 (Kupiec Test)**

*For $n$ observations and $x$ VaR exceedances (violations), test:*
- $H_0$: True violation rate = $1 - \alpha$
- $H_1$: True violation rate $\neq 1 - \alpha$

**Test Statistic**:

$$
\text{LR}_{\text{uc}} = -2\log\left[\frac{(1-\alpha)^{n-x}\alpha^x}{\hat{p}^{n-x}(1-\hat{p})^x}\right] \tag{17}
$$

*where $\hat{p} = x/n$.*

**Distribution**: Under $H_0$, $\text{LR}_{\text{uc}} \sim \chi^2_1$.

**Decision Rule**: Reject $H_0$ if $\text{LR}_{\text{uc}} > \chi^2_{1,1-\delta}$.

### 5.2 Christoffersen Test

**Definition 9 (Independence Test)**

*Test whether violations cluster:*

$$
\text{LR}_{\text{ind}} = -2\log\left[\frac{(1-\pi)^{n_{00}+n_{10}}\pi^{n_{01}+n_{11}}}{(1-\pi_0)^{n_{00}}\pi_0^{n_{01}}(1-\pi_1)^{n_{10}}\pi_1^{n_{11}}}\right] \tag{18}
$$

*where:*
- $n_{ij}$ = transitions from state $i$ to state $j$
- $\pi_i = n_{i1}/(n_{i0} + n_{i1})$
- $\pi = (n_{01} + n_{11})/n$

**Distribution**: Under $H_0$ (independence), $\text{LR}_{\text{ind}} \sim \chi^2_1$.

### 5.3 Combined Test

**Definition 10 (Conditional Coverage Test)**

$$
\text{LR}_{\text{cc}} = \text{LR}_{\text{uc}} + \text{LR}_{\text{ind}} \tag{19}
$$

**Distribution**: Under $H_0$, $\text{LR}_{\text{cc}} \sim \chi^2_2$.

### 5.4 Traffic Light System

**Definition 11 (Basel Traffic Light)**

| Zone | Exceedances (250 days, 99% VaR) | Action |
|------|--------------------------------|--------|
| Green | 0-4 | No action |
| Yellow | 5-9 | Increased monitoring |
| Red | 10+ | Capital add-on |

**Probabilities** (under correct model):
- Green: $\sum_{k=0}^{4} \binom{250}{k}(0.01)^k(0.99)^{250-k} \approx 89\%$
- Yellow: $\approx 10.5\%$
- Red: $\approx 0.5\%$

## 6. Variance Decomposition

### 6.1 Expected Loss Components

**Definition 12 (Loss Decomposition)**

$$
L = \underbrace{EL}_{\text{Expected}} + \underbrace{UL}_{\text{Unexpected}} \tag{20}
$$

*where:*
- $EL = \mathbb{E}[L]$ (expected loss)
- $UL = L - EL$ (unexpected loss)

### 6.2 Capital Requirements

**Definition 13 (Economic Capital)**

$$
EC_\alpha = \text{VaR}_\alpha(L) - \mathbb{E}[L] \tag{21}
$$

**Definition 14 (Risk Capital via CVaR)**

$$
RC_\alpha = \text{CVaR}_\alpha(L) - \mathbb{E}[L] \tag{22}
$$

### 6.3 Portfolio Variance Decomposition

**Theorem 7 (Marginal Contribution to VaR)**

*For portfolio loss $L = \sum_i w_i L_i$:*

$$
\text{MVaR}_i = \frac{\partial \text{VaR}_\alpha(L)}{\partial w_i} \approx \frac{\text{Cov}(L_i, L)}{\sigma_L} \cdot \Phi^{-1}(\alpha) \tag{23}
$$

*under normality assumption.*

**Euler Decomposition**:

$$
\text{VaR}_\alpha(L) = \sum_i w_i \cdot \text{MVaR}_i \tag{24}
$$

## 7. Numerical Example

### 7.1 Portfolio Setup

- 500 loans, total exposure $500M
- Expected default rate: 3%
- LGD: 40%
- Asset correlation: 15%

### 7.2 Risk Metric Computation

**Monte Carlo (10,000 simulations)**:

| Metric | Value | As % of Exposure |
|--------|-------|------------------|
| Expected Loss | $6.0M | 1.2% |
| Std Dev | $4.5M | 0.9% |
| VaR 95% | $12.5M | 2.5% |
| VaR 99% | $18.2M | 3.6% |
| CVaR 99% | $22.8M | 4.6% |

**Capital Calculations**:
- Economic Capital (99%): $18.2M - $6.0M = $12.2M$
- Risk Capital (99%): $22.8M - $6.0M = $16.8M$

### 7.3 Backtest Results

After 250 days with 99% VaR:
- Expected violations: 2.5
- Observed violations: 4

**Kupiec Test**:
$$
\text{LR}_{\text{uc}} = -2\log\left[\frac{0.01^4 \cdot 0.99^{246}}{(4/250)^4 \cdot (246/250)^{246}}\right] = 1.47
$$

$\chi^2_1(0.95) = 3.84$

**Conclusion**: Do not reject $H_0$ (model acceptable).

## 8. Implementation

### 8.1 Python Functions

```python
import numpy as np
from scipy import stats

def var_historical(losses, alpha):
    """Historical VaR."""
    return np.percentile(losses, alpha * 100)

def cvar_historical(losses, alpha):
    """Historical CVaR."""
    var = var_historical(losses, alpha)
    return losses[losses >= var].mean()

def var_parametric(mu, sigma, alpha):
    """Parametric VaR (Normal)."""
    return mu + sigma * stats.norm.ppf(alpha)

def cvar_parametric(mu, sigma, alpha):
    """Parametric CVaR (Normal)."""
    z = stats.norm.ppf(alpha)
    return mu + sigma * stats.norm.pdf(z) / (1 - alpha)
```

### 8.2 Backtest Implementation

```python
def kupiec_test(violations, n_obs, alpha):
    """Kupiec unconditional coverage test."""
    p_model = 1 - alpha
    x = violations
    p_hat = x / n_obs

    lr_num = (p_model ** (n_obs - x)) * ((1 - p_model) ** x)
    lr_den = (p_hat ** (n_obs - x)) * ((1 - p_hat) ** x)

    lr_stat = -2 * np.log(lr_num / lr_den)
    p_value = 1 - stats.chi2.cdf(lr_stat, 1)

    return lr_stat, p_value
```

## References

1. Artzner, P., et al. (1999). Coherent measures of risk. *Mathematical Finance*.
2. Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional value-at-risk. *Journal of Risk*.
3. Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *Journal of Derivatives*.
4. Christoffersen, P. F. (1998). Evaluating interval forecasts. *International Economic Review*.
5. Basel Committee on Banking Supervision. (2019). Minimum capital requirements for market risk.

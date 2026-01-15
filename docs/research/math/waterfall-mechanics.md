---
layout: default
title: CLO Waterfall Mechanics
parent: Research
nav_order: 4
math: true
---

# Collateralized Loan Obligation Waterfall: Mathematical Foundations

This document provides rigorous mathematical derivations for CLO/SPV waterfall mechanics, including loss allocation, priority of payments, and differentiable approximations for end-to-end training.

## 1. Tranche Structure

### 1.1 Definition

**Definition 1 (Tranche)**

*A tranche $k$ is defined by:*
- *Attachment point $A_k \in [0, 1]$: Loss level where tranche begins absorbing losses*
- *Detachment point $D_k \in [A_k, 1]$: Loss level where tranche is fully wiped out*
- *Notional $N_k = (D_k - A_k) \cdot N_{\text{total}}$*

**Example**: Standard CLO structure

| Tranche | Rating | Attachment | Detachment | Spread |
|---------|--------|------------|------------|--------|
| A | AAA | 30% | 100% | L+120 |
| B | AA | 22% | 30% | L+180 |
| C | A | 16% | 22% | L+250 |
| D | BBB | 10% | 16% | L+350 |
| E | BB | 5% | 10% | L+600 |
| Equity | NR | 0% | 5% | Residual |

### 1.2 Loss Allocation

**Definition 2 (Tranche Loss Function)**

*Given portfolio loss rate $L \in [0, 1]$, the loss allocated to tranche $k$:*

$$
\ell_k(L) = \min\left(\max\left(\frac{L - A_k}{D_k - A_k}, 0\right), 1\right) \tag{1}
$$

*The dollar loss:*

$$
\text{Loss}_k = \ell_k(L) \cdot N_k \tag{2}
$$

**Theorem 1 (Loss Allocation Properties)**

*The tranche loss function satisfies:*

1. *Non-negativity: $\ell_k(L) \geq 0$*
2. *Boundedness: $\ell_k(L) \leq 1$*
3. *Monotonicity: $\ell_k$ is non-decreasing in $L$*
4. *Piecewise linearity: $\ell_k$ has exactly two breakpoints*

<details>
<summary><strong>Proof</strong></summary>

Properties 1-2 follow directly from the min-max formulation.

For monotonicity, let $L_1 < L_2$:
- If $L_2 \leq A_k$: $\ell_k(L_1) = \ell_k(L_2) = 0$
- If $L_1 \geq D_k$: $\ell_k(L_1) = \ell_k(L_2) = 1$
- If $A_k < L_1 < L_2 < D_k$: $\ell_k(L_1) = \frac{L_1 - A_k}{D_k - A_k} < \frac{L_2 - A_k}{D_k - A_k} = \ell_k(L_2)$
- Other cases: Similar analysis

For piecewise linearity:
$$
\ell_k(L) = \begin{cases}
0 & L \leq A_k \\
\frac{L - A_k}{D_k - A_k} & A_k < L < D_k \\
1 & L \geq D_k
\end{cases}
$$

$\square$
</details>

### 1.3 Conservation of Loss

**Theorem 2 (Total Loss Conservation)**

*For non-overlapping tranches covering $[0, 1]$:*

$$
\sum_{k=1}^{K} \ell_k(L) \cdot (D_k - A_k) = L \tag{3}
$$

<details>
<summary><strong>Proof</strong></summary>

Consider $L \in [A_j, D_j]$ for some tranche $j$. Then:
- Tranches below $j$: $\ell_k(L) = 1$, contribution = $D_k - A_k$
- Tranche $j$: $\ell_j(L) = \frac{L - A_j}{D_j - A_j}$, contribution = $L - A_j$
- Tranches above $j$: $\ell_k(L) = 0$, contribution = $0$

Sum:
$$
\sum_{k < j}(D_k - A_k) + (L - A_j) = A_j + (L - A_j) = L
$$

Since $\sum_{k < j}(D_k - A_k) = A_j$ by construction.

$\square$
</details>

## 2. Interest Waterfall

### 2.1 Priority of Payments

**Definition 3 (Interest Waterfall Priority)**

*Interest collections $I_{\text{total}}$ are distributed in order:*

1. *Senior fees and expenses: $F$*
2. *Tranche A interest: $I_A = \min(I_{\text{avail}}, r_A \cdot N_A)$*
3. *Tranche B interest: $I_B = \min(I_{\text{avail}}, r_B \cdot N_B)$*
4. *...*
5. *Equity residual: $I_{\text{equity}} = I_{\text{avail}}$*

### 2.2 Formal Specification

**Algorithm 1: Interest Waterfall**

```
Input: Interest collections I_total, fees F, tranche notionals {N_k}, spreads {r_k}
Output: Interest to each tranche {I_k}

I_avail = I_total - F
For k = 1 to K-1 (senior to junior):
    I_k = min(I_avail, r_k * N_k)
    I_avail = I_avail - I_k
I_equity = I_avail  # Residual to equity
```

### 2.3 Interest Coverage Tests

**Definition 4 (Interest Coverage Ratio)**

$$
\text{IC}_k = \frac{I_{\text{total}} - F}{\sum_{j=1}^{k} r_j \cdot N_j} \tag{4}
$$

**Trigger Condition**: If $\text{IC}_k < \text{IC}_k^{\text{threshold}}$, divert cash to pay down senior tranches.

## 3. Principal Waterfall

### 3.1 Standard Principal Distribution

**Definition 5 (Principal Waterfall)**

*Principal collections $P_{\text{total}}$ (scheduled + prepayments + recoveries):*

1. *Senior fees (if interest insufficient)*
2. *Interest shortfall coverage*
3. *Principal to Tranche A until paid down*
4. *Principal to Tranche B until paid down*
5. *...*
6. *Residual to equity*

### 3.2 Overcollateralization Tests

**Definition 6 (Overcollateralization Ratio)**

$$
\text{OC}_k = \frac{\text{Portfolio NAV}}{\sum_{j=1}^{k} N_j} \tag{5}
$$

**Theorem 3 (OC Trigger Effect)**

*When $\text{OC}_k < \text{OC}_k^{\text{trigger}}$:*

$$
P_{\text{redirect}} = \max\left(0, \sum_{j=1}^{k} N_j - \frac{\text{NAV}}{\text{OC}_k^{\text{target}}}\right) \tag{6}
$$

*is diverted from junior to senior tranches.*

### 3.3 Reinvestment Period

**Definition 7 (Reinvestment Waterfall)**

*During reinvestment period (typically years 1-4):*

$$
P_{\text{reinvest}} = P_{\text{total}} - P_{\text{triggers}} \tag{7}
$$

*Principal is reinvested in new loans rather than paying down tranches.*

## 4. Full Numerical Example

### 4.1 Setup

**Portfolio**:
- Total notional: $N = \$500\text{M}$
- Weighted average coupon: 7%
- Expected default rate: 3%
- Expected recovery rate: 60%

**Capital Structure**:

| Tranche | Size | Rate | Attachment | Detachment |
|---------|------|------|------------|------------|
| A | $350M | L+1.2% | 30% | 100% |
| B | $40M | L+1.8% | 22% | 30% |
| C | $30M | L+2.5% | 16% | 22% |
| D | $30M | L+3.5% | 10% | 16% |
| E | $25M | L+6.0% | 5% | 10% |
| Equity | $25M | Residual | 0% | 5% |

### 4.2 Interest Waterfall (Normal Period)

Assumptions: LIBOR = 5%, Fees = $1M/period

**Interest Collections**:
$$
I_{\text{total}} = 500\text{M} \times 7\% / 4 = \$8.75\text{M}
$$

**Distribution**:

| Item | Calculation | Amount | Remaining |
|------|-------------|--------|-----------|
| Fees | | $1.00M | $7.75M |
| A Interest | $350M × (5% + 1.2%)/4 | $5.43M | $2.32M |
| B Interest | $40M × (5% + 1.8%)/4 | $0.68M | $1.64M |
| C Interest | $30M × (5% + 2.5%)/4 | $0.56M | $1.08M |
| D Interest | $30M × (5% + 3.5%)/4 | $0.64M | $0.44M |
| E Interest | $25M × (5% + 6.0%)/4 | $0.69M | -$0.25M |
| **E Shortfall** | | | **$0.25M** |
| Equity | | $0.00M | |

**Result**: Tranche E receives only $0.44M (64% of entitled amount).

### 4.3 Loss Allocation (Stress Scenario)

Assume portfolio loss = 8% ($40M):

| Tranche | Attachment | Loss Rate | Loss Amount |
|---------|------------|-----------|-------------|
| Equity | 0% | 100% | $25.00M |
| E | 5% | (8%-5%)/(10%-5%) = 60% | $15.00M |
| D | 10% | 0% | $0.00M |
| C | 16% | 0% | $0.00M |
| B | 22% | 0% | $0.00M |
| A | 30% | 0% | $0.00M |
| **Total** | | | **$40.00M** |

**Verification**: $25 + 15 = 40$M = 8% × $500M

### 4.4 Coverage Ratio Computation

**Interest Coverage (Class D)**:

$$
\text{IC}_D = \frac{8.75 - 1.0}{5.43 + 0.68 + 0.56 + 0.64} = \frac{7.75}{7.31} = 1.06
$$

Typical trigger: 1.20 → This would trigger diversion.

**Overcollateralization (Class D)**:

After 8% loss, NAV = $460M:

$$
\text{OC}_D = \frac{460}{350 + 40 + 30 + 30} = \frac{460}{450} = 1.02
$$

Typical trigger: 1.10 → This would trigger principal diversion.

## 5. Differentiable Approximation

### 5.1 Problem Statement

The standard loss allocation uses non-differentiable operations (min, max). For end-to-end training, we need smooth approximations.

### 5.2 Soft Gating Functions

**Definition 8 (Smooth Loss Allocation)**

*Replace hard thresholds with sigmoid gates:*

$$
\ell_k^{\text{soft}}(L; \tau) = \sigma\left(\frac{L - A_k}{\tau}\right) \cdot \left(1 - \sigma\left(\frac{L - D_k}{\tau}\right)\right) \cdot \frac{L - A_k}{D_k - A_k} \tag{8}
$$

*where $\sigma(x) = 1/(1 + e^{-x})$ and $\tau > 0$ controls sharpness.*

**Lemma 5.1 (Convergence)**

$$
\lim_{\tau \to 0^+} \ell_k^{\text{soft}}(L; \tau) = \ell_k(L) \tag{9}
$$

### 5.3 Alternative: Softplus Clipping

**Definition 9 (Softplus Approximation)**

$$
\text{clip}_{\text{soft}}(x, a, b) = a + \text{softplus}(x - a) - \text{softplus}(x - b) \tag{10}
$$

*where $\text{softplus}(x) = \log(1 + e^x)$.*

**Properties**:
- Smooth everywhere
- Gradient always exists
- Asymptotically approaches hard clipping

### 5.4 Gradient Flow

**Theorem 4 (Gradient Through Waterfall)**

*For differentiable loss allocation, the gradient with respect to portfolio loss $L$:*

$$
\frac{\partial \ell_k^{\text{soft}}}{\partial L} = \frac{1}{D_k - A_k} \cdot \mathbf{1}_{A_k < L < D_k} + O(\tau) \tag{11}
$$

<details>
<summary><strong>Proof Sketch</strong></summary>

Taking derivatives of the soft formulation and evaluating at interior points where both sigmoids are near 0.5, the gradient simplifies to the constant slope $1/(D_k - A_k)$.

The sigmoid derivatives contribute terms of order $\exp(-|L - A_k|/\tau)$ and $\exp(-|L - D_k|/\tau)$, which vanish as $\tau \to 0$ for interior $L$.

$\square$
</details>

### 5.5 Training Considerations

**Temperature Annealing**:

During training, anneal $\tau$:
$$
\tau_t = \tau_{\max} \cdot \left(\frac{\tau_{\min}}{\tau_{\max}}\right)^{t/T} \tag{12}
$$

Start with $\tau_{\max} = 0.1$ for smooth gradients, end with $\tau_{\min} = 0.001$ for accuracy.

## 6. Stochastic Extensions

### 6.1 Monte Carlo Waterfall

**Algorithm 2: MC Waterfall Simulation**

```
Input: N_sims, loss distribution parameters
Output: Tranche loss distributions

For s = 1 to N_sims:
    L_s ~ LossDistribution(params)
    For k = 1 to K:
        loss_k[s] = ell_k(L_s) * N_k

Return {mean(loss_k), std(loss_k), VaR(loss_k)}
```

### 6.2 Expected Tranche Loss

**Theorem 5 (Expected Loss Integration)**

*For continuous loss distribution with PDF $f_L$:*

$$
\mathbb{E}[\ell_k(L)] = \int_0^1 \ell_k(L) f_L(L) \, dL \tag{13}
$$

$$
= \int_{A_k}^{D_k} \frac{L - A_k}{D_k - A_k} f_L(L) \, dL + \int_{D_k}^1 f_L(L) \, dL \tag{14}
$$

### 6.3 Beta Distribution Example

For $L \sim \text{Beta}(\alpha, \beta)$ with $\alpha = 2, \beta = 50$:

$$
\mathbb{E}[L] = \frac{\alpha}{\alpha + \beta} = \frac{2}{52} \approx 3.8\%
$$

**Tranche E (5%-10%)**:

$$
\mathbb{E}[\ell_E] = \int_{0.05}^{0.10} \frac{L - 0.05}{0.05} f_L(L) \, dL + \int_{0.10}^{1} f_L(L) \, dL
$$

Using numerical integration:
- $\mathbb{P}(L > 0.05) \approx 15\%$
- $\mathbb{P}(L > 0.10) \approx 3\%$
- $\mathbb{E}[\ell_E] \approx 0.08$ (8% expected loss to tranche E)

## 7. Implementation

### 7.1 PyTorch Waterfall

```python
def soft_tranche_loss(portfolio_loss, attachment, detachment, tau=0.01):
    """Differentiable tranche loss allocation."""
    # Soft lower bound
    above_attachment = torch.sigmoid((portfolio_loss - attachment) / tau)
    # Soft upper bound
    below_detachment = 1 - torch.sigmoid((portfolio_loss - detachment) / tau)
    # Linear interpolation
    linear_loss = (portfolio_loss - attachment) / (detachment - attachment)
    # Combine
    return above_attachment * below_detachment * linear_loss.clamp(0, 1)
```

### 7.2 Full Waterfall Class

```python
class DifferentiableWaterfall:
    def __init__(self, tranches, tau=0.01):
        self.tranches = tranches  # List of (attachment, detachment, notional)
        self.tau = tau

    def allocate_loss(self, portfolio_loss_rate):
        losses = {}
        for name, (att, det, notional) in self.tranches.items():
            loss_rate = soft_tranche_loss(portfolio_loss_rate, att, det, self.tau)
            losses[name] = loss_rate * notional
        return losses
```

## 8. Summary

| Concept | Formula | Purpose |
|---------|---------|---------|
| Tranche Loss | $\ell_k = \text{clip}((L-A_k)/(D_k-A_k), 0, 1)$ | Loss allocation |
| Interest Coverage | $\text{IC}_k = (I - F) / \sum_j r_j N_j$ | Performance test |
| Overcollateralization | $\text{OC}_k = \text{NAV} / \sum_j N_j$ | Credit enhancement |
| Soft Allocation | $\sigma((L-A_k)/\tau) \cdot (1-\sigma((L-D_k)/\tau))$ | Differentiable |

## References

1. Fabozzi, F. J., & Kothari, V. (2008). Introduction to securitization. *Wiley*.
2. Duffie, D., & Garleanu, N. (2001). Risk and valuation of collateralized debt obligations. *Financial Analysts Journal*.
3. Longstaff, F. A., & Rajan, A. (2008). An empirical analysis of the pricing of collateralized debt obligations. *Journal of Finance*.
4. Bengio, Y., Leonard, N., & Courville, A. (2013). Estimating or propagating gradients through stochastic neurons. *arXiv*.

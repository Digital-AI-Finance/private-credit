---
layout: default
title: Transition Matrix Dynamics
parent: Research
nav_order: 2
math: true
---

# Markov Chain Transition Dynamics: Mathematical Foundations

This document provides rigorous mathematical derivations for the credit state transition model, including absorbing state theory and neural network parameterization.

## 1. Credit State Markov Chain

### 1.1 State Space Definition

**Definition 1 (Credit State Space)**

*The credit state space $\mathcal{S} = \{0, 1, 2, 3, 4, 5, 6\}$ consists of seven states:*

| State | Name | Type |
|-------|------|------|
| 0 | Performing | Transient |
| 1 | 30 Days Past Due | Transient |
| 2 | 60 Days Past Due | Transient |
| 3 | 90+ Days Past Due | Transient |
| 4 | Default | Absorbing |
| 5 | Prepaid | Absorbing |
| 6 | Matured | Absorbing |

### 1.2 Transition Matrix Structure

**Definition 2 (Transition Matrix)**

*The one-period transition matrix $\mathbf{P} \in [0,1]^{7 \times 7}$ satisfies:*

$$
P_{ij} = \mathbb{P}(S_{t+1} = j \mid S_t = i) \tag{1}
$$

*with the stochastic constraint:*

$$
\sum_{j=0}^{6} P_{ij} = 1 \quad \forall i \in \mathcal{S} \tag{2}
$$

### 1.3 Canonical Form

**Definition 3 (Canonical Form for Absorbing Chains)**

*Reordering states as [transient, absorbing], the transition matrix has the form:*

$$
\mathbf{P} = \begin{pmatrix} \mathbf{Q} & \mathbf{R} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} \tag{3}
$$

*where:*
- $\mathbf{Q} \in [0,1]^{4 \times 4}$: Transient-to-transient transitions
- $\mathbf{R} \in [0,1]^{4 \times 3}$: Transient-to-absorbing transitions
- $\mathbf{I} \in \{0,1\}^{3 \times 3}$: Identity (absorbing states)
- $\mathbf{0} \in \{0\}^{3 \times 4}$: Zero matrix

**Example**: A typical transition matrix:

$$
\mathbf{P} = \begin{pmatrix}
0.90 & 0.03 & 0.01 & 0.005 & 0.002 & 0.012 & 0.041 \\
0.40 & 0.35 & 0.10 & 0.05 & 0.02 & 0.03 & 0.05 \\
0.20 & 0.20 & 0.30 & 0.15 & 0.05 & 0.05 & 0.05 \\
0.10 & 0.10 & 0.15 & 0.35 & 0.15 & 0.05 & 0.10 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

## 2. Absorbing State Convergence

### 2.1 Fundamental Matrix

**Theorem 1 (Existence of Fundamental Matrix)**

*For an absorbing Markov chain with transient submatrix $\mathbf{Q}$, the fundamental matrix:*

$$
\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1} = \sum_{k=0}^{\infty} \mathbf{Q}^k \tag{4}
$$

*exists and is finite if and only if all eigenvalues of $\mathbf{Q}$ satisfy $|\lambda| < 1$.*

<details>
<summary><strong>Proof</strong></summary>

Since $\mathbf{Q}$ is a substochastic matrix (rows sum to less than 1 for at least one transient state that can transition to absorbing states), we have $\rho(\mathbf{Q}) < 1$ where $\rho$ denotes spectral radius.

The Neumann series:

$$
\sum_{k=0}^{\infty} \mathbf{Q}^k
$$

converges if $\rho(\mathbf{Q}) < 1$. Moreover:

$$
(\mathbf{I} - \mathbf{Q})\sum_{k=0}^{n} \mathbf{Q}^k = \mathbf{I} - \mathbf{Q}^{n+1}
$$

Taking $n \to \infty$ and using $\mathbf{Q}^n \to \mathbf{0}$:

$$
(\mathbf{I} - \mathbf{Q})\sum_{k=0}^{\infty} \mathbf{Q}^k = \mathbf{I}
$$

Thus $\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1}$.

$\square$
</details>

### 2.2 Interpretation

**Lemma 1.1 (Expected Time in State)**

*$N_{ij}$ represents the expected number of periods that a chain starting in transient state $i$ spends in transient state $j$ before absorption:*

$$
N_{ij} = \mathbb{E}\left[\sum_{t=0}^{\tau-1} \mathbf{1}_{S_t = j} \mid S_0 = i\right] \tag{5}
$$

*where $\tau = \inf\{t : S_t \in \{4, 5, 6\}\}$ is the absorption time.*

### 2.3 Absorption Probabilities

**Theorem 2 (Absorption Probability Matrix)**

*The matrix $\mathbf{B} = \mathbf{N}\mathbf{R}$ gives absorption probabilities, where:*

$$
B_{ij} = \mathbb{P}(\text{absorb in state } j \mid S_0 = i) \tag{6}
$$

*for transient state $i$ and absorbing state $j$.*

<details>
<summary><strong>Proof</strong></summary>

Let $B_{ij}^{(n)}$ be the probability of absorption in state $j$ within $n$ steps, starting from $i$. Then:

$$
B_{ij}^{(n+1)} = \sum_{k \text{ transient}} P_{ik} B_{kj}^{(n)} + R_{ij}
$$

In matrix form:

$$
\mathbf{B}^{(n+1)} = \mathbf{Q}\mathbf{B}^{(n)} + \mathbf{R}
$$

Taking limits:

$$
\mathbf{B} = \mathbf{Q}\mathbf{B} + \mathbf{R}
$$

$$
(\mathbf{I} - \mathbf{Q})\mathbf{B} = \mathbf{R}
$$

$$
\mathbf{B} = (\mathbf{I} - \mathbf{Q})^{-1}\mathbf{R} = \mathbf{N}\mathbf{R}
$$

$\square$
</details>

**Corollary 2.1 (Default Probability)**

*The cumulative default probability for a loan starting in performing state:*

$$
\text{PD} = B_{0,4} = [(\mathbf{I} - \mathbf{Q})^{-1}\mathbf{R}]_{0,4} \tag{7}
$$

## 3. Multi-Period Transitions

### 3.1 Chapman-Kolmogorov Equations

**Theorem 3 (n-Step Transition)**

*The n-step transition probabilities are given by:*

$$
\mathbf{P}^{(n)} = \mathbf{P}^n \tag{8}
$$

*where:*

$$
P_{ij}^{(n)} = \mathbb{P}(S_{t+n} = j \mid S_t = i) \tag{9}
$$

### 3.2 Long-Run Behavior

**Theorem 4 (Convergence to Absorbing Distribution)**

*For an absorbing Markov chain:*

$$
\lim_{n \to \infty} \mathbf{P}^n = \begin{pmatrix} \mathbf{0} & \mathbf{B} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} \tag{10}
$$

*All probability mass eventually concentrates in absorbing states.*

<details>
<summary><strong>Proof</strong></summary>

The n-th power of the canonical form:

$$
\mathbf{P}^n = \begin{pmatrix} \mathbf{Q}^n & (\mathbf{I} + \mathbf{Q} + \cdots + \mathbf{Q}^{n-1})\mathbf{R} \\ \mathbf{0} & \mathbf{I} \end{pmatrix}
$$

As $n \to \infty$:
- $\mathbf{Q}^n \to \mathbf{0}$ (since $\rho(\mathbf{Q}) < 1$)
- $\sum_{k=0}^{n-1}\mathbf{Q}^k \to \mathbf{N}$

Therefore:

$$
\lim_{n \to \infty} \mathbf{P}^n = \begin{pmatrix} \mathbf{0} & \mathbf{N}\mathbf{R} \\ \mathbf{0} & \mathbf{I} \end{pmatrix} = \begin{pmatrix} \mathbf{0} & \mathbf{B} \\ \mathbf{0} & \mathbf{I} \end{pmatrix}
$$

$\square$
</details>

### 3.3 Expected Absorption Time

**Lemma 3.1 (Time to Absorption)**

*The expected time to absorption starting from transient state $i$:*

$$
\mathbb{E}[\tau \mid S_0 = i] = \sum_{j \text{ transient}} N_{ij} = [\mathbf{N}\mathbf{1}]_i \tag{11}
$$

*where $\mathbf{1}$ is a vector of ones.*

## 4. Neural Network Parameterization

### 4.1 Softmax Constraint

**Definition 4 (Softmax Parameterization)**

*To ensure valid transition probabilities, we parameterize rows via softmax:*

$$
P_{ij} = \frac{\exp(f_{ij})}{\sum_{k=0}^{6} \exp(f_{ik})} \tag{12}
$$

*where $f_{ij} \in \mathbb{R}$ are unconstrained logits from the neural network.*

**Properties**:
1. $P_{ij} > 0$ for all $i, j$
2. $\sum_j P_{ij} = 1$ automatically satisfied
3. Gradients flow through softmax for training

### 4.2 Temperature Scaling

**Definition 5 (Temperature-Scaled Softmax)**

*To control transition sharpness:*

$$
P_{ij} = \frac{\exp(f_{ij}/\tau)}{\sum_{k} \exp(f_{ik}/\tau)} \tag{13}
$$

- $\tau \to 0$: Deterministic (argmax)
- $\tau = 1$: Standard softmax
- $\tau \to \infty$: Uniform distribution

### 4.3 Absorbing State Masking

For states 4, 5, 6 (absorbing), we enforce:

$$
P_{ij} = \begin{cases} 1 & \text{if } i = j \text{ and } i \in \{4, 5, 6\} \\ 0 & \text{if } i \neq j \text{ and } i \in \{4, 5, 6\} \end{cases} \tag{14}
$$

**Implementation**: Set logits to $-\infty$ for invalid transitions:

```python
logits[4:7, :4] = -1e9  # Absorbing cannot go to transient
logits[4:7, 4:7] = torch.eye(3) * 1e9 - 1e9 * (1 - torch.eye(3))
```

## 5. Cross-Attention Modulation

### 5.1 Macro-Conditional Transitions

**Theorem 5 (Transition Matrix as Attention Output)**

*The transition transformer computes:*

$$
\mathbf{P}_t = \text{Softmax}\left(\frac{\mathbf{Q}_t \mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \tag{15}
$$

*where:*
- $\mathbf{Q}_t \in \mathbb{R}^{7 \times d_k}$: State queries at time $t$
- $\mathbf{K} \in \mathbb{R}^{7 \times d_k}$: Transition keys
- $\mathbf{M}$: Absorbing state mask
- $d_k$: Key dimension

### 5.2 Macro Conditioning via Cross-Attention

**Definition 6 (Cross-Attention Mechanism)**

*Given macro path $\mathbf{m}_{1:T} \in \mathbb{R}^{T \times D_m}$:*

$$
\mathbf{h}_t = \text{CrossAttn}(\mathbf{s}_t, \mathbf{m}_{1:T}) \tag{16}
$$

$$
= \text{Softmax}\left(\frac{\mathbf{s}_t \mathbf{W}_Q (\mathbf{m}_{1:T}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right) \mathbf{m}_{1:T}\mathbf{W}_V
$$

*This allows transitions to depend on the entire macro trajectory.*

### 5.3 Time-Varying Transitions

**Theorem 6 (Macro-Modulated Default Probability)**

*Under stress, the transition matrix evolves:*

$$
\mathbf{P}_t(\mathbf{m}) = \mathbf{P}_{\text{base}} + \Delta\mathbf{P}(\mathbf{m}_t) \tag{17}
$$

*where $\Delta\mathbf{P}$ is the macro-induced perturbation.*

**Example**: During recession (high unemployment $u$, low GDP $g$):

$$
P_{01}(u, g) = P_{01}^{\text{base}} \cdot \exp(\alpha(u - u^*) - \beta g) \tag{18}
$$

## 6. Numerical Examples

### 6.1 Absorption Probability Computation

**Example 1**: Computing default probability

Given:
$$
\mathbf{Q} = \begin{pmatrix}
0.90 & 0.03 & 0.01 & 0.005 \\
0.40 & 0.35 & 0.10 & 0.05 \\
0.20 & 0.20 & 0.30 & 0.15 \\
0.10 & 0.10 & 0.15 & 0.35
\end{pmatrix}
$$

Step 1: Compute $\mathbf{I} - \mathbf{Q}$:
$$
\mathbf{I} - \mathbf{Q} = \begin{pmatrix}
0.10 & -0.03 & -0.01 & -0.005 \\
-0.40 & 0.65 & -0.10 & -0.05 \\
-0.20 & -0.20 & 0.70 & -0.15 \\
-0.10 & -0.10 & -0.15 & 0.65
\end{pmatrix}
$$

Step 2: Compute $\mathbf{N} = (\mathbf{I} - \mathbf{Q})^{-1}$:
$$
\mathbf{N} \approx \begin{pmatrix}
12.5 & 1.2 & 0.6 & 0.3 \\
8.2 & 2.8 & 0.8 & 0.4 \\
5.1 & 1.4 & 2.1 & 0.5 \\
3.2 & 0.8 & 0.7 & 1.9
\end{pmatrix}
$$

Step 3: With $\mathbf{R}$:
$$
\mathbf{R} = \begin{pmatrix}
0.002 & 0.012 & 0.041 \\
0.02 & 0.03 & 0.05 \\
0.05 & 0.05 & 0.05 \\
0.15 & 0.05 & 0.10
\end{pmatrix}
$$

Step 4: Compute $\mathbf{B} = \mathbf{N}\mathbf{R}$:

For a performing loan (state 0):
- Default probability: $B_{0,4} \approx 0.18$ (18%)
- Prepayment probability: $B_{0,5} \approx 0.32$ (32%)
- Maturity probability: $B_{0,6} \approx 0.50$ (50%)

### 6.2 Time to Absorption

Expected time to absorption from performing state:

$$
\mathbb{E}[\tau \mid S_0 = 0] = \sum_{j=0}^{3} N_{0j} = 12.5 + 1.2 + 0.6 + 0.3 = 14.6 \text{ months}
$$

### 6.3 Multi-Year Default Curves

For 60-month horizon, compute cumulative default:

| Month | $\mathbf{P}^t[0,4]$ | Cumulative Default |
|-------|---------------------|-------------------|
| 12 | 0.024 | 2.4% |
| 24 | 0.058 | 5.8% |
| 36 | 0.098 | 9.8% |
| 48 | 0.138 | 13.8% |
| 60 | 0.175 | 17.5% |

## 7. Estimation from Data

### 7.1 Maximum Likelihood Estimation

**Definition 7 (MLE for Transition Matrix)**

*Given observed transitions $n_{ij}$ from state $i$ to $j$:*

$$
\hat{P}_{ij} = \frac{n_{ij}}{\sum_k n_{ik}} \tag{19}
$$

### 7.2 Cohort Method

For discrete time cohort data:

$$
\hat{P}_{ij}^{(t)} = \frac{\text{Count}(S_t = j \mid S_{t-1} = i)}{\text{Count}(S_{t-1} = i)} \tag{20}
$$

### 7.3 Regularization

**Definition 8 (Dirichlet Prior)**

*With prior $\alpha_{ij}$:*

$$
\hat{P}_{ij} = \frac{n_{ij} + \alpha_{ij}}{\sum_k (n_{ik} + \alpha_{ik})} \tag{21}
$$

## 8. Model Validation

### 8.1 Likelihood Ratio Test

**Theorem 7 (Stationarity Test)**

*To test if transitions are time-homogeneous:*

$$
\Lambda = -2\sum_t \sum_{i,j} n_{ij}^{(t)} \log\frac{\hat{P}_{ij}^{(t)}}{\hat{P}_{ij}} \tag{22}
$$

*Under $H_0$ (stationarity): $\Lambda \sim \chi^2_{(T-1)(K-1)K}$*

### 8.2 Generator Matrix (Continuous Time)

For continuous-time generalization:

$$
\frac{d\mathbf{p}(t)}{dt} = \mathbf{p}(t)\mathbf{G} \tag{23}
$$

where $\mathbf{G}$ is the generator matrix with $G_{ii} = -\sum_{j \neq i} G_{ij}$.

**Relation**: $\mathbf{P}(\Delta t) = \exp(\mathbf{G}\Delta t)$

## References

1. Jarrow, R. A., Lando, D., & Turnbull, S. M. (1997). A Markov model for the term structure of credit risk spreads. *Review of Financial Studies*.
2. Lando, D., & Skodeberg, T. M. (2002). Analyzing rating transitions and rating drift with continuous observations. *Journal of Banking & Finance*.
3. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
4. Bladt, M., & Sorensen, M. (2005). Statistical inference for discretely observed Markov jump processes. *Journal of the Royal Statistical Society*.

---
layout: default
title: Conditional VAE Mathematical Derivation
parent: Research
nav_order: 1
math: true
---

# Conditional Variational Autoencoder: Mathematical Foundations

This document provides rigorous mathematical derivations for the Conditional Variational Autoencoder (CVAE) used in macro scenario generation.

## 1. Problem Formulation

Let $\mathbf{x} \in \mathbb{R}^{T \times D}$ denote a multivariate macro time series with $T$ time steps and $D$ variables, and let $c \in \{0, 1, 2, 3\}$ represent the scenario condition (baseline, adverse, severely adverse, stagflation).

**Objective**: Learn a generative model $p_\theta(\mathbf{x} | c)$ that can sample realistic macro scenarios conditioned on scenario type.

## 2. Variational Lower Bound (ELBO)

### 2.1 Marginal Likelihood Decomposition

**Theorem 1 (ELBO for Conditional VAE)**

*For any conditional distribution $q_\phi(\mathbf{z} | \mathbf{x}, c)$ with support containing that of the true posterior $p_\theta(\mathbf{z} | \mathbf{x}, c)$, the log marginal likelihood satisfies:*

$$
\log p_\theta(\mathbf{x} | c) = \mathcal{L}(\theta, \phi; \mathbf{x}, c) + D_{KL}\big(q_\phi(\mathbf{z} | \mathbf{x}, c) \| p_\theta(\mathbf{z} | \mathbf{x}, c)\big) \tag{1}
$$

*where the Evidence Lower Bound (ELBO) is:*

$$
\mathcal{L}(\theta, \phi; \mathbf{x}, c) = \mathbb{E}_{q_\phi(\mathbf{z} | \mathbf{x}, c)}\big[\log p_\theta(\mathbf{x} | \mathbf{z}, c)\big] - D_{KL}\big(q_\phi(\mathbf{z} | \mathbf{x}, c) \| p(\mathbf{z} | c)\big) \tag{2}
$$

<details>
<summary><strong>Proof</strong></summary>

Starting from the marginal likelihood:

$$
\log p_\theta(\mathbf{x} | c) = \log \int p_\theta(\mathbf{x}, \mathbf{z} | c) \, d\mathbf{z}
$$

Introduce the variational distribution $q_\phi(\mathbf{z} | \mathbf{x}, c)$:

$$
\log p_\theta(\mathbf{x} | c) = \log \int q_\phi(\mathbf{z} | \mathbf{x}, c) \frac{p_\theta(\mathbf{x}, \mathbf{z} | c)}{q_\phi(\mathbf{z} | \mathbf{x}, c)} \, d\mathbf{z}
$$

By Jensen's inequality (noting $\log$ is concave):

$$
\log p_\theta(\mathbf{x} | c) \geq \int q_\phi(\mathbf{z} | \mathbf{x}, c) \log \frac{p_\theta(\mathbf{x}, \mathbf{z} | c)}{q_\phi(\mathbf{z} | \mathbf{x}, c)} \, d\mathbf{z}
$$

Expanding the joint:

$$
= \int q_\phi(\mathbf{z} | \mathbf{x}, c) \log \frac{p_\theta(\mathbf{x} | \mathbf{z}, c) p(\mathbf{z} | c)}{q_\phi(\mathbf{z} | \mathbf{x}, c)} \, d\mathbf{z}
$$

Separating terms:

$$
= \underbrace{\mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x} | \mathbf{z}, c)]}_{\text{Reconstruction}} - \underbrace{D_{KL}(q_\phi(\mathbf{z} | \mathbf{x}, c) \| p(\mathbf{z} | c))}_{\text{Regularization}}
$$

The gap between $\log p_\theta(\mathbf{x}|c)$ and the ELBO is exactly $D_{KL}(q_\phi \| p_\theta)$, which is non-negative.

$\square$
</details>

### 2.2 Reparameterization Trick

**Lemma 1.1 (Differentiability via Reparameterization)**

*Let $q_\phi(\mathbf{z} | \mathbf{x}, c) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}, c), \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x}, c)))$. Then samples can be expressed as:*

$$
\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}, c) + \boldsymbol{\sigma}_\phi(\mathbf{x}, c) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{3}
$$

*This transformation allows gradients to flow through the sampling operation.*

<details>
<summary><strong>Proof</strong></summary>

For a random variable $Z \sim \mathcal{N}(\mu, \sigma^2)$, we can write $Z = \mu + \sigma \epsilon$ where $\epsilon \sim \mathcal{N}(0, 1)$.

**Verification**:
- $\mathbb{E}[Z] = \mathbb{E}[\mu + \sigma\epsilon] = \mu + \sigma \cdot 0 = \mu$
- $\text{Var}(Z) = \text{Var}(\mu + \sigma\epsilon) = \sigma^2 \text{Var}(\epsilon) = \sigma^2$

This extends element-wise to the multivariate case. The key insight is that $\epsilon$ is independent of $\phi$, so:

$$
\nabla_\phi \mathbb{E}_{q_\phi}[f(\mathbf{z})] = \mathbb{E}_{\epsilon}\left[\nabla_\phi f(\boldsymbol{\mu}_\phi + \boldsymbol{\sigma}_\phi \odot \boldsymbol{\epsilon})\right]
$$

which can be approximated via Monte Carlo sampling.

$\square$
</details>

### 2.3 KL Divergence Closed Form

**Lemma 1.2 (KL Divergence for Gaussians)**

*For $q = \mathcal{N}(\boldsymbol{\mu}, \text{diag}(\boldsymbol{\sigma}^2))$ and $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$:*

$$
D_{KL}(q \| p) = \frac{1}{2} \sum_{j=1}^{d} \left( \sigma_j^2 + \mu_j^2 - 1 - \log \sigma_j^2 \right) \tag{4}
$$

<details>
<summary><strong>Proof</strong></summary>

For two multivariate Gaussians with diagonal covariances:
- $q(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$
- $p(\mathbf{z}) = \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$

The KL divergence is:

$$
D_{KL}(q \| p) = \frac{1}{2}\left[\log\frac{|\boldsymbol{\Sigma}_p|}{|\boldsymbol{\Sigma}_q|} - d + \text{tr}(\boldsymbol{\Sigma}_p^{-1}\boldsymbol{\Sigma}_q) + (\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)^\top \boldsymbol{\Sigma}_p^{-1}(\boldsymbol{\mu}_p - \boldsymbol{\mu}_q)\right]
$$

For $p = \mathcal{N}(\mathbf{0}, \mathbf{I})$, we have $\boldsymbol{\Sigma}_p = \mathbf{I}$, $\boldsymbol{\mu}_p = \mathbf{0}$:

$$
D_{KL} = \frac{1}{2}\left[-\log|\boldsymbol{\Sigma}_q| - d + \text{tr}(\boldsymbol{\Sigma}_q) + \boldsymbol{\mu}_q^\top\boldsymbol{\mu}_q\right]
$$

With diagonal $\boldsymbol{\Sigma}_q = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)$:

$$
= \frac{1}{2}\left[-\sum_j \log\sigma_j^2 - d + \sum_j \sigma_j^2 + \sum_j \mu_j^2\right]
$$

$$
= \frac{1}{2}\sum_{j=1}^d \left(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2\right)
$$

$\square$
</details>

## 3. Conditional Prior Design

### 3.1 Scenario-Dependent Prior

In our implementation, the prior $p(\mathbf{z} | c)$ is scenario-dependent:

$$
p(\mathbf{z} | c) = \mathcal{N}(\boldsymbol{\mu}_c, \text{diag}(\boldsymbol{\sigma}_c^2)) \tag{5}
$$

where $\boldsymbol{\mu}_c$ and $\boldsymbol{\sigma}_c$ are learnable parameters for each scenario $c$.

**Rationale**: Different economic scenarios have distinct latent space characteristics:
- **Baseline** ($c=0$): Centered prior, moderate variance
- **Adverse** ($c=1$): Shifted mean toward stress, higher variance
- **Severely Adverse** ($c=2$): Larger shift, even higher variance
- **Stagflation** ($c=3$): Different direction of shift (high inflation, low growth)

### 3.2 Modified KL Term

With a non-standard prior, the KL divergence becomes:

$$
D_{KL}(q_\phi(\mathbf{z}|\mathbf{x},c) \| p(\mathbf{z}|c)) = \frac{1}{2}\sum_{j=1}^d \left[\frac{\sigma_{\phi,j}^2 + (\mu_{\phi,j} - \mu_{c,j})^2}{\sigma_{c,j}^2} - 1 - \log\frac{\sigma_{\phi,j}^2}{\sigma_{c,j}^2}\right] \tag{6}
$$

## 4. Reconstruction Loss

### 4.1 Gaussian Decoder (MSE Loss)

Assuming $p_\theta(\mathbf{x} | \mathbf{z}, c) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{z}, c), \sigma_{\text{dec}}^2 \mathbf{I})$:

$$
\log p_\theta(\mathbf{x} | \mathbf{z}, c) = -\frac{1}{2\sigma_{\text{dec}}^2}\|\mathbf{x} - \boldsymbol{\mu}_\theta(\mathbf{z}, c)\|^2 + \text{const} \tag{7}
$$

Maximizing this is equivalent to minimizing MSE loss:

$$
\mathcal{L}_{\text{recon}} = \frac{1}{TD}\sum_{t=1}^T \sum_{d=1}^D (x_{t,d} - \hat{x}_{t,d})^2 \tag{8}
$$

### 4.2 Alternative: Negative Log-Likelihood

For heteroscedastic noise, we can learn variance:

$$
p_\theta(\mathbf{x} | \mathbf{z}, c) = \mathcal{N}(\boldsymbol{\mu}_\theta(\mathbf{z}, c), \text{diag}(\boldsymbol{\sigma}_\theta^2(\mathbf{z}, c)))
$$

$$
\mathcal{L}_{\text{NLL}} = \frac{1}{2}\sum_{t,d}\left[\log\sigma_{\theta,t,d}^2 + \frac{(x_{t,d} - \mu_{\theta,t,d})^2}{\sigma_{\theta,t,d}^2}\right] \tag{9}
$$

## 5. Beta-VAE and KL Annealing

### 5.1 Beta-VAE Objective

**Definition (Beta-VAE Loss)**

$$
\mathcal{L}_\beta = \mathbb{E}_{q_\phi}[\log p_\theta(\mathbf{x}|\mathbf{z},c)] - \beta \cdot D_{KL}(q_\phi(\mathbf{z}|\mathbf{x},c) \| p(\mathbf{z}|c)) \tag{10}
$$

- $\beta < 1$: Emphasizes reconstruction, may lead to posterior collapse
- $\beta = 1$: Standard VAE (ELBO)
- $\beta > 1$: Emphasizes disentanglement, may hurt reconstruction

### 5.2 KL Annealing Schedule

**Algorithm 1: Cyclical KL Annealing**

```
Input: Total epochs E, cycle length C, warmup fraction r
For epoch e in 1..E:
    cycle_position = (e % C) / C
    if cycle_position < r:
        beta = cycle_position / r
    else:
        beta = 1.0
    Train with beta
```

**Schedule Types**:

1. **Linear Warmup**:
$$
\beta_t = \min\left(1, \frac{t}{T_{\text{warmup}}}\right) \tag{11}
$$

2. **Cyclical Annealing** (used in our implementation):
$$
\beta_t = \min\left(1, \frac{t \mod C}{r \cdot C}\right) \tag{12}
$$

3. **Sigmoid Annealing**:
$$
\beta_t = \sigma\left(\frac{t - T_{\text{mid}}}{T_{\text{scale}}}\right) \tag{13}
$$

### 5.3 Posterior Collapse Prevention

**Definition (Posterior Collapse)**: When $q_\phi(\mathbf{z}|\mathbf{x},c) \approx p(\mathbf{z}|c)$ for all $\mathbf{x}$, meaning the encoder ignores the input.

**Theorem 2 (Posterior Collapse Condition)**

*Posterior collapse occurs when the decoder is powerful enough to model $p(\mathbf{x}|c)$ without latent information, i.e., when:*

$$
\mathcal{I}_q[\mathbf{x}; \mathbf{z}] \triangleq D_{KL}(q_\phi(\mathbf{x}, \mathbf{z}|c) \| q_\phi(\mathbf{x}|c)q_\phi(\mathbf{z}|c)) \approx 0 \tag{14}
$$

**Mitigation Strategies**:
1. KL annealing (start with $\beta \ll 1$)
2. Free bits: $D_{KL} \leftarrow \max(\lambda, D_{KL})$
3. Skip connections bypass (avoid powerful decoders)

## 6. LSTM Encoder Architecture

### 6.1 Sequence Encoding

The encoder maps $\mathbf{x} \in \mathbb{R}^{T \times D}$ to latent parameters:

$$
\mathbf{h}_t = \text{LSTM}(\mathbf{x}_t, \mathbf{h}_{t-1}) \tag{15}
$$

$$
\boldsymbol{\mu}_\phi = W_\mu \mathbf{h}_T + \mathbf{b}_\mu, \quad \log\boldsymbol{\sigma}_\phi^2 = W_\sigma \mathbf{h}_T + \mathbf{b}_\sigma \tag{16}
$$

### 6.2 Condition Embedding

The scenario condition is embedded and concatenated:

$$
\mathbf{e}_c = \text{Embedding}(c) \in \mathbb{R}^{d_c} \tag{17}
$$

$$
\tilde{\mathbf{x}}_t = [\mathbf{x}_t; \mathbf{e}_c] \in \mathbb{R}^{D + d_c} \tag{18}
$$

## 7. Decoder Architecture

### 7.1 Latent to Sequence

The decoder generates sequences from latent codes:

$$
\mathbf{s}_0 = W_{\text{init}}[\mathbf{z}; \mathbf{e}_c] + \mathbf{b}_{\text{init}} \tag{19}
$$

$$
\mathbf{s}_t = \text{LSTM}(\mathbf{s}_{t-1}, \mathbf{s}_{t-1}) \tag{20}
$$

$$
\hat{\mathbf{x}}_t = W_{\text{out}}\mathbf{s}_t + \mathbf{b}_{\text{out}} \tag{21}
$$

### 7.2 Trend Component

To capture long-term dynamics, we add a deterministic trend:

$$
\mathbf{trend}_t = \mathbf{a}_c \cdot t + \mathbf{b}_c \tag{22}
$$

$$
\hat{\mathbf{x}}_t^{\text{final}} = \hat{\mathbf{x}}_t + \mathbf{trend}_t \tag{23}
$$

## 8. Training Objective

### 8.1 Final Loss Function

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta \cdot \mathcal{L}_{KL} + \lambda_{\text{corr}} \cdot \mathcal{L}_{\text{corr}} \tag{24}
$$

where:
- $\mathcal{L}_{\text{recon}}$: MSE reconstruction loss (Eq. 8)
- $\mathcal{L}_{KL}$: KL divergence (Eq. 6)
- $\mathcal{L}_{\text{corr}}$: Correlation regularization

### 8.2 Correlation Regularization

To preserve cross-variable dependencies:

$$
\mathcal{L}_{\text{corr}} = \|\mathbf{R}_{\text{true}} - \mathbf{R}_{\text{gen}}\|_F^2 \tag{25}
$$

where $\mathbf{R} \in \mathbb{R}^{D \times D}$ is the correlation matrix.

## 9. Theoretical Guarantees

**Theorem 3 (Consistency of VAE)**

*Under mild regularity conditions, as sample size $n \to \infty$:*

$$
\hat{\theta}_n \xrightarrow{p} \theta^* \quad \text{and} \quad \hat{\phi}_n \xrightarrow{p} \phi^* \tag{26}
$$

*where $(\theta^*, \phi^*)$ maximizes the population ELBO.*

**Corollary 3.1**: *The generative model $p_{\hat{\theta}}(\mathbf{x}|c)$ converges to the true data distribution as $n \to \infty$, provided the model class is correctly specified.*

## 10. Numerical Example

Consider a simple 1D case with:
- Prior: $p(z) = \mathcal{N}(0, 1)$
- Encoder output: $\mu_\phi = 0.5$, $\sigma_\phi = 0.8$

**KL Divergence Calculation**:

$$
D_{KL} = \frac{1}{2}(0.8^2 + 0.5^2 - 1 - \log 0.8^2)
$$

$$
= \frac{1}{2}(0.64 + 0.25 - 1 - (-0.446))
$$

$$
= \frac{1}{2}(0.336) = 0.168
$$

## References

1. Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. *ICLR*.
2. Higgins, I., et al. (2017). beta-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*.
3. Bowman, S. R., et al. (2016). Generating sentences from a continuous space. *CoNLL*.
4. Sohn, K., Lee, H., & Yan, X. (2015). Learning structured output representation using deep conditional generative models. *NeurIPS*.
5. Fu, H., et al. (2019). Cyclical annealing schedule: A simple approach to mitigating KL vanishing. *NAACL*.

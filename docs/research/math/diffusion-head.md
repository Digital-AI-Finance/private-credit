---
layout: default
title: Diffusion Model Derivation
parent: Research
nav_order: 3
math: true
---

# Denoising Diffusion Probabilistic Models: Mathematical Foundations

This document provides rigorous mathematical derivations for the diffusion head used in loan trajectory generation, including score matching and sampling algorithms.

## 1. Forward Diffusion Process

### 1.1 Definition

**Definition 1 (Forward Diffusion)**

*The forward diffusion process gradually adds Gaussian noise to data $\mathbf{x}_0 \sim q(\mathbf{x}_0)$:*

$$
q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I}) \tag{1}
$$

*where $\{\beta_t\}_{t=1}^T$ is the variance schedule with $\beta_t \in (0, 1)$.*

### 1.2 Noise Schedule

**Definition 2 (Linear Schedule)**

*The standard linear schedule:*

$$
\beta_t = \beta_{\text{min}} + \frac{t-1}{T-1}(\beta_{\text{max}} - \beta_{\text{min}}) \tag{2}
$$

*Typical values: $\beta_{\text{min}} = 10^{-4}$, $\beta_{\text{max}} = 0.02$, $T = 1000$.*

**Definition 3 (Cosine Schedule)**

*The cosine schedule (often superior for images):*

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2 \tag{3}
$$

*where $s = 0.008$ is a small offset.*

### 1.3 Cumulative Products

**Definition 4 (Alpha Parameters)**

$$
\alpha_t = 1 - \beta_t \tag{4}
$$

$$
\bar{\alpha}_t = \prod_{s=1}^t \alpha_s \tag{5}
$$

**Lemma 1.1 (Direct Sampling)**

*The noisy sample at any timestep can be computed directly:*

$$
q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I}) \tag{6}
$$

<details>
<summary><strong>Proof</strong></summary>

By induction. Base case ($t=1$):
$$
q(\mathbf{x}_1 | \mathbf{x}_0) = \mathcal{N}(\sqrt{\alpha_1}\mathbf{x}_0, (1-\alpha_1)\mathbf{I}) = \mathcal{N}(\sqrt{\bar{\alpha}_1}\mathbf{x}_0, (1-\bar{\alpha}_1)\mathbf{I})
$$

Inductive step: Assume true for $t-1$. Then:
$$
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1}
$$

$$
\mathbf{x}_t = \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t
$$

$$
= \sqrt{\alpha_t}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1}) + \sqrt{\beta_t}\boldsymbol{\epsilon}_t
$$

$$
= \sqrt{\alpha_t\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{\alpha_t(1-\bar{\alpha}_{t-1})}\boldsymbol{\epsilon}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_t
$$

The variance terms combine (sum of independent Gaussians):
$$
\text{Var} = \alpha_t(1-\bar{\alpha}_{t-1}) + \beta_t = \alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t
$$
$$
= 1 - \alpha_t\bar{\alpha}_{t-1} = 1 - \bar{\alpha}_t
$$

Therefore:
$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}
$$

$\square$
</details>

**Corollary 1.1 (Reparameterization)**

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \tag{7}
$$

## 2. Reverse Process

### 2.1 Reverse Transition

**Definition 5 (Reverse Process)**

*The reverse process is defined as:*

$$
p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \sigma_t^2\mathbf{I}) \tag{8}
$$

### 2.2 True Posterior

**Theorem 1 (Tractable Posterior)**

*When conditioned on $\mathbf{x}_0$, the reverse transition has closed form:*

$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t\mathbf{I}) \tag{9}
$$

*where:*

$$
\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t \tag{10}
$$

$$
\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}\beta_t \tag{11}
$$

<details>
<summary><strong>Proof</strong></summary>

Using Bayes' rule:
$$
q(\mathbf{x}_{t-1} | \mathbf{x}_t, \mathbf{x}_0) = \frac{q(\mathbf{x}_t | \mathbf{x}_{t-1}, \mathbf{x}_0) q(\mathbf{x}_{t-1} | \mathbf{x}_0)}{q(\mathbf{x}_t | \mathbf{x}_0)}
$$

All three terms are Gaussian:
- $q(\mathbf{x}_t | \mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$
- $q(\mathbf{x}_{t-1} | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$
- $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$

The product of Gaussians is Gaussian. The precision (inverse variance) adds:
$$
\frac{1}{\tilde{\beta}_t} = \frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}
$$

$$
= \frac{\alpha_t(1 - \bar{\alpha}_{t-1}) + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})}
$$

$$
= \frac{\alpha_t - \alpha_t\bar{\alpha}_{t-1} + 1 - \alpha_t}{\beta_t(1 - \bar{\alpha}_{t-1})}
$$

$$
= \frac{1 - \bar{\alpha}_t}{\beta_t(1 - \bar{\alpha}_{t-1})}
$$

Therefore:
$$
\tilde{\beta}_t = \frac{\beta_t(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}
$$

The mean is computed via precision-weighted average of means:
$$
\tilde{\boldsymbol{\mu}}_t = \tilde{\beta}_t\left(\frac{\alpha_t}{\beta_t}\mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}}\mathbf{x}_0\right)
$$

After algebra (substituting $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$):
$$
\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\mathbf{x}_t
$$

$\square$
</details>

### 2.3 Mean Parameterization

**Lemma 2.1 (Noise Prediction Form)**

*Expressing $\mathbf{x}_0$ in terms of noise:*

$$
\mathbf{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}) \tag{12}
$$

*Substituting into the posterior mean:*

$$
\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}\right) \tag{13}
$$

**Corollary 2.1 (Learned Mean)**

*The network predicts noise $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$:*

$$
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) \tag{14}
$$

## 3. Training Objective

### 3.1 Variational Lower Bound

**Theorem 2 (DDPM Loss Decomposition)**

*The variational lower bound decomposes as:*

$$
\mathcal{L} = \mathbb{E}_q\left[-\log p(\mathbf{x}_T) + \sum_{t=2}^T D_{KL}(q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \| p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)) - \log p_\theta(\mathbf{x}_0|\mathbf{x}_1)\right] \tag{15}
$$

### 3.2 Simplified Objective

**Theorem 3 (Simplified Training Loss)**

*The simplified training objective:*

$$
\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right] \tag{16}
$$

*where $t \sim \text{Uniform}\{1, \ldots, T\}$ and $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.*

<details>
<summary><strong>Proof</strong></summary>

The KL divergence between two Gaussians with same variance:
$$
D_{KL}(q \| p_\theta) = \frac{1}{2\sigma_t^2}\|\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta\|^2
$$

Substituting the parameterizations:
$$
\tilde{\boldsymbol{\mu}}_t - \boldsymbol{\mu}_\theta = \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1 - \bar{\alpha}_t}}(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \boldsymbol{\epsilon})
$$

Therefore:
$$
D_{KL} \propto \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2
$$

Dropping the time-dependent weighting factor gives the simplified objective.

$\square$
</details>

### 3.3 Score Matching Connection

**Theorem 4 (Equivalence to Score Matching)**

*Denoising score matching is equivalent to:*

$$
\mathcal{L}_{\text{DSM}} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\mathbf{s}_\theta(\mathbf{x}_t, t) - \nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t | \mathbf{x}_0)\|^2\right] \tag{17}
$$

*The score function relates to noise:*

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{\boldsymbol{\epsilon}}{\sqrt{1 - \bar{\alpha}_t}} \tag{18}
$$

<details>
<summary><strong>Proof</strong></summary>

From $q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$:

$$
\log q(\mathbf{x}_t | \mathbf{x}_0) = -\frac{1}{2(1-\bar{\alpha}_t)}\|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2 + \text{const}
$$

Taking gradient:
$$
\nabla_{\mathbf{x}_t}\log q = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1 - \bar{\alpha}_t}
$$

Since $\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0 = \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$:
$$
\nabla_{\mathbf{x}_t}\log q = -\frac{\sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}}{1 - \bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}
$$

$\square$
</details>

**Corollary 4.1 (Score-Noise Relationship)**

$$
\mathbf{s}_\theta(\mathbf{x}_t, t) = -\frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \tag{19}
$$

## 4. Sampling Algorithm

### 4.1 DDPM Sampling

**Algorithm 1: DDPM Sampling**

```
Input: Trained noise predictor ε_θ
Output: Sample x_0

1. Sample x_T ~ N(0, I)
2. For t = T, T-1, ..., 1:
   a. z ~ N(0, I) if t > 1, else z = 0
   b. x_{t-1} = (1/√α_t)(x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + σ_t · z
3. Return x_0
```

**Variance Choices**:
- $\sigma_t^2 = \beta_t$ (standard)
- $\sigma_t^2 = \tilde{\beta}_t$ (posterior variance)
- $\sigma_t^2 = 0$ (deterministic DDIM)

### 4.2 Closed-Form Updates

**Theorem 5 (Sampling Update Rule)**

*Each reverse step:*

$$
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t\mathbf{z} \tag{20}
$$

### 4.3 DDIM (Deterministic Sampling)

**Theorem 6 (DDIM Update)**

*For accelerated sampling with stride $\tau$:*

$$
\mathbf{x}_{t-\tau} = \sqrt{\bar{\alpha}_{t-\tau}}\left(\frac{\mathbf{x}_t - \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}_\theta}{\sqrt{\bar{\alpha}_t}}\right) + \sqrt{1 - \bar{\alpha}_{t-\tau}}\boldsymbol{\epsilon}_\theta \tag{21}
$$

*This allows generating samples with fewer steps (e.g., 50 instead of 1000).*

## 5. Conditional Generation

### 5.1 Classifier-Free Guidance

**Definition 6 (Conditional Score)**

*The conditional score combines unconditional and conditional predictions:*

$$
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, c) = (1 + w)\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c) - w\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing) \tag{22}
$$

*where $w > 0$ is the guidance scale and $\varnothing$ denotes null conditioning.*

### 5.2 Training with Dropout

During training, drop conditioning with probability $p_{\text{uncond}}$:

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, c') \text{ where } c' = \begin{cases} c & \text{with prob } 1 - p_{\text{uncond}} \\ \varnothing & \text{with prob } p_{\text{uncond}} \end{cases} \tag{23}
$$

### 5.3 Application to Loan Trajectories

In our model, conditioning includes:
- Loan characteristics (balance, rate, term)
- Macro scenario at time $t$
- Current credit state

$$
\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \to \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}_{\text{loan}}, \mathbf{m}_t, s_t) \tag{24}
$$

## 6. Variance Derivation

### 6.1 Learned Variance

**Definition 7 (Learned Variance Interpolation)**

*The network can predict variance via interpolation:*

$$
\sigma_t^2 = \exp(v \log\beta_t + (1-v)\log\tilde{\beta}_t) \tag{25}
$$

*where $v \in [0, 1]$ is predicted by the network.*

### 6.2 Bounds

**Lemma 6.1 (Variance Bounds)**

$$
\tilde{\beta}_t \leq \sigma_t^2 \leq \beta_t \tag{26}
$$

*since $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t \leq \beta_t$.*

## 7. Numerical Example

### 7.1 Forward Process Visualization

For $T = 10$ steps with linear schedule $\beta_t = 0.1$:

| $t$ | $\alpha_t$ | $\bar{\alpha}_t$ | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ |
|-----|------------|------------------|------------------------|---------------------------|
| 0 | - | 1.000 | 1.000 | 0.000 |
| 1 | 0.9 | 0.900 | 0.949 | 0.316 |
| 2 | 0.9 | 0.810 | 0.900 | 0.436 |
| 3 | 0.9 | 0.729 | 0.854 | 0.520 |
| 5 | 0.9 | 0.590 | 0.768 | 0.640 |
| 10 | 0.9 | 0.349 | 0.591 | 0.807 |

**Interpretation**: By $t=10$, signal is attenuated to 59% and noise accounts for 81% of variance.

### 7.2 Loss Computation Example

Given:
- $\mathbf{x}_0 = [0.5, -0.3]$
- $t = 5$, $\bar{\alpha}_5 = 0.59$
- $\boldsymbol{\epsilon} = [0.8, -1.2]$ (sampled)

Noisy sample:
$$
\mathbf{x}_5 = \sqrt{0.59} \cdot [0.5, -0.3] + \sqrt{0.41} \cdot [0.8, -1.2]
$$
$$
= [0.384, -0.230] + [0.512, -0.768]
$$
$$
= [0.896, -0.998]
$$

If network predicts $\hat{\boldsymbol{\epsilon}} = [0.75, -1.1]$:
$$
\mathcal{L} = \|[0.8, -1.2] - [0.75, -1.1]\|^2 = 0.05^2 + (-0.1)^2 = 0.0125
$$

## 8. Architecture Details

### 8.1 Time Embedding

**Definition 8 (Sinusoidal Embedding)**

$$
\text{PE}(t, 2i) = \sin(t / 10000^{2i/d}) \tag{27}
$$
$$
\text{PE}(t, 2i+1) = \cos(t / 10000^{2i/d}) \tag{28}
$$

### 8.2 U-Net Structure

For sequence data, we use a 1D U-Net:

```
Encoder: x → Conv1D → ResBlock → Downsample → ...
Bottleneck: Attention + ResBlock
Decoder: ... → Upsample → ResBlock → Conv1D → ε
```

### 8.3 Attention in Diffusion

**Definition 9 (Self-Attention Layer)**

$$
\text{Attn}(\mathbf{X}) = \text{Softmax}\left(\frac{\mathbf{X}\mathbf{W}_Q(\mathbf{X}\mathbf{W}_K)^\top}{\sqrt{d_k}}\right)\mathbf{X}\mathbf{W}_V \tag{29}
$$

## References

1. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*.
2. Song, Y., et al. (2021). Score-based generative modeling through stochastic differential equations. *ICLR*.
3. Nichol, A., & Dhariwal, P. (2021). Improved denoising diffusion probabilistic models. *ICML*.
4. Song, J., Meng, C., & Ermon, S. (2021). Denoising diffusion implicit models. *ICLR*.
5. Ho, J., & Salimans, T. (2022). Classifier-free diffusion guidance. *NeurIPS Workshop*.

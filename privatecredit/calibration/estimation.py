"""
Parameter Estimation - MLE and Bayesian inference for model parameters

Provides estimation tools for:
- Maximum Likelihood Estimation (MLE)
- Bayesian inference with MCMC
- Confidence intervals
- Model selection (AIC, BIC)
"""

import numpy as np
from scipy import optimize, stats
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EstimationConfig:
    """Configuration for parameter estimation"""
    # MLE settings
    max_iter: int = 1000
    tol: float = 1e-8
    method: str = 'L-BFGS-B'

    # MCMC settings
    n_samples: int = 10000
    n_burnin: int = 1000
    n_chains: int = 4
    proposal_scale: float = 0.1

    # Confidence level
    confidence_level: float = 0.95


# =============================================================================
# PARAMETER ESTIMATOR (MLE)
# =============================================================================

class ParameterEstimator:
    """
    Maximum Likelihood Estimation for credit risk parameters.

    Supports:
    - Default rate parameters
    - Transition matrix parameters
    - LGD distribution parameters
    - Correlation parameters
    """

    def __init__(self, config: EstimationConfig = None):
        self.config = config or EstimationConfig()
        self.fitted_params = {}
        self.fit_info = {}

    def fit_default_rate_mle(
        self,
        defaults: np.ndarray,
        exposures: np.ndarray,
        model: str = 'binomial'
    ) -> Dict[str, float]:
        """
        MLE for default rate parameters.

        Args:
            defaults: Number of defaults per period
            exposures: Number of exposures per period
            model: 'binomial', 'beta_binomial', or 'poisson'

        Returns:
            Estimated parameters
        """
        if model == 'binomial':
            # Simple binomial MLE
            total_defaults = defaults.sum()
            total_exposure = exposures.sum()
            p_hat = total_defaults / total_exposure

            # Standard error
            se = np.sqrt(p_hat * (1 - p_hat) / total_exposure)

            params = {
                'p': p_hat,
                'se': se,
                'ci_lower': p_hat - 1.96 * se,
                'ci_upper': p_hat + 1.96 * se
            }

        elif model == 'beta_binomial':
            # Beta-binomial MLE (accounts for overdispersion)
            observed_rates = defaults / (exposures + 1e-10)

            def neg_log_likelihood(params):
                alpha, beta = params
                if alpha <= 0 or beta <= 0:
                    return 1e10

                ll = 0
                for d, n in zip(defaults, exposures):
                    if n > 0:
                        # Beta-binomial log-likelihood
                        ll += stats.betabinom.logpmf(int(d), int(n), alpha, beta)
                return -ll

            # Initial guess from method of moments
            mean_rate = observed_rates.mean()
            var_rate = observed_rates.var()

            if var_rate > 0:
                # Method of moments
                common = mean_rate * (1 - mean_rate) / var_rate - 1
                alpha_init = mean_rate * common
                beta_init = (1 - mean_rate) * common
            else:
                alpha_init = 1
                beta_init = 1

            result = optimize.minimize(
                neg_log_likelihood,
                [max(0.1, alpha_init), max(0.1, beta_init)],
                method=self.config.method,
                bounds=[(1e-4, 1000), (1e-4, 1000)]
            )

            alpha, beta = result.x
            params = {
                'alpha': alpha,
                'beta': beta,
                'mean': alpha / (alpha + beta),
                'variance': alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)),
                'neg_log_likelihood': result.fun,
                'converged': result.success
            }

        elif model == 'poisson':
            # Poisson MLE for rare defaults
            def neg_log_likelihood(lambda_param):
                if lambda_param <= 0:
                    return 1e10
                return -stats.poisson.logpmf(defaults.astype(int), lambda_param * exposures).sum()

            result = optimize.minimize_scalar(
                neg_log_likelihood,
                bounds=(1e-6, 1),
                method='bounded'
            )

            lambda_hat = result.x
            params = {
                'lambda': lambda_hat,
                'mean_rate': lambda_hat,
                'neg_log_likelihood': result.fun
            }

        else:
            raise ValueError(f"Unknown model: {model}")

        self.fitted_params['default_rate'] = params
        return params

    def fit_lgd_mle(
        self,
        lgd_values: np.ndarray,
        model: str = 'beta'
    ) -> Dict[str, float]:
        """
        MLE for LGD distribution.

        Args:
            lgd_values: Observed LGD values (0-1)
            model: 'beta', 'truncated_normal', or 'mixture'

        Returns:
            Estimated parameters
        """
        # Filter valid values
        lgd = lgd_values[(lgd_values > 0) & (lgd_values < 1)]

        if model == 'beta':
            # Beta distribution MLE
            def neg_log_likelihood(params):
                alpha, beta = params
                if alpha <= 0 or beta <= 0:
                    return 1e10
                return -stats.beta.logpdf(lgd, alpha, beta).sum()

            # Method of moments initialization
            mean = lgd.mean()
            var = lgd.var()
            common = mean * (1 - mean) / var - 1
            alpha_init = mean * common
            beta_init = (1 - mean) * common

            result = optimize.minimize(
                neg_log_likelihood,
                [max(0.1, alpha_init), max(0.1, beta_init)],
                method=self.config.method,
                bounds=[(1e-4, 1000), (1e-4, 1000)]
            )

            alpha, beta = result.x
            params = {
                'alpha': alpha,
                'beta': beta,
                'mean': alpha / (alpha + beta),
                'mode': (alpha - 1) / (alpha + beta - 2) if alpha > 1 and beta > 1 else np.nan,
                'variance': alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)),
                'neg_log_likelihood': result.fun
            }

        elif model == 'truncated_normal':
            # Truncated normal on [0, 1]
            def neg_log_likelihood(params):
                mu, sigma = params
                if sigma <= 0:
                    return 1e10
                a, b = (0 - mu) / sigma, (1 - mu) / sigma
                return -stats.truncnorm.logpdf(lgd, a, b, loc=mu, scale=sigma).sum()

            result = optimize.minimize(
                neg_log_likelihood,
                [lgd.mean(), lgd.std()],
                method=self.config.method,
                bounds=[(-2, 2), (1e-4, 2)]
            )

            mu, sigma = result.x
            params = {
                'mu': mu,
                'sigma': sigma,
                'neg_log_likelihood': result.fun
            }

        elif model == 'mixture':
            # Two-component beta mixture
            def neg_log_likelihood(params):
                pi, a1, b1, a2, b2 = params
                if pi < 0 or pi > 1 or a1 <= 0 or b1 <= 0 or a2 <= 0 or b2 <= 0:
                    return 1e10

                pdf1 = stats.beta.pdf(lgd, a1, b1)
                pdf2 = stats.beta.pdf(lgd, a2, b2)
                mixture_pdf = pi * pdf1 + (1 - pi) * pdf2

                return -np.log(mixture_pdf + 1e-10).sum()

            # Initialize with k-means-like approach
            median = np.median(lgd)
            low_lgd = lgd[lgd < median]
            high_lgd = lgd[lgd >= median]

            result = optimize.minimize(
                neg_log_likelihood,
                [0.5, 2, 10, 2, 2],  # Initial guess
                method=self.config.method,
                bounds=[(0.01, 0.99), (0.1, 100), (0.1, 100), (0.1, 100), (0.1, 100)]
            )

            pi, a1, b1, a2, b2 = result.x
            params = {
                'mixing_weight': pi,
                'component1': {'alpha': a1, 'beta': b1},
                'component2': {'alpha': a2, 'beta': b2},
                'neg_log_likelihood': result.fun
            }

        else:
            raise ValueError(f"Unknown model: {model}")

        self.fitted_params['lgd'] = params
        return params

    def compute_standard_errors(
        self,
        neg_log_likelihood: Callable,
        params: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """
        Compute standard errors via numerical Hessian.

        Args:
            neg_log_likelihood: Negative log-likelihood function
            params: MLE parameter estimates
            eps: Step size for numerical differentiation

        Returns:
            Standard errors for each parameter
        """
        n_params = len(params)
        hessian = np.zeros((n_params, n_params))

        for i in range(n_params):
            for j in range(n_params):
                # Compute second derivative
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()

                params_pp[i] += eps
                params_pp[j] += eps
                params_pm[i] += eps
                params_pm[j] -= eps
                params_mp[i] -= eps
                params_mp[j] += eps
                params_mm[i] -= eps
                params_mm[j] -= eps

                hessian[i, j] = (
                    neg_log_likelihood(params_pp) -
                    neg_log_likelihood(params_pm) -
                    neg_log_likelihood(params_mp) +
                    neg_log_likelihood(params_mm)
                ) / (4 * eps ** 2)

        # Standard errors from inverse Hessian
        try:
            inv_hessian = np.linalg.inv(hessian)
            se = np.sqrt(np.diag(inv_hessian))
        except np.linalg.LinAlgError:
            se = np.full(n_params, np.nan)

        return se

    def model_selection(
        self,
        models: Dict[str, Dict],
        n_obs: int
    ) -> Dict[str, Dict]:
        """
        Compare models using AIC and BIC.

        Args:
            models: Dictionary of model name -> {'nll': neg_log_likelihood, 'k': n_params}
            n_obs: Number of observations

        Returns:
            Model comparison with AIC/BIC
        """
        results = {}

        for name, info in models.items():
            nll = info['nll']
            k = info['k']

            aic = 2 * k + 2 * nll
            bic = k * np.log(n_obs) + 2 * nll
            aicc = aic + 2 * k * (k + 1) / (n_obs - k - 1) if n_obs > k + 1 else np.inf

            results[name] = {
                'nll': nll,
                'k': k,
                'aic': aic,
                'bic': bic,
                'aicc': aicc
            }

        # Compute delta AIC and model weights
        min_aic = min(r['aic'] for r in results.values())
        for name in results:
            results[name]['delta_aic'] = results[name]['aic'] - min_aic
            results[name]['aic_weight'] = np.exp(-0.5 * results[name]['delta_aic'])

        # Normalize weights
        total_weight = sum(r['aic_weight'] for r in results.values())
        for name in results:
            results[name]['aic_weight'] /= total_weight

        return results


# =============================================================================
# BAYESIAN ESTIMATOR
# =============================================================================

class BayesianEstimator:
    """
    Bayesian inference for credit risk parameters using MCMC.

    Uses Metropolis-Hastings algorithm with adaptive proposal.
    """

    def __init__(self, config: EstimationConfig = None):
        self.config = config or EstimationConfig()
        self.samples = {}
        self.diagnostics = {}

    def metropolis_hastings(
        self,
        log_posterior: Callable,
        initial_params: np.ndarray,
        param_bounds: List[Tuple[float, float]] = None
    ) -> np.ndarray:
        """
        Metropolis-Hastings MCMC sampler.

        Args:
            log_posterior: Function computing log posterior
            initial_params: Starting parameter values
            param_bounds: Optional bounds for each parameter

        Returns:
            MCMC samples (n_samples, n_params)
        """
        n_params = len(initial_params)
        n_samples = self.config.n_samples
        n_burnin = self.config.n_burnin
        total_samples = n_samples + n_burnin

        # Initialize
        samples = np.zeros((total_samples, n_params))
        samples[0] = initial_params.copy()
        current_log_post = log_posterior(initial_params)

        # Adaptive proposal scale
        proposal_cov = np.eye(n_params) * self.config.proposal_scale ** 2
        accepted = 0

        for i in range(1, total_samples):
            # Propose new parameters
            proposal = np.random.multivariate_normal(samples[i - 1], proposal_cov)

            # Apply bounds
            if param_bounds is not None:
                for j, (lo, hi) in enumerate(param_bounds):
                    proposal[j] = np.clip(proposal[j], lo, hi)

            # Compute acceptance probability
            proposed_log_post = log_posterior(proposal)

            log_alpha = proposed_log_post - current_log_post
            alpha = min(1, np.exp(log_alpha))

            # Accept/reject
            if np.random.rand() < alpha:
                samples[i] = proposal
                current_log_post = proposed_log_post
                accepted += 1
            else:
                samples[i] = samples[i - 1]

            # Adapt proposal covariance during burn-in
            if i < n_burnin and i > 100 and i % 100 == 0:
                recent_samples = samples[max(0, i - 500):i]
                proposal_cov = np.cov(recent_samples.T) * 2.38 ** 2 / n_params + 1e-6 * np.eye(n_params)

        # Remove burn-in
        samples = samples[n_burnin:]

        self.diagnostics['acceptance_rate'] = accepted / total_samples

        return samples

    def fit_default_rate_bayesian(
        self,
        defaults: np.ndarray,
        exposures: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Bayesian estimation of default rate with beta prior.

        Args:
            defaults: Number of defaults per period
            exposures: Number of exposures per period
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter

        Returns:
            Posterior samples and summary statistics
        """
        total_defaults = int(defaults.sum())
        total_exposures = int(exposures.sum())

        # Conjugate update: Beta-Binomial
        posterior_alpha = prior_alpha + total_defaults
        posterior_beta = prior_beta + total_exposures - total_defaults

        # Sample from posterior
        samples = stats.beta.rvs(
            posterior_alpha,
            posterior_beta,
            size=self.config.n_samples
        )

        # Summary statistics
        results = {
            'samples': samples,
            'posterior_alpha': posterior_alpha,
            'posterior_beta': posterior_beta,
            'mean': posterior_alpha / (posterior_alpha + posterior_beta),
            'median': np.median(samples),
            'std': samples.std(),
            'ci_lower': np.percentile(samples, 2.5),
            'ci_upper': np.percentile(samples, 97.5),
            'hpd_lower': self._hpd_interval(samples)[0],
            'hpd_upper': self._hpd_interval(samples)[1]
        }

        self.samples['default_rate'] = results
        return results

    def fit_correlation_bayesian(
        self,
        defaults: np.ndarray,
        exposures: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Bayesian estimation of asset correlation using Vasicek model.

        Args:
            defaults: Number of defaults per period
            exposures: Number of exposures per period

        Returns:
            Posterior samples for PD and correlation
        """
        def log_posterior(params):
            pd, rho = params

            if pd <= 0 or pd >= 1 or rho <= 0 or rho >= 1:
                return -np.inf

            # Vasicek model log-likelihood
            ll = 0
            for d, n in zip(defaults, exposures):
                if n > 0:
                    # Approximate with normal approximation
                    k = d / n
                    # Vasicek model: transform
                    inv_norm_pd = stats.norm.ppf(pd)
                    factor = np.sqrt(1 - rho)

                    # Binomial approximation
                    expected = pd * n
                    variance = pd * (1 - pd) * n * (1 + (n - 1) * rho)
                    ll += stats.norm.logpdf(d, expected, np.sqrt(variance + 1e-10))

            # Priors: uniform on (0, 1)
            return ll

        # Run MCMC
        initial = np.array([defaults.sum() / exposures.sum(), 0.1])
        samples = self.metropolis_hastings(
            log_posterior,
            initial,
            param_bounds=[(0.001, 0.999), (0.001, 0.999)]
        )

        results = {
            'pd_samples': samples[:, 0],
            'rho_samples': samples[:, 1],
            'pd_mean': samples[:, 0].mean(),
            'rho_mean': samples[:, 1].mean(),
            'pd_ci': (np.percentile(samples[:, 0], 2.5), np.percentile(samples[:, 0], 97.5)),
            'rho_ci': (np.percentile(samples[:, 1], 2.5), np.percentile(samples[:, 1], 97.5))
        }

        self.samples['correlation'] = results
        return results

    def _hpd_interval(
        self,
        samples: np.ndarray,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """
        Compute Highest Posterior Density interval.

        Args:
            samples: MCMC samples
            alpha: Significance level (1 - credible level)

        Returns:
            HPD interval (lower, upper)
        """
        sorted_samples = np.sort(samples)
        n = len(samples)
        n_included = int(np.ceil((1 - alpha) * n))

        # Find narrowest interval
        min_width = np.inf
        hpd_min = sorted_samples[0]
        hpd_max = sorted_samples[n_included - 1]

        for i in range(n - n_included + 1):
            width = sorted_samples[i + n_included - 1] - sorted_samples[i]
            if width < min_width:
                min_width = width
                hpd_min = sorted_samples[i]
                hpd_max = sorted_samples[i + n_included - 1]

        return hpd_min, hpd_max

    def convergence_diagnostics(
        self,
        samples: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute MCMC convergence diagnostics.

        Args:
            samples: MCMC samples (n_samples, n_params)

        Returns:
            Convergence metrics
        """
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        n_samples, n_params = samples.shape

        diagnostics = {}

        for p in range(n_params):
            param_samples = samples[:, p]

            # Effective sample size
            acf = np.correlate(param_samples - param_samples.mean(), param_samples - param_samples.mean(), mode='full')
            acf = acf[n_samples - 1:] / acf[n_samples - 1]

            # Find first negative autocorrelation
            neg_idx = np.where(acf < 0)[0]
            if len(neg_idx) > 0:
                tau = 1 + 2 * acf[1:neg_idx[0]].sum()
            else:
                tau = 1 + 2 * acf[1:100].sum()

            ess = n_samples / tau

            # Geweke test (first 10% vs last 50%)
            n1 = int(0.1 * n_samples)
            n2 = int(0.5 * n_samples)
            mean1 = param_samples[:n1].mean()
            mean2 = param_samples[-n2:].mean()
            var1 = param_samples[:n1].var()
            var2 = param_samples[-n2:].var()
            geweke_z = (mean1 - mean2) / np.sqrt(var1 / n1 + var2 / n2)

            diagnostics[f'param_{p}'] = {
                'ess': ess,
                'geweke_z': geweke_z,
                'geweke_pvalue': 2 * (1 - stats.norm.cdf(abs(geweke_z)))
            }

        return diagnostics


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Demonstrate parameter estimation tools"""

    print("=" * 60)
    print("PARAMETER ESTIMATION TOOLS")
    print("=" * 60)

    np.random.seed(42)

    # Generate synthetic data
    true_pd = 0.025
    n_periods = 36
    avg_exposure = 1000

    exposures = np.random.poisson(avg_exposure, n_periods)
    defaults = np.random.binomial(exposures, true_pd)

    # MLE Estimation
    print("\n1. Maximum Likelihood Estimation")
    estimator = ParameterEstimator()

    # Binomial MLE
    params = estimator.fit_default_rate_mle(defaults, exposures, model='binomial')
    print(f"   Binomial MLE: PD = {params['p']:.4f} (true: {true_pd})")
    print(f"   95% CI: [{params['ci_lower']:.4f}, {params['ci_upper']:.4f}]")

    # Beta-binomial MLE
    params_bb = estimator.fit_default_rate_mle(defaults, exposures, model='beta_binomial')
    print(f"   Beta-Binomial MLE: PD = {params_bb['mean']:.4f}")

    # LGD Estimation
    print("\n2. LGD Distribution Estimation")
    lgd_data = np.random.beta(2, 5, size=500)
    lgd_params = estimator.fit_lgd_mle(lgd_data, model='beta')
    print(f"   Beta parameters: alpha={lgd_params['alpha']:.2f}, beta={lgd_params['beta']:.2f}")
    print(f"   Mean LGD: {lgd_params['mean']:.4f}")

    # Bayesian Estimation
    print("\n3. Bayesian Estimation")
    bayesian = BayesianEstimator()
    bayes_results = bayesian.fit_default_rate_bayesian(defaults, exposures)
    print(f"   Posterior mean: {bayes_results['mean']:.4f}")
    print(f"   95% HPD: [{bayes_results['hpd_lower']:.4f}, {bayes_results['hpd_upper']:.4f}]")

    # Model Selection
    print("\n4. Model Selection")
    n_obs = len(defaults)
    models = {
        'binomial': {'nll': -stats.binom.logpmf(defaults, exposures, params['p']).sum(), 'k': 1},
        'beta_binomial': {'nll': params_bb['neg_log_likelihood'], 'k': 2}
    }
    selection = estimator.model_selection(models, n_obs)
    for name, res in selection.items():
        print(f"   {name}: AIC={res['aic']:.2f}, Weight={res['aic_weight']:.4f}")

    print("\nEstimation complete!")


if __name__ == '__main__':
    main()

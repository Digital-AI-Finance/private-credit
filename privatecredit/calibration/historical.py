"""
Historical Calibration - Fit models to observed default rates and transitions

Provides calibration tools for:
- Default rate matching
- Transition matrix estimation
- Loss distribution calibration
- Moment matching
"""

import numpy as np
import pandas as pd
import torch
from scipy import optimize
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class CalibrationConfig:
    """Configuration for historical calibration"""
    # Credit states
    n_states: int = 7  # Performing, 30DPD, 60DPD, 90DPD, Default, Prepaid, Matured

    # Calibration targets
    target_default_rate: float = 0.02  # Annual default rate
    target_prepayment_rate: float = 0.15  # Annual prepayment rate

    # Optimization
    max_iter: int = 1000
    tol: float = 1e-6

    # Bootstrap
    n_bootstrap: int = 1000
    confidence_level: float = 0.95


# =============================================================================
# HISTORICAL CALIBRATOR
# =============================================================================

class HistoricalCalibrator:
    """
    Calibrate models to historical default and loss data.

    Methods:
    - Fit default rates by cohort/vintage
    - Match loss distribution moments
    - Calibrate LGD and recovery rates
    """

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.fitted_params = {}

    def fit_default_rates(
        self,
        observed_defaults: np.ndarray,
        exposure: np.ndarray,
        time_periods: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Fit default rate model to observed data.

        Args:
            observed_defaults: Number of defaults per period
            exposure: Total exposure per period
            time_periods: Optional time index

        Returns:
            Fitted parameters
        """
        # Observed default rates
        observed_rates = observed_defaults / (exposure + 1e-10)

        # Fit beta distribution to default rates
        alpha, beta, _, _ = stats.beta.fit(observed_rates, floc=0, fscale=1)

        # Fit time trend if available
        if time_periods is not None:
            slope, intercept, r_value, _, _ = stats.linregress(time_periods, observed_rates)
            trend_params = {'slope': slope, 'intercept': intercept, 'r_squared': r_value**2}
        else:
            trend_params = None

        params = {
            'mean_default_rate': observed_rates.mean(),
            'std_default_rate': observed_rates.std(),
            'beta_alpha': alpha,
            'beta_beta': beta,
            'trend': trend_params
        }

        self.fitted_params['default_rates'] = params
        return params

    def fit_loss_distribution(
        self,
        observed_losses: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Fit loss distribution to observed losses.

        Args:
            observed_losses: Array of observed losses
            weights: Optional sample weights

        Returns:
            Fitted parameters for various distributions
        """
        # Fit multiple distributions
        distributions = {}

        # Normal fit
        if weights is not None:
            mean = np.average(observed_losses, weights=weights)
            var = np.average((observed_losses - mean)**2, weights=weights)
            std = np.sqrt(var)
        else:
            mean = observed_losses.mean()
            std = observed_losses.std()

        distributions['normal'] = {'mean': mean, 'std': std}

        # Log-normal fit (for positive losses)
        positive_losses = observed_losses[observed_losses > 0]
        if len(positive_losses) > 10:
            log_losses = np.log(positive_losses)
            ln_mu = log_losses.mean()
            ln_sigma = log_losses.std()
            distributions['lognormal'] = {'mu': ln_mu, 'sigma': ln_sigma}

        # Beta fit (for losses as fraction of exposure)
        if observed_losses.max() <= 1:
            alpha, beta, _, _ = stats.beta.fit(observed_losses, floc=0, fscale=1)
            distributions['beta'] = {'alpha': alpha, 'beta': beta}

        # Fit GPD to tail
        threshold = np.percentile(observed_losses, 90)
        tail_losses = observed_losses[observed_losses > threshold] - threshold
        if len(tail_losses) > 20:
            shape, _, scale = stats.genpareto.fit(tail_losses)
            distributions['gpd_tail'] = {
                'shape': shape,
                'scale': scale,
                'threshold': threshold
            }

        # Goodness of fit tests
        distributions['ks_test_normal'] = stats.kstest(
            observed_losses, 'norm', args=(mean, std)
        ).pvalue

        self.fitted_params['loss_distribution'] = distributions
        return distributions

    def fit_lgd(
        self,
        recovery_amounts: np.ndarray,
        exposure_at_default: np.ndarray
    ) -> Dict[str, float]:
        """
        Fit Loss Given Default (LGD) distribution.

        Args:
            recovery_amounts: Amounts recovered
            exposure_at_default: Exposure at time of default

        Returns:
            Fitted LGD parameters
        """
        # Compute recovery rates and LGD
        recovery_rates = recovery_amounts / (exposure_at_default + 1e-10)
        recovery_rates = np.clip(recovery_rates, 0, 1)
        lgd = 1 - recovery_rates

        # Fit beta distribution to LGD
        alpha, beta, _, _ = stats.beta.fit(lgd[lgd > 0], floc=0, fscale=1)

        params = {
            'mean_lgd': lgd.mean(),
            'std_lgd': lgd.std(),
            'median_lgd': np.median(lgd),
            'beta_alpha': alpha,
            'beta_beta': beta,
            'q25': np.percentile(lgd, 25),
            'q75': np.percentile(lgd, 75)
        }

        self.fitted_params['lgd'] = params
        return params

    def bootstrap_confidence_intervals(
        self,
        data: np.ndarray,
        statistic: str = 'mean'
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals.

        Args:
            data: Input data array
            statistic: 'mean', 'std', 'median', or 'var'

        Returns:
            Confidence interval bounds
        """
        n = len(data)
        bootstrap_stats = []

        for _ in range(self.config.n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            if statistic == 'mean':
                bootstrap_stats.append(sample.mean())
            elif statistic == 'std':
                bootstrap_stats.append(sample.std())
            elif statistic == 'median':
                bootstrap_stats.append(np.median(sample))
            elif statistic == 'var':
                bootstrap_stats.append(sample.var())

        bootstrap_stats = np.array(bootstrap_stats)
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

        return {
            'point_estimate': np.mean(bootstrap_stats),
            'lower_bound': lower,
            'upper_bound': upper,
            'std_error': bootstrap_stats.std()
        }


# =============================================================================
# TRANSITION CALIBRATOR
# =============================================================================

class TransitionCalibrator:
    """
    Calibrate transition matrices from observed cohort data.

    Methods:
    - Count-based estimation
    - Duration-weighted estimation
    - Generator matrix estimation for continuous time
    """

    def __init__(self, config: CalibrationConfig = None):
        self.config = config or CalibrationConfig()
        self.fitted_matrices = {}

    def estimate_transition_matrix(
        self,
        state_history: np.ndarray,
        time_step: float = 1.0
    ) -> np.ndarray:
        """
        Estimate transition matrix from state history.

        Args:
            state_history: Array of shape (n_loans, n_periods) with state indices
            time_step: Time period length (e.g., 1 for monthly, 12 for annual)

        Returns:
            Estimated transition matrix (n_states, n_states)
        """
        n_states = self.config.n_states
        transition_counts = np.zeros((n_states, n_states))

        n_loans, n_periods = state_history.shape

        # Count transitions
        for t in range(n_periods - 1):
            for i in range(n_loans):
                from_state = state_history[i, t]
                to_state = state_history[i, t + 1]
                if from_state >= 0 and to_state >= 0:  # Valid states
                    transition_counts[int(from_state), int(to_state)] += 1

        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_counts,
            row_sums,
            where=row_sums > 0
        )

        # Handle rows with no transitions (absorbing states)
        for i in range(n_states):
            if row_sums[i, 0] == 0:
                transition_matrix[i, i] = 1.0

        self.fitted_matrices['discrete'] = transition_matrix
        return transition_matrix

    def estimate_generator_matrix(
        self,
        transition_matrix: np.ndarray,
        delta_t: float = 1.0
    ) -> np.ndarray:
        """
        Estimate continuous-time generator matrix from discrete transition matrix.

        Uses eigenvalue decomposition: Q = (1/dt) * log(P)

        Args:
            transition_matrix: Discrete transition matrix
            delta_t: Time step

        Returns:
            Generator matrix Q where P = exp(Q * delta_t)
        """
        n_states = transition_matrix.shape[0]

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)

        # Take log of eigenvalues (complex for negative eigenvalues)
        log_eigenvalues = np.log(eigenvalues.astype(complex))

        # Reconstruct generator
        Q = eigenvectors @ np.diag(log_eigenvalues) @ np.linalg.inv(eigenvectors)
        Q = Q.real / delta_t

        # Ensure valid generator: off-diagonal >= 0, rows sum to 0
        Q = self._regularize_generator(Q)

        self.fitted_matrices['generator'] = Q
        return Q

    def _regularize_generator(self, Q: np.ndarray) -> np.ndarray:
        """Ensure generator matrix is valid"""
        n_states = Q.shape[0]

        # Make off-diagonal non-negative
        Q_reg = Q.copy()
        for i in range(n_states):
            for j in range(n_states):
                if i != j:
                    Q_reg[i, j] = max(0, Q_reg[i, j])

        # Make rows sum to zero
        for i in range(n_states):
            Q_reg[i, i] = -Q_reg[i, :].sum() + Q_reg[i, i]

        return Q_reg

    def estimate_cohort_matrices(
        self,
        state_history: np.ndarray,
        cohort_labels: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Estimate separate transition matrices per cohort.

        Args:
            state_history: Array (n_loans, n_periods) of state indices
            cohort_labels: Array (n_loans,) of cohort identifiers

        Returns:
            Dictionary mapping cohort to transition matrix
        """
        unique_cohorts = np.unique(cohort_labels)
        matrices = {}

        for cohort in unique_cohorts:
            mask = cohort_labels == cohort
            cohort_history = state_history[mask]
            matrices[cohort] = self.estimate_transition_matrix(cohort_history)

        self.fitted_matrices['cohort'] = matrices
        return matrices

    def fit_to_target_rates(
        self,
        target_default_rate: float,
        target_prepayment_rate: float,
        maturity: int = 60
    ) -> np.ndarray:
        """
        Calibrate transition matrix to match target default and prepayment rates.

        Args:
            target_default_rate: Annual default rate
            target_prepayment_rate: Annual prepayment rate
            maturity: Loan maturity in periods

        Returns:
            Calibrated transition matrix
        """
        n_states = self.config.n_states

        # Initialize with identity-like matrix
        P = np.eye(n_states) * 0.9
        for i in range(n_states - 1):
            P[i, i + 1] = 0.1

        # Absorbing states
        P[4, :] = 0  # Default
        P[4, 4] = 1
        P[5, :] = 0  # Prepaid
        P[5, 5] = 1
        P[6, :] = 0  # Matured
        P[6, 6] = 1

        def objective(params):
            # Unpack parameters
            p_stay = params[:4]
            p_worsen = params[4:8]
            p_improve = params[8:12]
            p_default = params[12:16]
            p_prepay = params[16:20]

            # Build matrix
            P_test = np.zeros((n_states, n_states))
            for i in range(4):  # Non-absorbing states
                P_test[i, i] = p_stay[i]
                if i < 3:
                    P_test[i, i + 1] = p_worsen[i]
                if i > 0:
                    P_test[i, i - 1] = p_improve[i]
                P_test[i, 4] = p_default[i]  # To default
                P_test[i, 5] = p_prepay[i]  # To prepaid

            # Normalize rows
            for i in range(4):
                row_sum = P_test[i, :].sum()
                if row_sum > 0:
                    P_test[i, :] /= row_sum

            # Absorbing states
            P_test[4, 4] = 1
            P_test[5, 5] = 1
            P_test[6, 6] = 1

            # Compute cumulative default/prepayment
            state_probs = np.zeros(n_states)
            state_probs[0] = 1  # Start in performing

            cumulative_default = 0
            cumulative_prepay = 0

            for t in range(maturity):
                state_probs = state_probs @ P_test
                cumulative_default = state_probs[4]
                cumulative_prepay = state_probs[5]

            # Annual rates
            annual_default = 1 - (1 - cumulative_default) ** (12 / maturity)
            annual_prepay = 1 - (1 - cumulative_prepay) ** (12 / maturity)

            # Objective: minimize squared error
            error = (annual_default - target_default_rate) ** 2
            error += (annual_prepay - target_prepayment_rate) ** 2

            return error

        # Initial guess
        x0 = np.concatenate([
            np.ones(4) * 0.9,   # p_stay
            np.ones(4) * 0.05,  # p_worsen
            np.ones(4) * 0.02,  # p_improve
            np.ones(4) * 0.01,  # p_default
            np.ones(4) * 0.02,  # p_prepay
        ])

        # Bounds
        bounds = [(0.01, 0.99)] * 20

        # Optimize
        result = optimize.minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': self.config.max_iter}
        )

        # Build final matrix from optimized params
        params = result.x
        P_calibrated = np.zeros((n_states, n_states))

        for i in range(4):
            P_calibrated[i, i] = params[i]
            if i < 3:
                P_calibrated[i, i + 1] = params[4 + i]
            if i > 0:
                P_calibrated[i, i - 1] = params[8 + i]
            P_calibrated[i, 4] = params[12 + i]
            P_calibrated[i, 5] = params[16 + i]

        # Normalize and set absorbing
        for i in range(4):
            P_calibrated[i, :] /= P_calibrated[i, :].sum()
        P_calibrated[4, 4] = 1
        P_calibrated[5, 5] = 1
        P_calibrated[6, 6] = 1

        self.fitted_matrices['calibrated'] = P_calibrated
        return P_calibrated


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_distributions(
    observed: np.ndarray,
    simulated: np.ndarray,
    metrics: List[str] = ['mean', 'std', 'median', 'q05', 'q95']
) -> Dict[str, Dict[str, float]]:
    """
    Compare observed and simulated distributions.

    Args:
        observed: Observed data
        simulated: Simulated data
        metrics: List of metrics to compare

    Returns:
        Comparison results
    """
    results = {}

    for metric in metrics:
        if metric == 'mean':
            obs_val = observed.mean()
            sim_val = simulated.mean()
        elif metric == 'std':
            obs_val = observed.std()
            sim_val = simulated.std()
        elif metric == 'median':
            obs_val = np.median(observed)
            sim_val = np.median(simulated)
        elif metric.startswith('q'):
            q = int(metric[1:])
            obs_val = np.percentile(observed, q)
            sim_val = np.percentile(simulated, q)
        else:
            continue

        results[metric] = {
            'observed': obs_val,
            'simulated': sim_val,
            'error': sim_val - obs_val,
            'relative_error': (sim_val - obs_val) / (abs(obs_val) + 1e-10)
        }

    # Statistical tests
    ks_stat, ks_pval = stats.ks_2samp(observed, simulated)
    results['ks_test'] = {'statistic': ks_stat, 'p_value': ks_pval}

    return results


def main():
    """Demonstrate calibration tools"""

    print("=" * 60)
    print("HISTORICAL CALIBRATION TOOLS")
    print("=" * 60)

    # Create calibrator
    calibrator = HistoricalCalibrator()

    # Generate synthetic historical data
    np.random.seed(42)
    n_periods = 60

    # Simulated default data
    exposure = np.ones(n_periods) * 1000000
    default_rate = 0.02 + 0.005 * np.random.randn(n_periods)
    default_rate = np.clip(default_rate, 0.005, 0.05)
    defaults = np.round(exposure * default_rate)

    # Fit default rates
    print("\n1. Fitting Default Rates")
    params = calibrator.fit_default_rates(defaults, exposure, np.arange(n_periods))
    print(f"   Mean default rate: {params['mean_default_rate']:.4f}")
    print(f"   Std default rate: {params['std_default_rate']:.4f}")

    # Simulated loss data
    losses = np.random.beta(2, 10, size=1000) * 100000

    print("\n2. Fitting Loss Distribution")
    dist_params = calibrator.fit_loss_distribution(losses)
    print(f"   Normal mean: {dist_params['normal']['mean']:.2f}")
    print(f"   Normal std: {dist_params['normal']['std']:.2f}")

    # Bootstrap CI
    print("\n3. Bootstrap Confidence Intervals")
    ci = calibrator.bootstrap_confidence_intervals(losses, 'mean')
    print(f"   Mean: {ci['point_estimate']:.2f}")
    print(f"   95% CI: [{ci['lower_bound']:.2f}, {ci['upper_bound']:.2f}]")

    # Transition calibration
    print("\n4. Transition Matrix Calibration")
    trans_cal = TransitionCalibrator()

    P = trans_cal.fit_to_target_rates(
        target_default_rate=0.02,
        target_prepayment_rate=0.15,
        maturity=60
    )
    print(f"   Calibrated matrix shape: {P.shape}")
    print(f"   Default probability from Performing: {P[0, 4]:.4f}")
    print(f"   Prepayment probability from Performing: {P[0, 5]:.4f}")

    print("\nCalibration complete!")


if __name__ == '__main__':
    main()

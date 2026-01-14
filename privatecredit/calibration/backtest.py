"""
Backtesting Framework - Walk-forward validation and coverage tests

Provides backtesting tools for:
- Walk-forward validation
- VaR and CVaR coverage tests
- Brier scores for PD predictions
- Model comparison metrics
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class CoverageTestType(Enum):
    """Types of coverage tests"""
    KUPIEC = "kupiec"  # Unconditional coverage
    CHRISTOFFERSEN = "christoffersen"  # Conditional coverage
    JOINT = "joint"  # Christoffersen joint test


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    # Walk-forward settings
    train_window: int = 36  # Training window size (months)
    test_window: int = 12  # Test window size (months)
    step_size: int = 1  # Step size for rolling window

    # VaR settings
    confidence_levels: List[float] = None  # Default: [0.95, 0.99]

    # Significance
    alpha: float = 0.05

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.95, 0.99]


# =============================================================================
# BACKTEST FRAMEWORK
# =============================================================================

class BacktestFramework:
    """
    Walk-forward backtesting for credit risk models.

    Methods:
    - Rolling window validation
    - Out-of-sample metrics
    - Model comparison
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results = {}

    def walk_forward_backtest(
        self,
        data: np.ndarray,
        model_train_fn: Callable,
        model_predict_fn: Callable,
        target_col: int = -1
    ) -> Dict[str, np.ndarray]:
        """
        Perform walk-forward backtest.

        Args:
            data: Time series data (n_periods, n_features)
            model_train_fn: Function to train model, takes train_data
            model_predict_fn: Function to generate predictions
            target_col: Index of target column

        Returns:
            Dictionary with predictions and actuals
        """
        n_periods = len(data)
        train_window = self.config.train_window
        test_window = self.config.test_window
        step = self.config.step_size

        predictions = []
        actuals = []
        train_ends = []

        # Walk forward
        t = train_window
        while t + test_window <= n_periods:
            # Training data
            train_data = data[t - train_window:t]

            # Fit model
            model = model_train_fn(train_data)

            # Test data
            test_data = data[t:t + test_window]

            # Generate predictions
            preds = model_predict_fn(model, test_data)

            # Store results
            predictions.append(preds)
            actuals.append(test_data[:, target_col] if target_col >= 0 else test_data)
            train_ends.append(t)

            t += step

        results = {
            'predictions': np.concatenate(predictions),
            'actuals': np.concatenate(actuals),
            'train_ends': np.array(train_ends),
            'n_windows': len(train_ends)
        }

        self.results['walk_forward'] = results
        return results

    def compute_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute backtest metrics.

        Args:
            predictions: Model predictions
            actuals: Actual values

        Returns:
            Dictionary of metrics
        """
        # Basic errors
        errors = actuals - predictions
        abs_errors = np.abs(errors)

        metrics = {
            # Point forecast metrics
            'mae': abs_errors.mean(),
            'rmse': np.sqrt((errors ** 2).mean()),
            'mape': (abs_errors / (np.abs(actuals) + 1e-10)).mean(),
            'bias': errors.mean(),

            # Correlation
            'correlation': np.corrcoef(predictions.flatten(), actuals.flatten())[0, 1],

            # R-squared
            'r_squared': 1 - (errors ** 2).sum() / ((actuals - actuals.mean()) ** 2).sum(),
        }

        return metrics

    def compute_brier_score(
        self,
        predicted_probs: np.ndarray,
        actual_events: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute Brier score for probability predictions.

        Args:
            predicted_probs: Predicted probabilities (0-1)
            actual_events: Binary outcomes (0 or 1)

        Returns:
            Brier score and decomposition
        """
        # Basic Brier score
        brier = ((predicted_probs - actual_events) ** 2).mean()

        # Decomposition (Murphy)
        n = len(actual_events)
        base_rate = actual_events.mean()

        # Reliability (calibration)
        # Group by predicted probability bins
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        reliability = 0
        resolution = 0

        for i in range(n_bins):
            mask = (predicted_probs >= bins[i]) & (predicted_probs < bins[i + 1])
            if mask.sum() > 0:
                bin_prob = predicted_probs[mask].mean()
                bin_actual = actual_events[mask].mean()
                bin_count = mask.sum()

                reliability += bin_count * (bin_actual - bin_prob) ** 2
                resolution += bin_count * (bin_actual - base_rate) ** 2

        reliability /= n
        resolution /= n
        uncertainty = base_rate * (1 - base_rate)

        return {
            'brier_score': brier,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'skill_score': 1 - brier / uncertainty if uncertainty > 0 else 0
        }

    def discrimination_metrics(
        self,
        predicted_probs: np.ndarray,
        actual_events: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute discrimination metrics (AUC, Gini, etc.)

        Args:
            predicted_probs: Predicted default probabilities
            actual_events: Actual default indicators

        Returns:
            Discrimination metrics
        """
        # Sort by predicted probability
        sorted_idx = np.argsort(predicted_probs)
        sorted_actual = actual_events[sorted_idx]
        sorted_pred = predicted_probs[sorted_idx]

        # ROC curve
        n = len(actual_events)
        n_pos = actual_events.sum()
        n_neg = n - n_pos

        if n_pos == 0 or n_neg == 0:
            return {'auc': 0.5, 'gini': 0, 'ks_statistic': 0}

        # Compute TPR and FPR at each threshold
        tpr = []
        fpr = []
        thresholds = np.unique(sorted_pred)

        for thresh in thresholds:
            pred_pos = predicted_probs >= thresh
            tp = (pred_pos & (actual_events == 1)).sum()
            fp = (pred_pos & (actual_events == 0)).sum()
            tpr.append(tp / n_pos)
            fpr.append(fp / n_neg)

        tpr = np.array(tpr)
        fpr = np.array(fpr)

        # AUC (trapezoidal rule)
        auc = np.trapz(tpr, fpr)

        # Gini coefficient
        gini = 2 * auc - 1

        # KS statistic
        ks_stat = np.max(np.abs(tpr - fpr))

        return {
            'auc': abs(auc),
            'gini': abs(gini),
            'ks_statistic': ks_stat
        }


# =============================================================================
# VAR BACKTEST
# =============================================================================

class VaRBacktest:
    """
    VaR and CVaR backtesting.

    Tests:
    - Kupiec unconditional coverage test
    - Christoffersen conditional coverage test
    - Traffic light approach
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()
        self.results = {}

    def compute_var_breaches(
        self,
        returns: np.ndarray,
        var_forecasts: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute VaR breaches (exceptions).

        Args:
            returns: Actual returns/losses
            var_forecasts: Forecasted VaR (negative for losses)

        Returns:
            Breach indicators and counts
        """
        # Breach occurs when return is worse than VaR
        breaches = returns < var_forecasts
        n_breaches = breaches.sum()
        n_obs = len(returns)

        return {
            'breaches': breaches.astype(int),
            'n_breaches': n_breaches,
            'n_obs': n_obs,
            'breach_rate': n_breaches / n_obs
        }

    def kupiec_test(
        self,
        breaches: np.ndarray,
        confidence_level: float
    ) -> Dict[str, float]:
        """
        Kupiec unconditional coverage test.

        Tests if the observed breach rate equals the expected rate.

        Args:
            breaches: Binary array of VaR breaches
            confidence_level: VaR confidence level (e.g., 0.99)

        Returns:
            Test results
        """
        n = len(breaches)
        x = breaches.sum()  # Number of breaches
        p = 1 - confidence_level  # Expected breach probability

        # Avoid edge cases
        if x == 0:
            x = 0.5
        if x == n:
            x = n - 0.5

        # Likelihood ratio statistic
        p_hat = x / n  # Observed breach rate

        if p_hat == 0 or p_hat == 1:
            lr_stat = 0
        else:
            lr_stat = -2 * (
                np.log(p ** x * (1 - p) ** (n - x)) -
                np.log(p_hat ** x * (1 - p_hat) ** (n - x))
            )

        # p-value from chi-squared distribution with 1 df
        p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

        return {
            'n_obs': n,
            'n_breaches': int(x),
            'expected_breaches': n * p,
            'observed_rate': p_hat,
            'expected_rate': p,
            'lr_statistic': lr_stat,
            'p_value': p_value,
            'reject_h0': p_value < self.config.alpha
        }

    def christoffersen_test(
        self,
        breaches: np.ndarray,
        confidence_level: float
    ) -> Dict[str, float]:
        """
        Christoffersen independence and conditional coverage test.

        Tests for serial independence of VaR breaches.

        Args:
            breaches: Binary array of VaR breaches
            confidence_level: VaR confidence level

        Returns:
            Test results
        """
        n = len(breaches) - 1

        # Transition counts
        n00 = 0  # No breach -> No breach
        n01 = 0  # No breach -> Breach
        n10 = 0  # Breach -> No breach
        n11 = 0  # Breach -> Breach

        for i in range(n):
            if breaches[i] == 0 and breaches[i + 1] == 0:
                n00 += 1
            elif breaches[i] == 0 and breaches[i + 1] == 1:
                n01 += 1
            elif breaches[i] == 1 and breaches[i + 1] == 0:
                n10 += 1
            else:
                n11 += 1

        # Probabilities
        p01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
        p11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
        p = (n01 + n11) / n if n > 0 else 0

        # Independence test statistic
        if p01 > 0 and p11 > 0 and p01 < 1 and p11 < 1 and p > 0 and p < 1:
            lr_ind = -2 * (
                np.log((1 - p) ** (n00 + n10) * p ** (n01 + n11)) -
                np.log((1 - p01) ** n00 * p01 ** n01 * (1 - p11) ** n10 * p11 ** n11)
            )
        else:
            lr_ind = 0

        # Get Kupiec test result
        kupiec = self.kupiec_test(breaches, confidence_level)

        # Joint test (conditional coverage)
        lr_joint = kupiec['lr_statistic'] + lr_ind

        return {
            'transition_counts': {'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11},
            'p01': p01,
            'p11': p11,
            'lr_independence': lr_ind,
            'p_value_independence': 1 - stats.chi2.cdf(lr_ind, df=1),
            'lr_joint': lr_joint,
            'p_value_joint': 1 - stats.chi2.cdf(lr_joint, df=2),
            'reject_independence': (1 - stats.chi2.cdf(lr_ind, df=1)) < self.config.alpha,
            'reject_joint': (1 - stats.chi2.cdf(lr_joint, df=2)) < self.config.alpha
        }

    def traffic_light_test(
        self,
        breaches: np.ndarray,
        confidence_level: float = 0.99,
        n_days: int = 250
    ) -> Dict[str, str]:
        """
        Basel traffic light approach for VaR backtesting.

        Args:
            breaches: Binary array of VaR breaches
            confidence_level: VaR confidence level
            n_days: Number of days in the test period

        Returns:
            Traffic light zone and add-on
        """
        n_breaches = breaches.sum()

        # Basel III zones for 99% VaR over 250 days
        if confidence_level == 0.99 and n_days == 250:
            if n_breaches <= 4:
                zone = 'green'
                add_on = 0.0
            elif n_breaches <= 9:
                zone = 'yellow'
                # Increasing add-ons: 5->0.4, 6->0.5, 7->0.65, 8->0.75, 9->0.85
                add_ons = {5: 0.4, 6: 0.5, 7: 0.65, 8: 0.75, 9: 0.85}
                add_on = add_ons.get(n_breaches, 0.5)
            else:
                zone = 'red'
                add_on = 1.0
        else:
            # Scale thresholds for different parameters
            expected = n_days * (1 - confidence_level)
            if n_breaches <= expected * 1.5:
                zone = 'green'
                add_on = 0.0
            elif n_breaches <= expected * 3:
                zone = 'yellow'
                add_on = 0.5
            else:
                zone = 'red'
                add_on = 1.0

        return {
            'zone': zone,
            'capital_add_on': add_on,
            'n_breaches': int(n_breaches),
            'expected_breaches': n_days * (1 - confidence_level)
        }

    def backtest_var(
        self,
        returns: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.99
    ) -> Dict[str, Dict]:
        """
        Complete VaR backtest suite.

        Args:
            returns: Actual returns/losses
            var_forecasts: VaR forecasts
            confidence_level: VaR confidence level

        Returns:
            Combined test results
        """
        # Compute breaches
        breach_info = self.compute_var_breaches(returns, var_forecasts)
        breaches = breach_info['breaches']

        results = {
            'summary': breach_info,
            'kupiec': self.kupiec_test(breaches, confidence_level),
            'christoffersen': self.christoffersen_test(breaches, confidence_level),
            'traffic_light': self.traffic_light_test(breaches, confidence_level)
        }

        self.results[f'var_{int(confidence_level*100)}'] = results
        return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_models(
    model_results: Dict[str, Dict],
    metric: str = 'rmse'
) -> pd.DataFrame:
    """
    Compare multiple models on backtest metrics.

    Args:
        model_results: Dictionary of model name -> backtest results
        metric: Metric to compare

    Returns:
        Comparison DataFrame
    """
    comparison = []
    for model_name, results in model_results.items():
        if 'metrics' in results:
            comparison.append({
                'model': model_name,
                metric: results['metrics'].get(metric, np.nan)
            })

    return pd.DataFrame(comparison).sort_values(metric)


def main():
    """Demonstrate backtesting tools"""

    print("=" * 60)
    print("BACKTESTING FRAMEWORK")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n_periods = 500

    # Simulated returns
    returns = np.random.normal(-0.001, 0.02, n_periods)

    # Simulated VaR forecasts (99% VaR)
    var_99 = np.percentile(returns[:100], 1) * np.ones(n_periods)  # Static VaR
    var_99 += np.random.normal(0, 0.001, n_periods)  # Add noise

    # VaR Backtest
    print("\n1. VaR Backtesting")
    var_backtest = VaRBacktest()
    results = var_backtest.backtest_var(returns, var_99, confidence_level=0.99)

    print(f"   Breaches: {results['summary']['n_breaches']} / {results['summary']['n_obs']}")
    print(f"   Breach rate: {results['summary']['breach_rate']:.4f}")
    print(f"   Kupiec p-value: {results['kupiec']['p_value']:.4f}")
    print(f"   Traffic light: {results['traffic_light']['zone'].upper()}")

    # Brier Score
    print("\n2. Probability Calibration")
    bt_framework = BacktestFramework()

    # Simulated PD predictions
    predicted_pds = np.random.beta(2, 50, n_periods)
    actual_defaults = (np.random.rand(n_periods) < predicted_pds).astype(int)

    brier = bt_framework.compute_brier_score(predicted_pds, actual_defaults)
    print(f"   Brier score: {brier['brier_score']:.4f}")
    print(f"   Skill score: {brier['skill_score']:.4f}")

    # Discrimination
    print("\n3. Discrimination Metrics")
    disc = bt_framework.discrimination_metrics(predicted_pds, actual_defaults)
    print(f"   AUC: {disc['auc']:.4f}")
    print(f"   Gini: {disc['gini']:.4f}")
    print(f"   KS Statistic: {disc['ks_statistic']:.4f}")

    print("\nBacktesting complete!")


if __name__ == '__main__':
    main()

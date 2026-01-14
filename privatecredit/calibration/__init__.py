"""
Calibration tools for credit risk models.

This module provides:
- HistoricalCalibrator: Fit models to observed default rates and transitions
- BacktestFramework: Walk-forward validation and VaR coverage tests
- ParameterEstimator: MLE and Bayesian inference for model parameters
"""

from privatecredit.calibration.historical import HistoricalCalibrator, TransitionCalibrator
from privatecredit.calibration.backtest import BacktestFramework, VaRBacktest
from privatecredit.calibration.estimation import ParameterEstimator, BayesianEstimator

__all__ = [
    # Historical calibration
    "HistoricalCalibrator",
    "TransitionCalibrator",
    # Backtesting
    "BacktestFramework",
    "VaRBacktest",
    # Parameter estimation
    "ParameterEstimator",
    "BayesianEstimator",
]

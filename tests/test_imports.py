"""Test that all package imports work correctly."""

import pytest


def test_version():
    """Test that version is defined."""
    import privatecredit
    assert hasattr(privatecredit, '__version__')
    assert privatecredit.__version__ == '0.1.0'


def test_data_imports():
    """Test data module imports."""
    from privatecredit.data import LoanTapeGenerator, MacroScenarioGenerator
    assert LoanTapeGenerator is not None
    assert MacroScenarioGenerator is not None


def test_models_imports():
    """Test models module imports."""
    from privatecredit.models import (
        MacroVAE,
        MacroGAN,
        MacroFlow,
        MacroEnsemble,
        TransitionTransformer,
        LoanTrajectoryModel,
        PortfolioAggregator,
        MarkovTransitionModel
    )
    assert MacroVAE is not None
    assert MacroGAN is not None
    assert MacroFlow is not None
    assert MacroEnsemble is not None
    assert TransitionTransformer is not None
    assert LoanTrajectoryModel is not None
    assert PortfolioAggregator is not None
    assert MarkovTransitionModel is not None


def test_calibration_imports():
    """Test calibration module imports."""
    from privatecredit.calibration import (
        HistoricalCalibrator,
        TransitionCalibrator,
        BacktestFramework,
        VaRBacktest,
        ParameterEstimator,
        BayesianEstimator
    )
    assert HistoricalCalibrator is not None
    assert TransitionCalibrator is not None
    assert BacktestFramework is not None
    assert VaRBacktest is not None
    assert ParameterEstimator is not None
    assert BayesianEstimator is not None

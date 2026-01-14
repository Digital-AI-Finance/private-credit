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
        TransitionTransformer,
        LoanTrajectoryModel,
        PortfolioAggregator,
        MarkovTransitionModel
    )
    assert MacroVAE is not None
    assert TransitionTransformer is not None
    assert LoanTrajectoryModel is not None
    assert PortfolioAggregator is not None
    assert MarkovTransitionModel is not None

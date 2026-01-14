"""
Private Credit: Deep Generative Models for SPV Analytics

A hierarchical framework for private credit portfolio modeling:
- Macro VAE for scenario generation
- Transition Transformer for cohort dynamics
- Loan Trajectory Model for individual paths
- Portfolio Aggregator for waterfall and loss distribution
"""

__version__ = "0.1.0"
__author__ = "Digital Finance Research"
__email__ = "research@digital-finance.org"
__license__ = "MIT"

from privatecredit.data import LoanTapeGenerator, MacroScenarioGenerator
from privatecredit.models import (
    MacroVAE,
    TransitionTransformer,
    LoanTrajectoryModel,
    PortfolioAggregator,
    MarkovTransitionModel,
)

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Data generators
    "LoanTapeGenerator",
    "MacroScenarioGenerator",
    # Models
    "MacroVAE",
    "TransitionTransformer",
    "LoanTrajectoryModel",
    "PortfolioAggregator",
    "MarkovTransitionModel",
]

"""
Deep generative models for credit risk.

This module provides:
- MacroVAE: Conditional VAE for macro scenario generation
- MacroGAN: Wasserstein GAN with gradient penalty for macro generation
- MacroFlow: Normalizing flows for exact likelihood macro generation
- MacroEnsemble: Ensemble combining VAE, GAN, and Flow models
- TransitionTransformer: Cohort-level transition dynamics
- LoanTrajectoryModel: Individual loan path generation
- PortfolioAggregator: Waterfall and loss aggregation
- MarkovTransitionModel: Baseline benchmark model
"""

from privatecredit.models.macro_vae import MacroVAE, MacroVAEConfig
from privatecredit.models.macro_gan import MacroGAN, MacroGANConfig
from privatecredit.models.macro_flow import MacroFlow, MacroFlowConfig
from privatecredit.models.ensemble import MacroEnsemble, EnsembleConfig, EnsembleMethod
from privatecredit.models.transition_transformer import TransitionTransformer, TransitionTransformerConfig
from privatecredit.models.loan_trajectory import LoanTrajectoryModel, LoanTrajectoryConfig
from privatecredit.models.portfolio_aggregator import PortfolioAggregator, DifferentiableAggregator
from privatecredit.models.baseline_markov import MarkovTransitionModel, PortfolioLossModel

__all__ = [
    # Macro VAE
    "MacroVAE",
    "MacroVAEConfig",
    # Macro GAN
    "MacroGAN",
    "MacroGANConfig",
    # Macro Flow
    "MacroFlow",
    "MacroFlowConfig",
    # Ensemble
    "MacroEnsemble",
    "EnsembleConfig",
    "EnsembleMethod",
    # Transition Transformer
    "TransitionTransformer",
    "TransitionTransformerConfig",
    # Loan Trajectory
    "LoanTrajectoryModel",
    "LoanTrajectoryConfig",
    # Portfolio Aggregator
    "PortfolioAggregator",
    "DifferentiableAggregator",
    # Baseline
    "MarkovTransitionModel",
    "PortfolioLossModel",
]

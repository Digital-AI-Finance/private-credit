"""
Portfolio Aggregator - Waterfall and Loss Distribution

Aggregates loan-level trajectories to compute:
- Portfolio cashflows
- Loss distribution (VaR, CVaR)
- Tranche-level returns
- Coverage ratio tests (OC, IC)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# =============================================================================
# SPV STRUCTURE
# =============================================================================

@dataclass
class Tranche:
    """SPV tranche definition"""
    name: str
    size: float  # Notional
    attachment: float  # Loss attachment point
    detachment: float  # Loss detachment point
    coupon: float  # Annual coupon rate
    priority: int  # Waterfall priority (1 = senior)


@dataclass
class SPVStructure:
    """Complete SPV structure"""
    tranches: List[Tranche]
    senior_fees_pct: float = 0.01  # Annual
    junior_fees_pct: float = 0.005
    reserve_target: float = 0.02  # As % of pool


# Example CLO structure
DEFAULT_CLO = SPVStructure(
    tranches=[
        Tranche('Senior A', 700_000_000, 0.0, 0.70, 0.055, 1),
        Tranche('Mezzanine B', 150_000_000, 0.70, 0.85, 0.075, 2),
        Tranche('Junior C', 100_000_000, 0.85, 0.95, 0.10, 3),
        Tranche('Equity', 50_000_000, 0.95, 1.0, 0.0, 4),
    ]
)


# =============================================================================
# PORTFOLIO AGGREGATOR
# =============================================================================

class PortfolioAggregator:
    """
    Aggregates loan trajectories to portfolio and tranche level.
    """

    def __init__(
        self,
        spv_structure: SPVStructure = DEFAULT_CLO,
        lgd_mean: float = 0.45,
        lgd_std: float = 0.15
    ):
        self.spv = spv_structure
        self.lgd_mean = lgd_mean
        self.lgd_std = lgd_std

        self.total_pool = sum(t.size for t in spv_structure.tranches)

    def aggregate_cashflows(
        self,
        loan_balances: np.ndarray,
        loan_payments: np.ndarray,
        loan_states: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate loan-level data to portfolio cashflows.

        Args:
            loan_balances: (n_loans, n_months) - Outstanding balances
            loan_payments: (n_loans, n_months) - Payment amounts
            loan_states: (n_loans, n_months) - Loan states (int)

        Returns:
            Dict with portfolio-level cashflows
        """
        n_loans, n_months = loan_balances.shape

        # Aggregate
        total_balance = loan_balances.sum(axis=0)
        total_payments = loan_payments.sum(axis=0)

        # Count states
        n_performing = (loan_states == 0).sum(axis=0)
        n_delinquent = ((loan_states >= 1) & (loan_states <= 3)).sum(axis=0)
        n_default = (loan_states == 4).sum(axis=0)
        n_prepaid = (loan_states == 5).sum(axis=0)

        # Calculate losses from defaults
        rng = np.random.default_rng(42)
        losses = np.zeros(n_months)

        for t in range(1, n_months):
            # New defaults this month
            new_defaults = (loan_states[:, t] == 4) & (loan_states[:, t-1] != 4)
            for i in np.where(new_defaults)[0]:
                lgd = np.clip(rng.normal(self.lgd_mean, self.lgd_std), 0, 1)
                losses[t] += loan_balances[i, t-1] * lgd

        return {
            'total_balance': total_balance,
            'total_payments': total_payments,
            'losses': losses,
            'cumulative_losses': np.cumsum(losses),
            'n_performing': n_performing,
            'n_delinquent': n_delinquent,
            'n_default': n_default,
            'n_prepaid': n_prepaid,
            'pool_factor': total_balance / total_balance[0]
        }

    def apply_waterfall(
        self,
        portfolio_cashflows: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Apply waterfall to distribute cashflows to tranches.

        Args:
            portfolio_cashflows: From aggregate_cashflows

        Returns:
            Dict[tranche_name -> Dict of cashflow arrays]
        """
        n_months = len(portfolio_cashflows['total_payments'])
        collections = portfolio_cashflows['total_payments']
        cumulative_losses = portfolio_cashflows['cumulative_losses']

        tranche_results = {}

        for tranche in self.spv.tranches:
            tranche_results[tranche.name] = {
                'interest': np.zeros(n_months),
                'principal': np.zeros(n_months),
                'loss': np.zeros(n_months),
                'balance': np.full(n_months, tranche.size)
            }

        # Simple waterfall (monthly)
        for t in range(n_months):
            available = collections[t]
            loss_rate = cumulative_losses[t] / self.total_pool

            # Pay fees first
            senior_fees = self.total_pool * self.spv.senior_fees_pct / 12
            available -= min(available, senior_fees)

            # Pay tranches in priority order
            for tranche in sorted(self.spv.tranches, key=lambda x: x.priority):
                result = tranche_results[tranche.name]

                # Interest due
                interest_due = tranche.size * tranche.coupon / 12
                interest_paid = min(available, interest_due)
                result['interest'][t] = interest_paid
                available -= interest_paid

                # Check for losses hitting tranche
                if loss_rate > tranche.attachment:
                    loss_pct = min(loss_rate - tranche.attachment,
                                  tranche.detachment - tranche.attachment)
                    loss_pct = loss_pct / (tranche.detachment - tranche.attachment)
                    result['loss'][t] = tranche.size * loss_pct
                    result['balance'][t] = tranche.size * (1 - loss_pct)

        return tranche_results

    def compute_loss_distribution(
        self,
        simulated_losses: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute loss distribution statistics.

        Args:
            simulated_losses: (n_simulations,) - Total losses per simulation

        Returns:
            Dict with EL, VaR, CVaR, etc.
        """
        return {
            'expected_loss': simulated_losses.mean(),
            'std_loss': simulated_losses.std(),
            'var_90': np.percentile(simulated_losses, 90),
            'var_95': np.percentile(simulated_losses, 95),
            'var_99': np.percentile(simulated_losses, 99),
            'var_999': np.percentile(simulated_losses, 99.9),
            'cvar_95': simulated_losses[simulated_losses >= np.percentile(simulated_losses, 95)].mean(),
            'cvar_99': simulated_losses[simulated_losses >= np.percentile(simulated_losses, 99)].mean(),
            'min_loss': simulated_losses.min(),
            'max_loss': simulated_losses.max()
        }

    def compute_tranche_metrics(
        self,
        tranche_cashflows: Dict[str, Dict[str, np.ndarray]],
        discount_rate: float = 0.05
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute tranche-level metrics.

        Args:
            tranche_cashflows: From apply_waterfall
            discount_rate: Annual discount rate

        Returns:
            Dict[tranche_name -> metrics]
        """
        metrics = {}

        for name, flows in tranche_cashflows.items():
            n_months = len(flows['interest'])
            tranche = next(t for t in self.spv.tranches if t.name == name)

            # Total cashflows
            total_interest = flows['interest'].sum()
            total_principal = flows['principal'].sum()
            total_loss = flows['loss'].max()

            # IRR approximation via NPV
            monthly_rate = discount_rate / 12
            discount_factors = np.array([(1 + monthly_rate) ** -t for t in range(n_months)])

            pv_interest = (flows['interest'] * discount_factors).sum()
            pv_principal = (flows['principal'] * discount_factors).sum()

            # Return metrics
            metrics[name] = {
                'total_interest': total_interest,
                'total_principal': total_principal,
                'total_loss': total_loss,
                'loss_rate': total_loss / tranche.size,
                'pv_cashflows': pv_interest + pv_principal,
                'annualized_return': (total_interest + total_principal - total_loss) / tranche.size / (n_months / 12),
                'credit_enhancement': tranche.attachment
            }

        return metrics


# =============================================================================
# DIFFERENTIABLE AGGREGATOR (for end-to-end training)
# =============================================================================

class DifferentiableAggregator(nn.Module):
    """
    PyTorch-based aggregator for end-to-end training.
    """

    def __init__(
        self,
        n_tranches: int = 4,
        attachment_points: List[float] = [0.0, 0.70, 0.85, 0.95],
        detachment_points: List[float] = [0.70, 0.85, 0.95, 1.0]
    ):
        super().__init__()
        self.n_tranches = n_tranches

        self.register_buffer('attachments', torch.tensor(attachment_points))
        self.register_buffer('detachments', torch.tensor(detachment_points))

    def forward(
        self,
        loan_losses: torch.Tensor,
        loan_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute tranche losses from loan-level losses.

        Args:
            loan_losses: (batch, n_loans) - Loss per loan
            loan_weights: (batch, n_loans) - Exposure weight per loan

        Returns:
            tranche_losses: (batch, n_tranches)
        """
        # Portfolio loss rate
        portfolio_loss = (loan_losses * loan_weights).sum(dim=-1)
        total_weight = loan_weights.sum(dim=-1)
        loss_rate = portfolio_loss / (total_weight + 1e-8)

        # Tranche losses
        tranche_losses = torch.zeros(loan_losses.shape[0], self.n_tranches, device=loan_losses.device)

        for i in range(self.n_tranches):
            # Loss absorbed by tranche
            excess = torch.clamp(loss_rate - self.attachments[i], min=0)
            tranche_width = self.detachments[i] - self.attachments[i]
            tranche_loss_rate = torch.clamp(excess / tranche_width, max=1.0)
            tranche_losses[:, i] = tranche_loss_rate

        return tranche_losses


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Demonstrate portfolio aggregation"""

    print("=" * 60)
    print("PORTFOLIO AGGREGATOR - WATERFALL SIMULATION")
    print("=" * 60)

    # Create aggregator
    aggregator = PortfolioAggregator()

    # Generate synthetic loan data
    n_loans = 1000
    n_months = 60
    rng = np.random.default_rng(42)

    # Balances (decreasing over time)
    initial_balances = rng.lognormal(12, 1, n_loans)  # Mean ~100k
    loan_balances = np.zeros((n_loans, n_months))
    for i in range(n_loans):
        amort_rate = rng.uniform(0.01, 0.03)
        for t in range(n_months):
            loan_balances[i, t] = initial_balances[i] * (1 - amort_rate) ** t

    # Payments (roughly matching amortization)
    loan_payments = loan_balances * rng.uniform(0.01, 0.02, (n_loans, 1))

    # States (mostly performing, some defaults)
    loan_states = np.zeros((n_loans, n_months), dtype=int)
    default_probs = rng.uniform(0.001, 0.01, n_loans)  # Per-month
    for i in range(n_loans):
        for t in range(1, n_months):
            if loan_states[i, t-1] == 4:  # Already defaulted
                loan_states[i, t] = 4
            elif rng.random() < default_probs[i]:
                loan_states[i, t] = 4

    # Aggregate
    print("\nAggregating portfolio cashflows...")
    cashflows = aggregator.aggregate_cashflows(loan_balances, loan_payments, loan_states)

    print(f"\nPortfolio Statistics:")
    print(f"  Initial pool: EUR {cashflows['total_balance'][0]:,.0f}")
    print(f"  Final pool: EUR {cashflows['total_balance'][-1]:,.0f}")
    print(f"  Total payments: EUR {cashflows['total_payments'].sum():,.0f}")
    print(f"  Cumulative losses: EUR {cashflows['cumulative_losses'][-1]:,.0f}")
    print(f"  Loss rate: {cashflows['cumulative_losses'][-1] / cashflows['total_balance'][0]:.2%}")
    print(f"  Final default count: {cashflows['n_default'][-1]}")

    # Apply waterfall
    print("\nApplying waterfall...")
    tranche_flows = aggregator.apply_waterfall(cashflows)

    # Compute metrics
    metrics = aggregator.compute_tranche_metrics(tranche_flows)

    print("\nTranche Metrics:")
    print("-" * 60)
    for name, m in metrics.items():
        print(f"\n{name}:")
        print(f"  Total Interest: EUR {m['total_interest']:,.0f}")
        print(f"  Loss Rate: {m['loss_rate']:.2%}")
        print(f"  Credit Enhancement: {m['credit_enhancement']:.0%}")

    # Monte Carlo loss distribution
    print("\n" + "=" * 60)
    print("MONTE CARLO LOSS DISTRIBUTION")
    print("=" * 60)

    n_sims = 1000
    simulated_losses = np.zeros(n_sims)

    for sim in range(n_sims):
        # Vary default rates
        sim_default_probs = default_probs * rng.lognormal(0, 0.3, n_loans)
        sim_states = np.zeros((n_loans, n_months), dtype=int)

        for i in range(n_loans):
            for t in range(1, n_months):
                if sim_states[i, t-1] == 4:
                    sim_states[i, t] = 4
                elif rng.random() < sim_default_probs[i]:
                    sim_states[i, t] = 4

        sim_cashflows = aggregator.aggregate_cashflows(loan_balances, loan_payments, sim_states)
        simulated_losses[sim] = sim_cashflows['cumulative_losses'][-1]

    # Loss statistics
    loss_stats = aggregator.compute_loss_distribution(simulated_losses)

    print(f"\nLoss Distribution (n={n_sims} simulations):")
    print(f"  Expected Loss: EUR {loss_stats['expected_loss']:,.0f}")
    print(f"  VaR 95%: EUR {loss_stats['var_95']:,.0f}")
    print(f"  VaR 99%: EUR {loss_stats['var_99']:,.0f}")
    print(f"  CVaR 99%: EUR {loss_stats['cvar_99']:,.0f}")

    # As percentage
    initial_pool = cashflows['total_balance'][0]
    print(f"\nAs % of initial pool:")
    print(f"  Expected Loss: {loss_stats['expected_loss'] / initial_pool:.2%}")
    print(f"  VaR 95%: {loss_stats['var_95'] / initial_pool:.2%}")
    print(f"  VaR 99%: {loss_stats['var_99'] / initial_pool:.2%}")


if __name__ == '__main__':
    main()

"""
Baseline Markov Chain Model for Credit Risk

A simple multi-state Markov model for loan transitions:
- States: Performing, 30DPD, 60DPD, 90DPD, Default, Prepaid, Matured
- Transition probabilities can be conditioned on macro factors
- Serves as benchmark for deep generative models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

STATES = ['performing', '30dpd', '60dpd', '90dpd', 'default', 'prepaid', 'matured']
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
IDX_TO_STATE = {i: s for s, i in STATE_TO_IDX.items()}
N_STATES = len(STATES)

# Absorbing states (terminal)
ABSORBING_STATES = {'default', 'prepaid', 'matured'}


# =============================================================================
# MARKOV MODEL
# =============================================================================

class MarkovTransitionModel:
    """
    Multi-state Markov chain for credit risk modeling.

    Features:
    - Macro-conditioned transition probabilities
    - Loan-level heterogeneity via credit score adjustment
    - Cohort-level calibration
    """

    def __init__(
        self,
        asset_class: str = 'corporate',
        random_seed: int = 42
    ):
        self.asset_class = asset_class
        self.rng = np.random.default_rng(random_seed)

        # Base transition matrix (rows: from, cols: to)
        self.base_matrix = self._get_base_transition_matrix(asset_class)

        # Macro sensitivity parameters
        self.macro_elasticities = self._get_macro_elasticities()

    def _get_base_transition_matrix(self, asset_class: str) -> np.ndarray:
        """
        Get base monthly transition matrix for asset class.

        Matrix structure:
        - Rows: from states (performing, 30dpd, 60dpd, 90dpd, default, prepaid, matured)
        - Cols: to states (same order)
        - Row sums = 1
        """

        # Default corporate transitions
        if asset_class == 'corporate':
            matrix = np.array([
                # perf    30dpd   60dpd   90dpd   default prepaid matured
                [0.9750,  0.0150, 0.0000, 0.0000, 0.0000, 0.0080, 0.0020],  # performing
                [0.4000,  0.2950, 0.3000, 0.0000, 0.0000, 0.0050, 0.0000],  # 30dpd
                [0.2000,  0.0000, 0.3950, 0.4000, 0.0000, 0.0050, 0.0000],  # 60dpd
                [0.1000,  0.0000, 0.0000, 0.3950, 0.5000, 0.0050, 0.0000],  # 90dpd
                [0.0000,  0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],  # default (absorbing)
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],  # prepaid (absorbing)
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # matured (absorbing)
            ])
        elif asset_class == 'consumer':
            matrix = np.array([
                [0.9600,  0.0250, 0.0000, 0.0000, 0.0000, 0.0130, 0.0020],
                [0.3500,  0.2900, 0.3500, 0.0000, 0.0000, 0.0100, 0.0000],
                [0.1500,  0.0000, 0.3900, 0.4500, 0.0000, 0.0100, 0.0000],
                [0.0800,  0.0000, 0.0000, 0.3600, 0.5500, 0.0100, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
            ])
        elif asset_class == 'realestate':
            matrix = np.array([
                [0.9850,  0.0080, 0.0000, 0.0000, 0.0000, 0.0050, 0.0020],
                [0.4500,  0.2900, 0.2500, 0.0000, 0.0000, 0.0100, 0.0000],
                [0.2500,  0.0000, 0.3900, 0.3500, 0.0000, 0.0100, 0.0000],
                [0.1200,  0.0000, 0.0000, 0.4200, 0.4500, 0.0100, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
            ])
        else:  # receivables
            matrix = np.array([
                [0.9850,  0.0100, 0.0000, 0.0000, 0.0000, 0.0000, 0.0050],
                [0.5000,  0.2800, 0.2000, 0.0000, 0.0000, 0.0000, 0.0200],
                [0.3000,  0.0000, 0.3900, 0.3000, 0.0000, 0.0000, 0.0100],
                [0.1500,  0.0000, 0.0000, 0.4400, 0.4000, 0.0000, 0.0100],
                [0.0000,  0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000],
                [0.0000,  0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
            ])

        # Ensure rows sum to 1
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

        return matrix

    def _get_macro_elasticities(self) -> Dict[str, np.ndarray]:
        """
        Elasticities for how transition probabilities respond to macro factors.

        Structure: For each macro variable, a matrix of elasticities
        where elasticity[i,j] is d(log(p_ij))/d(macro)
        """

        # Unemployment elasticity
        # Higher unemployment -> more transitions to worse states
        unemp_elasticity = np.zeros((N_STATES, N_STATES))
        unemp_elasticity[0, 1] = 1.5   # perf -> 30dpd
        unemp_elasticity[1, 2] = 1.0   # 30 -> 60
        unemp_elasticity[2, 3] = 0.8   # 60 -> 90
        unemp_elasticity[3, 4] = 0.6   # 90 -> default
        unemp_elasticity[1, 0] = -0.8  # 30 -> cure
        unemp_elasticity[2, 0] = -0.6  # 60 -> cure
        unemp_elasticity[3, 0] = -0.4  # 90 -> cure
        unemp_elasticity[0, 5] = -0.3  # perf -> prepay

        # GDP elasticity (inverted)
        gdp_elasticity = -1.5 * unemp_elasticity  # Opposite of unemployment

        # Credit spread elasticity
        spread_elasticity = np.zeros((N_STATES, N_STATES))
        spread_elasticity[0, 1] = 0.8
        spread_elasticity[1, 2] = 0.5
        spread_elasticity[2, 3] = 0.4
        spread_elasticity[3, 4] = 0.3
        spread_elasticity[1, 0] = -0.3
        spread_elasticity[2, 0] = -0.2

        return {
            'unemployment': unemp_elasticity,
            'gdp': gdp_elasticity,
            'spread': spread_elasticity
        }

    def adjust_matrix_for_macro(
        self,
        base_matrix: np.ndarray,
        macro_factors: Dict[str, float]
    ) -> np.ndarray:
        """
        Adjust transition matrix based on macro conditions.

        Args:
            base_matrix: Base transition matrix
            macro_factors: Dict with keys 'unemployment_dev', 'gdp_dev', 'spread_dev'
                          representing deviations from baseline (in std units)

        Returns:
            Adjusted transition matrix
        """

        adjusted = base_matrix.copy()

        for macro_var, deviation in macro_factors.items():
            if macro_var in self.macro_elasticities:
                elasticity = self.macro_elasticities[macro_var]
                # Log-linear adjustment
                adjustment = np.exp(elasticity * deviation * 0.1)
                adjusted = adjusted * adjustment

        # Re-normalize rows
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

        # Ensure absorbing states stay absorbing
        for state in ABSORBING_STATES:
            idx = STATE_TO_IDX[state]
            adjusted[idx, :] = 0.0
            adjusted[idx, idx] = 1.0

        return adjusted

    def adjust_matrix_for_loan_quality(
        self,
        base_matrix: np.ndarray,
        credit_score: float,
        baseline_score: float = 75.0
    ) -> np.ndarray:
        """
        Adjust transition matrix based on loan quality (credit score).

        Args:
            base_matrix: Transition matrix
            credit_score: Loan's internal credit score (0-100)
            baseline_score: Baseline score for calibration

        Returns:
            Quality-adjusted transition matrix
        """

        # Score deviation (positive = better quality)
        score_dev = (credit_score - baseline_score) / 25.0

        adjusted = base_matrix.copy()

        # Better quality -> lower default transitions, higher cure
        for i in range(N_STATES):
            for j in range(N_STATES):
                if STATES[j] in ['30dpd', '60dpd', '90dpd', 'default']:
                    adjusted[i, j] *= np.exp(-0.2 * score_dev)
                elif STATES[j] == 'performing' and STATES[i] in ['30dpd', '60dpd', '90dpd']:
                    adjusted[i, j] *= np.exp(0.15 * score_dev)

        # Re-normalize
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)

        # Ensure absorbing states
        for state in ABSORBING_STATES:
            idx = STATE_TO_IDX[state]
            adjusted[idx, :] = 0.0
            adjusted[idx, idx] = 1.0

        return adjusted

    def simulate_path(
        self,
        initial_state: str,
        n_months: int,
        macro_path: Optional[pd.DataFrame] = None,
        credit_score: float = 75.0,
        maturity_month: Optional[int] = None
    ) -> List[str]:
        """
        Simulate state path for a single loan.

        Args:
            initial_state: Starting state
            n_months: Number of months to simulate
            macro_path: DataFrame with macro factors by month
            credit_score: Loan's credit score
            maturity_month: Month when loan matures (0-indexed)

        Returns:
            List of states for each month
        """

        path = [initial_state]
        current_state = initial_state

        for t in range(1, n_months):
            # Check if already in absorbing state
            if current_state in ABSORBING_STATES:
                path.append(current_state)
                continue

            # Check maturity
            if maturity_month is not None and t >= maturity_month:
                if current_state == 'performing':
                    current_state = 'matured'
                    path.append(current_state)
                    continue

            # Get transition matrix
            matrix = self.base_matrix.copy()

            # Adjust for credit quality
            matrix = self.adjust_matrix_for_loan_quality(matrix, credit_score)

            # Adjust for macro
            if macro_path is not None and t < len(macro_path):
                row = macro_path.iloc[t]
                macro_factors = {
                    'unemployment': (row.get('unemployment_rate', 0.05) - 0.05) / 0.02,
                    'gdp': (row.get('gdp_growth_yoy', 0.02) - 0.02) / 0.01,
                    'spread': (row.get('credit_spread_hy', 400) - 400) / 200
                }
                matrix = self.adjust_matrix_for_macro(matrix, macro_factors)

            # Sample next state
            current_idx = STATE_TO_IDX[current_state]
            probs = matrix[current_idx]
            next_idx = self.rng.choice(N_STATES, p=probs)
            current_state = IDX_TO_STATE[next_idx]

            path.append(current_state)

        return path

    def simulate_cohort(
        self,
        n_loans: int,
        n_months: int,
        credit_scores: Optional[np.ndarray] = None,
        macro_path: Optional[pd.DataFrame] = None,
        maturity_months: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Simulate state paths for a cohort of loans.

        Args:
            n_loans: Number of loans
            n_months: Simulation horizon
            credit_scores: Array of credit scores
            macro_path: Macro scenario DataFrame
            maturity_months: Array of maturity months

        Returns:
            Tuple of (state_matrix [n_loans x n_months], summary_df)
        """

        if credit_scores is None:
            credit_scores = self.rng.uniform(50, 100, n_loans)

        if maturity_months is None:
            maturity_months = np.full(n_loans, n_months + 12)  # Beyond simulation

        # Simulate all paths
        states = np.zeros((n_loans, n_months), dtype=int)

        for i in range(n_loans):
            path = self.simulate_path(
                initial_state='performing',
                n_months=n_months,
                macro_path=macro_path,
                credit_score=credit_scores[i],
                maturity_month=maturity_months[i]
            )
            states[i] = [STATE_TO_IDX[s] for s in path]

        # Compute statistics
        summary = self._compute_cohort_statistics(states)

        return states, summary

    def _compute_cohort_statistics(self, states: np.ndarray) -> pd.DataFrame:
        """Compute summary statistics for simulated cohort"""

        n_loans, n_months = states.shape

        stats = []
        for t in range(n_months):
            month_states = states[:, t]
            state_counts = {s: (month_states == i).sum() / n_loans
                          for s, i in STATE_TO_IDX.items()}

            stats.append({
                'month': t,
                'pct_performing': state_counts['performing'],
                'pct_30dpd': state_counts['30dpd'],
                'pct_60dpd': state_counts['60dpd'],
                'pct_90dpd': state_counts['90dpd'],
                'pct_default': state_counts['default'],
                'pct_prepaid': state_counts['prepaid'],
                'pct_matured': state_counts['matured'],
                'cumulative_default': state_counts['default'],
                'cumulative_prepaid': state_counts['prepaid']
            })

        df = pd.DataFrame(stats)
        return df

    def estimate_from_data(
        self,
        panel_df: pd.DataFrame,
        state_column: str = 'loan_state',
        loan_id_column: str = 'loan_id',
        month_column: str = 'reporting_month'
    ) -> np.ndarray:
        """
        Estimate transition matrix from historical panel data.

        Args:
            panel_df: Loan-month panel with state observations
            state_column: Column with loan state
            loan_id_column: Column with loan ID
            month_column: Column with reporting month

        Returns:
            Estimated transition matrix
        """

        # Sort by loan and month
        df = panel_df.sort_values([loan_id_column, month_column])

        # Create lagged state
        df['prev_state'] = df.groupby(loan_id_column)[state_column].shift(1)
        df = df.dropna(subset=['prev_state'])

        # Count transitions
        transition_counts = np.zeros((N_STATES, N_STATES))

        for _, row in df.iterrows():
            from_state = row['prev_state']
            to_state = row[state_column]

            if from_state in STATE_TO_IDX and to_state in STATE_TO_IDX:
                i = STATE_TO_IDX[from_state]
                j = STATE_TO_IDX[to_state]
                transition_counts[i, j] += 1

        # Normalize to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        estimated_matrix = transition_counts / row_sums

        return estimated_matrix

    def compute_pd_curve(
        self,
        horizon_months: int = 60,
        macro_path: Optional[pd.DataFrame] = None,
        credit_score: float = 75.0
    ) -> np.ndarray:
        """
        Compute PD term structure (cumulative default probability by month).

        Args:
            horizon_months: Projection horizon
            macro_path: Macro scenario
            credit_score: Loan quality

        Returns:
            Array of cumulative PD by month
        """

        # Start with performing
        state_prob = np.zeros(N_STATES)
        state_prob[STATE_TO_IDX['performing']] = 1.0

        pd_curve = np.zeros(horizon_months)

        for t in range(horizon_months):
            # Get transition matrix for this month
            matrix = self.base_matrix.copy()
            matrix = self.adjust_matrix_for_loan_quality(matrix, credit_score)

            if macro_path is not None and t < len(macro_path):
                row = macro_path.iloc[t]
                macro_factors = {
                    'unemployment': (row.get('unemployment_rate', 0.05) - 0.05) / 0.02,
                    'gdp': (row.get('gdp_growth_yoy', 0.02) - 0.02) / 0.01,
                    'spread': (row.get('credit_spread_hy', 400) - 400) / 200
                }
                matrix = self.adjust_matrix_for_macro(matrix, macro_factors)

            # Propagate state distribution
            state_prob = state_prob @ matrix

            # Record cumulative default probability
            pd_curve[t] = state_prob[STATE_TO_IDX['default']]

        return pd_curve


# =============================================================================
# LOSS DISTRIBUTION
# =============================================================================

class PortfolioLossModel:
    """
    Monte Carlo simulation for portfolio loss distribution.
    """

    def __init__(
        self,
        markov_model: MarkovTransitionModel,
        lgd_mean: float = 0.45,
        lgd_std: float = 0.15,
        random_seed: int = 42
    ):
        self.markov = markov_model
        self.lgd_mean = lgd_mean
        self.lgd_std = lgd_std
        self.rng = np.random.default_rng(random_seed)

    def simulate_portfolio_loss(
        self,
        loans_df: pd.DataFrame,
        macro_path: pd.DataFrame,
        n_simulations: int = 1000
    ) -> Dict[str, np.ndarray]:
        """
        Simulate portfolio loss distribution.

        Args:
            loans_df: Loan tape with static features
            macro_path: Macro scenario
            n_simulations: Number of Monte Carlo paths

        Returns:
            Dict with 'losses' array and statistics
        """

        n_loans = len(loans_df)
        n_months = len(macro_path)

        total_exposure = loans_df['original_balance'].sum()
        balances = loans_df['original_balance'].values
        scores = loans_df['internal_score_origination'].values
        terms = loans_df['term_months'].values

        portfolio_losses = np.zeros(n_simulations)

        for sim in range(n_simulations):
            if sim % 100 == 0:
                print(f"Simulation {sim}/{n_simulations}")

            # Simulate cohort
            states, _ = self.markov.simulate_cohort(
                n_loans=n_loans,
                n_months=n_months,
                credit_scores=scores,
                macro_path=macro_path,
                maturity_months=terms
            )

            # Calculate losses
            sim_loss = 0.0
            for i in range(n_loans):
                # Check if defaulted
                final_state = IDX_TO_STATE[states[i, -1]]
                if final_state == 'default':
                    # Find default month
                    default_month = np.argmax(states[i] == STATE_TO_IDX['default'])
                    # Remaining balance at default (simplified)
                    remaining_pct = 1.0 - default_month / terms[i]
                    remaining_balance = balances[i] * max(0, remaining_pct)
                    # LGD
                    lgd = np.clip(self.rng.normal(self.lgd_mean, self.lgd_std), 0, 1)
                    sim_loss += remaining_balance * lgd

            portfolio_losses[sim] = sim_loss

        # Compute statistics
        results = {
            'losses': portfolio_losses,
            'total_exposure': total_exposure,
            'expected_loss': portfolio_losses.mean(),
            'expected_loss_pct': portfolio_losses.mean() / total_exposure,
            'std_loss': portfolio_losses.std(),
            'var_95': np.percentile(portfolio_losses, 95),
            'var_99': np.percentile(portfolio_losses, 99),
            'cvar_95': portfolio_losses[portfolio_losses >= np.percentile(portfolio_losses, 95)].mean(),
            'cvar_99': portfolio_losses[portfolio_losses >= np.percentile(portfolio_losses, 99)].mean()
        }

        return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run baseline Markov model demonstration"""

    print("=" * 60)
    print("BASELINE MARKOV TRANSITION MODEL")
    print("=" * 60)

    # Load or generate data
    data_dir = Path(__file__).parent.parent.parent / 'data'

    if (data_dir / 'loans_static.csv').exists():
        loans_df = pd.read_csv(data_dir / 'loans_static.csv')
        macro_df = pd.read_csv(data_dir / 'macro_baseline.csv')
    else:
        print("Generating synthetic data first...")
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'data'))
        from simulate_loans import LoanTapeGenerator
        from simulate_macro import MacroScenarioGenerator

        macro_gen = MacroScenarioGenerator(n_months=60, start_date='2020-01-01')
        macro_df = macro_gen.generate_baseline()

        loan_gen = LoanTapeGenerator(n_loans=1000, n_months=60)
        loans_df, _ = loan_gen.generate(macro_df)

    # Initialize model
    model = MarkovTransitionModel(asset_class='corporate')

    print("\nBase transition matrix:")
    print(pd.DataFrame(
        model.base_matrix,
        index=STATES,
        columns=STATES
    ).round(4))

    # Compute PD curves under different scenarios
    print("\n" + "=" * 60)
    print("PD TERM STRUCTURES")
    print("=" * 60)

    pd_baseline = model.compute_pd_curve(horizon_months=60, credit_score=75)
    pd_high_quality = model.compute_pd_curve(horizon_months=60, credit_score=90)
    pd_low_quality = model.compute_pd_curve(horizon_months=60, credit_score=60)

    print(f"\n12-month PD (baseline score): {pd_baseline[11]:.2%}")
    print(f"12-month PD (high score 90): {pd_high_quality[11]:.2%}")
    print(f"12-month PD (low score 60): {pd_low_quality[11]:.2%}")
    print(f"\nLifetime PD (60 months, baseline): {pd_baseline[59]:.2%}")

    # Simulate cohort
    print("\n" + "=" * 60)
    print("COHORT SIMULATION")
    print("=" * 60)

    states, summary = model.simulate_cohort(
        n_loans=1000,
        n_months=60,
        macro_path=macro_df
    )

    print(f"\nFinal state distribution (month 60):")
    print(f"  Performing: {summary.iloc[-1]['pct_performing']:.2%}")
    print(f"  Default: {summary.iloc[-1]['pct_default']:.2%}")
    print(f"  Prepaid: {summary.iloc[-1]['pct_prepaid']:.2%}")
    print(f"  Matured: {summary.iloc[-1]['pct_matured']:.2%}")

    # Portfolio loss simulation
    print("\n" + "=" * 60)
    print("PORTFOLIO LOSS DISTRIBUTION")
    print("=" * 60)

    loss_model = PortfolioLossModel(model)
    results = loss_model.simulate_portfolio_loss(
        loans_df.head(1000),
        macro_df,
        n_simulations=500
    )

    print(f"\nPortfolio Exposure: EUR {results['total_exposure']:,.0f}")
    print(f"Expected Loss: EUR {results['expected_loss']:,.0f} ({results['expected_loss_pct']:.2%})")
    print(f"VaR 95%: EUR {results['var_95']:,.0f}")
    print(f"VaR 99%: EUR {results['var_99']:,.0f}")
    print(f"CVaR 95%: EUR {results['cvar_95']:,.0f}")
    print(f"CVaR 99%: EUR {results['cvar_99']:,.0f}")


if __name__ == '__main__':
    main()

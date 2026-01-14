"""
Macro Scenario Generator for Private Credit SPV Modeling

Generates correlated macroeconomic time series for:
- Baseline scenarios
- Adverse scenarios (recession)
- Severely adverse scenarios (financial crisis)

Includes VAR-based simulation for realistic cross-correlations.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.linalg import cholesky


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MacroVariable:
    """Configuration for a macroeconomic variable"""
    name: str
    baseline_mean: float
    baseline_std: float
    min_val: float
    max_val: float
    persistence: float  # AR(1) coefficient
    unit: str


MACRO_VARIABLES = {
    'gdp_growth_yoy': MacroVariable(
        name='GDP Growth YoY',
        baseline_mean=0.02,
        baseline_std=0.01,
        min_val=-0.15,
        max_val=0.15,
        persistence=0.7,
        unit='annual %'
    ),
    'unemployment_rate': MacroVariable(
        name='Unemployment Rate',
        baseline_mean=0.05,
        baseline_std=0.01,
        min_val=0.02,
        max_val=0.25,
        persistence=0.95,
        unit='%'
    ),
    'inflation_rate': MacroVariable(
        name='Inflation Rate',
        baseline_mean=0.02,
        baseline_std=0.005,
        min_val=-0.05,
        max_val=0.15,
        persistence=0.8,
        unit='annual %'
    ),
    'policy_rate': MacroVariable(
        name='Policy Rate',
        baseline_mean=0.03,
        baseline_std=0.01,
        min_val=0.0,
        max_val=0.15,
        persistence=0.98,
        unit='%'
    ),
    'yield_10y': MacroVariable(
        name='10Y Government Yield',
        baseline_mean=0.035,
        baseline_std=0.008,
        min_val=0.0,
        max_val=0.10,
        persistence=0.95,
        unit='%'
    ),
    'credit_spread_ig': MacroVariable(
        name='IG Credit Spread',
        baseline_mean=100,
        baseline_std=30,
        min_val=50,
        max_val=500,
        persistence=0.9,
        unit='bps'
    ),
    'credit_spread_hy': MacroVariable(
        name='HY Credit Spread',
        baseline_mean=400,
        baseline_std=100,
        min_val=200,
        max_val=2000,
        persistence=0.85,
        unit='bps'
    ),
    'property_price_index': MacroVariable(
        name='Property Price Index',
        baseline_mean=100,
        baseline_std=5,
        min_val=50,
        max_val=200,
        persistence=0.98,
        unit='index'
    ),
    'equity_return': MacroVariable(
        name='Equity Market Return',
        baseline_mean=0.08,
        baseline_std=0.15,
        min_val=-0.50,
        max_val=0.50,
        persistence=0.1,
        unit='annual %'
    )
}

# Cross-correlation matrix (approximate)
# Order: gdp, unemp, infl, policy, yield10y, ig_spread, hy_spread, property, equity
CORRELATION_MATRIX = np.array([
    [ 1.0, -0.6,  0.3,  0.4,  0.3, -0.5, -0.6,  0.5,  0.4],  # gdp
    [-0.6,  1.0, -0.2, -0.3, -0.2,  0.6,  0.7, -0.4, -0.3],  # unemployment
    [ 0.3, -0.2,  1.0,  0.7,  0.6,  0.1,  0.0,  0.3,  0.0],  # inflation
    [ 0.4, -0.3,  0.7,  1.0,  0.8, -0.2, -0.3,  0.2,  0.2],  # policy
    [ 0.3, -0.2,  0.6,  0.8,  1.0, -0.1, -0.2,  0.2,  0.1],  # yield10y
    [-0.5,  0.6,  0.1, -0.2, -0.1,  1.0,  0.9, -0.3, -0.5],  # ig spread
    [-0.6,  0.7,  0.0, -0.3, -0.2,  0.9,  1.0, -0.4, -0.6],  # hy spread
    [ 0.5, -0.4,  0.3,  0.2,  0.2, -0.3, -0.4,  1.0,  0.4],  # property
    [ 0.4, -0.3,  0.0,  0.2,  0.1, -0.5, -0.6,  0.4,  1.0],  # equity
])


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

@dataclass
class ScenarioConfig:
    """Configuration for a macro scenario"""
    name: str
    # Shifts from baseline (in standard deviations or absolute)
    gdp_shift: float = 0.0
    unemployment_shift: float = 0.0
    inflation_shift: float = 0.0
    spread_multiplier: float = 1.0
    property_drawdown: float = 0.0
    shock_start_month: int = 6  # When stress begins
    shock_duration_months: int = 12  # How long stress lasts
    recovery_months: int = 24  # Recovery period


SCENARIOS = {
    'baseline': ScenarioConfig(
        name='Baseline',
        gdp_shift=0.0,
        unemployment_shift=0.0,
        inflation_shift=0.0,
        spread_multiplier=1.0,
        property_drawdown=0.0
    ),
    'adverse': ScenarioConfig(
        name='Adverse (Recession)',
        gdp_shift=-0.03,
        unemployment_shift=0.03,
        inflation_shift=-0.005,
        spread_multiplier=2.0,
        property_drawdown=0.15,
        shock_start_month=6,
        shock_duration_months=12,
        recovery_months=24
    ),
    'severely_adverse': ScenarioConfig(
        name='Severely Adverse (Crisis)',
        gdp_shift=-0.06,
        unemployment_shift=0.08,
        inflation_shift=-0.01,
        spread_multiplier=4.0,
        property_drawdown=0.35,
        shock_start_month=6,
        shock_duration_months=18,
        recovery_months=36
    ),
    'stagflation': ScenarioConfig(
        name='Stagflation',
        gdp_shift=-0.02,
        unemployment_shift=0.03,
        inflation_shift=0.05,
        spread_multiplier=2.5,
        property_drawdown=0.10,
        shock_start_month=6,
        shock_duration_months=24,
        recovery_months=36
    )
}


# =============================================================================
# MACRO GENERATOR
# =============================================================================

class MacroScenarioGenerator:
    """
    Generates correlated macroeconomic scenarios using VAR-like dynamics.
    """

    def __init__(
        self,
        n_months: int = 60,
        start_date: str = '2020-01-01',
        random_seed: int = 42
    ):
        self.n_months = n_months
        self.start_date = pd.Timestamp(start_date)
        self.rng = np.random.default_rng(random_seed)

        self.variables = list(MACRO_VARIABLES.keys())
        self.n_vars = len(self.variables)

        # Ensure correlation matrix is positive definite
        self.corr_matrix = self._ensure_positive_definite(CORRELATION_MATRIX)
        self.chol = cholesky(self.corr_matrix, lower=True)

    def _ensure_positive_definite(self, matrix: np.ndarray) -> np.ndarray:
        """Ensure correlation matrix is positive definite"""
        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)

        # Replace negative eigenvalues with small positive values
        eigenvalues = np.maximum(eigenvalues, 1e-6)

        # Reconstruct matrix
        matrix_fixed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        # Normalize to correlation matrix
        d = np.sqrt(np.diag(matrix_fixed))
        matrix_fixed = matrix_fixed / np.outer(d, d)

        return matrix_fixed

    def generate_correlated_shocks(self) -> np.ndarray:
        """Generate correlated standard normal shocks"""
        # Independent shocks
        z = self.rng.standard_normal((self.n_months, self.n_vars))

        # Apply Cholesky to induce correlation
        correlated = z @ self.chol.T

        return correlated

    def generate_baseline(self) -> pd.DataFrame:
        """Generate baseline macroeconomic scenario"""

        shocks = self.generate_correlated_shocks()

        data = {}
        dates = pd.date_range(self.start_date, periods=self.n_months, freq='MS')
        data['reporting_month'] = [d.strftime('%Y-%m') for d in dates]

        for i, var_name in enumerate(self.variables):
            config = MACRO_VARIABLES[var_name]

            # AR(1) process
            series = np.zeros(self.n_months)
            series[0] = config.baseline_mean

            for t in range(1, self.n_months):
                mean_reversion = config.persistence * (series[t-1] - config.baseline_mean)
                innovation = config.baseline_std * shocks[t, i] * np.sqrt(1 - config.persistence**2)
                series[t] = config.baseline_mean + mean_reversion + innovation

            # Apply bounds
            series = np.clip(series, config.min_val, config.max_val)
            data[var_name] = series

        return pd.DataFrame(data)

    def generate_scenario(
        self,
        scenario_name: str = 'adverse'
    ) -> pd.DataFrame:
        """Generate stressed macroeconomic scenario"""

        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario = SCENARIOS[scenario_name]

        if scenario_name == 'baseline':
            return self.generate_baseline()

        # Start with baseline
        df = self.generate_baseline()

        # Apply scenario shocks
        shock_start = scenario.shock_start_month
        shock_end = shock_start + scenario.shock_duration_months
        recovery_end = shock_end + scenario.recovery_months

        for t in range(self.n_months):
            if t < shock_start:
                # Pre-shock: no adjustment
                multiplier = 0.0
            elif t < shock_end:
                # During shock: ramp up
                progress = (t - shock_start) / scenario.shock_duration_months
                multiplier = np.sin(progress * np.pi / 2)  # Smooth ramp
            elif t < recovery_end:
                # Recovery: ramp down
                progress = (t - shock_end) / scenario.recovery_months
                multiplier = 1.0 - progress
            else:
                # Post-recovery: back to normal
                multiplier = 0.0

            # Apply shifts
            df.loc[t, 'gdp_growth_yoy'] += scenario.gdp_shift * multiplier
            df.loc[t, 'unemployment_rate'] += scenario.unemployment_shift * multiplier
            df.loc[t, 'inflation_rate'] += scenario.inflation_shift * multiplier

            # Spread multiplier
            base_ig = MACRO_VARIABLES['credit_spread_ig'].baseline_mean
            base_hy = MACRO_VARIABLES['credit_spread_hy'].baseline_mean
            spread_adj = 1.0 + (scenario.spread_multiplier - 1.0) * multiplier
            df.loc[t, 'credit_spread_ig'] = base_ig * spread_adj + (df.loc[t, 'credit_spread_ig'] - base_ig)
            df.loc[t, 'credit_spread_hy'] = base_hy * spread_adj + (df.loc[t, 'credit_spread_hy'] - base_hy)

            # Property price drawdown
            if t >= shock_start:
                property_adj = 1.0 - scenario.property_drawdown * multiplier
                df.loc[t, 'property_price_index'] *= property_adj

        # Apply bounds
        for var_name in self.variables:
            config = MACRO_VARIABLES[var_name]
            df[var_name] = df[var_name].clip(config.min_val, config.max_val)

        return df

    def generate_monte_carlo(
        self,
        n_paths: int = 1000,
        scenario_weights: Dict[str, float] = None
    ) -> List[pd.DataFrame]:
        """Generate Monte Carlo paths with scenario mixing"""

        if scenario_weights is None:
            scenario_weights = {'baseline': 0.7, 'adverse': 0.2, 'severely_adverse': 0.1}

        scenarios = list(scenario_weights.keys())
        weights = list(scenario_weights.values())

        paths = []
        for i in range(n_paths):
            # Select scenario
            scenario = self.rng.choice(scenarios, p=weights)

            # Re-seed for variety
            self.rng = np.random.default_rng(42 + i)

            # Generate path
            path = self.generate_scenario(scenario)
            path['scenario'] = scenario
            path['path_id'] = i
            paths.append(path)

        return paths

    def compute_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute summary statistics for macro scenario"""

        stats_data = []
        for var_name in self.variables:
            if var_name not in df.columns:
                continue

            series = df[var_name]
            config = MACRO_VARIABLES[var_name]

            stats_data.append({
                'variable': var_name,
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'baseline_mean': config.baseline_mean,
                'deviation_from_baseline': (series.mean() - config.baseline_mean) / config.baseline_std
            })

        return pd.DataFrame(stats_data)


# =============================================================================
# SECTOR DEFAULT RATES
# =============================================================================

class SectorDefaultGenerator:
    """
    Generates sector-specific default rates based on macro conditions.
    """

    # Base annual default rates by sector
    BASE_DEFAULT_RATES = {
        'A_agriculture': 0.015,
        'C_manufacturing': 0.020,
        'F_construction': 0.035,
        'G_wholesale_retail': 0.025,
        'H_transport': 0.020,
        'I_hospitality': 0.040,
        'J_information': 0.025,
        'K_finance': 0.015,
        'L_real_estate': 0.020,
        'M_professional': 0.015,
        'N_admin': 0.030,
        'Q_health': 0.010
    }

    # Sensitivity to macro factors (elasticities)
    MACRO_SENSITIVITY = {
        'A_agriculture': {'gdp': -1.0, 'unemployment': 0.5, 'spread': 0.3},
        'C_manufacturing': {'gdp': -2.0, 'unemployment': 1.0, 'spread': 0.5},
        'F_construction': {'gdp': -2.5, 'unemployment': 1.5, 'spread': 0.8},
        'G_wholesale_retail': {'gdp': -1.5, 'unemployment': 1.2, 'spread': 0.4},
        'H_transport': {'gdp': -1.8, 'unemployment': 0.8, 'spread': 0.5},
        'I_hospitality': {'gdp': -3.0, 'unemployment': 2.0, 'spread': 0.6},
        'J_information': {'gdp': -1.2, 'unemployment': 0.6, 'spread': 0.4},
        'K_finance': {'gdp': -1.5, 'unemployment': 0.5, 'spread': 1.0},
        'L_real_estate': {'gdp': -2.0, 'unemployment': 1.0, 'spread': 0.8},
        'M_professional': {'gdp': -1.0, 'unemployment': 0.5, 'spread': 0.3},
        'N_admin': {'gdp': -1.8, 'unemployment': 1.5, 'spread': 0.5},
        'Q_health': {'gdp': -0.5, 'unemployment': 0.3, 'spread': 0.2}
    }

    def __init__(self, random_seed: int = 42):
        self.rng = np.random.default_rng(random_seed)

    def generate_sector_defaults(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Generate sector-specific default rates based on macro path"""

        # Baseline macro values
        baseline_gdp = 0.02
        baseline_unemp = 0.05
        baseline_spread = 400

        data = {'reporting_month': macro_df['reporting_month'].values}

        for sector in self.BASE_DEFAULT_RATES:
            base_rate = self.BASE_DEFAULT_RATES[sector]
            sens = self.MACRO_SENSITIVITY[sector]

            rates = []
            for idx, row in macro_df.iterrows():
                # Compute macro deviations
                gdp_dev = (row['gdp_growth_yoy'] - baseline_gdp) / 0.01
                unemp_dev = (row['unemployment_rate'] - baseline_unemp) / 0.01
                spread_dev = (row['credit_spread_hy'] - baseline_spread) / 100

                # Apply sensitivities (multiplicative adjustment)
                adjustment = 1.0
                adjustment *= (1 + sens['gdp'] * gdp_dev * 0.1)
                adjustment *= (1 + sens['unemployment'] * unemp_dev * 0.1)
                adjustment *= (1 + sens['spread'] * spread_dev * 0.1)

                # Add noise
                noise = self.rng.normal(0, 0.1)
                adjustment *= (1 + noise)

                # Compute adjusted rate (monthly)
                monthly_rate = (base_rate * adjustment) / 12
                monthly_rate = max(0.0001, min(0.05, monthly_rate))
                rates.append(monthly_rate)

            data[f'default_rate_{sector}'] = rates

        return pd.DataFrame(data)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate and save macro scenarios"""

    output_dir = Path(__file__).parent.parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("MACRO SCENARIO GENERATOR")
    print("=" * 60)

    generator = MacroScenarioGenerator(
        n_months=60,
        start_date='2020-01-01',
        random_seed=42
    )

    # Generate all scenarios
    scenarios = ['baseline', 'adverse', 'severely_adverse', 'stagflation']

    for scenario_name in scenarios:
        print(f"\nGenerating {scenario_name} scenario...")

        df = generator.generate_scenario(scenario_name)

        # Add sector default rates
        sector_gen = SectorDefaultGenerator()
        sector_df = sector_gen.generate_sector_defaults(df)

        # Merge
        df = df.merge(sector_df, on='reporting_month')

        # Save
        output_file = output_dir / f'macro_{scenario_name}.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved to {output_file}")

        # Print statistics
        stats = generator.compute_statistics(df)
        print(f"\nStatistics for {scenario_name}:")
        print(stats[['variable', 'mean', 'min', 'max', 'deviation_from_baseline']].to_string(index=False))

    # Generate Monte Carlo paths
    print("\n" + "=" * 60)
    print("GENERATING MONTE CARLO PATHS")
    print("=" * 60)

    mc_paths = generator.generate_monte_carlo(n_paths=100)

    # Save combined
    mc_combined = pd.concat(mc_paths, ignore_index=True)
    mc_combined.to_parquet(output_dir / 'macro_monte_carlo.parquet', index=False)
    print(f"Saved 100 Monte Carlo paths to {output_dir / 'macro_monte_carlo.parquet'}")

    # Scenario distribution
    print("\nScenario distribution:")
    print(mc_combined.groupby('path_id')['scenario'].first().value_counts())


if __name__ == '__main__':
    main()

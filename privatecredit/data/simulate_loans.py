"""
Synthetic Loan Tape Generator for Private Credit SPV Modeling

Generates realistic loan-level data with:
- Static loan characteristics at origination
- Monthly performance panel with state transitions
- Macro-conditioned transition probabilities
- Multiple asset classes with distinct risk profiles
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AssetClassConfig:
    """Configuration for each asset class"""
    name: str
    balance_range: Tuple[float, float]
    rate_range: Tuple[float, float]
    term_range: Tuple[int, int]
    ltv_range: Tuple[float, float]
    dscr_range: Tuple[float, float]
    score_range: Tuple[int, int]
    annual_default_rate: Tuple[float, float]
    lgd_range: Tuple[float, float]
    annual_prepay_rate: Tuple[float, float]

    # Base monthly transition probabilities
    p_current_to_30dpd: float = 0.015
    p_30dpd_to_60dpd: float = 0.30
    p_30dpd_cure: float = 0.40
    p_60dpd_to_90dpd: float = 0.40
    p_60dpd_cure: float = 0.20
    p_90dpd_to_default: float = 0.50
    p_90dpd_cure: float = 0.10
    p_prepay: float = 0.008


ASSET_CONFIGS = {
    'corporate': AssetClassConfig(
        name='corporate',
        balance_range=(100_000, 5_000_000),
        rate_range=(0.04, 0.12),
        term_range=(12, 84),
        ltv_range=(0.50, 0.80),
        dscr_range=(1.1, 2.5),
        score_range=(50, 100),
        annual_default_rate=(0.01, 0.05),
        lgd_range=(0.30, 0.60),
        annual_prepay_rate=(0.05, 0.15),
        p_current_to_30dpd=0.015,
        p_30dpd_to_60dpd=0.30,
        p_30dpd_cure=0.40,
        p_60dpd_to_90dpd=0.40,
        p_60dpd_cure=0.20,
        p_90dpd_to_default=0.50,
        p_90dpd_cure=0.10,
        p_prepay=0.008
    ),
    'consumer': AssetClassConfig(
        name='consumer',
        balance_range=(1_000, 100_000),
        rate_range=(0.05, 0.20),
        term_range=(6, 84),
        ltv_range=(0.0, 0.0),  # Typically unsecured
        dscr_range=(0.0, 0.0),  # Use DTI instead
        score_range=(300, 850),
        annual_default_rate=(0.02, 0.08),
        lgd_range=(0.60, 0.90),
        annual_prepay_rate=(0.10, 0.30),
        p_current_to_30dpd=0.025,
        p_30dpd_to_60dpd=0.35,
        p_30dpd_cure=0.35,
        p_60dpd_to_90dpd=0.45,
        p_60dpd_cure=0.15,
        p_90dpd_to_default=0.55,
        p_90dpd_cure=0.08,
        p_prepay=0.015
    ),
    'realestate': AssetClassConfig(
        name='realestate',
        balance_range=(50_000, 10_000_000),
        rate_range=(0.02, 0.08),
        term_range=(60, 360),
        ltv_range=(0.50, 0.90),
        dscr_range=(1.0, 1.8),
        score_range=(60, 100),
        annual_default_rate=(0.005, 0.03),
        lgd_range=(0.15, 0.40),
        annual_prepay_rate=(0.03, 0.10),
        p_current_to_30dpd=0.008,
        p_30dpd_to_60dpd=0.25,
        p_30dpd_cure=0.45,
        p_60dpd_to_90dpd=0.35,
        p_60dpd_cure=0.25,
        p_90dpd_to_default=0.45,
        p_90dpd_cure=0.12,
        p_prepay=0.005
    ),
    'receivables': AssetClassConfig(
        name='receivables',
        balance_range=(1_000, 5_000_000),
        rate_range=(0.03, 0.10),
        term_range=(1, 6),
        ltv_range=(0.0, 0.0),
        dscr_range=(0.0, 0.0),
        score_range=(60, 100),
        annual_default_rate=(0.005, 0.02),
        lgd_range=(0.20, 0.50),
        annual_prepay_rate=(0.0, 0.0),  # Receivables typically don't prepay
        p_current_to_30dpd=0.010,
        p_30dpd_to_60dpd=0.20,
        p_30dpd_cure=0.50,
        p_60dpd_to_90dpd=0.30,
        p_60dpd_cure=0.30,
        p_90dpd_to_default=0.40,
        p_90dpd_cure=0.15,
        p_prepay=0.0
    )
}

# Industry sectors (NACE codes simplified)
INDUSTRY_SECTORS = [
    'A_agriculture', 'C_manufacturing', 'F_construction',
    'G_wholesale_retail', 'H_transport', 'I_hospitality',
    'J_information', 'K_finance', 'L_real_estate',
    'M_professional', 'N_admin', 'Q_health'
]

# Geography distribution
COUNTRIES = {
    'DE': 0.25, 'FR': 0.20, 'IT': 0.15, 'ES': 0.10,
    'NL': 0.08, 'BE': 0.05, 'AT': 0.05, 'CH': 0.05,
    'UK': 0.04, 'Other': 0.03
}

# =============================================================================
# LOAN GENERATOR
# =============================================================================

class LoanTapeGenerator:
    """
    Generates synthetic loan tape with realistic characteristics
    """

    def __init__(
        self,
        n_loans: int = 10000,
        n_months: int = 60,
        n_vintages: int = 24,
        start_date: str = '2020-01-01',
        asset_class_weights: Dict[str, float] = None,
        random_seed: int = 42
    ):
        self.n_loans = n_loans
        self.n_months = n_months
        self.n_vintages = n_vintages
        self.start_date = pd.Timestamp(start_date)
        self.asset_class_weights = asset_class_weights or {
            'corporate': 0.40, 'consumer': 0.25,
            'realestate': 0.25, 'receivables': 0.10
        }
        self.rng = np.random.default_rng(random_seed)

        # State encoding
        self.states = ['performing', '30dpd', '60dpd', '90dpd',
                       'default', 'prepaid', 'matured']
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}

    def generate_static_features(self) -> pd.DataFrame:
        """Generate loan-level static features at origination"""

        loans = []
        loan_id = 0

        # Assign loans to asset classes
        asset_classes = self.rng.choice(
            list(self.asset_class_weights.keys()),
            size=self.n_loans,
            p=list(self.asset_class_weights.values())
        )

        # Assign loans to vintages (monthly cohorts)
        vintages = self.rng.integers(0, self.n_vintages, size=self.n_loans)

        for i in range(self.n_loans):
            asset_class = asset_classes[i]
            config = ASSET_CONFIGS[asset_class]
            vintage = vintages[i]

            # Origination date from vintage
            orig_date = self.start_date + relativedelta(months=int(vintage))

            # Loan terms
            term_months = self.rng.integers(config.term_range[0], config.term_range[1] + 1)
            maturity_date = orig_date + relativedelta(months=int(term_months))

            # Balance (log-normal distribution)
            log_balance = np.log(config.balance_range[0]) + \
                self.rng.random() * (np.log(config.balance_range[1]) - np.log(config.balance_range[0]))
            original_balance = np.exp(log_balance)

            # Interest rate (correlated with credit quality)
            base_rate = config.rate_range[0] + \
                self.rng.random() * (config.rate_range[1] - config.rate_range[0])

            # Credit scores
            internal_score = self.rng.uniform(config.score_range[0], config.score_range[1])
            if asset_class == 'consumer':
                external_score = int(self.rng.uniform(300, 850))
            else:
                external_score = int(internal_score * 5 + 350 + self.rng.normal(0, 30))
                external_score = min(850, max(300, external_score))

            # Adjust rate for credit quality (higher score = lower rate)
            score_adjustment = (internal_score - 75) / 100 * 0.02  # +/- 2% for score
            interest_rate = max(config.rate_range[0],
                              min(config.rate_range[1], base_rate - score_adjustment))

            # Rate type
            rate_type = self.rng.choice(['fixed', 'float'], p=[0.6, 0.4])
            spread_bps = int((interest_rate - 0.03) * 10000) if rate_type == 'float' else 0

            # LTV and DSCR
            if config.ltv_range[1] > 0:
                ltv = self.rng.uniform(config.ltv_range[0], config.ltv_range[1])
                collateral_value = original_balance / ltv
                collateral_type = self.rng.choice(['real_estate', 'equipment', 'inventory'])
            else:
                ltv = 0.0
                collateral_value = 0.0
                collateral_type = 'unsecured'

            if config.dscr_range[1] > 0:
                dscr = self.rng.uniform(config.dscr_range[0], config.dscr_range[1])
            else:
                # For consumer: DTI (inverse concept)
                dscr = self.rng.uniform(0.2, 0.5)  # DTI

            # Industry and geography
            industry = self.rng.choice(INDUSTRY_SECTORS) if asset_class != 'consumer' else 'consumer'
            country = self.rng.choice(list(COUNTRIES.keys()), p=list(COUNTRIES.values()))

            # Borrower type
            if asset_class == 'corporate':
                if original_balance < 500_000:
                    borrower_type = 'SME'
                elif original_balance < 5_000_000:
                    borrower_type = 'midmarket'
                else:
                    borrower_type = 'large_corporate'
            elif asset_class == 'consumer':
                borrower_type = 'consumer'
            else:
                borrower_type = 'corporate'

            # Amortization type
            if asset_class == 'receivables':
                amort_type = 'bullet'
            elif asset_class == 'realestate':
                amort_type = self.rng.choice(['amortizing', 'balloon'], p=[0.7, 0.3])
            else:
                amort_type = self.rng.choice(['amortizing', 'bullet'], p=[0.8, 0.2])

            # Payment frequency
            if asset_class == 'receivables':
                payment_freq = 'bullet'  # Single payment at maturity
            else:
                payment_freq = 'monthly'

            loan = {
                'loan_id': f'L{loan_id:06d}',
                'origination_date': orig_date,
                'maturity_date': maturity_date,
                'original_balance': round(original_balance, 2),
                'interest_rate': round(interest_rate, 6),
                'rate_type': rate_type,
                'spread_bps': spread_bps,
                'payment_frequency': payment_freq,
                'amortization_type': amort_type,
                'asset_class': asset_class,
                'term_months': term_months,
                'ltv_origination': round(ltv, 4),
                'dti_dscr_origination': round(dscr, 4),
                'internal_score_origination': round(internal_score, 2),
                'external_score_origination': external_score,
                'industry_sector': industry,
                'geography_country': country,
                'borrower_type': borrower_type,
                'collateral_type': collateral_type,
                'collateral_value_origination': round(collateral_value, 2),
                'originator_id': f'ORIG{self.rng.integers(1, 11):02d}',
                'vintage_cohort': orig_date.strftime('%Y-%m')
            }

            loans.append(loan)
            loan_id += 1

        return pd.DataFrame(loans)

    def calculate_scheduled_payment(
        self,
        balance: float,
        rate: float,
        remaining_term: int,
        amort_type: str
    ) -> float:
        """Calculate scheduled monthly payment"""

        if remaining_term <= 0:
            return 0.0

        monthly_rate = rate / 12

        if amort_type == 'bullet':
            # Interest only, principal at maturity
            return balance * monthly_rate
        elif amort_type == 'balloon':
            # Amortizing with 30% balloon
            balloon = balance * 0.30
            amort_balance = balance - balloon
            if monthly_rate > 0:
                pmt = amort_balance * (monthly_rate * (1 + monthly_rate)**remaining_term) / \
                      ((1 + monthly_rate)**remaining_term - 1)
            else:
                pmt = amort_balance / remaining_term
            return pmt + balance * monthly_rate
        else:  # amortizing
            if monthly_rate > 0:
                pmt = balance * (monthly_rate * (1 + monthly_rate)**remaining_term) / \
                      ((1 + monthly_rate)**remaining_term - 1)
            else:
                pmt = balance / remaining_term
            return pmt

    def get_transition_probabilities(
        self,
        config: AssetClassConfig,
        macro_factor: float = 0.0,
        loan_quality: float = 0.0
    ) -> Dict[Tuple[str, str], float]:
        """
        Get state transition probabilities adjusted for macro and loan quality.

        Args:
            config: Asset class configuration
            macro_factor: Macro stress factor (-1 to +1, positive = worse economy)
            loan_quality: Loan quality factor (-1 to +1, positive = better quality)
        """

        # Base probabilities
        probs = {
            ('performing', '30dpd'): config.p_current_to_30dpd,
            ('performing', 'prepaid'): config.p_prepay,
            ('30dpd', '60dpd'): config.p_30dpd_to_60dpd,
            ('30dpd', 'performing'): config.p_30dpd_cure,
            ('60dpd', '90dpd'): config.p_60dpd_to_90dpd,
            ('60dpd', 'performing'): config.p_60dpd_cure,
            ('90dpd', 'default'): config.p_90dpd_to_default,
            ('90dpd', 'performing'): config.p_90dpd_cure,
        }

        # Macro adjustment (worse economy = higher default, lower cure)
        macro_mult_default = 1 + 0.5 * macro_factor
        macro_mult_cure = 1 - 0.3 * macro_factor
        macro_mult_prepay = 1 - 0.2 * macro_factor

        # Quality adjustment (better quality = lower default, higher cure)
        quality_mult_default = 1 - 0.3 * loan_quality
        quality_mult_cure = 1 + 0.2 * loan_quality

        adjusted = {}
        for (from_state, to_state), base_prob in probs.items():
            if to_state in ['30dpd', '60dpd', '90dpd', 'default']:
                adj_prob = base_prob * macro_mult_default * quality_mult_default
            elif to_state == 'performing':  # Cure
                adj_prob = base_prob * macro_mult_cure * quality_mult_cure
            elif to_state == 'prepaid':
                adj_prob = base_prob * macro_mult_prepay
            else:
                adj_prob = base_prob

            adjusted[(from_state, to_state)] = min(0.99, max(0.001, adj_prob))

        return adjusted

    def simulate_loan_path(
        self,
        loan: pd.Series,
        macro_path: pd.DataFrame,
        end_date: pd.Timestamp
    ) -> List[Dict]:
        """Simulate monthly path for a single loan"""

        config = ASSET_CONFIGS[loan['asset_class']]
        records = []

        # Initial state
        current_state = 'performing'
        current_balance = loan['original_balance']
        months_in_state = 0
        cumulative_principal = 0.0
        cumulative_interest = 0.0

        # Loan quality based on credit score (normalized)
        if loan['asset_class'] == 'consumer':
            loan_quality = (loan['external_score_origination'] - 575) / 275  # 300-850 → -1 to 1
        else:
            loan_quality = (loan['internal_score_origination'] - 75) / 25  # 50-100 → -1 to 1
        loan_quality = max(-1, min(1, loan_quality))

        # Simulate each month
        current_date = loan['origination_date'] + relativedelta(months=1)

        while current_date <= end_date and current_state not in ['prepaid', 'matured', 'default']:

            # Check if loan has matured
            if current_date >= loan['maturity_date']:
                if current_state == 'performing':
                    current_state = 'matured'
                    current_balance = 0.0
                # If delinquent at maturity, stays delinquent/defaults

            # Get macro factor for this month
            month_str = current_date.strftime('%Y-%m')
            if month_str in macro_path.index:
                macro_row = macro_path.loc[month_str]
                # Composite macro factor based on unemployment and credit spreads
                macro_factor = (macro_row.get('unemployment_rate', 0.05) - 0.05) / 0.10 + \
                              (macro_row.get('credit_spread_hy', 400) - 400) / 800
                macro_factor = max(-1, min(1, macro_factor))
            else:
                macro_factor = 0.0

            # Get transition probabilities
            trans_probs = self.get_transition_probabilities(config, macro_factor, loan_quality)

            # Calculate remaining term
            remaining_term = (loan['maturity_date'].year - current_date.year) * 12 + \
                           (loan['maturity_date'].month - current_date.month)
            remaining_term = max(0, remaining_term)

            # Scheduled payment
            scheduled_payment = self.calculate_scheduled_payment(
                current_balance, loan['interest_rate'],
                remaining_term, loan['amortization_type']
            )

            # State transitions
            new_state = current_state
            actual_payment = 0.0
            prepayment = 0.0
            loss_amount = 0.0

            if current_state == 'performing':
                rand = self.rng.random()
                p_30 = trans_probs.get(('performing', '30dpd'), 0.0)
                p_prepay = trans_probs.get(('performing', 'prepaid'), 0.0)

                if rand < p_30:
                    new_state = '30dpd'
                    actual_payment = scheduled_payment * self.rng.uniform(0.0, 0.5)
                elif rand < p_30 + p_prepay:
                    new_state = 'prepaid'
                    actual_payment = current_balance + scheduled_payment * loan['interest_rate'] / 12
                    prepayment = current_balance
                else:
                    new_state = 'performing'
                    actual_payment = scheduled_payment
                    # Occasional extra prepayment
                    if self.rng.random() < 0.02:
                        extra = current_balance * self.rng.uniform(0.01, 0.10)
                        prepayment = extra
                        actual_payment += extra

            elif current_state == '30dpd':
                rand = self.rng.random()
                p_60 = trans_probs.get(('30dpd', '60dpd'), 0.0)
                p_cure = trans_probs.get(('30dpd', 'performing'), 0.0)

                if rand < p_cure:
                    new_state = 'performing'
                    actual_payment = scheduled_payment * 2  # Catch up
                elif rand < p_cure + p_60:
                    new_state = '60dpd'
                    actual_payment = scheduled_payment * self.rng.uniform(0.0, 0.3)
                else:
                    new_state = '30dpd'
                    actual_payment = scheduled_payment * self.rng.uniform(0.3, 0.7)

            elif current_state == '60dpd':
                rand = self.rng.random()
                p_90 = trans_probs.get(('60dpd', '90dpd'), 0.0)
                p_cure = trans_probs.get(('60dpd', 'performing'), 0.0)

                if rand < p_cure:
                    new_state = 'performing'
                    actual_payment = scheduled_payment * 3  # Catch up
                elif rand < p_cure + p_90:
                    new_state = '90dpd'
                    actual_payment = 0.0
                else:
                    new_state = '60dpd'
                    actual_payment = scheduled_payment * self.rng.uniform(0.0, 0.3)

            elif current_state == '90dpd':
                rand = self.rng.random()
                p_default = trans_probs.get(('90dpd', 'default'), 0.0)
                p_cure = trans_probs.get(('90dpd', 'performing'), 0.0)

                if rand < p_cure:
                    new_state = 'performing'
                    actual_payment = scheduled_payment * 4  # Catch up
                elif rand < p_cure + p_default:
                    new_state = 'default'
                    # LGD calculation
                    lgd = self.rng.uniform(config.lgd_range[0], config.lgd_range[1])
                    loss_amount = current_balance * lgd
                    actual_payment = 0.0
                else:
                    new_state = '90dpd'
                    actual_payment = 0.0

            # Calculate principal and interest split
            interest_portion = current_balance * loan['interest_rate'] / 12
            principal_portion = max(0, actual_payment - interest_portion)

            if new_state == 'prepaid':
                principal_portion = current_balance

            # Update balance
            if new_state not in ['default']:
                current_balance = max(0, current_balance - principal_portion - prepayment)

            # Days past due
            dpd_map = {'performing': 0, '30dpd': 30, '60dpd': 60, '90dpd': 90,
                      'default': 180, 'prepaid': 0, 'matured': 0}
            days_past_due = dpd_map.get(new_state, 0)

            # Update months in state
            if new_state == current_state:
                months_in_state += 1
            else:
                months_in_state = 1

            # Cure flag
            cure_flag = (current_state in ['30dpd', '60dpd', '90dpd'] and
                        new_state == 'performing')

            # Record
            record = {
                'loan_id': loan['loan_id'],
                'reporting_month': current_date.strftime('%Y-%m'),
                'current_balance': round(current_balance, 2),
                'scheduled_payment': round(scheduled_payment, 2),
                'actual_payment': round(actual_payment, 2),
                'principal_paid': round(principal_portion, 2),
                'interest_paid': round(min(interest_portion, actual_payment), 2),
                'prepayment_amount': round(prepayment, 2),
                'days_past_due': days_past_due,
                'delinquency_bucket': new_state if new_state in ['30dpd', '60dpd', '90dpd'] else 'current',
                'months_in_bucket': months_in_state,
                'loan_state': new_state,
                'cure_flag': 1 if cure_flag else 0,
                'modification_flag': 0,
                'forbearance_flag': 0,
                'loss_amount': round(loss_amount, 2) if new_state == 'default' else 0.0
            }

            records.append(record)
            current_state = new_state
            current_date = current_date + relativedelta(months=1)

            cumulative_principal += principal_portion
            cumulative_interest += min(interest_portion, actual_payment)

        return records

    def generate_panel(
        self,
        loans_df: pd.DataFrame,
        macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate monthly performance panel for all loans"""

        # Set macro index
        if 'reporting_month' in macro_df.columns:
            macro_df = macro_df.set_index('reporting_month')

        # End date for simulation
        end_date = self.start_date + relativedelta(months=self.n_months)

        all_records = []
        for idx, loan in loans_df.iterrows():
            if idx % 1000 == 0:
                print(f"Simulating loan {idx}/{len(loans_df)}")

            records = self.simulate_loan_path(loan, macro_df, end_date)
            all_records.extend(records)

        return pd.DataFrame(all_records)

    def generate(self, macro_df: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate complete loan tape (static + panel)

        Returns:
            Tuple of (loans_df, panel_df)
        """

        print(f"Generating {self.n_loans} loans across {self.n_vintages} vintages...")
        loans_df = self.generate_static_features()

        # Generate or use provided macro data
        if macro_df is None:
            from simulate_macro import MacroScenarioGenerator
            macro_gen = MacroScenarioGenerator(
                n_months=self.n_months,
                start_date=self.start_date.strftime('%Y-%m-%d')
            )
            macro_df = macro_gen.generate_baseline()

        print(f"Simulating {self.n_months} months of performance...")
        panel_df = self.generate_panel(loans_df, macro_df)

        print(f"Generated {len(panel_df)} loan-month observations")

        return loans_df, panel_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate and save synthetic loan tape"""

    output_dir = Path(__file__).parent.parent.parent / 'data'
    output_dir.mkdir(exist_ok=True)

    # Generate macro first
    print("=" * 60)
    print("STEP 1: Generating Macro Scenarios")
    print("=" * 60)

    try:
        from simulate_macro import MacroScenarioGenerator
        macro_gen = MacroScenarioGenerator(
            n_months=60,
            start_date='2020-01-01'
        )
        macro_df = macro_gen.generate_baseline()
        macro_df.to_csv(output_dir / 'macro_baseline.csv', index=False)
        print(f"Saved macro data to {output_dir / 'macro_baseline.csv'}")
    except ImportError:
        print("Warning: simulate_macro.py not found, using default macro")
        # Create simple macro DataFrame
        dates = pd.date_range('2020-01-01', periods=60, freq='MS')
        macro_df = pd.DataFrame({
            'reporting_month': [d.strftime('%Y-%m') for d in dates],
            'gdp_growth_yoy': np.random.normal(0.02, 0.01, 60),
            'unemployment_rate': np.random.normal(0.05, 0.01, 60),
            'inflation_rate': np.random.normal(0.02, 0.005, 60),
            'policy_rate': np.random.normal(0.03, 0.01, 60),
            'credit_spread_hy': np.random.normal(400, 100, 60)
        })

    # Generate loans
    print("\n" + "=" * 60)
    print("STEP 2: Generating Loan Tape")
    print("=" * 60)

    generator = LoanTapeGenerator(
        n_loans=10000,
        n_months=60,
        n_vintages=24,
        start_date='2020-01-01',
        random_seed=42
    )

    loans_df, panel_df = generator.generate(macro_df)

    # Save
    loans_df.to_csv(output_dir / 'loans_static.csv', index=False)
    panel_df.to_parquet(output_dir / 'loan_monthly.parquet', index=False)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Loans generated: {len(loans_df)}")
    print(f"Loan-month observations: {len(panel_df)}")
    print(f"\nAsset class distribution:")
    print(loans_df['asset_class'].value_counts())
    print(f"\nFinal state distribution:")
    final_states = panel_df.groupby('loan_id')['loan_state'].last()
    print(final_states.value_counts())
    print(f"\nDefault rate: {(final_states == 'default').mean():.2%}")
    print(f"Prepay rate: {(final_states == 'prepaid').mean():.2%}")
    print(f"\nFiles saved to: {output_dir}")


if __name__ == '__main__':
    main()

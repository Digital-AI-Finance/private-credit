# Private Credit SPV - Data Schema Specification

## Overview

This document defines the data architecture for the Private Credit SPV deep generative modeling framework. The schema supports four asset classes: corporate loans, consumer loans, real estate loans, and trade receivables.

---

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         LOANS (Static)                          │
│  PK: loan_id                                                    │
│  ────────────────────────────────────────────────────────────── │
│  origination_date, maturity_date, original_balance,            │
│  interest_rate, rate_type, spread_bps, payment_frequency,      │
│  amortization_type, asset_class, ltv_origination,              │
│  dti_dscr_origination, internal_score, external_score,         │
│  industry_sector, geography, borrower_type, collateral,        │
│  originator_id, vintage_cohort, term_months                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 1:N
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LOAN_MONTHLY (Time-Varying)                  │
│  PK: (loan_id, reporting_month)                                 │
│  FK: loan_id → LOANS                                            │
│  ────────────────────────────────────────────────────────────── │
│  current_balance, scheduled_payment, actual_payment,            │
│  principal_paid, interest_paid, prepayment_amount,              │
│  days_past_due, delinquency_bucket, months_in_bucket,          │
│  loan_state, cure_flag, modification_flag, forbearance_flag,   │
│  collateral_value_current, loss_amount, recovery_amount        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ N:1
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MACRO_MONTHLY (External)                     │
│  PK: reporting_month                                            │
│  ────────────────────────────────────────────────────────────── │
│  gdp_growth_yoy, unemployment_rate, inflation_rate,            │
│  policy_rate, yield_10y, credit_spread_ig, credit_spread_hy,   │
│  property_price_index, sector_default_rates                    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      SPV_STRUCTURE (Static)                     │
│  PK: (spv_id, tranche_id)                                       │
│  ────────────────────────────────────────────────────────────── │
│  tranche_name, tranche_size, attachment_point,                 │
│  detachment_point, coupon_rate, priority_order,                │
│  oc_trigger, ic_trigger, waterfall_rules                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Loan State Transitions

```
                    ┌──────────────┐
                    │  PERFORMING  │
                    │   (Current)  │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   ┌─────────┐       ┌─────────┐        ┌─────────┐
   │ PREPAID │       │ 30 DPD  │        │ MATURED │
   └─────────┘       └────┬────┘        └─────────┘
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        ┌─────────┐             ┌─────────┐
        │  CURE   │             │ 60 DPD  │
        │(→Perf)  │             └────┬────┘
        └─────────┘                  │
                          ┌──────────┴──────────┐
                          ▼                     ▼
                    ┌─────────┐           ┌─────────┐
                    │  CURE   │           │ 90 DPD  │
                    │(→Perf)  │           └────┬────┘
                    └─────────┘                │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              ┌─────────┐           ┌─────────┐
                              │  CURE   │           │ DEFAULT │
                              │(→Perf)  │           └────┬────┘
                              └─────────┘                │
                                              ┌──────────┴──────────┐
                                              ▼                     ▼
                                        ┌───────────┐         ┌─────────┐
                                        │ RECOVERY  │         │  LOSS   │
                                        │ (Workout) │         │(Write-off)
                                        └───────────┘         └─────────┘
```

---

## Asset Class Specifications

### Corporate Loans (SME / Mid-Market)

| Parameter | Typical Range |
|-----------|---------------|
| Original Balance | EUR 100K - 50M |
| Interest Rate | 4% - 12% |
| Term | 12 - 84 months |
| LTV (if secured) | 50% - 80% |
| DSCR | 1.1 - 2.5 |
| Annual Default Rate | 1% - 5% |
| LGD | 30% - 60% |
| Prepayment Rate | 5% - 15% annual |

### Consumer Loans

| Parameter | Typical Range |
|-----------|---------------|
| Original Balance | EUR 1K - 100K |
| Interest Rate | 5% - 20% |
| Term | 6 - 84 months |
| DTI | 0.2 - 0.5 |
| External Score | 300 - 850 |
| Annual Default Rate | 2% - 8% |
| LGD | 60% - 90% |
| Prepayment Rate | 10% - 30% annual |

### Real Estate Loans

| Parameter | Typical Range |
|-----------|---------------|
| Original Balance | EUR 50K - 10M |
| Interest Rate | 2% - 8% |
| Term | 60 - 360 months |
| LTV | 50% - 90% |
| DSCR | 1.0 - 1.8 |
| Annual Default Rate | 0.5% - 3% |
| LGD | 15% - 40% |
| Prepayment Rate | 3% - 10% annual |

### Trade Receivables

| Parameter | Typical Range |
|-----------|---------------|
| Original Balance | EUR 1K - 5M |
| Interest Rate | 3% - 10% |
| Term | 1 - 6 months |
| Dilution Rate | 2% - 10% |
| Annual Default Rate | 0.5% - 2% |
| LGD | 20% - 50% |

---

## Transition Probability Parameters

### Base Case Monthly Transition Rates

| From State | To State | Corporate | Consumer | RealEstate | Receivables |
|------------|----------|-----------|----------|------------|-------------|
| Current | 30 DPD | 0.015 | 0.025 | 0.008 | 0.010 |
| 30 DPD | 60 DPD | 0.30 | 0.35 | 0.25 | 0.20 |
| 30 DPD | Cure | 0.40 | 0.35 | 0.45 | 0.50 |
| 60 DPD | 90 DPD | 0.40 | 0.45 | 0.35 | 0.30 |
| 60 DPD | Cure | 0.20 | 0.15 | 0.25 | 0.30 |
| 90 DPD | Default | 0.50 | 0.55 | 0.45 | 0.40 |
| 90 DPD | Cure | 0.10 | 0.08 | 0.12 | 0.15 |
| Current | Prepay | 0.008 | 0.015 | 0.005 | 0.02 |

### Macro Sensitivity (Elasticities)

| Transition | GDP Growth | Unemployment | Credit Spread |
|------------|------------|--------------|---------------|
| Current → Delinquent | -2.0 | +1.5 | +0.8 |
| Delinquent → Default | -1.5 | +1.0 | +0.5 |
| Delinquent → Cure | +1.0 | -0.8 | -0.3 |
| Current → Prepay | +0.5 | -0.3 | -0.2 |

*Elasticity interpretation: 1% increase in unemployment → 1.5% increase in delinquency transition rate*

---

## SPV Waterfall Structure

### Typical CLO Waterfall

```
Collections Pool
      │
      ▼
┌─────────────────────┐
│ 1. Senior Fees      │  Trustee, servicer, admin fees
│    (0.5-1% p.a.)    │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 2. Class A Interest │  Senior tranche coupon
│    (SOFR + 150 bps) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 3. Class A OC Test  │  If OC < 120%, redirect to A principal
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 4. Class B Interest │  Mezzanine coupon
│    (SOFR + 300 bps) │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 5. Class B OC Test  │  If OC < 110%, redirect to B principal
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 6. Junior Fees      │  Subordinated management fees
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ 7. Equity Residual  │  Remaining to equity tranche
└─────────────────────┘
```

### Tranche Structure Example

| Tranche | Size | Attachment | Detachment | Coupon | CE |
|---------|------|------------|------------|--------|-----|
| Senior A | 70% | 0% | 70% | SOFR+150 | 30% |
| Mezz B | 15% | 70% | 85% | SOFR+300 | 15% |
| Junior C | 10% | 85% | 95% | SOFR+500 | 5% |
| Equity | 5% | 95% | 100% | Residual | 0% |

---

## Data Quality Rules

### Mandatory Fields

All loans must have:
- loan_id (unique)
- origination_date <= reporting_month
- maturity_date > origination_date
- original_balance > 0
- interest_rate in [0.01, 0.30]
- asset_class in valid set
- geography_country (ISO-2)

### Consistency Checks

1. current_balance <= original_balance (unless PIK)
2. days_past_due consistent with delinquency_bucket
3. loan_state transitions follow allowed paths
4. sum(principal_paid) + current_balance <= original_balance
5. If defaulted: loss_amount + recovery_amount <= current_balance at default

### Missing Data Handling

| Field | If Missing |
|-------|-----------|
| ltv_origination | Impute from asset_class median |
| external_score | Impute from internal_score mapping |
| collateral_value | Set to 0 (unsecured) |
| spread_bps | Impute from rate minus benchmark |

---

## Derived Features (Feature Engineering)

### Loan-Level Derived

```python
# Age features
loan_age_months = reporting_month - origination_date
remaining_term = maturity_date - reporting_month
seasoning_ratio = loan_age_months / term_months

# Payment behavior
payment_ratio = actual_payment / scheduled_payment
cumulative_payment_ratio = sum(actual_payment) / sum(scheduled_payment)
months_since_last_full_payment = ...

# Balance trajectory
balance_reduction_rate = 1 - (current_balance / original_balance)
expected_balance = amortization_schedule(t)
balance_gap = current_balance - expected_balance

# Risk metrics
current_ltv = current_balance / collateral_value_current
credit_score_delta = external_score_current - external_score_origination
```

### Cohort-Level Derived

```python
# Vintage performance
vintage_default_rate = defaults / loans_in_vintage
vintage_prepay_rate = prepayments / performing_loans
vintage_delinquency_rate = delinquent / performing_loans

# Roll rates
roll_rate_30_to_60 = loans_moving_30_to_60 / loans_in_30dpd
cure_rate_from_30 = loans_curing_from_30 / loans_in_30dpd
```

### Portfolio-Level Derived

```python
# Concentration metrics
hhi_industry = sum((exposure_i / total_exposure)^2)
hhi_geography = sum((exposure_i / total_exposure)^2)
top10_concentration = sum(top10_exposures) / total_exposure

# Portfolio quality
weighted_avg_score = sum(balance * score) / sum(balance)
weighted_avg_ltv = sum(balance * ltv) / sum(balance)
current_pool_factor = sum(current_balance) / sum(original_balance)
```

---

## File Format Specifications

### Loan Tape (CSV)

```
loan_id,origination_date,maturity_date,original_balance,...
L000001,2020-01-15,2025-01-15,250000.00,...
L000002,2020-02-01,2025-02-01,175000.00,...
```

### Monthly Panel (Parquet recommended)

```
loan_id | reporting_month | current_balance | loan_state | ...
L000001 | 2020-02 | 247500.00 | performing | ...
L000001 | 2020-03 | 245000.00 | performing | ...
L000001 | 2020-04 | 242500.00 | 30dpd | ...
```

### Macro Series (CSV)

```
reporting_month,gdp_growth_yoy,unemployment_rate,inflation_rate,...
2020-01,0.023,0.036,0.018,...
2020-02,0.021,0.037,0.019,...
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01 | Initial specification |

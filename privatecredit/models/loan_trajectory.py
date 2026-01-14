"""
Loan Trajectory Model - Autoregressive Generation of Individual Loan Paths

Generates loan-level trajectories including:
- State sequences (discrete): performing, delinquent, default, prepaid, matured
- Payment sequences (continuous): actual payment amounts
- Default timing: survival analysis component

Architecture:
- Transformer decoder for autoregressive state prediction
- Diffusion head for continuous payment generation
- Hazard rate module for default timing
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass
import math


# =============================================================================
# CONFIGURATION
# =============================================================================

STATES = ['performing', '30dpd', '60dpd', '90dpd', 'default', 'prepaid', 'matured']
N_STATES = len(STATES)
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
ABSORBING_STATES = {4, 5, 6}  # default, prepaid, matured


@dataclass
class LoanTrajectoryConfig:
    """Configuration for Loan Trajectory Model"""
    # Input dimensions
    n_loan_features: int = 20  # Static loan features
    n_macro_vars: int = 9
    n_states: int = N_STATES

    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_decoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_length: int = 120

    # Diffusion for payments
    diffusion_steps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    n_epochs: int = 50


# =============================================================================
# LOAN EMBEDDING
# =============================================================================

class LoanEmbedding(nn.Module):
    """Embed static loan features"""

    def __init__(self, config: LoanTrajectoryConfig):
        super().__init__()

        # Categorical embeddings
        self.asset_class_embed = nn.Embedding(4, config.d_model // 4)
        self.borrower_type_embed = nn.Embedding(4, config.d_model // 4)
        self.rate_type_embed = nn.Embedding(2, config.d_model // 8)
        self.amort_type_embed = nn.Embedding(3, config.d_model // 8)

        # Continuous features projection
        n_continuous = 12  # balance, rate, ltv, dscr, score, term, etc.
        self.continuous_proj = nn.Sequential(
            nn.Linear(n_continuous, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model // 4)
        )

        # Final projection
        self.output_proj = nn.Linear(config.d_model, config.d_model)

    def forward(
        self,
        categorical_features: Dict[str, torch.Tensor],
        continuous_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            categorical_features: Dict with tensors (batch,)
            continuous_features: (batch, n_continuous)

        Returns:
            Loan embedding (batch, d_model)
        """
        ac_emb = self.asset_class_embed(categorical_features['asset_class'])
        bt_emb = self.borrower_type_embed(categorical_features['borrower_type'])
        rt_emb = self.rate_type_embed(categorical_features['rate_type'])
        at_emb = self.amort_type_embed(categorical_features['amort_type'])
        cont_emb = self.continuous_proj(continuous_features)

        combined = torch.cat([ac_emb, bt_emb, rt_emb, at_emb, cont_emb], dim=-1)
        return self.output_proj(combined)


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x of shape (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# =============================================================================
# DIFFUSION MODULE (for continuous payments)
# =============================================================================

class DiffusionHead(nn.Module):
    """Diffusion-based head for continuous payment generation"""

    def __init__(self, config: LoanTrajectoryConfig):
        super().__init__()
        self.config = config

        # Beta schedule
        betas = torch.linspace(config.beta_start, config.beta_end, config.diffusion_steps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

        # Denoising network
        self.denoise_net = nn.Sequential(
            nn.Linear(config.d_model + 2, config.d_model),  # +2 for noisy payment and timestep
            nn.SiLU(),
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Linear(config.d_model, 1)  # Predict noise
        )

        # Timestep embedding
        self.time_embed = nn.Embedding(config.diffusion_steps, 1)

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)

        x_t = sqrt_alpha * x_0 + sqrt_one_minus * noise
        return x_t, noise

    def forward(
        self,
        context: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training: Predict noise given noisy target
        Inference: Generate payment from noise

        Args:
            context: Conditioning from trajectory model (batch, d_model)
            target: True payment (training only)
            t: Diffusion timestep (training only)

        Returns:
            Predicted noise (training) or generated payment (inference)
        """
        if self.training and target is not None:
            # Training: denoise
            x_t, noise = self.q_sample(target, t)
            t_emb = self.time_embed(t)

            model_input = torch.cat([context, x_t, t_emb], dim=-1)
            pred_noise = self.denoise_net(model_input)
            return pred_noise, noise
        else:
            # Inference: DDPM sampling
            return self.sample(context)

    @torch.no_grad()
    def sample(self, context: torch.Tensor) -> torch.Tensor:
        """Generate payment via reverse diffusion"""
        batch_size = context.shape[0]
        device = context.device

        # Start from noise
        x = torch.randn(batch_size, 1, device=device)

        for t in reversed(range(self.config.diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            t_emb = self.time_embed(t_tensor)

            # Predict noise
            model_input = torch.cat([context, x, t_emb], dim=-1)
            pred_noise = self.denoise_net(model_input)

            # Denoise step
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (1 / torch.sqrt(alpha)) * (
                x - beta / torch.sqrt(1 - alpha_cumprod) * pred_noise
            ) + torch.sqrt(beta) * noise

        return x


# =============================================================================
# HAZARD RATE MODULE
# =============================================================================

class HazardRateModule(nn.Module):
    """Survival analysis component for default timing"""

    def __init__(self, config: LoanTrajectoryConfig):
        super().__init__()

        self.hazard_net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, 1),
            nn.Sigmoid()  # Hazard rate in [0, 1]
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Predict instantaneous hazard rate.

        Args:
            context: (batch, d_model)

        Returns:
            Hazard rate (batch, 1)
        """
        return self.hazard_net(context) * 0.1  # Scale to reasonable range

    def survival_probability(
        self,
        context_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute survival probability curve.

        Args:
            context_seq: (batch, seq_len, d_model)

        Returns:
            Survival probability (batch, seq_len)
        """
        batch_size, seq_len, _ = context_seq.shape

        # Compute hazard at each time
        context_flat = context_seq.view(-1, context_seq.shape[-1])
        hazards = self.forward(context_flat).view(batch_size, seq_len)

        # Cumulative hazard
        cum_hazard = torch.cumsum(hazards, dim=1)

        # Survival = exp(-cumulative hazard)
        survival = torch.exp(-cum_hazard)

        return survival


# =============================================================================
# LOAN TRAJECTORY MODEL
# =============================================================================

class LoanTrajectoryModel(nn.Module):
    """
    Autoregressive model for loan-level trajectory generation.

    Components:
    - Transformer decoder for state sequence
    - Diffusion head for payment amounts
    - Hazard rate module for default timing
    """

    def __init__(self, config: LoanTrajectoryConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.loan_embed = LoanEmbedding(config)
        self.state_embed = nn.Embedding(config.n_states, config.d_model)
        self.macro_proj = nn.Linear(config.n_macro_vars, config.d_model)
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_length)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_decoder_layers
        )

        # Output heads
        self.state_head = nn.Linear(config.d_model, config.n_states)
        self.diffusion_head = DiffusionHead(config)
        self.hazard_module = HazardRateModule(config)

        # Causal mask
        self.register_buffer(
            'causal_mask',
            nn.Transformer.generate_square_subsequent_mask(config.max_seq_length)
        )

    def forward(
        self,
        loan_features: Dict[str, torch.Tensor],
        macro_seq: torch.Tensor,
        state_seq: torch.Tensor,
        payment_seq: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            loan_features: Dict with categorical and continuous features
            macro_seq: (batch, seq_len, n_macro_vars)
            state_seq: (batch, seq_len) - true state sequence
            payment_seq: (batch, seq_len) - true payments (optional)

        Returns:
            Dict with state_logits, payment_pred, hazard_rates
        """
        batch_size, seq_len = state_seq.shape
        device = state_seq.device

        # Embed loan
        loan_emb = self.loan_embed(
            loan_features['categorical'],
            loan_features['continuous']
        )  # (batch, d_model)

        # Embed macro sequence
        macro_emb = self.macro_proj(macro_seq)  # (batch, seq_len, d_model)

        # Embed state sequence (shifted for AR)
        state_emb = self.state_embed(state_seq)  # (batch, seq_len, d_model)

        # Combine: loan + macro + state
        loan_emb_expanded = loan_emb.unsqueeze(1).expand(-1, seq_len, -1)
        decoder_input = state_emb + macro_emb + loan_emb_expanded
        decoder_input = self.pos_encoder(decoder_input)

        # Memory: loan embedding (used as context)
        memory = loan_emb.unsqueeze(1)  # (batch, 1, d_model)

        # Causal mask
        tgt_mask = self.causal_mask[:seq_len, :seq_len].to(device)

        # Decode
        decoder_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=memory,
            tgt_mask=tgt_mask
        )  # (batch, seq_len, d_model)

        # State prediction
        state_logits = self.state_head(decoder_output)

        # Payment prediction (with diffusion)
        results = {'state_logits': state_logits}

        if payment_seq is not None and self.training:
            # Sample random timesteps
            t = torch.randint(
                0, self.config.diffusion_steps,
                (batch_size * seq_len,),
                device=device
            )

            # Flatten for diffusion
            context_flat = decoder_output.view(-1, self.config.d_model)
            payment_flat = payment_seq.view(-1, 1)

            pred_noise, true_noise = self.diffusion_head(context_flat, payment_flat, t)
            results['pred_noise'] = pred_noise
            results['true_noise'] = true_noise

        # Hazard rates
        hazard_rates = self.hazard_module(decoder_output)
        results['hazard_rates'] = hazard_rates

        return results

    @torch.no_grad()
    def generate(
        self,
        loan_features: Dict[str, torch.Tensor],
        macro_seq: torch.Tensor,
        max_length: int = 60
    ) -> Dict[str, torch.Tensor]:
        """
        Generate loan trajectories autoregressively.

        Args:
            loan_features: Static loan features
            macro_seq: (batch, max_length, n_macro_vars)
            max_length: Maximum sequence length

        Returns:
            Dict with generated states and payments
        """
        self.eval()
        batch_size = macro_seq.shape[0]
        device = macro_seq.device

        # Initialize
        states = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        payments = torch.zeros(batch_size, max_length, device=device)

        # Embed loan
        loan_emb = self.loan_embed(
            loan_features['categorical'],
            loan_features['continuous']
        )

        # Start with performing state
        states[:, 0] = STATE_TO_IDX['performing']

        for t in range(1, max_length):
            # Get context up to t
            state_seq_t = states[:, :t]
            macro_seq_t = macro_seq[:, :t]

            # Embed
            state_emb = self.state_embed(state_seq_t)
            macro_emb = self.macro_proj(macro_seq_t)

            loan_emb_exp = loan_emb.unsqueeze(1).expand(-1, t, -1)
            decoder_input = state_emb + macro_emb + loan_emb_exp
            decoder_input = self.pos_encoder(decoder_input)

            memory = loan_emb.unsqueeze(1)
            tgt_mask = self.causal_mask[:t, :t].to(device)

            # Decode
            decoder_output = self.transformer_decoder(
                tgt=decoder_input,
                memory=memory,
                tgt_mask=tgt_mask
            )

            # Get last position
            last_output = decoder_output[:, -1]  # (batch, d_model)

            # Predict next state
            state_logits = self.state_head(last_output)

            # Apply absorbing state constraints
            for i in range(batch_size):
                prev_state = states[i, t-1].item()
                if prev_state in ABSORBING_STATES:
                    # Stay in absorbing state
                    states[i, t] = prev_state
                else:
                    # Sample from valid transitions
                    probs = F.softmax(state_logits[i], dim=-1)
                    next_state = torch.multinomial(probs, 1).item()
                    states[i, t] = next_state

            # Generate payment
            generated_payment = self.diffusion_head.sample(last_output)
            payments[:, t] = generated_payment.squeeze(-1).clamp(min=0)

        return {'states': states, 'payments': payments}


# =============================================================================
# DATASET
# =============================================================================

class LoanTrajectoryDataset(Dataset):
    """Dataset for loan trajectory training"""

    def __init__(
        self,
        loans_df: pd.DataFrame,
        panel_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        max_length: int = 60
    ):
        self.max_length = max_length

        # Process loans
        self.process_data(loans_df, panel_df, macro_df)

    def process_data(
        self,
        loans_df: pd.DataFrame,
        panel_df: pd.DataFrame,
        macro_df: pd.DataFrame
    ):
        """Process and align data"""

        # Extract features for each loan
        self.loan_ids = loans_df['loan_id'].values
        n_loans = len(self.loan_ids)

        # Categorical features
        asset_class_map = {'corporate': 0, 'consumer': 1, 'realestate': 2, 'receivables': 3}
        borrower_map = {'SME': 0, 'midmarket': 1, 'large_corporate': 2, 'consumer': 3}
        rate_map = {'fixed': 0, 'float': 1}
        amort_map = {'amortizing': 0, 'bullet': 1, 'balloon': 2}

        self.categorical = {
            'asset_class': np.array([asset_class_map.get(x, 0) for x in loans_df['asset_class']]),
            'borrower_type': np.array([borrower_map.get(x, 0) for x in loans_df['borrower_type']]),
            'rate_type': np.array([rate_map.get(x, 0) for x in loans_df['rate_type']]),
            'amort_type': np.array([amort_map.get(x, 0) for x in loans_df['amortization_type']])
        }

        # Continuous features (normalized)
        continuous_cols = [
            'original_balance', 'interest_rate', 'term_months',
            'ltv_origination', 'dti_dscr_origination',
            'internal_score_origination', 'external_score_origination',
            'spread_bps', 'collateral_value_origination'
        ]

        continuous_data = []
        for col in continuous_cols:
            if col in loans_df.columns:
                vals = loans_df[col].fillna(0).values
                # Normalize
                vals = (vals - vals.mean()) / (vals.std() + 1e-8)
                continuous_data.append(vals)
            else:
                continuous_data.append(np.zeros(n_loans))

        # Pad to fixed size
        while len(continuous_data) < 12:
            continuous_data.append(np.zeros(n_loans))

        self.continuous = np.column_stack(continuous_data)

        # State and payment sequences
        self.states = np.zeros((n_loans, self.max_length), dtype=np.int64)
        self.payments = np.zeros((n_loans, self.max_length))
        self.seq_lengths = np.zeros(n_loans, dtype=np.int64)

        for i, loan_id in enumerate(self.loan_ids):
            loan_panel = panel_df[panel_df['loan_id'] == loan_id].sort_values('reporting_month')

            seq_len = min(len(loan_panel), self.max_length)
            self.seq_lengths[i] = seq_len

            for t, (_, row) in enumerate(loan_panel.iterrows()):
                if t >= self.max_length:
                    break
                state_str = row.get('loan_state', 'performing')
                self.states[i, t] = STATE_TO_IDX.get(state_str, 0)
                self.payments[i, t] = row.get('actual_payment', 0) / 1000  # Scale

        # Macro data
        numeric_cols = ['gdp_growth_yoy', 'unemployment_rate', 'inflation_rate',
                       'policy_rate', 'yield_10y', 'credit_spread_ig',
                       'credit_spread_hy', 'property_price_index', 'equity_return']

        macro_vals = []
        for col in numeric_cols:
            if col in macro_df.columns:
                vals = macro_df[col].values[:self.max_length]
                vals = (vals - vals.mean()) / (vals.std() + 1e-8)
                macro_vals.append(vals)
            else:
                macro_vals.append(np.zeros(self.max_length))

        self.macro = np.column_stack(macro_vals)

    def __len__(self):
        return len(self.loan_ids)

    def __getitem__(self, idx):
        categorical = {k: torch.tensor(v[idx]) for k, v in self.categorical.items()}
        continuous = torch.FloatTensor(self.continuous[idx])
        states = torch.LongTensor(self.states[idx])
        payments = torch.FloatTensor(self.payments[idx])
        macro = torch.FloatTensor(self.macro)

        return {
            'categorical': categorical,
            'continuous': continuous,
            'states': states,
            'payments': payments,
            'macro': macro
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Demonstrate Loan Trajectory Model"""

    print("=" * 60)
    print("LOAN TRAJECTORY MODEL - AUTOREGRESSIVE GENERATION")
    print("=" * 60)

    config = LoanTrajectoryConfig(
        n_loan_features=20,
        n_macro_vars=9,
        d_model=128,
        n_heads=8,
        n_decoder_layers=4,
        n_epochs=20
    )

    # Create model
    model = LoanTrajectoryModel(config)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Demonstrate generation with random inputs
    print("\nGenerating sample trajectories...")

    batch_size = 4
    seq_length = 60

    # Random features
    loan_features = {
        'categorical': {
            'asset_class': torch.randint(0, 4, (batch_size,)),
            'borrower_type': torch.randint(0, 4, (batch_size,)),
            'rate_type': torch.randint(0, 2, (batch_size,)),
            'amort_type': torch.randint(0, 3, (batch_size,))
        },
        'continuous': torch.randn(batch_size, 12)
    }

    macro_seq = torch.randn(batch_size, seq_length, config.n_macro_vars)

    # Generate
    with torch.no_grad():
        results = model.generate(loan_features, macro_seq, max_length=seq_length)

    print("\nGenerated state sequences:")
    for i in range(batch_size):
        states = results['states'][i].numpy()
        state_names = [STATES[s] for s in states[:10]]
        print(f"  Loan {i}: {state_names}...")

    # Show final states
    print("\nFinal states:")
    for i in range(batch_size):
        final_state = STATES[results['states'][i, -1].item()]
        print(f"  Loan {i}: {final_state}")

    # Save model
    output_dir = Path(__file__).parent.parent.parent / 'models'
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'loan_trajectory.pt')
    print(f"\nModel saved to {output_dir / 'loan_trajectory.pt'}")


if __name__ == '__main__':
    main()

"""
Transition Transformer - Cohort-Level Transition Matrix Prediction

Predicts time-varying transition matrices for loan cohorts based on:
- Macro economic conditions
- Cohort characteristics (vintage, asset class, geography)
- Historical transition patterns

Uses Transformer architecture for temporal dependencies.
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

# Loan states
STATES = ['performing', '30dpd', '60dpd', '90dpd', 'default', 'prepaid', 'matured']
N_STATES = len(STATES)
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}

# Valid transitions (from_state, to_state)
VALID_TRANSITIONS = [
    (0, 0), (0, 1), (0, 5), (0, 6),  # performing -> {performing, 30dpd, prepaid, matured}
    (1, 0), (1, 1), (1, 2), (1, 5),  # 30dpd -> {cure, 30dpd, 60dpd, prepaid}
    (2, 0), (2, 2), (2, 3), (2, 5),  # 60dpd -> {cure, 60dpd, 90dpd, prepaid}
    (3, 0), (3, 3), (3, 4), (3, 5),  # 90dpd -> {cure, 90dpd, default, prepaid}
    (4, 4),  # default -> default (absorbing)
    (5, 5),  # prepaid -> prepaid (absorbing)
    (6, 6),  # matured -> matured (absorbing)
]
N_TRANSITIONS = len(VALID_TRANSITIONS)


@dataclass
class TransitionTransformerConfig:
    """Configuration for Transition Transformer"""
    # Input dimensions
    n_macro_vars: int = 9
    n_cohort_features: int = 10
    seq_length: int = 60

    # Model architecture
    d_model: int = 128
    n_heads: int = 8
    n_encoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # Output
    n_states: int = N_STATES
    n_transitions: int = N_TRANSITIONS

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    n_epochs: int = 100
    warmup_steps: int = 1000


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
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# =============================================================================
# COHORT EMBEDDING
# =============================================================================

class CohortEmbedding(nn.Module):
    """Embed cohort characteristics"""

    def __init__(
        self,
        n_asset_classes: int = 4,
        n_geographies: int = 10,
        n_vintages: int = 60,
        d_model: int = 128
    ):
        super().__init__()

        # Categorical embeddings
        self.asset_class_embed = nn.Embedding(n_asset_classes, d_model // 4)
        self.geography_embed = nn.Embedding(n_geographies, d_model // 4)
        self.vintage_embed = nn.Embedding(n_vintages, d_model // 4)

        # Continuous features projection
        self.continuous_proj = nn.Linear(4, d_model // 4)  # avg_score, avg_ltv, avg_balance, n_loans

        # Final projection
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        asset_class: torch.Tensor,
        geography: torch.Tensor,
        vintage: torch.Tensor,
        continuous_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            asset_class: (batch,)
            geography: (batch,)
            vintage: (batch,)
            continuous_features: (batch, 4)

        Returns:
            Cohort embedding (batch, d_model)
        """
        ac_emb = self.asset_class_embed(asset_class)
        geo_emb = self.geography_embed(geography)
        vin_emb = self.vintage_embed(vintage)
        cont_emb = self.continuous_proj(continuous_features)

        combined = torch.cat([ac_emb, geo_emb, vin_emb, cont_emb], dim=-1)
        return self.output_proj(combined)


# =============================================================================
# TRANSITION TRANSFORMER
# =============================================================================

class TransitionTransformer(nn.Module):
    """
    Transformer model for predicting cohort-level transition matrices.

    Architecture:
    - Input: Macro time series + cohort features
    - Encoder: Transformer encoder for temporal patterns
    - Output: Transition probabilities for each time step
    """

    def __init__(self, config: TransitionTransformerConfig):
        super().__init__()
        self.config = config

        # Input projections
        self.macro_proj = nn.Linear(config.n_macro_vars, config.d_model)
        self.cohort_embed = CohortEmbedding(d_model=config.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            config.d_model,
            max_len=config.seq_length,
            dropout=config.dropout
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            batch_first=False  # (seq, batch, d_model)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers
        )

        # Output heads for each state (predict transitions from that state)
        self.transition_heads = nn.ModuleDict()
        for from_state in range(config.n_states):
            # Get valid transitions from this state
            valid_to_states = [t[1] for t in VALID_TRANSITIONS if t[0] == from_state]
            n_outputs = len(valid_to_states)
            if n_outputs > 0:
                self.transition_heads[str(from_state)] = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model // 2),
                    nn.ReLU(),
                    nn.Linear(config.d_model // 2, n_outputs)
                )

        # Store valid transitions per state for softmax
        self.valid_transitions_per_state = {}
        for from_state in range(config.n_states):
            self.valid_transitions_per_state[from_state] = [
                t[1] for t in VALID_TRANSITIONS if t[0] == from_state
            ]

    def forward(
        self,
        macro_seq: torch.Tensor,
        cohort_features: Dict[str, torch.Tensor]
    ) -> Dict[int, torch.Tensor]:
        """
        Predict transition probabilities.

        Args:
            macro_seq: Macro time series (batch, seq_len, n_macro_vars)
            cohort_features: Dict with 'asset_class', 'geography', 'vintage', 'continuous'

        Returns:
            Dict mapping from_state -> transition probabilities (batch, seq_len, n_to_states)
        """
        batch_size, seq_len, _ = macro_seq.shape

        # Project macro sequence
        macro_emb = self.macro_proj(macro_seq)  # (batch, seq, d_model)

        # Get cohort embedding
        cohort_emb = self.cohort_embed(
            cohort_features['asset_class'],
            cohort_features['geography'],
            cohort_features['vintage'],
            cohort_features['continuous']
        )  # (batch, d_model)

        # Add cohort embedding to each time step
        cohort_emb = cohort_emb.unsqueeze(1).expand(-1, seq_len, -1)
        combined = macro_emb + cohort_emb  # (batch, seq, d_model)

        # Transformer expects (seq, batch, d_model)
        combined = combined.permute(1, 0, 2)
        combined = self.pos_encoder(combined)
        encoded = self.transformer_encoder(combined)
        encoded = encoded.permute(1, 0, 2)  # (batch, seq, d_model)

        # Compute transition probabilities for each state
        transition_probs = {}

        for from_state in range(self.config.n_states):
            if str(from_state) in self.transition_heads:
                logits = self.transition_heads[str(from_state)](encoded)
                probs = F.softmax(logits, dim=-1)
                transition_probs[from_state] = probs

        return transition_probs

    def get_transition_matrix(
        self,
        macro_seq: torch.Tensor,
        cohort_features: Dict[str, torch.Tensor],
        time_idx: int
    ) -> torch.Tensor:
        """
        Get full transition matrix for a specific time step.

        Args:
            macro_seq: Macro sequence
            cohort_features: Cohort features
            time_idx: Time index

        Returns:
            Transition matrix (batch, n_states, n_states)
        """
        probs = self.forward(macro_seq, cohort_features)
        batch_size = macro_seq.shape[0]

        # Initialize matrix
        matrix = torch.zeros(
            batch_size, self.config.n_states, self.config.n_states,
            device=macro_seq.device
        )

        # Fill in valid transitions
        for from_state, to_states in self.valid_transitions_per_state.items():
            if from_state in probs:
                state_probs = probs[from_state][:, time_idx, :]  # (batch, n_to)
                for idx, to_state in enumerate(to_states):
                    matrix[:, from_state, to_state] = state_probs[:, idx]

        # Ensure absorbing states
        for state in [4, 5, 6]:  # default, prepaid, matured
            matrix[:, state, :] = 0
            matrix[:, state, state] = 1.0

        return matrix


# =============================================================================
# DATASET
# =============================================================================

class TransitionDataset(Dataset):
    """Dataset for transition probability training"""

    def __init__(
        self,
        macro_data: np.ndarray,
        cohort_features: Dict[str, np.ndarray],
        transition_targets: np.ndarray
    ):
        """
        Args:
            macro_data: (n_samples, seq_len, n_macro_vars)
            cohort_features: Dict with numpy arrays
            transition_targets: (n_samples, seq_len, n_states, n_states)
        """
        self.macro_data = torch.FloatTensor(macro_data)
        self.cohort_features = {
            k: torch.LongTensor(v) if k != 'continuous' else torch.FloatTensor(v)
            for k, v in cohort_features.items()
        }
        self.transition_targets = torch.FloatTensor(transition_targets)

    def __len__(self):
        return len(self.macro_data)

    def __getitem__(self, idx):
        cohort = {k: v[idx] for k, v in self.cohort_features.items()}
        return self.macro_data[idx], cohort, self.transition_targets[idx]


# =============================================================================
# TRAINING
# =============================================================================

class TransitionTrainer:
    """Trainer for Transition Transformer"""

    def __init__(
        self,
        model: TransitionTransformer,
        config: TransitionTransformerConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        self.scheduler = self._get_scheduler()
        self.step = 0

    def _get_scheduler(self):
        """Warmup + cosine decay scheduler"""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            else:
                progress = (step - self.config.warmup_steps) / (
                    self.config.n_epochs * 100 - self.config.warmup_steps
                )
                return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def compute_loss(
        self,
        predictions: Dict[int, torch.Tensor],
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for transition predictions.

        Args:
            predictions: Dict from_state -> probs (batch, seq, n_to)
            targets: Target transition matrices (batch, seq, n_states, n_states)
        """
        total_loss = 0.0
        n_terms = 0

        for from_state, to_states in self.model.valid_transitions_per_state.items():
            if from_state not in predictions:
                continue

            pred_probs = predictions[from_state]  # (batch, seq, n_to)
            target_probs = targets[:, :, from_state, to_states]  # (batch, seq, n_to)

            # KL divergence
            eps = 1e-8
            kl = target_probs * (torch.log(target_probs + eps) - torch.log(pred_probs + eps))
            loss = kl.sum(dim=-1).mean()

            total_loss += loss
            n_terms += 1

        return total_loss / max(n_terms, 1)

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""

        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for macro_seq, cohort_features, targets in dataloader:
            macro_seq = macro_seq.to(self.device)
            cohort_features = {k: v.to(self.device) for k, v in cohort_features.items()}
            targets = targets.to(self.device)

            # Forward
            predictions = self.model(macro_seq, cohort_features)

            # Loss
            loss = self.compute_loss(predictions, targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop"""

        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(self.config.n_epochs):
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)

            if val_loader:
                self.model.eval()
                val_loss = 0.0
                n_batches = 0
                with torch.no_grad():
                    for macro_seq, cohort_features, targets in val_loader:
                        macro_seq = macro_seq.to(self.device)
                        cohort_features = {k: v.to(self.device) for k, v in cohort_features.items()}
                        targets = targets.to(self.device)

                        predictions = self.model(macro_seq, cohort_features)
                        loss = self.compute_loss(predictions, targets)
                        val_loss += loss.item()
                        n_batches += 1

                val_loss /= n_batches
                history['val_loss'].append(val_loss)

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.config.n_epochs} | Train Loss: {train_loss:.4f}"
                if val_loader:
                    msg += f" | Val Loss: {val_loss:.4f}"
                print(msg)

        return history


# =============================================================================
# SYNTHETIC DATA GENERATION
# =============================================================================

def generate_synthetic_transition_data(
    n_cohorts: int = 500,
    seq_length: int = 60
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Generate synthetic training data for transition model.

    Returns:
        macro_data, cohort_features, transition_targets
    """
    rng = np.random.default_rng(42)

    # Generate macro data
    macro_data = np.zeros((n_cohorts, seq_length, 9))
    for i in range(n_cohorts):
        # Base macro with some variation
        base_gdp = rng.normal(0.02, 0.005)
        base_unemp = rng.normal(0.05, 0.01)

        for t in range(seq_length):
            # Add temporal dynamics
            cycle = np.sin(2 * np.pi * t / 24) * 0.01  # 2-year cycle

            macro_data[i, t, 0] = base_gdp + cycle + rng.normal(0, 0.002)  # GDP
            macro_data[i, t, 1] = base_unemp - cycle * 2 + rng.normal(0, 0.002)  # Unemployment
            macro_data[i, t, 2] = 0.02 + rng.normal(0, 0.003)  # Inflation
            macro_data[i, t, 3] = 0.03 + rng.normal(0, 0.005)  # Policy rate
            macro_data[i, t, 4] = 0.035 + rng.normal(0, 0.005)  # 10y yield
            macro_data[i, t, 5] = 100 + rng.normal(0, 20)  # IG spread
            macro_data[i, t, 6] = 400 - cycle * 100 + rng.normal(0, 50)  # HY spread
            macro_data[i, t, 7] = 100 + rng.normal(0, 5)  # Property index
            macro_data[i, t, 8] = 0.08 + rng.normal(0, 0.05)  # Equity return

    # Generate cohort features
    cohort_features = {
        'asset_class': rng.integers(0, 4, n_cohorts),
        'geography': rng.integers(0, 10, n_cohorts),
        'vintage': rng.integers(0, 24, n_cohorts),
        'continuous': rng.uniform(0, 1, (n_cohorts, 4))  # normalized features
    }

    # Generate transition targets
    # Base transition matrix
    base_matrix = np.array([
        [0.975, 0.015, 0.000, 0.000, 0.000, 0.008, 0.002],
        [0.400, 0.295, 0.300, 0.000, 0.000, 0.005, 0.000],
        [0.200, 0.000, 0.395, 0.400, 0.000, 0.005, 0.000],
        [0.100, 0.000, 0.000, 0.395, 0.500, 0.005, 0.000],
        [0.000, 0.000, 0.000, 0.000, 1.000, 0.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 1.000, 0.000],
        [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 1.000],
    ])

    transition_targets = np.zeros((n_cohorts, seq_length, N_STATES, N_STATES))

    for i in range(n_cohorts):
        for t in range(seq_length):
            # Adjust based on macro
            unemp = macro_data[i, t, 1]
            spread = macro_data[i, t, 6]

            # Create adjusted matrix
            adj_matrix = base_matrix.copy()

            # Higher unemployment/spread -> more defaults
            stress_factor = (unemp - 0.05) / 0.05 + (spread - 400) / 400
            stress_factor = max(-0.5, min(0.5, stress_factor))

            adj_matrix[0, 1] *= (1 + stress_factor)  # More delinquency
            adj_matrix[1, 2] *= (1 + stress_factor * 0.8)
            adj_matrix[2, 3] *= (1 + stress_factor * 0.6)
            adj_matrix[3, 4] *= (1 + stress_factor * 0.5)

            adj_matrix[1, 0] *= (1 - stress_factor * 0.5)  # Less cure
            adj_matrix[2, 0] *= (1 - stress_factor * 0.5)
            adj_matrix[3, 0] *= (1 - stress_factor * 0.5)

            # Normalize rows
            adj_matrix = adj_matrix / adj_matrix.sum(axis=1, keepdims=True)

            transition_targets[i, t] = adj_matrix

    return macro_data, cohort_features, transition_targets


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train and demonstrate Transition Transformer"""

    print("=" * 60)
    print("TRANSITION TRANSFORMER - COHORT-LEVEL DYNAMICS")
    print("=" * 60)

    config = TransitionTransformerConfig(
        n_macro_vars=9,
        seq_length=60,
        d_model=128,
        n_heads=8,
        n_encoder_layers=4,
        n_epochs=50
    )

    # Generate synthetic data
    print("\nGenerating synthetic training data...")
    macro_data, cohort_features, transition_targets = generate_synthetic_transition_data(
        n_cohorts=500,
        seq_length=60
    )

    # Create dataset
    dataset = TransitionDataset(macro_data, cohort_features, transition_targets)

    # Split
    n_train = int(0.8 * len(dataset))
    train_idx = np.random.choice(len(dataset), n_train, replace=False)
    val_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)

    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # Create and train model
    print("\nTraining Transition Transformer...")
    model = TransitionTransformer(config)
    trainer = TransitionTrainer(model, config)
    history = trainer.train(train_loader, val_loader, verbose=True)

    # Demonstrate prediction
    print("\n" + "=" * 60)
    print("PREDICTING TRANSITION MATRICES")
    print("=" * 60)

    model.eval()
    with torch.no_grad():
        # Get sample batch
        macro_sample = torch.FloatTensor(macro_data[:1]).to(trainer.device)
        cohort_sample = {
            k: torch.LongTensor([v[0]]).to(trainer.device) if k != 'continuous'
            else torch.FloatTensor([v[0]]).to(trainer.device)
            for k, v in cohort_features.items()
        }

        # Get transition matrix at month 0 and month 30
        matrix_t0 = model.get_transition_matrix(macro_sample, cohort_sample, 0)
        matrix_t30 = model.get_transition_matrix(macro_sample, cohort_sample, 30)

        print("\nTransition matrix at t=0:")
        print(pd.DataFrame(
            matrix_t0[0].cpu().numpy(),
            index=STATES,
            columns=STATES
        ).round(4))

        print("\nTransition matrix at t=30:")
        print(pd.DataFrame(
            matrix_t30[0].cpu().numpy(),
            index=STATES,
            columns=STATES
        ).round(4))

    # Save model
    output_dir = Path(__file__).parent.parent.parent / 'models'
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'transition_transformer.pt')
    print(f"\nModel saved to {output_dir / 'transition_transformer.pt'}")


if __name__ == '__main__':
    main()

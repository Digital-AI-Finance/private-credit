"""
Macro Scenario VAE - Conditional Variational Autoencoder for Macro Paths

Generates correlated macroeconomic time series with:
- Conditional generation based on scenario type (baseline/adverse/severe)
- Learned cross-correlations between macro variables
- Temporal dynamics via LSTM/Transformer encoder-decoder
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


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MacroVAEConfig:
    """Configuration for Macro VAE"""
    # Data dimensions
    n_macro_vars: int = 9
    seq_length: int = 60
    n_scenarios: int = 4  # baseline, adverse, severely_adverse, stagflation

    # Model architecture
    hidden_dim: int = 128
    latent_dim: int = 32
    n_lstm_layers: int = 2
    dropout: float = 0.2

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 100
    kl_weight: float = 0.1
    kl_annealing_epochs: int = 20


# =============================================================================
# DATASET
# =============================================================================

class MacroDataset(Dataset):
    """Dataset for macro time series"""

    def __init__(
        self,
        data: np.ndarray,
        scenario_labels: np.ndarray,
        normalize: bool = True
    ):
        """
        Args:
            data: Array of shape (n_samples, seq_length, n_vars)
            scenario_labels: Array of scenario indices (n_samples,)
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(scenario_labels)
        self.normalize = normalize

        if normalize:
            self.mean = self.data.mean(dim=(0, 1), keepdim=True)
            self.std = self.data.std(dim=(0, 1), keepdim=True) + 1e-8
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = torch.zeros(1, 1, data.shape[-1])
            self.std = torch.ones(1, 1, data.shape[-1])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def denormalize(self, data: torch.Tensor) -> torch.Tensor:
        """Convert normalized data back to original scale"""
        return data * self.std + self.mean


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class LSTMEncoder(nn.Module):
    """LSTM-based encoder for time series"""

    def __init__(self, config: MacroVAEConfig):
        super().__init__()
        self.config = config

        self.lstm = nn.LSTM(
            input_size=config.n_macro_vars,
            hidden_size=config.hidden_dim,
            num_layers=config.n_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.n_lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Scenario embedding
        self.scenario_embed = nn.Embedding(
            config.n_scenarios,
            config.hidden_dim
        )

        # Project to latent space
        encoder_output_dim = config.hidden_dim * 2  # Bidirectional
        self.fc_mu = nn.Linear(encoder_output_dim + config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_dim + config.hidden_dim, config.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        scenario: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input sequence (batch, seq_len, n_vars)
            scenario: Scenario labels (batch,)

        Returns:
            mu, logvar: Latent distribution parameters
        """
        # Encode sequence
        _, (h_n, _) = self.lstm(x)

        # Concatenate forward and backward hidden states
        h_n = h_n.view(self.config.n_lstm_layers, 2, -1, self.config.hidden_dim)
        h_forward = h_n[-1, 0]  # Last layer, forward
        h_backward = h_n[-1, 1]  # Last layer, backward
        h_combined = torch.cat([h_forward, h_backward], dim=-1)

        # Add scenario embedding
        scenario_emb = self.scenario_embed(scenario)
        h_final = torch.cat([h_combined, scenario_emb], dim=-1)

        # Compute latent parameters
        mu = self.fc_mu(h_final)
        logvar = self.fc_logvar(h_final)

        return mu, logvar


class LSTMDecoder(nn.Module):
    """LSTM-based decoder for time series generation"""

    def __init__(self, config: MacroVAEConfig):
        super().__init__()
        self.config = config

        # Project latent to initial hidden state
        self.fc_init = nn.Linear(
            config.latent_dim + config.hidden_dim,  # z + scenario
            config.hidden_dim * config.n_lstm_layers * 2  # h + c for each layer
        )

        # Scenario embedding
        self.scenario_embed = nn.Embedding(
            config.n_scenarios,
            config.hidden_dim
        )

        # Decoder LSTM
        self.lstm = nn.LSTM(
            input_size=config.n_macro_vars + config.hidden_dim,  # prev output + scenario
            hidden_size=config.hidden_dim,
            num_layers=config.n_lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.n_lstm_layers > 1 else 0
        )

        # Output projection
        self.fc_out = nn.Linear(config.hidden_dim, config.n_macro_vars)

    def forward(
        self,
        z: torch.Tensor,
        scenario: torch.Tensor,
        seq_length: int
    ) -> torch.Tensor:
        """
        Generate sequence from latent code.

        Args:
            z: Latent code (batch, latent_dim)
            scenario: Scenario labels (batch,)
            seq_length: Length of sequence to generate

        Returns:
            Generated sequence (batch, seq_length, n_vars)
        """
        batch_size = z.shape[0]

        # Get scenario embedding
        scenario_emb = self.scenario_embed(scenario)

        # Initialize hidden state from latent
        z_cond = torch.cat([z, scenario_emb], dim=-1)
        h_init = self.fc_init(z_cond)

        # Reshape to (n_layers, batch, hidden_dim) for h and c
        h_init = h_init.view(
            batch_size,
            self.config.n_lstm_layers,
            2,
            self.config.hidden_dim
        )
        h_0 = h_init[:, :, 0, :].permute(1, 0, 2).contiguous()
        c_0 = h_init[:, :, 1, :].permute(1, 0, 2).contiguous()

        # Generate autoregressively
        outputs = []
        prev_output = torch.zeros(batch_size, self.config.n_macro_vars, device=z.device)
        hidden = (h_0, c_0)

        for t in range(seq_length):
            # Input: previous output + scenario embedding
            decoder_input = torch.cat([prev_output, scenario_emb], dim=-1)
            decoder_input = decoder_input.unsqueeze(1)

            # LSTM step
            lstm_out, hidden = self.lstm(decoder_input, hidden)

            # Output projection
            output = self.fc_out(lstm_out.squeeze(1))
            outputs.append(output)
            prev_output = output

        return torch.stack(outputs, dim=1)


# =============================================================================
# MACRO VAE MODEL
# =============================================================================

class MacroVAE(nn.Module):
    """
    Conditional Variational Autoencoder for Macro Scenario Generation

    Architecture:
    - Bidirectional LSTM encoder
    - Scenario-conditioned latent space
    - Autoregressive LSTM decoder
    """

    def __init__(self, config: MacroVAEConfig):
        super().__init__()
        self.config = config

        self.encoder = LSTMEncoder(config)
        self.decoder = LSTMDecoder(config)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        x: torch.Tensor,
        scenario: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.

        Args:
            x: Input sequence (batch, seq_len, n_vars)
            scenario: Scenario labels (batch,)

        Returns:
            recon: Reconstructed sequence
            mu: Latent mean
            logvar: Latent log-variance
        """
        # Encode
        mu, logvar = self.encoder(x, scenario)

        # Sample latent
        z = self.reparameterize(mu, logvar)

        # Decode
        recon = self.decoder(z, scenario, x.shape[1])

        return recon, mu, logvar

    def generate(
        self,
        scenario: torch.Tensor,
        seq_length: int,
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate macro scenarios from prior.

        Args:
            scenario: Scenario labels (batch,) or single int
            seq_length: Length of sequence to generate
            n_samples: Number of samples per scenario
            temperature: Sampling temperature (higher = more diversity)

        Returns:
            Generated sequences (n_samples * batch, seq_length, n_vars)
        """
        if isinstance(scenario, int):
            scenario = torch.tensor([scenario])

        batch_size = len(scenario)

        # Repeat for multiple samples
        scenario = scenario.repeat(n_samples)

        # Sample from prior
        z = torch.randn(
            n_samples * batch_size,
            self.config.latent_dim,
            device=next(self.parameters()).device
        ) * temperature

        # Generate
        with torch.no_grad():
            generated = self.decoder(z, scenario, seq_length)

        return generated

    def loss_function(
        self,
        recon: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        kl_weight: float = 1.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute VAE loss.

        Args:
            recon: Reconstructed sequence
            x: Original sequence
            mu: Latent mean
            logvar: Latent log-variance
            kl_weight: Weight for KL divergence term

        Returns:
            total_loss, loss_dict
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x, reduction='mean')

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        total_loss = recon_loss + kl_weight * kl_loss

        loss_dict = {
            'total': total_loss.item(),
            'recon': recon_loss.item(),
            'kl': kl_loss.item()
        }

        return total_loss, loss_dict


# =============================================================================
# TRAINING
# =============================================================================

class MacroVAETrainer:
    """Trainer for Macro VAE"""

    def __init__(
        self,
        model: MacroVAE,
        config: MacroVAEConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch"""

        self.model.train()

        # KL annealing
        if epoch < self.config.kl_annealing_epochs:
            kl_weight = self.config.kl_weight * epoch / self.config.kl_annealing_epochs
        else:
            kl_weight = self.config.kl_weight

        total_losses = {'total': 0, 'recon': 0, 'kl': 0}
        n_batches = 0

        for batch_x, batch_scenario in dataloader:
            batch_x = batch_x.to(self.device)
            batch_scenario = batch_scenario.to(self.device)

            # Forward pass
            recon, mu, logvar = self.model(batch_x, batch_scenario)

            # Compute loss
            loss, loss_dict = self.model.loss_function(
                recon, batch_x, mu, logvar, kl_weight
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            for k, v in loss_dict.items():
                total_losses[k] += v
            n_batches += 1

        self.scheduler.step()

        # Average losses
        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""

        self.model.eval()
        total_losses = {'total': 0, 'recon': 0, 'kl': 0}
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_scenario in dataloader:
                batch_x = batch_x.to(self.device)
                batch_scenario = batch_scenario.to(self.device)

                recon, mu, logvar = self.model(batch_x, batch_scenario)
                _, loss_dict = self.model.loss_function(
                    recon, batch_x, mu, logvar, self.config.kl_weight
                )

                for k, v in loss_dict.items():
                    total_losses[k] += v
                n_batches += 1

        avg_losses = {k: v / n_batches for k, v in total_losses.items()}
        return avg_losses

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop"""

        history = {
            'train_loss': [],
            'train_recon': [],
            'train_kl': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': []
        }

        for epoch in range(self.config.n_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            history['train_loss'].append(train_losses['total'])
            history['train_recon'].append(train_losses['recon'])
            history['train_kl'].append(train_losses['kl'])

            # Validate
            if val_loader is not None:
                val_losses = self.evaluate(val_loader)
                history['val_loss'].append(val_losses['total'])
                history['val_recon'].append(val_losses['recon'])
                history['val_kl'].append(val_losses['kl'])

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.config.n_epochs} | "
                msg += f"Train Loss: {train_losses['total']:.4f} "
                msg += f"(Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.4f})"
                if val_loader:
                    msg += f" | Val Loss: {val_losses['total']:.4f}"
                print(msg)

        return history


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

def generate_scenarios(
    model: MacroVAE,
    n_samples: int,
    scenario_type: str,
    seq_length: int = 60,
    dataset: Optional[MacroDataset] = None
) -> np.ndarray:
    """
    Generate macro scenarios of specified type.

    Args:
        model: Trained MacroVAE
        n_samples: Number of scenarios to generate
        scenario_type: 'baseline', 'adverse', 'severely_adverse', 'stagflation'
        seq_length: Sequence length
        dataset: Dataset for denormalization

    Returns:
        Generated scenarios (n_samples, seq_length, n_vars)
    """
    scenario_map = {
        'baseline': 0,
        'adverse': 1,
        'severely_adverse': 2,
        'stagflation': 3
    }

    scenario_idx = scenario_map.get(scenario_type, 0)

    model.eval()
    scenario_tensor = torch.tensor([scenario_idx], device=next(model.parameters()).device)

    generated = model.generate(
        scenario=scenario_tensor,
        seq_length=seq_length,
        n_samples=n_samples,
        temperature=1.0
    )

    generated = generated.cpu().numpy()

    # Denormalize if dataset provided
    if dataset is not None:
        generated = generated * dataset.std.numpy() + dataset.mean.numpy()

    return generated


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train and demonstrate Macro VAE"""

    print("=" * 60)
    print("MACRO VAE - CONDITIONAL SCENARIO GENERATOR")
    print("=" * 60)

    # Configuration
    config = MacroVAEConfig(
        n_macro_vars=9,
        seq_length=60,
        n_scenarios=4,
        hidden_dim=128,
        latent_dim=32,
        n_epochs=50,
        batch_size=16
    )

    # Generate synthetic training data
    print("\nGenerating synthetic training data...")

    from simulate_macro import MacroScenarioGenerator

    macro_gen = MacroScenarioGenerator(n_months=60, start_date='2020-01-01')

    # Generate multiple paths per scenario
    n_paths_per_scenario = 100
    all_data = []
    all_labels = []

    scenario_names = ['baseline', 'adverse', 'severely_adverse', 'stagflation']

    for scenario_idx, scenario_name in enumerate(scenario_names):
        print(f"Generating {n_paths_per_scenario} paths for {scenario_name}...")

        for i in range(n_paths_per_scenario):
            macro_gen.rng = np.random.default_rng(42 + i + scenario_idx * 1000)
            df = macro_gen.generate_scenario(scenario_name)

            # Extract numeric columns
            numeric_cols = ['gdp_growth_yoy', 'unemployment_rate', 'inflation_rate',
                          'policy_rate', 'yield_10y', 'credit_spread_ig',
                          'credit_spread_hy', 'property_price_index', 'equity_return']
            data = df[numeric_cols].values
            all_data.append(data)
            all_labels.append(scenario_idx)

    all_data = np.array(all_data)
    all_labels = np.array(all_labels)

    print(f"Training data shape: {all_data.shape}")

    # Create dataset
    dataset = MacroDataset(all_data, all_labels, normalize=True)

    # Split train/val
    n_train = int(0.8 * len(dataset))
    train_idx = np.random.choice(len(dataset), n_train, replace=False)
    val_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)

    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # Create and train model
    print("\nTraining Macro VAE...")
    model = MacroVAE(config)
    trainer = MacroVAETrainer(model, config)
    history = trainer.train(train_loader, val_loader, verbose=True)

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATING SCENARIOS")
    print("=" * 60)

    for scenario_name in scenario_names:
        samples = generate_scenarios(
            model,
            n_samples=5,
            scenario_type=scenario_name,
            seq_length=60,
            dataset=dataset
        )
        print(f"\n{scenario_name.upper()} (5 samples):")
        print(f"  Mean GDP growth: {samples[:, :, 0].mean():.4f}")
        print(f"  Mean unemployment: {samples[:, :, 1].mean():.4f}")
        print(f"  Mean HY spread: {samples[:, :, 6].mean():.1f}")

    # Save model
    output_dir = Path(__file__).parent.parent.parent / 'models'
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'macro_vae.pt')
    print(f"\nModel saved to {output_dir / 'macro_vae.pt'}")


if __name__ == '__main__':
    main()

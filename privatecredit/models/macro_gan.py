"""
Macro Scenario GAN - Wasserstein GAN with Gradient Penalty for Macro Paths

Generates correlated macroeconomic time series using WGAN-GP:
- Wasserstein distance for stable training
- Gradient penalty for Lipschitz constraint
- Conditional generation based on scenario type
- Alternative to VAE for sharper distributions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class MacroGANConfig:
    """Configuration for Macro GAN"""
    # Data dimensions
    n_macro_vars: int = 9
    seq_length: int = 60
    n_scenarios: int = 4  # baseline, adverse, severely_adverse, stagflation

    # Model architecture
    latent_dim: int = 64
    hidden_dim: int = 128
    n_layers: int = 3
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate_g: float = 1e-4
    learning_rate_d: float = 4e-4
    n_epochs: int = 200
    n_critic: int = 5  # Discriminator updates per generator update
    gp_weight: float = 10.0  # Gradient penalty weight


# =============================================================================
# GENERATOR
# =============================================================================

class Generator(nn.Module):
    """LSTM-based generator for time series"""

    def __init__(self, config: MacroGANConfig):
        super().__init__()
        self.config = config

        # Scenario embedding
        self.scenario_embed = nn.Embedding(config.n_scenarios, config.hidden_dim)

        # Project latent + scenario to initial hidden
        self.fc_init = nn.Linear(
            config.latent_dim + config.hidden_dim,
            config.hidden_dim * config.n_layers * 2  # h + c for each layer
        )

        # LSTM generator
        self.lstm = nn.LSTM(
            input_size=config.n_macro_vars + config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0
        )

        # Output projection with skip connection
        self.fc_out = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.hidden_dim, config.n_macro_vars)
        )

        # Trend component
        self.trend_net = nn.Sequential(
            nn.Linear(config.latent_dim + config.hidden_dim, config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(config.hidden_dim, config.n_macro_vars * config.seq_length)
        )

    def forward(
        self,
        z: torch.Tensor,
        scenario: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate macro sequence from noise.

        Args:
            z: Latent noise (batch, latent_dim)
            scenario: Scenario labels (batch,)

        Returns:
            Generated sequence (batch, seq_length, n_vars)
        """
        batch_size = z.shape[0]

        # Scenario embedding
        scenario_emb = self.scenario_embed(scenario)

        # Initialize hidden state
        z_cond = torch.cat([z, scenario_emb], dim=-1)
        h_init = self.fc_init(z_cond)

        # Reshape to (n_layers, batch, hidden_dim) for h and c
        h_init = h_init.view(
            batch_size,
            self.config.n_layers,
            2,
            self.config.hidden_dim
        )
        h_0 = h_init[:, :, 0, :].permute(1, 0, 2).contiguous()
        c_0 = h_init[:, :, 1, :].permute(1, 0, 2).contiguous()

        # Generate trend component
        trend = self.trend_net(z_cond)
        trend = trend.view(batch_size, self.config.seq_length, self.config.n_macro_vars)

        # Generate autoregressively with residuals
        outputs = []
        prev_output = torch.zeros(batch_size, self.config.n_macro_vars, device=z.device)
        hidden = (h_0, c_0)

        for t in range(self.config.seq_length):
            # Input: previous output + scenario embedding
            lstm_input = torch.cat([prev_output, scenario_emb], dim=-1).unsqueeze(1)

            # LSTM step
            lstm_out, hidden = self.lstm(lstm_input, hidden)

            # Residual output
            residual = self.fc_out(lstm_out.squeeze(1))
            output = trend[:, t, :] + residual
            outputs.append(output)
            prev_output = output

        return torch.stack(outputs, dim=1)


# =============================================================================
# DISCRIMINATOR (CRITIC)
# =============================================================================

class Discriminator(nn.Module):
    """Bidirectional LSTM critic for time series"""

    def __init__(self, config: MacroGANConfig):
        super().__init__()
        self.config = config

        # Scenario embedding
        self.scenario_embed = nn.Embedding(config.n_scenarios, config.hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(
            config.n_macro_vars + config.hidden_dim,
            config.hidden_dim
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.n_layers > 1 else 0
        )

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1)
        )

        # Output head (no sigmoid - Wasserstein)
        self.fc_out = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        scenario: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Wasserstein critic score.

        Args:
            x: Input sequence (batch, seq_length, n_vars)
            scenario: Scenario labels (batch,)

        Returns:
            Critic scores (batch, 1)
        """
        batch_size = x.shape[0]
        seq_length = x.shape[1]

        # Scenario embedding - broadcast across time
        scenario_emb = self.scenario_embed(scenario)
        scenario_emb = scenario_emb.unsqueeze(1).expand(-1, seq_length, -1)

        # Concatenate with input
        x_cond = torch.cat([x, scenario_emb], dim=-1)

        # Project input
        x_proj = self.input_proj(x_cond)

        # LSTM encoding
        lstm_out, _ = self.lstm(x_proj)

        # Attention over time
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)

        # Output score
        score = self.fc_out(context)

        return score


# =============================================================================
# MACRO GAN MODEL
# =============================================================================

class MacroGAN(nn.Module):
    """
    Wasserstein GAN with Gradient Penalty for Macro Scenario Generation

    Architecture:
    - LSTM-based generator with trend + residual
    - Bidirectional LSTM critic with attention
    - Gradient penalty for Lipschitz constraint
    """

    def __init__(self, config: MacroGANConfig):
        super().__init__()
        self.config = config

        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def generate(
        self,
        scenario: torch.Tensor,
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate macro scenarios.

        Args:
            scenario: Scenario labels (batch,) or single int
            n_samples: Number of samples per scenario
            temperature: Noise scaling (higher = more diversity)

        Returns:
            Generated sequences (n_samples * batch, seq_length, n_vars)
        """
        if isinstance(scenario, int):
            scenario = torch.tensor([scenario], device=next(self.parameters()).device)

        batch_size = len(scenario)
        scenario = scenario.repeat(n_samples)

        z = torch.randn(
            n_samples * batch_size,
            self.config.latent_dim,
            device=next(self.parameters()).device
        ) * temperature

        with torch.no_grad():
            generated = self.generator(z, scenario)

        return generated

    def compute_gradient_penalty(
        self,
        real_data: torch.Tensor,
        fake_data: torch.Tensor,
        scenario: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute gradient penalty for WGAN-GP.

        Args:
            real_data: Real sequences (batch, seq_length, n_vars)
            fake_data: Generated sequences (batch, seq_length, n_vars)
            scenario: Scenario labels (batch,)

        Returns:
            Gradient penalty loss
        """
        batch_size = real_data.shape[0]
        device = real_data.device

        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        # Critic score for interpolated
        d_interpolated = self.discriminator(interpolated, scenario)

        # Compute gradients
        gradients = grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]

        # Compute gradient norm
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)

        # Gradient penalty
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()

        return gradient_penalty


# =============================================================================
# TRAINING
# =============================================================================

class MacroGANTrainer:
    """Trainer for Macro GAN"""

    def __init__(
        self,
        model: MacroGAN,
        config: MacroGANConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Separate optimizers for G and D
        self.optimizer_g = torch.optim.Adam(
            model.generator.parameters(),
            lr=config.learning_rate_g,
            betas=(0.5, 0.9)
        )
        self.optimizer_d = torch.optim.Adam(
            model.discriminator.parameters(),
            lr=config.learning_rate_d,
            betas=(0.5, 0.9)
        )

    def train_discriminator_step(
        self,
        real_data: torch.Tensor,
        scenario: torch.Tensor
    ) -> Dict[str, float]:
        """Single discriminator update step"""

        batch_size = real_data.shape[0]

        # Generate fake data
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        with torch.no_grad():
            fake_data = self.model.generator(z, scenario)

        # Critic scores
        d_real = self.model.discriminator(real_data, scenario)
        d_fake = self.model.discriminator(fake_data, scenario)

        # Wasserstein loss
        d_loss = d_fake.mean() - d_real.mean()

        # Gradient penalty
        gp = self.model.compute_gradient_penalty(real_data, fake_data, scenario)

        # Total discriminator loss
        total_d_loss = d_loss + self.config.gp_weight * gp

        # Update
        self.optimizer_d.zero_grad()
        total_d_loss.backward()
        self.optimizer_d.step()

        return {
            'd_loss': d_loss.item(),
            'gp': gp.item(),
            'w_dist': -d_loss.item()  # Wasserstein distance estimate
        }

    def train_generator_step(
        self,
        batch_size: int,
        scenario: torch.Tensor
    ) -> Dict[str, float]:
        """Single generator update step"""

        # Generate fake data
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        fake_data = self.model.generator(z, scenario)

        # Critic score
        d_fake = self.model.discriminator(fake_data, scenario)

        # Generator loss (maximize critic score)
        g_loss = -d_fake.mean()

        # Update
        self.optimizer_g.zero_grad()
        g_loss.backward()
        self.optimizer_g.step()

        return {'g_loss': g_loss.item()}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""

        self.model.train()
        epoch_losses = {'d_loss': 0, 'g_loss': 0, 'gp': 0, 'w_dist': 0}
        n_batches = 0

        for batch_x, batch_scenario in dataloader:
            batch_x = batch_x.to(self.device)
            batch_scenario = batch_scenario.to(self.device)
            batch_size = batch_x.shape[0]

            # Train discriminator n_critic times
            for _ in range(self.config.n_critic):
                d_losses = self.train_discriminator_step(batch_x, batch_scenario)

            # Train generator once
            g_losses = self.train_generator_step(batch_size, batch_scenario)

            # Accumulate
            epoch_losses['d_loss'] += d_losses['d_loss']
            epoch_losses['g_loss'] += g_losses['g_loss']
            epoch_losses['gp'] += d_losses['gp']
            epoch_losses['w_dist'] += d_losses['w_dist']
            n_batches += 1

        # Average
        return {k: v / n_batches for k, v in epoch_losses.items()}

    def train(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop"""

        history = {'d_loss': [], 'g_loss': [], 'gp': [], 'w_dist': []}

        for epoch in range(self.config.n_epochs):
            losses = self.train_epoch(dataloader)

            for k, v in losses.items():
                history[k].append(v)

            if verbose and (epoch + 1) % 20 == 0:
                print(
                    f"Epoch {epoch+1}/{self.config.n_epochs} | "
                    f"D Loss: {losses['d_loss']:.4f} | "
                    f"G Loss: {losses['g_loss']:.4f} | "
                    f"W Dist: {losses['w_dist']:.4f}"
                )

        return history


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

def generate_scenarios_gan(
    model: MacroGAN,
    n_samples: int,
    scenario_type: str,
    dataset=None
) -> np.ndarray:
    """
    Generate macro scenarios using trained GAN.

    Args:
        model: Trained MacroGAN
        n_samples: Number of scenarios to generate
        scenario_type: 'baseline', 'adverse', 'severely_adverse', 'stagflation'
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
        n_samples=n_samples,
        temperature=1.0
    )

    generated = generated.cpu().numpy()

    # Denormalize if dataset provided
    if dataset is not None and hasattr(dataset, 'std'):
        generated = generated * dataset.std.numpy() + dataset.mean.numpy()

    return generated


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train and demonstrate Macro GAN"""

    print("=" * 60)
    print("MACRO GAN - WASSERSTEIN GAN WITH GRADIENT PENALTY")
    print("=" * 60)

    # Configuration
    config = MacroGANConfig(
        n_macro_vars=9,
        seq_length=60,
        n_scenarios=4,
        hidden_dim=128,
        latent_dim=64,
        n_epochs=100,
        batch_size=16,
        n_critic=5
    )

    # Generate synthetic training data
    print("\nGenerating synthetic training data...")

    from privatecredit.data.simulate_macro import MacroScenarioGenerator
    from privatecredit.models.macro_vae import MacroDataset

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
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Create and train model
    print("\nTraining Macro GAN...")
    model = MacroGAN(config)
    trainer = MacroGANTrainer(model, config)
    history = trainer.train(dataloader, verbose=True)

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATING SCENARIOS")
    print("=" * 60)

    for scenario_name in scenario_names:
        samples = generate_scenarios_gan(
            model,
            n_samples=5,
            scenario_type=scenario_name,
            dataset=dataset
        )
        print(f"\n{scenario_name.upper()} (5 samples):")
        print(f"  Mean GDP growth: {samples[:, :, 0].mean():.4f}")
        print(f"  Mean unemployment: {samples[:, :, 1].mean():.4f}")
        print(f"  Mean HY spread: {samples[:, :, 6].mean():.1f}")

    # Save model
    output_dir = Path(__file__).parent.parent.parent / 'models'
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'macro_gan.pt')
    print(f"\nModel saved to {output_dir / 'macro_gan.pt'}")


if __name__ == '__main__':
    main()

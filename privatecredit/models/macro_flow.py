"""
Macro Scenario Normalizing Flows - Real NVP for Macro Paths

Generates correlated macroeconomic time series using normalizing flows:
- Exact likelihood computation (unlike VAE/GAN)
- Real NVP (Real-valued Non-Volume Preserving) architecture
- Conditional generation based on scenario type
- Better for tail risk modeling due to flexible density estimation
"""

import numpy as np
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
class MacroFlowConfig:
    """Configuration for Macro Normalizing Flow"""
    # Data dimensions
    n_macro_vars: int = 9
    seq_length: int = 60
    n_scenarios: int = 4  # baseline, adverse, severely_adverse, stagflation

    # Model architecture
    hidden_dim: int = 128
    n_flows: int = 8  # Number of coupling layers
    n_blocks: int = 2  # MLP blocks per coupling layer
    dropout: float = 0.1

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 100
    weight_decay: float = 1e-5


# =============================================================================
# COUPLING LAYERS
# =============================================================================

class CouplingLayer(nn.Module):
    """
    Real NVP Coupling Layer

    Splits input in half, applies affine transformation conditioned on other half.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        condition_dim: int,
        n_blocks: int = 2,
        mask_type: str = 'odd'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mask_type = mask_type

        # Create mask
        mask = torch.zeros(input_dim)
        if mask_type == 'odd':
            mask[::2] = 1
        else:
            mask[1::2] = 1
        self.register_buffer('mask', mask)

        # Scale and translation networks
        n_masked = int(mask.sum().item())
        n_unmasked = input_dim - n_masked

        # Conditioner network
        layers = []
        in_dim = n_masked + condition_dim
        for i in range(n_blocks):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1) if i < n_blocks - 1 else nn.Identity()
            ])
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)

        # Output heads for scale and translation
        self.scale_net = nn.Sequential(
            nn.Linear(hidden_dim, n_unmasked),
            nn.Tanh()  # Bound scale
        )
        self.translate_net = nn.Linear(hidden_dim, n_unmasked)

        # Scale factor for numerical stability
        self.scale_factor = nn.Parameter(torch.ones(1) * 2)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward (encode) or reverse (decode) pass.

        Args:
            x: Input tensor (batch, input_dim)
            condition: Conditioning tensor (batch, condition_dim)
            reverse: If True, compute inverse transformation

        Returns:
            y: Transformed tensor
            log_det: Log determinant of Jacobian
        """
        # Split based on mask
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)

        # Get non-zero elements for conditioning
        mask_bool = self.mask.bool()
        x_cond = x[:, mask_bool]

        # Compute scale and translation
        h = self.net(torch.cat([x_cond, condition], dim=-1))
        s = self.scale_net(h) * self.scale_factor
        t = self.translate_net(h)

        # Expand to full dimension
        s_full = torch.zeros_like(x)
        t_full = torch.zeros_like(x)
        s_full[:, ~mask_bool] = s
        t_full[:, ~mask_bool] = t

        if not reverse:
            # Forward: y = x * exp(s) + t
            y = x_masked + x_unmasked * torch.exp(s_full) + t_full * (1 - self.mask)
            log_det = s.sum(dim=-1)
        else:
            # Reverse: x = (y - t) * exp(-s)
            y = x_masked + (x_unmasked - t_full * (1 - self.mask)) * torch.exp(-s_full)
            log_det = -s.sum(dim=-1)

        return y, log_det


class ActNorm(nn.Module):
    """Activation Normalization layer with data-dependent initialization"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.log_scale = nn.Parameter(torch.zeros(1, dim))
        self.bias = nn.Parameter(torch.zeros(1, dim))
        self.initialized = False

    def initialize(self, x: torch.Tensor):
        """Initialize with data statistics"""
        with torch.no_grad():
            mean = x.mean(dim=0, keepdim=True)
            std = x.std(dim=0, keepdim=True) + 1e-6
            self.bias.data = -mean
            self.log_scale.data = -torch.log(std)
        self.initialized = True

    def forward(
        self,
        x: torch.Tensor,
        reverse: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward or reverse pass"""

        if not self.initialized and not reverse:
            self.initialize(x)

        if not reverse:
            y = (x + self.bias) * torch.exp(self.log_scale)
            log_det = self.log_scale.sum() * torch.ones(x.shape[0], device=x.device)
        else:
            y = x * torch.exp(-self.log_scale) - self.bias
            log_det = -self.log_scale.sum() * torch.ones(x.shape[0], device=x.device)

        return y, log_det


# =============================================================================
# NORMALIZING FLOW MODEL
# =============================================================================

class MacroFlow(nn.Module):
    """
    Normalizing Flow for Macro Scenario Generation

    Architecture:
    - Stack of Real NVP coupling layers
    - ActNorm for stable training
    - Conditional on scenario type
    - Exact log-likelihood computation
    """

    def __init__(self, config: MacroFlowConfig):
        super().__init__()
        self.config = config

        # Total dimension (flattened time series)
        self.data_dim = config.n_macro_vars * config.seq_length

        # Scenario embedding
        self.scenario_embed = nn.Embedding(config.n_scenarios, config.hidden_dim)

        # Build flow layers
        self.flows = nn.ModuleList()

        for i in range(config.n_flows):
            # Alternate mask type
            mask_type = 'odd' if i % 2 == 0 else 'even'

            # Add ActNorm
            self.flows.append(ActNorm(self.data_dim))

            # Add coupling layer
            self.flows.append(
                CouplingLayer(
                    input_dim=self.data_dim,
                    hidden_dim=config.hidden_dim,
                    condition_dim=config.hidden_dim,
                    n_blocks=config.n_blocks,
                    mask_type=mask_type
                )
            )

        # Prior
        self.register_buffer('prior_mean', torch.zeros(self.data_dim))
        self.register_buffer('prior_logvar', torch.zeros(self.data_dim))

    def encode(
        self,
        x: torch.Tensor,
        scenario: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Transform data to latent space.

        Args:
            x: Input sequences (batch, seq_length, n_vars)
            scenario: Scenario labels (batch,)

        Returns:
            z: Latent representation
            log_det: Log determinant of transformation
        """
        batch_size = x.shape[0]

        # Flatten time series
        x_flat = x.view(batch_size, -1)

        # Get scenario embedding
        scenario_emb = self.scenario_embed(scenario)

        # Pass through flows
        z = x_flat
        total_log_det = torch.zeros(batch_size, device=x.device)

        for layer in self.flows:
            if isinstance(layer, ActNorm):
                z, log_det = layer(z, reverse=False)
            else:
                z, log_det = layer(z, scenario_emb, reverse=False)
            total_log_det = total_log_det + log_det

        return z, total_log_det

    def decode(
        self,
        z: torch.Tensor,
        scenario: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform latent to data space.

        Args:
            z: Latent representation (batch, data_dim)
            scenario: Scenario labels (batch,)

        Returns:
            x: Generated sequences (batch, seq_length, n_vars)
        """
        batch_size = z.shape[0]

        # Get scenario embedding
        scenario_emb = self.scenario_embed(scenario)

        # Pass through flows in reverse
        x_flat = z
        for layer in reversed(self.flows):
            if isinstance(layer, ActNorm):
                x_flat, _ = layer(x_flat, reverse=True)
            else:
                x_flat, _ = layer(x_flat, scenario_emb, reverse=True)

        # Reshape to time series
        x = x_flat.view(batch_size, self.config.seq_length, self.config.n_macro_vars)

        return x

    def log_prob(
        self,
        x: torch.Tensor,
        scenario: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute exact log probability of data.

        Args:
            x: Input sequences (batch, seq_length, n_vars)
            scenario: Scenario labels (batch,)

        Returns:
            Log probability for each sample
        """
        # Encode to latent
        z, log_det = self.encode(x, scenario)

        # Log probability under prior (standard normal)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)

        # Total log probability
        log_px = log_pz + log_det

        return log_px

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
            temperature: Sampling temperature

        Returns:
            Generated sequences (n_samples * batch, seq_length, n_vars)
        """
        if isinstance(scenario, int):
            scenario = torch.tensor([scenario], device=next(self.parameters()).device)

        batch_size = len(scenario)
        scenario = scenario.repeat(n_samples)

        # Sample from prior
        z = torch.randn(
            n_samples * batch_size,
            self.data_dim,
            device=next(self.parameters()).device
        ) * temperature

        # Decode
        with torch.no_grad():
            generated = self.decode(z, scenario)

        return generated

    def forward(
        self,
        x: torch.Tensor,
        scenario: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning latent and log determinant.

        Args:
            x: Input sequences (batch, seq_length, n_vars)
            scenario: Scenario labels (batch,)

        Returns:
            z: Latent representation
            log_det: Log determinant
        """
        return self.encode(x, scenario)


# =============================================================================
# TRAINING
# =============================================================================

class MacroFlowTrainer:
    """Trainer for Macro Flow"""

    def __init__(
        self,
        model: MacroFlow,
        config: MacroFlowConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.n_epochs
        )

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""

        self.model.train()
        total_nll = 0
        n_batches = 0

        for batch_x, batch_scenario in dataloader:
            batch_x = batch_x.to(self.device)
            batch_scenario = batch_scenario.to(self.device)

            # Compute negative log likelihood
            log_prob = self.model.log_prob(batch_x, batch_scenario)
            nll = -log_prob.mean()

            # Update
            self.optimizer.zero_grad()
            nll.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_nll += nll.item()
            n_batches += 1

        self.scheduler.step()

        return {'nll': total_nll / n_batches}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""

        self.model.eval()
        total_nll = 0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_scenario in dataloader:
                batch_x = batch_x.to(self.device)
                batch_scenario = batch_scenario.to(self.device)

                log_prob = self.model.log_prob(batch_x, batch_scenario)
                nll = -log_prob.mean()

                total_nll += nll.item()
                n_batches += 1

        return {'nll': total_nll / n_batches}

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop"""

        history = {'train_nll': [], 'val_nll': []}

        for epoch in range(self.config.n_epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            history['train_nll'].append(train_losses['nll'])

            # Validate
            if val_loader is not None:
                val_losses = self.evaluate(val_loader)
                history['val_nll'].append(val_losses['nll'])

            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch+1}/{self.config.n_epochs} | "
                msg += f"Train NLL: {train_losses['nll']:.4f}"
                if val_loader:
                    msg += f" | Val NLL: {val_losses['nll']:.4f}"
                print(msg)

        return history


# =============================================================================
# GENERATION UTILITIES
# =============================================================================

def generate_scenarios_flow(
    model: MacroFlow,
    n_samples: int,
    scenario_type: str,
    dataset=None
) -> np.ndarray:
    """
    Generate macro scenarios using trained flow.

    Args:
        model: Trained MacroFlow
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


def compute_tail_metrics(
    model: MacroFlow,
    scenario_type: str,
    n_samples: int = 10000,
    quantiles: List[float] = [0.01, 0.05, 0.95, 0.99]
) -> Dict[str, np.ndarray]:
    """
    Compute tail risk metrics using exact likelihood.

    Args:
        model: Trained MacroFlow
        scenario_type: Scenario to analyze
        n_samples: Number of samples for estimation
        quantiles: Quantiles to compute

    Returns:
        Dictionary with VaR and other tail metrics
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

    # Generate samples
    samples = model.generate(
        scenario=scenario_tensor,
        n_samples=n_samples,
        temperature=1.0
    )
    samples = samples.cpu().numpy()

    # Compute metrics for each variable
    metrics = {}

    for q in quantiles:
        var_key = f'VaR_{int(q*100)}'
        metrics[var_key] = np.percentile(samples, q * 100, axis=0)

    # Expected shortfall (CVaR)
    for q in [0.01, 0.05]:
        es_key = f'ES_{int(q*100)}'
        threshold = np.percentile(samples, q * 100, axis=0, keepdims=True)
        tail_samples = np.where(samples <= threshold, samples, np.nan)
        metrics[es_key] = np.nanmean(tail_samples, axis=0)

    return metrics


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Train and demonstrate Macro Flow"""

    print("=" * 60)
    print("MACRO FLOW - NORMALIZING FLOW FOR EXACT LIKELIHOOD")
    print("=" * 60)

    # Configuration
    config = MacroFlowConfig(
        n_macro_vars=9,
        seq_length=60,
        n_scenarios=4,
        hidden_dim=128,
        n_flows=8,
        n_epochs=50,
        batch_size=16
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

    # Split train/val
    n_train = int(0.8 * len(dataset))
    train_idx = np.random.choice(len(dataset), n_train, replace=False)
    val_idx = np.setdiff1d(np.arange(len(dataset)), train_idx)

    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)

    # Create and train model
    print("\nTraining Macro Flow...")
    model = MacroFlow(config)
    trainer = MacroFlowTrainer(model, config)
    history = trainer.train(train_loader, val_loader, verbose=True)

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATING SCENARIOS")
    print("=" * 60)

    for scenario_name in scenario_names:
        samples = generate_scenarios_flow(
            model,
            n_samples=5,
            scenario_type=scenario_name,
            dataset=dataset
        )
        print(f"\n{scenario_name.upper()} (5 samples):")
        print(f"  Mean GDP growth: {samples[:, :, 0].mean():.4f}")
        print(f"  Mean unemployment: {samples[:, :, 1].mean():.4f}")
        print(f"  Mean HY spread: {samples[:, :, 6].mean():.1f}")

    # Compute tail metrics
    print("\n" + "=" * 60)
    print("TAIL RISK METRICS (Adverse Scenario)")
    print("=" * 60)

    metrics = compute_tail_metrics(model, 'adverse', n_samples=1000)
    print(f"VaR 1%: GDP growth = {metrics['VaR_1'][:, 0].mean():.4f}")
    print(f"VaR 5%: GDP growth = {metrics['VaR_5'][:, 0].mean():.4f}")
    print(f"ES 5%: GDP growth = {metrics['ES_5'][:, 0].mean():.4f}")

    # Save model
    output_dir = Path(__file__).parent.parent.parent / 'models'
    output_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), output_dir / 'macro_flow.pt')
    print(f"\nModel saved to {output_dir / 'macro_flow.pt'}")


if __name__ == '__main__':
    main()

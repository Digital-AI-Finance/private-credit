"""
Ensemble Model - Combining VAE, GAN, and Flow for Robust Scenario Generation

Combines multiple generative models:
- Weighted averaging of generated scenarios
- Model selection based on likelihood
- Stacking with learned combination weights
- Uncertainty quantification via model disagreement
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CONFIGURATION
# =============================================================================

class EnsembleMethod(Enum):
    """Ensemble combination methods"""
    AVERAGE = "average"  # Simple averaging
    WEIGHTED = "weighted"  # Learned weights
    STACKING = "stacking"  # Meta-learner
    SELECTION = "selection"  # Best model per scenario


@dataclass
class EnsembleConfig:
    """Configuration for Ensemble Model"""
    # Component model configs (can be overridden)
    n_macro_vars: int = 9
    seq_length: int = 60
    n_scenarios: int = 4

    # Ensemble settings
    method: EnsembleMethod = EnsembleMethod.WEIGHTED
    use_vae: bool = True
    use_gan: bool = True
    use_flow: bool = True

    # Stacking meta-learner
    meta_hidden_dim: int = 64
    meta_n_layers: int = 2

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    n_epochs: int = 50


# =============================================================================
# ENSEMBLE MODEL
# =============================================================================

class MacroEnsemble(nn.Module):
    """
    Ensemble of Generative Models for Macro Scenarios

    Combines VAE, GAN, and Flow models for robust generation:
    - Average: Simple mean of all model outputs
    - Weighted: Learned per-scenario weights
    - Stacking: Neural network meta-learner
    - Selection: Best model based on validation likelihood
    """

    def __init__(
        self,
        config: EnsembleConfig,
        vae_model=None,
        gan_model=None,
        flow_model=None
    ):
        super().__init__()
        self.config = config

        # Store component models
        self.vae = vae_model
        self.gan = gan_model
        self.flow = flow_model

        # Count active models
        self.n_models = sum([
            vae_model is not None and config.use_vae,
            gan_model is not None and config.use_gan,
            flow_model is not None and config.use_flow
        ])

        if self.n_models == 0:
            raise ValueError("At least one model must be provided")

        # Learned weights per scenario (for weighted method)
        if config.method == EnsembleMethod.WEIGHTED:
            self.weights = nn.Parameter(
                torch.ones(config.n_scenarios, self.n_models) / self.n_models
            )

        # Meta-learner (for stacking method)
        if config.method == EnsembleMethod.STACKING:
            # Input: concatenated outputs from all models
            input_dim = self.n_models * config.n_macro_vars * config.seq_length

            layers = []
            in_dim = input_dim
            for i in range(config.meta_n_layers):
                layers.extend([
                    nn.Linear(in_dim, config.meta_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1) if i < config.meta_n_layers - 1 else nn.Identity()
                ])
                in_dim = config.meta_hidden_dim

            layers.append(nn.Linear(config.meta_hidden_dim, config.n_macro_vars * config.seq_length))

            self.meta_learner = nn.Sequential(*layers)

        # Model selection scores (for selection method)
        if config.method == EnsembleMethod.SELECTION:
            self.selection_scores = nn.Parameter(
                torch.zeros(config.n_scenarios, self.n_models)
            )

    def _get_active_models(self) -> List[Tuple[str, nn.Module]]:
        """Get list of active models"""
        models = []
        if self.vae is not None and self.config.use_vae:
            models.append(('vae', self.vae))
        if self.gan is not None and self.config.use_gan:
            models.append(('gan', self.gan))
        if self.flow is not None and self.config.use_flow:
            models.append(('flow', self.flow))
        return models

    def generate_all(
        self,
        scenario: torch.Tensor,
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Generate samples from all component models.

        Args:
            scenario: Scenario labels
            n_samples: Number of samples per model
            temperature: Sampling temperature

        Returns:
            Dictionary of model name to generated samples
        """
        outputs = {}

        for name, model in self._get_active_models():
            with torch.no_grad():
                samples = model.generate(
                    scenario=scenario,
                    n_samples=n_samples,
                    temperature=temperature
                )
            outputs[name] = samples

        return outputs

    def generate(
        self,
        scenario: torch.Tensor,
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Generate combined samples using ensemble method.

        Args:
            scenario: Scenario labels
            n_samples: Number of samples
            temperature: Sampling temperature

        Returns:
            Generated sequences
        """
        if isinstance(scenario, int):
            scenario = torch.tensor([scenario], device=self._get_device())

        batch_size = len(scenario)
        scenario = scenario.repeat(n_samples)

        # Get all model outputs
        all_outputs = self.generate_all(scenario, n_samples=1, temperature=temperature)

        # Stack outputs: (n_models, batch, seq, vars)
        model_outputs = torch.stack(list(all_outputs.values()), dim=0)

        # Apply ensemble method
        if self.config.method == EnsembleMethod.AVERAGE:
            combined = model_outputs.mean(dim=0)

        elif self.config.method == EnsembleMethod.WEIGHTED:
            # Get weights for each scenario (softmax normalized)
            weights = F.softmax(self.weights, dim=-1)  # (n_scenarios, n_models)

            # Gather weights for batch scenarios
            scenario_weights = weights[scenario]  # (batch, n_models)

            # Apply weights: (batch, n_models) -> (n_models, batch, 1, 1)
            w = scenario_weights.permute(1, 0).unsqueeze(-1).unsqueeze(-1)

            combined = (model_outputs * w).sum(dim=0)

        elif self.config.method == EnsembleMethod.STACKING:
            # Flatten and concatenate
            flat_outputs = model_outputs.view(self.n_models, -1, self.config.n_macro_vars * self.config.seq_length)
            concat_flat = flat_outputs.permute(1, 0, 2).reshape(-1, self.n_models * self.config.n_macro_vars * self.config.seq_length)

            # Meta-learner
            combined_flat = self.meta_learner(concat_flat)
            combined = combined_flat.view(-1, self.config.seq_length, self.config.n_macro_vars)

        elif self.config.method == EnsembleMethod.SELECTION:
            # Select best model per scenario
            scores = F.softmax(self.selection_scores, dim=-1)
            best_model_idx = scores.argmax(dim=-1)

            # Gather best model output for each sample
            batch_best = best_model_idx[scenario]  # (batch,)
            combined = torch.zeros_like(model_outputs[0])

            for i in range(self.n_models):
                mask = (batch_best == i).view(-1, 1, 1)
                combined = combined + model_outputs[i] * mask.float()

        else:
            combined = model_outputs.mean(dim=0)

        return combined

    def _get_device(self) -> torch.device:
        """Get device from any available model"""
        for _, model in self._get_active_models():
            return next(model.parameters()).device
        return torch.device('cpu')

    def compute_disagreement(
        self,
        scenario: torch.Tensor,
        n_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        Compute model disagreement as uncertainty measure.

        Args:
            scenario: Scenario labels
            n_samples: Number of samples for estimation

        Returns:
            Disagreement metrics
        """
        # Generate from all models
        all_outputs = self.generate_all(scenario, n_samples=n_samples)

        # Stack: (n_models, n_samples * batch, seq, vars)
        model_outputs = torch.stack(list(all_outputs.values()), dim=0)

        # Compute statistics
        mean_output = model_outputs.mean(dim=0)
        var_between = model_outputs.var(dim=0)  # Between-model variance

        # Model-wise means
        model_means = model_outputs.mean(dim=1)  # (n_models, seq, vars)

        # Pairwise disagreement
        n_models = len(all_outputs)
        pairwise_diff = torch.zeros(n_models, n_models)
        model_names = list(all_outputs.keys())

        for i in range(n_models):
            for j in range(i + 1, n_models):
                diff = (model_means[i] - model_means[j]).abs().mean()
                pairwise_diff[i, j] = diff
                pairwise_diff[j, i] = diff

        return {
            'mean': mean_output,
            'variance': var_between,
            'pairwise_disagreement': pairwise_diff,
            'model_names': model_names
        }

    def calibrate_weights(
        self,
        val_data: torch.Tensor,
        val_labels: torch.Tensor,
        n_iter: int = 100
    ):
        """
        Calibrate ensemble weights on validation data.

        Args:
            val_data: Validation sequences (n, seq, vars)
            val_labels: Scenario labels (n,)
            n_iter: Optimization iterations
        """
        if self.config.method not in [EnsembleMethod.WEIGHTED, EnsembleMethod.SELECTION]:
            return

        device = self._get_device()
        val_data = val_data.to(device)
        val_labels = val_labels.to(device)

        # Only optimize weights
        if self.config.method == EnsembleMethod.WEIGHTED:
            optimizer = torch.optim.Adam([self.weights], lr=0.01)
        else:
            optimizer = torch.optim.Adam([self.selection_scores], lr=0.01)

        for _ in range(n_iter):
            # Generate samples
            generated = self.generate(val_labels, n_samples=1)

            # MSE loss
            loss = F.mse_loss(generated, val_data)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# =============================================================================
# TRAINING
# =============================================================================

class EnsembleTrainer:
    """Trainer for Ensemble stacking"""

    def __init__(
        self,
        ensemble: MacroEnsemble,
        config: EnsembleConfig,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.ensemble = ensemble.to(device)
        self.config = config
        self.device = device

        # Only train meta-learner if stacking
        if config.method == EnsembleMethod.STACKING:
            self.optimizer = torch.optim.Adam(
                ensemble.meta_learner.parameters(),
                lr=config.learning_rate
            )
        elif config.method in [EnsembleMethod.WEIGHTED, EnsembleMethod.SELECTION]:
            params = [ensemble.weights] if config.method == EnsembleMethod.WEIGHTED else [ensemble.selection_scores]
            self.optimizer = torch.optim.Adam(params, lr=config.learning_rate)
        else:
            self.optimizer = None

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""

        if self.optimizer is None:
            return {'loss': 0}

        self.ensemble.train()
        total_loss = 0
        n_batches = 0

        for batch_x, batch_scenario in dataloader:
            batch_x = batch_x.to(self.device)
            batch_scenario = batch_scenario.to(self.device)

            # Forward
            generated = self.ensemble.generate(batch_scenario, n_samples=1)

            # Loss
            loss = F.mse_loss(generated, batch_x)

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return {'loss': total_loss / n_batches}

    def train(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """Full training loop"""

        history = {'loss': []}

        for epoch in range(self.config.n_epochs):
            losses = self.train_epoch(dataloader)
            history['loss'].append(losses['loss'])

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.n_epochs} | Loss: {losses['loss']:.4f}")

        return history


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_ensemble(
    config: EnsembleConfig,
    vae_path: Optional[str] = None,
    gan_path: Optional[str] = None,
    flow_path: Optional[str] = None,
    device: str = 'cpu'
) -> MacroEnsemble:
    """
    Create ensemble from saved model checkpoints.

    Args:
        config: Ensemble configuration
        vae_path: Path to VAE checkpoint
        gan_path: Path to GAN checkpoint
        flow_path: Path to Flow checkpoint
        device: Device to load models on

    Returns:
        MacroEnsemble instance
    """
    from privatecredit.models.macro_vae import MacroVAE, MacroVAEConfig
    from privatecredit.models.macro_gan import MacroGAN, MacroGANConfig
    from privatecredit.models.macro_flow import MacroFlow, MacroFlowConfig

    vae_model = None
    gan_model = None
    flow_model = None

    # Load VAE
    if vae_path and config.use_vae:
        vae_config = MacroVAEConfig(
            n_macro_vars=config.n_macro_vars,
            seq_length=config.seq_length,
            n_scenarios=config.n_scenarios
        )
        vae_model = MacroVAE(vae_config)
        vae_model.load_state_dict(torch.load(vae_path, map_location=device))
        vae_model.eval()

    # Load GAN
    if gan_path and config.use_gan:
        gan_config = MacroGANConfig(
            n_macro_vars=config.n_macro_vars,
            seq_length=config.seq_length,
            n_scenarios=config.n_scenarios
        )
        gan_model = MacroGAN(gan_config)
        gan_model.load_state_dict(torch.load(gan_path, map_location=device))
        gan_model.eval()

    # Load Flow
    if flow_path and config.use_flow:
        flow_config = MacroFlowConfig(
            n_macro_vars=config.n_macro_vars,
            seq_length=config.seq_length,
            n_scenarios=config.n_scenarios
        )
        flow_model = MacroFlow(flow_config)
        flow_model.load_state_dict(torch.load(flow_path, map_location=device))
        flow_model.eval()

    ensemble = MacroEnsemble(
        config=config,
        vae_model=vae_model,
        gan_model=gan_model,
        flow_model=flow_model
    )

    return ensemble.to(device)


def compare_models(
    ensemble: MacroEnsemble,
    scenario_type: str,
    n_samples: int = 1000
) -> Dict[str, np.ndarray]:
    """
    Compare generated distributions across models.

    Args:
        ensemble: Ensemble model
        scenario_type: Scenario to compare
        n_samples: Number of samples

    Returns:
        Statistics for each model
    """
    scenario_map = {
        'baseline': 0,
        'adverse': 1,
        'severely_adverse': 2,
        'stagflation': 3
    }
    scenario_idx = scenario_map.get(scenario_type, 0)
    scenario_tensor = torch.tensor([scenario_idx], device=ensemble._get_device())

    # Generate from all models
    all_outputs = ensemble.generate_all(scenario_tensor, n_samples=n_samples)

    # Compute statistics
    stats = {}
    for name, samples in all_outputs.items():
        samples_np = samples.cpu().numpy()
        stats[name] = {
            'mean': samples_np.mean(axis=0),
            'std': samples_np.std(axis=0),
            'q05': np.percentile(samples_np, 5, axis=0),
            'q95': np.percentile(samples_np, 95, axis=0),
            'skew': ((samples_np - samples_np.mean(axis=0)) ** 3).mean(axis=0) / (samples_np.std(axis=0) ** 3 + 1e-8),
            'kurtosis': ((samples_np - samples_np.mean(axis=0)) ** 4).mean(axis=0) / (samples_np.std(axis=0) ** 4 + 1e-8) - 3
        }

    return stats


# =============================================================================
# MAIN
# =============================================================================

# Import F for softmax
import torch.nn.functional as F


def main():
    """Demonstrate ensemble model"""

    print("=" * 60)
    print("MACRO ENSEMBLE - COMBINING VAE, GAN, AND FLOW")
    print("=" * 60)

    # Create simple mock models for demonstration
    from privatecredit.models.macro_vae import MacroVAE, MacroVAEConfig, MacroDataset
    from privatecredit.models.macro_gan import MacroGAN, MacroGANConfig
    from privatecredit.models.macro_flow import MacroFlow, MacroFlowConfig

    config = EnsembleConfig(
        n_macro_vars=9,
        seq_length=60,
        n_scenarios=4,
        method=EnsembleMethod.WEIGHTED
    )

    # Create component models (untrained for demo)
    vae_config = MacroVAEConfig(n_macro_vars=9, seq_length=60, n_scenarios=4)
    gan_config = MacroGANConfig(n_macro_vars=9, seq_length=60, n_scenarios=4)
    flow_config = MacroFlowConfig(n_macro_vars=9, seq_length=60, n_scenarios=4)

    vae = MacroVAE(vae_config)
    gan = MacroGAN(gan_config)
    flow = MacroFlow(flow_config)

    # Create ensemble
    ensemble = MacroEnsemble(
        config=config,
        vae_model=vae,
        gan_model=gan,
        flow_model=flow
    )

    print(f"\nEnsemble with {ensemble.n_models} models")
    print(f"Method: {config.method.value}")

    # Generate samples
    print("\nGenerating baseline scenarios...")
    scenario = torch.tensor([0])  # baseline

    samples = ensemble.generate(scenario, n_samples=10)
    print(f"Generated shape: {samples.shape}")

    # Compute disagreement
    print("\nComputing model disagreement...")
    disagreement = ensemble.compute_disagreement(scenario, n_samples=100)
    print(f"Between-model variance (mean): {disagreement['variance'].mean():.4f}")
    print(f"Pairwise disagreement matrix:\n{disagreement['pairwise_disagreement']}")

    print("\nEnsemble ready for use!")


if __name__ == '__main__':
    main()

"""
Model Trainer for ISRO PS14 Radiation Forecasting.

Full training pipeline with:
- Mixed precision (AMP) for RTX 5060
- CosineAnnealingWarmRestarts scheduler
- Early stopping with patience
- Gradient clipping
- Best model checkpointing
- Training/validation loss logging
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from pathlib import Path
from typing import Optional, Dict, Tuple, List

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains and validates forecasting models with production-grade features.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Config] = None,
        model_name: str = 'model',
    ):
        """
        Args:
            model: PyTorch model to train.
            config: Configuration object.
            model_name: Name for checkpointing (e.g., 'transformer', 'lstm').
        """
        self.config = config or Config()
        self.model = model
        self.model_name = model_name
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        self.model = self.model.to(self.device)
        
        # Training params
        train_cfg = self.config.training
        self.lr = train_cfg.get('learning_rate', 1e-4)
        self.weight_decay = train_cfg.get('weight_decay', 1e-5)
        self.max_epochs = train_cfg.get('max_epochs', 200)
        self.patience = train_cfg.get('patience', 15)
        self.grad_clip = train_cfg.get('grad_clip_norm', 1.0)
        self.use_amp = train_cfg.get('use_amp', True) and self.device.type == 'cuda'
        
        # Loss function: MSE on log10(flux)
        self.criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        sched_cfg = train_cfg.get('scheduler', {})
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=sched_cfg.get('T_0', 10),
            T_mult=sched_cfg.get('T_mult', 2),
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        
        # Checkpointing
        self.checkpoint_dir = Path(train_cfg.get('checkpoint_dir', 'models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = str(
            self.checkpoint_dir / f'best_{model_name}.pt'
        )
        
        # Training state
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training DataLoader.
        
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (X, Y) in enumerate(train_loader):
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with autocast('cuda', enabled=self.use_amp):
                predictions = self.model(X)
                loss = self.criterion(predictions, Y)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: Validation DataLoader.
        
        Returns:
            Tuple of (average loss, per-horizon metrics dict).
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_targets = []
        
        for X, Y in val_loader:
            X = X.to(self.device, non_blocking=True)
            Y = Y.to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=self.use_amp):
                predictions = self.model(X)
                loss = self.criterion(predictions, Y)
            
            total_loss += loss.item()
            num_batches += 1
            
            all_preds.append(predictions.cpu().numpy())
            all_targets.append(Y.cpu().numpy())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Per-horizon metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        horizon_labels = self.config.features.get(
            'target_horizons_labels', ['30min', '6h', '12h']
        )
        
        metrics = {}
        for i, label in enumerate(horizon_labels[:all_preds.shape[1]]):
            pred_i = all_preds[:, i]
            true_i = all_targets[:, i]
            
            mse = np.mean((pred_i - true_i) ** 2)
            rmse = np.sqrt(mse)
            
            # Prediction Efficiency
            ss_res = np.sum((true_i - pred_i) ** 2)
            ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
            pe = 1 - ss_res / max(ss_tot, 1e-10)
            
            # Pearson correlation
            corr = np.corrcoef(true_i, pred_i)[0, 1] if len(true_i) > 1 else 0
            
            metrics[label] = {
                'RMSE_log': rmse,
                'PE': pe,
                'R': corr,
            }
        
        return avg_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
        
        Returns:
            Dictionary with training history.
        """
        logger.info("=" * 60)
        logger.info(f"Training {self.model_name} model")
        logger.info(f"  Max epochs: {self.max_epochs}")
        logger.info(f"  Patience: {self.patience}")
        logger.info(f"  LR: {self.lr}")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        for epoch in range(1, self.max_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            # Scheduler step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            self.learning_rates.append(current_lr)
            
            # Logging
            epoch_time = time.time() - epoch_start
            metrics_str = " | ".join(
                f"{h}: PE={m['PE']:.3f}, R={m['R']:.3f}"
                for h, m in val_metrics.items()
            )
            
            logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
                f"LR: {current_lr:.2e} | {epoch_time:.1f}s | "
                f"{metrics_str}"
            )
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, val_loss, val_metrics)
                logger.info(f"  * New best model saved! (val_loss={val_loss:.5f})")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    logger.info(
                        f"\nEarly stopping at epoch {epoch} "
                        f"(no improvement for {self.patience} epochs)"
                    )
                    break
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining complete in {total_time/60:.1f} minutes")
        logger.info(f"Best validation loss: {self.best_val_loss:.5f}")
        
        # Load best model
        self.load_best_model()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'best_val_loss': self.best_val_loss,
            'total_epochs': len(self.train_losses),
            'total_time_minutes': total_time / 60,
        }
    
    def _save_checkpoint(self, epoch: int, val_loss: float, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics': metrics,
            'model_name': self.model_name,
        }
        torch.save(checkpoint, self.best_model_path)
    
    def load_best_model(self):
        """Load the best model checkpoint."""
        if os.path.exists(self.best_model_path):
            checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', '?')}")
        else:
            logger.warning(f"No checkpoint found at {self.best_model_path}")
    
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for all data in a DataLoader.
        
        Args:
            data_loader: DataLoader to predict on.
        
        Returns:
            Tuple of (predictions, targets, uncertainties) as numpy arrays.
            uncertainties is a dict with 'p5' and 'p95' bounds.
        """
        self.model.eval()
        
        # Enable dropout for MC Dropout uncertainty
        use_mc_dropout = True
        num_samples = 20
        if use_mc_dropout:
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        
        all_preds = []
        all_targets = []
        all_preds_mc = []
        
        for X, Y in data_loader:
            X = X.to(self.device, non_blocking=True)
            
            with autocast('cuda', enabled=self.use_amp):
                # Standard mean prediction
                self.model.eval() # ensure non-dropout are eval
                pred = self.model(X)
                
                # MC Dropout predictions
                if use_mc_dropout:
                    for m in self.model.modules():
                        if m.__class__.__name__.startswith('Dropout'):
                            m.train()
                    
                    mc_samples = []
                    for _ in range(num_samples):
                        mc_samples.append(self.model(X).detach().cpu().numpy())
                    all_preds_mc.append(np.stack(mc_samples, axis=0))
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(Y.numpy())
        
        predictions = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        uncertainties = {}
        if use_mc_dropout:
            # shape: (samples, batches*batch_size, horizons)
            mc_preds = np.concatenate(all_preds_mc, axis=1)
            uncertainties['p5'] = np.percentile(mc_preds, 5, axis=0)
            uncertainties['p95'] = np.percentile(mc_preds, 95, axis=0)
            uncertainties['std'] = np.std(mc_preds, axis=0)
            
        return predictions, targets, uncertainties

    @torch.no_grad()
    def compute_feature_importance(self, data_loader: DataLoader, feature_names: List[str]) -> Dict[str, float]:
        """
        Compute Permutation Feature Importance.
        """
        self.model.eval()
        
        # 1. Get baseline loss
        baseline_preds, targets, _ = self.predict(data_loader)
        # Using MSE of the first horizon as the scoring metric
        baseline_mse = np.mean((baseline_preds[:, 0] - targets[:, 0])**2)
        
        importance = {}
        for feat_idx, feat_name in enumerate(feature_names):
            all_preds_shuffled = []
            
            for X, Y in data_loader:
                # Shuffle feature across the batch dimension
                X_shuffled = X.clone()
                batch_size = X.size(0)
                perm = torch.randperm(batch_size)
                X_shuffled[:, :, feat_idx] = X_shuffled[perm, :, feat_idx]
                
                X_shuffled = X_shuffled.to(self.device, non_blocking=True)
                with autocast('cuda', enabled=self.use_amp):
                    pred = self.model(X_shuffled)
                all_preds_shuffled.append(pred.cpu().numpy())
            
            preds_shuffled = np.concatenate(all_preds_shuffled, axis=0)
            shuffled_mse = np.mean((preds_shuffled[:, 0] - targets[:, 0])**2)
            
            # Importance is the INCREASE in error (higher = more important)
            importance[feat_name] = float(shuffled_mse - baseline_mse)
            
        # Normalize to percentages
        total_importance = sum(v for v in importance.values() if v > 0)
        if total_importance > 0:
            importance = {k: (v / total_importance * 100 if v > 0 else 0.0) for k, v in importance.items()}
            
        return importance


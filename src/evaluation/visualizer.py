"""
Visualization module for ISRO PS14 Radiation Forecasting.

Creates publication-quality plots for:
- Data exploration (time series, distributions, correlations)
- Model predictions vs observations
- Performance metrics comparison
- Storm case studies
- GRASP comparison
- Training history
"""
# Force IDE cache clear

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)

# Publication-quality settings
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color scheme
COLORS = {
    'observed': '#2196F3',
    'predicted': '#FF5722',
    'lstm': '#4CAF50',
    'transformer': '#FF5722',
    'confidence': '#FFE0B2',
    'storm': '#F44336',
    'quiet': '#4CAF50',
    'grasp': '#9C27B0',
}


class Visualizer:
    """Creates all visualization outputs for the project."""
    
    def __init__(self, config: Optional[Config] = None, output_dir: str = None):
        self.config = config or Config()
        self.output_dir = Path(output_dir or str(self.config.root / 'outputs' / 'plots'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _save(self, fig, name: str):
        """Save figure and close."""
        path = self.output_dir / f'{name}.png'
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        logger.info(f"Saved plot: {path}")
        return str(path)
    
    # =========================================================================
    # Data Exploration Plots
    # =========================================================================
    
    def plot_timeseries_overview(
        self,
        df: pd.DataFrame,
        flux_col: str = 'electron_flux_gt2MeV'
    ) -> str:
        """
        Multi-panel time-series overview of all data.
        
        Shows electron flux, solar wind speed, IMF Bz, density,
        and geomagnetic indices stacked vertically.
        """
        fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True)
        fig.suptitle('11-Year Data Overview: Electron Flux & Solar Wind Parameters',
                     fontsize=14, fontweight='bold')
        
        # Panel 1: Electron flux
        if flux_col in df.columns:
            axes[0].semilogy(df.index, df[flux_col], color=COLORS['observed'],
                           linewidth=0.3, alpha=0.7)
            axes[0].set_ylabel('>2 MeV Electron Flux\n(pfu)', color=COLORS['observed'])
            axes[0].set_title('GOES >2 MeV Integral Electron Flux')
            axes[0].axhline(y=1000, color='orange', linestyle='--', alpha=0.7, label='Warning (1000 pfu)')
            axes[0].legend(loc='upper right')
        
        # Panel 2: Solar wind speed
        if 'Vsw' in df.columns:
            axes[1].plot(df.index, df['Vsw'], color='#E91E63',
                        linewidth=0.3, alpha=0.7)
            axes[1].set_ylabel('Vsw (km/s)', color='#E91E63')
            axes[1].set_title('Solar Wind Speed')
            axes[1].axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='HSS threshold')
            axes[1].legend(loc='upper right')
        
        # Panel 3: IMF Bz
        bz_col = 'Bz_GSM' if 'Bz_GSM' in df.columns else 'Bz_GSE'
        if bz_col in df.columns:
            axes[2].plot(df.index, df[bz_col], color='#00BCD4',
                        linewidth=0.3, alpha=0.7)
            axes[2].axhline(y=0, color='gray', linewidth=0.5)
            axes[2].set_ylabel('IMF Bz (nT)', color='#00BCD4')
            axes[2].set_title('Interplanetary Magnetic Field Bz (GSM)')
            axes[2].fill_between(df.index, df[bz_col], 0,
                               where=df[bz_col] < 0, alpha=0.2, color='red',
                               label='Southward (geoeffective)')
            axes[2].legend(loc='upper right')
        
        # Panel 4: Density
        if 'Np' in df.columns:
            axes[3].plot(df.index, df['Np'], color='#FF9800',
                        linewidth=0.3, alpha=0.7)
            axes[3].set_ylabel('Np (cm⁻³)', color='#FF9800')
            axes[3].set_title('Solar Wind Proton Density')
        
        # Panel 5: Dst index
        if 'Dst' in df.columns:
            axes[4].plot(df.index, df['Dst'], color='#673AB7',
                        linewidth=0.3, alpha=0.7)
            axes[4].axhline(y=-50, color='red', linestyle='--', alpha=0.5,
                           label='Storm threshold')
            axes[4].set_ylabel('Dst (nT)', color='#673AB7')
            axes[4].set_title('Disturbance Storm-Time Index')
            axes[4].legend(loc='upper right')
        
        axes[-1].set_xlabel('Date')
        axes[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        return self._save(fig, 'timeseries_overview')
    
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> str:
        """Feature correlation heatmap."""
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cols_to_plot = [c for c in numeric_cols if df[c].notna().mean() > 0.5][:20]
        
        corr = df[cols_to_plot].corr()
        
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, ax=ax,
                    square=True, linewidths=0.5)
        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return self._save(fig, 'correlation_heatmap')
    
    def plot_flux_distribution(
        self,
        df: pd.DataFrame,
        flux_col: str = 'electron_flux_gt2MeV'
    ) -> str:
        """Histogram of log10(flux) showing log-normal distribution."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        if flux_col in df.columns:
            flux = df[flux_col].dropna()
            log_flux = np.log10(flux.clip(lower=1e-10))
            
            # Linear scale histogram
            ax1.hist(flux, bins=100, color=COLORS['observed'], alpha=0.7, edgecolor='white')
            ax1.set_xlabel('Electron Flux (pfu)')
            ax1.set_ylabel('Count')
            ax1.set_title('Flux Distribution (Linear Scale)')
            ax1.set_xscale('log')
            
            # Log scale histogram
            ax2.hist(log_flux, bins=100, color=COLORS['observed'], alpha=0.7, edgecolor='white')
            ax2.set_xlabel('log₁₀(Electron Flux)')
            ax2.set_ylabel('Count')
            ax2.set_title('Flux Distribution (Log₁₀ Scale)')
            
            # Annotate statistics
            ax2.axvline(log_flux.mean(), color='red', linestyle='--',
                       label=f'Mean: {log_flux.mean():.2f}')
            ax2.axvline(log_flux.median(), color='green', linestyle='--',
                       label=f'Median: {log_flux.median():.2f}')
            ax2.legend()
        
        fig.suptitle('>2 MeV Electron Flux Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, 'flux_distribution')
    
    def plot_data_coverage(self, df: pd.DataFrame) -> str:
        """Data quality / coverage heatmap by year and month."""
        # Compute validity fraction per month
        monthly_valid = df.resample('ME').apply(lambda x: x.notna().mean())
        
        # Select key columns
        key_cols = [c for c in ['electron_flux_gt2MeV', 'Vsw', 'Bz_GSE', 'Bz_GSM', 'Np', 'Dst', 'AE'] 
                   if c in monthly_valid.columns]
        
        if not key_cols:
            key_cols = monthly_valid.columns[:6].tolist()
        
        fig, ax = plt.subplots(figsize=(16, max(4, len(key_cols) * 0.8)))
        
        data = monthly_valid[key_cols].T
        data.columns = [f"{d.year}-{d.month:02d}" for d in data.columns]
        
        # Sample columns to avoid overcrowding
        if len(data.columns) > 36:
            step = max(1, len(data.columns) // 36)
            data = data.iloc[:, ::step]
        
        sns.heatmap(data, cmap='RdYlGn', vmin=0, vmax=1, ax=ax,
                    annot=False, linewidths=0.5)
        ax.set_title('Data Coverage (fraction of valid values per month)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Month')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return self._save(fig, 'data_coverage')
    
    # =========================================================================
    # Model Prediction Plots
    # =========================================================================
    
    def plot_predictions_timeseries(
        self,
        timestamps: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_labels: List[str] = None,
        model_name: str = 'Transformer',
        max_days: int = 60,
    ) -> str:
        """
        Predicted vs observed time-series for each forecast horizon.
        """
        if horizon_labels is None:
            horizon_labels = ['30min', '6h', '12h']
        
        n_horizons = min(y_true.shape[1], len(horizon_labels))
        fig, axes = plt.subplots(n_horizons, 1, figsize=(16, 4 * n_horizons), sharex=True)
        
        if n_horizons == 1:
            axes = [axes]
        
        fig.suptitle(f'{model_name} — Predicted vs Observed Electron Flux',
                     fontsize=14, fontweight='bold')
        
        # Limit display range
        if len(timestamps) > max_days * 288:  # 288 = 5-min steps per day
            step = len(timestamps) // (max_days * 288)
            idx = slice(None, None, max(step, 1))
        else:
            idx = slice(None)
        
        for i, (ax, label) in enumerate(zip(axes, horizon_labels)):
            if i >= y_true.shape[1]:
                break
            
            t = timestamps[idx] if isinstance(timestamps, np.ndarray) else timestamps
            true_vals = y_true[idx, i] if isinstance(idx, slice) else y_true[:, i]
            pred_vals = y_pred[idx, i] if isinstance(idx, slice) else y_pred[:, i]
            
            ax.plot(t[:len(true_vals)], true_vals, color=COLORS['observed'],
                   linewidth=0.5, alpha=0.7, label='Observed')
            ax.plot(t[:len(pred_vals)], pred_vals, color=COLORS['predicted'],
                   linewidth=0.5, alpha=0.7, label='Predicted')
            
            ax.set_ylabel(f'log₁₀(Flux)\n{label} ahead')
            ax.legend(loc='upper right')
            ax.set_title(f'Forecast Horizon: {label}')
        
        axes[-1].set_xlabel('Date')
        plt.tight_layout()
        return self._save(fig, f'predictions_timeseries_{model_name.lower()}')
    
    def plot_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_labels: List[str] = None,
        model_name: str = 'Transformer',
    ) -> str:
        """
        Scatter plot of predicted vs observed (log scale) with 1:1 line.
        """
        if horizon_labels is None:
            horizon_labels = ['30min', '6h', '12h']
        
        n_horizons = min(y_true.shape[1], len(horizon_labels))
        fig, axes = plt.subplots(1, n_horizons, figsize=(5 * n_horizons, 5))
        
        if n_horizons == 1:
            axes = [axes]
        
        fig.suptitle(f'{model_name} — Scatter: Predicted vs Observed',
                     fontsize=14, fontweight='bold')
        
        for i, (ax, label) in enumerate(zip(axes, horizon_labels)):
            if i >= y_true.shape[1]:
                break
            
            true_i = y_true[:, i]
            pred_i = y_pred[:, i]
            mask = np.isfinite(true_i) & np.isfinite(pred_i)
            
            ax.scatter(true_i[mask], pred_i[mask], alpha=0.05, s=1,
                      color=COLORS['predicted'], rasterized=True)
            
            # 1:1 line
            lims = [min(true_i[mask].min(), pred_i[mask].min()),
                    max(true_i[mask].max(), pred_i[mask].max())]
            ax.plot(lims, lims, 'k--', linewidth=1, alpha=0.5, label='1:1')
            
            # Stats
            r = np.corrcoef(true_i[mask], pred_i[mask])[0, 1]
            ax.text(0.05, 0.95, f'R = {r:.4f}', transform=ax.transAxes,
                   verticalalignment='top', fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Observed log₁₀(Flux)')
            ax.set_ylabel('Predicted log₁₀(Flux)')
            ax.set_title(f'{label} Forecast')
            ax.set_aspect('equal')
            ax.legend()
        
        plt.tight_layout()
        return self._save(fig, f'scatter_{model_name.lower()}')
    
    def plot_metrics_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, Dict[str, float]]],
        metric_name: str = 'PE',
    ) -> str:
        """
        Bar chart comparing metrics across models and horizons.
        
        Args:
            metrics_dict: {model_name: {horizon: {metric: value}}}
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(metrics_dict.keys())
        horizons = list(next(iter(metrics_dict.values())).keys())
        
        x = np.arange(len(horizons))
        width = 0.8 / len(models)
        
        colors = [COLORS.get('transformer', '#FF5722'), COLORS.get('lstm', '#4CAF50'),
                  '#2196F3', '#FF9800']
        
        for i, model in enumerate(models):
            values = [metrics_dict[model].get(h, {}).get(metric_name, 0) for h in horizons]
            ax.bar(x + i * width, values, width, label=model,
                   color=colors[i % len(colors)], alpha=0.8)
        
        ax.set_xlabel('Forecast Horizon')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} by Model and Forecast Horizon',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(horizons)
        ax.legend()
        
        if metric_name == 'PE':
            ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Target (0.8)')
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return self._save(fig, f'metrics_comparison_{metric_name.lower()}')
    
    def plot_training_history(
        self,
        history: Dict,
        model_name: str = 'Transformer'
    ) -> str:
        """Plot training and validation loss curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_losses'], label='Train', color=COLORS['observed'])
        ax1.plot(epochs, history['val_losses'], label='Validation', color=COLORS['predicted'])
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title(f'{model_name} — Training History')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Learning rate
        ax2.plot(epochs, history['learning_rates'], color='#4CAF50')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        
        fig.suptitle(f'{model_name} Training', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self._save(fig, f'training_history_{model_name.lower()}')
    
    def plot_storm_case_study(
        self,
        df: pd.DataFrame,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: np.ndarray,
        storm_start: str,
        storm_end: str,
        storm_name: str = 'Geomagnetic Storm',
        horizon_idx: int = 0,
    ) -> str:
        """
        Detailed case study plot for a specific storm event.
        
        Shows electron flux prediction overlaid with solar wind drivers
        during the storm period.
        """
        # Find indices within storm period
        start = np.datetime64(storm_start)
        end = np.datetime64(storm_end)
        mask = (timestamps >= start) & (timestamps <= end)
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1])
        
        # Panel 1: Electron flux prediction
        ax1 = fig.add_subplot(gs[0])
        t = timestamps[mask]
        ax1.plot(t, y_true[mask, horizon_idx], color=COLORS['observed'],
                linewidth=1.5, label='Observed', alpha=0.8)
        ax1.plot(t, y_pred[mask, horizon_idx], color=COLORS['predicted'],
                linewidth=1.5, label='Predicted', alpha=0.8)
        ax1.set_ylabel('log₁₀(Flux)')
        ax1.set_title(f'Storm Case Study: {storm_name}', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.fill_between(t, y_true[mask, horizon_idx], y_pred[mask, horizon_idx],
                         alpha=0.15, color='gray')
        
        # Panel 2: Solar wind
        storm_df = df.loc[storm_start:storm_end] if isinstance(df.index, pd.DatetimeIndex) else df
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        if 'Vsw' in storm_df.columns:
            ax2.plot(storm_df.index, storm_df['Vsw'], color='#E91E63', linewidth=1)
            ax2.set_ylabel('Vsw (km/s)')
        
        # Panel 3: Dst
        ax3 = fig.add_subplot(gs[2], sharex=ax1)
        if 'Dst' in storm_df.columns:
            ax3.plot(storm_df.index, storm_df['Dst'], color='#673AB7', linewidth=1)
            ax3.axhline(y=-50, color='red', linestyle='--', alpha=0.5)
            ax3.set_ylabel('Dst (nT)')
        
        ax3.set_xlabel('Date')
        plt.tight_layout()
        return self._save(fig, f'storm_case_{storm_name.replace(" ", "_").lower()}')
    
    def plot_grasp_comparison(
        self,
        goes_pred: np.ndarray,
        grasp_obs: np.ndarray,
        timestamps: np.ndarray,
    ) -> str:
        """
        Compare model predictions (trained on GOES) with GRASP observations.
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
        
        fig.suptitle('GOES-based Prediction vs GRASP/GSAT-19 Observation',
                     fontsize=14, fontweight='bold')
        
        # Time series comparison
        ax1.plot(timestamps, goes_pred, color=COLORS['predicted'],
                linewidth=0.8, label='Model Prediction (GOES-based)', alpha=0.7)
        ax1.plot(timestamps, grasp_obs, color=COLORS['grasp'],
                linewidth=0.8, label='GRASP Observation', alpha=0.7)
        ax1.set_ylabel('log₁₀(Flux)')
        ax1.set_title('Time Series Comparison')
        ax1.legend()
        
        # Scatter
        mask = np.isfinite(goes_pred) & np.isfinite(grasp_obs)
        ax2.scatter(grasp_obs[mask], goes_pred[mask], alpha=0.1, s=2,
                   color=COLORS['grasp'], rasterized=True)
        lims = [min(grasp_obs[mask].min(), goes_pred[mask].min()),
                max(grasp_obs[mask].max(), goes_pred[mask].max())]
        ax2.plot(lims, lims, 'k--', linewidth=1)
        ax2.set_xlabel('GRASP Observed log₁₀(Flux)')
        ax2.set_ylabel('Model Predicted log₁₀(Flux)')
        ax2.set_title('Scatter: Model vs GRASP')
        ax2.set_aspect('equal')
        
        r = np.corrcoef(grasp_obs[mask], goes_pred[mask])[0, 1]
        ax2.text(0.05, 0.95, f'R = {r:.4f}', transform=ax2.transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return self._save(fig, 'grasp_comparison')
    
    def plot_feature_importance(
        self,
        importance: Dict[str, float],
        model_name: str = 'Model',
    ) -> str:
        """
        Plot permutation feature importance as a horizontal bar chart.
        
        Args:
            importance: Dict of {feature_name: importance_score}.
            model_name: Name of the model for the title.
        
        Returns:
            Path to saved plot.
        """
        # Sort by importance
        sorted_items = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items[:15]]  # Top 15
        scores = [item[1] for item in sorted_items[:15]]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
        bars = ax.barh(range(len(names)), scores, color=colors)
        
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance (%)', fontsize=12)
        ax.set_title(f'Feature Importance — {model_name}\n(Permutation-based, higher = more important)',
                     fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{score:.1f}%', va='center', fontsize=9)
        
        ax.set_facecolor('#1a1a2e')
        fig.patch.set_facecolor('#0d1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')
        
        plt.tight_layout()
        return self._save(fig, f'feature_importance_{model_name.lower()}')
    
    def plot_uncertainty_bands(
        self,
        timestamps: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        p5: np.ndarray,
        p95: np.ndarray,
        horizon_labels: List[str],
        model_name: str = 'Model',
    ) -> str:
        """
        Plot predictions with 90% confidence interval from MC Dropout.
        
        Args:
            timestamps: Array of datetime timestamps.
            y_true: Observed values (N, num_horizons).
            y_pred: Predicted values (N, num_horizons).
            p5: 5th percentile predictions (N, num_horizons).
            p95: 95th percentile predictions (N, num_horizons).
            horizon_labels: Label for each horizon.
            model_name: Name of the model for the title.
        
        Returns:
            Path to saved plot.
        """
        n_horizons = min(y_true.shape[1], len(horizon_labels))
        fig, axes = plt.subplots(n_horizons, 1, figsize=(16, 4 * n_horizons), sharex=True)
        if n_horizons == 1:
            axes = [axes]
        
        # Show a representative subset (last 2000 points)
        n_show = min(2000, len(timestamps))
        sl = slice(-n_show, None)
        t = timestamps[sl]
        
        for i, (ax, label) in enumerate(zip(axes, horizon_labels)):
            if i >= y_true.shape[1]:
                break
            
            true_vals = y_true[sl, i]
            pred_vals = y_pred[sl, i]
            lower = p5[sl, i]
            upper = p95[sl, i]
            
            ax.fill_between(t[:len(lower)], lower, upper,
                          alpha=0.25, color='#667eea', label='90% CI (MC Dropout)')
            ax.plot(t[:len(true_vals)], true_vals, color=COLORS['observed'],
                   linewidth=0.8, label='Observed', alpha=0.8)
            ax.plot(t[:len(pred_vals)], pred_vals, color=COLORS['predicted'],
                   linewidth=0.8, label='Predicted', alpha=0.8)
            
            ax.set_ylabel(f'log₁₀(Flux) — {label}', fontsize=11)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            ax.set_facecolor('#1a1a2e')
            ax.tick_params(colors='white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#333')
        
        fig.suptitle(f'{model_name} — Predictions with 90% Uncertainty Bounds',
                    fontsize=14, fontweight='bold', color='white')
        fig.patch.set_facecolor('#0d1117')
        plt.tight_layout()
        return self._save(fig, f'uncertainty_{model_name.lower()}')
    
    def create_all_eda_plots(self, df: pd.DataFrame) -> List[str]:
        """Generate all exploratory data analysis plots."""
        plots = []
        
        logger.info("Generating EDA plots...")
        plots.append(self.plot_timeseries_overview(df))
        plots.append(self.plot_flux_distribution(df))
        plots.append(self.plot_correlation_heatmap(df))
        plots.append(self.plot_data_coverage(df))
        
        logger.info(f"Generated {len(plots)} EDA plots")
        return plots


"""
Evaluation Metrics for ISRO PS14 Radiation Forecasting.

All metrics computed in log10 space (standard for radiation belt research).

Includes:
- Prediction Efficiency (PE)
- RMSE on log10(flux)
- Pearson correlation coefficient
- Mean Absolute Error (log space)
- Heidke Skill Score (categorical)
- Persistence and 27-day recurrence baselines
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ForecastMetrics:
    """
    Comprehensive evaluation metrics for electron flux forecasting.
    """
    
    @staticmethod
    def prediction_efficiency(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Prediction Efficiency (PE) — the standard space weather skill score.
        
        PE = 1 - Σ(obs - pred)² / Σ(obs - mean(obs))²
        
        Interpretation:
        - PE = 1.0: Perfect prediction
        - PE = 0.0: No better than predicting the mean (climatology)
        - PE < 0:   Worse than climatology
        
        Args:
            y_true: Observed values (log10 flux).
            y_pred: Predicted values (log10 flux).
        
        Returns:
            PE score.
        """
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1.0 - ss_res / ss_tot
    
    @staticmethod
    def rmse_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Square Error in log10 space.
        
        An RMSE(log10) of 0.3 means predictions are off by about
        a factor of 2 (10^0.3 ≈ 2), which is considered good.
        
        Args:
            y_true: Observed log10 flux.
            y_pred: Predicted log10 flux.
        
        Returns:
            RMSE in log10 units.
        """
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        return np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
    
    @staticmethod
    def mae_log(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error in log10 space.
        
        Args:
            y_true: Observed log10 flux.
            y_pred: Predicted log10 flux.
        
        Returns:
            MAE in log10 units.
        """
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        return mean_absolute_error(y_true[mask], y_pred[mask])
    
    @staticmethod
    def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Pearson correlation coefficient.
        
        Measures linear correlation between predictions and observations.
        R > 0.9 is considered excellent for flux forecasting.
        
        Args:
            y_true: Observed values.
            y_pred: Predicted values.
        
        Returns:
            Pearson R.
        """
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) < 2:
            return 0.0
        
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    
    @staticmethod
    def heidke_skill_score(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = None,
    ) -> float:
        """
        Heidke Skill Score for categorical forecast.
        
        Evaluates ability to predict flux exceedance above a threshold.
        Commonly used in space weather operations.
        
        HSS = 2(ad - bc) / ((a+c)(c+d) + (a+b)(b+d))
        
        where a=hits, b=false alarms, c=misses, d=correct negatives.
        
        If threshold is None, uses the 75th percentile of y_true as an
        adaptive threshold. This ensures a meaningful event/non-event
        split even when extreme events (>1000 pfu) are absent.
        
        Args:
            y_true: Observed log10 flux.
            y_pred: Predicted log10 flux.
            threshold: Log10 flux threshold. If None, uses 75th percentile.
        
        Returns:
            HSS value (-1 to 1, 1 is perfect).
        """
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) < 10:
            return 0.0
        
        # Adaptive threshold: use 75th percentile if not specified
        if threshold is None:
            threshold = float(np.percentile(y_true, 75))
        
        obs_exceed = y_true >= threshold
        pred_exceed = y_pred >= threshold
        
        a = np.sum(obs_exceed & pred_exceed)      # Hits
        b = np.sum(~obs_exceed & pred_exceed)     # False alarms
        c = np.sum(obs_exceed & ~pred_exceed)     # Misses
        d = np.sum(~obs_exceed & ~pred_exceed)    # Correct negatives
        
        denominator = (a + c) * (c + d) + (a + b) * (b + d)
        
        if denominator == 0:
            return 0.0
        
        return float(2 * (a * d - b * c) / denominator)
    
    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_label: str = "",
    ) -> Dict[str, float]:
        """
        Compute all metrics for a single forecast horizon.
        
        Args:
            y_true: Observed log10 flux values.
            y_pred: Predicted log10 flux values.
            horizon_label: Label for logging (e.g., '30min', '6h', '12h').
        
        Returns:
            Dictionary of all metrics.
        """
        metrics = {
            'PE': cls.prediction_efficiency(y_true, y_pred),
            'RMSE_log': cls.rmse_log(y_true, y_pred),
            'MAE_log': cls.mae_log(y_true, y_pred),
            'Pearson_R': cls.pearson_r(y_true, y_pred),
            'HSS': cls.heidke_skill_score(y_true, y_pred),
        }
        
        if horizon_label:
            logger.info(
                f"  {horizon_label}: "
                f"PE={metrics['PE']:.4f}, "
                f"RMSE={metrics['RMSE_log']:.4f}, "
                f"R={metrics['Pearson_R']:.4f}, "
                f"MAE={metrics['MAE_log']:.4f}, "
                f"HSS={metrics['HSS']:.4f}"
            )
        
        return metrics
    
    @classmethod
    def compute_multi_horizon(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        horizon_labels: list = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for all forecast horizons.
        
        Args:
            y_true: Array of shape (N, num_horizons).
            y_pred: Array of shape (N, num_horizons).
            horizon_labels: Names for each horizon.
        
        Returns:
            Nested dict: {horizon_label: {metric_name: value}}.
        """
        if horizon_labels is None:
            horizon_labels = [f'h{i}' for i in range(y_true.shape[1])]
        
        logger.info("Multi-horizon evaluation:")
        
        all_metrics = {}
        for i, label in enumerate(horizon_labels):
            if i < y_true.shape[1]:
                all_metrics[label] = cls.compute_all(
                    y_true[:, i], y_pred[:, i], label
                )
        
        return all_metrics
    
    @staticmethod
    def persistence_baseline(
        y_true: np.ndarray,
        horizon_steps: List[int],
        full_series: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Compute persistence baseline: predict flux stays the same.
        
        This is the simplest possible forecast — tomorrow's flux equals
        today's flux. Any useful model must beat this.
        
        Args:
            y_true: Full time series of log10 flux.
            horizon_steps: Steps ahead for each horizon.
            full_series: The complete series (if different from y_true).
        
        Returns:
            Dictionary of PE scores for persistence at each horizon.
        """
        series = full_series if full_series is not None else y_true
        
        results = {}
        for h in horizon_steps:
            if h < len(series):
                pred = series[:-h]
                actual = series[h:]
                n = min(len(pred), len(actual))
                pe = ForecastMetrics.prediction_efficiency(actual[:n], pred[:n])
                results[f'persistence_{h}step'] = pe
        
        return results
    
    @staticmethod
    def evaluate_storm_quiet(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dst_values: np.ndarray,
        storm_threshold: float = -50,
    ) -> Tuple[Dict, Dict]:
        """
        Separate evaluation for storm-time and quiet conditions.
        
        Storm periods (Dst < -50 nT) are the most important to predict
        correctly, as they pose the greatest threat to satellites.
        
        Args:
            y_true: Observed log10 flux.
            y_pred: Predicted log10 flux.
            dst_values: Dst index values aligned with predictions.
            storm_threshold: Dst threshold for storm classification.
        
        Returns:
            Tuple of (storm_metrics, quiet_metrics).
        """
        storm_mask = dst_values < storm_threshold
        quiet_mask = ~storm_mask
        
        storm_metrics = {}
        quiet_metrics = {}
        
        if storm_mask.any():
            storm_metrics = {
                'PE': ForecastMetrics.prediction_efficiency(
                    y_true[storm_mask], y_pred[storm_mask]
                ),
                'RMSE_log': ForecastMetrics.rmse_log(
                    y_true[storm_mask], y_pred[storm_mask]
                ),
                'R': ForecastMetrics.pearson_r(
                    y_true[storm_mask], y_pred[storm_mask]
                ),
                'count': int(storm_mask.sum()),
            }
        
        if quiet_mask.any():
            quiet_metrics = {
                'PE': ForecastMetrics.prediction_efficiency(
                    y_true[quiet_mask], y_pred[quiet_mask]
                ),
                'RMSE_log': ForecastMetrics.rmse_log(
                    y_true[quiet_mask], y_pred[quiet_mask]
                ),
                'R': ForecastMetrics.pearson_r(
                    y_true[quiet_mask], y_pred[quiet_mask]
                ),
                'count': int(quiet_mask.sum()),
            }
        
        logger.info(f"Storm-time (Dst<{storm_threshold}): {storm_metrics}")
        logger.info(f"Quiet-time: {quiet_metrics}")
        
        return storm_metrics, quiet_metrics

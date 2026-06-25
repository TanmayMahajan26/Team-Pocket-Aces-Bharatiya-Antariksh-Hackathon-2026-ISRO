"""
Main Pipeline Orchestrator for ISRO PS14 Radiation Forecasting.

Runs the complete end-to-end pipeline:
1. Data download (or load from cache)
2. Preprocessing
3. Feature engineering
4. Model training (Transformer + LSTM)
5. Evaluation
6. Visualization
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import Config
from src.utils.logger import setup_logger, get_logger
from src.data.downloader import DataDownloader
from src.data.preprocessor import DataPreprocessor
from src.data.dataset import create_dataloaders
from src.features.feature_engineer import FeatureEngineer
from src.models.lstm_model import LSTMForecaster
from src.models.transformer_model import TransformerForecaster
from src.models.trainer import ModelTrainer
from src.evaluation.metrics import ForecastMetrics
from src.evaluation.visualizer import Visualizer

logger = get_logger("main")


def run_download(config: Config):
    """Step 1: Download all required data."""
    logger.info("=" * 70)
    logger.info("STEP 1: DATA DOWNLOAD")
    logger.info("=" * 70)
    
    downloader = DataDownloader(config)
    results = downloader.download_all()
    return results


def run_preprocessing(config: Config) -> pd.DataFrame:
    """Step 2: Preprocess raw data into unified dataset."""
    logger.info("=" * 70)
    logger.info("STEP 2: PREPROCESSING")
    logger.info("=" * 70)
    
    preprocessor = DataPreprocessor(config)
    
    # Check for cached processed data
    cache_path = config.data.get('cache_file', 'data/processed/merged_dataset.parquet')
    if Path(cache_path).exists():
        logger.info(f"Loading cached processed data from {cache_path}")
        return preprocessor.load_processed()
    
    # Run full pipeline
    df = preprocessor.run_full_pipeline(save=True)
    return df


def run_feature_engineering(
    config: Config,
    df: pd.DataFrame
) -> tuple:
    """Step 3: Generate features and normalize."""
    logger.info("=" * 70)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    engineer = FeatureEngineer(config)
    df = engineer.generate_all_features(df)
    
    # Get feature columns
    feature_cols = engineer.get_feature_columns()
    # Only keep columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    if not feature_cols:
        # Fallback: use all numeric columns
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'electron_flux_gt2MeV' in feature_cols:
            feature_cols.remove('electron_flux_gt2MeV')
        if 'proton_flux_gt10MeV' in feature_cols:
            feature_cols.remove('proton_flux_gt10MeV')
    
    logger.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")
    
    # Fit scaler on training data
    split = config.training.get('split', {})
    train_years = split.get('train_years', [2011, 2018])
    train_mask = (df.index.year >= train_years[0]) & (df.index.year <= train_years[1])
    
    engineer.fit_scaler(df[train_mask], feature_cols)
    df = engineer.transform(df)
    
    # Save scaler
    engineer.save_scaler()
    
    return df, engineer, feature_cols


def run_eda(config: Config, df: pd.DataFrame):
    """Step 3b: Generate EDA visualizations."""
    logger.info("=" * 70)
    logger.info("STEP 3b: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 70)
    
    viz = Visualizer(config)
    plots = viz.create_all_eda_plots(df)
    logger.info(f"Generated {len(plots)} EDA plots")
    return plots


def run_training(
    config: Config,
    df: pd.DataFrame,
    feature_cols: list,
    model_type: str = 'both',
) -> dict:
    """Step 4: Train models."""
    logger.info("=" * 70)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("=" * 70)
    
    results = {}
    
    # Create DataLoaders
    if model_type in ['transformer', 'both']:
        logger.info("\n--- Training Transformer Model ---")
        train_loader, val_loader, test_loader, metadata = create_dataloaders(
            df, feature_cols, config, model_type='transformer'
        )
        
        # Create model
        model = TransformerForecaster.from_config(
            config._config, num_features=metadata['num_features']
        )
        
        # Train
        trainer = ModelTrainer(model, config, model_name='transformer')
        history = trainer.train(train_loader, val_loader)
        
        # Test predictions
        test_preds, test_targets, test_uncertainties = trainer.predict(test_loader)
        
        # Feature Importance
        importance = trainer.compute_feature_importance(test_loader, feature_cols)
        
        results['transformer'] = {
            'trainer': trainer,
            'history': history,
            'test_preds': test_preds,
            'test_targets': test_targets,
            'test_uncertainties': test_uncertainties,
            'importance': importance,
            'metadata': metadata,
            'test_loader': test_loader,
        }
        
        # Save predictions
        pred_dir = config.root / 'outputs' / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            pred_dir / 'transformer_test_predictions.npz',
            predictions=test_preds,
            targets=test_targets,
            p5=test_uncertainties.get('p5', np.zeros_like(test_preds)),
            p95=test_uncertainties.get('p95', np.zeros_like(test_preds)),
        )
        import json
        with open(pred_dir / 'transformer_importance.json', 'w') as f:
            json.dump(importance, f)
    
    if model_type in ['lstm', 'both']:
        logger.info("\n--- Training LSTM Model ---")
        train_loader_lstm, val_loader_lstm, test_loader_lstm, metadata_lstm = create_dataloaders(
            df, feature_cols, config, model_type='lstm'
        )
        
        # Create model
        model_lstm = LSTMForecaster.from_config(
            config._config, num_features=metadata_lstm['num_features']
        )
        
        # Train
        trainer_lstm = ModelTrainer(model_lstm, config, model_name='lstm')
        history_lstm = trainer_lstm.train(train_loader_lstm, val_loader_lstm)
        
        # Test predictions
        test_preds_lstm, test_targets_lstm, test_uncertainties_lstm = trainer_lstm.predict(test_loader_lstm)
        
        # Feature Importance
        importance_lstm = trainer_lstm.compute_feature_importance(test_loader_lstm, feature_cols)
        
        results['lstm'] = {
            'trainer': trainer_lstm,
            'history': history_lstm,
            'test_preds': test_preds_lstm,
            'test_targets': test_targets_lstm,
            'test_uncertainties': test_uncertainties_lstm,
            'importance': importance_lstm,
            'metadata': metadata_lstm,
            'test_loader': test_loader_lstm,
        }
        
        # Save predictions
        pred_dir = config.root / 'outputs' / 'predictions'
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            pred_dir / 'lstm_test_predictions.npz',
            predictions=test_preds_lstm,
            targets=test_targets_lstm,
            p5=test_uncertainties_lstm.get('p5', np.zeros_like(test_preds_lstm)),
            p95=test_uncertainties_lstm.get('p95', np.zeros_like(test_preds_lstm)),
        )
        import json
        with open(pred_dir / 'lstm_importance.json', 'w') as f:
            json.dump(importance_lstm, f)
    
    return results


def run_evaluation(config: Config, training_results: dict) -> dict:
    """Step 5: Evaluate models."""
    logger.info("=" * 70)
    logger.info("STEP 5: EVALUATION")
    logger.info("=" * 70)
    
    horizon_labels = config.features.get('target_horizons_labels', ['30min', '6h', '12h'])
    all_metrics = {}
    
    for model_name, result in training_results.items():
        logger.info(f"\n--- {model_name.upper()} Evaluation ---")
        
        metrics = ForecastMetrics.compute_multi_horizon(
            result['test_targets'],
            result['test_preds'],
            horizon_labels
        )
        
        all_metrics[model_name] = metrics
    
    # Print comparison table
    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Model':<15} {'Horizon':<10} {'PE':<10} {'RMSE':<10} {'R':<10} {'MAE':<10}")
    logger.info("-" * 65)
    
    for model_name, metrics in all_metrics.items():
        for horizon, m in metrics.items():
            logger.info(
                f"{model_name:<15} {horizon:<10} "
                f"{m['PE']:<10.4f} {m['RMSE_log']:<10.4f} "
                f"{m['Pearson_R']:<10.4f} {m['MAE_log']:<10.4f} "
                f"HSS={m['HSS']:.4f}"
            )
    
    return all_metrics


def run_visualization(
    config: Config,
    training_results: dict,
    all_metrics: dict,
):
    """Step 6: Generate result visualizations."""
    logger.info("=" * 70)
    logger.info("STEP 6: VISUALIZATION")
    logger.info("=" * 70)
    
    viz = Visualizer(config)
    horizon_labels = config.features.get('target_horizons_labels', ['30min', '6h', '12h'])
    
    for model_name, result in training_results.items():
        # Training history
        viz.plot_training_history(result['history'], model_name.capitalize())
        
        # Predictions time series
        test_ds = result['test_loader'].dataset
        timestamps = np.array([test_ds.get_timestamp(i) for i in range(len(test_ds))])
        
        viz.plot_predictions_timeseries(
            timestamps,
            result['test_targets'],
            result['test_preds'],
            horizon_labels,
            model_name.capitalize()
        )
        
        # Scatter plots
        viz.plot_scatter(
            result['test_targets'],
            result['test_preds'],
            horizon_labels,
            model_name.capitalize()
        )
    
    # Metrics comparison
    if len(all_metrics) > 1:
        for metric in ['PE', 'RMSE_log', 'Pearson_R', 'HSS']:
            viz.plot_metrics_comparison(all_metrics, metric)
    
    # Feature importance charts (XAI)
    for model_name, result in training_results.items():
        if result.get('importance'):
            viz.plot_feature_importance(result['importance'], model_name.capitalize())
    
    # Uncertainty band plots (MC Dropout)
    for model_name, result in training_results.items():
        uncertainties = result.get('test_uncertainties', {})
        if uncertainties.get('p5') is not None and uncertainties.get('p95') is not None:
            test_ds = result['test_loader'].dataset
            timestamps = np.array([test_ds.get_timestamp(i) for i in range(len(test_ds))])
            viz.plot_uncertainty_bands(
                timestamps,
                result['test_targets'],
                result['test_preds'],
                uncertainties['p5'],
                uncertainties['p95'],
                horizon_labels,
                model_name.capitalize()
            )
    
    logger.info("All visualizations generated!")


def main(
    mode: str = 'full',
    config_path: str = None,
    model_type: str = 'both',
    skip_download: bool = False,
):
    """
    Run the ISRO PS14 pipeline.
    
    Args:
        mode: 'full', 'download', 'preprocess', 'train', 'evaluate'
        config_path: Path to config.yaml
        model_type: 'transformer', 'lstm', or 'both'
        skip_download: Skip data download step
    """
    # Setup
    config = Config(config_path)
    config.ensure_dirs()
    
    setup_logger("isro_ps14", log_file=str(config.root / 'outputs' / 'training.log'))
    
    logger.info("=" * 62)
    logger.info("  ISRO PS14 - Energetic Particle Radiation Forecasting")
    logger.info("  Geostationary Electron Flux Prediction System")
    logger.info("=" * 62)
    
    # Device info
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("Running on CPU (training will be slower)")
    
    if mode in ['full', 'download']:
        if not skip_download:
            run_download(config)
        if mode == 'download':
            return
    
    if mode in ['full', 'preprocess']:
        df = run_preprocessing(config)
        if mode == 'preprocess':
            return
    else:
        # Load cached data
        preprocessor = DataPreprocessor(config)
        df = preprocessor.load_processed()
    
    # Feature engineering
    df_featured, engineer, feature_cols = run_feature_engineering(config, df)
    
    # EDA plots (on un-normalized data)
    run_eda(config, df)
    
    if mode in ['full', 'train']:
        # Training
        training_results = run_training(config, df_featured, feature_cols, model_type)
        
        # Evaluation
        all_metrics = run_evaluation(config, training_results)
        
        # Visualization
        run_visualization(config, training_results, all_metrics)
        
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {config.root / 'outputs'}")
        logger.info(f"Models saved to: {config.training.get('checkpoint_dir')}")
        logger.info("Run the Streamlit dashboard: streamlit run app/dashboard.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='ISRO PS14 - Energetic Particle Radiation Forecasting Pipeline'
    )
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['full', 'download', 'preprocess', 'train', 'evaluate'],
        help='Pipeline mode to run'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to config.yaml'
    )
    parser.add_argument(
        '--model', type=str, default='both',
        choices=['transformer', 'lstm', 'both'],
        help='Which model(s) to train'
    )
    parser.add_argument(
        '--skip-download', action='store_true',
        help='Skip data download step'
    )
    
    args = parser.parse_args()
    main(
        mode=args.mode,
        config_path=args.config,
        model_type=args.model,
        skip_download=args.skip_download,
    )

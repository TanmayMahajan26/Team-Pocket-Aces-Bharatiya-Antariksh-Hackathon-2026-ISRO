import os
import time
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

from src.utils.logger import get_logger
from src.utils.config import Config
from src.data.live_fetcher import LiveDataFetcher
from src.features.feature_engineer import FeatureEngineer
from src.models.transformer_model import TransformerForecaster

logger = get_logger(__name__)

class LiveInferenceEngine:
    """
    Runs real-time inference on incoming space weather data.
    """
    
    def __init__(self, config_path: str = None):
        self.config = Config(config_path)
        self.fetcher = LiveDataFetcher()
        self.engineer = FeatureEngineer(self.config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Output paths
        self.output_dir = self.config.root / 'outputs' / 'live'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.output_dir / 'live_state.json'
        
        # Load scaler
        scaler_path = self.config.root / 'models' / 'feature_scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Must train first.")
        self.engineer.load_scaler(str(scaler_path))
        self.feature_cols = self.engineer.get_feature_columns()
        
        # Load best model
        checkpoint_path = self.config.root / 'models' / 'checkpoints' / 'best_transformer.pt'
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found at {checkpoint_path}. Must train first.")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        metadata = checkpoint.get('metadata', {})
        
        # Backward compatibility for the old 9-feature model checkpoint
        if 'num_features' not in metadata:
            self.feature_cols = ['log10_flux', 'flux_diff_1h', 'flux_rolling_std_6h', 
                                 'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 
                                 'bartels_sin', 'bartels_cos']
            num_features = 9
            self.engineer.feature_columns = self.feature_cols
        else:
            num_features = metadata['num_features']
        
        self.model = TransformerForecaster.from_config(self.config._config, num_features=num_features)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Settings
        self.seq_len = self.config.features.get('sequence_length', 72)
        
    def run_inference(self):
        """Fetch latest data, generate features, and predict."""
        logger.info("Starting live inference cycle...")
        
        # 1. Fetch live data (need enough history for sequence length + feature engineer padding)
        # Sequence is 72 * 5m = 6 hours. Let's grab 12 hours to be safe for rolling windows.
        raw_df = self.fetcher.get_latest_aligned_data(hours_back=12)
        
        if len(raw_df) < self.seq_len:
            logger.warning(f"Not enough live data: got {len(raw_df)} rows, need {self.seq_len}")
            return
            
        # 2. Feature Engineering
        df = self.engineer.generate_all_features(raw_df.copy())
        
        # Ensure all columns exist
        for col in self.feature_cols:
            if col not in df.columns:
                logger.warning(f"Feature {col} missing! Filling with 0.")
                df[col] = 0.0
                
        print("DEBUG feature_cols inside live_inference:", self.feature_cols)
        print("DEBUG engineer feature_columns:", self.engineer.feature_columns)
        df = self.engineer.transform(df)
        
        # 3. Prepare sequence (last `seq_len` steps)
        # Drop rows with NaNs in features
        df_clean = df.dropna(subset=self.feature_cols)
        if len(df_clean) < self.seq_len:
            logger.warning("Not enough valid data after feature engineering.")
            return
            
        # Get the very last sequence
        seq_df = df_clean.iloc[-self.seq_len:]
        X = seq_df[self.feature_cols].values
        
        # Add batch dimension and tensor
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 4. Predict (with MC Dropout for uncertainty)
        num_samples = 20
        all_preds = []
        
        with torch.amp.autocast('cuda', enabled=(self.device.type=='cuda')):
            for m in self.model.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
                    
            for _ in range(num_samples):
                pred = self.model(X_tensor).detach().cpu().numpy()
                all_preds.append(pred)
                
        mc_preds = np.concatenate(all_preds, axis=0) # shape: (samples, horizons)
        
        mean_pred = np.mean(mc_preds, axis=0)
        p5 = np.percentile(mc_preds, 5, axis=0)
        p95 = np.percentile(mc_preds, 95, axis=0)
        
        # 5. Format results
        latest_time = seq_df.index[-1]
        
        # Un-log10 the flux to get raw pfu
        mean_pfu = 10 ** mean_pred
        p95_pfu = 10 ** p95
        
        results = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'latest_data_time': latest_time.isoformat(),
            'current_flux_gt2MeV': float(raw_df['electron_flux_gt2MeV'].iloc[-1]),
            'horizons': ['30min', '6h', '12h'],
            'predictions_log10': mean_pred.tolist(),
            'p5_log10': p5.tolist(),
            'p95_log10': p95.tolist(),
            'predictions_pfu': mean_pfu.tolist(),
            'p95_pfu': p95_pfu.tolist(),
            'status': 'Normal'
        }
        
        # Alert logic based on 95th percentile
        if any(val > 1000 for val in p95_pfu):
            results['status'] = 'CRITICAL'
        elif any(val > 100 for val in p95_pfu):
            results['status'] = 'WARNING'
            
        # Save to state file for dashboard
        with open(self.state_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Live inference complete. Status: {results['status']}")
        return results

if __name__ == "__main__":
    engine = LiveInferenceEngine()
    engine.run_inference()

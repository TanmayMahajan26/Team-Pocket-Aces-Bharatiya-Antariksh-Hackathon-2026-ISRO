from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import numpy as np
import torch
from typing import Dict, List, Any
import datetime
from src.models.transformer_model import TransformerForecaster
from src.utils.config import Config
from pathlib import Path

app = FastAPI(
    title="ISRO PS-14 Radiation Forecast API",
    description="Production REST API for energetic particle radiation forecasting.",
    version="1.0.0"
)

# Global variables to hold model state
model = None
config = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ForecastResponse(BaseModel):
    timestamp: str
    horizon_30min: Dict[str, float]
    horizon_6h: Dict[str, float]
    horizon_12h: Dict[str, float]

@app.on_event("startup")
async def startup_event():
    """Load model weights and config on server start."""
    global model, config
    try:
        config = Config()
        checkpoint_path = Path("models/checkpoints/transformer_best.pth")
        
        if not checkpoint_path.exists():
            print(f"Warning: Model not found at {checkpoint_path}. API endpoints will fail until trained.")
            return

        model = TransformerForecaster(
            seq_len=config.model.get('seq_len', 144),
            pred_len=config.model.get('pred_len', 3),
            num_features=9, # Will need to be dynamic based on training
            d_model=config.model.get('d_model', 64),
            n_heads=config.model.get('n_heads', 4),
            e_layers=config.model.get('e_layers', 2),
            d_ff=config.model.get('d_ff', 256),
            dropout=config.model.get('dropout', 0.1)
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("Transformer model successfully loaded for inference.")
        
    except Exception as e:
        print(f"Failed to load model: {e}")

@app.get("/")
def read_root():
    return {"status": "healthy", "service": "ISRO Radiation Forecasting API"}

@app.get("/predict/latest", response_model=ForecastResponse)
def get_latest_forecast():
    """
    In a real production environment, this would pull the latest streaming data,
    pass it through the preprocessor, and return a live forecast.
    For this demo, we read the latest prediction from our npz output.
    """
    try:
        pred_file = Path("outputs/predictions/transformer_test_predictions.npz")
        if not pred_file.exists():
            raise HTTPException(status_code=404, detail="Predictions not available. Run training pipeline first.")
            
        data = np.load(pred_file)
        preds = data['predictions']
        p5 = data.get('p5', None)
        p95 = data.get('p95', None)
        
        # Get the absolute last prediction in the test set
        latest_pred = preds[-1]
        
        response = {
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "horizon_30min": {"mean_log10_flux": float(latest_pred[0])},
            "horizon_6h": {"mean_log10_flux": float(latest_pred[1])},
            "horizon_12h": {"mean_log10_flux": float(latest_pred[2])}
        }
        
        # Add uncertainty if available
        if p5 is not None and p95 is not None:
            response["horizon_30min"].update({"p5": float(p5[-1][0]), "p95": float(p95[-1][0])})
            response["horizon_6h"].update({"p5": float(p5[-1][1]), "p95": float(p95[-1][1])})
            response["horizon_12h"].update({"p5": float(p5[-1][2]), "p95": float(p95[-1][2])})
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

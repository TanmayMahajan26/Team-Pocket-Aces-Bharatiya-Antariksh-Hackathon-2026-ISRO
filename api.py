import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from src.data.database import DatabaseManager

import logging

logger = logging.getLogger("api")

app = FastAPI(
    title="ISRO PS14 Space Weather API",
    description="Real-time Geostationary Electron Flux Prediction API",
    version="1.0.0"
)

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ROOT = Path(__file__).parent
LIVE_STATE_FILE = PROJECT_ROOT / "outputs" / "live" / "live_state.json"

db = DatabaseManager()

class ForecastData(BaseModel):
    timestamp: str
    latest_data_time: str
    connection_status: str
    current_flux_gt2MeV: float
    status: str
    predictions_pfu: List[float]
    p95_pfu: List[float]
    explainability: Optional[Dict[str, Any]] = None

@app.get("/api/status")
async def get_system_status():
    """Check if the backend is running."""
    return {"status": "operational", "service": "ISRO PS14 Backend API"}

@app.get("/api/live", response_model=ForecastData)
async def get_live_forecast():
    """Get the latest real-time predictions."""
    try:
        history = db.get_latest_predictions(limit=1)
        if not history:
            logger.warning("No predictions found in database. Is the daemon running?")
            raise HTTPException(status_code=503, detail="Live state not available. Ensure the daemon is running.")
        return history[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal Server Error while fetching live forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/api/history", response_model=List[ForecastData])
async def get_history(limit: int = Query(100, ge=1, le=1000)):
    """Get historical predictions for plotting drift."""
    try:
        history = db.get_latest_predictions(limit=limit)
        return history
    except Exception as e:
        logger.error(f"Internal Server Error while fetching history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    # Run the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

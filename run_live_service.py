import time
import os
import sys
from datetime import datetime
from src.utils.logger import get_logger
from src.models.live_inference import LiveInferenceEngine

logger = get_logger(__name__)

def run_service():
    """
    Run the live inference service indefinitely.
    Fetches new data and runs inference every 5 minutes.
    """
    logger.info("=" * 60)
    logger.info("Starting ISRO PS14 Live Inference Service")
    logger.info("=" * 60)
    
    try:
        engine = LiveInferenceEngine()
    except Exception as e:
        logger.error(f"Failed to initialize LiveInferenceEngine: {e}")
        logger.error("Make sure you have trained a model first!")
        sys.exit(1)
        
    while True:
        try:
            logger.info(f"--- Running Inference Cycle at {datetime.now().isoformat()} ---")
            engine.run_inference()
        except Exception as e:
            logger.error(f"Error during inference cycle: {e}")
            
        # Sleep for 5 minutes (300 seconds)
        logger.info("Sleeping for 5 minutes until next cycle...")
        time.sleep(300)

if __name__ == "__main__":
    run_service()

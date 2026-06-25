import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time
from typing import Dict, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

class LiveDataFetcher:
    """
    Fetches near-real-time (NRT) space weather data from NOAA SWPC APIs.
    """
    
    def __init__(self):
        # NOAA SWPC JSON endpoints
        self.flux_url = "https://services.swpc.noaa.gov/json/goes/primary/integral-electrons-3-day.json"
        self.plasma_url = "https://services.swpc.noaa.gov/products/solar-wind/plasma-3-day.json"
        self.mag_url = "https://services.swpc.noaa.gov/products/solar-wind/mag-3-day.json"
    
    def fetch_electron_flux(self) -> pd.DataFrame:
        """Fetch the last 3 days of >2 MeV electron flux from primary GOES."""
        logger.info("Fetching live GOES electron flux...")
        response = requests.get(self.flux_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse JSON
        records = []
        for item in data:
            if item.get('energy') == '>=2 MeV':
                records.append({
                    'datetime': pd.to_datetime(item['time_tag']),
                    'electron_flux_gt2MeV': float(item['flux'])
                })
        
        df = pd.DataFrame(records)
        if df.empty:
            return df
            
        df = df.set_index('datetime').sort_index()
        # Drop duplicates just in case
        df = df[~df.index.duplicated(keep='last')]
        return df
        
    def fetch_solar_wind_plasma(self) -> pd.DataFrame:
        """Fetch live solar wind speed and density."""
        logger.info("Fetching live solar wind plasma...")
        response = requests.get(self.plasma_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Format: ["time_tag", "density", "speed", "temperature"]
        if len(data) <= 1:
            return pd.DataFrame()
            
        df = pd.DataFrame(data[1:], columns=data[0])
        df['datetime'] = pd.to_datetime(df['time_tag'])
        df = df.set_index('datetime').sort_index()
        
        # Convert types
        df['speed'] = pd.to_numeric(df['speed'], errors='coerce')
        df['density'] = pd.to_numeric(df['density'], errors='coerce')
        
        # Rename to match historical (Vsw, Np)
        df = df.rename(columns={'speed': 'Vsw', 'density': 'Np'})
        return df[['Vsw', 'Np']]
        
    def fetch_solar_wind_mag(self) -> pd.DataFrame:
        """Fetch live solar wind magnetic field."""
        logger.info("Fetching live solar wind magnetic field...")
        response = requests.get(self.mag_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Format: ["time_tag", "bx_gsm", "by_gsm", "bz_gsm", "lon_gsm", "lat_gsm", "bt"]
        if len(data) <= 1:
            return pd.DataFrame()
            
        df = pd.DataFrame(data[1:], columns=data[0])
        df['datetime'] = pd.to_datetime(df['time_tag'])
        df = df.set_index('datetime').sort_index()
        
        # Convert types
        df['bz_gsm'] = pd.to_numeric(df['bz_gsm'], errors='coerce')
        df['bt'] = pd.to_numeric(df['bt'], errors='coerce')
        
        # Rename to match historical
        df = df.rename(columns={'bz_gsm': 'Bz', 'bt': 'B'})
        return df[['Bz', 'B']]
        
    def get_latest_aligned_data(self, hours_back: int = 12) -> pd.DataFrame:
        """
        Fetches all live data, merges it, resamples to 5-min intervals,
        and returns the last N hours.
        """
        try:
            flux_df = self.fetch_electron_flux()
            plasma_df = self.fetch_solar_wind_plasma()
            mag_df = self.fetch_solar_wind_mag()
            
            flux_df.index = flux_df.index.tz_localize(None)
            plasma_df.index = plasma_df.index.tz_localize(None)
            mag_df.index = mag_df.index.tz_localize(None)
            
            # Merge solar wind
            sw_df = pd.merge(plasma_df, mag_df, left_index=True, right_index=True, how='outer')
            
            # 5-min resampling (forward fill small gaps, mean for downsampling)
            flux_5m = flux_df.resample('5min').mean()
            sw_5m = sw_df.resample('5min').mean()
            
            # Merge all
            merged = pd.merge(flux_5m, sw_5m, left_index=True, right_index=True, how='outer')
            
            # We don't have real-time OMNIWeb Dst/Kp in this simple JSON, so we approximate
            # or just fill with neutral values since our model relies most heavily on L1 data.
            if 'Dst' not in merged.columns:
                merged['Dst'] = 0.0
            if 'Kp' not in merged.columns:
                merged['Kp'] = 1.0
                
            # Filter to last N hours
            cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=hours_back)
            # Remove timezone info for compatibility with our pipeline which is tz-naive UTC
            cutoff = cutoff.replace(tzinfo=None)
            
            merged.index = merged.index.tz_localize(None)
            merged = merged[merged.index >= cutoff]
            
            # Forward fill remaining NaNs up to 1 hour
            merged = merged.ffill(limit=12).bfill(limit=12)
            
            return merged
            
        except Exception as e:
            logger.error(f"Live data fetch failed: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    fetcher = LiveDataFetcher()
    df = fetcher.get_latest_aligned_data(hours_back=6)
    print(df.tail())

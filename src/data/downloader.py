"""
Data Downloader for ISRO PS14 Radiation Forecasting.

Downloads CDF data files from:
- SPDF/CDAWeb: GOES electron flux, Wind SWE & MFI
- OMNIWeb: Geomagnetic indices (Kp, Dst, AE, SYM-H)
- PRADAN: GRASP/GSAT-19 data (manual download guidance)

Supports resumable downloads and year-by-year processing.
"""

import os
import re
import time
import requests
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from urllib.parse import urljoin

from src.utils.logger import get_logger
from src.utils.config import Config

logger = get_logger(__name__)


class DataDownloader:
    """
    Downloads space weather data from NASA and NOAA archives.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.config.ensure_dirs()
        self.session = requests.Session()
        # Reduced max_workers to 1 to prevent NASA SPDF connection resets
        self.max_workers = 1
        self.session.headers.update({
            'User-Agent': 'ISRO-PS14-RadiationForecast/1.0 (research; python/requests)'
        })
    
    # =========================================================================
    # GOES Electron Flux Data
    # =========================================================================
    
    def download_goes(
        self,
        start_year: int = None,
        end_year: int = None,
        satellites: List[int] = None
    ) -> List[str]:
        """
        Download GOES EPS (Energetic Particle Sensor) CDF files from SPDF.
        
        GOES data path structure:
          https://spdf.gsfc.nasa.gov/pub/data/goes/goes{N}/eps_k0/{year}/
        
        Args:
            start_year: Start year (default from config).
            end_year: End year (default from config).
            satellites: List of GOES satellite numbers (e.g., [13, 15]).
        
        Returns:
            List of downloaded file paths.
        """
        start_year = start_year or self.config.data.get('start_year', 2011)
        end_year = end_year or self.config.data.get('end_year', 2021)
        satellites = satellites or self.config.get('data', 'goes', 'satellites', default=[13, 15])
        
        base_url = "https://spdf.gsfc.nasa.gov/pub/data/goes"
        output_dir = Path(self.config.data.get('raw_dir', 'data/raw')) / 'goes'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        
        for sat_num in satellites:
            # Try different URL patterns for different GOES versions
            url_patterns = [
                f"{base_url}/goes{sat_num:02d}/eps_k0",
                f"{base_url}/goes{sat_num}/eps_k0",
                f"{base_url}/goes{sat_num:02d}/epead_e13ew_1min",
                f"{base_url}/goes{sat_num}/epead-electrons/e13ew_1min",
            ]
            
            for year in range(start_year, end_year + 1):
                found = False
                for pattern in url_patterns:
                    year_url = f"{pattern}/{year}/"
                    
                    try:
                        files = self._list_remote_files(year_url, '.cdf')
                        if files:
                            logger.info(f"GOES-{sat_num} {year}: Found {len(files)} CDF files")
                            for fname in files:
                                file_url = urljoin(year_url, fname)
                                local_path = output_dir / f"goes{sat_num}" / str(year) / fname
                                if self._download_file(file_url, local_path):
                                    downloaded.append(str(local_path))
                            found = True
                            break
                    except Exception as e:
                        logger.debug(f"Pattern {pattern}/{year}/ failed: {e}")
                
                if not found:
                    logger.warning(f"GOES-{sat_num} {year}: No data found")
        
        logger.info(f"GOES download complete: {len(downloaded)} files")
        return downloaded
    
    # =========================================================================
    # Wind Solar Wind Data
    # =========================================================================
    
    def download_wind_swe(
        self,
        start_year: int = None,
        end_year: int = None
    ) -> List[str]:
        """
        Download Wind SWE (Solar Wind Experiment) key parameter CDF files.
        
        Path: https://spdf.gsfc.nasa.gov/pub/data/wind/swe/swe_k0/{year}/
        
        Args:
            start_year: Start year.
            end_year: End year.
        
        Returns:
            List of downloaded file paths.
        """
        start_year = start_year or self.config.data.get('start_year', 2011)
        end_year = end_year or self.config.data.get('end_year', 2021)
        
        base_url = "https://spdf.gsfc.nasa.gov/pub/data/wind/swe/swe_k0"
        output_dir = Path(self.config.data.get('raw_dir', 'data/raw')) / 'wind_swe'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        
        for year in range(start_year, end_year + 1):
            year_url = f"{base_url}/{year}/"
            
            try:
                files = self._list_remote_files(year_url, '.cdf')
                if files:
                    logger.info(f"Wind SWE {year}: Found {len(files)} CDF files")
                    for fname in files:
                        file_url = urljoin(year_url, fname)
                        local_path = output_dir / str(year) / fname
                        if self._download_file(file_url, local_path):
                            downloaded.append(str(local_path))
                else:
                    logger.warning(f"Wind SWE {year}: No CDF files found")
            except Exception as e:
                logger.error(f"Wind SWE {year} download failed: {e}")
        
        logger.info(f"Wind SWE download complete: {len(downloaded)} files")
        return downloaded
    
    def download_wind_mfi(
        self,
        start_year: int = None,
        end_year: int = None
    ) -> List[str]:
        """
        Download Wind MFI (Magnetic Field Investigation) CDF files.
        
        Path: https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h0/{year}/
        
        Args:
            start_year: Start year.
            end_year: End year.
        
        Returns:
            List of downloaded file paths.
        """
        start_year = start_year or self.config.data.get('start_year', 2011)
        end_year = end_year or self.config.data.get('end_year', 2021)
        
        base_url = "https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/mfi_h0"
        output_dir = Path(self.config.data.get('raw_dir', 'data/raw')) / 'wind_mfi'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        downloaded = []
        
        for year in range(start_year, end_year + 1):
            year_url = f"{base_url}/{year}/"
            
            try:
                files = self._list_remote_files(year_url, '.cdf')
                if files:
                    logger.info(f"Wind MFI {year}: Found {len(files)} CDF files")
                    for fname in files:
                        file_url = urljoin(year_url, fname)
                        local_path = output_dir / str(year) / fname
                        if self._download_file(file_url, local_path):
                            downloaded.append(str(local_path))
                else:
                    logger.warning(f"Wind MFI {year}: No CDF files found")
            except Exception as e:
                logger.error(f"Wind MFI {year} download failed: {e}")
        
        logger.info(f"Wind MFI download complete: {len(downloaded)} files")
        return downloaded
    
    # =========================================================================
    # OMNIWeb Geomagnetic Indices
    # =========================================================================
    
    def download_omniweb_indices(
        self,
        start_year: int = None,
        end_year: int = None,
        resolution: str = 'hourly'
    ) -> str:
        """
        Download geomagnetic indices from OMNIWeb.
        
        Fetches Kp, Dst, AE, SYM-H, and F10.7 index.
        
        Args:
            start_year: Start year.
            end_year: End year.
            resolution: 'hourly' or '1min' (1-min has SYM-H).
        
        Returns:
            Path to saved CSV file.
        """
        start_year = start_year or self.config.data.get('start_year', 2011)
        end_year = end_year or self.config.data.get('end_year', 2021)
        
        output_dir = Path(self.config.data.get('raw_dir', 'data/raw')) / 'omniweb'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f'omniweb_{resolution}_{start_year}_{end_year}.csv'
        
        if output_file.exists():
            logger.info(f"OMNIWeb data already exists: {output_file}")
            return str(output_file)
        
        logger.info(f"Downloading OMNIWeb {resolution} data for {start_year}-{end_year}")
        
        # OMNIWeb CGI interface
        if resolution == 'hourly':
            url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
            # Variable codes for hourly OMNI:
            # 38=Kp, 40=Dst, 41=AE, 13=F10.7, 
            # 24=Vsw, 23=Np, 28=Bz_GSM, 25=Bt
            params = {
                'activity': 'retrieve',
                'res': 'hour',
                'spacecraft': 'omni2',
                'start_date': f'{start_year}0101',
                'end_date': f'{end_year}1231',
                'vars': '13,23,24,25,28,38,40,41',
                'output_type': 'file',
            }
        else:
            url = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
            params = {
                'activity': 'retrieve',
                'res': 'min',
                'spacecraft': 'omni_min',
                'start_date': f'{start_year}0101',
                'end_date': f'{end_year}1231',
                'vars': '13,23,24,25,28,38,40,41,44',  # 44=SYM-H
                'output_type': 'file',
            }
        
        try:
            resp = self.session.get(url, params=params, timeout=300)
            resp.raise_for_status()
            
            # Parse the OMNIWeb text format
            lines = resp.text.strip().split('\n')
            
            # Find data lines (start with year)
            data_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        year = int(parts[0])
                        if 1900 < year < 2100:
                            data_lines.append(parts)
                    except ValueError:
                        continue
            
            if data_lines:
                import pandas as pd
                import numpy as np
                
                if resolution == 'hourly':
                    columns = ['Year', 'DOY', 'Hour', 'F107', 'Np', 'Vsw', 'Bt', 'Bz_GSM', 'Kp', 'Dst', 'AE']
                else:
                    columns = ['Year', 'DOY', 'Hour', 'Minute', 'F107', 'Np', 'Vsw', 'Bt', 'Bz_GSM', 'Kp', 'Dst', 'AE', 'SYM_H']
                
                # Trim columns to match data
                max_cols = min(len(columns), len(data_lines[0]))
                columns = columns[:max_cols]
                
                df = pd.DataFrame(data_lines, columns=columns[:len(data_lines[0])])
                df = df.apply(pd.to_numeric, errors='coerce')
                
                # Replace fill values (OMNIWeb uses 999.9, 9999, etc.)
                fill_values = {
                    'Kp': 99, 'Dst': 99999, 'AE': 99999, 'SYM_H': 99999,
                    'F107': 999.9, 'Vsw': 9999.9, 'Np': 999.99, 'Bt': 999.9,
                    'Bz_GSM': 999.9
                }
                for col, fv in fill_values.items():
                    if col in df.columns:
                        df.loc[df[col] >= fv, col] = np.nan
                
                df.to_csv(output_file, index=False)
                logger.info(f"OMNIWeb data saved: {output_file} ({len(df)} records)")
                return str(output_file)
            else:
                logger.error("No data lines found in OMNIWeb response")
                # Save raw response for debugging
                raw_file = output_dir / f'omniweb_raw_{start_year}_{end_year}.txt'
                with open(raw_file, 'w') as f:
                    f.write(resp.text)
                logger.info(f"Raw response saved to {raw_file}")
                return str(raw_file)
                
        except Exception as e:
            logger.error(f"OMNIWeb download failed: {e}")
            return ""
    
    # =========================================================================
    # GRASP/GSAT-19 Data (Manual)
    # =========================================================================
    
    @staticmethod
    def print_grasp_instructions():
        """Print instructions for manual GRASP data download."""
        instructions = """
==================================================================
           GRASP/GSAT-19 Data Download Instructions              
==================================================================
                                                                  
  1. Visit: https://pradan.issdc.gov.in                          
  2. Register / Log in to the portal                              
  3. Navigate to GSAT-19 -> GRASP payload                         
  4. Select time range: 2017-06 to 2019-06 (or available)        
  5. Download electron flux data (CDF or CSV format)              
  6. Place files in: data/raw/grasp/                             
                                                                  
  Alternative: https://www.issdc.gov.in -> ISDA -> GSAT-19         
                                                                  
  The pipeline will auto-detect GRASP files in data/raw/grasp/   
==================================================================
        """
        print(instructions)
    
    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def _list_remote_files(self, url: str, extension: str = '.cdf') -> List[str]:
        """
        List files at a remote HTTP directory listing.
        
        Args:
            url: URL of the directory listing page.
            extension: File extension to filter for.
        
        Returns:
            List of filenames.
        """
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            
            # Parse HTML directory listing for file links
            pattern = rf'href="([^"]*{re.escape(extension)})"'
            files = re.findall(pattern, resp.text, re.IGNORECASE)
            
            # Clean up: remove path prefixes, keep just filenames
            files = [f.split('/')[-1] for f in files if not f.startswith('..')]
            
            return sorted(set(files))
            
        except requests.exceptions.RequestException as e:
            logger.debug(f"Could not list {url}: {e}")
            return []
    
    def _download_file(
        self,
        url: str,
        local_path: str,
        overwrite: bool = False
    ) -> bool:
        """
        Download a single file with resume support.
        
        Args:
            url: Remote file URL.
            local_path: Local path to save to.
            overwrite: Whether to re-download existing files.
        
        Returns:
            True if download successful or file already exists.
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Skip if already exists
        if local_path.exists() and not overwrite:
            logger.debug(f"  Already exists: {local_path.name}")
            return True
        
        try:
            resp = self.session.get(url, stream=True, timeout=60)
            resp.raise_for_status()
            
            total_size = int(resp.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f:
                downloaded = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.debug(f"  Downloaded: {local_path.name} ({size_mb:.1f} MB)")
            return True
            
        except Exception as e:
            logger.warning(f"  Download failed {local_path.name}: {e}")
            if local_path.exists():
                local_path.unlink()
            return False
    
    def download_all(self) -> Dict:
        """
        Download all required datasets.
        
        Returns:
            Dictionary with download results per source.
        """
        results = {}
        
        logger.info("=" * 60)
        logger.info("Starting full data download")
        logger.info("=" * 60)
        
        # GOES electron flux
        logger.info("\n[1/4] Downloading GOES electron flux data...")
        results['goes'] = self.download_goes()
        
        # Wind SWE
        logger.info("\n[2/4] Downloading Wind SWE solar wind data...")
        results['wind_swe'] = self.download_wind_swe()
        
        # Wind MFI
        logger.info("\n[3/4] Downloading Wind MFI magnetic field data...")
        results['wind_mfi'] = self.download_wind_mfi()
        
        # OMNIWeb
        logger.info("\n[4/4] Downloading OMNIWeb geomagnetic indices...")
        results['omniweb'] = self.download_omniweb_indices()
        
        # GRASP instructions
        self.print_grasp_instructions()
        
        logger.info("=" * 60)
        logger.info("Download summary:")
        for source, files in results.items():
            count = len(files) if isinstance(files, list) else (1 if files else 0)
            logger.info(f"  {source}: {count} files")
        logger.info("=" * 60)
        
        return results


# Convenience function for command-line usage
def download_all_data(config_path: Optional[str] = None):
    """Download all datasets using default or specified config."""
    config = Config(config_path) if config_path else Config()
    downloader = DataDownloader(config)
    return downloader.download_all()


if __name__ == "__main__":
    download_all_data()

"""
Data Preprocessor for ISRO PS14 Radiation Forecasting.

Handles:
- Spike removal (sigma-clipping)
- Solar Proton Event (SPE) filtering
- Gap interpolation (linear, cubic spline)
- Time alignment and resampling to 5-min cadence
- Solar wind propagation delay correction (L1 → bow shock)
- Merging GOES + Wind + OMNIWeb into unified dataset
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict
from scipy import interpolate

from src.utils.logger import get_logger
from src.utils.config import Config
from src.data.cdf_reader import CDFReader

logger = get_logger(__name__)


class DataPreprocessor:
    """
    Preprocesses raw space weather data into a clean, unified dataset
    suitable for ML training.
    """
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.preprocessing = self.config.preprocessing
        self.reader = CDFReader()
    
    # =========================================================================
    # 1. Load Raw Data from CDF Files
    # =========================================================================
    
    def load_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all raw data from CDF files in data/raw/ directories.
        
        Returns:
            Dictionary with DataFrames: 'goes', 'wind_swe', 'wind_mfi', 'omniweb', 'grasp'
        """
        raw_dir = Path(self.config.data.get('raw_dir', 'data/raw'))
        data = {}
        
        # GRASP (ISRO GSAT-19)
        grasp_dir = raw_dir / 'grasp'
        has_grasp = False
        if grasp_dir.exists():
            logger.info("Loading GRASP data...")
            grasp_files = sorted(grasp_dir.rglob('*.cdf')) + sorted(grasp_dir.rglob('*.csv')) + sorted(grasp_dir.rglob('*.txt'))
            if grasp_files:
                dfs = []
                for f in grasp_files:
                    df = self.reader.read_grasp(str(f))
                    if not df.empty:
                        dfs.append(df)
                if dfs:
                    data['grasp'] = pd.concat(dfs).sort_index()
                    data['grasp'] = data['grasp'][~data['grasp'].index.duplicated(keep='first')]
                    logger.info(f"  GRASP: {len(data['grasp'])} records")
                    has_grasp = True
                    
        # GOES electron flux (Skip if we have GRASP data to focus on ISRO satellite)
        goes_dir = raw_dir / 'goes'
        if goes_dir.exists() and not has_grasp:
            logger.info("Loading GOES electron flux data...")
            # Recursively find all CDF files
            cdf_files = sorted(goes_dir.rglob('*.cdf'))
            if cdf_files:
                dfs = []
                for f in cdf_files:
                    df = self.reader.read_goes_flux(str(f))
                    if not df.empty:
                        dfs.append(df)
                if dfs:
                    data['goes'] = pd.concat(dfs).sort_index()
                    data['goes'] = data['goes'][~data['goes'].index.duplicated(keep='first')]
                    logger.info(f"  GOES: {len(data['goes'])} records")
        elif goes_dir.exists() and has_grasp:
            logger.info("Skipping GOES data because GRASP data was found.")
        
        # Wind SWE
        swe_dir = raw_dir / 'wind_swe'
        if swe_dir.exists():
            logger.info("Loading Wind SWE data...")
            cdf_files = sorted(swe_dir.rglob('*.cdf'))
            if cdf_files:
                dfs = []
                for f in cdf_files:
                    df = self.reader.read_wind_swe(str(f))
                    if not df.empty:
                        dfs.append(df)
                if dfs:
                    data['wind_swe'] = pd.concat(dfs).sort_index()
                    data['wind_swe'] = data['wind_swe'][~data['wind_swe'].index.duplicated(keep='first')]
                    logger.info(f"  Wind SWE: {len(data['wind_swe'])} records")
        
        # Wind MFI
        mfi_dir = raw_dir / 'wind_mfi'
        if mfi_dir.exists():
            logger.info("Loading Wind MFI data...")
            cdf_files = sorted(mfi_dir.rglob('*.cdf'))
            if cdf_files:
                dfs = []
                for f in cdf_files:
                    df = self.reader.read_wind_mfi(str(f))
                    if not df.empty:
                        dfs.append(df)
                if dfs:
                    data['wind_mfi'] = pd.concat(dfs).sort_index()
                    data['wind_mfi'] = data['wind_mfi'][~data['wind_mfi'].index.duplicated(keep='first')]
                    logger.info(f"  Wind MFI: {len(data['wind_mfi'])} records")
        
        # OMNIWeb indices (CSV)
        omniweb_dir = raw_dir / 'omniweb'
        if omniweb_dir.exists():
            logger.info("Loading OMNIWeb indices...")
            csv_files = sorted(omniweb_dir.glob('omniweb_*.csv'))
            if csv_files:
                data['omniweb'] = self._load_omniweb_csv(str(csv_files[0]))
                logger.info(f"  OMNIWeb: {len(data['omniweb'])} records")
        
        # (GRASP data handled earlier)
        
        return data
    
    def _load_omniweb_csv(self, filepath: str) -> pd.DataFrame:
        """Load and parse OMNIWeb CSV file into DatetimeIndex DataFrame."""
        df = pd.read_csv(filepath)
        
        # Construct datetime from Year, DOY, Hour columns
        if 'Year' in df.columns and 'DOY' in df.columns:
            hour = df.get('Hour', 0)
            minute = df.get('Minute', 0)
            
            # Convert Year+DOY to datetime
            dates = pd.to_datetime(
                df['Year'].astype(str) + df['DOY'].astype(str).str.zfill(3),
                format='%Y%j'
            )
            dates = dates + pd.to_timedelta(hour, unit='h')
            if 'Minute' in df.columns:
                dates = dates + pd.to_timedelta(minute, unit='m')
            
            df.index = pd.DatetimeIndex(dates, name='datetime')
            df = df.drop(columns=['Year', 'DOY', 'Hour'] + 
                        (['Minute'] if 'Minute' in df.columns else []), 
                        errors='ignore')
        
        return df
    
    # =========================================================================
    # 2. Spike Removal
    # =========================================================================
    
    def remove_spikes(
        self,
        df: pd.DataFrame,
        column: str = 'electron_flux_gt2MeV',
        sigma: float = None,
        window_hours: float = None
    ) -> pd.DataFrame:
        """
        Remove spikes using sigma-clipping in rolling windows.
        
        Operates on log10(flux) to handle the log-normal distribution.
        
        Args:
            df: DataFrame with flux data.
            column: Column name to clean.
            sigma: Number of standard deviations for clipping.
            window_hours: Rolling window size in hours.
        
        Returns:
            Cleaned DataFrame (spikes replaced with NaN).
        """
        sigma = sigma or self.preprocessing.get('spike_removal', {}).get('sigma_threshold', 3.0)
        window_hours = window_hours or self.preprocessing.get('spike_removal', {}).get('window_hours', 6)
        
        if column not in df.columns:
            return df
        
        df = df.copy()
        original_count = df[column].notna().sum()
        
        # Work in log space
        log_flux = np.log10(df[column].clip(lower=1e-10))
        
        # Rolling statistics
        window = f'{int(window_hours)}h'
        rolling_mean = log_flux.rolling(window, center=True, min_periods=10).mean()
        rolling_std = log_flux.rolling(window, center=True, min_periods=10).std()
        
        # Flag points more than sigma*std from rolling mean
        deviation = np.abs(log_flux - rolling_mean)
        spike_mask = deviation > (sigma * rolling_std)
        
        # Also flag physically impossible values
        spike_mask |= df[column] > 1e8  # Unrealistically high
        spike_mask |= df[column] < 1e-4  # Below instrument threshold
        
        df.loc[spike_mask, column] = np.nan
        
        removed_count = original_count - df[column].notna().sum()
        logger.info(f"Spike removal: removed {removed_count} points "
                    f"({100*removed_count/max(original_count,1):.2f}%)")
        
        return df
    
    # =========================================================================
    # 3. Solar Proton Event (SPE) Filtering
    # =========================================================================
    
    def filter_spe_events(
        self,
        df: pd.DataFrame,
        proton_col: str = 'proton_flux_gt10MeV',
        flux_col: str = 'electron_flux_gt2MeV'
    ) -> pd.DataFrame:
        """
        Flag and remove electron flux data contaminated by Solar Proton Events.
        
        During SPEs, proton contamination makes electron flux readings unreliable.
        
        Args:
            df: DataFrame with both electron and proton flux.
            proton_col: Column name for proton flux.
            flux_col: Column name for electron flux.
        
        Returns:
            DataFrame with SPE-contaminated periods marked as NaN.
        """
        spe_config = self.preprocessing.get('spe_filter', {})
        threshold = spe_config.get('proton_threshold_pfu', 10)
        hours_before = spe_config.get('exclusion_hours_before', 2)
        hours_after = spe_config.get('exclusion_hours_after', 24)
        
        if proton_col not in df.columns:
            logger.info("No proton flux data available — skipping SPE filter")
            return df
        
        df = df.copy()
        
        # Find SPE periods
        spe_mask = df[proton_col] > threshold
        
        if spe_mask.any():
            # Expand the mask by hours_before and hours_after
            expanded_mask = spe_mask.copy()
            
            for idx in df.index[spe_mask]:
                start = idx - pd.Timedelta(hours=hours_before)
                end = idx + pd.Timedelta(hours=hours_after)
                expanded_mask |= (df.index >= start) & (df.index <= end)
            
            contaminated_count = expanded_mask.sum()
            df.loc[expanded_mask, flux_col] = np.nan
            
            logger.info(f"SPE filter: flagged {contaminated_count} points "
                       f"({100*contaminated_count/len(df):.2f}%)")
        else:
            logger.info("No SPE events detected")
        
        return df
    
    # =========================================================================
    # 4. Gap Interpolation
    # =========================================================================
    
    def interpolate_gaps(
        self,
        df: pd.DataFrame,
        columns: Optional[list] = None,
        max_linear_hours: float = None,
        max_spline_hours: float = None
    ) -> pd.DataFrame:
        """
        Interpolate data gaps with method selection based on gap duration.
        
        - Short gaps (≤ max_linear_hours): Linear interpolation
        - Medium gaps (≤ max_spline_hours): Cubic spline interpolation
        - Long gaps: Left as NaN
        
        All interpolation done in log-space for flux variables.
        
        Args:
            df: DataFrame with gaps (NaN values).
            columns: Columns to interpolate. If None, interpolates all.
            max_linear_hours: Maximum gap for linear interpolation.
            max_spline_hours: Maximum gap for spline interpolation.
        
        Returns:
            DataFrame with gaps filled where appropriate.
        """
        gap_config = self.preprocessing.get('gap_handling', {})
        max_linear_hours = max_linear_hours or gap_config.get('max_linear_interp_hours', 2)
        max_spline_hours = max_spline_hours or gap_config.get('max_spline_interp_hours', 6)
        
        if columns is None:
            columns = df.columns.tolist()
        
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col].copy()
            total_nans_before = series.isna().sum()
            
            if total_nans_before == 0:
                continue
            
            # Create a limit based on maximum allowed gap size in points
            cadence_minutes = self.config.data.get('resolution_minutes', 5)
            linear_limit = int(max_linear_hours * 60 / cadence_minutes)
            spline_limit = int(max_spline_hours * 60 / cadence_minutes)
            
            # Identify gap boundaries and sizes
            is_nan = series.isna()
            if not is_nan.any():
                continue
                
            gap_groups = (is_nan != is_nan.shift()).cumsum()
            # Calculate size of each gap
            gap_sizes = gap_groups.map(gap_groups.value_counts())
            
            # Create masks for gaps that fall within our limits
            linear_mask = is_nan & (gap_sizes <= linear_limit)
            spline_mask = is_nan & (gap_sizes > linear_limit) & (gap_sizes <= spline_limit)
            
            # Interpolate all gaps (we will only keep the valid ones)
            series_linear = series.interpolate(method='linear')
            try:
                series_spline = series.interpolate(method='cubic')
            except Exception:
                series_spline = series_linear.copy()
            
            # Apply interpolations only to the appropriate gaps
            series[linear_mask] = series_linear[linear_mask]
            series[spline_mask] = series_spline[spline_mask]
            
            df[col] = series
            filled = total_nans_before - df[col].isna().sum()
            logger.debug(f"  {col}: filled {filled}/{total_nans_before} gaps")
        
        return df
    
    # =========================================================================
    # 5. Time Alignment & Resampling
    # =========================================================================
    
    def resample_to_cadence(
        self,
        df: pd.DataFrame,
        cadence_minutes: int = None
    ) -> pd.DataFrame:
        """
        Resample data to uniform time cadence.
        
        Uses median for flux variables (robust to outliers),
        mean for other variables.
        
        Args:
            df: DataFrame with DatetimeIndex.
            cadence_minutes: Target cadence in minutes.
        
        Returns:
            Resampled DataFrame.
        """
        cadence_minutes = cadence_minutes or self.config.data.get('resolution_minutes', 5)
        cadence = f'{cadence_minutes}min'
        
        logger.info(f"Resampling to {cadence} cadence...")
        
        # Identify flux columns for median resampling
        flux_cols = [c for c in df.columns if 'flux' in c.lower()]
        other_cols = [c for c in df.columns if c not in flux_cols]
        
        resampled_parts = []
        
        if flux_cols:
            flux_resampled = df[flux_cols].resample(cadence).median()
            resampled_parts.append(flux_resampled)
        
        if other_cols:
            other_resampled = df[other_cols].resample(cadence).mean()
            resampled_parts.append(other_resampled)
        
        if resampled_parts:
            result = pd.concat(resampled_parts, axis=1)
        else:
            result = df.resample(cadence).mean()
        
        logger.info(f"  Resampled: {len(df)} → {len(result)} records")
        return result
    
    # =========================================================================
    # 6. Solar Wind Propagation Delay
    # =========================================================================
    
    def apply_propagation_delay(
        self,
        wind_df: pd.DataFrame,
        vsw_col: str = 'Vsw'
    ) -> pd.DataFrame:
        """
        Apply ballistic propagation delay to shift Wind (L1) data to bow shock.
        
        The Wind spacecraft is at L1, ~1.5 million km (235 Re) from Earth.
        Solar wind takes Δt = ΔX / Vsw to travel from L1 to bow shock.
        
        Args:
            wind_df: DataFrame with solar wind data from Wind spacecraft.
            vsw_col: Column name for solar wind speed.
        
        Returns:
            Time-shifted DataFrame.
        """
        prop_config = self.preprocessing.get('propagation', {})
        l1_distance_Re = prop_config.get('l1_distance_Re', 235.0)
        default_vsw = prop_config.get('default_vsw_kms', 400.0)
        
        Re_km = 6371.0  # Earth radius in km
        distance_km = l1_distance_Re * Re_km  # ~1.5 million km
        
        df = wind_df.copy()
        
        if vsw_col in df.columns:
            # Variable delay based on actual solar wind speed
            vsw = df[vsw_col].fillna(default_vsw).clip(lower=200)
            delay_seconds = distance_km / vsw  # seconds
            
            # Apply individual time shifts
            new_times = df.index + pd.to_timedelta(delay_seconds, unit='s')
            df.index = new_times
            
            avg_delay_min = delay_seconds.mean() / 60
            logger.info(f"Propagation delay applied: avg {avg_delay_min:.0f} min "
                       f"(range: {delay_seconds.min()/60:.0f}-{delay_seconds.max()/60:.0f} min)")
        else:
            # Constant delay with default speed
            delay_seconds = distance_km / default_vsw
            df.index = df.index + pd.Timedelta(seconds=delay_seconds)
            logger.info(f"Constant propagation delay applied: {delay_seconds/60:.0f} min")
        
        return df
    
    # =========================================================================
    # 7. Full Merge Pipeline
    # =========================================================================
    
    def merge_datasets(
        self,
        goes_df: pd.DataFrame,
        wind_swe_df: pd.DataFrame = None,
        wind_mfi_df: pd.DataFrame = None,
        omniweb_df: pd.DataFrame = None,
        cadence_minutes: int = None
    ) -> pd.DataFrame:
        """
        Merge all data sources into a single time-aligned DataFrame.
        
        Args:
            goes_df: GOES electron flux data.
            wind_swe_df: Wind SWE solar wind data.
            wind_mfi_df: Wind MFI magnetic field data.
            omniweb_df: OMNIWeb geomagnetic indices.
            cadence_minutes: Target time cadence.
        
        Returns:
            Merged, time-aligned DataFrame.
        """
        cadence_minutes = cadence_minutes or self.config.data.get('resolution_minutes', 5)
        cadence = f'{cadence_minutes}min'
        
        logger.info("Merging all datasets...")
        
        # Start with GOES as the base (it defines our time range)
        merged = self.resample_to_cadence(goes_df, cadence_minutes)
        
        # Apply propagation delay to Wind data and merge
        if wind_swe_df is not None and not wind_swe_df.empty:
            swe_shifted = self.apply_propagation_delay(wind_swe_df)
            swe_resampled = self.resample_to_cadence(swe_shifted, cadence_minutes)
            merged = merged.join(swe_resampled, how='left', rsuffix='_swe')
        
        if wind_mfi_df is not None and not wind_mfi_df.empty:
            mfi_shifted = self.apply_propagation_delay(wind_mfi_df)
            mfi_resampled = self.resample_to_cadence(mfi_shifted, cadence_minutes)
            merged = merged.join(mfi_resampled, how='left', rsuffix='_mfi')
        
        # OMNIWeb indices (already at Earth, no propagation needed)
        if omniweb_df is not None and not omniweb_df.empty:
            omni_resampled = self.resample_to_cadence(omniweb_df, cadence_minutes)
            merged = merged.join(omni_resampled, how='left', rsuffix='_omni')
        
        # Remove duplicate columns
        merged = merged.loc[:, ~merged.columns.duplicated()]
        
        logger.info(f"Merged dataset: {len(merged)} records, {len(merged.columns)} columns")
        logger.info(f"  Columns: {list(merged.columns)}")
        logger.info(f"  Time range: {merged.index[0]} to {merged.index[-1]}")
        logger.info(f"  Missing data: {merged.isna().mean().mean()*100:.1f}% average")
        
        return merged
    
    # =========================================================================
    # 8. Full Processing Pipeline
    # =========================================================================
    
    def run_full_pipeline(self, save: bool = True) -> pd.DataFrame:
        """
        Execute the complete preprocessing pipeline.
        
        Steps:
        1. Load all raw CDF data
        2. Remove spikes from flux data
        3. Filter SPE events
        4. Merge and time-align all sources
        5. Interpolate remaining gaps
        6. Save processed Parquet
        
        Args:
            save: Whether to save the result as Parquet.
        
        Returns:
            Fully preprocessed DataFrame.
        """
        logger.info("=" * 60)
        logger.info("Running full preprocessing pipeline")
        logger.info("=" * 60)
        
        # Step 1: Load raw data
        raw_data = self.load_all_raw_data()
        
        if 'goes' not in raw_data or raw_data['goes'].empty:
            if 'grasp' in raw_data and not raw_data['grasp'].empty:
                logger.info("GOES data not found. Falling back to using GRASP/GSAT-19 data as the primary target!")
                raw_data['goes'] = raw_data['grasp'].rename(columns={'grasp_electron_flux': 'electron_flux_gt2MeV'})
            else:
                raise ValueError("GOES electron flux data is required but not found. "
                               "Run the data downloader first.")
        
        # Step 2: Spike removal on GOES data
        logger.info("\nStep 2: Removing spikes...")
        goes_clean = self.remove_spikes(raw_data['goes'])
        
        # Step 3: SPE filtering
        logger.info("\nStep 3: Filtering SPE events...")
        goes_clean = self.filter_spe_events(goes_clean)
        
        # Step 4: Merge datasets
        logger.info("\nStep 4: Merging datasets...")
        merged = self.merge_datasets(
            goes_df=goes_clean,
            wind_swe_df=raw_data.get('wind_swe'),
            wind_mfi_df=raw_data.get('wind_mfi'),
            omniweb_df=raw_data.get('omniweb'),
        )
        
        # Step 5: Interpolate gaps
        logger.info("\nStep 5: Interpolating gaps...")
        merged = self.interpolate_gaps(merged)
        
        # Step 6: Save
        if save:
            cache_path = self.config.data.get('cache_file', 'data/processed/merged_dataset.parquet')
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            merged.to_parquet(cache_path, engine='pyarrow')
            logger.info(f"\nSaved processed data to: {cache_path}")
            logger.info(f"  File size: {Path(cache_path).stat().st_size / 1e6:.1f} MB")
        
        logger.info("\n" + "=" * 60)
        logger.info("Preprocessing pipeline complete!")
        logger.info(f"  Records: {len(merged)}")
        logger.info(f"  Columns: {len(merged.columns)}")
        logger.info(f"  Time range: {merged.index[0]} to {merged.index[-1]}")
        logger.info("=" * 60)
        
        return merged
    
    def load_processed(self) -> pd.DataFrame:
        """Load previously processed data from Parquet cache."""
        cache_path = self.config.data.get('cache_file', 'data/processed/merged_dataset.parquet')
        if not Path(cache_path).exists():
            raise FileNotFoundError(
                f"Processed data not found at {cache_path}. "
                "Run preprocessor.run_full_pipeline() first."
            )
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded processed data: {len(df)} records, {len(df.columns)} columns")
        return df

import os
import pandas as pd
import numpy as np
from astropy.time import Time
import astropy.units as u

from redback.get_data.getter import DataGetter
import redback.get_data.directory
from redback.utils import logger, calc_flux_density_from_ABmag, \
    calc_flux_density_error_from_monochromatic_magnitude, \
    bandpass_magnitude_to_flux, bands_to_reference_flux, calc_flux_error_from_magnitude

# Try to import OTTER components
OTTER_INSTALLED = False
OTTER_IMPORT_ERROR = None

try:
    # Apply scipy compatibility fix before importing otter
    import sys
    import scipy.integrate
    if not hasattr(scipy.integrate, 'trapz'):
        # scipy >= 1.14 moved trapz to trapezoid
        scipy.integrate.trapz = scipy.integrate.trapezoid
        # Also patch it in sys.modules so submodules see it
        if 'scipy.integrate' in sys.modules:
            sys.modules['scipy.integrate'].trapz = scipy.integrate.trapezoid
    
    from otter import Otter
    from otter.io.transient import Transient as OtterTransient
    OTTER_INSTALLED = True
except ImportError as e:
    OTTER_IMPORT_ERROR = f"ImportError: {str(e)}"
    logger.debug(f"OTTER not available: {e}")
except Exception as e:
    OTTER_IMPORT_ERROR = f"Error: {str(e)}"
    logger.debug(f"OTTER import failed: {e}")


class OtterDataGetter(DataGetter):
    """
    Data getter for OTTER (Open multiwavelength Transient Event Repository)
    
    Follows redback pattern: converts raw OTTER data to CSV with ALL data modes
    (flux_density, flux, magnitude) computed in convert_raw_data_to_csv().
    The Transient class handles data_mode selection when loading.
    
    Parameters
    ----------
    transient : str
        Name of the transient (e.g., 'AT2017gfo', 'SN2011fe')
    transient_type : str
        Type of transient (kilonova, supernova, tidal_disruption_event)
    obs_type : str, optional
        Observation type to retrieve: 'uvoir' (default), 'radio', 'xray'
    """
    
    VALID_TRANSIENT_TYPES = [
        'kilonova', 'supernova', 'tidal_disruption_event'
    ]
    
    VALID_OBS_TYPES = ['uvoir', 'radio', 'xray']
    
    def __init__(self, transient: str, transient_type: str, obs_type: str = 'uvoir') -> None:
        """Constructor class for OTTER data getter.
        
        :param transient: Name of the transient, e.g., 'AT2017gfo'
        :type transient: str
        :param transient_type: Type of transient. 
                               Must be from `redback.get_data.otter.OtterDataGetter.VALID_TRANSIENT_TYPES`.
        :type transient_type: str
        :param obs_type: Observation type: 'uvoir', 'radio', or 'xray'. Default is 'uvoir'.
        :type obs_type: str, optional
        """
        if not OTTER_INSTALLED:
            error_msg = "OTTER is not available. "
            if 'OTTER_IMPORT_ERROR' in globals():
                error_msg += f"Import error: {OTTER_IMPORT_ERROR}. "
            error_msg += "Try: pip install astro-otter"
            raise ImportError(error_msg)
        
        if obs_type not in self.VALID_OBS_TYPES:
            raise ValueError(
                f"obs_type must be one of {self.VALID_OBS_TYPES}, got {obs_type}"
            )
        
        super().__init__(transient, transient_type)
        self._obs_type = obs_type
        
        # Create directory structure with obs_type as subdirectory (like Swift uses data_mode)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            self._create_directory_structure()
    
    @property
    def obs_type(self) -> str:
        """Return observation type"""
        return self._obs_type
    
    def _create_directory_structure(self):
        """Create directory structure based on obs_type"""
        # Base directory: transient_type/transient/obs_type/
        # Like Swift does: afterglow/GRB/flux/ or afterglow/GRB/counts/
        base_dir = f"{self.transient_type}/{self.transient}/{self.obs_type}/"
        raw_file = f"{base_dir}{self.transient}_rawdata.csv"
        processed_file = f"{base_dir}{self.transient}.csv"
        
        from collections import namedtuple
        DirectoryStructure = namedtuple('DirectoryStructure', 
                                       ['directory_path', 'raw_file_path', 'processed_file_path'])
        return DirectoryStructure(base_dir, raw_file, processed_file)
    
    @property
    def metadata_path(self):
        """
        :return: Path to the metadata file.
        :rtype: str
        """
        return f"{self.directory_path}{self.transient}_metadata.csv"
    
    def collect_data(self) -> None:
        """Query OTTER and save raw data"""
        if os.path.isfile(self.raw_file_path):
            logger.warning(f"Raw data file already exists: {self.raw_file_path}")
            return
        
        try:
            # Initialize OTTER connection
            otter = Otter()
            
            # Get metadata for this transient
            meta = otter.get_meta(names=self.transient)
            
            if len(meta) == 0:
                raise ValueError(
                    f"Transient {self.transient} not found in OTTER database"
                )
            
            meta_obj = meta[0]
            
            # Determine flux unit based on obs_type
            # UVOIR: magnitudes (AB)
            # Radio/X-ray: flux density (mJy)
            if self.obs_type == 'uvoir':
                flux_unit = 'mag(AB)'
            else:  # radio or xray
                flux_unit = 'mJy'
            
            # Get photometry
            phot = otter.get_phot(
                names=self.transient,
                obs_type=self.obs_type,
                return_type="pandas",
                flux_unit=flux_unit,
                date_unit="MJD"
            )
            
            if len(phot) == 0:
                raise ValueError(
                    f"No {self.obs_type} photometry found for {self.transient} in OTTER database"
                )
            
            # Save the raw photometry as CSV
            phot.to_csv(self.raw_file_path, index=False)
            
            # Save basic metadata
            metadata = {
                'redshift': meta_obj.get_redshift(),
                'ra': meta_obj.get_ra() if hasattr(meta_obj, 'get_ra') else None,
                'dec': meta_obj.get_dec() if hasattr(meta_obj, 'get_dec') else None,
                'discovery_date': str(meta_obj.get_discovery_date()),
                'classification': str(meta_obj.get_classification()) if hasattr(meta_obj, 'get_classification') else None,
                'obs_type': self.obs_type
            }
            pd.DataFrame([metadata]).to_csv(self.metadata_path, index=False)
            
            logger.info(f"Retrieved {self.obs_type} data for {self.transient} from OTTER")
            
        except Exception as e:
            logger.error(f"Failed to retrieve data from OTTER: {e}")
            raise
    
    def convert_raw_data_to_csv(self) -> pd.DataFrame:
        """
        Convert OTTER data to redback format with ALL data modes.
        
        Following OpenDataGetter pattern:
        - Compute magnitude, flux_density, and flux columns
        - Add time relative to discovery/trigger
        - Include band and system information
        
        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        
        if os.path.isfile(self.processed_file_path):
            logger.info(f"Processed file exists: {self.processed_file_path}")
            return pd.read_csv(self.processed_file_path)
        
        # Load raw photometry data
        phot = pd.read_csv(self.raw_file_path)
        
        # Load metadata
        metadata = pd.read_csv(self.metadata_path)
        
        logger.info(f'Processing data for transient {self.transient}.')
        
        # Convert to redback format (ALL data modes in one CSV)
        data = self._convert_to_redback_format(phot, metadata)
        
        # Save processed data
        data.to_csv(self.processed_file_path, index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')
        
        return data
    
    def _convert_to_redback_format(
        self, 
        phot: pd.DataFrame, 
        metadata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert OTTER photometry DataFrame to redback expected format.
        
        Handles different obs_types:
        - uvoir: magnitude (AB) → convert to flux_density and flux
        - radio/xray: flux_density (mJy) → convert to magnitude and flux
        
        OTTER columns:
        - converted_date (MJD)
        - converted_flux (magnitude for uvoir, mJy for radio/xray)
        - converted_flux_err
        - converted_freq (GHz) - for radio/xray
        - filter_name (filter name) - for uvoir
        - upperlimit (boolean)
        
        Redback expects:
        - time (MJD) - absolute MJD for phase models
        - time (days) - relative to discovery
        - magnitude + e_magnitude (for optical)
        - flux_density(mjy) + flux_density_error
        - flux(erg/cm2/s) + flux_error
        - band or frequency
        - system (AB for optical)
        
        :param phot: OTTER photometry DataFrame
        :type phot: pandas.DataFrame
        :param metadata: Metadata DataFrame
        :type metadata: pandas.DataFrame
        :return: Converted data in redback format
        :rtype: pandas.DataFrame
        """
        
        # Filter to only detections (not upper limits) and valid data
        data = phot.copy()
        if 'upperlimit' in data.columns:
            data = data[data['upperlimit'] == False].copy()
        
        # Remove rows with NaN flux values
        data = data.dropna(subset=['converted_flux', 'converted_flux_err'])
        
        # Get observation type from metadata
        obs_type = metadata['obs_type'].iloc[0] if 'obs_type' in metadata.columns else self.obs_type
        
        # Get time reference (discovery date or first detection)
        if not pd.isna(metadata['discovery_date'].iloc[0]) and metadata['discovery_date'].iloc[0] != 'None':
            try:
                time_ref = Time(metadata['discovery_date'].iloc[0])
            except:
                logger.warning("Could not parse discovery date, using first photometry point")
                time_ref = Time(data['converted_date'].min(), format='mjd')
        else:
            logger.warning("No discovery date found, using first photometry point")
            time_ref = Time(data['converted_date'].min(), format='mjd')
        
        # Calculate time since reference
        times_mjd = Time(data['converted_date'].values, format='mjd')
        time_days = (times_mjd - time_ref).to(u.day).value
        
        # Build output based on observation type
        if obs_type == 'uvoir':
            output = self._convert_uvoir_data(data, time_days)
        else:  # radio or xray
            output = self._convert_radio_xray_data(data, time_days, obs_type)
        
        # Add common columns
        output['time'] = data['converted_date'].values
        output = output[['time', 'time (days)'] + [col for col in output.columns if col not in ['time', 'time (days)']]]
        
        # Add metadata if available
        if not pd.isna(metadata['redshift'].iloc[0]):
            output['redshift'] = metadata['redshift'].iloc[0]
        
        return output
    
    def _convert_uvoir_data(self, data: pd.DataFrame, time_days: np.ndarray) -> pd.DataFrame:
        """Convert UV/optical/IR magnitude data to all formats"""
        
        output = pd.DataFrame({
            'time (days)': time_days,
            'magnitude': data['converted_flux'].values,
            'e_magnitude': data['converted_flux_err'].values,
            'band': data['filter_name'].values,
            'system': 'AB'
        })
        
        # Compute flux_density from magnitude
        output['flux_density(mjy)'] = calc_flux_density_from_ABmag(
            output['magnitude'].values).value
        output['flux_density_error'] = calc_flux_density_error_from_monochromatic_magnitude(
            magnitude=output['magnitude'].values, 
            magnitude_error=output['e_magnitude'].values, 
            reference_flux=3631,
            magnitude_system='AB')
        
        # Compute flux from magnitude
        output['flux(erg/cm2/s)'] = bandpass_magnitude_to_flux(
            output['magnitude'].values, 
            output['band'].values)
        output['flux_error'] = calc_flux_error_from_magnitude(
            magnitude=output['magnitude'].values,
            magnitude_error=output['e_magnitude'].values,
            reference_flux=bands_to_reference_flux(output['band'].values))
        
        return output
    
    def _convert_radio_xray_data(self, data: pd.DataFrame, time_days: np.ndarray, obs_type: str) -> pd.DataFrame:
        """Convert radio/X-ray flux density data to all formats"""
        
        output = pd.DataFrame({
            'time (days)': time_days,
            'flux_density(mjy)': data['converted_flux'].values,
            'flux_density_error': data['converted_flux_err'].values,
            'frequency': data['converted_freq'].values if 'converted_freq' in data.columns else None,
        })
        
        # For radio/X-ray, we typically work in flux_density mode
        # Magnitude conversion may not be meaningful for all bands
        # Leave magnitude columns as NaN for now
        output['magnitude'] = np.nan
        output['e_magnitude'] = np.nan
        output['band'] = f"{obs_type}_band"
        output['system'] = 'flux_density'
        
        # Compute flux from flux_density (if frequency is available)
        # flux (erg/s/cm2) = flux_density (mJy) * frequency_bandwidth
        # For single-frequency observations, we can't directly convert
        # Leave as NaN for now - user can compute if needed
        output['flux(erg/cm2/s)'] = np.nan
        output['flux_error'] = np.nan
        
        return output

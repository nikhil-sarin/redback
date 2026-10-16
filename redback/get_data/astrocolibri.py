import io
import json
import os
import time
from typing import Union

import astropy.units as uu
import numpy as np
import pandas as pd
import requests
from astropy.time import Time

import redback
import redback.get_data.directory
from redback.get_data.getter import DataGetter
from redback.utils import logger, calc_flux_density_from_ABmag, \
    calc_flux_density_error_from_monochromatic_magnitude, bandpass_magnitude_to_flux, bands_to_reference_flux, \
    calc_flux_error_from_magnitude

dirname = os.path.dirname(__file__)

ASTRO_COLIBRI_URL = "https://astro-colibri.science"
UID_ENV_VAR = "ASTROCOLIBRI_UID"
UID_INSTRUCTIONS = (
    "Astro-COLIBRI lightcurve requests require a personal user id. Get yours from "
    "https://astro-colibri.com/account and pass it as `uid=...` or set the "
    f"`{UID_ENV_VAR}` environment variable.")


class AstroColibriDataGetter(DataGetter):
    """Data getter for the Astro-COLIBRI API (https://astro-colibri.science).

    Downloads multi-survey optical photometry (ATLAS, ZTF, LSST, ASAS-SN and AAVSO)
    for any transient known to Astro-COLIBRI through the /lightcurve endpoint, and
    event metadata (trigger id, trigger time, position, redshift) through the free
    /event endpoint. The processed data follow the same format as the other optical
    data getters, plus `detected` and `limiting_magnitude` columns following the
    simulated-data convention, so non-detections can be kept as upper limits.

    Lightcurve requests count against the daily Astro-COLIBRI API quota of the
    account (each request costs several units); event lookups are free.

    Note that ASAS-SN provides apparent magnitudes (host + transient) while
    ZTF/LSST/ATLAS provide difference magnitudes. The AAVSO and ASAS-SN
    Johnson/Cousins bands (B, V, R, I) are natively in the Vega system and are
    converted to AB magnitudes (offsets computed from the sncosmo bandpasses),
    so the processed data are all in the AB system. Check the data before
    combining surveys.
    """

    VALID_TRANSIENT_TYPES = ["afterglow", "kilonova", "supernova", "tidal_disruption_event", "unknown"]

    ASTROCOLIBRI_TO_REDBACK_BANDS = {
        "ztf_g": "ztfg", "ztf_r": "ztfr", "ztf_i": "ztfi",
        "lsst_u": "lsstu", "lsst_g": "lsstg", "lsst_r": "lsstr",
        "lsst_i": "lssti", "lsst_z": "lsstz", "lsst_y": "lssty",
        "atlas_c": "atlasc", "atlas_o": "atlaso",
        "asas-sn_g": "sdssg", "asas-sn_V": "bessellv",
        "aavso_B": "bessellb", "aavso_V": "bessellv", "aavso_R": "bessellr",
        "aavso_I": "besselli",
    }

    DETECTION_FLAGS = ["good", "original", "average"]
    UPPER_LIMIT_FLAG = "ul"
    DROPPED_FLAGS = ["bad", "magerr_missing"]

    # AAVSO and ASAS-SN Johnson/Cousins photometry is reported in the Vega
    # system. All processed data are converted to AB (see
    # VEGA_TO_AB_OFFSETS) so the file is internally consistent, like the
    # other optical data getters.
    VEGA_BANDS = ["bessellb", "bessellv", "bessellr", "besselli"]

    @classmethod
    def _vega_to_ab_offsets(cls) -> dict:
        """Computes the Vega-to-AB magnitude offset per band.

        The offset satisfies AB = Vega + offset and is computed from the
        sncosmo bandpasses and magnitude systems.

        :return: The offset per redback band name.
        :rtype: dict
        """
        import sncosmo

        vega = sncosmo.get_magsystem("vega")
        ab = sncosmo.get_magsystem("ab")
        offsets = {}
        for band in cls.VEGA_BANDS:
            bandpass = sncosmo.get_bandpass(band)
            offsets[band] = float(
                -2.5 * np.log10(vega.zpbandflux(bandpass) / ab.zpbandflux(bandpass)))
        return offsets

    COMPLETED_STATUSES = ["completed", "complete", "done", "success", "finished"]
    FAILED_STATUSES = ["failed", "error", "cancelled", "canceled"]

    POLL_INTERVAL = 5.0

    PROCESSED_FILE_COLUMNS = [
        "time", "magnitude", "e_magnitude", "system", "band",
        "flux_density(mjy)", "flux_density_error",
        "flux(erg/cm2/s)", "flux_error", "time (days)",
        "detected", "limiting_magnitude",
    ]

    _decoder = json.JSONDecoder()

    def __init__(self, transient: str, transient_type: str, uid: str = None,
                 start: Union[str, None] = None, end: Union[str, None] = None,
                 ztf: bool = True, lsst: bool = True, atlas: bool = True,
                 asas_sn: bool = True, aavso: bool = True, aavso_all_filters: bool = False,
                 rapas: bool = False, include_upper_limits: bool = False,
                 request_timeout: float = 60.0, polling_timeout: float = 600.0,
                 **kwargs: None) -> None:
        """Constructor class for a data getter. The instance will be able to download
        Astro-COLIBRI data for the given transient.

        :param transient: Name of the transient, e.g. 'AT2017gfo', 'SN2026ejy' or 'GRB250101A'.
                          Any alias known to Astro-COLIBRI works.
        :type transient: str
        :param transient_type: Type of the transient. Must be from
                               `redback.get_data.astrocolibri.AstroColibriDataGetter.VALID_TRANSIENT_TYPES`.
        :type transient_type: str
        :param uid: Astro-COLIBRI user id from https://astro-colibri.com/account.
                    If None, it is read from the ASTROCOLIBRI_UID environment variable.
        :type uid: str, optional
        :param start: Start of the requested time window, e.g. '2024-05-02'.
                      Defaults to 5 days before the event.
        :type start: str, optional
        :param end: End of the requested time window, e.g. '2024-06-06'.
                    Defaults to 30 days after the event.
        :type end: str, optional
        :param ztf: Whether ZTF data should be requested. (Default value = True)
        :type ztf: bool, optional
        :param lsst: Whether LSST data should be requested. (Default value = True)
        :type lsst: bool, optional
        :param atlas: Whether ATLAS data should be requested. (Default value = True)
        :type atlas: bool, optional
        :param asas_sn: Whether ASAS-SN data should be requested. (Default value = True)
        :type asas_sn: bool, optional
        :param aavso: Whether AAVSO data should be requested. (Default value = True)
        :type aavso: bool, optional
        :param aavso_all_filters: Whether to include all other AAVSO filters,
                                  including visual estimates. (Default value = False)
        :type aavso_all_filters: bool, optional
        :param rapas: Whether RAPAS data should be requested (members only). (Default value = False)
        :type rapas: bool, optional
        :param include_upper_limits: If True, non-detections are kept in the processed
                                     data with `detected=0` and the limiting magnitude in
                                     `limiting_magnitude`. If False (default), non-detections
                                     are dropped like in the other optical data getters.
        :type include_upper_limits: bool, optional
        :param request_timeout: Timeout in seconds for API requests. (Default value = 60.0)
        :type request_timeout: float, optional
        :param polling_timeout: Timeout in seconds for waiting on lightcurve generation. (Default value = 600.0)
        :type polling_timeout: float, optional
        :param kwargs: Placeholder to prevent TypeErrors.
        :type kwargs: None
        """
        super().__init__(transient, transient_type)
        self.directory_path, self.raw_file_path, self.processed_file_path = \
            redback.get_data.directory.astrocolibri_directory_structure(
                transient=self.transient, transient_type=self.transient_type)
        self.uid = uid if uid is not None else (os.environ.get(UID_ENV_VAR) or None)
        self.start = start
        self.end = end
        self.surveys = {"ztf": ztf, "lsst": lsst, "atlas": atlas, "asas-sn": asas_sn,
                        "aavso": aavso, "aavso_all_filters": aavso_all_filters, "rapas": rapas}
        self.include_upper_limits = include_upper_limits
        self.request_timeout = request_timeout
        self.polling_timeout = polling_timeout
        self.event_metadata = None

    @property
    def url(self) -> str:
        """
        :return: The Astro-COLIBRI lightcurve url.
        :rtype: str
        """
        return f"{ASTRO_COLIBRI_URL}/lightcurve"

    @property
    def event_url(self) -> str:
        """
        :return: The Astro-COLIBRI event url.
        :rtype: str
        """
        return f"{ASTRO_COLIBRI_URL}/event"

    @property
    def metadata_path(self) -> str:
        """
        :return: The path to the Astro-COLIBRI event metadata file.
        :rtype: str
        """
        return f"{self.directory_path}{self.transient}_colibri_event.json"

    @classmethod
    def _decode_documents(cls, buffer: str) -> tuple:
        """Decodes as many concatenated JSON documents as possible from the buffer.

        The Astro-COLIBRI API returns several JSON objects in a single response body,
        without a separator, so json.loads cannot be used directly.

        :param buffer: The text to decode.
        :return: The decoded documents and the undecodable remainder.
        :rtype: tuple[list, str]
        """
        documents = []
        remainder = buffer.strip()
        while remainder:
            try:
                document, index = cls._decoder.raw_decode(remainder)
            except ValueError:
                break
            documents.append(document)
            remainder = remainder[index:].strip()
        return documents, remainder

    @classmethod
    def parse_json_documents(cls, text: str) -> list:
        """Parses all JSON documents from an Astro-COLIBRI API response body.

        :param text: The response body.
        :type text: str
        :return: All JSON documents that could be decoded.
        :rtype: list
        """
        documents, _ = cls._decode_documents(text)
        return documents

    @classmethod
    def iter_json_documents(cls, response):
        """Yields JSON documents from a streaming Astro-COLIBRI API response.

        The /lightcurve_status endpoint streams status documents as they are
        produced, so documents are decoded incrementally from the stream.

        :param response: A streaming requests response.
        :return: A generator yielding one dictionary per JSON document.
        """
        buffer = ""
        for chunk in response.iter_content(chunk_size=512, decode_unicode=True):
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="ignore")
            buffer += chunk
            documents, buffer = cls._decode_documents(buffer)
            for document in documents:
                yield document

    def get_event_metadata(self) -> dict:
        """Downloads the Astro-COLIBRI event metadata and saves it as a JSON file.

        The metadata contain the trigger id, trigger time, position, redshift and
        classification of the event. The lookup is free and does not require a
        user id. The metadata are used to reference the processed data times to
        the trigger time of the event.

        :return: The event metadata.
        :rtype: dict
        """
        if self.event_metadata is not None:
            return self.event_metadata
        if os.path.isfile(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.event_metadata = json.load(f)
            return self.event_metadata

        response = requests.get(url=self.event_url, params={"source_name": self.transient},
                                timeout=self.request_timeout)
        documents = self.parse_json_documents(getattr(response, "text", ""))
        metadata = documents[0] if documents else {}
        if isinstance(metadata, list):
            metadata = metadata[0] if metadata else {}
        if "trigger_id" not in metadata and "source_name" not in metadata:
            message = metadata.get("message", "event not found")
            raise ValueError(
                f"Transient {self.transient} does not exist in the Astro-COLIBRI database. "
                f"Are you sure you are using the right alias? ({message})")
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Astro-COLIBRI metadata for {self.transient} saved to {self.metadata_path}.")
        self.event_metadata = metadata
        return metadata

    def collect_data(self) -> None:
        """Downloads the photometry from the Astro-COLIBRI API and saves it into the raw file path."""
        if os.path.isfile(self.raw_file_path):
            logger.warning('The raw data file already exists.')
            return None

        if self.uid is None:
            raise ValueError(UID_INSTRUCTIONS)

        logger.info(f"Collecting Astro-COLIBRI data for {self.transient}.")
        metadata = self.get_event_metadata()
        trigger_id = metadata.get("trigger_id")
        if trigger_id is None:
            raise ValueError(
                f"Could not resolve a trigger id for {self.transient} from Astro-COLIBRI. "
                f"Are you sure you are using the right alias?")

        params = {"trigger_id": trigger_id, "uid": self.uid}
        params.update({key: str(value).lower() for key, value in self.surveys.items()})
        if self.start is not None:
            params["start"] = self.start
        if self.end is not None:
            params["end"] = self.end

        response = requests.get(url=self.url, params=params, timeout=self.request_timeout)
        documents = self._raise_for_api_error(response)
        csv_url = self._find_csv_url(documents)

        if csv_url is None:
            lightcurve_id = None
            for document in documents:
                if "lightcurve_id" in document:
                    lightcurve_id = document["lightcurve_id"]
                    break
            if lightcurve_id is None:
                raise RuntimeError(
                    f"The Astro-COLIBRI API did not return a lightcurve or a lightcurve id "
                    f"for {self.transient}: {documents}")
            urls = self.wait_for_lightcurve(lightcurve_id)
            csv_url = urls.get("csv_url")
            if csv_url is None:
                # The status documents carry no download urls, so request the
                # lightcurve once more. The generation is finished by now, so
                # the API serves the result from its cache instead of starting
                # a new (quota-consuming) generation.
                logger.info(f"Requesting the finished lightcurve of {self.transient} "
                            "to get the download links.")
                response = requests.get(url=self.url, params=params, timeout=self.request_timeout)
                documents = self._raise_for_api_error(response)
                csv_url = self._find_csv_url(documents)
            if csv_url is None:
                raise RuntimeError(
                    f"The Astro-COLIBRI lightcurve for {self.transient} did not provide a CSV download.")

        logger.info(f"Downloading Astro-COLIBRI photometry for {self.transient}.")
        response = requests.get(url=csv_url, timeout=self.request_timeout)
        response.raise_for_status()
        try:
            data = pd.read_csv(io.BytesIO(response.content))
        except pd.errors.EmptyDataError:
            data = pd.DataFrame()
        if len(data) == 0:
            raise ValueError(
                f"The Astro-COLIBRI lightcurve for {self.transient} does not contain any photometry. "
                f"Try a wider time range or additional surveys.")
        data.to_csv(self.raw_file_path, index=False)
        logger.info(f"Retrieved data for {self.transient}.")

    @staticmethod
    def _raise_for_api_error(response) -> list:
        """Raises a RuntimeError with the API message for non-200 responses.

        :param response: The requests response.
        :return: The JSON documents of the response body.
        :rtype: list
        """
        documents = AstroColibriDataGetter.parse_json_documents(getattr(response, "text", ""))
        if response.status_code != 200:
            message = documents[0].get("message", "") if documents else getattr(response, "text", "")[:200]
            raise RuntimeError(
                f"The Astro-COLIBRI API returned status code {response.status_code}"
                f"{': ' + message if message else ''}")
        return documents

    @staticmethod
    def _find_csv_url(documents: list) -> Union[str, None]:
        """Finds a CSV download url in the API response documents.

        :param documents: The JSON documents of an API response.
        :type documents: list
        :return: The CSV url, or None if the response contains none.
        :rtype: str or None
        """
        for document in documents:
            if "csv_url" in document and document["csv_url"]:
                return document["csv_url"]
        return None

    def wait_for_lightcurve(self, lightcurve_id: str) -> dict:
        """Waits for an Astro-COLIBRI lightcurve generation to finish.

        Repeatedly connects to the /lightcurve_status endpoint and consumes
        status documents until a terminal status (completed or failed) is
        received. The endpoint streams status documents while the generation
        is running; if a connection ends without a terminal status, it is
        treated as an incomplete poll and reconnected, so both streaming and
        poll-style behaviour are supported. Any download urls found in the
        status documents are collected.

        :param lightcurve_id: The lightcurve id returned by the /lightcurve endpoint.
        :type lightcurve_id: str
        :return: Any download urls found, e.g. {'url': ..., 'csv_url': ..., 'votable_url': ...}.
        :rtype: dict
        """
        status_url = f"{ASTRO_COLIBRI_URL}/lightcurve_status/{lightcurve_id}"
        logger.info(f"Waiting for the Astro-COLIBRI lightcurve of {self.transient} to be generated.")
        urls = {}
        deadline = time.monotonic() + self.polling_timeout
        while True:
            response = requests.get(url=status_url, stream=True, timeout=self.polling_timeout)
            try:
                response.raise_for_status()
                terminal = False
                for document in self.iter_json_documents(response):
                    for key in ("url", "csv_url", "votable_url"):
                        if document.get(key):
                            urls[key] = document[key]
                    status = str(document.get("status", "")).lower()
                    if "progress" in document:
                        logger.debug(f"Lightcurve generation for {self.transient}: "
                                     f"{document.get('message', '')} ({document.get('progress')})")
                    if status in self.FAILED_STATUSES:
                        raise RuntimeError(
                            f"The Astro-COLIBRI lightcurve generation for {self.transient} failed: "
                            f"{document.get('message', status)}")
                    if status in self.COMPLETED_STATUSES:
                        terminal = True
                        break
                if terminal:
                    return urls
                # the connection ended without a terminal status; the generation
                # is still running, so poll again until the deadline
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"The Astro-COLIBRI lightcurve generation for {self.transient} did not "
                        f"finish within {self.polling_timeout} seconds.")
                time.sleep(self.POLL_INTERVAL)
            finally:
                response.close()

    def get_time_of_event(self, data: pd.DataFrame) -> Time:
        """Gets the trigger time of the event from the Astro-COLIBRI metadata.

        Falls back to 0.1 days before the first data point if no trigger time
        is available, like the other optical data getters.

        :param data: The half-processed data.
        :type data: pandas.DataFrame
        :return: The time of the event.
        :rtype: astropy.time.Time
        """
        trigger_time = None
        try:
            metadata = self.get_event_metadata()
            trigger_time = metadata.get("time")
        except (ValueError, requests.RequestException, OSError) as e:
            logger.warning(f"Could not retrieve Astro-COLIBRI metadata: {e}")
        if trigger_time:
            # the API returns 'YYYY-MM-DD HH:MM:SS.fff'; astropy isot needs a 'T' separator
            trigger_time = str(trigger_time).strip().replace(" ", "T")
            return Time(trigger_time, format="isot", scale="utc")
        logger.warning("No trigger time in Astro-COLIBRI metadata. "
                       "Temporarily using 0.1d before the first data point as a start time.")
        return Time(min(data["time"]) - 0.1, format='mjd')

    def convert_raw_data_to_csv(self) -> Union[pd.DataFrame, None]:
        """Converts the raw data into processed data and saves it into the processed file path.
        The data columns are in `AstroColibriDataGetter.PROCESSED_FILE_COLUMNS`.

        :return: The processed data.
        :rtype: pandas.DataFrame
        """
        if os.path.isfile(self.processed_file_path):
            logger.warning('The processed data file already exists. Returning.')
            return pd.read_csv(self.processed_file_path)

        raw_data = pd.read_csv(self.raw_file_path)
        logger.info('Processing data for transient {}.'.format(self.transient))

        required_columns = ["mjd", "filter", "mag", "mag_err", "flag"]
        missing_columns = [column for column in required_columns if column not in raw_data.columns]
        if missing_columns:
            raise ValueError(
                f"The Astro-COLIBRI raw data for {self.transient} is missing columns: {missing_columns}.")

        keep_flags = list(self.DETECTION_FLAGS)
        if self.include_upper_limits:
            keep_flags.append(self.UPPER_LIMIT_FLAG)
        dropped_flags = sorted(set(raw_data["flag"].dropna()) - set(keep_flags))
        if dropped_flags:
            logger.info(f"Dropping data points with flags {dropped_flags}.")
        data = raw_data[raw_data["flag"].isin(keep_flags)].copy()
        if "mag_ul" not in data.columns:
            data["mag_ul"] = np.nan

        bands = data["filter"].map(self.ASTROCOLIBRI_TO_REDBACK_BANDS)
        unknown_bands = sorted(set(data.loc[bands.isna(), "filter"].dropna()))
        if unknown_bands:
            logger.warning(
                f"Dropping bands that are not defined in redback: {unknown_bands}. "
                f"Request different surveys if you need them.")
        keep_bands = bands.notna()
        data = data[keep_bands].copy()
        data["band"] = bands[keep_bands]

        if any(band in self.VEGA_BANDS for band in data["band"]):
            logger.warning(
                "The lightcurve contains ASAS-SN and/or AAVSO photometry. ASAS-SN provides apparent "
                "(host+transient) magnitudes, unlike the difference magnitudes of ZTF/LSST/ATLAS. "
                "The Johnson/Cousins bands (B, V, R, I) of AAVSO and ASAS-SN are natively in the "
                "Vega system and are converted to AB magnitudes here (offsets computed from the "
                "sncosmo bandpasses). Check the data before combining it with other surveys.")

        is_upper_limit = (data["flag"] == self.UPPER_LIMIT_FLAG) | (
            data["mag"].isna() & data["mag_ul"].notna())
        data["detected"] = (~is_upper_limit).astype(int)
        data["limiting_magnitude"] = np.where(is_upper_limit, data["mag_ul"], np.nan)
        data["magnitude"] = np.where(is_upper_limit, np.nan, data["mag"])
        data["e_magnitude"] = np.where(is_upper_limit, np.nan, data["mag_err"])
        data["time"] = data["mjd"]
        data["system"] = "AB"

        # Convert the natively Vega-system magnitudes of the AAVSO and
        # ASAS-SN Johnson/Cousins bands to AB, so the processed data are
        # all in one system (matching the other optical data getters and
        # the flux computations below). AB = Vega + offset.
        vega_to_ab_offsets = self._vega_to_ab_offsets()
        vega_mask = data["band"].isin(vega_to_ab_offsets.keys())
        if vega_mask.any():
            offsets = data.loc[vega_mask, "band"].map(vega_to_ab_offsets)
            for column in ["magnitude", "limiting_magnitude"]:
                data.loc[vega_mask, column] = data.loc[vega_mask, column] + offsets

        # Keep detections with a magnitude and non-detections with a limiting magnitude.
        data = data[data["magnitude"].notna() | (data["detected"] == 0)].copy()

        magnitude = data["magnitude"].to_numpy(dtype=float)
        magnitude_error = data["e_magnitude"].to_numpy(dtype=float)
        detected = data["detected"].to_numpy(dtype=bool)
        limiting_magnitude = data["limiting_magnitude"].to_numpy(dtype=float)
        band_values = data["band"].to_numpy()
        flux_density = np.full(len(data), np.nan)
        flux_density_error = np.full(len(data), np.nan)
        flux = np.full(len(data), np.nan)
        flux_error = np.full(len(data), np.nan)
        if detected.any():
            flux_density[detected] = calc_flux_density_from_ABmag(magnitude[detected]).value
            flux_density_error[detected] = calc_flux_density_error_from_monochromatic_magnitude(
                magnitude=magnitude[detected], magnitude_error=magnitude_error[detected],
                reference_flux=3631, magnitude_system="AB")
            flux[detected] = bandpass_magnitude_to_flux(magnitude[detected], band_values[detected])
            flux_error[detected] = calc_flux_error_from_magnitude(
                magnitude=magnitude[detected], magnitude_error=magnitude_error[detected],
                reference_flux=bands_to_reference_flux(band_values[detected]))
        upper_limits = ~detected
        if upper_limits.any():
            # Populate the flux values of non-detections from the limiting
            # magnitude, so upper-limit likelihoods get finite y-values. The
            # errors are left as NaN; the upper-limit likelihood does not use
            # the errors of non-detections.
            flux_density[upper_limits] = calc_flux_density_from_ABmag(
                limiting_magnitude[upper_limits]).value
            flux[upper_limits] = bandpass_magnitude_to_flux(
                limiting_magnitude[upper_limits], band_values[upper_limits])
        data["flux_density(mjy)"] = flux_density
        data["flux_density_error"] = flux_density_error
        data["flux(erg/cm2/s)"] = flux
        data["flux_error"] = flux_error

        data = data.sort_values(by="time")
        time_of_event = self.get_time_of_event(data=data)
        tt = Time(np.asarray(data["time"], dtype=float), format='mjd')
        data['time (days)'] = ((tt - time_of_event).to(uu.day)).value
        data = data.reset_index(drop=True)
        data = data[self.PROCESSED_FILE_COLUMNS]
        data.to_csv(self.processed_file_path, sep=',', index=False)
        logger.info(f'Congratulations, you now have a nice data file: {self.processed_file_path}')
        return data

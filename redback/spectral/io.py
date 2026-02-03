from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy.io import fits

from redback.utils import logger

@dataclass
class OGIPPHASpectrum:
    channel: np.ndarray
    counts: np.ndarray
    exposure: float
    backscale: float
    areascal: float
    quality: Optional[np.ndarray]
    grouping: Optional[np.ndarray]
    backfile: Optional[str]
    respfile: Optional[str]
    ancrfile: Optional[str]
    header: fits.Header


@dataclass
class OGIPLightCurve:
    time: np.ndarray
    rate: np.ndarray
    error: np.ndarray
    fracexp: Optional[np.ndarray]
    timedel: Optional[float]
    header: fits.Header


def _get_spectrum_hdu(hdul: fits.HDUList) -> fits.BinTableHDU:
    if "SPECTRUM" in hdul:
        return hdul["SPECTRUM"]
    for hdu in hdul:
        if isinstance(hdu, fits.BinTableHDU):
            if hdu.header.get("HDUCLAS1", "").upper() == "SPECTRUM":
                return hdu
    raise ValueError("No SPECTRUM HDU found in PHA file")


def _select_spectrum_row(table: fits.BinTableHDU, spectrum_index: Optional[int]) -> int:
    nrows = len(table.data)
    if nrows == 1:
        return 0
    if spectrum_index is None:
        return 0
    if not (0 <= spectrum_index < nrows):
        raise IndexError(f"spectrum_index={spectrum_index} out of range [0, {nrows - 1}]")
    return spectrum_index


def _get_column_or_header_value(table: fits.BinTableHDU, name: str, row: int, default: Optional[float] = None):
    if name in table.columns.names:
        value = table.data[name][row]
        if np.ndim(value) == 0:
            return float(value)
        return value
    if name in table.header:
        return float(table.header.get(name))
    return default


def read_pha(path: str, spectrum_index: Optional[int] = None) -> OGIPPHASpectrum:
    logger.info("Reading PHA file: %s", path)
    with fits.open(path) as hdul:
        spectrum_hdu = _get_spectrum_hdu(hdul)
        data = spectrum_hdu.data
        counts_col = data["COUNTS"]

        is_type1 = counts_col.ndim == 1 and len(counts_col) > 1 and np.ndim(counts_col[0]) == 0
        if is_type1:
            channel = np.asarray(data["CHANNEL"], dtype=int)
            counts = np.asarray(counts_col, dtype=float)
            row = 0
        else:
            row = _select_spectrum_row(spectrum_hdu, spectrum_index)
            channel = np.atleast_1d(data["CHANNEL"][row]).astype(int)
            counts = np.atleast_1d(counts_col[row]).astype(float)

        exposure = float(spectrum_hdu.header.get("EXPOSURE", 0.0))
        backscale = _get_column_or_header_value(spectrum_hdu, "BACKSCAL", row, default=1.0)
        areascal = _get_column_or_header_value(spectrum_hdu, "AREASCAL", row, default=1.0)

        quality = None
        if "QUALITY" in spectrum_hdu.columns.names:
            qcol = spectrum_hdu.data["QUALITY"]
            quality = np.asarray(qcol if is_type1 else qcol[row], dtype=int)
        elif "QUALITY" in spectrum_hdu.header:
            quality = np.asarray([int(spectrum_hdu.header.get("QUALITY"))] * len(channel), dtype=int)

        grouping = None
        if "GROUPING" in spectrum_hdu.columns.names:
            gcol = spectrum_hdu.data["GROUPING"]
            grouping = np.asarray(gcol if is_type1 else gcol[row], dtype=int)
        elif "GROUPING" in spectrum_hdu.header:
            grouping = np.asarray([int(spectrum_hdu.header.get("GROUPING"))] * len(channel), dtype=int)

        backfile = spectrum_hdu.header.get("BACKFILE")
        respfile = spectrum_hdu.header.get("RESPFILE")
        ancrfile = spectrum_hdu.header.get("ANCRFILE")

        return OGIPPHASpectrum(
            channel=channel,
            counts=counts,
            exposure=exposure,
            backscale=float(backscale),
            areascal=float(areascal),
            quality=quality,
            grouping=grouping,
            backfile=backfile,
            respfile=respfile,
            ancrfile=ancrfile,
            header=spectrum_hdu.header.copy(),
        )


def read_lc(path: str) -> OGIPLightCurve:
    logger.info("Reading lightcurve file: %s", path)
    with fits.open(path) as hdul:
        rate_hdu = hdul["RATE"] if "RATE" in hdul else hdul[1]
        time = np.asarray(rate_hdu.data["TIME"], dtype=float)
        rate = np.asarray(rate_hdu.data["RATE"], dtype=float)
        error = np.asarray(rate_hdu.data["ERROR"], dtype=float)
        fracexp = None
        if "FRACEXP" in rate_hdu.columns.names:
            fracexp = np.asarray(rate_hdu.data["FRACEXP"], dtype=float)
        timedel = rate_hdu.header.get("TIMEDEL")
        return OGIPLightCurve(
            time=time,
            rate=rate,
            error=error,
            fracexp=fracexp,
            timedel=float(timedel) if timedel is not None else None,
            header=rate_hdu.header.copy(),
        )


def read_rmf(path: str):
    from redback.spectral.response import ResponseMatrix

    with fits.open(path) as hdul:
        matrix_hdu = hdul["MATRIX"]
        ebounds_hdu = hdul["EBOUNDS"]

        e_min = np.asarray(matrix_hdu.data["ENERG_LO"], dtype=float)
        e_max = np.asarray(matrix_hdu.data["ENERG_HI"], dtype=float)

        channel = np.asarray(ebounds_hdu.data["CHANNEL"], dtype=int)
        emin_chan = np.asarray(ebounds_hdu.data["E_MIN"], dtype=float)
        emax_chan = np.asarray(ebounds_hdu.data["E_MAX"], dtype=float)

        detchans = int(matrix_hdu.header.get("DETCHANS", len(channel)))
        if len(channel) != detchans:
            detchans = len(channel)

        channel_min = int(channel.min()) if len(channel) > 0 else 0
        matrix = _build_rmf_matrix(matrix_hdu, detchans, channel_min)

        return ResponseMatrix(
            e_min=e_min,
            e_max=e_max,
            channel=channel,
            emin_chan=emin_chan,
            emax_chan=emax_chan,
            matrix=matrix,
        )


def _build_rmf_matrix(matrix_hdu: fits.BinTableHDU, detchans: int, channel_min: int) -> np.ndarray:
    n_energy = len(matrix_hdu.data)
    matrix = np.zeros((detchans, n_energy), dtype=float)

    n_grp = matrix_hdu.data["N_GRP"]
    f_chan = matrix_hdu.data["F_CHAN"]
    n_chan = matrix_hdu.data["N_CHAN"]
    mat = matrix_hdu.data["MATRIX"]

    for i in range(n_energy):
        if np.ndim(n_grp[i]) == 0:
            groups = int(n_grp[i])
            fch = np.atleast_1d(f_chan[i])
            nch = np.atleast_1d(n_chan[i])
            vals = np.atleast_1d(mat[i])
        else:
            groups = len(n_grp[i])
            fch = np.asarray(f_chan[i], dtype=int)
            nch = np.asarray(n_chan[i], dtype=int)
            vals = np.asarray(mat[i], dtype=float)

        offset = 0
        for g in range(groups):
            start = int(fch[g]) - channel_min
            length = int(nch[g])
            end = start + length
            matrix[start:end, i] = vals[offset:offset + length]
            offset += length

    return matrix


def read_arf(path: str):
    from redback.spectral.response import EffectiveArea

    with fits.open(path) as hdul:
        specresp_hdu = hdul["SPECRESP"]
        e_min = np.asarray(specresp_hdu.data["ENERG_LO"], dtype=float)
        e_max = np.asarray(specresp_hdu.data["ENERG_HI"], dtype=float)
        area = np.asarray(specresp_hdu.data["SPECRESP"], dtype=float)
        return EffectiveArea(e_min=e_min, e_max=e_max, area=area)

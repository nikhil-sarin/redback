from typing import Union

import numpy as np
from sncosmo import TimeSeriesSource

from redback.constants import *
from redback.utils import nu_to_lambda, bandpass_magnitude_to_flux

def _bandflux_single_redback(model, band, time_or_phase):
    """

    Synthetic photometry of a model through a single bandpass

    :param model: Source object
    :param band: Bandpass
    :param time_or_phase: Time or phase numpy array
    :return: bandflux through the bandpass
    """
    from sncosmo.utils import integration_grid
    HC_ERG_AA = 1.9864458571489284e-08 # planck * speed_of_light in AA/s
    MODEL_BANDFLUX_SPACING = 5.0  # Angstroms

    if (band.minwave() < model.minwave() or band.maxwave() > model.maxwave()):
        raise ValueError('bandpass {0!r:s} [{1:.6g}, .., {2:.6g}] '
                         'outside spectral range [{3:.6g}, .., {4:.6g}]'
                         .format(band.name, band.minwave(), band.maxwave(),
                                 model.minwave(), model.maxwave()))

        # Set up wavelength grid. Spacing (dwave) evenly divides the bandpass,
        # closest to 5 angstroms without going over.
    wave, dwave = integration_grid(band.minwave(), band.maxwave(),
                                   MODEL_BANDFLUX_SPACING)
    trans = band(wave)
    f = model._flux(time_or_phase, wave)
    f = np.abs(f)
    return np.sum(wave * trans * f, axis=1) * dwave / HC_ERG_AA


def _bandflux_redback(model, band, time_or_phase, zp, zpsys):
    """
    Support function for bandflux in Source and Model. Follows SNCOSMO

    This is necessary to have outside because ``phase`` is used in Source
    and ``time`` is used in Model, and we want the method signatures to
    have the right variable name.
    """
    from sncosmo.magsystems import get_magsystem
    from sncosmo.bandpasses import get_bandpass

    if zp is not None and zpsys is None:
        raise ValueError('zpsys must be given if zp is not None')

    # broadcast arrays
    if zp is None:
        time_or_phase, band = np.broadcast_arrays(time_or_phase, band)
    else:
        time_or_phase, band, zp, zpsys = \
            np.broadcast_arrays(time_or_phase, band, zp, zpsys)

    # Convert all to 1-d arrays.
    ndim = time_or_phase.ndim  # Save input ndim for return val.
    time_or_phase = np.atleast_1d(time_or_phase)
    band = np.atleast_1d(band)
    if zp is not None:
        zp = np.atleast_1d(zp)
        zpsys = np.atleast_1d(zpsys)

    # initialize output arrays
    bandflux = np.zeros(time_or_phase.shape, dtype=float)

    # Loop over unique bands.
    for b in set(band):
        mask = band == b
        b = get_bandpass(b)

        fsum = _bandflux_single_redback(model, b, time_or_phase[mask])

        if zp is not None:
            zpnorm = 10. ** (0.4 * zp[mask])
            bandzpsys = zpsys[mask]
            for ms in set(bandzpsys):
                mask2 = bandzpsys == ms
                ms = get_magsystem(ms)
                zpnorm[mask2] = zpnorm[mask2] / ms.zpbandflux(b)
            fsum *= zpnorm

        bandflux[mask] = fsum

    if ndim == 0:
        return bandflux[0]
    return bandflux

def _bandmag_redback(model, band, magsys, time_or_phase):
    """
    Support function for bandflux in Source and Model.
    This is necessary to have outside the models because ``phase`` is used in
    Source and ``time`` is used in Model.
    """
    from sncosmo.magsystems import get_magsystem

    bandflux = _bandflux_redback(model, band, time_or_phase, None, None)
    band, magsys, bandflux = np.broadcast_arrays(band, magsys, bandflux)
    return_scalar = (band.ndim == 0)
    band = band.ravel()
    magsys = magsys.ravel()
    bandflux = bandflux.ravel()

    result = np.empty(bandflux.shape, dtype=float)
    for i, (b, ms, f) in enumerate(zip(band, magsys, bandflux)):
        ms = get_magsystem(ms)
        zpf = ms.zpbandflux(b)
        result[i] = -2.5 * np.log10(f / zpf)

    if return_scalar:
        return result[0]
    return result

class RedbackTimeSeriesSource(TimeSeriesSource):
        def __init__(self, phase, wave, flux, **kwargs):
            """
            RedbackTimeSeriesSource is a subclass of sncosmo.TimeSeriesSource that adds the ability to return the
            flux density at a given time and wavelength, and changes
            the behaviour of the _flux method to better handle models with very low flux values.

            :param phase: phase/time array
            :param wave: wavelength array in Angstrom
            :param spectra: spectra in erg/cm^2/s/A evaluated at all times and frequencies; shape (len(times), len(frequency_array))
            :param kwargs: additional arguments
            """
            super(RedbackTimeSeriesSource, self).__init__(phase=phase, wave=wave, flux=flux, **kwargs)

        def get_flux_density(self, time, wavelength):
            """
            Get the flux density at a given time and wavelength.

            :param time: time in days
            :param wavelength: wavelength in Angstrom
            :return: flux density in erg/cm^2/s/A
            """
            return self._flux(time, wavelength)

        def bandflux(self, band, phase, zp=None, zpsys=None):
            """
            Flux through the given bandpass(es) at the given phase(s).

            Default return value is flux in photons / s / cm^2. If zp and zpsys
            are given, flux(es) are scaled to the requested zeropoints.

            Parameters
            ----------
            band : str or list_like
                Name(s) of bandpass(es) in registry.
            phase : float or list_like, optional
                Phase(s) in days. Default is `None`, which corresponds to the full
                native phase sampling of the model.
            zp : float or list_like, optional
                If given, zeropoint to scale flux to (must also supply ``zpsys``).
                If not given, flux is not scaled.
            zpsys : str or list_like, optional
                Name of a magnitude system in the registry, specifying the system
                that ``zp`` is in.

            Returns
            -------
            bandflux : float or `~numpy.ndarray`
                Flux in photons / s /cm^2, unless `zp` and `zpsys` are
                given, in which case flux is scaled so that it corresponds
                to the requested zeropoint. Return value is `float` if all
                input parameters are scalars, `~numpy.ndarray` otherwise.
            """
            return _bandflux_redback(self, band, phase, zp, zpsys)

        def bandmag(self, band, magsys, phase):
            """Magnitude at the given phase(s) through the given
            bandpass(es), and for the given magnitude system(s).

            Parameters
            ----------
            band : str or list_like
                Name(s) of bandpass in registry.
            magsys : str or list_like
                Name(s) of `~sncosmo.MagSystem` in registry.
            phase : float or list_like
                Phase(s) in days.

            Returns
            -------
            mag : float or `~numpy.ndarray`
                Magnitude for each item in band, magsys, phase.
                The return value is a float if all parameters are not iterables.
                The return value is an `~numpy.ndarray` if any are iterable.
            """
            return _bandmag_redback(self, band, magsys, phase)



def blackbody_to_flux_density(temperature, r_photosphere, dl, frequency):
    """
    A general blackbody_to_flux_density formula

    :param temperature: effective temperature in kelvin
    :param r_photosphere: photosphere radius in cm
    :param dl: luminosity_distance in cm
    :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                      In source frame
    :return: flux_density in erg/s/Hz/cm^2
    """
    # adding units back in to ensure dimensions are correct
    frequency = frequency * uu.Hz
    radius = r_photosphere * uu.cm
    dl = dl * uu.cm
    temperature = temperature * uu.K
    planck = cc.h.cgs
    speed_of_light = cc.c.cgs
    boltzmann_constant = cc.k_B.cgs
    num = 2 * np.pi * planck * frequency ** 3 * radius ** 2
    denom = dl ** 2 * speed_of_light ** 2
    frac = 1. / (np.expm1((planck * frequency) / (boltzmann_constant * temperature)))
    flux_density = num / denom * frac
    return flux_density


class _SED(object):

    # sed units are erg/s/Angstrom - need to turn them into flux density compatible units
    UNITS = uu.erg / uu.s / uu.Hz / uu.cm**2

    def __init__(self, frequency: Union[np.ndarray, float], luminosity_distance: float, length=1) -> None:
        self.sed = None
        if isinstance(frequency, (float, int)):
            self.frequency = np.array([frequency] * length)
        else:
            self.frequency = frequency
        self.luminosity_distance = luminosity_distance

    @property
    def flux_density(self):
        flux_density = self.sed.copy()
        # add distance units
        flux_density /= (4 * np.pi * self.luminosity_distance ** 2)
        # get rid of Angstrom and normalise to frequency
        flux_density *= nu_to_lambda(self.frequency)
        flux_density /= self.frequency

        # add units
        flux_density = flux_density << self.UNITS

        # convert to mJy
        return flux_density.to(uu.mJy)

def boosted_bolometric_luminosity(temperature, radius, lambda_cut):
    """
    Compute the boosted bolometric luminosity for a blackbody with temperature T (K)
    and radius R (cm) to account for missing blue flux.

    It uses:
        L_bb = 4π R² σ_SB T⁴,
    F_tot = σ_SB T⁴, and integrates the flux density redward of lambda_cut.

    Parameters
    ----------
    temperature : float
        Temperature (K)
    radius : float
        Photospheric radius (cm)
    lambda_cut : float
        Cutoff wavelength in centimeters (i.e. converted from angstroms by multiplying by 1e-8)

    Returns
    -------
    L_boosted : float
        The corrected bolometric luminosity (erg/s)
    """
    from scipy.integrate import quad
    sigma_SB = sigma_sb  # Stefan–Boltzmann constant in cgs

    # Compute pure-blackbody bolometric luminosity.
    L_bb = 4.0 * np.pi * radius ** 2 * sigma_SB * temperature ** 4
    # Total flux per unit area (erg/s/cm^2)
    F_tot = sigma_SB * temperature ** 4

    # Define the Planck function B_lambda (in erg/s/cm^2/cm/sr)
    def planck_lambda(lam, T):
        h = planck  # erg s
        c = speed_of_light  # cm/s
        k = boltzmann_constant  # erg/K
        return (2.0 * h * c ** 2) / (lam ** 5) / (np.exp(h * c / (lam * k * T)) - 1.0)

    # Integrand: π * B_lambda gives flux per unit wavelength (erg/s/cm²/cm)
    integrand = lambda lam, T: np.pi * planck_lambda(lam, T)
    # Integrate from lambda_cut to infinity to get the red flux.
    F_red, integration_error = quad(integrand, lambda_cut, np.inf, args=(temperature,))
    # Compute boost factor.
    Boost = F_tot / F_red
    # Corrected luminosity.
    return Boost * L_bb, L_bb


class CutoffBlackbody(_SED):

    X_CONST = planck * speed_of_light / boltzmann_constant
    FLUX_CONST = 4 * np.pi * 2 * np.pi * planck * speed_of_light ** 2 * angstrom_cgs

    reference = "https://ui.adsabs.harvard.edu/abs/2017ApJ...850...55N/abstract"

    def __init__(self, time: np.ndarray, temperature: np.ndarray, luminosity: np.ndarray, r_photosphere: np.ndarray,
                 frequency: Union[float, np.ndarray], luminosity_distance: float, cutoff_wavelength: float,
                 **kwargs: None) -> None:
        """
        Blackbody SED with a cutoff

        :param time: time in source frame
        :param temperature: temperature in kelvin
        :param luminosity: luminosity in cgs
        :param r_photosphere: photosphere radius in cm
        :param frequency: frequency in Hz - must be a single number or same length as time array
        :param luminosity_distance: dl in cm
        :param cutoff_wavelength: cutoff wavelength in Angstrom
        :param kwargs: None
        """
        super(CutoffBlackbody, self).__init__(
            frequency=frequency, luminosity_distance=luminosity_distance, length=len(time))
        self.time = time
        self.unique_times = np.unique(self.time)
        tsort = np.argsort(self.time)
        self.uniq_is = np.searchsorted(self.time, self.unique_times, sorter=tsort)

        self.luminosity = luminosity
        self.temperature = temperature
        self.r_photosphere = r_photosphere
        self.cutoff_wavelength = cutoff_wavelength * angstrom_cgs

        self.norms = None

        self.sed = np.zeros(len(self.time))
        self.calculate_flux_density()

    @property
    def wavelength(self):
        if len(self.frequency) == 1:
            self.frequency = np.ones(len(self.time)) * self.frequency
        wavelength = nu_to_lambda(self.frequency) * angstrom_cgs
        return wavelength

    @property
    def mask(self):
        return self.wavelength < self.cutoff_wavelength

    @property
    def nxcs(self):
        return self.X_CONST * np.array(range(1, 11))

    def _set_sed(self):
        self.sed[self.mask] = \
            self.FLUX_CONST * (self.r_photosphere[self.mask]**2 / self.cutoff_wavelength /
                               self.wavelength[self.mask] ** 4) \
            / np.expm1(self.X_CONST / self.wavelength[self.mask] / self.temperature[self.mask])
        self.sed[~self.mask] = \
            self.FLUX_CONST * (self.r_photosphere[~self.mask]**2 / self.wavelength[~self.mask]**5) \
            / np.expm1(self.X_CONST / self.wavelength[~self.mask] / self.temperature[~self.mask])
        # Apply renormalisation
        self.sed *= self.norms[np.searchsorted(self.unique_times, self.time)]

    def _set_norm(self):
        self.norms = self.luminosity[self.uniq_is] / \
                     (self.FLUX_CONST / angstrom_cgs * self.r_photosphere[self.uniq_is] ** 2 * self.temperature[
                         self.uniq_is])

        tp = self.temperature[self.uniq_is].reshape(len(self.unique_times), 1)
        tp2 = tp ** 2
        tp3 = tp ** 3

        c1 = np.exp(-self.nxcs / (self.cutoff_wavelength * tp))

        term_1 = \
             c1 * (self.nxcs ** 2 + 2 * (self.nxcs * self.cutoff_wavelength * tp + self.cutoff_wavelength ** 2 * tp2)) \
            / (self.nxcs ** 3 * self.cutoff_wavelength ** 3)
        term_2 = \
            (6 * tp3 - c1 *
             (self.nxcs ** 3 + 3 * self.nxcs ** 2 * self.cutoff_wavelength * tp + 6 *
              (self.nxcs * self.cutoff_wavelength ** 2 * tp2 + self.cutoff_wavelength ** 3 * tp3))
             / self.cutoff_wavelength ** 3) / self.nxcs ** 4
        f_blue_reds = np.sum(term_1 + term_2, 1)
        self.norms /= f_blue_reds

    def calculate_flux_density(self):
        self._set_norm()
        self._set_sed()
        return self.flux_density


class Blackbody(object):

    reference = "It is a blackbody - Do you really need a reference for this?"

    def __init__(self, temperature: np.ndarray, r_photosphere: np.ndarray, frequency: np.ndarray,
                 luminosity_distance: float, **kwargs: None) -> None:
        """
        Simple Blackbody SED

        :param temperature: effective temperature in kelvin
        :param r_photosphere: photosphere radius in cm
        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                          In source frame
        :param luminosity_distance: luminosity_distance in cm
        :param kwargs: None
        """
        self.temperature = temperature
        self.r_photosphere = r_photosphere
        self.frequency = frequency
        self.luminosity_distance = luminosity_distance

        self.flux_density = self.calculate_flux_density()

    def calculate_flux_density(self):
        self.flux_density = blackbody_to_flux_density(
            temperature=self.temperature, r_photosphere=self.r_photosphere,
            frequency=self.frequency, dl=self.luminosity_distance)
        return self.flux_density


class Synchrotron(_SED):

    reference = "https://ui.adsabs.harvard.edu/abs/2004rvaa.conf...13H/abstract"

    def __init__(self, frequency: Union[np.ndarray, float], luminosity_distance: float, pp: float, nu_max: float,
                 source_radius: float = 1e13, f0: float = 1e-26, **kwargs: None) -> None:
        """
        Synchrotron SED

        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                          In source frame.
        :param luminosity_distance: luminosity_distance in cm
        :param pp: synchrotron power law slope
        :param nu_max: max frequency
        :param source_radius: emitting source radius
        :param f0: frequency normalization
        :param kwargs: None
        """
        super(Synchrotron, self).__init__(frequency=frequency, luminosity_distance=luminosity_distance)
        self.pp = pp
        self.nu_max = nu_max
        self.source_radius = source_radius
        self.f0 = f0
        self.sed = None

        self.calculate_flux_density()

    @property
    def f_max(self):
        return self.f0 * self.source_radius**2 * self.nu_max ** 2.5  # for SSA

    @property
    def mask(self):
        return self.frequency < self.nu_max

    def _set_sed(self):
        self.sed = np.zeros(len(self.frequency))
        if self.frequency.ndim == 2:
            self.sed = np.zeros((len(self.frequency), 1))
        self.sed[self.mask] = \
            self.f0 * self.source_radius**2 * (self.frequency[self.mask]/self.nu_max) ** 2.5 \
            * angstrom_cgs / speed_of_light * self.frequency[self.mask] ** 2
        self.sed[~self.mask] = \
            self.f_max * (self.frequency[~self.mask]/self.nu_max)**(-(self.pp - 1.)/2.) \
            * angstrom_cgs / speed_of_light * self.frequency[~self.mask] ** 2

    def calculate_flux_density(self):
        self._set_sed()
        return self.flux_density


class Line(_SED):

    reference = "https://ui.adsabs.harvard.edu/abs/2018ApJS..236....6G/abstract"

    def __init__(self, time: np.ndarray, luminosity: np.ndarray, frequency: Union[np.ndarray, float],
                 sed: Union[_SED, Blackbody], luminosity_distance: float, line_wavelength: float = 7.5e3,
                 line_width: float = 500, line_time: float = 50, line_duration: float = 25,
                 line_amplitude: float = 0.3, **kwargs: None) -> None:
        """
        Modifies the input SED by accounting for absorption lines

        :param time: time in source frame
        :param luminosity: luminosity in cgs
        :param frequency: frequency to calculate in Hz - Must be same length as time array or a single number.
                          In source frame.
        :param sed: instantiated SED class object.
        :param luminosity_distance: luminosity_distance in cm
        :param line_wavelength: line wavelength in angstrom
        :param line_width: line width in angstrom
        :param line_time: line time
        :param line_duration: line duration
        :param line_amplitude: line amplitude
        :param kwargs: None
        """
        super(Line, self).__init__(frequency=frequency, luminosity_distance=luminosity_distance, length=len(time))
        self.time = time
        self.luminosity = luminosity
        self.SED = sed
        self.sed = None
        self.line_wavelength = line_wavelength
        self.line_width = line_width
        self.line_time = line_time
        self.line_duration = line_duration
        self.line_amplitude = line_amplitude

        self.calculate_flux_density()

    @property
    def wavelength(self):
        return nu_to_lambda(self.frequency)

    def calculate_flux_density(self):
        # Mostly from Mosfit/SEDs
        self._set_sed()
        return self.flux_density

    def _set_sed(self):
        amplitude = self.line_amplitude * np.exp(-0.5 * ((self.time - self.line_time) / self.line_duration) ** 2)
        self.sed = self.SED.sed * (1 - amplitude)

        amplitude *= self.luminosity / (self.line_width * (2 * np.pi) ** 0.5)

        amp_new = np.exp(-0.5 * ((self.wavelength - self.line_wavelength) / self.line_width) ** 2)
        self.sed += amplitude * amp_new

def get_correct_output_format_from_spectra(time, time_eval, spectra, lambda_array, **kwargs):
    """
    Use SNcosmo to get the bandpass flux or magnitude in AB from spectra at given times.

    :param time: times in observer frame in days to evaluate the model on
    :param time_eval: times in observer frame where spectra are evaluated. A densely sampled array for accuracy
    :param bands: band array - must be same length as time array or a single band
    :param spectra: spectra in erg/cm^2/s/A evaluated at all times and frequencies; shape (len(times), len(frequency_array))
    :param lambda_array: wavelenth array in Angstrom in observer frame
    :param kwargs: Additional parameters
    :param output_format: 'flux', 'magnitude', 'sncosmo_source', 'flux_density'
    :return: flux, magnitude or SNcosmo TimeSeries Source depending on output format kwarg
    """
    # clean up spectrum to remove nonsensical values before creating sncosmo source
    spectra = np.nan_to_num(spectra)
    spectra[spectra.value == np.nan_to_num(np.inf)] = 1e-30 * np.mean(spectra[5])
    spectra[spectra.value == 0.] = 1e-30 * np.mean(spectra[5])

    source = RedbackTimeSeriesSource(phase=time_eval, wave=lambda_array, flux=spectra)
    if kwargs['output_format'] == 'flux':
        bands = kwargs['bands']
        magnitude = source.bandmag(phase=time, band=bands, magsys='ab')
        return bandpass_magnitude_to_flux(magnitude=magnitude, bands=bands)
    elif kwargs['output_format'] == 'magnitude':
        bands = kwargs['bands']
        magnitude = source.bandmag(phase=time, band=bands, magsys='ab')
        return magnitude
    elif kwargs['output_format'] == 'sncosmo_source':
        return source
    else:
        raise ValueError("Output format must be 'flux', 'magnitude', 'sncosmo_source', or 'flux_density'")
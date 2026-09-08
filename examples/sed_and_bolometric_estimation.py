"""Estimate SEDs and bolometric luminosities from multiband photometry.

This example uses deterministic synthetic flux-density data so that it runs
without downloads. The final epoch has too few filters on purpose, showing how
structured SED results retain failed epochs for inspection.
"""

import matplotlib.pyplot as plt
import numpy as np

import redback
from redback.sed_analysis import CutoffBlackbodySEDModel, ExtinctionConfig


def make_synthetic_transient():
    """Return dust-attenuated photometry with two complete and one failed epoch."""
    redshift = 0.05
    distance = 7.0e26
    rest_wavelength = np.array([2600.0, 3300.0, 4100.0, 5000.0, 6200.0, 7600.0])
    observer_frequency = redback.utils.lambda_to_nu(rest_wavelength * (1.0 + redshift))
    extinction = ExtinctionConfig(av_host=0.18, av_mw=0.04)
    source = CutoffBlackbodySEDModel(
        distance=distance,
        redshift=redshift,
        cutoff_wavelength=4200.0,
        absorption_index=2.0,
        extinction=extinction,
    )

    times = []
    frequencies = []
    flux_densities = []
    flux_density_errors = []
    epoch_parameters = [
        (10.0, 11500.0, 8.0e14, len(observer_frequency)),
        (12.0, 9500.0, 9.0e14, len(observer_frequency)),
        (14.0, 8000.0, 9.5e14, 2),
    ]
    fractional_offsets = np.array([-0.015, 0.01, 0.0, 0.012, -0.008, 0.006])
    for epoch, temperature, radius, n_filters in epoch_parameters:
        epoch_frequency = observer_frequency[:n_filters]
        rest_frequency, _ = redback.utils.calc_kcorrected_properties(
            frequency=epoch_frequency, redshift=redshift, time=0.0)
        flux_density = source.evaluate_photometry(
            rest_frequency,
            {"temperature": temperature, "radius": radius,
             "cutoff_wavelength": 4200.0, "absorption_index": 2.0},
            {"coordinate_type": "frequency", "data_mode": "flux_density"},
        )
        flux_density *= 1.0 + fractional_offsets[:n_filters]
        times.extend(epoch + np.linspace(0.0, 0.2, n_filters))
        frequencies.extend(epoch_frequency)
        flux_densities.extend(flux_density)
        flux_density_errors.extend(0.04 * flux_density)

    transient = redback.transient.OpticalTransient(
        name="SyntheticSED",
        time=np.asarray(times),
        flux_density=np.asarray(flux_densities),
        flux_density_err=np.asarray(flux_density_errors),
        frequency=np.asarray(frequencies),
        data_mode="flux_density",
        redshift=redshift,
        use_phase_model=False,
    )
    return transient, distance, extinction


def run_example(show=True):
    transient, distance, extinction = make_synthetic_transient()
    common = dict(
        distance=distance,
        bin_width=1.0,
        min_filters=4,
        extinction=extinction,
    )

    blackbody = transient.estimate_sed(method="blackbody", **common)
    cutoff_blackbody = transient.estimate_sed(
        method="cutoff_blackbody",
        cutoff_wavelength=4200.0,
        absorption_index=2.0,
        **common,
    )
    direct = transient.estimate_sed(
        method="direct_integration",
        uv_tail="cutoff_blackbody",
        ir_tail="blackbody",
        cutoff_wavelength=4200.0,
        absorption_index=2.0,
        **common,
    )

    diagnostics = cutoff_blackbody.to_dataframe()
    successful_fits = cutoff_blackbody.to_dataframe(successful_only=True)
    failed_fits = diagnostics[~diagnostics["success"]]
    components = direct.integrate().to_dataframe(successful_only=True)[[
        "epoch_times", "lum_ultraviolet", "lum_observed", "lum_infrared", "lum_bol"
    ]]

    print("Cutoff-blackbody diagnostics (failures are retained by default):")
    print(diagnostics[[
        "epoch_times", "success", "status", "n_filters", "reduced_chi_square"
    ]].to_string(index=False))
    print("\nSuccessful cutoff-blackbody fits:")
    print(successful_fits[["epoch_times", "temperature", "radius"]].to_string(index=False))
    print("\nFailed epochs:")
    print(failed_fits[["epoch_times", "status", "message"]].to_string(index=False))
    print("\nDirect-integration luminosity components [10^50 erg/s]:")
    print(components.to_string(index=False))

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    cutoff_blackbody.plot_epoch(0, ax=axes[0])
    blackbody.plot_evolution(axes=axes[1:])
    figure.tight_layout()

    _, luminosity_axis = plt.subplots()
    for result, label in [
        (blackbody, "blackbody"),
        (cutoff_blackbody, "cutoff blackbody"),
        (direct, "direct integration"),
    ]:
        result.integrate().plot_evolution(ax=luminosity_axis, label=label)
    luminosity_axis.legend()

    if show:
        plt.show()
    return {
        "blackbody": blackbody,
        "cutoff_blackbody": cutoff_blackbody,
        "direct_integration": direct,
    }


if __name__ == "__main__":
    run_example()

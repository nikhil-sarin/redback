import matplotlib.pyplot as plt
import numpy as np
import redback
import bilby.core.prior
from bilby.core.prior import LogUniform, Uniform, Sine
from bnspopkne.redback_interface import redback_S22_BNS_popkNe as saeev
from multiprocessing import Pool, cpu_count

run_pool = Pool(cpu_count() - 1)

sampler = "dynesty"
# lots of different models implemented, including
# afterglow/magnetar varieties/n_dimensional_fireball/shapelets/band function/kilonova/SNe/TDE
model = saeev

kne = "at2017gfo"
# gets the magnitude data for AT2017gfo, the KN associated with GW170817
data = redback.get_data.get_kilonova_data_from_open_transient_catalog_data(
    transient=kne
)
# creates a GRBDir with GRB
kilonova = redback.kilonova.Kilonova.from_open_access_catalogue(
    name=kne,
    data_mode="flux_density",
    active_bands=np.array(["u", "g", "r", "i", "z", "y", "J"]),
)

# use default priors
priors = bilby.core.prior.PriorDict()
priors["m1"] = Uniform(1.0, 2.0, "m1", latex_label=r"$M_1$")
priors["m2"] = Uniform(1.0, 2.0, "m2", latex_label=r"$M_2$")
priors["theta_obs"] = Sine(name="theta_obs", latex_label=r"$\iota$")
priors["redshift"] = Uniform(0.009, 0.011, "redshift", latex_label=r"$z$")
priors["disk_eff"] = Uniform(0.1, 0.4, "disk_eff", latex_label=r"$\eta_\mathrm{disk}$")
priors["ra"] = np.deg2rad(197.4503542)
priors["dec"] = np.deg2rad(-23.3814842)


model_kwargs = dict(
    frequency=kilonova.filtered_frequencies, output_format="flux_density"
)

result = redback.fit_model(
    transient=kilonova,
    model=model,
    sampler=sampler,
    model_kwargs=model_kwargs,
    prior=priors,
    sample="rslice",
    nlive=1000,
    pool=run_pool,
    queue_size=cpu_count() - 1,
    use_pool=True,
    resume=True,
)
result.save_to_file(
    filename="S22kne_test",
    overwrite=False,
    outdir="/cfs/data/chse9649/output_data/kne_data/redback_fits/",
    extension="hdf5",
    gzip=False,
)

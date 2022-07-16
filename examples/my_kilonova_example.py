import matplotlib.pyplot as plt
import numpy as np
import redback
import bilby.core.prior
from bilby.core.prior import Uniform, Sine, Gaussian
from bnspopkne.redback_interface import redback_S22_BNS_popkNe_streamlined as saeev
from bnspopkne import equation_of_state as eos
from bnspopkne import mappings
from multiprocessing import cpu_count


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
    name=kne, data_mode="flux_density", active_bands=["u", "g", "r", "i", "z", "y"],
)
EOS = "sfho"
EOS_path = "/cfs/data/chse9649/input_data/kne_data/eos_data/"
# EOS_path = "/Users/cnsetzer/Documents/Project1_kNe/kne_modeling/eos_data/"
E1 = eos.get_EOS_table(EOS=EOS, EOS_path=EOS_path)
tov_mass = eos.get_max_EOS_mass(E1)

# use default priors
priors = bilby.core.prior.PriorDict()
priors["redshift"] = 0.01
priors["m1"] = Uniform(1.0, tov_mass, "m1", latex_label=r"$M_1$")
priors["m2"] = Uniform(1.0, tov_mass, "m2", latex_label=r"$M_2$")
priors["theta_obs"] = Sine(name="theta_obs", latex_label=r"$\iota$")
priors["disk_eff"] = Uniform(0.1, 0.4, "disk_eff", latex_label=r"$\eta_\mathrm{disk}$")
priors["peculiar_velocity"] = Gaussian(
    0.0, 300.0, name="peculiar_velocity", latex_label=r"$v_\mathrm{p}$"
)

kappa_grid_path = "/cfs/data/chse9649/input_data/kne_data/opacity_data/Setzer2022_popkNe_opacities.csv"
# kappa_grid_path = "Setzer2022_popkNe_opacities.csv"
EOS_mass_to_rad = eos.get_radius_from_EOS(E1)
opacity_data, gp = mappings.construct_opacity_gaussian_process(kappa_grid_path, None)

model_kwargs = dict(
    frequency=kilonova.filtered_frequencies,
    output_format="flux_density",
    sim_gw=False,
    tov_mass=tov_mass,
    EOS_name=EOS,
    EOS_mass_to_rad=EOS_mass_to_rad,
    opacity_data=opacity_data,
    grey_opacity_interp=gp,
    save=False,
    consistency_check=False,
    ra=np.deg2rad(197.4503542),
    dec=np.deg2rad(-23.3814842),
)

result = redback.fit_model(
    transient=kilonova,
    model=model,
    sampler=sampler,
    model_kwargs=model_kwargs,
    prior=priors,
    sample="rslice",
    nlive=1000,
    queue_size=cpu_count() - 1,
    resume=True,
)
result.save_to_file(
    filename="S22kne_test",
    overwrite=False,
    outdir="/cfs/data/chse9649/output_data/kne_data/redback_fits/",
    extension="hdf5",
    gzip=False,
)

import numpy as np
import redback

# In this example we show how you can update the physics in a ejecta relation
# and pass them into a redback model by creating a new ejecta relation class.

class custom_ejecta_relation(object):
    """
    A relation connecting the intrinsic gravitational-wave parameters
    (component masses: mass_1, mass_2; tidal deformability: lambda_1, lambda_2)
    to the extrinsic kilonova parameters (ejecta velocity: vej; ejecta mass: mej)
    """
    def __init__(self, mass_1, mass_2, lambda_1, lambda_2):
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reference = 'Totally made up relation'
        self.ejecta_mass = self.calculate_ejecta_mass
        self.ejecta_velocity = self.calculate_ejecta_velocity

    @property
    def calculate_ejecta_velocity(self):
        c1 = redback.ejecta_relations.calc_compactness_from_lambda(self.lambda_1)
        c2 = redback.ejecta_relations.calc_compactness_from_lambda(self.lambda_2)

        vej = 1000 * (self.mass_1 / self.mass_2) * (1 + 0.5) + 0.99 * (self.mass_2 / self.mass_1)
        return vej

    @property
    def calculate_ejecta_mass(self):
        c1 = redback.ejecta_relations.calc_compactness_from_lambda(self.lambda_1)
        c2 = redback.ejecta_relations.calc_compactness_from_lambda(self.lambda_2)

        log10_mej = self.mass_1 * (self.mass_2 / self.mass_1)**c1

        mej = 10 ** log10_mej
        return mej


model = 'one_component_ejecta_relation'
kne = 'at2017gfo'

# gets the magnitude data for AT2017gfo
data = redback.get_data.get_kilonova_data_from_open_transient_catalog_data(transient=kne)
kilonova = redback.kilonova.Kilonova.from_open_access_catalogue(name=kne, data_mode="flux_density", active_bands=np.array(["g", "i"]))

priors = redback.priors.get_priors(model=model)
sampler = 'nestle'

model_kwargs = {
    'ejecta_relation': custom_ejecta_relation, # uses custom physics in ejecta relation
    'frequency': kilonova.filtered_frequencies,
    'output_format': 'flux_density',
}

result = redback.fit_model(transient=kilonova, model=model, sampler=sampler, model_kwargs=model_kwargs,
                           prior=priors, sample='rslice', nlive=1000, resume=True)

result.plot_corner()

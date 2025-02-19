import george.modeling
from redback.transient_models import phenomenological_models as pm

def calculate_flux_with_labels(time, t0, tau_rise, tau_fall, labels, **kwargs):
    """
    Calculate flux for multiple sets of parameters using the Bazin function,
    where parameters are indexed by the provided labels.

    :param time: time array in arbitrary units
    :param t0: start time
    :param tau_rise: exponential rise time
    :param tau_fall: exponential fall time
    :param labels: list of strings to generate parameter names
    :param kwargs: keyword arguments for parameters in the form a_{label}, b_{label}, etc.
    :return: dictionary with labels as keys and flux arrays as values
    """
    a_values = []
    b_values = []

    for label in labels:
        a_key = f'a_{label}'
        b_key = f'b_{label}'
        if a_key in kwargs and b_key in kwargs:
            a_values.append(kwargs[a_key])
            b_values.append(kwargs[b_key])
        else:
            raise ValueError(f"Missing parameters for label '{label}'.")

    # Call bazin_sne once with all aa and bb values
    flux_matrix = pm.bazin_sne(time, a_values, b_values, t0, tau_rise, tau_fall)

    # Construct the result dictionary with labels
    flux_results = {f'{label}': flux_matrix[i] for i, label in enumerate(labels)}

    return flux_results

class BazinGPModel(george.modeling.Model):
    def __init__(self, band_labels):
        self.parameter_names = ['t0', 'tau_rise', 'tau_fall']
        self.band_labels = band_labels
        for label in band_labels:
            self.parameter_names.extend([f'a_{label}', f'b_{label}'])

        def get_flux(self, t):
            return calculate_flux_with_labels(t, **self.get_parameter_vector())
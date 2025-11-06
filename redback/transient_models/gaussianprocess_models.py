from redback.transient_models import phenomenological_models as pm

def calculate_flux_with_labels(time, t0, tau_rise, tau_fall, labels, **kwargs):
    """
    Calculate flux for multiple sets of parameters using the Bazin function.

    Parameters are indexed by the provided labels.

    Parameters
    ----------
    time : np.ndarray
        Time array in arbitrary units.
    t0 : float
        Start time.
    tau_rise : float
        Exponential rise time.
    tau_fall : float
        Exponential fall time.
    labels : list of str
        List of strings to generate parameter names.
    **kwargs : dict
        Additional keyword arguments:

        - a_{label} : float
            Amplitude parameter for each label.
        - b_{label} : float
            Second parameter for each label.

    Returns
    -------
    dict
        Dictionary with labels as keys and flux arrays as values.
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
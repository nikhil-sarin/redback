from . import afterglow, kilonova, prompt, supernova, tde, transient

TRANSIENT_DICT = dict(afterglow=afterglow.Afterglow, lgrb=afterglow.LGRB, sgrb=afterglow.SGRB,
                      kilonova=kilonova.Kilonova, prompt=prompt.PromptTimeSeries, supernova=supernova.Supernova,
                      tde=tde.TDE, transient=transient.Transient)


def from_file(filename, transient_type, name, data_mode='flux'):
    """
    Instantiate the object from a private datafile.
    You do not need all the attributes just some.
    If a user has their own data for a type of transient, they can input the
    :param data_mode: flux, flux_density, luminosity
    :return: transient object with corresponding data loaded into an object
    """
    return transient_object
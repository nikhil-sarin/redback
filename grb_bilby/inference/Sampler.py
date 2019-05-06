import numpy as np
import grb_bilby.processing.GRB as tools
import bilby
import inspect
import os
import sys
from grb_bilby.models import models as mm

class GRB_GaussianLikelihood(bilby.Likelihood):
    def __init__(self, x, y, sigma, function):
        """

        Parameters
        ----------
        x, y: array_like
            The data to analyse
        sigma: array_like
            The standard deviation of the noise
        function:
            The python function to fit to the data
        """
        self.x = x
        self.y = y
        self.sigma = sigma
        self.function = function
        parameters = inspect.getargspec(function).args
        parameters.pop(0)
        self.parameters = dict.fromkeys(parameters)

    def log_likelihood(self):
        res = self.y - self.function(self.x, **self.parameters)
        return -0.5 * (np.sum((res / self.sigma)**2
                       + np.log(2*np.pi*self.sigma**2)))

def fit_model(name, path, model, sampler = 'dynesty', nlive = 3000, prior = False, walks = 1000, truncate = True, **kwargs):
    """

    Parameters
    ----------
    :param name: Telephone number of SGRB, e.g., GRB 140903A
    :param path: Path to the GRB folder which contains GRB data files, if using inbuilt data files then 'GRBData'
    :param model: String to indicate which model to fit to data
    :param sampler: String to indicate which sampler to use, default is dynesty
    and nested samplers are encouraged to allow evidence calculation
    :param nlive: number of live points
    :param priors: if Prior is true user needs to pass a dictionary with priors defined the bilby way
    """

    data = tools.SGRB(name, path)
    data.load_and_truncate_data(truncate = truncate)

    if prior == False:
        priors = bilby.prior.PriorDict(filename = 'Priors/'+model+'.prior')

    if model == 'collapsing_magnetar':
        function = mm.collapsing_mag

    if model == 'collapsing_losses':
        function = mm.collapsing_losses

    if model == 'radiative_losses':
        function = mm.radiative_losses

    if model == 'radiative_losses_full':
        function = mm.radiative_losses_full

    outdir = path+'/GRB' + name +'/'+model
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    label = 'result'

    likelihood = GRB_GaussianLikelihood(x = data.time, y = data.Lum50, sigma = data.Lum50_err, function = function)

    result = bilby.run_sampler(likelihood, priors=priors, label=label, sampler=sampler, nlive=nlive,
                                   outdir=outdir, plot=True, use_ratio=False, walks = walks, resume = False, maxmcmc = walks, nthreads = 4, save_bounds=False,**kwargs)

    return result, data

if __name__=="__main__":
    result, data = fit_model(name = sys.argv[1], path = 'GRBData', model = sys.argv[2], sampler = 'pymultinest', nlive = 1000, prior = False, walks = 100)

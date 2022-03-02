Welcome to REDBACK's documentation!
===================================
.. automodule:: redback
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Installation
   Code motivation
   Basics of Bayesian inference and parameter estimation
   Likelihood
   Priors
      Constraints
   Getting Data
      Private data
      Swift
      Open access catalog
      BATSE
      Data modes
   Transients
      Flux to luminosity conversion
      Supernova
      Kilonova
      Broadband Afterglow
      magnetar powered exotica
      Tidal disruption event
      Prompt
      Generic transient
   Models
      Afterglow models
      Supernova models
      Kilonova models
      Magnetar boosted ejecta models
      Millisecond magnetar models
      Tidal disruption models
      Phase models
      Phenomenological and fireball models
      Modifying your models
   Fitting
      Active bands
      Samplers
   Result object
      Plotting lightcurves
      Plotting corner
      Further analysis
      Loading a result file
   Acknowledgements
   Contributing guidelines

#Indices and tables
#==================

#* :ref:`genindex`
#* :ref:`modindex`
#* :ref:`search`

API:
----

.. autosummary::
   :toctree: api
   :template: custom-module-template.rst
   :caption: API:
   :recursive:

    core
    gw
    hyper
    bilby_mcmc

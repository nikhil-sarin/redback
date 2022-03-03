Welcome to REDBACK's documentation!
===================================
.. automodule:: redback
    :members:

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   code_motivation
   pe_basics
   likelihood
   priors
   getting_data
   transients
   models
   dependency_injections
   fitting
   results
   acknowledgements
   contributing

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   broadband_afterglow_private_data_example
   fit_your_own_model_example
   kilonova_example
   magnetar_boosted_example
   magnetar_example
   prompt_example
   supernova_example
   tde_example
   SN2011kl_sample_in_t0_example

.. currentmodule:: redback

API:
----

.. autosummary::
   :toctree: api
   :template: custom-module-template.rst
   :caption: API:
   :recursive:

   get_data
   priors
   plotting
   eos
   constraints
   analysis
   ejecta_relations
   interaction_processes
   utils
   transient
   transient_models
   sampler
   result
   sed
   photosphere
   likelihoods

redback.result.RedbackResult
============================

.. currentmodule:: redback.result

.. autoclass:: RedbackResult
   :members:
   :show-inheritance:
   :inherited-members:

   
   .. automethod:: __init__
   .. automethod:: __call__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~RedbackResult.__init__
      ~RedbackResult.calculate_prior_values
      ~RedbackResult.from_hdf5
      ~RedbackResult.from_json
      ~RedbackResult.from_pickle
      ~RedbackResult.get_all_injection_credible_levels
      ~RedbackResult.get_injection_credible_level
      ~RedbackResult.get_latex_labels_from_parameter_keys
      ~RedbackResult.get_one_dimensional_median_and_error_bar
      ~RedbackResult.get_weights_by_new_prior
      ~RedbackResult.occam_factor
      ~RedbackResult.plot_corner
      ~RedbackResult.plot_data
      ~RedbackResult.plot_lightcurve
      ~RedbackResult.plot_marginals
      ~RedbackResult.plot_multiband
      ~RedbackResult.plot_single_density
      ~RedbackResult.plot_walkers
      ~RedbackResult.plot_with_data
      ~RedbackResult.posterior_probability
      ~RedbackResult.prior_volume
      ~RedbackResult.samples_to_posterior
      ~RedbackResult.save_posterior_samples
      ~RedbackResult.save_to_file
      ~RedbackResult.to_arviz
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~RedbackResult.bayesian_model_dimensionality
      ~RedbackResult.covariance_matrix
      ~RedbackResult.kde
      ~RedbackResult.log_10_bayes_factor
      ~RedbackResult.log_10_evidence
      ~RedbackResult.log_10_evidence_err
      ~RedbackResult.log_10_noise_evidence
      ~RedbackResult.meta_data
      ~RedbackResult.model
      ~RedbackResult.model_kwargs
      ~RedbackResult.name
      ~RedbackResult.nburn
      ~RedbackResult.nested_samples
      ~RedbackResult.num_likelihood_evaluations
      ~RedbackResult.path
      ~RedbackResult.posterior
      ~RedbackResult.posterior_volume
      ~RedbackResult.priors
      ~RedbackResult.samples
      ~RedbackResult.transient
      ~RedbackResult.transient_type
      ~RedbackResult.version
      ~RedbackResult.walkers
   
   
[![Documentation Status](https://readthedocs.org/projects/redback/badge/?version=latest)](https://redback.readthedocs.io/en/latest/?badge=latest)
![Python package](https://github.com/nikhil-sarin/redback/workflows/Python%20application/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/nikhil-sarin/redback/badge.svg?branch=master)](https://coveralls.io/github/nikhil-sarin/redback?branch=master)
![PyPI](https://img.shields.io/pypi/v/redback)
[![arXiv](https://img.shields.io/badge/arXiv-2308.12806-00ff00.svg)](https://arxiv.org/abs/2308.12806)
# Redback
Introducing REDBACK, A software package for end-to-end modelling, fitting, and interpretation of electromagnetic transients via Bayesian Inference

### Online documentation

- [Installation](https://redback.readthedocs.io/en/latest/)
- [Examples](https://github.com/nikhil-sarin/redback/tree/master/examples)
- [Documentation](https://redback.readthedocs.io/en/latest/)


### Motivation and why redback might be useful to you.
The launch of new telescopes/surveys is leading to an explosion of transient observations. 
Redback is a software package for modelling, end-to-end interpretation and parameter estimation of these transients.

- Download data for supernovae, tidal disruption events, gamma-ray burst afterglows, kilonovae, prompt emission from 
  different catalogs/telescopes; Swift, BATSE, Open access catalogs, FINK and LASAIR brokers. 
  Users can also provide their own data or use simulated data
- Redback processes the data into a homogeneous transient object. Making it easy to plot lightcurves and do any other processing e.g., estimating bolometric luminosities or blackbody properties. 
- The user can then fit one of the models implemented in redback. Or fit their own model. Models for several different types of electromagnetic transients are implemented and range from simple analytical models to numerical surrogates.
- All models are implemented as functions and can be used to simulate populations, without needing to provide data. This way redback can be used simply as a tool to simulate realistic populations, no need to actually fit anything.
- Simulate realistic transient lightcurves for Rubin LSST Survey using the latest cadence tables and for ZTF. Or make your own survey. 
Simulate single transients or populations or simulate a full survey including non-detections and realistic cadences and noise.
- Redback uses [Bilby](https://lscsoft.docs.ligo.org/bilby/index.html) under the hood for sampling. Can easily switch samplers/likelihoods etc. By default the choice is made depending on the data.
- Fitting returns a homogenous result object, with functionality to plot fitted lightcurves and the posterior/evidence. Or importance sample etc.

### Contributing
If you are interested in contributing please join the redback 
[slack](https://join.slack.com/t/redback-group/shared_invite/zt-2503mmkaq-EMEAgz7i3mY0pg1o~VUdqw)
and email [Nikhil Sarin](mailto:nikhil.sarin@su.se?subject=Contributing%20to%20redback).

To make changes to redback, we require users to use a merge request system. 

### User/Dev calls
We have regular calls for users and developers. 
These include tutorials on specific redback functionality as well as discussions of new features/feature requests, 
and Q/A. Please join the slack to get details of these calls.

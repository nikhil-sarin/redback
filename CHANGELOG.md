# All notable changes will be documented in this file

## [1.0.3] 2025-02-019
Version 1.0.3 release of redback

## What's Changed
* fixed the lanthanide angle, vej units, and time conversion issues in 'nicholl_bns', 'mosfit_rprocess', and 'mosfit_kilonova' models by @Xiaofei13 in https://github.com/nikhil-sarin/redback/pull/223
* Fix setup.py to install dependencies by @robertdstein in https://github.com/nikhil-sarin/redback/pull/225
* Add astroquery as optional dependency by @robertdstein in https://github.com/nikhil-sarin/redback/pull/227
* Rtd hotfix and misc by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/229
* Implementation of TDE model in mosfit following Guillochon+13 etc by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/230
* Rename model by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/231
* hotfix by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/232
* Fbots and etc by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/233
* fix output format error by @wfw23 in https://github.com/nikhil-sarin/redback/pull/214
* BUGFIX: sncosmo models predicting nan magnitudes by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/235
* Csm cocoon and stuff by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/236
* Lateral spreading revision by @GPLamb in https://github.com/nikhil-sarin/redback/pull/243
* ReadTheDocs-312 by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/244
* Change docstring/default prior  by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/237
* Deleted an extra ckm unit; changed the default start time in the rest… by @Xiaofei13 in https://github.com/nikhil-sarin/redback/pull/238
* Get set for v1.03 by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/241
* Switch to the new Fink API URL by @JulienPeloton in https://github.com/nikhil-sarin/redback/pull/246
* Sbo and arnett model by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/248
* Synchrotron + PWN by @conoromand in https://github.com/nikhil-sarin/redback/pull/247
* Jetsimpy and TDE Synchrotron by @conoromand in https://github.com/nikhil-sarin/redback/pull/249
* Change sbo arnett range + misc  by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/250
* Fitted by @conoromand in https://github.com/nikhil-sarin/redback/pull/251
* Tdemass by @conoromand in https://github.com/nikhil-sarin/redback/pull/252

## New Contributors
* @robertdstein made their first contribution in https://github.com/nikhil-sarin/redback/pull/225
* @JulienPeloton made their first contribution in https://github.com/nikhil-sarin/redback/pull/246

**Full Changelog**: https://github.com/nikhil-sarin/redback/compare/V1.02...v1.03

## [1.0.2] 2024-04-18
Version 1.0.2 release of redback

### Added
- Tophat_redback surrogate model
- New SNCosmo fitting and lightcurve generation example
- New ToO simulations example 
- New heating rate for kilonova following Rosswog and Korobkin 2022c
- New kilonova models using Rosswog and Korobkin 2022c heating rate prescription 
- Fractional error likelihood. 
- Phenomenological model for emission/absorption lines in spectrum.

### Changed
- Changed inbuilt constraints to be easier for end users to define in priors
- Logger now outputs info as standard
- Maximum likelihood estimation for all likelihoods
- Bugfix for SNCosmo source when evaluating magnitudes with spectrum that go to zero. 
- Update simulation interface for changing noise types.  
- Fix CSM prior and logic for forward and reverse shock masks. 
- LASAIR API

## [1.0.0] 2023-08-25
Version 1.0.1 release of redback

### Changed
- Plotting behaviour for band scaling with magnitudes.
- Added reference and citation to redback paper now on arXiv to documentation/tutorials/readme

## [1.0.0] 2023-08-24
Version 1.0.0 release of redback

### Added
- Simulation interface for generic transients or for Surveys such as ZTF and LSST
- New TDE model (cooling envelope)
- New magnetar driven model for SLSN/Ic-BL/Fbot
- ~15 new afterglow models 
- New kilonova afterglow models
- Joint kilonova + afterglow models
- Added ability to add filters to redback and SNcosmo
- Plotting spectrum tools
- Several new examples
- Several new plotting options e.g., band scaling
- Redback paper is submitted!

### Changed
- Plotting max likelihood and random sample colour to a 
different colour to the data now requires an additional keyword argument 

## [0.5.0] 2023-04-14
Version 0.5.0 release of redback

### Added
- A lot of unit tests
- All models with different output formats are tested for whether they can be evaluated, plotted for random draws from the prior

### Changed
- Interfaces for Fink and Lasair.
- Plotting options e.g., what bands to plot, labels 

## [0.4.0] 2023-02-07
Version 0.4.0 release of redback

### Added
- New models for type 1A 
- type 1c supernovae model 
- mosfit r-process model
- mosfit kilonova model
- Nicholl BNS model 
- Power law stratified kilonova 
- Redback surrogate models for Bulla BNS, NSBH and Kasen BNS simulations
- Interface with redback surrogates

### Changed
- Phase models explicitly output 0 for t < t0 
- Extinction models now work with new SNCosmo style interface
- Updated shock powered models 
- Updated default frequency array for spectra, and added option to change it as a keyword argument
- Cleaner interface to fit models in magnitude/flux space

## [0.3.1] 2022-12-22
Version 0.3.1 release of redback

### Changed
- Fitting compatibility with `SNCosmo` filter definitions.

## [0.3.0] 2022-12-05
Version 0.3.0 release of redback

### Added
- Fink broker data getter
- New model interface allows spectra, SNcosmosource, flux output for all optical models. Backward compatible with old flux density approximation output.
- Optical transients can now be fit in flux space with proper calculation of filters
- Overhaul of plotting flux for optical transients. 
- Plot spectra for optical transients in analysis 
- New models for TDE, kilonovae and shocked ejecta

### Changed
- Lasair data getter. This may be removed in a future release.

## [0.2.1] 2022-03-12
Version 0.2.1 release of redback

### Added
- Pypi installation
- Standardized logging 
- Several new models

## [0.2.0] 2022-03-11
Version 0.2.0 release of redback

### Added
- Basic functionality
- Several examples 
- Docs
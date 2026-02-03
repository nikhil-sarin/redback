# All notable changes will be documented in this file

## [1.14.0] 2026-02-03
Version 1.14.0 release of redback

## Added and changed
* Added VegasAfterglow integration: 6 unified afterglow models with comprehensive priors and documentation
* Added spectral line profile models and velocity fitting tools (P-Cygni profiles, Voigt profiles)
* Replaced Swift data retrieval with swifttools API (removed Selenium dependency)
* Enhanced logging throughout the package for improved debugging and monitoring
* Added frequency parameter support to LightCurveLynx data reader
* Added toy partial TDE support for cooling envelope models
* Improved mixing models and TDE model stability
* Fixed spectral plotting options for data visualization
* Refactored afterglow_models into organized submodule structure
* Comprehensive test coverage improvements across all new features
* Multiple bug fixes for array handling in shock and TDE models

**Full Changelog**: https://github.com/nikhil-sarin/redback/compare/v1.13.1...v1.14.0

## [1.13.1] 2026-01-27
Version 1.13.1 release of redback

## Changed
* Automatically link __version__ to setup.py for single source of truth

## What's Changed
* Automatically link __version__ to setup.py by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/336

**Full Changelog**: https://github.com/nikhil-sarin/redback/compare/v1.13.0...v1.13.1

## [1.13.0] 2026-01-27
Version 1.13.0 release of redback

## Added and changed
* Added high-energy spectra models (gamma-ray) with photon counting observations
* Added full population synthesis workflow with custom rates and realistic distributions
* Added learned surrogate models (Type II supernova surrogates)
* Added comprehensive simulation classes (SimulateTransientWithCadence, SimulateGammaRayTransient)
* Added radio/X-ray support to simulation framework
* Added template-based models (SN 1998bw)
* Added LightCurveLynx data reader
* Added extensive plotter customization options
* Added nickel mixing models with temperature-dependent opacities
* Added flexible SED models with spectral features (blackbody + lines)
* Added mixture and student-t likelihoods for robust fitting
* Fixed learned model units and k-correction issues
* Fixed SVD convergence issue in nicholl_bns model
* Fixed missing factors in general synchrotron models
* Fixed thermal_synchrotron syntax
* Fixed afterglow SED model time array behavior
* Numba-ified afterglow models for performance improvements
* Automated PyPI releases via GitHub Actions
* Updated Python 3.13 compatibility
* Comprehensive test coverage improvements (85%+)
* CI/CD improvements: speed optimizations, parallel testing, disk space management

## What's Changed
* Add cocoon cooling proper by @conoromand in https://github.com/nikhil-sarin/redback/pull/334
* Add plotter customization options by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/316
* Fix learned model units by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/331
* Fix thermal_synchrotron syntax by @conoromand in https://github.com/nikhil-sarin/redback/pull/330
* Fix missing factors in general synchrotron by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/329
* Add learned surrogates by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/312
* Update plotting defaults by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/327
* Fix SVD convergence in nicholl_bns by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/321
* Fix afterglow SED model by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/322
* Speed up CI workflow by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/320
* Update flux mode by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/313
* Fix CI disk space issues by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/314
* Update docstrings by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/306
* Remove duplicate likelihood classes by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/308
* Add template based models by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/301
* Numba-ify afterglow models by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/287
* Add reference result tests by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/292
* Automate PyPI releases by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/298
* Add k-corrections by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/299
* Update workflow to Python 3.13 by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/300
* Add LightCurveLynx reader by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/293
* Add nickel mixing model by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/291
* Add flexible SED by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/290
* Fix parallelisation by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/289
* Add conditional JIT by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/288

**Full Changelog**: https://github.com/nikhil-sarin/redback/compare/v1.12.1...v1.13.0

## [1.12.1] 2025-08-28
Version 1.12.1 release of redback

## Added and changed
* Added partial disruptions to cooling envelope TDE model
* Added few new parameterisations for TDE model
* Added some phenomenological models
* Changed heating rate kilonova model grid file to use npz format
* Changed dependency on numpy/scipy 
* Several new models
* Type II supernova surrogate model based on Sarin et al. 2025
* New non-detection likelihood + example 

## What's Changed
* added nickel + fallback model by @conoromand in https://github.com/nikhil-sarin/redback/pull/273
* Fixed variable name to prevent noise-term problems by @conoromand in https://github.com/nikhil-sarin/redback/pull/274
* Vectorized magnetar by @conoromand in https://github.com/nikhil-sarin/redback/pull/275
* Fix for effective widths by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/271
* Stellar interactions by @conoromand in https://github.com/nikhil-sarin/redback/pull/276
* Non detection likelihood + plotting options + extinction functionality overhaul by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/277
* Type ii surr by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/278
* Update afterglow_models.py by @GPLamb in https://github.com/nikhil-sarin/redback/pull/279
* Extra spectrum models by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/280
* Remove extra day_to_s division by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/281
* Update cooling envelope model + misc by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/283
* Fix basic install by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/285

**Full Changelog**: https://github.com/nikhil-sarin/redback/compare/v1.12.0...v1.12.1

## [1.12] 2025-04-13
Version 1.12 release of redback

## Added and changed
* Bugfix for shock cooling and arnett
* Added bolometric version of shock cooling and arnett
* Added basic salt2 model direct interface
* Added some unit tests

## What's Changed
* shock cooling and arnett bolometric + bugfix  by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/268

## [1.1] 2025-04-13
Version 1.1 release of redback

## Added and changed
* Added new models for TDE, CSM and shock cooling
* New thermal synchrotron model 
* New interface for spectra analysis
* New interface/functionality for GP-based interpolation. 
* New interface for user-defined cosmology 
* New interface for estimating blackbody temperature and radius, and bolometric luminosity estimation. 
* StudentT and MixtureModel Likelihoods 
* Several new examples

## What's Changed
* Spectral analysis features by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/256
* added Chavalier/Fransson synchrotron by @conoromand in https://github.com/nikhil-sarin/redback/pull/255
* Added vectorized Fitted and TDEmass by @conoromand in https://github.com/nikhil-sarin/redback/pull/259
* MQ24 thermal synchrotron by @conoromand in https://github.com/nikhil-sarin/redback/pull/257
* GP implementation by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/253
* Update some filters andbugfix by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/262
* New shock cooling models and some changes by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/264
* utils and docs for UserCosmology functionality by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/265
* A big overhaul by @nikhil-sarin in https://github.com/nikhil-sarin/redback/pull/266

**Full Changelog**: https://github.com/nikhil-sarin/redback/compare/v1.03...v1.1

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
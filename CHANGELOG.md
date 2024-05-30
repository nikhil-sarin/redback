# All notable changes will be documented in this file

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
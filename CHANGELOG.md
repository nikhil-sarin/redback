# All notable changes will be documented in this file

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
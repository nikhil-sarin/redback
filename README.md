# Redback
Introducing REDBACK, a robust bayesian inference pipeline for electromagnetic transients

`How to Install:`
for the time being its recommended that you install with a symlink
this can be done using:
 `pip install -e .`
 This may not work on a cluster so installing locally is advised:
 pip install . --user

 This software relies on bilby and associated samplers, and other packages for models/transient catalogues etc.

 # Basic use #this is out of date.
 `Process GRB data:`

To download all the XRT data from the Swift Website you can use the helper functions:
`grb_bilby.processing.process_data.process_long_grbs()`

Helper functions also exist for short GRBs and for processing specific GRBs in a list.

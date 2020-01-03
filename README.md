# gbags.bilby
The full blown GRB pipeline with Bilby

`How to Install:`
for the time being its recommended that you install with a symlink
this can be done using:
 `pip install -e .`
 This may not work on a cluster so installing locally is advised:
 pip install . --user
 
 This software relies on bilby and associated samplers. 
 
 This README is a work in progress.
 
 # Basic use
 `Process GRB data:`
 
To download all the XRT data from the Swift Website you can use the helper functions:
`grb_bilby.processing.process_data.process_long_grbs(use_default_directory=True)`

Here the flag `use_default_directory` can be switched to either save to the data folder or
in to a new directory. In the latter case, the code will make a new directory called `GRBData`
with each GRB having its own folder.

Helper functions also exist for short GRBs and for processing specific GRBs in a list.

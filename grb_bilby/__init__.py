"""
grb_bilby is a python package for end to end GRB analysis.
It serves as library to enable fast Bayesian analysis of GRBs through Bilby.
It also does data processing for swift data products, k-correction, automated data gathering etc.
"""
from . import inference, processing, analysis, models
from .inference.Sampler import fit_model
from .processing.GRB import SGRB


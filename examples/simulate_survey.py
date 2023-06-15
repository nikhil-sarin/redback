#This example will show how to simulate a survey for kilonovae in 1 month of ZTF.
import redback
from redback.simulate_transients import SimulateFullOpticalSurvey
import numpy as np

# We now set up the parameters for the kilonova population. A full survey simulation requires a prior and a rate.
rate = 10 #Gpc^-3 yr^-1
prior = redback.priors.get_priors(model='one_component_kilonova_model')
survey_start_date = 58288
survey_duration = 0.5 #years

kn_survey = SimulateFullOpticalSurvey(model='one_component_kilonova_model', survey='ztf',
                                      rate=rate, prior=prior,end_transient_time=10., snr_threshold=5.,
                                      survey_fov_sqdeg=36, survey_start_date=survey_start_date,
                                      survey_duration=survey_duration)
#This example will show how to simulate a survey for kilonovae in 1 month of ZTF.
import redback
from redback.simulate_transients import SimulateFullOpticalSurvey
import numpy as np

# We now set up the parameters for the kilonova population. A full survey simulation requires a prior and a rate.
rate = 10 #Gpc^-3 yr^-1
prior = redback.priors.get_priors(model='one_component_kilonova_model')
survey_start_date = 58288 #MJD
survey_duration = 1 #years

kn_survey = SimulateFullOpticalSurvey(model='one_component_kilonova_model', survey='ztf',
                                      rate=rate, prior=prior,end_transient_time=10., snr_threshold=5.,
                                      survey_fov_sqdeg=36, survey_start_date=survey_start_date,
                                      survey_duration=survey_duration, model_kwargs={})
# This will return the kn survey object as the other survey simulations.
#let's check the parameters of the simulated kilonovae
print(kn_survey.parameters)
# We can now extract the list of observations from the survey.
print(kn_survey.list_of_observations)
#Unsurprisingly, with a rate of 10 and a survey duration of 1 year, we get 0 observations but lots of non-detections etc

# We can save the survey to a file using the save method. Which will save all the observations.
# These can be looked at to understand things like detection efficiency or non-detections etc etc.
kn_survey.save_survey(survey='my_survey')
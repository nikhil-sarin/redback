from redback.transient import afterglow, kilonova, prompt, supernova, tde, transient

TRANSIENT_DICT = dict(afterglow=afterglow.Afterglow, lgrb=afterglow.LGRB, sgrb=afterglow.SGRB,
                      kilonova=kilonova.Kilonova, prompt=prompt.PromptTimeSeries, supernova=supernova.Supernova,
                      tde=tde.TDE, transient=transient.Transient)

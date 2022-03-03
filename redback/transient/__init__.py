from redback.transient import afterglow, kilonova, prompt, supernova, tde, transient
from redback.transient.afterglow import Afterglow, LGRB, SGRB
from redback.transient.kilonova import Kilonova
from redback.transient.prompt import PromptTimeSeries
from redback.transient.supernova import Supernova
from redback.transient.tde import TDE
from redback.transient.transient import Transient


TRANSIENT_DICT = dict(
    afterglow=Afterglow, lgrb=LGRB, sgrb=SGRB, kilonova=Kilonova, prompt=PromptTimeSeries, supernova=Supernova,
    tde=TDE, transient=Transient)

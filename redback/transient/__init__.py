from redback.transient import afterglow, kilonova, prompt, supernova, tde, transient
from afterglow import Afterglow, LGRB, SGRB
from kilonova import Kilonova
from prompt import PromptTimeSeries
from supernova import Supernova
from tde import TDE
from transient import Transient


TRANSIENT_DICT = dict(
    afterglow=Afterglow, lgrb=LGRB, sgrb=SGRB, kilonova=Kilonova, prompt=PromptTimeSeries, supernova=Supernova,
    tde=TDE, transient=Transient)

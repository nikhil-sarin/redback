import os
import pandas as pd

dirname = os.path.dirname(__file__)


def get_trigger_number(grb: str) -> str:
    grb_table = get_grb_table()
    trigger = grb_table.query('GRB == @grb')['Trigger Number']
    if len(trigger) == 0:
        raise TriggerNotFoundError(f"The trigger for {grb} does not exist in the table.")
    else:
        return trigger.values[0]


def get_grb_table() -> pd.DataFrame:
    short_table = os.path.join(dirname, '../tables/SGRB_table.txt')
    long_table = os.path.join(dirname, '../tables/LGRB_table.txt')
    sgrb = pd.read_csv(short_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    lgrb = pd.read_csv(long_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    frames = [lgrb, sgrb]
    return pd.concat(frames, ignore_index=True)


class TriggerNotFoundError(Exception):
    """ Exceptions raised when trigger is not found."""

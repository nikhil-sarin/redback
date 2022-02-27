import os
import pandas as pd


_dirname = os.path.dirname(__file__)


def get_trigger_number(grb: str) -> str:
    grb_table = get_grb_table()
    trigger = grb_table.query('GRB == @grb')['Trigger Number']
    if len(trigger) == 0:
        raise TriggerNotFoundError(f"The trigger for {grb} does not exist in the table.")
    else:
        return trigger.values[0]


def get_grb_table() -> pd.DataFrame:
    short_table = os.path.join(_dirname, '../tables/SGRB_table.txt')
    long_table = os.path.join(_dirname, '../tables/LGRB_table.txt')
    sgrb = pd.read_csv(short_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    lgrb = pd.read_csv(long_table, header=0,
                       error_bad_lines=False, delimiter='\t', dtype='str')
    frames = [lgrb, sgrb]
    return pd.concat(frames, ignore_index=True)


class TriggerNotFoundError(Exception):
    """ Exceptions raised when trigger is not found."""


def get_batse_trigger_from_grb(grb):
    ALPHABET = "ABCDEFGHIJKLMNOP"
    dat = ascii.read(f"{_dirname}/../tables/BATSE_trigger_table.txt")
    batse_triggers = list(dat['col1'])
    object_labels = list(dat['col2'])

    label_locations = dict()
    for i, label in enumerate(object_labels):
        if label in label_locations:
            label_locations[label].append(i)
        else:
            label_locations[label] = [i]

    for label, location in label_locations.items():
        if len(location) != 1:
            for i, loc in enumerate(location):
                object_labels[loc] = object_labels[loc] + ALPHABET[i]

    grb = "GRB" + grb.lstrip("GRB")

    index = object_labels.index(grb)
    return int(batse_triggers[index])

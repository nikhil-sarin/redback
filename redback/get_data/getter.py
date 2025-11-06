import pandas as pd


class DataGetter(object):
    """ """

    def __init__(self, transient: str, transient_type: str) -> None:
        self.transient = transient
        self.transient_type = transient_type

    def get_data(self) -> pd.DataFrame:
        """Downloads the raw data and produces a processed .csv file.

    Returns
    -------
    pandas.DataFrame
        The processed data
    """
        self.collect_data()
        return self.convert_raw_data_to_csv()

    @property
    def transient_type(self) -> str:
        """Checks if the transient type is valid when setting.

    Returns
    -------
    str
        The transient type.
    """
        return self._transient_type

    @transient_type.setter
    def transient_type(self, transient_type: str) -> None:
        """

    Parameters
    ----------
    transient_type : str
        The transient type.
    """
        if transient_type not in self.VALID_TRANSIENT_TYPES:
            raise ValueError("Transient type does not have Lasair data.")
        self._transient_type = transient_type


class GRBDataGetter(DataGetter):
    """ """

    def __init__(self, grb: str, transient_type: str) -> None:
        super().__init__(transient=grb, transient_type=transient_type)

    @property
    def grb(self) -> str:
        """

    Returns
    -------
    str
        The GRB number with prepended 'GRB'.
    """
        return self.transient

    @grb.setter
    def grb(self, grb: str) -> None:
        """

    Parameters
    ----------
    grb : str
        The GRB name with or without the prepended 'GRB'
    """
        self.transient = "GRB" + grb.lstrip("GRB")

    @property
    def stripped_grb(self) -> str:
        """

    Returns
    -------
    str
        The GRB number without prepended 'GRB'.
    """
        return self.grb.lstrip('GRB')

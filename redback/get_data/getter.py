import pandas as pd


class DataGetter(object):
    """ """

    def __init__(self, transient: str, transient_type: str) -> None:
        self.transient = transient
        self.transient_type = transient_type

    def get_data(self) -> pd.DataFrame:
        """
        Download raw data and produce a processed .csv file.

        Returns
        -------
        pandas.DataFrame
            The processed data with standardized columns

        Examples
        --------
        >>> from redback.get_data.swift import SwiftDataGetter
        >>> getter = SwiftDataGetter('GRB170817A', 'afterglow', 'flux')
        >>> data = getter.get_data()
        """
        self.collect_data()
        return self.convert_raw_data_to_csv()

    @property
    def transient_type(self) -> str:
        """
        Get the transient type.

        Checks if the transient type is valid when setting.

        Returns
        -------
        str
            The transient type
        """
        return self._transient_type

    @transient_type.setter
    def transient_type(self, transient_type: str) -> None:
        """
        Set the transient type.

        Parameters
        ----------
        transient_type : str
            The transient type. Must be in VALID_TRANSIENT_TYPES
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
        Get the GRB name with prepended 'GRB'.

        Returns
        -------
        str
            The GRB name with 'GRB' prefix
        """
        return self.transient

    @grb.setter
    def grb(self, grb: str) -> None:
        """
        Set the GRB name.

        Parameters
        ----------
        grb : str
            The GRB name with or without the prepended 'GRB'
        """
        self.transient = "GRB" + grb.lstrip("GRB")

    @property
    def stripped_grb(self) -> str:
        """
        Get the GRB number without prepended 'GRB'.

        Returns
        -------
        str
            The GRB number without 'GRB' prefix
        """
        return self.grb.lstrip('GRB')

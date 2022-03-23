import pandas as pd


class DataGetter(object):
    """ """

    def __init__(self, transient: str, transient_type: str) -> None:
        self.transient = transient
        self.transient_type = transient_type

    def get_data(self) -> pd.DataFrame:
        """Downloads the raw data and produces a processed .csv file.

        :return: The processed data
        :rtype: pandas.DataFrame
        """
        self.collect_data()
        return self.convert_raw_data_to_csv()

    @property
    def transient_type(self) -> str:
        """Checks if the transient type is valid when setting.

        :return: The transient type.
        :rtype: str
        """
        return self._transient_type

    @transient_type.setter
    def transient_type(self, transient_type: str) -> None:
        """
        :param transient_type: The transient type.
        :type transient_type: str
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
        :return: The GRB number with prepended 'GRB'.
        :rtype: str
        """
        return self.transient

    @grb.setter
    def grb(self, grb: str) -> None:
        """
        :param grb: The GRB name with or without the prepended 'GRB'
        :type grb: str
        """
        self.transient = "GRB" + grb.lstrip("GRB")

    @property
    def stripped_grb(self) -> str:
        """
        :return: The GRB number without prepended 'GRB'.
        :rtype: str
        """
        return self.grb.lstrip('GRB')

from .transient import Transient


class Kilonova(Transient):
    def __init__(self, name):

        self.flux_density = []
        self.flux_density_err = []
        self.magnitude = []
        self.magnitude_error = []
        self.name = name

        self._set_data()
        self._set_photon_index()
        self._set_t90()
        self._get_redshift()

    def _set_data(self):
        pass

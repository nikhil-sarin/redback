from redback.utils import calc_compactness_from_lambda

class Dietrich_ujevic_17(object):
    """
    Relations to connect intrinsic GW parameters to extrinsic kilonova parameters from Dietrich and Ujevic 2017
    """
    def __init__(self, mass_1, mass_2, lambda_1, lambda_2):
        self.mass_1 = mass_1
        self.mass_2 = mass_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.reference = 'https://ui.adsabs.harvard.edu/abs/2017CQGra..34j5014D/abstract'
        self.ejecta_mass = self.calculate_ejecta_mass()
        self.ejecta_velocity = self.calculate_ejecta_velocity()

    @property
    def calculate_ejecta_velocity(self):
        c1 = calc_compactness_from_lambda(self.lambda_1)
        c2 = calc_compactness_from_lambda(self.lambda_2)

        a = -0.3090
        b = 0.657
        c = -1.879
        vej = a * (self.mass_1 / self.mass_2) * (1 + c * c1) + a * (self.mass_2 / self.mass_1) * (1 + c * c2) + b
        return vej

    @property
    def calculate_ejecta_mass(self):
        c1 = calc_compactness_from_lambda(self.lambda_1)
        c2 = calc_compactness_from_lambda(self.lambda_2)

        a = -0.0719
        b = 0.2116
        d = -2.42
        n = -2.905

        log10_mej = a * (self.mass_1 * (1 - 2 * c1) / c1 + self.mass_2 * (1 - 2 * c2) / c2) + b * \
                    (self.mass_1 * (self.mass_2 / self.mass_1) ** n + self.mass_2 * (self.mass_1 / self.mass_2) ** n) + d

        mej = 10 ** log10_mej
        return mej

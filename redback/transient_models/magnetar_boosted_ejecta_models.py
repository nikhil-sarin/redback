from redback.constants import *
import numpy as np

def metzger_magnetar_boosted_kilonova_model(time, **kwargs):
    pass

def trapped_magnetar_lum(time, **kwargs):
    alpha = (1 + nn)/(1 - nn)
    omegat = omega0 * (1. + time/tau)**(alpha)
    lsd = eta * bp**2 * radius**6 * omegat**4
    doppler = 1/(gamma * (1 - beta*np.cos(theta)))
    lnu_x_bb = (8*np.pi**2*doppler**2 *radius**2)/(planck**3*speed_of_light**2)
    tau = kappa * (mej/vprime) * (radius/lorentz_factor)
    lum = e**(-tau) * lsd + (lnu_x_bb)
    return lum


def trapped_magnetar_flux(time, **kwargs):
    lum = trapped_magnetar_lum(time, **kwargs)
    kcorr = (1. + redshift)**(photon_index - 2)
    flux = lum/(4*np.pi*dl**2 * kcorr)

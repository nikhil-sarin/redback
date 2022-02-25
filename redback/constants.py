import astropy.constants as cc
import astropy.units as uu

ev_to_erg = 1.60218e-12
speed_of_light = cc.c.cgs.value
planck = cc.h.cgs.value
proton_mass = cc.m_p.cgs.value
solar_mass = cc.M_sun.cgs.value
sigma_sb = cc.sigma_sb.cgs.value
radiation_constant = sigma_sb*4 / speed_of_light
boltzmann_constant = cc.k_B.cgs.value
km_cgs = uu.km.cgs.scale
day_to_s = 86400
au_cgs = uu.au.cgs.scale
solar_radius = cc.R_sun.cgs.value
graviational_constant = cc.G.cgs.value
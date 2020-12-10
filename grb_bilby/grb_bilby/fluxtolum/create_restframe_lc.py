# script to create restframe BAT-XRT lightcurves
from sherpa.astro import ui as sherpa
#import sherpa.astro.ui as sherpa
import numpy as np
import sys

bol_elow =  1. # bolometric restframe low frequency in keV
bol_ehigh =  10000. # bolometric restframe high frequency in keV

grbs, z, alpha, beta, Ecut, obs_elow, obs_ehigh, cts2flux_abs, cts2flux_unabs = np.loadtxt("GRBs.dat", skiprows=1, dtype = '|S12', unpack = True, delimiter = ' ')
grbs = ['GRB050319','GRB060729','GRB061121','GRB080430']

def create_restframe_lc(grb,z,alpha,beta,Ecut,obs_elow,obs_ehigh,bol_elow,bol_ehigh,cts2flx_unabs,cts2flx_abs):
    obs_lc = np.genfromtxt('GRBData/'+grb+'/'+grb+'_flux.dat',comments='#')
    sherpa.dataspace1d(obs_elow, bol_ehigh, 0.01)
    sherpa.set_source(sherpa.bpl1d.band)
    band.gamma1=alpha
    band.gamma2=beta
    band.eb=Ecut
    sherpa.show_source()
    kcorr = sherpa.calc_kcorr(z, obs_elow, obs_ehigh, bol_elow, bol_ehigh, id=1)
    print(kcorr)

    z_dl = np.genfromtxt('z_dl.dat',skip_header=1)
    dl = (np.interp(z,z_dl[:,0],z_dl[:,1]))*3.08568e24  # interpolate to find dl and then convert dl into cm
    fbiso=(dl**2.)*4.*np.pi*kcorr
    Liso = (obs_lc[:,3]*(cts2flx_unabs/cts2flx_abs))*fbiso*1e-50  # calculate restframe luminosity in 10^50 erg s^-1 including conversion to unabs
    Liso_pos_err = obs_lc[:,4]*fbiso*1e-50
    Liso_neg_err = obs_lc[:,5]*fbiso*1e-50
    rest_time = obs_lc[:,0]/(1.+z)
    rest_time_pos_err = obs_lc[:,1]/(1.+z)
    rest_time_neg_err = obs_lc[:,2]/(1.+z)
    rest_lc = np.hstack((np.matrix(rest_time).T,np.matrix(rest_time_pos_err).T,np.matrix(rest_time_neg_err).T,np.matrix(Liso).T,np.matrix(Liso_pos_err).T,np.matrix(Liso_neg_err).T))
    return rest_lc

for x in range(len(grbs)):
    #  NB when using the burst analyser website the XRT data is unabsorbed.
    rest_lc = create_restframe_lc(grb = grbs[x],z = float(z[x]),
                                  alpha = float(alpha[x]),
                                  beta = float(beta[x]),
                                  Ecut=float(Ecut[x]),
                                  obs_elow=float(obs_elow[x]),
                                  obs_ehigh=float(obs_ehigh[x]),
                                  bol_elow=bol_elow,
                                  bol_ehigh=bol_ehigh,
                                  cts2flx_unabs=float(cts2flux_unabs[x]),
                                  cts2flx_abs=float(cts2flux_abs[x]))

    np.savetxt('GRBData/'+grbs[x]+'/'+grbs[x]+'.dat', rest_lc)
    print('processed',grbs[x])

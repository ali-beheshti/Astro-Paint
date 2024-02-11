"""
library containing [projected] halo profiles
"""

__author__ = ["Siavash Yasini"]
__email__ = ["yasini@usc.edu"]

import numpy as np

from astropy import units as u
from astropy.constants import sigma_T, m_p
#from astropy.cosmology import Planck18_arXiv_v2 as cosmo
from astropy.cosmology import Planck18 as cosmo

from astropaint.lib.utils import interpolate, LOS_integrate
from astropaint.lib import transform

# ---------------Caching----------------
# To cache templates use
#  the @memory.cache decorator
from joblib import Memory
cachedir = 'cache'
memory = Memory(cachedir, verbose=False)
# --------------------------------------

sigma_T = sigma_T.to(u.Mpc**2).value # [Mpc^2]
m_p = m_p.to(u.M_sun).value # [M_sun]
f_b = cosmo.Ob0/cosmo.Om0
c = 299792. #km/s
h = cosmo.h
T_cmb = 2.7251
Gcm2 = 4.785E-20 # G/c^2 (Mpc/M_sun)


# -----------
# 3D profiles
# -----------

def rho_3D(r, rho_s, r_s):
    """
    Calculate the 3D NFW density profile #TODO: add reference Eq.

    Parameters
    ----------
    r:
        distance from the center
    rho_s:
        density at radius r_s
    r_s:
        characterisic radius R_200c/c_200c

    Returns
    -------
    rho = 4 * rho_s * r_s ** 3 / r / (r + r_s) ** 2
    """

    rho = 4 * rho_s * r_s ** 3 / r / (r + r_s) ** 2

    return rho


# ---------------
# LOS projections
# ---------------

def rho_2D_bartlemann(R, rho_s, R_s):
    """
    projected NFW mass profile
    Eq. 7 in Bartlemann 1996: https://arxiv.org/abs/astro-ph/9602053

    Returns
    -------
    surface mass density: [M_sun/Mpc^2]
    """

    x = np.asarray(R / R_s, dtype=np.complex)
    f = 1 - 2 / np.sqrt(1 - x ** 2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))
    f = f.real
    f = np.true_divide(f, x ** 2 - 1)
    Sigma = 8 * rho_s * R_s * f
    return Sigma


@LOS_integrate
def rho_2D(R, rho_s, R_s):
    """
    3D NFW profile intgrated along the line of sight

    Returns
    -------
    surface mass density: [M_sun/Mpc^2]
    """

    return rho_3D(R, rho_s, R_s)


@interpolate(n_samples=20, sampling_method="logspace")
@LOS_integrate
def rho_2D_interp(R, rho_s, R_s):
    """
    3D NFW profile intgrated along a sampled number of line of sights
    and then interpolated

    Returns
    -------
    surface mass density: [M_sun/Mpc^2]
    """

    return rho_3D(R, rho_s, R_s)


def deflection_angle(R, c_200c, R_200c, M_200c, redshift, *, suppress=False, suppression_R=8):
    """
    calculate the deflection angle of a halo with NFW profile

    Parameters
    ----------
    R:
        distance from the center of halo (physical)[Mpc]
    c_200c:
        halo concentration parameter
    R_200c:
        halo 200c radius in (physical)[Mpc]
    M_200c:
        halo 200c mass of halo in [Msun]
    redshift:
        you guessed it, right! :D

    Returns
    -------
        the deflection angle at distance (physical) R from the center of halo
    """
    
    # constants for making units right
    msun = 1.989e30   # solar mass in kg
    mpc = 3.08567758e22   # Mpc in m
    c_kms = 299792.458  # Speed of light in km/s
    c_ms = c_kms*10**3   # Speed of light in m/s
    G = 6.67e-11*(msun/mpc)  # from [m^3.kg^-1.s^-2] to [m^2.Mpc.c^-2.Msun^-1]
    
    # objects info
    Z = np.array(redshift)
    a = 1/(1+Z) # scale factor
    Rs = R_200c/c_200c/a # (comoving) [Mpc]
    rhoS = M_200c /(4.*np.pi*(Rs**3)) /(np.log(1.+c_200c) - c_200c/(1.+c_200c)) # (comoving) [Msun/Mpc^3]
    c_tr = 1 * c_200c

 
    # preparing x from R
    x = (R/a) / Rs # dimensionless, rad
    x = x.astype(np.complex).value

    
    fArray = []
    # looping over all segments in x
    for i in np.arange(len(x)):
        # array to store f values at each segment
        # f truncation
        # if the segment is inside truncation calculate f inside
        if x[i] < c_tr:
            # f inside
            fIn = x[i]*((-np.log(np.sqrt(c_tr**2 - x[i]**2) + c_tr) + \
                    (np.log(-x[i]) -2j*(np.log(-x[i]).imag) \
                    - np.log(np.sqrt(1 - x[i]**2)*np.sqrt(c_tr**2 - x[i]**2) \
                    - c_tr - x[i]**2))/ np.sqrt(1 - x[i]**2) + np.log(x[i]))/x[i]**2 + \
                    ((1/np.sqrt(1 - x[i]**2) + 1)*np.log(c_tr + 1))/x[i]**2 - \
                    1/((c_tr + 1)*(np.sqrt((c_tr - x[i])*(c_tr + x[i])) + c_tr))) # dimensionless
            # store it in the ff array and go to the next segment
            fArray.append(fIn)
            #print("in",fIn)
        # if that segment is outside truncation calculate f outside
        else:
            # f outside
            fOut = (1/x[i])*( (-c_tr/(1+c_tr)) + np.log(1+c_tr) ) # dimensionless
            # store it in the ff array and go to the next segment
            fArray.append(fOut)
            #print("out",fOut)            
    
    # put the fArray values into an array f
    f = np.array(fArray)    
    
    # calculate the deflection angle
    beta = (16. *np.pi *G *rhoS *(Rs**2) /(a*(c_ms**2)) ) *(f.real) # dimensionless, rad 

    # suppress alpha at large radii
    if suppress:
        suppress_radius = suppression_R * R_200c
        beta *= np.exp(-(R.value / suppress_radius) ** 3)
        
        
    return beta


def tau_2D(R, rho_s, R_s):
    """
    projected NFW tau profile
    Eq. 7 in Battaglia 2016 :

    Returns
    -------
    tau: [NA]
    """
    X_H = 0.76
    x_e = (X_H + 1) / 2 * X_H
    f_s = 0.02
    mu = 4 / (2 * X_H + 1 + X_H * x_e)

    Sigma = rho_2D_bartlemann(R, rho_s, R_s)
    tau = sigma_T * x_e * X_H * (1 - f_s) * f_b * Sigma / mu / m_p
    return tau


def kSZ_T(R, rho_s, R_s, v_r, *, T_cmb=T_cmb):
    """kinetic Sunyaev Zeldovich effect
    #TODO: add reference"""
    tau = tau_2D(R, rho_s, R_s)
    dT = -tau * v_r / c * T_cmb

    return dT


def BG(R_vec, c_200c, R_200c, M_200c, redshift, theta, phi, v_r, v_th, v_ph, *, T_cmb=T_cmb):
    """
    Birkinshaw-Gull effect
    aka moving lens
    aka Rees-Sciama (moving gravitational potential)
    
    if we want output in uK then T_cmb should come as uK
    v_th, v_ph are supposed to be in km/s
    
    """
    c_kms = 299792.458  # Speed of light in km/s

    R = np.linalg.norm(R_vec, axis=-1)
    R_hat = np.true_divide(R_vec, np.expand_dims(R, axis=-1))
    beta = deflection_angle(R, c_200c, R_200c, M_200c, redshift)
    v_vec = transform.convert_velocity_sph2cart(theta, phi, 0 , v_th, v_ph)

    dT = -beta * np.dot(R_hat, v_vec) / c_kms * T_cmb  

    return dT
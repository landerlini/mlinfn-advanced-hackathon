import numpy as np
from scipy.interpolate import RectBivariateSpline

def interpolating_function(x,y, z):
    '''
    This function interpolate the known distribution in order to get all possible value in the domain.
    The function used is RectBivariateSpline by scipy.interpolate
    x   :   list of known x-coordinates
    y   :   list of known y-coordinates
    z   :   2D array, distribution
    '''
    cs=RectBivariateSpline(x,y, z)
    return cs
import numpy as np
from FTCS.FTCS_scheme import diffusion_2D
from PINN.interpolation import interpolating_function
import copy

def sigmoid(x, x_0 ,a,reverse=False):
    if reverse:
        return 1/(1+np.exp((x-x_0)*a))
    else:
        return 1/(1+np.exp(-(x-x_0)*a))    
    

def circle_sigmoid(geo: diffusion_2D, xc : float ,yc : float ,R : float, k : float):
    """
    This function returns a 2D circle distribution which decays with a sigmoid function. The radius is defined as the distance between the centre and the inflection point of the sigmoid profile:\n
    $sig(x)=\frac{1}{1+e^{-k\rho}}$\n
    where $\rho$ is the distance from the centre.

    geo :   geometry class defined with FTCS_scheme.diffusion_2D
    xc  :   x-coordinate of the centre
    yc  :   y-coordinate of the centre
    R   :   Radius of the circle
    k   :   decay constant of the sigmoid
    """
    couples=geo.get_domain()
    coo=np.array(np.meshgrid(np.arange(geo.Ny),np.arange(geo.Nx))).T.reshape(-1,2)
    ini=geo.get_ini_empty_array()
    i=0
    for x,y in couples:
        rho=np.sqrt((x-xc)**2+(y-yc)**2)
        ini[coo[i,0],coo[i,1]]=sigmoid(rho,R,k,True)
        i+=1
    return ini


def rectangle(geo: diffusion_2D, xc: float ,yc: float, H: float, W:float, T_evo: float):
    
    couples=geo.get_domain()
    coo=np.array(np.meshgrid(np.arange(geo.Ny),np.arange(geo.Nx))).T.reshape(-1,2)
    ini=geo.get_ini_empty_array()
    i=0
    for x,y in couples:
        if np.abs(x-xc)<H and np.abs(y-yc)<W:
            ini[coo[i,0],coo[i,1]]=1
        i+=1

    geo.set_initial_condition(ini)
    T_old=geo.size_t
    geo.size_t=T_evo
    finale=geo.Neumann_evo()
    geo.size_t=T_old
    Lx=geo.size_x
    Ly=geo.size_y
    dx=geo.dx
    dy=geo.dy

    x=np.arange(0,Lx,dx)
    y=np.arange(0,Ly,dy)
    cs=interpolating_function(x,y,finale)
    interp_f=cs(x,y)

    
    return interp_f

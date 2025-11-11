from FTCS.FTCS_scheme import diffusion_2D
from FTCS.Initial_condition import circle_sigmoid
import matplotlib.pylab as plt
from PINN.datagrid import geometry
import matplotlib.pylab as plt
from PINN import network
import os
from PINN.interpolation import interpolating_function
import numpy as np
from PINN.fit_model import fit_model
import tensorflow as tf


# Problem data
Lx=1    #dimension along x
Ly=1    #dimension along y
Tmax=0.2    #final time
D=0.005     #diffusion coefficient
dx=0.01     #space interval along x
dy=0.01     #space interval along y

#initial condition: uniform circle in the middle, sigmoid smooth
#centre
xc=0.5
yc=0.5
#sigmoid parmameter
R=0.25  #radius
k=50    #dacay constant


#initialising geometry for FTCS scheme
geo=diffusion_2D(Lx,Ly,Tmax,dx,dy,D)

#initial condition
ini=circle_sigmoid(geo,xc,yc,R,k)
geo.set_initial_condition(ini)

#FTCS evolution
finale=geo.Neumann_evo()
x=np.arange(0,Lx,dx)
y=np.arange(0,Ly,dy)
cs=interpolating_function(x,y,finale)

#set PINN geometry
geo=geometry(0,Lx,0,Ly,0,Tmax,True)


#Numero di epoche per ciascun addestramento
N_epochs=100000




#numero di seed
seeds=[8767,5358,271292,1268]
#ciclo for sui seed
for seed in range(seeds):
    tf.keras.backend.clear_session()
    #generazione seed random
    #Neural network
    NN=network.NN(3,1,in_seed=seed)
    NN.summary()

    #nome modello
    nome='seed_'+str(seed)
    fit_model(NN,geo,cs,N_epochs,D,nome,Tmax)





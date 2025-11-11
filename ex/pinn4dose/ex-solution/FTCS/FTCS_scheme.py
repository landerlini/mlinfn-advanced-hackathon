import numpy as np
import os


class diffusion_2D():
    def __init__(self, size_x : float, size_y : float, size_t : float, dx : float, dy : float, D : float,source : float = 0, dt_stable=True, dt=None):
        """
        This class sets the 2D geometry for the ingration of the forward diffusion equation

        size_x      :   size of the domain along the x-direction
        size_y      :   size of the domain along the y-direction
        size_t      :   size of the domain along the time-direction
        dx          :   space interval along x-direction
        dy          :   space interval along y-direction
        D           :   diffusion coefficient
        source      :   optional, if you want to add a source term in the equation
        dt_stable   :   True if you want to get the time interval from the stable condition
        dt          :   time interval
        
        
        """


        self.size_x=size_x
        self.size_y=size_y
        self.size_t=size_t
        self.dx=dx
        self.dy=dy
        self.D=D
        self.source=source
        if dt_stable:
            self.dt=self.time_stable()
        else:
            self.dt=dt
        
        self.Nx=int(round(self.size_x/dx))
        self.Ny=int(round(self.size_y/dy))
        self.Nstep=int(round(self.size_t/self.dt))

    def time_stable(self):
        """
        Calculate the time interval from the stable condition\n
        it gets the maximum time interval and return its fifth.
        """
        dt=(self.dx*self.dy)**2/(self.dx**2+self.dy**2)/10/self.D
        return dt
    
    def get_ini_empty_array(self):
        """
        this function returns a zeros array with correct size of the problem
        """
        return np.zeros((self.Ny,self.Nx))
    
    def get_domain(self):
        """
        This function returns all space couples of the domain (you can use it to set an analytical initial condition)
        """
        x=np.arange(0,self.size_x,self.dx)
        y=np.arange(0,self.size_y,self.dy)
        domain=np.array(np.meshgrid(x,y)).T.reshape(-1,2)
        return domain


    def set_initial_condition(self, initial : np.array):
        """
        This function set the initial condition of the problem

        initial :   2D matrix containing the inital condition. Warning: it has to be compatible with the size you of the geometry, if you are not sure use the get_ini_empy_array() to get an empy array with the correct dimension.
        """

        self.initial= initial

    def Neumann_BC(self):
        self.u[0,:]=self.u[1,:]
        self.u[-1,:]=self.u[-2,:]
        self.u[:,0]=self.u[:,1]
        self.u[:,-1]=self.u[:,-2]

    def integration_step(self):
        self.u[1:-1,1:-1]=self.u[1:-1,1:-1]+self.fact_x*(-2*self.u[1:-1,1:-1]+self.u[:-2,1:-1]+self.u[2:,1:-1])+self.fact_y*(-2*self.u[1:-1,1:-1]+self.u[1:-1,:-2]+self.u[1:-1,2:])+self.source*self.dt
        self.Neumann_BC()
    
    def save_distribution_at_time(self,t_step,folder):
        np.savez(folder+'/time_'+str(t_step)+'.npz', self.u)

    def Neumann_evo(self, save_distr : list = None, save_folder : str = None):
        """
        save_distr  :   Optional. list of inters containing time step to save
        save_folder :   if save_distr insert the path of the folder where you want to save in
        """
        self.u=self.initial.copy()
        self.Neumann_BC()
        self.fact_x=self.dt*self.D/self.dx**2
        self.fact_y=self.dt*self.D/self.dy**2
        if save_distr== None:
            t_save=None
        else:
            k=0
            t_save=save_distr[k]
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)

        
        self.Nstep=int(round(self.size_t/self.dt))
        for t in range(self.Nstep):
            self.integration_step()
            if t == t_save:
                self.save_distribution_at_time(t_save, save_folder)
                k+=1
                if k < len(save_distr):
                    t_save=save_distr[k]
        
        return self.u







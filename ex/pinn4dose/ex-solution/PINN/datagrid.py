import numpy as np

class geometry():
    def __init__(self, x0,xL,y0,yL,tmin,tmax,resample=False,seed=500):
        '''
        This class generates the whole geometry of the problem
        x0        : lower x
        xL        : higher x
        y0        : lowest y
        yL        : highest y
        tmin      : lower time
        tmax      : higher time
        resample  : resample data at each call?
        seed      : random seed for replication

        '''
        
        
        self.x0=x0
        self.xL=xL
        self.y0=y0
        self.yL=yL        
        self.t0=tmin
        self.tmax=tmax
        self.resample=resample
        self.seed=seed

    def get_random_points(self,n_point):
        self.resample_seed()    
        np.random.seed(self.seed)
        x=np.random.uniform(self.x0,self.xL,n_point)
        y=np.random.uniform(self.y0,self.yL,n_point)
        t=np.random.uniform(self.t0,self.tmax,n_point)
        return np.stack([x,y,t],-1)

    def get_random_boundary_points(self,n_point):
        self.resample_seed()
        np.random.seed(self.seed)
        x_left=np.random.uniform(self.x0,self.xL,n_point//4)
        t=np.random.uniform(self.t0,self.tmax,n_point//4)
        arr=np.stack([x_left,self.y0*np.ones_like(x_left),t],-1)

        x_right=np.random.uniform(self.x0,self.xL,n_point//4)
        t=np.random.uniform(self.t0,self.tmax,n_point//4)
        arr=np.append(arr,np.stack([x_right,self.yL*np.ones_like(x_right),t],-1),0)
        
        y_down=np.random.uniform(self.y0,self.yL,n_point//4)
        t=np.random.uniform(self.t0,self.tmax,n_point//4)
        arr=np.append(arr,np.stack([self.x0*np.ones_like(y_down),y_down,t],-1),0)
        
        y_up=np.random.uniform(self.y0,self.yL,n_point//4)
        t=np.random.uniform(self.t0,self.tmax,n_point//4)
        arr=np.append(arr,np.stack([self.xL*np.ones_like(y_up),y_up,t],-1),0)
        
        np.random.shuffle(arr)
        return arr
    
    def get_random_initial_points(self,n_point):
        self.resample_seed()
        np.random.seed(self.seed)
        x=np.random.uniform(self.x0,self.xL,n_point)
        y=np.random.uniform(self.y0,self.yL,n_point)
        t=np.zeros_like(x)
        return np.stack([x,y,t],-1)
    
    def get_random_points_atsometime(self,n_point, T):
        self.resample_seed()
        np.random.seed(self.seed)
        x=np.random.uniform(self.x0,self.xL,n_point)
        y=np.random.uniform(self.y0,self.yL,n_point)
        t=np.ones_like(x)*T
        return np.stack([x,y,t],-1)
    
    def resample_seed(self):
        if self.resample:
            self.seed=np.random.randint(1000000)

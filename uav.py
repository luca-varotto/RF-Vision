import numpy as np
from camera import Cam

'''
Define the UAV searching platform
'''

class UAV:

    ''' Constructor 
        c_init: initial position
        E_tot: total energy capacity
    '''
    def __init__(self,c_init, E_tot, scenario):
        self.scenario = scenario # 0: no energy constraint , 1: drone can not move, 2: full  
        self.c = c_init
        self.E_tot = E_tot 
        self.E_t = 0 # amount of energy used
        self.rx = None # Rx object associated to the UAV
        self.camera = None # embedded camera
    
    ''' automatically align camera optical center with the UAV position '''
    def __setattr__(self, key, value):
        try:
            self.__dict__[key] = value
            if key == "c" and self.camera:
                self.camera.center = self.c
        except:
            pass

    ''' update current level of energy used
        E_t: energy used at time t
    '''
    def update_energy_used(self, E_last):
        self.E_t += E_last if self.scenario == 3 else 0

    ''' draw UAV 
        ax: axis where to draw
    ''' 
    def draw_uav(self,ax):
        ax.scatter3D(self.c[0], self.c[1],self.c[2], c='k', marker='o', s=60)

    ''' draw UAV reference frame (the same as the world reference frame, but shifted) 
        ax : axis where to draw
    ''' 
    def draw_uav_frame(self,ax):
        ax.quiver(
            self.c[0], self.c[1],self.c[2], 
            1,0,0,
            length=2, color = 'red', alpha = .5, lw = 1,
            )
        ax.quiver(
            self.c[0], self.c[1],self.c[2], 
            0,1,0, 
            length=2, color = 'blue', alpha = .5, lw = 1,
        )
        ax.quiver(
            self.c[0], self.c[1],self.c[2], 
            0,0,1, 
            length=2,color = 'green', alpha = .5, lw = 1,
        )
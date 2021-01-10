import numpy as np

'''
Define a moving agent
'''

class Agent:

    ''' Constructor 
        p_init: initial position
    '''
    def __init__(self,p_init):
        self.p = p_init # 3D position
        self.tx = None # Tx object associated to the agent

    ''' Agent underlying motion model (usually unknown)
        mu_q: mean of the driving stochastic input
        sigma_q: variance of the driving stochastic input
    '''
    def motion(self, mu_q, sigma_q):
        q = mu_q*np.ones((2,1), dtype=float) + sigma_q * np.random.normal(size=(2,1)) # generate stochastic input
        q = np.vstack(( q, [0.0] ))
        self.p += q 

    ''' draw agent 
        ax: axis where to draw
    '''
    def draw_agent(self,ax):
        ax.scatter3D(self.p[0], self.p[1],self.p[2], c='r', marker='o')

    ''' draw agent reference frame (the same as the world reference frame, but shifted) 
        ax: axis where to draw
    '''
    def draw_agent_frame(self,ax):
        ax.quiver(
            self.p[0], self.p[1],self.p[2], 
            1,0,0,
            length=1, color = 'red', alpha = .8, lw = 1,
            )
        ax.quiver(
            self.p[0], self.p[1],self.p[2], 
            0,1,0, 
            length=1, color = 'blue', alpha = .8, lw = 1,
        )
        ax.quiver(
            self.p[0], self.p[1],self.p[2], 
            0,0,1, 
            length=1,color = 'green', alpha = .8, lw = 1,
        )
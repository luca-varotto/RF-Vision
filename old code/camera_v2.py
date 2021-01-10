import numpy as np
import matplotlib
import math
from matplotlib.patches import Polygon
import mpl_toolkits.mplot3d.art3d as art3d

class Cam:

    ''' Constructor '''
    def __init__(self, alpha, beta,f_range,f,c,resolution,T_a=1,T_clock=1/100):
        self.PT = [alpha,beta]  # pan and tilt angles
        self.f_range = f_range
        self.f = f # half-angle of view (zoom) [rad]
        self.center = c # optical center
        self.L = resolution # image sensor resolution (supposed to be squared) in [mm]
        self.T_a = T_a # image acquisition rate
        self.T_clock = T_clock # controller clock time

    @property
    def theta(self):
        return  math.atan2(self.L/(2*self.f))

    @property
    def R(self):
        return self.compute_R()

    @property
    def G(self):
        return np.hstack((self.R,self.center))

    @property
    def K(self):
        return np.array([[self.f,0,0],[0,self.f,0],[0,0,1]])

    @property
    def P(self):
        return np.dot(self.K,self.G)

    @property
    def c_fov(self):
        return self.compute_center_FoV()

    def compute_R(self):
        R_f = np.array([[-1,0,0],[0,1,0],[0,0,-1]]) # default conventional rotation wrt UAV reference frame
        R_alpha = np.array([[1,0,0],[0,np.cos(self.PT[0]), -np.sin(self.PT[0])],[0,np.sin(self.PT[0]), np.cos(self.PT[0])]]) # rotation associate to pan
        R_beta = np.array([[np.cos(self.PT[1]),0,np.sin(self.PT[1])],[0,1,0],[-np.sin(self.PT[1]),0,np.cos(self.PT[1])]]) # rotation associate to tilt
        return np.linalg.multi_dot([R_f,R_alpha,R_beta]) 

    def compute_FoV(self):
        Q = self.P[:,:-1]
        Q_inv = np.linalg.inv(Q)
        # compute the projection of each image plane vertex onto the 3D groundplane, 
        # using the inverse 3D projection rule
        vertices = np.array([[-self.L/2,-self.L/2], \
                            [-self.L/2,self.L/2],\
                            [self.L/2,self.L/2],\
                            [self.L/2,-self.L/2]])
                            
        for i in range(np.shape(vertices)[0]):
            m = np.vstack((vertices[i,:].reshape(-1,1),[1])) # homogeneous coordinates
            lamda = - self.center[-1] / (np.dot(Q_inv[-1,:],m)) 
            M = self.center + lamda*np.dot(Q_inv,m)
            if i ==0:
                FoV = M[:-1].T
            else:
                FoV = np.vstack((FoV, M[:-1].T))
        return FoV # 4x2 array

    def draw_FoV(self,ax):
        # compute the FoV
        FoV = self.compute_FoV()
        # draw the FoV
        p = Polygon(FoV, alpha=0.5)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0,  zdir="z")
        # draw optical axis
        # self.c_fov = self.compute_center_FoV()
        ax.plot3D([np.asscalar(self.center[0]), np.asscalar(self.c_fov[0])], \
                [np.asscalar(self.center[1]), np.asscalar(self.c_fov[1])], \
                [np.asscalar(self.center[2]), np.asscalar(self.c_fov[2])], \
                'b-.', LineWidth=1, alpha=0.5)
        # draw cone of view
        for i in range(np.shape(FoV)[0]):
            ax.plot3D([np.asscalar(self.center[0]), FoV[i,0]], \
                    [np.asscalar(self.center[1]), FoV[i,1]], \
                    [np.asscalar(self.center[2]), np.asscalar(self.c_fov[2])], \
                    'b--', LineWidth=1, alpha=0.5)           


    def compute_center_FoV(self):
        # from geometry theory
        # n = np.array([[0],[0],[1]]) # normal of the intersecting plane (z=0)
        # s = np.asscalar((-np.dot(self.center.T,n))/(np.dot(self.R[:,-1].T,n))) # scalar on the optical ray
        # l = np.reshape(self.R[:,-1], np.shape(self.center))
        # c_fov = self.center + s*l # intersection optical ray and z=0 (FoV center)
        # from camera inverse 3D projection rule (equivalent of above in case of optical axis)
        Q = self.P[:,:-1]
        Q_inv = np.linalg.inv(Q)
        m_c = np.vstack((np.zeros((2,1)),[1])) # homogeneous coordinates
        lamda = - self.center[-1] / (np.dot(Q_inv[-1,:],m_c))
        c_fov = self.center + lamda*np.dot(Q_inv,m_c)
        return c_fov

    def draw_center_FoV(self,ax):
        ax.scatter3D(self.c_fov[0], self.c_fov[1],self.c_fov[2], c='k', marker='o', s=10)

    def draw_cam_frame(self,ax,c):
        R_T = self.R.T # WHY????????
        ax.quiver(
            c[0], c[1],c[2], 
            R_T[0,0],R_T[1,0],R_T[2,0],
            length=1, color = 'red', alpha = .8, lw = 2,
            )
        ax.quiver(
            c[0], c[1],c[2], 
            R_T[0,1],R_T[1,1],R_T[2,1],
            length=1, color = 'blue', alpha = .8, lw = 2,
        )
        ax.quiver(
            c[0], c[1],c[2], 
            R_T[0,2],R_T[1,2],R_T[2,2],
            length=1,color = 'green', alpha = .8, lw = 2,
        )


    ''' Tracking cost function '''
    def cost_function(self,target):
        h = 0.5 * np.linalg.norm(target-self.c_fov)**2
        return h

    ''' Controller for camera center '''
    def compute_gradient(self, target):
        e_3 = np.array([[0],[0],[1]])
        mc = e_3
        
        R_f = np.array([[-1,0,0],[0,1,0],[0,0,-1]]) # default conventional rotation wrt UAV reference frame
        R_alpha = np.array([[1,0,0],[0,np.cos(self.PT[0]), -np.sin(self.PT[0])],[0,np.sin(self.PT[0]), np.cos(self.PT[0])]]) # rotation associate to pan
        R_beta = np.array([[np.cos(self.PT[1]),0,np.sin(self.PT[1])],[0,1,0],[-np.sin(self.PT[1]),0,np.cos(self.PT[1])]]) # rotation associate to tilt
        
        R_alpha_derivative = np.array([[0,0,0],[0,-np.sin(self.PT[0]), -np.cos(self.PT[0])],[0,np.cos(self.PT[0]), -np.sin(self.PT[0])]]) 

        R_beta_derivative = np.array([[-np.sin(self.PT[1]),0,np.cos(self.PT[1])],[0,0,0],[-np.cos(self.PT[1]),0,-np.sin(self.PT[1])]])
        
        c_fov_derivative_alpha =  + self.center[-1] /(np.asscalar( np.linalg.multi_dot([e_3.T,np.linalg.inv(self.K),R_f,R_alpha,R_beta,mc]) ))**2 \
                                    *np.linalg.multi_dot([R_alpha_derivative,R_beta.T,R_alpha.T,R_f.T,np.linalg.inv(self.K),mc]) +\
                                     self.center[-1] / np.asscalar( np.linalg.multi_dot([e_3.T,np.linalg.inv(self.K),R_f,R_alpha,R_beta,mc])) \
                                    *np.linalg.multi_dot([R_beta.T,R_alpha_derivative.T,R_f.T,np.linalg.inv(self.K),mc])
        
        c_fov_derivative_beta =  + self.center[-1] /(np.asscalar( np.linalg.multi_dot([e_3.T,np.linalg.inv(self.K),R_f,R_alpha,R_beta,mc]) ))**2 \
                                    *np.linalg.multi_dot([R_beta_derivative,R_beta.T,R_alpha.T,R_f.T,np.linalg.inv(self.K),mc]) +\
                                     self.center[-1] / np.asscalar( np.linalg.multi_dot([e_3.T,np.linalg.inv(self.K),R_f,R_alpha,R_beta,mc])) \
                                    *np.linalg.multi_dot([R_beta_derivative.T,R_alpha.T,R_f.T,np.linalg.inv(self.K),mc])

        gradient_alpha =  np.asscalar(np.dot(target.T,c_fov_derivative_alpha)) - np.asscalar(np.dot(self.c_fov.T,c_fov_derivative_alpha))
        gradient_beta =  np.asscalar(np.dot(target.T,c_fov_derivative_beta)) - np.asscalar(np.dot(self.c_fov.T,c_fov_derivative_beta))
        return gradient_alpha, gradient_beta

    def controller(self,target,K_alpha=1.0E-3, K_beta=1.0E-3, max_control_iterations=100):
        cost_function = []
        for iteration in range(max_control_iterations):
            # update camera pan
            gradient_alpha, gradient_beta = self.compute_gradient(target)

            # account for saturations
            self.PT = [self.PT[0]  - K_alpha*gradient_alpha, self.PT[1] - K_beta*gradient_beta]
            
            # consider saturations

            cost_function.append(self.cost_function(target))

        return cost_function
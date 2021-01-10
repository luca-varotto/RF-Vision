import numpy as np
import matplotlib
import math
from matplotlib.patches import Polygon as Polygon_plt
from shapely.geometry import Point, Polygon
import mpl_toolkits.mplot3d.art3d as art3d

''' 
Camera object
'''

class Cam:

    ''' Constructor '''
    def __init__(self, alpha, beta,f_range,f,c,resolution_px, px2mm,T_a=1,T_clock=1/100):
        self.PT = [alpha,beta]  #[rad]; pan and tilt angles
        self.f_range = f_range # focal length range
        self.f = f # [mm]; focal length (zoom)
        self.center = c # [m]; optical center
        self.resolution_px = resolution_px # [pixels]; image sensor resolution (supposed to be squared)
        self.px2mm = px2mm # [pixels] --> [mm] conversion
        self.T_a = T_a # image acquisition rate
        self.T_clock = T_clock # controller clock time
        
    @property
    def theta(self):
        L = self.resolution_px * self.px2mm
        return  math.atan(L/(2*self.f))

    @property
    def R(self):
        return self.compute_R()

    @property
    def G(self):
        T = - np.dot(self.R.T, self.center)
        return np.block([ [ self.R.T, T ], [np.zeros((1,3)), 1] ]) 

    @property
    def K(self):
        u0 = int(self.resolution_px/2) # image plane offset along X_c direction
        v0 = int(self.resolution_px/2) # image plane offset along Y_c direction
        return  np.array([[self.f*(1/self.px2mm),0,u0],\
                        [0,self.f*(1/self.px2mm),v0],\
                        [0,0,1]])

    @property
    def P(self):
        Pi_0 = np.hstack( (np.eye(3) , np.zeros((3,1))))
        return np.linalg.multi_dot([self.K, Pi_0 ,self.G ])

    @property
    def c_Pi(self):
        return self.compute_center_FoV()

    ''' compute camera orientation
    '''
    def compute_R(self):
        R_0 = np.array([[0,-1,0],\
                        [-1,0,0],   
                        [0,0,-1]]) # default conventional rotation wrt world
        # rotations are wrt the world reference frame, wilst the angles (alpha and beta)
        # are wrt the inertial reference frame
        R_alpha = np.array([[np.cos(-self.PT[0]),0,np.sin(-self.PT[0])],\
                        [0,1,0],\
                        [-np.sin(-self.PT[0]),0,np.cos(-self.PT[0])]])
        R_beta = np.array([[1,0,0],\
                        [0,np.cos(-self.PT[1]), -np.sin(-self.PT[1])],    
                        [0,np.sin(-self.PT[1]), np.cos(-self.PT[1])]])
        return np.linalg.multi_dot([R_beta,R_alpha,R_0])
        

    ''' compute FoV center (i.e., optical axis in Z=0)
    '''
    def compute_center_FoV(self):
        Q = self.P[:,:-1]
        q = self.P[:,-1].reshape(-1,1)
        Q_inv = np.linalg.inv(Q)
        c_I = np.array([[int(self.resolution_px/2)], \
                        [int(self.resolution_px/2)], \
                        [1]]) # image plane center in homogeneous coordinates
        lamda_c = - self.center[-1] /(np.dot(Q_inv[-1,:],c_I))
        # print(np.dot(Q_inv,c_I), self.R[:,-1],"\n \n")
        c_Pi = self.center + lamda_c*np.dot(Q_inv,c_I)
        return c_Pi

    ''' draw FoV center
    '''
    def draw_center_FoV(self,ax):
        ax.scatter3D(self.c_Pi[0], self.c_Pi[1],self.c_Pi[2], c='k', marker='o', s=10)

    ''' compute FoV vertices
    '''
    def compute_FoV(self):
        Q = self.P[:,:-1]
        Q_inv = np.linalg.inv(Q)
        # vector of image plane vertices 
        vertices = np.array([[0,0], \
                            [0,self.resolution_px],\
                            [self.resolution_px,self.resolution_px],\
                            [self.resolution_px,0]])
                            
        for i in range(np.shape(vertices)[0]):
            m = np.vstack((vertices[i,:].reshape(-1,1),[1])) # i-th image plane vertex in homogeneous coordinates
            lamda = - self.center[-1] / (np.dot(Q_inv[-1,:],m)) 
            M = self.center + lamda*np.dot(Q_inv,m)
            FoV = M[:-1].T if i == 0 else np.vstack((FoV, M[:-1].T))
        return FoV # 4x2 array

    ''' draw FoV (i.e., image plane projected onto the groundplane)
        ax: axis where to draw
    '''
    def draw_FoV(self,ax):
        # compute the FoV
        FoV = self.compute_FoV()
        # draw the FoV
        p = Polygon_plt(FoV, alpha=0.1)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0,  zdir="z")
        # draw optical axis
        ax.plot3D([np.asscalar(self.center[0]), np.asscalar(self.c_Pi[0])], \
                [np.asscalar(self.center[1]), np.asscalar(self.c_Pi[1])], \
                [np.asscalar(self.center[2]), np.asscalar(self.c_Pi[2])], \
                'b-.', LineWidth=1, alpha=0.5)
        # draw cone of view
        for i in range(np.shape(FoV)[0]):
            ax.plot3D([np.asscalar(self.center[0]), FoV[i,0]], \
                    [np.asscalar(self.center[1]), FoV[i,1]], \
                    [np.asscalar(self.center[2]), np.asscalar(self.c_Pi[2])], \
                    'b--', LineWidth=1, alpha=0.5)          
        ax.scatter3D(FoV[0,0], FoV[0,1],0, c='b', marker='x', s=10) # draw also the ptojection of point (0,0)                  


    ''' draw camera reference frame
        ax: axis where to draw
    '''
    def draw_cam_frame(self,ax):
        R = self.R
        ax.quiver(
            self.center[0], self.center[1],self.center[2], 
            R[0,0],R[1,0],R[2,0],
            length=1, color = 'red', alpha = .8, lw = 2,
            )
        ax.quiver(
            self.center[0], self.center[1],self.center[2],
            R[0,1],R[1,1],R[2,1],
            length=1, color = 'blue', alpha = .8, lw = 2,
        )
        ax.quiver(
            self.center[0], self.center[1],self.center[2],
            R[0,2],R[1,2],R[2,2],
            length=1,color = 'green', alpha = .8, lw = 2,
        )

    ''' project any 3D point in the image plane
        p: 3D point
    '''
    # ATTENTION!!!! 
    # The 3D -> 2D projection does not work due to the unknwon scale factor.
    def projector(self, p):
        C = np.vstack( (self.center, [1]) )      
        p = np.vstack((p.reshape(-1,1),[1])) # rewrite in homogeneous coordinates
        h = np.linalg.norm(p[:-1] - C[:-1])#  normalization factor of the homogeneous coordinates of p
        p_I = np.round(np.dot( self.P, p)/h)[:-1]   # p[-1] = 1 because of the homogeneous coordinates
        if p_I[0] < 0 or p_I[0] > self.resolution_px or p_I[1] < 0 or p_I[1] > self.resolution_px: # out of FoV
            p_I = None
        return p_I

    ''' check if any 3D point is inside the FoV
        p: 3D point
    '''
    def is_inside_FoV(self, p):
        FoV = self.compute_FoV()
        polygon = Polygon(FoV)
        pt = Point(p)
        p_inside_FoV = True if polygon.contains(pt) else False
        return p_inside_FoV
        

    ''' detector of the target 
        p_I: point in the image plane 
        d: true camera-target distance
    '''
    # N.B.: once the target is in the FoV, it has a certain probability p_D of being detected. 
    #       Here detection is model as a noise-free Bernulli process with success probability 
    #       p_D related to the camera focal length and the distance to the target.
    #       Being a noise-free process, it returns the input point if the detection has been
    #       successful, otherwise it returns None
    def detector(self, p_I, d):
        f_m = self.f * 1.0E-3 # focal length in [m]
        res = d/f_m # "target rsolution"
        gamma = 1.0E-2 # first hyperparameter
        eps = 0.3*1.0E+3 # second hyperparameter
        p_D = ( 1 + np.exp( gamma*( res - eps ) ) )**(-1) *( 1 + np.exp( -gamma*eps )) # detection success probability
        D = np.random.binomial(1,p_D) # detection event drawn from Bernoulli experiment 
        p_detector = p_I if D else None 
        return p_detector, p_D

    ''' detector of the target (once the target is in the FoV, it has a certain probability of being detected)
        p_inside_FoV: boolean true if a point is in the image plane 
        d: true camera-target distance
    '''
    # N.B.:  w.r.t. detector(self, p_I, d), here it is returned only if the point has been detected or not
    def is_detected(self, p_inside_FoV, d):
        d_critic = 50
        f_critic = 80.0*1.0E-3
        gamma = 1.0E-3
        eps = d_critic/f_critic
        f_m = self.f * 1.0E-3 # focal length in [m]
        res = d/f_m
        p_D = ( 1 + np.exp( gamma*( res - eps ) ) )**(-1) *( 1 + np.exp( -gamma*eps )) * int(p_inside_FoV) # detection probability
        D = np.random.binomial(1,p_D) # detection event drawn from Bernoulli experiment 
        is_detected = True if D else False 
        return is_detected, p_D


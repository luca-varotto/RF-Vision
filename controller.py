import numpy as np
import math
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt


''' 
Controller object
'''

class Controller:

    ''' Constructor '''
    def __init__(self, uav):
        self.uav = uav # searching platform associated to the controller


    ''' Tracking cost function '''
    def cost_function(self,target):
        h = 0.5 * np.linalg.norm(target - self.uav.camera.c_Pi)**2
        return h

    ''' compute gradient of the tracking cost function wrt PTZ parametersnd camera position 
        target: target position
    '''
    def compute_gradient(self, target):
                
        e_3 = np.array([[0],[0],[1]])
        c_I = np.array([[int(self.uav.camera.resolution_px/2)], \
                        [int(self.uav.camera.resolution_px/2)], \
                        [1]]) # image plane center in homogeneous coordinates
        
        R_0 = np.array([[0,-1,0],\
                        [-1,0,0],   
                        [0,0,-1]]) # default conventional rotation wrt world
        R_alpha = np.array([[np.cos(-self.uav.camera.PT[0]),0,np.sin(-self.uav.camera.PT[0])],\
                        [0,1,0],\
                        [-np.sin(-self.uav.camera.PT[0]),0,np.cos(-self.uav.camera.PT[0])]])
        R_beta = np.array([[1,0,0],\
                        [0,np.cos(-self.uav.camera.PT[1]), -np.sin(-self.uav.camera.PT[1])],    
                        [0,np.sin(-self.uav.camera.PT[1]), np.cos(-self.uav.camera.PT[1])]])

        R_alpha_derivative = np.array([[np.sin(-self.uav.camera.PT[0]),0,-np.cos(-self.uav.camera.PT[0])],\
                            [0,0,0],\
                            [np.cos(-self.uav.camera.PT[0]),0,np.sin(-self.uav.camera.PT[0])]]) # d R_alpha / d alpha
        R_beta_derivative = np.array([[0,0,0],\
                            [0,np.sin(-self.uav.camera.PT[1]), np.cos(-self.uav.camera.PT[1])],\
                            [0,-np.cos(-self.uav.camera.PT[1]), np.sin(-self.uav.camera.PT[1])]]) # d R_beta / d beta

        K_inv = np.linalg.inv(self.uav.camera.K)
        
        # derivative of c_Pi wrt alpha 
        den_alpha = np.asscalar( np.linalg.multi_dot([ e_3.T, R_beta, R_alpha, R_0, K_inv, c_I ]) )
        c_Pi_derivative_alpha =  self.uav.camera.center[-1] / (den_alpha)**2 \
                                * ( np.linalg.multi_dot([ e_3.T, R_beta, R_alpha_derivative, R_0, K_inv, c_I ]) ) \
                                * np.linalg.multi_dot([ R_beta, R_alpha, R_0, K_inv, c_I ]) \
                                - self.uav.camera.center[-1] / den_alpha  \
                                * np.linalg.multi_dot([ R_beta, R_alpha_derivative, R_0, K_inv, c_I ])
        
        # derivative of c_Pi wrt beta
        den_beta = np.asscalar( np.linalg.multi_dot([ e_3.T, R_beta, R_alpha, R_0, K_inv, c_I ]) )
        c_Pi_derivative_beta =  self.uav.camera.center[-1] / (den_beta)**2 \
                                * ( np.linalg.multi_dot([ e_3.T, R_beta_derivative, R_alpha, R_0, K_inv, c_I ]) ) \
                                * np.linalg.multi_dot([ R_beta, R_alpha, R_0, K_inv, c_I ]) \
                                - self.uav.camera.center[-1] / den_beta  \
                                * np.linalg.multi_dot([ R_beta_derivative, R_alpha, R_0, K_inv, c_I ])
                
        # gradient wrt c
        target = target.reshape(-1,1)
        gradient_c = ( self.uav.camera.c_Pi[:-1] - target[:-1] ) / np.linalg.norm(self.uav.camera.c_Pi - target) 
        # gradient_c = gradient_c - 0.5*np.sign(gradient_c)*abs(gradient_c)
        # gradient wrt f
        gradient_f = 0
        # gradient wrt alpha
        gradient_alpha = np.dot( ( self.uav.camera.c_Pi - target ).T, c_Pi_derivative_alpha )[0] / np.linalg.norm(self.uav.camera.c_Pi - target)
        # gradient wrt beta
        gradient_beta =  np.dot( ( self.uav.camera.c_Pi - target ).T , c_Pi_derivative_beta )[0] / np.linalg.norm(self.uav.camera.c_Pi - target)
        return gradient_c, gradient_f, gradient_alpha, gradient_beta

    def apply_control(self,target,max_control_iterations, K):
        cost_function = []
        K_c = K[0]
        K_alpha = K[1]
        K_beta = K[2]
        K_f = K[3]
        Delta_E = self.uav.E_tot - self.uav.E_t # UAV available energy

        iteration = 0
        while iteration <= max_control_iterations:
            # compute gradient
            gradient_c, gradient_f, gradient_alpha, gradient_beta = self.compute_gradient(target)
            
            # if the drone can move...
            if self.uav.scenario != 2:
                # ...update UAV position ...
                if Delta_E > 0: # ..., but only if there is energy available
                    # proposed next position
                    c_proposed = self.uav.c[:-1] -  K_c* gradient_c
                    if np.linalg.norm( self.uav.c[:-1] - c_proposed ) <= Delta_E: # is it compatible with energy constraint?
                        self.uav.c = np.vstack(( self.uav.c[:-1] - K_c* gradient_c , self.uav.c[-1]))
                
            # if the camera can move...
            counter = 0
            if self.uav.scenario != 1:
                # compute the proposed pan and tilt
                pan_proposed = self.uav.camera.PT[0]  - K_alpha*gradient_alpha[0]
                tilt_proposed = self.uav.camera.PT[1] - K_beta*gradient_beta[0]
                # lateral saturation
                sat_interval = [-np.pi/2 + self.uav.camera.theta*(1+0.5) , np.pi/2 - self.uav.camera.theta*(1+0.5)]
                # compute the diagonal displacement 
                L = self.uav.camera.resolution_px * self.uav.camera.px2mm # image plane size in [mm]
                gamma = math.atan(L/(np.sqrt(2)*self.uav.camera.f)) # diagonal angle of view
                delta_proposed = math.atan( np.sqrt( math.tan(pan_proposed)**2 + math.tan(tilt_proposed)**2) ) # diagonal angle between Z0 axis and cone of view
                # apply pan-tilt update only if the diagonal saturation is not met
                counter_th = 5
                while delta_proposed < - np.pi/2 + gamma*(1+0.2) or delta_proposed > np.pi/2 - gamma*(1+0.2):
                    K_alpha *= 0.2
                    K_beta *= 0.2
                    pan_proposed = self.uav.camera.PT[0]  - K_alpha*gradient_alpha[0]
                    tilt_proposed = self.uav.camera.PT[1] - K_beta*gradient_beta[0]
                    delta_proposed = math.atan( np.sqrt( math.tan(pan_proposed)**2 + math.tan(tilt_proposed)**2) )
                    counter +=1
                    if counter == counter_th: # avoid infinite loops
                        break
                if counter < counter_th:
                    # pan update
                    self.uav.camera.PT[0] = min(max(sat_interval[0],pan_proposed),sat_interval[1])
                    # tilt update
                    self.uav.camera.PT[1] = min(max(sat_interval[0],tilt_proposed),sat_interval[1])
                else:
                #     # compute the saturated value of the diagonal angle
                #     delta_sat = min(max(- np.pi/2 + gamma*(1+0.2),delta_proposed),np.pi/2 - gamma*(1+0.2))
                    move_pan = np.random.binomial(1,0.5)
                    if move_pan:
                        self.uav.camera.PT[0] = min(max(sat_interval[0],self.uav.camera.PT[0]  - K[0]*gradient_alpha[0]),sat_interval[1])
                        self.uav.camera.PT[1] = 0#min(max(sat_interval[0],tilt_proposed),sat_interval[1])
                    else:
                        self.uav.camera.PT[0] = 0
                        self.uav.camera.PT[1] = min(max(sat_interval[0],self.uav.camera.PT[1]  - K[1]*gradient_beta[0]),sat_interval[1])
                #     if math.tan(delta_sat)**2 - math.tan(self.uav.camera.PT[0])**2 >= 0:
                #         # tilt update
                #         self.uav.camera.PT[1] = math.atan( np.sqrt( math.tan(delta_sat)**2 - math.tan(self.uav.camera.PT[0])**2) )
                #     else: 
                #         self.uav.camera.PT[1] = min(max(sat_interval[0],tilt_proposed),sat_interval[1])
                #         # if math.tan(delta_sat)**2 - math.tan(self.uav.camera.PT[1])**2 >= 0:
                #         self.uav.camera.PT[0] = math.atan( np.sqrt( math.tan(delta_sat)**2 - math.tan(self.uav.camera.PT[1])**2) )
        
            iteration += (1 + counter)
        # print(-K_alpha*gradient_alpha[0]*180/np.pi, self.uav.camera.PT[0]*180/np.pi, pan_proposed*180/np.pi, sat_interval[1]*180/np.pi)
        # print(-K_beta*gradient_beta[0]*180/np.pi, self.uav.camera.PT[1]*180/np.pi, tilt_proposed*180/np.pi, sat_interval[1]*180/np.pi)
        cost_function.append(self.cost_function(target))

        return cost_function


    # target tracking utility as function of pan and tilt angles
    def J(self,x,y, target):

            e_3 = np.array([[0],[0],[1]])
            c_I = np.array([[int(self.uav.camera.resolution_px/2)], \
                            [int(self.uav.camera.resolution_px/2)], \
                            [1]]) # image plane center in homogeneous coordinates
            
            R_0 = np.array([[0,-1,0],\
                            [-1,0,0],   
                            [0,0,-1]]) # default conventional rotation wrt world
            R_alpha = np.array([[np.cos(-x),0,np.sin(-x)],\
                            [0,1,0],\
                            [-np.sin(-x),0,np.cos(-x)]])
            R_beta = np.array([[1,0,0],\
                            [0,np.cos(-y), -np.sin(-y)],    
                            [0,np.sin(-y), np.cos(-y)]])

            K_inv = np.linalg.inv(self.uav.camera.K)

            c_Pi = self.uav.camera.center -  self.uav.camera.center[-1] \
                / np.linalg.multi_dot([ e_3.T, R_beta, R_alpha, R_0, K_inv, c_I ]) \
                    * np.linalg.multi_dot([ R_beta, R_alpha, R_0, K_inv, c_I ]) 
            J = 0.5 * np.linalg.norm(target - c_Pi)**2
            return J

    # plot a 2D function
    def plot_J(self, target, prev_state, state):

        alpha = np.arange(-np.pi/2+self.uav.camera.theta,np.pi/2-self.uav.camera.theta,np.pi/100)
        beta = np.arange(-np.pi/2+self.uav.camera.theta,np.pi/2-self.uav.camera.theta,np.pi/100)
        X,Y = meshgrid(alpha,beta) # grid of point
        Z = self.J(X, Y, target) # evaluation of the function on the grid

        fig= plt.figure()
        # ax = plt.gca()
        # im = imshow(Z,cmap=cm.RdBu, \
        #     extent=[-np.pi/2+self.uav.camera.theta,\
        #             np.pi/2-self.uav.camera.theta,\
        #             -np.pi/2+self.uav.camera.theta,\
        #             np.pi/2-self.uav.camera.theta]) # cmap on the plane
        # # cset = contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
        # clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
        # colorbar(im) # adding the colobar on the right
        
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                            cmap=cm.RdBu,linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # if prev_state and state:
        #     plt.arrow(prev_state[0], prev_state[1], (state[0]-prev_state[0]),(state[1]-prev_state[1]), fc="k", ec="k", head_width=0.1, head_length=0.1)

        # ax.zaxis.set_major_locator(LinearLocator(10))
        # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        title(r'$J(\alpha,\beta)$', fontsize=18)
        ax.set_xlabel(r'$\alpha$ [rad]',fontsize=18, labelpad=20)
        ax.set_ylabel(r'$\beta$ [rad]',fontsize=18, labelpad=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        # ax.zaxis.set_tick_params(labelsize=18)
        plt.show()
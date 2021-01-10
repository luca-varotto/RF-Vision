                    # ***** PLOT DETECTION PROBABILITY FUNCTION *****

import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

# detection probability as function od zoom and camera-target distance
def p_D(x,y):
    d_critic = 50
    f_critic = 80.0*1.0E-3
    gamma = 1.0E-3
    eps = d_critic/f_critic
    print(gamma, eps)
    x_min = np.amin(x)
    y_min = np.amin(y)
    z0 = ( 1 + np.exp( gamma*( x_min/y_min - eps) ))**(-1)
    return ( ( 1 + np.exp( gamma*( x/y - eps) ))**(-1) / z0 )

# plot a 2D p_D
def plot2D():
    d = np.arange(0.0,40.0,0.5)
    f = np.arange(10.0*1.0E-3,100.0*1.0E-3,1.0E-3)
    X,Y = meshgrid(d, f) # grid of point
    Z = p_D(X, Y) # evaluation of the function on the grid

    # im = imshow(Z,cmap=cm.RdBu) # drawing the function
    # # adding the Contour lines with labels
    # cset = contour(Z,np.arange(-1,1.5,0.2),linewidths=2,cmap=cm.Set2)
    # clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
    # colorbar(im) # adding the colobar on the right
    # # latex fashion title
    # title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
    # show()

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
                        cmap=cm.RdBu,linewidth=0, antialiased=False)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(surf, shrink=0.5, aspect=5)
    title(r'$\left[ 1 + e^{\gamma(\varepsilon_t - \epsilon)}\right]^{-1}\left( 1 + e^{-\gamma \epsilon} \right)$', fontsize=18)
    ax.set_xlabel(r'$d\left( \mathbf{c}_t, \mathbf{p}_t  \right)$ [m]',fontsize=18, labelpad=20)
    ax.set_ylabel(r'$f_t$ [m]',fontsize=18, labelpad=20)
    ax.xaxis.set_tick_params(labelsize=18)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.zaxis.set_tick_params(labelsize=18)
    plt.show()


                    # ***** SAVE/LOAD DATA *****
import pickle
import os

'''
Object that saves/loads data
'''

class DataImporter:

        ''' Constructor '''
        def __init__(self):
                pass

        # save data
        def save_obj(self, obj, name ):
                with open(name + '.pkl', 'wb') as f:
                        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        # load data
        def load_obj(self, name):
                with open(name + '.pkl', 'rb') as f:
                        return pickle.load(f)

        def save(self, dir_list, obj,name):
                basic_dir = dir_list[0]
                current_dir = basic_dir
                for i in range(1,len(dir_list)):
                        try:
                                os.mkdir(os.path.join(current_dir, dir_list[i]))
                        except OSError as error:
                                pass
                        current_dir += dir_list[i] + "/" 
                self.save_obj(obj, current_dir + name)
                print("Saved: " + name)

                    # ***** PLOTTER OF THE ENVIRONMENT *****

'''
Object that plots everything of the environment 
'''
class Plotter:

        ''' Constructor '''
        def __init__(self, agent, uav, pf):
            self.agent = agent
            self.uav = uav
            self.pf = pf

        ''' draw environment (frames, objects, ...) 
            ax: axis where to draw
        '''
        def env_plot(self, ax):

                # dimension of the (square) environment
                W = np.max((abs(self.agent.p[0]),abs(self.agent.p[1]),abs(self.uav.c[0]),abs(self.uav.c[1]),10)) # width and length
                W_z = self.uav.c[-1]+1 # height

                # plot particles and estimate
                self.pf.pf_plotting_tool(ax)

                # draw agent and frame
                self.agent.draw_agent(ax)
                # agent.draw_agent_frame(ax)
                # draw UAV and frame
                self.uav.draw_uav(ax)
                self.uav.draw_uav_frame(ax)
                # draw camera, frame, Fov and FoV center
                self.uav.camera.draw_cam_frame(ax)
                self.uav.camera.draw_FoV(ax)
                self.uav.camera.draw_center_FoV(ax) 

                # world reference frame
                ax.quiver(0,0,0, 
                1,0,0,
                length=2, color = 'red', alpha = .5, lw = 1,
                )
                ax.quiver(0, 0, 0, 
                0,1,0, 
                length=2, color = 'blue', alpha = .5, lw = 1,
                )
                ax.quiver(0, 0, 0, 
                0,0,1, 
                length=2,color = 'green', alpha = .5, lw = 1,
                )

                ax.set_xlabel('X')
                ax.xaxis.label.set_color('red')
                ax.set_ylabel('Y')
                ax.yaxis.label.set_color('blue')
                ax.set_zlabel('Z')
                ax.zaxis.label.set_color('green') 
                ax.set_xlim3d(-2*W, 2*W)
                ax.set_ylim3d(-2*W, 2*W)
                ax.set_zlim3d(0, W_z)
                title_str = r"$\mathbf{c}=$ (%2.1f,%2.1f), $\alpha=$ %2.1f, $\beta=$ %2.1f, $f=$ %2.1f" \
                                %(self.uav.c[0],self.uav.c[1],self.uav.camera.PT[0]*180/np.pi,self.uav.camera.PT[1]*180/np.pi, self.uav.camera.f) 
                ax.set_title(title_str)

        ''' plot data collected/computed during the simulation
        '''
        def plot_results(self, Tx_data, Rx_data, uav_energy_level, est_performance,control_cost_functions, vision_events, est_variance):
                fig = plt.figure(figsize=(10,6))
                
                # how much did it take to the UAV to find the target (if found)
                search_duration = len(uav_energy_level)
                
                ax_sub1 = plt.subplot(3,2,1) # Tx vs Rx RSSI
                plt.xlabel(r"$t$", fontsize=18)
                plt.ylabel("RSSI",fontsize=18)
                plt.plot(Tx_data,label='Tx data', marker='o')
                plt.plot(Rx_data,label='Rx data',marker='o')
                plt.legend()
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax_sub2 = plt.subplot(3,2,2) # UAV energy level 
                plt.xlabel(r"$t$", fontsize=18)
                plt.ylabel(r"$\Delta E_t$",fontsize=18)
                plt.plot(uav_energy_level)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                ax_sub3 = plt.subplot(3,2,3) # estimation performance
                plt.xlabel(r"$t$", fontsize=18)
                plt.ylabel(r"$ norm( \mathbf{p},\hat{\mathbf{p}} ) $",fontsize=18)
                plt.plot(est_performance)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                ax_sub4 = plt.subplot(3,2,4) # control cost function (at the end of each control session)
                plt.xlabel(r"$t$", fontsize=18)
                plt.ylabel(r"$ J( \mathbf{c}_{\Pi}}] ) $",fontsize=18)
                plt.plot(control_cost_functions)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                ax_sub5 = plt.subplot(3,2,5) # vision events and detection probabilities
                plt.xlabel(r"$t$", fontsize=18)
                plt.plot(vision_events[0,:search_duration], label='vision events', marker='o')
                plt.plot(vision_events[1,:search_duration], label=r'$p_D$', lineStyle='-')
                plt.legend()
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)

                ax_sub6 = plt.subplot(3,2,6) # estimate variance
                plt.plot(est_variance)
                plt.xlabel(r"$t$", fontsize=18)
                plt.ylabel("est. variance",fontsize=18)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                
                plt.tight_layout()
                plt.show()
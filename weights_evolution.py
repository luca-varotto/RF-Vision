import numpy as np    
import matplotlib.pyplot as plt
from tqdm import tqdm

############################################################################################################################

# define some parameters
Ns = 50 # number of particles
n = 2.0 # true attenuation gain
r_0 = -35.0 # true RSSI at d_0
n_hat = np.random.normal(n,1.0E-1) # estimated attenuation gain
r_0_hat = np.random.normal(r_0,1.0E-0) # estimated RSSI at d_0
p = np.array([[10.0],[0.0]]) # true target location (suppose target fixed)
Delta = ( r_0 - r_0_hat ) - 10*(n-n_hat)*np.log10(np.linalg.norm(p))
sigma_omega = 1.0E-7 # process noise std.dev.

def l_RF(v_prime,p_i):
  l =  np.exp( - ( (v_prime + 10*n_hat*np.log10(np.linalg.norm(p_i)/np.linalg.norm(p))) / (np.sqrt(2)*sigma_hat) )**2 ) 
  return l#/(np.sqrt(2*np.pi)*sigma_hat)

def l_v(p_i):
  f_m = 100 * 1.0E-3 # focal length in [m]
  res = np.linalg.norm(p_i)/f_m # "target resolution"
  gamma = 1.0E-2 # first hyperparameter
  eps = 0.3*1.0E+3 # second hyperparameter
  p_D = ( 1 + np.exp( gamma*( res - eps ) ) )**(-1) *( 1 + np.exp( -gamma*eps ))
  return p_D

# decide what you want to do
# weight_evolution --> 0
# minimum importance loss --> 1
# distribution multiplicative ranging error --> 2
task = 0

if task == 0:
  # define and plot the particles
  # P_opt: the set of particles whose distance to Rx is approx Rx-Tx distance (in this case, 
  # whose norm is approx the norm of p)
  for th_ratio in [0.75,0.9]: # percentage of particles outside P_opt
    p_closest = np.array([ [9.0], [0.0] ]) # closests particle to p
    ax = plt.gca()
    delta = 0.3 # for annotations
    particles = [] # particles set
    th = int(th_ratio*Ns)
    opt_th = 5 # threshold to discriminate particles inside and outside P_opt
    for i in range(Ns+1):
      if i < th: # particles outside P_opt
        particle = np.random.uniform(-30,30,size=(2,1)) # random placement in [-30,30]x[-30,30]
        while abs(np.linalg.norm(particle) - np.linalg.norm(p)) < opt_th: # if inside P_opt ... 
          particle = np.random.uniform(-30,30,size=(2,1)) # ...repeat random placement
      elif i == th: # closest particle
        particle = p_closest
      else: # particles inside P_opt
        particle = np.random.uniform(-np.linalg.norm(p)-opt_th,np.linalg.norm(p)+opt_th,size=(2,1))
        while abs(np.linalg.norm(particle) - np.linalg.norm(p)) > opt_th: 
          particle = np.random.uniform(-np.linalg.norm(p)-opt_th,np.linalg.norm(p)+opt_th,size=(2,1))
        ax.annotate(str(i), (particle[0] + delta,particle[1] + delta),size=18)
      plt.scatter(particle[0],particle[1], marker = 'o', c='g', s=50)
      particles.append(particle)
    plt.scatter(0,0, marker = 'o', c='k', s=100)
    ax.annotate("Rx", (delta,delta),size=18)
    plt.scatter(p[0], p[1], marker = 's', c='r', s=100)
    ax.annotate("Tx", (p[0] + 2*delta,p[1]-5*delta),size=18)
    theta = np.linspace(-np.pi, np.pi, 200)
    radius = max(p)
    plt.plot(radius*np.sin(theta), radius*np.cos(theta), c='r', linestyle='--')
    ax.axis('equal')
    plt.grid('on')
    # plt.savefig("../Report/images/particles.png")

    # time interval
    T = 100
    time = range(T)

    # define the number of MC tests
    num_MC_tests = 150

    for T_RF in [2]: # RSSI transmission rates to test
      for sigma in [2]: # true noise std.dev. values to test
        sigma_hat = np.random.normal(sigma,1.0E-1) # estimated noise std.dev.

        # storage buffers
        i_MAP_RF = np.empty((num_MC_tests,T),dtype=int) # store index of MAP estimate according to RF 
        i_MAP_RFV = np.empty((num_MC_tests,T),dtype=int) # store index of MAP estimate according to RF+V
        i_MAP_V = np.empty((num_MC_tests,T),dtype=int) # store index of MAP estimate according to V
        i_closest = np.empty((num_MC_tests,T),dtype=int) # store index of the closest particle to the target (optimal particle)
        
        e_RF = np.empty((num_MC_tests,T)) #
        e_RFV = np.empty((num_MC_tests,T)) #
        e_V = np.empty((num_MC_tests,T)) # 

        # MC experiment
        for MC_sim in tqdm(range(num_MC_tests)): 
          
          # weights evolution buffers
          weight_ev_RF = 1/Ns*np.ones((len(particles),T)) # according to RF likelihood
          weight_ev_RFV = 1/Ns*np.ones((len(particles),T)) # according to RF+V likelihood
          weight_ev_V = 1/Ns*np.ones((len(particles),T)) # according to V likelihood

          # initialize randomly MAP estimate index (since weights are uniform)
          i_MAP_RF[MC_sim,0] = np.random.uniform(0,len(particles))
          i_MAP_RFV[MC_sim,0] = i_MAP_RF[MC_sim,0]
          i_MAP_V[MC_sim,0] = i_MAP_RF[MC_sim,0]
          i_closest[MC_sim,0] = th

          e_RF[MC_sim,0] = np.linalg.norm(particles[i_MAP_RF[MC_sim,0]] - p_closest)
          e_RFV[MC_sim,0] = np.linalg.norm(particles[i_MAP_RFV[MC_sim,0]] - p_closest)
          e_V[MC_sim,0] = np.linalg.norm(particles[i_MAP_V[MC_sim,0]] - p_closest)

          for t in time[1:]: # t=1:T-1

            # (modified) measurement noise
            v = np.random.normal(Delta,sigma) 
            
            # compute the distance between the target and each particle
            distances = np.empty(len(particles))
            for i in range(len(particles)):
              particles[i] += np.random.normal(0.0, sigma_omega, size=np.shape(particles[i])) # process model applied on particles
              distances[i] = np.linalg.norm(particles[i] - p)
              # apply RF likelihood
              if t % T_RF ==0:
                weight_ev_RF[i,t] = weight_ev_RF[i,t-1] * l_RF(v,particles[i])  
                weight_ev_RFV[i,t] = weight_ev_RFV[i,t-1] * l_RF(v,particles[i])
              else:
                weight_ev_RF[i,t] = weight_ev_RF[i,t-1]
                weight_ev_RFV[i,t] = weight_ev_RFV[i,t-1]
              weight_ev_V[i,t] = weight_ev_V[i,t-1]
            # compute the index of the closest particle (should be constant)
            i_closest[MC_sim,t] = np.argmin(distances)
            
            # according to the camera control law, move the FoV towards the 
            # MAP estimate of the previous time instant
            # N.B.: in the following we suppose only MAP particle inside FoV
            # RF
            is_in_FoV = i_MAP_RF[MC_sim,t-1] == i_closest[MC_sim,t] # target (opt. particle) in the FoV 
            p_D = l_v(particles[i_MAP_RF[MC_sim,t-1]])*int(is_in_FoV) # probability of detection in particle of MAP estimate
            is_detected = np.random.binomial(1,p_D)  # target detected?
            if is_detected: 
              mask = range(len(particles))
              mask = np.delete(mask,i_MAP_RF[MC_sim,t-1])
              weight_ev_RF[mask,t] *= 0 # stop the search
            # RF+V
            is_in_FoV = i_MAP_RFV[MC_sim,t-1] == i_closest[MC_sim,t] # target (opt. particle) in the FoV 
            p_D = l_v(particles[i_MAP_RFV[MC_sim,t-1]])*int(is_in_FoV) # probability of detection in particle of MAP estimate
            is_detected = np.random.binomial(1,p_D) # target detected?
            for i in range(len(particles)):
              if i != i_MAP_RFV[MC_sim,t-1]: # out of FoV particle
                  if not is_detected:
                    weight_ev_RFV[i,t] *= 1 
                  else:
                    weight_ev_RFV[i,t] *= 0
              else: # MAP particle (inside FoV)
                if is_detected:
                  weight_ev_RFV[i,t] *= l_v(particles[i])  
                else:
                  weight_ev_RFV[i,t] *= 1-l_v(particles[i])
            # V
            is_in_FoV = i_MAP_V[MC_sim,t-1] == i_closest[MC_sim,t] # target (opt. particle) in the FoV 
            p_D = l_v(particles[i_MAP_V[MC_sim,t-1]])*int(is_in_FoV) # probability of detection in particle of MAP estimate
            is_detected = np.random.binomial(1,p_D) # target detected?
            for i in range(len(particles)):
              if i != i_MAP_V[MC_sim,t-1]: # out of FoV particle
                  if not is_detected:
                    weight_ev_V[i,t] *= 1 
                  else:
                    weight_ev_V[i,t] *= 0
              else: # MAP particle (inside FoV)
                if is_detected:
                  weight_ev_V[i,t] *= l_v(particles[i])  
                else:
                  weight_ev_V[i,t] *= 1-l_v(particles[i])

            # compute the index of the MAP estimate
            if t % T_RF ==0:
              max_index_list_RF = [idx for idx, w in enumerate(weight_ev_RF[:,t]) if w == np.max(weight_ev_RF[:,t])]
              i_MAP_RF[MC_sim,t] = max_index_list_RF[np.random.randint(0,len(max_index_list_RF))]
            else:
              i_MAP_RF[MC_sim,t] = i_MAP_RF[MC_sim,t-1]
            max_index_list_RFV = [idx for idx, w in enumerate(weight_ev_RFV[:,t]) if w == np.max(weight_ev_RFV[:,t])]
            i_MAP_RFV[MC_sim,t] = max_index_list_RFV[np.random.randint(0,len(max_index_list_RFV))]
            max_index_list_V = [idx for idx, w in enumerate(weight_ev_V[:,t]) if w == np.max(weight_ev_V[:,t])]
            i_MAP_V[MC_sim,t] = max_index_list_V[np.random.randint(0,len(max_index_list_V))]

            e_RF[MC_sim,t] = np.linalg.norm(particles[i_MAP_RF[MC_sim,t]] - p_closest)
            e_RFV[MC_sim,t] = np.linalg.norm(particles[i_MAP_RFV[MC_sim,t]] - p_closest)
            e_V[MC_sim,t] = np.linalg.norm(particles[i_MAP_V[MC_sim,t]] - p_closest)

        # plot weights of the last MC test
        # fig = plt.figure(figsize=(10,8))
        # plt.subplot(2,1,1)
        # for i in range(th,len(particles)):
        #   plt.plot(weight_ev_RF[i,:10], label=r"$w_t^{("+str(i)+")}$") # plot weights of the last MC sim
        # plt.legend(fontsize=25,fancybox=True, framealpha=0.5)
        # plt.subplot(2,1,2)
        # for i in range(th,len(particles)):
        #   plt.plot(range(1,10),weight_ev_RF[i,1:10]-weight_ev_RF[i,:9], label=r"$w_t^{("+str(i)+")}$") # plot weights of the last MC sim
        # plt.legend(fontsize=25,fancybox=True, framealpha=0.5)

        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()

        mean_RF = np.mean(i_MAP_RF-i_closest,axis=0)
        plt.plot(time,mean_RF,linestyle='-',linewidth=2, c='b', label='RF')
        std_RF = np.std(i_MAP_RF-i_closest,axis=0)
        plt.plot(time,mean_RF-std_RF,linestyle='--',linewidth=1, c='b')
        plt.plot(time,mean_RF+std_RF,linestyle='--',linewidth=1, c='b')
        ax.fill_between(time, mean_RF-std_RF, mean_RF+std_RF ,alpha=0.2, facecolor='b')

        mean_RFV = np.mean(i_MAP_RFV-i_closest,axis=0)
        plt.plot(time,mean_RFV,linestyle='-',linewidth=2, c='g', label='RF+V')
        std_RFV = np.std(i_MAP_RFV-i_closest,axis=0)
        plt.plot(time,mean_RFV-std_RFV,linestyle='--',linewidth=1, c='g')
        plt.plot(time,mean_RFV+std_RFV,linestyle='--',linewidth=1, c='g')
        ax.fill_between(time, mean_RFV-std_RFV, mean_RFV+std_RFV ,alpha=0.2, facecolor='g')

        mean_V = np.mean(i_MAP_V-i_closest,axis=0)
        plt.plot(time,mean_V,linestyle='-',linewidth=2, c='y', label='V')
        std_V = np.std(i_MAP_V-i_closest,axis=0)
        plt.plot(time,mean_V-std_V,linestyle='--',linewidth=1, c='y')
        plt.plot(time,mean_V+std_V,linestyle='--',linewidth=1, c='y')
        ax.fill_between(time, mean_V-std_V, mean_V+std_V ,alpha=0.2, facecolor='y')

        plt.plot(time,np.zeros(len(time)),linestyle=':', c='k')

        plt.legend(fontsize=25,fancybox=True, framealpha=0.5)
        plt.xlabel(r'$t$',fontsize=30)
        plt.ylabel(r'$\Delta i_t$',fontsize=30)
        plt.xticks(range(0,T+1,10),fontsize=25)
        plt.yticks(fontsize=25)
        # plt.savefig("../Report/images/weights_evolution_sigma" + str(sigma)+\
        #   "_Trf" + str(T_RF) +".png")
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.grid(ls = ':', lw = 0.5)
        plt.tight_layout()
        plt.show()

        fig = plt.figure(figsize=(9,6))
        ax = plt.gca()

        mean_RF = np.mean(e_RF,axis=0)
        plt.plot(time,mean_RF,linestyle='-',linewidth=2, c='b', label='RF')
        std_RF = np.std(e_RF,axis=0)
        plt.plot(time,mean_RF-std_RF,linestyle='--',linewidth=1, c='b')
        plt.plot(time,mean_RF+std_RF,linestyle='--',linewidth=1, c='b')
        ax.fill_between(time, mean_RF-std_RF, mean_RF+std_RF ,alpha=0.2, facecolor='b')

        mean_RFV = np.mean(e_RFV,axis=0)
        plt.plot(time,mean_RFV,linestyle='-',linewidth=2, c='g', label='RF+V')
        std_RFV = np.std(e_RFV,axis=0)
        plt.plot(time,mean_RFV-std_RFV,linestyle='--',linewidth=1, c='g')
        plt.plot(time,mean_RFV+std_RFV,linestyle='--',linewidth=1, c='g')
        ax.fill_between(time, mean_RFV-std_RFV, mean_RFV+std_RFV ,alpha=0.2, facecolor='g')

        mean_V = np.mean(e_V,axis=0)
        plt.plot(time,mean_V,linestyle='-',linewidth=2, c='y', label='V')
        std_V = np.std(e_V,axis=0)
        plt.plot(time,mean_V-std_V,linestyle='--',linewidth=1, c='y')
        plt.plot(time,mean_V+std_V,linestyle='--',linewidth=1, c='y')
        ax.fill_between(time, mean_V-std_V, mean_V+std_V ,alpha=0.2, facecolor='y')

        plt.plot(time,np.zeros(len(time)),linestyle=':', c='k')

        plt.legend(fontsize=25,fancybox=True, framealpha=0.5)
        plt.xlabel(r'$t$',fontsize=30)
        plt.ylabel(r'$e_t \; [m]$',fontsize=30)
        plt.xticks(range(0,T+1,10),fontsize=25)
        plt.yticks(fontsize=25)
        # plt.savefig("../Report/images/weights_evolution_sigma" + str(sigma)+\
        #   "_Trf" + str(T_RF) +".png")
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.grid(ls = ':', lw = 0.5)
        plt.tight_layout()
        plt.show()


############################################################################################################################

from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.stats import norm, lognorm

# underlying noise level 
sigma=2
sigma_hat = np.random.normal(sigma,1.0E-1) # estimated noise std.dev.

# relative importance loss ...
# ... as function of particle position
def relative_importance_loss(x,y, v_prime):
    p_i_norm = np.sqrt(x**2+y**2) # particle distance to Rx
    p_norm = np.linalg.norm(p) # Tx-Rx distance
    sigma_hat = np.random.normal(sigma,1.0E-1) # estimated noise std.dev.
    # relative importance loss
    z = abs( np.exp(- ((v_prime + 10*n_hat*np.log10(p_i_norm/p_norm) )/(np.sqrt(2)*sigma_hat) )**2  )\
      / (np.sqrt(2*np.pi)*sigma_hat) - 1 )
    return z
# ... as function of particle distance
def relative_importance_loss(d_i,v_prime):
    p_norm = np.linalg.norm(p) # Tx-Rx distance
    # relative importance loss
    z = abs( np.exp(- ((v_prime + 10*n_hat*np.log10(d_i/p_norm))/(np.sqrt(2)*sigma_hat))**2  )\
      / (np.sqrt(2*np.pi)*sigma_hat) - 1 )
    return z

if task == 1:
  # # plot relative importance loss (without noise)
  # X,Y = meshgrid(np.arange(-30.0,30.0,0.5),np.arange(-30.0,30.0,0.5)) # grid of point
  # v_prime = 0
  # Z = relative_importance_loss(X,Y,v_prime) # evaluation of the function on the grid
  # fig = plt.figure()
  # ax = fig.gca(projection='3d')
  # surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, 
  #                     cmap=cm.RdBu,linewidth=0, antialiased=False)
  # # ax.zaxis.set_major_locator(LinearLocator(10))
  # # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
  # fig.colorbar(surf, shrink=0.5, aspect=5)
  # title(r'$\Delta w$', fontsize=30)
  # ax.set_xlabel(r'$X$ [m]',fontsize=30, labelpad=20)
  # ax.set_ylabel(r'$Y$ [m]',fontsize=30, labelpad=20)
  # ax.xaxis.set_tick_params(labelsize=18)
  # ax.yaxis.set_tick_params(labelsize=18)
  # ax.zaxis.set_tick_params(labelsize=18)

  # compare the minimum of Delta_w with and without noise 
  distances = np.arange(1.0E-2,30,0.1) # all distances in which Delta_w is evaluated
  num_tests = 200 # number od MC tests 
  # Delta_w without noise
  Delta_w = np.empty(len(distances))   
  # Delta_w with noise
  Delta_w_noise = np.empty((num_tests,len(distances)))
  min_list = [] # list of min for every test
  for test in range(num_tests):
    for j,dist in enumerate(distances):
      if test ==0:
        Delta_w[j] = relative_importance_loss(dist,0)
      v_prime = np.random.normal(Delta,sigma) # noise
      Delta_w_noise[test,j] = relative_importance_loss(dist,v_prime)
    min_list.append(distances[np.argmin(Delta_w_noise[test,:])])
  fig = plt.figure(figsize=(10,6))
  ax = plt.gca()
  # average minimum 
  mean = np.mean(min_list)
  # std of of the minimums
  std = np.std(min_list) 
  pdf = norm.pdf(distances,mean,std) # Gaussian pdf fit
  # plot pdf
  plt.plot(distances,pdf, c='b', linewidth=2, \
    label=r'$p(\min \{ \Delta w_t \})$, $v_{Rx}^{\prime}\sim \mathcal{N}(\Delta_t,\sigma_{Rx})$')
  ax.fill_between(distances,pdf, color='b', alpha=0.1)
  plt.vlines(mean,0,norm.pdf(mean,mean,std), colors='b', linestyles='--') # highlight mean value
  plt.vlines(distances[np.argmin(Delta_w)],0,norm.pdf(mean,mean,std), colors='k', \
    linestyles='--', label=r'$\min \{ \Delta w_t \} $, $v_{Rx}^{\prime}=0$') # highlight mean value
  plt.xlabel(r'$d$ [m]',fontsize=30)
  plt.ylabel('Gaussian distribution',fontsize=30)
  plt.xlim([0,distances[-1]])
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.legend(fontsize=25,fancybox=True, framealpha=0.5)
  plt.show()
  # plot relative importance loss as function of distance
  # plt.plot(distances,Delta_w,linestyle='-',linewidth=2, c='g', label='Delta')
  # mean = np.mean(Delta_w_noise,axis=0)
  # plt.plot(distances,mean,linestyle='-',linewidth=2, c='b', label='Delta_noisy')
  # std = np.std(Delta_w_noise,axis=0)
  # plt.plot(distances,mean-std,linestyle='--',linewidth=1, c='b')
  # plt.plot(distances,mean+std,linestyle='--',linewidth=1, c='b')
  # ax.fill_between(distances,mean-std, mean+std ,alpha=0.2, facecolor='b')
  # plt.legend( fontsize=25,fancybox=True, framealpha=0.5)
  # plt.xlabel(r'$d$ [m]',fontsize=30)
  # plt.ylabel(r'$\Delta w$',fontsize=30)
  # plt.xticks(fontsize=25)
  # plt.yticks(fontsize=25)
  # plt.show()

if task == 2:
  # plot log-normal distribution of the multiplicative ranging error
  fig =plt.figure(figsize=(10,6))
  ax = plt.gca()
  x_all = np.arange(0, 2.5, 0.01)
  Deltas = []
  for n_std in [1.0E-1]:
    for r_0_std in [1.0E-0]:
      Deltas.append( ( r_0 - np.random.normal(r_0,r_0_std) ) - \
        10*(n-np.random.normal(n,n_std))*np.log10(np.linalg.norm(p)) )
  for Delta in Deltas:
    for sigma in [2,4,6]:
      pdf = lognorm.pdf(x_all,sigma/(10*n_hat),Delta/(10*n_hat)) #lognorm.pdf(x_all,sigma/(10*n_hat),Delta/(10*n_hat))
      plt.plot(x_all,pdf, label="Delta= %2.1f, sigma= %d" %(Delta,sigma))
  plt.vlines(1,0,4.5, colors='k', linestyles='--', label="Delta= %2.1f, sigma= %d" %(0,0)) 
  plt.xlim([min(x_all), max(x_all)])
  plt.xticks(fontsize=25)
  plt.yticks(fontsize=25)
  plt.xlabel(r'$10^{-v_{Rx}^{\prime}}/10\hat{n}$',fontsize=30)
  plt.ylabel(r'$p(10^{-v_{Rx}^{\prime}}/10\hat{n})$',fontsize=30)
  plt.legend(fontsize=25,fancybox=True, framealpha=0.5)
  plt.show()
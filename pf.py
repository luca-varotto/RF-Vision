from numpy.random import uniform, randn, multivariate_normal
import numpy as np
from filterpy.monte_carlo import systematic_resample
from numpy.linalg import norm
import scipy.stats
import matplotlib.pyplot as plt

''' 
SIR Particle Filter
'''

class PF:

    ''' Constructor '''
    def __init__(self, uav,  strategy, p0_mu=[0.0,0.0,0.0], p0_sigma=100.0 ,N=100,  draw_particles_flag=False):
        self.strategy = strategy # which information is used to updat the particles? 1: R, 2:V, 3: RF+V
        self.N = N # number of particles 
        self.particles = np.zeros((self.N, 3))
        self.draw_particles_flag = draw_particles_flag
        self.uav = uav # searching platform that uses the PF
        self.particles[:,:-1] = multivariate_normal(p0_mu[:-1], (p0_sigma**2)*np.eye(2),size=self.N) # initialize particles
        self.weights = np.ones(self.N) / self.N # initialize weights
        self.estimate = None # current estimate
        self.estimate_var = None # current estimate variance
        self.estimation() # compute current estimate

    ''' compute PF estimate
        est_type: flag for MMSE or MAP estimate
    '''
    def estimation(self, est_type='MMSE'):
        # compute the expected state
        mean = np.average(self.particles, weights=self.weights, axis=0)
        # compute the variance of the state 
        self.estimate_var  = np.trace(np.cov(self.particles-mean, rowvar=False, aweights=self.weights) ) # np.average((self.particles - mean)**2, weights=self.weights, axis=0)
        if est_type == 'MAP':
            # compute the MAP estimate
            argmax_weight = np.argmax(self.weights)
            self.estimate = self.particles[argmax_weight]
        else:
            # compute MMSE estimate
            self.estimate = mean

    ''' prediction step
        mu_omega: mean of the process model
        sigma_omega: variance of the process model
    '''
    def predict(self, mu_omega,sigma_omega):
        # move particles according to the process model
        omega = multivariate_normal(mu_omega[:-1], (sigma_omega*2)**np.eye(2),size=self.N)
        self.particles[:, :-1] += omega

    ''' update step
        z_RF: RSSI measurement
        z_c: camera measurement
    '''
    # def update(self, z_RF, z_c):
    #     # RF likelihood
    #     l_RF = self.likelihood_RF(z_RF) if self.strategy == 0 or self.strategy == 2 else 1 
    #     # visual likelihood
    #     l_c = self.likelihood_c(z_c) if self.strategy == 1 or self.strategy == 2 else 1
    #     self.weights *= l_RF*l_c
    #     self.weights += 1.e-300      # avoid round-off to zero
    #     self.weights /= sum(self.weights) # normalize

    ''' update step
        z_RF: RSSI measurement
        p_is_detected: boolean if the agent has been detected
    '''
    def update(self, z_RF, p_is_detected):
        # RF likelihood
        l_RF = 1
        for k in range(len(z_RF)):
            l_RF *= self.likelihood_RF(z_RF[k]) if self.strategy == 0 or self.strategy == 2 else 1 
        # visual likelihood
        l_c = self.likelihood_c(p_is_detected) if self.strategy == 1 or self.strategy == 2 else 1
        self.weights *= l_RF*l_c
        self.weights += 1.e-300      # avoid round-off to zero
        self.weights /= sum(self.weights) # normalize

    ''' sample subset of elements according to a list of indices
        indexes: list of indices that guides the elements sampling
    '''
    def resample_from_index(self, indexes):
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights.fill(1.0 / len(self.weights))

    ''' SIS resampling '''
    def SIS_resampling(self):
        # resample if too few effective particles
        n_eff = 1. / np.sum(np.square(self.weights))
        if n_eff < self.N/2:
            indexes = systematic_resample(self.weights)
            self.resample_from_index(indexes)
            assert np.allclose(self.weights, 1/self.N)

    ''' RF likelihood
        z_RF: RSSI datum
    '''
    def likelihood_RF(self, z_RF):
        if not np.isnan(z_RF): # in case of collected datum
            # distances between each particle and the searching platform
            distance = np.linalg.norm(self.particles - self.uav.c.T, axis=1)
            rssi = self.uav.rx.r_0_hat - 10*self.uav.rx.n_hat*np.log10(distance/self.uav.rx.d_0)
            l_RF = scipy.stats.norm.pdf(z_RF,rssi,self.uav.rx.sigma_hat)
        else: # in case of non-collected datum
            l_RF = 1 
        return l_RF

    ''' visual likelihood
        z_c: visual detection datum
    '''
    # def likelihood_c(self, z_c):
    #     # distances between each particle and the searching platform
    #     distance = np.linalg.norm(self.particles - self.uav.c.T, axis=1)
    #     l_c = np.ones(self.N)
    #     for i in range(self.N):
    #         p_i = self.particles[i,:] # i-th particle
    #         p_i_inside_FoV = self.uav.camera.is_inside_FoV(p_i) # project onto the image plane
    #         if p_i_inside_FoV: # inside FoV
    #             # probability of having current measure, given the detection event (suppose no detection error/noise).
    #             # Suppose 1 for every z_c in image plane (so that it is not necessary to have the 3D coordinates) 
    #             p_zc_given_D = 1# scipy.stats.multivariate_normal.pdf(p_i_I[:,0], mean=p_i_I[:,0], cov=0*np.eye(2)) 
    #             d_i = distance[i]
    #             z_detector, p_D = self.uav.camera.detector(z_c,d_i)
    #             l_c[i] = p_zc_given_D*p_D 
    #         else: # out of FoV
    #             l_c[i] = 1
    #     return l_c

    ''' visual likelihood
        p_is_detected: boolean if the agent has been detected
    '''
    def likelihood_c(self, p_is_detected):
        # distances between each particle and the searching platform
        distance = np.linalg.norm(self.particles - self.uav.c.T, axis=1)
        l_c = np.ones(self.N)
        for i in range(self.N):
            p_i = self.particles[i,:] # i-th particle
            p_i_inside_FoV = self.uav.camera.is_inside_FoV(p_i) # project onto the image plane
            if not p_i_inside_FoV: # out of FoV particle
                l_c[i] = 1 if not p_is_detected else 0
            else:
                d_i = distance[i]
                p_i_is_detected, p_D = self.uav.camera.is_detected(p_i_inside_FoV,d_i)
                l_c[i]= p_D if p_is_detected else 1-p_D 
        return l_c

    ''' PF plotting tool
    '''
    def pf_plotting_tool(self, ax):
        if self.draw_particles_flag:
            self.plot_particles(ax)
        self.plot_estimate(ax)

    ''' draw particles
        ax: axis where to draw
    '''    
    def plot_particles(self,ax):
        alpha = .20
        if self.N > 5000:
            alpha *= np.sqrt(5000)/np.sqrt(self.N)  
        rgba_colors = np.zeros((self.N,4))
        rgba_colors[:,1] = 1.0
        rgba_colors[:,3] = 10 + (self.weights**3)*100
        ax.scatter3D(self.particles[:, 0], self.particles[:, 1],0, c=rgba_colors[:,:-1], s=rgba_colors[:,-1], marker='o', alpha=0.1)

    ''' draw particle filter estimate
        ax: axis where to draw
    '''
    def plot_estimate(self,ax):
        ax.scatter3D(self.estimate[0], self.estimate[1],0, color='g', marker='s')

        
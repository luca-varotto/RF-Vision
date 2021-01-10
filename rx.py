import numpy as np

'''
Define a radio receiver (Rx)
'''

class Rx:

    ''' Constructor 
        d_0_hat: estimated reference distance (often = d_0)
        r_0_hat: estimated reference RSSI
        n_hat: estimated attenuation gain
        sigma: std. dev. of the noise in the underlying generative model
        sigma_hat: estimated noise std. dev.
    '''
    def __init__(self,d_0,r_0_hat,n_hat,sigma, sigma_hat):
        # suppose underlying generative model of RSSI data
        # follows the log-distance path loss model
        self.d_0 = d_0 
        self.r_0_hat = r_0_hat 
        self.n_hat = n_hat 
        self.sigma = sigma
        self.sigma_hat = sigma_hat 

    ''' Generate (noisy) RSSI data according to the log-distance path loss model 
        r: deterministic RSSI datum
    '''
    def receive(self,r):
        r_noisy = r + np.random.normal(0.0,self.sigma)
        return r_noisy

    ''' Compute distance from incoming (noisy) RSSI data through propagation model inversion
        r_noisy: noisy RSSI datum
    '''
    def inverse_formula(self, r_noisy):
        d_hat = self.d_0*10**( (self.r_0_hat - r_noisy) / (10*self.n_hat) ) 
        return d_hat
        

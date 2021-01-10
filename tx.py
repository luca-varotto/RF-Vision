import numpy as np

'''
Define a radio transmitter (Tx)
'''

class Tx:

    ''' Constructor 
        d_0: calibration reference distance
        r_0: calibration reference RSSI
        n: attenuation gain
        T_RF: transmission rate
    '''
    def __init__(self,d_0,r_0,n,T_RF):
        # suppose the underlying generative model of RSSI data
        # follows the log-distance path loss model
        self.d_0 = d_0 
        self.r_0 = r_0 
        self.n = n 
        self.T_RF = T_RF

    ''' Generate (noisy-free deterministic) RSSI data according to the log-distance path loss model 
        d: Rx-Tx distance 
    '''
    def send(self, d):
        return self.r_0 - 10*self.n*np.log10(d/self.d_0) 



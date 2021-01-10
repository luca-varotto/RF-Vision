import numpy as np
from tqdm import tqdm

############################################################################################################################

from agent import Agent 
from uav import UAV
from tx import Tx
from rx import Rx
from camera import Cam
from controller import Controller
from pf import PF
from tools import Plotter

############################################################################################################################


def main_MC(agent_motion, user_params):

    flag_create_agent_motion = len(agent_motion) == 0

                        # ***** CREATE THE TARGET (MOVING AGENT) ***** 
    agent = Agent(user_params['p_init'])
    # associate a Tx object to the agent 
    sigma = user_params['sigma'] # (not used in Tx, but to produce noisy data in Rx) 
    agent.tx = Tx(user_params['d_0'],\
                user_params['r_0'],\
                user_params['n'], \
                user_params['T_RF'])

                        # ***** CREATE THE UAV SEARCHING PLATFORM ***** 
    uav = UAV(user_params['c_init'],\
            user_params['E_tot'], \
            user_params['scenario']) 
    # associate a Rx to the UAV. 
    # Define the ESTIMATED propagation model parameters (suppose a calibration process have already been done) 
    d_0_hat = agent.tx.d_0 # calibration reference distance is user-defined during calibration, hence it is knwon 
    # r_0, n and sigma are usually estimated from data.
    #  Suppose to apply a bayesian estimator that produces unbiased estimates with given std. dev.  
    r_0_hat = np.random.normal(agent.tx.r_0,user_params['sigma_r_0']) # estimated reference RSSI
    n_hat = np.max((1,np.random.normal(agent.tx.n,user_params['sigma_n']))) # estimated attenuation gain
    sigma_hat= np.max((1.0E-3,np.random.normal(sigma,user_params['sigma_sigma'])))
    uav.rx = Rx(d_0_hat,\
                r_0_hat,\
                n_hat, \
                sigma, \
                sigma_hat)
    # associate a camera to the UAV (with initial PTZ configuration)
    f_range = user_params['f_range']
    f_init = user_params['f_init']   # zoom (focal length)
    uav.camera = Cam(user_params['alpha_init']*np.pi/180,\
                    user_params['beta_init']*np.pi/180, \
                    f_range,\
                    f_init, 
                    uav.c, 
                    user_params['resolution_px'],
                    user_params['px2mm'],
                    user_params['T_a'],
                    user_params['T_clock'])

                        # ***** CREATE THE CONTROLLER EMBEDDED IN THE SEARCHING PLATFORM *****
    controller = Controller(uav)

                        # ***** CREATE THE RBE MAP GENERATOR *****
    # create Particle Filter object
    N_s= user_params['N_s']
    draw_particles_flag=True  
    pf = PF(uav, \
        user_params['strategy'],
        user_params['p0_mu'], \
        user_params['p0_sigma'],\
        N_s,\
        draw_particles_flag)

    print("r_0 = %4.2f, n = %4.2f, sigma = %4.2f \nr_0_hat = %4.2f, n_hat = %4.2f, sigma_hat = %4.2f \n"\
        %(agent.tx.r_0,agent.tx.n, sigma,\
        uav.rx.r_0_hat,uav.rx.n_hat, uav.rx.sigma_hat))

                        # ***** INSTANTIATE SOME BUFFERS FOR DATA STORAGE *****
    # store (noise-free) RSSI transmitted by the Tx 
    Tx_data = []
    # store noisy RSSI received by the Rx
    num_rx = 3
    Rx_data = np.empty((num_rx,user_params['sim_duration']))
    # store true distances between UAV and target
    d_true = []
    # store distances estimated by the Rx
    d_hat = np.empty((num_rx,user_params['sim_duration']))
    # store control cost functions (distance c_Pi to PF estimate)
    control_cost_functions = []
    # store estimation performance (distance PF estimate to true target position)
    est_performance = []
    # store estimate variance
    est_variance = []
    # UAV energy level
    uav_energy_level = []
    # store vision events (out of FoV, missed detection, detection) and the corresponding detection probabilities
    vision_events = np.empty((2,user_params['sim_duration'])) # 2xsim_duration matrix, first row: vision events; second row: detection probabilities

                        # ***** SIMULATION *****
    # simulation duration
    sim_duration = user_params['sim_duration']
    plotter = Plotter(agent, uav, pf)
    for t in tqdm(range(sim_duration)): 

        if t>0 and p_is_detected: # in case of detection, finish
            print("Target detected!!")
            break

        # ** TARGET MOTION **
        # move agent according to the underlying motion law
        mu_q = user_params['mu_q'] 
        sigma_q = user_params['sigma_q']
        if flag_create_agent_motion or t >= len(agent_motion):
            agent.motion(mu_q,sigma_q)
            agent_motion.append([agent.p[0][0],agent.p[1][0],agent.p[2][0]]) # save movement
        else:
            agent.p = np.array(agent_motion[t]).reshape(-1,1)
        
        # ** SENSING **
        # actual true distance between UAV and target
        d_true.append(np.linalg.norm(agent.p - uav.c))
        # RSSI
        if t % agent.tx.T_RF ==0: # case of transmission
            # transmission 
            r = agent.tx.send(d_true[-1])
            Tx_data.append(r)

            # receive noisy datum and estimate distance
            for k in range(num_rx):
                r_noisy = uav.rx.receive(r)
                Rx_data[k,t] = r_noisy
                d_hat[k,t] = uav.rx.inverse_formula(r_noisy)
        else: # case of non-transmission
            Tx_data.append(None) # Tx does not transmit anything
            for k in range(num_rx):
                Rx_data[k,t] = None # Rx does not receive anything...
                d_hat[k,t] = None # ... hence, it can not estimate the distance
        # visual
        p_inside_FoV = uav.camera.is_inside_FoV(agent.p) # project target onto camera image plane
        # p_I = uav.camera.projector(agent.p)
        p_is_detected, p_D = uav.camera.is_detected(p_inside_FoV, d_true[-1]) # simulate detection process
        # z_c,p_D = uav.camera.detector(p_I)
        vision_events[0,t] = float(p_inside_FoV) + float(p_is_detected) # 0.0 -> out of FoV, 1.0 -> missed detection, 2.0 -> detection 
        vision_events[1,t] = p_D 

        # ** RBE **
        # prediction step
        # suppose brownian motion as process model
        mu_omega = user_params['mu_omega']
        sigma_omega = user_params['sigma_omega']
        pf.predict(mu_omega,sigma_omega)
        # update particle filter
        # pf.update(Rx_data[-1], z_c)
        pf.update(Rx_data[:,t], p_is_detected)
        # resampling
        pf.SIS_resampling()
        # compute target position estimate
        pf.estimation(est_type='MAP')
        # estimation performance 
        est_performance.append(  np.linalg.norm(agent.p -  pf.estimate) )
        # estimate variance
        est_variance.append(pf.estimate_var)

        # ** CONTROL & SEARCHING PLATFORM MOTION **
        uav_c_old = uav.c # save actual UAV position
        # apply control law
        setpoint = pf.estimate # the control setpoint is the actual PF estimate
        J = controller.apply_control(setpoint,int(1/uav.camera.T_clock),user_params['K'])   
        # save cost function value at the end of the control session
        control_cost_functions.append(J)
        # compute actual UAV energy consumption and update amount of energy used
        uav.update_energy_used(np.linalg.norm(uav.c - uav_c_old))
        uav_energy_level.append( uav.E_tot - uav.E_t )

    return plotter,Tx_data, Rx_data, uav_energy_level, est_performance,control_cost_functions, vision_events, est_variance
import numpy as np
from tqdm import tqdm
from datetime import datetime 
import pickle

############################################################################################################################

from main_MC import main_MC
from agent import Agent 
from uav import UAV
from tx import Tx
from rx import Rx
from camera import Cam
from controller import Controller
from pf import PF
from tools import Plotter, DataImporter

############################################################################################################################

                        # ***** MC SETTING *****
num_MC_simulations = 20
scenario = 3
strategy = 2

# object to manage data load/saving
dataImporter = DataImporter()
count_succ = 0 # count how many success searches
init_counter = 0
for test_number in range(init_counter,init_counter+num_MC_simulations):

    name_config = "scenario0/strategy2/user_params" + str(test_number) 
    name_agent_motion = "scenario0/strategy2/agent_motion" + str(test_number)
                  
                        # ***** DATA LOADING *****
    # load pre-defined configuration
    user_params = None
    dir_path_configurations = "./data/configurations/"
    if name_config != "":
        user_params = dataImporter.load_obj(dir_path_configurations+name_config)
        if user_params['scenario'] != scenario:
            user_params['scenario'] = scenario
        if user_params['strategy'] != strategy:
            user_params['strategy'] = strategy
        print("\n *************************** \nLoaded configuration: " + name_config)
    else:
        print("\n *************************** \nNo configuration loaded")

    # load pre-defined agent motion
    dir_path_agent_motions = "./data/agent_motions/"
    if name_agent_motion != "":
        agent_motion = dataImporter.load_obj(dir_path_agent_motions+name_agent_motion)
        print("Loaded agent motion: " + name_agent_motion)
    else:
        agent_motion = [] # buffer to save successive agent motion
        print("No agent motion loaded")

                        # ***** DEFINE THE SETUP *****
    if user_params is None: 
        
        user_params = {
    
            # AGENT ...
            # ... underlying initial position
            'p_init' : np.array([ [np.random.uniform(10.0,20.0)], [np.random.uniform(10.0,20.0)] , [0.0] ]), # [m]; initial target position
            # ... parameters of the stochastic input of the underlying motion law (supposed to be biased brownian motion)
            'mu_q' : 1.0, # mean of the driving stochastic input q
            'sigma_q' : 0.2, # variance of the driving stochastic input q 
            # ... Tx, UNDERLYING (unknown) RSSI propagation model parameters  
            'd_0' : 1, # calibration reference distance
            'r_0' : -35.0, # calibration reference RSSI
            'n' : 2.0, # attenuation gain
            'sigma' : 2, # RSSI noise (std. dev.)
            'T_RF' : 10, # RSSI transmission rate

            # SEARCHING PLATFORM...
            # ... UAV
            # 'c_init' : np.array([[0.0],[0.0],[10.0]]),
            'c_init' : np.array([ [np.random.uniform(-50.0,50.0)], [np.random.uniform(-50.0,50.0)], [10.0] ]), # [m]; initial UAV position
            'E_tot' : 100, # total energy capacity 
            # ... Rx
            'sigma_r_0' : 1.0E-3, # uncertainty.T on r_0 of the propagation model
            'sigma_n' : 1.0E-3, # uncertainty on n of the propagation model
            'sigma_sigma': 1.0E-3, # uncertainty on the level of the propagation model noise
            # ... camera
            'alpha_init': 0.0, # [°]; initial camera pan angle
            'beta_init': 0.0, # [°]; initial camera tilt angle
            'f_range' : (20,100), # [mm]; range of the focal length
            'f_init': np.random.uniform(20,100), # [mm]; initial focal length
            'resolution_px' : 300, # [pixels]; resolution of the (square) camera sensor  
            'px2mm' : 0.25, # [pixel] --> [mm] conversion: 1 [px] = px2mm [mm]
            'T_a': 1, # image acquisition rate
            'T_clock': 1/200, # camera PTZ controller clock time

            # CONTROLLER
            'K' : [1*1.0E-1, 5*1.0E-4,5*1.0E-4, 1*1.0E-0], # control gains: K_c, K_alpha, K_beta, K_f

            # RBE MAP GENERATOR
            'N_s' : 300, # number of particles
            'p0_mu': [np.random.uniform(-50.0,-30.0), np.random.uniform(-50.0,-30.0), 0.0], # initial belief (pdf mean)
            'p0_sigma': 100.0, # initial belief (pdf std. dev.) 
            'mu_omega': [0.0, 0.0, 0.0], # mean of the process model
            'sigma_omega': 5, # std. dev. of the process model 

            # SCENARIO
            'scenario' : 3, # 0 -> NO ENERGY CONSTRAINT, 1 -> CAMERA CAN NOT MOVE, 2- >  DRONE CAN NOT MOVE, 3 -> FULL
            'sim_duration': 100, # simulation duration

            # SEARCH STRATEGY
            'strategy' : 1 # 0 -> RF, 1-> V, 2 -> RF+V
        }
     
    print("MC simulation %d / %d \nScenario: %d, strategy: %d " \
            %(test_number, init_counter+num_MC_simulations-1,user_params['scenario'],user_params['strategy']))

                            # ***** RUN SIMULATION *****
 
    plotter, Tx_data, Rx_data, uav_energy_level, est_performance,control_cost_functions, vision_events, est_variance = \
        main_MC(agent_motion, user_params)
    
    count_succ += 1 if len(uav_energy_level) < user_params['sim_duration'] else 0
    print("\nSuccess searches: %d\n" %(count_succ))

                            # ***** PLOT *****

    # plotter.plot_results(Tx_data, Rx_data, uav_energy_level, est_performance, control_cost_functions, vision_events)

                            # ***** SAVINGS *****

    now = datetime.now()
    date_time = now.strftime("%m%d%Y-%H%M%S")

    # save configuration 
    # define the path where to save the configuration as a list of folders 
    scenario_dir = "scenario" + str(user_params['scenario'])
    strategy_dir = "strategy" + str(user_params['strategy'])
    dir_list = [dir_path_configurations, scenario_dir, strategy_dir]
    name = "user_params" + str(test_number) # date_time
    dataImporter.save(dir_list,user_params, name)
    

    # save agent motion
    dir_list[0] = dir_path_agent_motions
    name = "agent_motion" + str(test_number) # date_time
    dataImporter.save(dir_list,agent_motion, name)

    # save results
    dir_path_results = "./results/" 
    dir_list[0] = dir_path_results
    results_dict = {"Tx_data": Tx_data, \
                    "Rx_data": Rx_data,\
                    "uav_energy_level":uav_energy_level,\
                    "est_performance":est_performance,\
                    "control_cost_functions":control_cost_functions,\
                    "vision_events":vision_events,\
                    "est_variance":est_variance}
    for item in results_dict.items():
        name = item[0] + str(test_number) # date_time # item[0]: key
        dataImporter.save(dir_list,item[1], name) # item[1]: value

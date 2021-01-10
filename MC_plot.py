import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm, gamma

############################################################################################################################

from tools import Plotter, DataImporter

############################################################################################################################

# Analyse results from a Monte Carlo experiment

scenarios2plot = [0,1,2] # number of scenarios compared during the MC experiment
strategies2plot = [2] # number of strategies compared during the MC experiment
num_MC_simulations = 150 # number of simulations of the MC experiment
sim_duration = 100 # duration of each simulation
E_tot = 300 # initial UAV energy

dict_labels_strategy = {0: 'RF',
                        1: 'V',
                        2: 'RF+V',
                        3:'2RF',
                        4:'3RF',
                        5:'5RF',
                        6:'10RF'}

col_strategy = {0:'b',
                1:'y',
                2:'g',
                3:'k',
                4:'c',
                5:[0.5,0.5,0.5],
                6:'m'}

dataImporter = DataImporter()
# dir_path_results = "./results/"
test_type = "case4"
dir_path_results = "./history/" +  test_type +"/results/"
results_dict = {"Tx_data":None, \
                "Rx_data":None,\
                "uav_energy_level":None,\
                "est_performance":None,\
                "control_cost_functions":None,\
                "vision_events":None,\
                'est_variance':None}

# keep trace of successful searches for each scenario and strategy
successful_searches = np.zeros((max(scenarios2plot)+1,max(strategies2plot)+1))
# keep trace of the time of search for each test
times_of_searches = np.zeros((max(scenarios2plot)+1,max(strategies2plot)+1,num_MC_simulations))
# keep trace of the energy used for each test
UAV_energy = np.zeros((max(scenarios2plot)+1,max(strategies2plot)+1,num_MC_simulations))
# keep trace of the estimation errors over the simulations
est_performance= np.zeros((max(scenarios2plot)+1,max(strategies2plot)+1,num_MC_simulations))
# keep trace of the estimation variance over the simulations
est_variance= np.zeros((max(scenarios2plot)+1,max(strategies2plot)+1,num_MC_simulations))
for scenario in scenarios2plot:
    for strategy in strategies2plot:
        for test_number in range(num_MC_simulations):
            for key in results_dict.keys():
                name = key + str(test_number)
                results_dict[key] = dataImporter.load_obj(dir_path_results+ \
                                                    "scenario" + str(scenario) + "/" +\
                                                    "strategy" + str(strategy) + "/" +\
                                                    name)
            est_performance[scenario][strategy][test_number] = np.sum(results_dict["est_performance"])
            est_variance[scenario][strategy][test_number] = np.mean(results_dict["est_variance"])
            time_of_search = int(len(results_dict["uav_energy_level"]))
            times_of_searches[scenario,strategy, test_number] = time_of_search if time_of_search < sim_duration else None 
            UAV_energy[scenario,strategy,test_number] = results_dict["uav_energy_level"][-1]
            successful_searches[scenario,strategy] += 1 if results_dict["vision_events"][0,time_of_search-1] == 2 else 0 

# print strategies instants of failure
# for scenario in scenarios2plot:
#         for strategy in strategies2plot:
#                 failures = [t for t in range(num_MC_simulations) if np.isnan(times_of_searches[scenario,strategy,t]) ]
#                 print("Failures of scenario %d, strategy %d" %(scenario, strategy), failures)

# show ECDF of time searches
# and succuess rates
linestyles = ['-',':','-.'] #['--',None,None,'-']
label_complement = ['',': $\mathbf{G}_{\mathbf{\Psi}}=\mathbf{0}$',': $\mathbf{G}_{\mathbf{c}}=\mathbf{0}$'] # ['',None,None,' (Full)']
fig =plt.figure(figsize=(9,6))
for scenario in scenarios2plot:
    for strategy in strategies2plot:
        # success rate
        print("Success rate " + dict_labels_strategy[strategy]+": %4.1f%%" %(100*successful_searches[scenario, strategy]/num_MC_simulations))
        # time search ecdf
        ecdf = ECDF(times_of_searches[scenario,strategy,:])
        plt.plot(ecdf.x, ecdf.y, \
                label=dict_labels_strategy[strategy], \
                c=col_strategy[strategy],linestyle=linestyles[scenario],
                linewidth=2)
        # plt.legend(loc='upper left', ncol=2,fontsize=25,fancybox=True, framealpha=0.5)
        delta_x = -35 if scenario > 0 else -20
        if scenario == 0:
                delta_y = 0
        elif scenario == 1:
                delta_y=0.01
        else:
                delta_y= -0.1
        plt.text(max(ecdf.x)+delta_x, ecdf.y[np.argmax(ecdf.x)]+delta_y,\
                dict_labels_strategy[strategy]+label_complement[scenario],fontsize=25)
        plt.xlabel(r'$t$',fontsize=30)
        plt.ylabel('$p(t_D \leq t)$',fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.xlim([0,sim_duration])
        plt.ylim([0,1.01])
        ax = plt.gca()
        ax.patch.set_edgecolor('black')  
        ax.patch.set_linewidth('2')
        ax.grid(ls = ':', lw = 0.5)
        plt.tight_layout()
        plt.savefig("ECDFscenario"+ str(scenario)+test_type+".png")
plt.show()

# compute resilience score
for scenario in scenarios2plot:
    for strategy in strategies2plot:
        resilience = 0
        for test_number in range(num_MC_simulations): 
                time_of_search = times_of_searches[scenario,strategy, test_number]
                if time_of_search < sim_duration and UAV_energy[scenario,strategy,test_number] <= 0 :
                        resilience += 1/  (time_of_search /sim_duration )  
        print("Resilience " + dict_labels_strategy[strategy]+ " %5.3f" %(resilience/num_MC_simulations))

# show estimation performance
# fig =plt.figure(figsize=(15,6))
# for scenario in scenarios2plot:
#         for strategy in strategies2plot:
#                 ax = plt.gca()
#                 bias_mean = est_performance[scenario,strategy,:]
#                 shape, loc, scale = gamma.fit(bias_mean, floc=0)
#                 x_all = np.arange(0, 3000, 0.1)
#                 plt.plot(x_all,gamma.pdf(x_all, shape, loc, scale), label=dict_labels_strategy[strategy], c=col_strategy[strategy])
#                 ax.fill_between(x_all,gamma.pdf(x_all, shape, loc, scale),0, alpha=0.3, color=col_strategy[strategy])
#                 plt.vlines(shape*scale,0,gamma.pdf(shape*scale, shape, loc, scale), colors=col_strategy[strategy], linestyles='--')
#                 plt.xlabel(r'$bias$',fontsize=30)
#                 plt.ylabel(r'$p(bias)$',fontsize=30)
#                 plt.xticks(fontsize=25)
#                 plt.yticks(fontsize=25)
#                 plt.legend(fontsize=25,fancybox=True, framealpha=0.5)
# # plt.savefig("../Report/images/UAVEnergy"+ str(scenario)+test_type+".png")
# plt.show()
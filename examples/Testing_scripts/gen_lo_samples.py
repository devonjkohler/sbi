
from bioscrape.types import Model
from bioscrape.simulator import SSASimulator, ModelCSimInterface
import pickle
import numpy as np
import pandas as pd
import torch
import sbi.utils as utils
import multiprocessing as mp

## Define sim
def lo_simulation(i, rates):

    sbml_model = Model(sbml_filename=r"/home/kohler.d/code_output/biosim_project/sbi/examples/LacOperon_stochastic.xml", sbml_warnings=False)
    # sbml_model = Model(sbml_filename=r"../LacOperon_stochastic.xml", sbml_warnings=False)

    ## Update rates based on sample from prior
    rate_names = [
        'LacPermease_vmax',
        'LacPermease_Kd',
    ]
    sbml_model.set_params(dict(zip(rate_names, rates)))

    timepoints = np.arange(0, 20000, 20.)

    #Create an Interface Model --> Cython
    interface = ModelCSimInterface(sbml_model)

    #Create a Simulator
    simulator = SSASimulator()

    ## Run simulation
    output = simulator.py_simulate(interface, timepoints)
    output = output.py_get_dataframe(Model=sbml_model)
    # output = py_simulate_model(timepoints, Model=sbml_model, stochastic = True)

    sim_path = torch.tensor(np.array(output))
    #time = torch.tensor(timepoints).reshape(len(timepoints), 1)
    #sim_path = torch.cat((species_values, time), 1)

    # trace = torch.flatten(sim_path)

    return (i, sim_path.T)

def main():
    pool = mp.Pool(mp.cpu_count())
    obs_list = list()
    def collect_result(result):
        obs_list.append(result)

    prior = utils.BoxUniform(
        low=torch.tensor([20.,  120000.]),
        high=torch.tensor([40.,  180000.]),)
    print("entering for loop")
    labels = list()
    for i in range(1000):
        prior_sample = prior.sample()
        labels.append(prior_sample)
        pool.apply_async(lo_simulation, args=(i, prior_sample), callback=collect_result)

    pool.close()
    pool.join()
    print("obs Gen")
    obs_list.sort(key=lambda x: x[0])
    obs_list_final = [r for i, r in obs_list]
    print("obs sorted")
    x = torch.stack(obs_list_final,axis=0)
    y = torch.stack(labels,axis=0)

    ## Save observations
    print("trying to save cnn obs")
    with open(r'/scratch/kohler.d/code_output/biosim/lac_operon_obs.pickle', 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("trying to save cnn labels")
    with open(r'/scratch/kohler.d/code_output/biosim/lac_operon_labels.pickle', 'wb') as handle:
        pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("saved")

if __name__ == '__main__':
    main()
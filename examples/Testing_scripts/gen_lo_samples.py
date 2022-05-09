
from bioscrape.types import Model
from bioscrape.simulator import SSASimulator, ModelCSimInterface
import pickle
import numpy as np
import pandas as pd
import torch
import sbi.utils as utils
from sbi.inference.base import simulate
from joblib import Parallel

## Define sim
def lo_simulation(rates):

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

    return sim_path.T

def main():

    prior = utils.BoxUniform(
        low=torch.tensor([20.,  120000.]),
        high=torch.tensor([40.,  180000.]),)
    print("entering for loop")

    y, x = simulate(
        lo_simulation,
        prior,
        num_workers=-1,
        num_simulations=750
    )

    print("obs Gen")
    # x = torch.stack(obs_list_final,axis=0)
    # y = torch.stack(labels,axis=0)
    print(x)
    print(y)
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
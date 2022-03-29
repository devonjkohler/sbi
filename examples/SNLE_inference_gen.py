import time

import sbi.utils as utils
from sbi.inference.base import infer
import numpy as np
import torch
from bioscrape.types import Model
from bioscrape.simulator import SSASimulator, ModelCSimInterface
import pickle
import itertools

## Lac Operon
def lo_simulation(rates):

    sbml_model = Model(sbml_filename=r"/home/kohler.d/biosim_project/sbi/examples/LacOperon_stochastic.xml", sbml_warnings=False)

    ## Update rates based on sample from prior
    rate_names = [
        'LacPermease_vmax',
        'LacPermease_Kd',
    ]
    sbml_model.set_params(dict(zip(rate_names, rates)))

    timepoints = np.arange(0, 20000, 10.)

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

    trace = torch.flatten(sim_path)

    return trace[12:]

class InferPosterior:

    def __init__(self, init_params):
        self.posterior_samples = None
        self.posterior = None
        self.prior = init_params[0]
        self.nn_sims = init_params[1]
        self.inference_method = init_params[2]
        self.observations = init_params[3]

    def train(self):

        posterior = infer(
            lo_simulation,
            self.prior,
            method=self.inference_method,
            num_workers=-1,
            num_simulations=self.nn_sims
        )

        self.posterior = posterior

    def sample_posterior(self):

        samples = self.posterior.sample((300,), x=self.observations)

        self.posterior_samples = samples


def main():

    ## Define Prior
    prior_params = [utils.NormalPrior(
        torch.tensor([35.8, 156576.]),
        torch.tensor([2., 800.])
    )]

    # Num sims to use to train nn
    n_sims = [500]

    ## Type of inference to use - one of "SNPE", "SNLE", "SNRE"
    inference_methods = ["SNLE"]

    ## Num obs to use to learn posterior
    n_obs = [1,5,20]
    obs = list()
    temp_obs = torch.tensor(np.array([lo_simulation([35.8, 156576.]).numpy() for _ in range(20)]))
    ## Generate observations
    for i in n_obs:
        obs.append(temp_obs[:i])

    ## just create a list of tuples of all parameters combos
    parameters = list(itertools.product(*[prior_params, n_sims, inference_methods, obs]))

    for i in range(len(parameters)):

        start = time.time()

        ## Run inference
        print("starting inference")
        inference_run = InferPosterior(parameters[i])
        inference_run.train()
        print("inference finished")
        inference_run.sample_posterior()
        print("posterior_sampled")

        end = time.time()
        run_time = end - start

        ## What parameters to save
        save_params = dict(
            number_sims=parameters[i][1],
            method=parameters[i][2],
            number_obs=len(parameters[i][3]))

        ## Save all needed data
        results_file = dict(
            parameters=save_params,
            run_time=run_time,
            samples=inference_run.posterior_samples)

        print("trying to save")
        with open(r'/scratch/kohler.d/code_output/biosim/sim_results_SNLE_500_{0}.pickle'.format(str(i)), 'wb') as handle:
            pickle.dump(results_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("saved")
if __name__ == '__main__':
    main()
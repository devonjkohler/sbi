import pickle5
import pickle
import numpy as np

import torch
from torch.nn import Conv2d, ReLU, Linear, Sequential, MaxPool2d, AvgPool2d, Dropout, Module

from bioscrape.types import Model
from bioscrape.simulator import SSASimulator, ModelCSimInterface

import sbi.utils as utils
from sbi.inference.base import infer

def lo_simulation(rates):

    sbml_model = Model(sbml_filename=r"/home/kohler.d/code_output/biosim_project/sbi/examples/LacOperon_stochastic.xml", sbml_warnings=False)
    # sbml_model = Model(sbml_filename=r"LacOperon_stochastic.xml", sbml_warnings=False)

    ## Update rates based on sample from prior
    rate_names = [
        'LacPermease_vmax',
        'LacPermease_Kd',
    ]
    sbml_model.set_params(dict(zip(rate_names, rates)))

    timepoints = np.arange(0, 20000, 20.)

    # Create an Interface Model --> Cython
    interface = ModelCSimInterface(sbml_model)

    # Create a Simulator
    simulator = SSASimulator()

    ## Run simulation
    output = simulator.py_simulate(interface, timepoints)
    output = output.py_get_dataframe(Model=sbml_model)
    # output = py_simulate_model(timepoints, Model=sbml_model, stochastic = True)

    sim_path = torch.tensor(np.array(output))
    # time = torch.tensor(timepoints).reshape(len(timepoints), 1)
    # sim_path = torch.cat((species_values, time), 1)

    # trace = torch.flatten(sim_path)

    nn_input = sim_path.T
    nn_input = nn_input[[x for x in range(12) if x  != 9] ,:]
    for i in range(11):
        col_min = min_list[i]
        col_max = max_list[i]
        nn_input[:, i] = (nn_input[:, i] - col_min) / (col_max - col_min)

    nn_input = nn_input.reshape(1 ,1 ,11 ,1000).float()
    summary = model(nn_input)

    return summary

class Net(Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(

            # Defining a 2D convolution layer
            Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            MaxPool2d((1, 5)),
            Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            MaxPool2d((1, 5)),
            Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            MaxPool2d((1, 5)),
            Conv2d(128, 128, kernel_size=(3, 3), padding=(1, 0)),
            ReLU(inplace=True),
            AvgPool2d(kernel_size=(1, 5))
        )

        self.linear_layers = Sequential(
            Linear(1408, 720)
        )

        self.middle_linear_layers = Sequential(
            Linear(720, 720)
        )

        self.relu = Sequential(
            ReLU(inplace=True),
            Dropout(p=.15)
        )

        self.output = Sequential(
            Linear(720, 2),
        )

    # Defining the forward pass
    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.relu(x)
        # x = x.view(-1)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        x = self.relu(x)
        x = self.middle_linear_layers(x)
        x = self.relu(x)
        x = self.middle_linear_layers(x)
        x = self.relu(x)
        x = self.middle_linear_layers(x)
        x = self.relu(x)

        x = self.output(x)

        return x

def main():

    ## Calculate summary statistics
    with open(r'/scratch/kohler.d/code_output/biosim/lac_operon_obs.pickle', 'rb') as handle:
        x = pickle5.load(handle)

    x = x.reshape(6000, 12, 1000)
    x = x[:, [x for x in range(12) if x != 9], :]

    global min_list
    global max_list

    min_list = list()
    max_list = list()
    for i in range(11):
        col_min = x[:, i].min() - 1
        min_list.append(col_min)
        col_max = x[:, i].max()
        max_list.append(col_max)

    ## Load Model
    global model
    model = Net()
    state = torch.load(r'/scratch/kohler.d/code_output/biosim/cnn_model_lo.pth')
    model.load_state_dict(state['state_dict'])

    sbi_prior = utils.NormalPrior(
        torch.tensor([35.8, 156576]),
        torch.tensor([5., 5000])
    )

    num_sim = 300
    method = 'SNLE'  # SNPE or SNLE or SNRE
    posterior = infer(
        lo_simulation,
        sbi_prior,
        # See glossary for explanation of methods.
        #    SNRE newer than SNLE newer than SNPE.
        method=method,
        num_workers=-1,
        num_simulations=num_sim
    )

    with open(r'/scratch/kohler.d/code_output/biosim/snle_model_lo.pickle', 'wb') as handle:
        pickle.dump(posterior, handle, protocol=pickle.HIGHEST_PROTOCOL)

    obs = torch.tensor([lo_simulation([35.8, 156576.]).numpy() for _ in range(1)])
    samples = posterior.sample((200,), x=obs)

    with open(r'/scratch/kohler.d/code_output/biosim/one_obs_samples_snle_lo.pickle', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    obs = torch.tensor([lo_simulation([35.8, 156576.]).numpy() for _ in range(10)])
    samples = posterior.sample((200,), x=obs)

    with open(r'/scratch/kohler.d/code_output/biosim/ten_obs_samples_snle_lo.pickle', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    obs = torch.tensor([lo_simulation([34., 156576.]).numpy() for _ in range(10)])
    samples = posterior.sample((200,), x=obs)

    with open(r'/scratch/kohler.d/code_output/biosim/ten_obs_new_post_samples_snle_lo.pickle', 'wb') as handle:
        pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
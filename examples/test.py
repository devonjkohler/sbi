import time

import sbi.utils as utils
from sbi.inference.base import infer
import numpy as np
import torch
# from bioscrape_cobra.simulate import simulate_bioscrape
from bioscrape.sbmlutil import import_sbml
from vivarium_bioscrape.processes.bioscrape import Bioscrape
from vivarium.core.composition import simulate_process
from bioscrape.types import Model
from bioscrape.simulator import py_simulate_model, SSASimulator, ModelCSimInterface


import random
import matplotlib.pyplot as plt

# def gillespie_simulator(propose_rates):
#     t = 0.
#     stop_time = 5000.
#     s = torch.tensor([20., 40.])
#     path = np.insert(s, 0, t, axis=0).reshape(1, 3)
#
#     rate_functions = [lambda s: propose_rates[0] * s[0],
#                       lambda s: propose_rates[1] * s[1] * s[0],
#                       lambda s: propose_rates[2] * s[1]]
#     n_func = len(rate_functions)
#
#     transition_matrix = torch.tensor([[1, 0], [-1, 1], [0, -1]])
#
#     while True:
#
#         sampling_weights = [f(s) for f in rate_functions]
#         total_weight = sum(sampling_weights)
#
#         probs = np.array([weight / total_weight for weight in sampling_weights])
#         sample = np.random.choice(n_func, p=probs)
#         t += np.random.exponential(1.0 / total_weight)
#
#         if t >= stop_time:
#             break
#
#         s = s + transition_matrix[sample]
#         s = torch.normal(s, .25)
#         s[0] = max(1, s[0])
#         s[1] = max(1, s[1])
#
#         path = torch.cat((path, np.insert(s, 0, t, axis=0).reshape(1, 3)), axis=0)
#
#     path = torch.cat((path, np.insert(s, 0, stop_time, axis = 0).reshape(1, 3)), axis=0)
#     path = torch.flatten(path)
#
#     # Mask values onto tensor so that all samples are of equal length
#     mask_len = 7500 - len(path)
#     mask = torch.tensor([-1 for _ in range(mask_len)])
#     path = torch.cat((path, mask), 0)
#
#     return path[3:]
#     # return np.insert(s, 0, t, axis = 0).reshape(1, 3)
#
# prior = utils.BoxUniform(
#     low=torch.tensor([0.0005, 0.0001, 0.0005]),
#     high=torch.tensor([0.003, 0.0005, 0.005])
# )
#
# prior = utils.LogNormalPrior(
#     torch.tensor([np.log(.0015),
#                   np.log(.0002),
#                   np.log(.003)]), .25)
#
# num_sim = 200
# method = 'SNRE' #SNPE or SNLE or SNRE
# posterior = infer(
#     gillespie_simulator,
#     prior,
#     # See glossary for explanation of methods.
#     #    SNRE newer than SNLE newer than SNPE.
#     method='SNLE',
#     num_workers=-1,
#     num_simulations=num_sim
# )
#
# ## Generate some observations
# obs = torch.tensor([gillespie_simulator([0.001, 0.0002, 0.003]).numpy() for _ in range(10)])
#
# plot_df = np.array([i for i in obs[0][:-1] if i > 0.])
# plot_df = plot_df.reshape((int(len(plot_df)/3),3))
# plot_df1 = np.array([i for i in obs[1][:-1] if i > 0.])
# plot_df1 = plot_df1.reshape((int(len(plot_df1)/3),3))
# plot_df2 = np.array([i for i in obs[2][:-1] if i > 0.])
# plot_df2 = plot_df2.reshape((int(len(plot_df2)/3),3))
#
# fig, ax = plt.subplots(1, 3, figsize=(12,10))
# ax[0].plot(plot_df[:, 0], plot_df[:, 1], color = "blue")
# ax[0].plot(plot_df[:, 0], plot_df[:, 2], color = "red")
# ax[1].plot(plot_df1[:, 0], plot_df1[:, 1], color = "blue")
# ax[1].plot(plot_df1[:, 0], plot_df1[:, 2], color = "red")
# ax[2].plot(plot_df2[:, 0], plot_df2[:, 1], color = "blue")
# ax[2].plot(plot_df2[:, 0], plot_df2[:, 2], color = "red")
# plt.show()
#
# ## Sample posterior
# samples = posterior.sample((200,), x=obs)
#
# fig, ax = plt.subplots(1, 3, figsize=(10,6))
# ax[0].hist(samples[:, 0].numpy(), bins=15)
# ax[1].hist(samples[:, 1].numpy(), bins=15)
# ax[2].hist(samples[:, 2].numpy(), bins=15)
# ax[0].axvline(x = .001, color = "red")
# ax[1].axvline(x = .0002, color = "red")
# ax[2].axvline(x = .003, color = "red")
#
# ax[0].set_title("r1 samples")
# ax[1].set_title("r2 samples")
# ax[2].set_title("r3 samples")
#
# plt.suptitle("fixed priors, lots of sims")
# plt.show()

## Lac Operon
def lo_simulation(rates):

    sbml_model = Model(sbml_filename=r"examples/LacOperon_stochastic.xml", sbml_warnings=False)
    # sbml_model = Model(sbml_filename=r"LacOperon_stochastic.xml", sbml_warnings=False)

    ## Update rates based on sample from prior
    rate_names = [
        'GluPermease_Kd__',
        'LacPermease_Kd__',
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

start = time.time()
lo_simulation([9033., 8800000.])
end = time.time()
end-start
#173
#~60

# Only uniform prior implemented into sbi package currently
# Easy to add more manually, for example log normal
prior = utils.BoxUniform(
    low=torch.tensor([9000.,  8799900.]),
    high=torch.tensor([9050.,  8800100.]),)

# Train model
num_sim = 200
method = 'SNRE' #SNPE or SNLE or SNRE
posterior = infer(
    lo_simulation,
    prior,
    method=method,
    num_workers=-1,
    num_simulations=num_sim
)


# Sample some observations with known ground truth
obs = torch.tensor(np.array([lo_simulation([9033., 8800000.]).numpy() for _ in range(10)]))
obs0 = lo_simulation([15000., 8800000.])
obs1 = lo_simulation([9033., 7000000.])
obs = [obs0, obs1]

plot_df = obs[0].reshape((int(len(obs[0])/12),12))
plot_df1 = obs[1].reshape((int(len(obs[1])/12),12))
# plot_df2 = obs[2].reshape((int(len(obs[2])/12),12))

fig, ax = plt.subplots(1,2,figsize=(12,10))
ax[0].plot(plot_df[:, 11], plot_df[:, 1], color = "blue")
ax[0].plot(plot_df[:, 11], plot_df[:, 2], color = "red")
ax[0].plot(plot_df[:, 11], plot_df[:, 3], color = "blue")
ax[0].plot(plot_df[:, 11], plot_df[:, 4], color = "red")
ax[0].plot(plot_df[:, 11], plot_df[:, 5], color = "blue")
ax[0].plot(plot_df[:, 11], plot_df[:, 6], color = "red")
ax[0].plot(plot_df[:, 11], plot_df[:, 7], color = "blue")
ax[0].plot(plot_df[:, 11], plot_df[:, 8], color = "red")
ax[0].plot(plot_df[:, 11], plot_df[:, 9], color = "red")
ax[0].plot(plot_df[:, 11], plot_df[:, 10], color = "red")
ax[0].plot(plot_df[:, 11], plot_df[:, 0], color = "red")

ax[1].plot(plot_df1[:, 11], plot_df1[:, 1], color = "blue")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 2], color = "red")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 3], color = "blue")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 4], color = "red")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 5], color = "blue")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 6], color = "red")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 7], color = "blue")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 8], color = "red")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 9], color = "red")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 10], color = "red")
ax[1].plot(plot_df1[:, 11], plot_df1[:, 0], color = "red")

plt.show()

samples = posterior.sample((200,), x=obs) # SNPE only works with one obs, SNLE will work with more

# plot results
fig, ax = plt.subplots(1, 2, figsize=(10,6))
ax[0].hist(samples[:, 0].numpy(), bins=15)
ax[1].hist(samples[:, 1].numpy(), bins=15)
ax[0].axvline(x = 9033., color = "red")
ax[1].axvline(x = 8800000., color = "red")

ax[0].set_title("GluPermease_Kd samples")
ax[1].set_title("LacPermease_Kd samples")

plt.suptitle("Lac Operon")
plt.show()

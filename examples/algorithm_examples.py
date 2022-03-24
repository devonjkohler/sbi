
# Imports
import sbi.utils as utils
from sbi.inference.base import infer
import numpy as np
import torch

import matplotlib.pyplot as plt

# Define simulator
def gillespie_simulator(propose_rates):

    t = 0.
    stop_time = 5000. # Length of sim
    s = torch.tensor([20., 40.]) # Starting points
    path = np.insert(s, 0, t, axis=0).reshape(1, 3)

    ## Hazard functions
    rate_functions = [lambda s: propose_rates[0] * s[0],
                      lambda s: propose_rates[1] * s[1] * s[0],
                      lambda s: propose_rates[2] * s[1]]
    n_func = len(rate_functions)

    transition_matrix = torch.tensor([[1, 0], [-1, 1], [0, -1]])

    # Run sim until time limit is reached
    while True:

        sampling_weights = [f(s) for f in rate_functions]
        total_weight = sum(sampling_weights)

        # Sample a rate
        probs = np.array([weight / total_weight for weight in sampling_weights])
        sample = np.random.choice(n_func, p=probs)
        t += np.random.exponential(1.0 / total_weight)

        if t >= stop_time:
            break

        # Update species
        s = s + transition_matrix[sample]

        # Add some noise to outputs
        s = torch.normal(s, .25)
        s[0] = max(1, s[0])
        s[1] = max(1, s[1])

        path = torch.cat((path, np.insert(s, 0, t, axis=0).reshape(1, 3)), axis=0)

    path = torch.cat((path, np.insert(s, 0, stop_time, axis = 0).reshape(1, 3)), axis=0)
    path = torch.flatten(path)

    # Mask values onto long tensor so that all samples are of equal length
    mask_len = 7500 - len(path)
    mask = torch.tensor([-1 for _ in range(mask_len)])
    path = torch.cat((path, mask), 0)

    return path[3:]
    # return np.insert(s, 0, t, axis = 0).reshape(1, 3)

# Choose a prior
prior = utils.BoxUniform(
    low=torch.tensor([0.00001, 0.00001, 0.00001]),
    high=torch.tensor([0.01, 0.01, 0.01])
)

# prior = utils.LogNormalPrior(
#     torch.tensor([np.log(.0015),
#                   np.log(.0002),
#                   np.log(.003)]), .25)

# Run inference
num_sim = 200
method = 'SNLE' #SNPE or SNLE or SNRE
posterior = infer(
    gillespie_simulator,
    prior,
    method=method,
    num_workers=-1,
    num_simulations=num_sim
)

obs = torch.tensor([gillespie_simulator([0.001, 0.0002, 0.003]).numpy() for _ in range(10)])

plot_df = np.array([i for i in obs[0][:-1] if i > 0.])
plot_df = plot_df.reshape((int(len(plot_df)/3),3))
plot_df1 = np.array([i for i in obs[1][:-1] if i > 0.])
plot_df1 = plot_df1.reshape((int(len(plot_df1)/3),3))
plot_df2 = np.array([i for i in obs[2][:-1] if i > 0.])
plot_df2 = plot_df2.reshape((int(len(plot_df2)/3),3))

fig, ax = plt.subplots(1, 3, figsize=(12,10))
ax[0].plot(plot_df[:, 0], plot_df[:, 1], color = "blue")
ax[0].plot(plot_df[:, 0], plot_df[:, 2], color = "red")
ax[1].plot(plot_df1[:, 0], plot_df1[:, 1], color = "blue")
ax[1].plot(plot_df1[:, 0], plot_df1[:, 2], color = "red")
ax[2].plot(plot_df2[:, 0], plot_df2[:, 1], color = "blue")
ax[2].plot(plot_df2[:, 0], plot_df2[:, 2], color = "red")
plt.show()

# Sample posterior
samples = posterior.sample((200,), x=obs[0])

fig, ax = plt.subplots(1, 3, figsize=(10,6))
ax[0].hist(samples[:, 0].numpy(), bins=15)
ax[1].hist(samples[:, 1].numpy(), bins=15)
ax[2].hist(samples[:, 2].numpy(), bins=15)
ax[0].axvline(x = .001, color = "red")
ax[1].axvline(x = .0002, color = "red")
ax[2].axvline(x = .003, color = "red")

ax[0].set_title("r1 samples")
ax[1].set_title("r2 samples")
ax[2].set_title("r3 samples")

plt.suptitle("SNLE")
plt.show()

## Try out SNPE
num_sim = 500
method = 'SNPE' #SNPE or SNLE or SNRE
posterior = infer(
    gillespie_simulator,
    prior,
    method=method,
    num_workers=-1,
    num_simulations=num_sim
)

all_samples = list()
for i in range(10):
    samples = posterior.sample((10000,), x=obs[i])
    all_samples.append(samples)

all_samples = torch.cat(all_samples)

fig, ax = plt.subplots(1, 3, figsize=(10,6))
ax[0].hist(all_samples[:, 0].numpy(), bins=15)
ax[1].hist(all_samples[:, 1].numpy(), bins=15)
ax[2].hist(all_samples[:, 2].numpy(), bins=15)
ax[0].axvline(x = .001, color = "red")
ax[1].axvline(x = .0002, color = "red")
ax[2].axvline(x = .003, color = "red")

ax[0].set_title("r1 samples")
ax[1].set_title("r2 samples")
ax[2].set_title("r3 samples")

plt.suptitle("SNPE_all")
plt.show()

## SNLE with more obs
num_sim = 200
method = 'SNLE' #SNPE or SNLE or SNRE
posterior = infer(
    gillespie_simulator,
    prior,
    method=method,
    num_workers=-1,
    num_simulations=num_sim
)

samples = posterior.sample((200,), x=obs)

fig, ax = plt.subplots(1, 3, figsize=(10,6))
ax[0].hist(samples[:, 0].numpy(), bins=15)
ax[1].hist(samples[:, 1].numpy(), bins=15)
ax[2].hist(samples[:, 2].numpy(), bins=15)
ax[0].axvline(x = .001, color = "red")
ax[1].axvline(x = .0002, color = "red")
ax[2].axvline(x = .003, color = "red")

ax[0].set_title("r1 samples")
ax[1].set_title("r2 samples")
ax[2].set_title("r3 samples")

plt.suptitle("SNLE with 10 obs")
plt.show()

import numpy as np
import os
import sys
import torch

sys.path.append("../")
sys.path.append("../../")
import problem_data as pbd

# Dummy class to avoid changing any code in the training loop
class UtilsWrapper:
    @staticmethod
    def build_kan(*args, **kwargs):
        from src import models
        return models.build_kan(*args, **kwargs)

    @staticmethod
    def train_adam(*args, **kwargs):
        from src import train
        return train.train_adam(*args, **kwargs)

utils = UtilsWrapper()


ITERATIONS = 10000
LR = 0.001
REPETITIONS = 10 # Number of training runs for each hyperparameter configuration

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collocation points
NUM_POINTS_X = 100
NUM_POINTS_T = 100

x = torch.linspace(pbd.INTERVAL_X[0], pbd.INTERVAL_X[1], NUM_POINTS_X, requires_grad=True)
t = torch.linspace(pbd.INTERVAL_T[0], pbd.INTERVAL_T[1], NUM_POINTS_T, requires_grad=True)

x, t = torch.meshgrid(x, t, indexing='ij')

x = x.reshape(-1, 1).to(device)
t = t.reshape(-1, 1).to(device)


# Hyperparameter sweep
for hpt_config, (n_hidden, hidden_width) in enumerate([(1,23), (2,7), (3,5)]):
    # Create directory for model
    G = 5
    k = 3
    dir = f"./hpt_results/PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/"
    os.makedirs(dir, exist_ok=True)

    for rep in range(REPETITIONS):
        print(f"Hyperparameter configuration: {hpt_config+1} | Repetition: {rep+1}", flush=True)

        # KAN
        model = utils.build_kan(pbd.NUM_INPUTS, pbd.NUM_OUTPUTS, n_hidden, hidden_width, G, k)
        model.to(device)
        model.train(True)
        model.speed() # Deactivate symbolic branch

        # Physics-informed initialization if supported by the model
        if hasattr(model, 'physics_informed_init'):
            t_zero_mask = (t == 0.0).flatten()
            x_init = x[t_zero_mask]
            t_init = t[t_zero_mask]
            xt_init = torch.cat([x_init, t_init], dim=-1)
            y_init = pbd.gaussian_ic(x_init, sigma_g=0.1)
            model.physics_informed_init(xt_init, y_init)

        # Train
        total_losses, physics_losses, initial_conds_losses = utils.train_adam(model, [x, t], pbd.losses, ITERATIONS, LR, outdir=dir + f"model{rep+1}.pkl")

        # Save losses
        np.savez(
            file=dir + f"losses{rep+1}.npz",
            physics=np.array(physics_losses),
            initial_conds=np.array(initial_conds_losses),
            total=np.array(total_losses)
        )
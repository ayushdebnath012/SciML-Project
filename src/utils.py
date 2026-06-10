import itertools
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import pickle as pkl
import torch
from typing import Callable

# Relative package imports
from .models import build_mlp, build_kan, num_parameters_mlp, num_parameters_kan
from .physics import dy_dx

# For LaTeX in plots
plt.rcParams.update({'font.size': 18, 'font.family': 'serif'})
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{lmodern}'


# --------------------
# METRICS
# --------------------

def summarize_results(network_type, n_inputs, n_outputs, n_hidden_list, hidden_width_list, x_eval, ref_sol, ref_grad, repetitions, dir, G=None, k=None):
    assert network_type in ['PINN', 'PIKAN']
    assert network_type == 'PINN' or (G is not None and k is not None)

    summary = {}
    for n_hidden, hidden_width in zip(n_hidden_list, hidden_width_list):
        if network_type == 'PINN':
            model = build_mlp(n_inputs, n_outputs, n_hidden, hidden_width)
            subdir = dir + f"PINN/pinn_hidden{n_hidden}_width{hidden_width}/"
        else:
            model = build_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k)
            subdir = dir + f"PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/"

        summary[(n_hidden, hidden_width)] = compute_error_stats(model, x_eval, ref_sol, ref_grad, subdir, repetitions)

        if network_type == 'PINN':
            summary[(n_hidden, hidden_width)]["n_params"] = num_parameters_mlp(n_inputs, n_outputs, n_hidden, hidden_width)
        else:
            summary[(n_hidden, hidden_width)]["n_params"] = num_parameters_kan(n_inputs, n_outputs, n_hidden, hidden_width, G, k)

    with open(dir + f"{network_type}/summary.pkl", 'wb') as file:
        pkl.dump(summary, file)


def rel_l2_error(residual, ref_sol):
    return 100 * LA.norm(residual) / LA.norm(ref_sol)


def max_abs_error(residual):
    return np.max(np.abs(residual))


def compute_error_stats(model, x_eval, ref_sol, ref_grad, dir, repetitions):
    rel_l2_errors = []
    max_abs_errors = []
    rel_l2_errors_grad = []
    max_abs_errors_grad = []
    model.to(x_eval.device)

    for rep in range(repetitions):
        model.load_state_dict(torch.load(dir + f"model{rep+1}.pkl", weights_only=True))
        model.eval()
        pred_sol = model(x_eval)
        pred_grad = dy_dx(pred_sol, x_eval)
        with torch.no_grad():
            # Solution error
            pred_sol = pred_sol.to('cpu').detach().numpy()
            residual = ref_sol - pred_sol
            rel_l2_errors.append(rel_l2_error(residual, ref_sol))
            max_abs_errors.append(max_abs_error(residual))

            # Gradient error 
            pred_grad = pred_grad.to('cpu').detach().numpy() # shape(num_points, input_dim)
            grad_residual_norm = LA.norm(ref_grad - pred_grad, axis=-1, keepdims=True)
            ref_grad_norm = LA.norm(ref_grad, axis=-1, keepdims=True)
            rel_l2_errors_grad.append(rel_l2_error(grad_residual_norm, ref_grad_norm))
            max_abs_errors_grad.append(max_abs_error(grad_residual_norm))
            
    summary = {
        'relative_l2_errors': rel_l2_errors, 
        'avg_rel_l2_err': np.mean(rel_l2_errors), 
        'std_rel_l2_err': np.std(rel_l2_errors), 
        'max_abs_errors': max_abs_errors, 
        'avg_max_abs_err': np.mean(max_abs_errors), 
        'std_max_abs_err': np.std(max_abs_errors), 
        'relative_l2_grad_errors': rel_l2_errors_grad, 
        'avg_rel_l2_grad_err': np.mean(rel_l2_errors_grad), 
        'std_rel_l2_grad_err': np.std(rel_l2_errors_grad), 
        'max_abs_grad_errors': max_abs_errors_grad, 
        'avg_max_abs_grad_err': np.mean(max_abs_errors_grad), 
        'std_max_abs_grad_err': np.std(max_abs_errors_grad), 
        'lowest_relative_l2_ix': np.argmin(rel_l2_errors)
    }

    return summary


def print_summary(summary):
    for (n_hidden, hidden_width), local_summary in summary.items():
        print(f"Nº hidden layers: {n_hidden} | Hidden layer width: {hidden_width} | Nº paramers: {local_summary['n_params']}")
        print(f"    Relative L2 error (mean): {local_summary['avg_rel_l2_err']}%")
        print(f"    Relative L2 error (std): {local_summary['std_rel_l2_err']}%")
        print(f"    Maximum absolute error (mean): {local_summary['avg_max_abs_err']}")
        print(f"    Maximum absolute error (std): {local_summary['std_max_abs_err']}")
        print(f"    Relative L2 gradient error (mean): {local_summary['avg_rel_l2_grad_err']}%")
        print(f"    Relative L2 gradient error (std): {local_summary['std_rel_l2_grad_err']}%")
        print(f"    Maximum absolute gradient error (mean): {local_summary['avg_max_abs_grad_err']}")
        print(f"    Maximum absolute gradient error (std): {local_summary['std_max_abs_grad_err']}")


# --------------------
# Plots
# --------------------

def plot_losses(n_hidden_pinn_list, hidden_width_pinn_list, best_pinn_ixs, n_hidden_pikan_list, hidden_width_pikan_list, best_pikan_ixs, G, k, results_dir, figsize=(14, 6), smoothness=0.98):
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#ff0000","#00b3ff","#000000","#43b813","#f700ff","#ff8f00","#0000ff","#767676"]
    color_cycle = itertools.cycle(colors)
    ax.set_yscale('log')
    ax.set_xlabel(r'Iteration')
    ax.set_ylabel(r'Total Loss')

    # Grid + minor ticks
    ax.grid(True, which='major', linestyle='solid', color='gainsboro')

    alpha = 0.2

    # PINN
    for n_hidden, hidden_width, best_ix in zip(n_hidden_pinn_list, hidden_width_pinn_list, best_pinn_ixs):
        losses = np.load(results_dir + f"PINN/pinn_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        ax.plot(losses['total'], color=next(color_cycle), alpha=alpha)

    # PIKAN
    for n_hidden, hidden_width, best_ix in zip(n_hidden_pikan_list, hidden_width_pikan_list, best_pikan_ixs):
        losses = np.load(results_dir + f"PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        ax.plot(losses['total'], color=next(color_cycle), alpha=alpha)

    color_cycle = itertools.cycle(colors)
    # PINN smoothed
    for n_hidden, hidden_width, best_ix in zip(n_hidden_pinn_list, hidden_width_pinn_list, best_pinn_ixs):
        losses = np.load(results_dir + f"PINN/pinn_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        losses = smooth_losses(losses['total'], smoothness)
        ax.plot(losses, label=rf"PINN: $L={n_hidden}$, $W={hidden_width}$", color=next(color_cycle))

    # PIKAN smoothed
    for n_hidden, hidden_width, best_ix in zip(n_hidden_pikan_list, hidden_width_pikan_list, best_pikan_ixs):
        losses = np.load(results_dir + f"PIKAN/pikan_G{G}_k{k}_hidden{n_hidden}_width{hidden_width}/losses{best_ix+1}.npz")
        losses = smooth_losses(losses['total'], smoothness)
        ax.plot(losses, label=rf"PIKAN: $L={n_hidden}$, $W={hidden_width}$", color=next(color_cycle))

    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

    # Tight layout to avoid clipping
    fig.tight_layout()

    # Save the figure
    fig.savefig(results_dir + "losses_fig.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_loss(ax, losses, label=None, color='blue', plot_smoothed=True, smoothness=0.8):
    ax.plot(losses, label=(None if plot_smoothed else label), color=color, alpha=(0.3 if plot_smoothed else 1))
    if plot_smoothed:
        smoothed_losses = smooth_losses(losses, smoothness)
        ax.plot(smoothed_losses, label=label, color=color)


def smooth_losses(losses, smoothness):
    smoothed_losses = np.copy(losses)
    for i in range(1, smoothed_losses.size):
        smoothed_losses[i] = smoothness * smoothed_losses[i-1] + (1 - smoothness) * smoothed_losses[i]
    return smoothed_losses


def decorate_loss_plot(ax, title, x_label, y_label):
    ax.set_title(title)
    ax.set_yscale('log')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.get_major_locator().set_params(numticks=99, subs='all')
    ax.grid(True, which='both', linestyle='solid', color='gainsboro')




def plot_pde_solution(sol, x, y, title, x_label, y_label, outdir, figsize=(6, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, rotation=0, ha='right')
    ax.set_xticks(ax.get_yticks()) # Same ticks for both axes
    # Grid
    ax.grid(True, which='both', linestyle='dotted')

    # Plot
    img = ax.imshow(np.flip(np.transpose(sol), 0), extent=[x.min(), x.max(), y.min(), y.max()], cmap='plasma', aspect='equal')
    fig.colorbar(img, ax=ax)

    # Tight layout to avoid clipping
    fig.tight_layout()

    # Save the figure
    fig.savefig(outdir + "solution_fig.png", dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()


def plot_pde_predVref(sol_ref, sol_pred, x, y, ref_label, pred_label, x_label, y_label, outfile):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))

    # Plot reference
    axes[0].set_title(ref_label, pad = 20)
    img = axes[0].imshow(np.flip(np.transpose(sol_ref), 0), extent=[x.min(), x.max(), y.min(), y.max()], cmap='plasma', aspect='equal')
    fig.colorbar(img, ax=axes[0])

    # Plot prediction
    axes[1].set_title(pred_label, pad = 20)
    img = axes[1].imshow(np.flip(np.transpose(sol_pred), 0), extent=[x.min(), x.max(), y.min(), y.max()], cmap='plasma', aspect='equal')
    fig.colorbar(img, ax=axes[1])

    # Plot residual
    residual = sol_ref - sol_pred
    axes[2].set_title(f"{ref_label} $-$ {pred_label}", pad = 20)
    img = axes[2].imshow(np.flip(np.transpose(residual), 0), extent=[x.min(), x.max(), y.min(), y.max()], cmap='coolwarm', vmin=-abs(residual).max(), vmax=abs(residual).max(), aspect='equal')
    fig.colorbar(img, ax=axes[2])

    for ax in axes:
        ax.set_xticks(np.linspace(x.min(), x.max(), 5))
        ax.set_yticks(np.linspace(y.min(), y.max(), 5))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label, rotation=0, ha='right')
        ax.set_xticks(ax.get_yticks()) # Same ticks for both axes
        # Grid
        ax.grid(True, which='both', linestyle='dotted')

    # Tight layout to avoid clipping
    fig.tight_layout()

    # Save the figure
    fig.savefig(outfile, dpi=500, bbox_inches='tight', pad_inches=0)
    plt.show()
    

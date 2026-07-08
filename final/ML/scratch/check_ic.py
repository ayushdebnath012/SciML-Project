import torch
import numpy as np

# Mocking the MaterialModel and gaussian_ic from the files
def gaussian_ic_pikan(x, sigma_g=0.1, x0=0.0):
    f = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdx = -(x - x0) / sigma_g**2 * f
    return dfdx / (dfdx.abs().max() + 1e-12)

def gaussian_ic_benchmark(x, sigma_g=0.1, x0=0.0):
    f    = torch.exp(-0.5 * ((x - x0) / sigma_g) ** 2)
    dfdx = -(x - x0) / sigma_g**2 * f
    return dfdx / (dfdx.abs().max() + 1e-12)

x = torch.linspace(-1, 1, 11)
g_pikan = gaussian_ic_pikan(x)
g_bench = gaussian_ic_benchmark(x)

print("x coords:", x.numpy())
print("PIKAN IC:", g_pikan.numpy())
print("Bench IC:", g_bench.numpy())

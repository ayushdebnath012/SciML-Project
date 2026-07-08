#!/bin/bash

#SBATCH -c 48
#SBATCH --mem=96GB
#SBATCH -p serc --gres gpu:0
#SBATCH --time=120:00:00
#SBATCH --job-name="code0_hybrid"

module load py-jupyter/1.0.0_py39
module load py-numpy/1.20.3_py39
module load py-scipy/1.10.1_py39
module load py-tensorflow/2.9.1_py39

pip install --upgrade pip
pip install pennylane --upgrade
pip install pennylane-lightning
 
srun python3 DC_PINN_hybrid.py

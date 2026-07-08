#!/bin/bash

#SBATCH -c 2
#SBATCH -p serc --gres gpu:1
#SBATCH --time=24:00:00

singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif pip install --user pennylane==0.38.0 
singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif pip install --user pennylane-qiskit==0.36.0
singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif pip install --user qiskit==1.1.0 
singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif pip install --user qiskit-aer==0.13.3
singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif pip install --user qiskit-ibm-runtime==0.23.0
singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif pip install --user dimod
singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif pip install --user dwave-ocean-sdk
singularity exec --nv ../../../groups/mukerji/divakarv/pennylane-tf.sif python3.10 noisy_qaoa_10k_shots.py
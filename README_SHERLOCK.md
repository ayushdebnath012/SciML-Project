# Wave Equation Experiments on Stanford Sherlock HPC
This guide contains instructions to set up the environment and run the parameter sweep experiments on Stanford's Sherlock cluster.

---

## 📂 Sherlock Files & Scripts

- **`wave/setup_sherlock.sh`**: One-time environment setup script. Loads Python, builds a virtual environment in `$SCRATCH` (avoiding NFS home quota lags), upgrades pip, installs PyTorch with CUDA 12.1 support, installs dependencies, and creates output folders.
- **`wave/slurm_array.sh`**: Slurm Array Job script. Runs all 360 parameter combinations in parallel, using one GPU per task. Throttled by default to run up to 30 tasks concurrently.
- **`wave/slurm_h100.sh`**: Slurm job script to run the entire 360 parameter configurations sweep sequentially on a single requested H100 GPU (takes ~12-18 hours).
- **`wave/run_experiment.py`**: The underlying Python runner that executes either a single configuration mapping to a `--run-id` or the sequential sweep.



##### ⚙️ How to Choose How Many GPUs to Use:
By default, `wave/slurm_array.sh` caps concurrency at **30 GPUs** (`--array=0-359%30`) so it doesn't hog the group/cluster queue. You can change this concurrency limit (the `%N` suffix) at submission time:
- **Use 10 GPUs concurrently**:
  ```bash
  sbatch --array=0-359%10 wave/slurm_array.sh
  ```
- **Use 50 GPUs concurrently**:
  ```bash
  sbatch --array=0-359%50 wave/slurm_array.sh
  ```
- **Use 1 GPU at a time (sequential execution as array tasks)**:
  ```bash
  sbatch --array=0-359%1 wave/slurm_array.sh
  ```

#### Option B: Submit a Single H100 GPU Job (Sequential Sweep)
To request a single H100 node and run all 360 configurations one-by-one sequentially:
```bash
sbatch wave/slurm_h100.sh
```
*(To target a different GPU type like an A100 or V100, open `wave/slurm_h100.sh` and edit the `#SBATCH --gres=gpu:h100:1` line to `gpu:a100:1` or `gpu:v100:1` respectively).*

---

If you need to change hyperparameter options, you can do so in **`wave/run_experiment.py`**:
- **Iterative Limits**: Set `adam_iterations` and `lbfgs_iterations` in the `CONFIG` dictionary (around line 80).
- **Collocation Points**: Adjust `Nx_collocation` and `Nt_collocation` in `CONFIG` to control grid resolution.
- **Model Sweeps**: Modify the `MODEL_CONFIGS` array (around line 37) to add/remove configurations of PINN, FourierFeaturePINN, PirateNet, KAN, or WavKAN.

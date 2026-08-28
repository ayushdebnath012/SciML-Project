import os
import torch
import numpy as np

MODEL_DIR = "Models"

def get_hyperparams(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    
    filename = os.path.basename(path)
    # Filename format: prefix_arch_model.pt
    parts = filename.split('_')
    arch = parts[-2]
    experiment = "_".join(parts[:-2])
    
    params = {
        "Experiment": experiment,
        "Architecture": arch,
        "File": filename
    }
    
    try:
        if arch == "vanilla":
            units = sd['net.0.weight'].shape[0]
            n_layers = sum(1 for k in sd.keys() if 'net.' in k and '.weight' in k) - 1
            params.update({
                "Hidden Layers": n_layers,
                "Hidden Units": units
            })
        
        elif arch == "fourier":
            n_fourier = sd['B'].shape[0]
            units = sd['net.0.weight'].shape[0]
            n_layers = sum(1 for k in sd.keys() if 'net.' in k and '.weight' in k) - 1
            # Estimate sigma from B
            sigma_est = sd['B'].std().item() / (1.0 / np.sqrt(2)) # roughly, if initialized with randn * sigma
            # Actually, B is initialized as randn(n_fourier, 2) * sigma. 
            # The std of randn is 1.0. So std(B) should be approx sigma.
            sigma = sd['B'].std().item()
            params.update({
                "Hidden Layers": n_layers,
                "Hidden Units": units,
                "N Fourier": n_fourier,
                "Estimated Sigma": round(sigma, 2)
            })
        
        elif arch == "pirate":
            n_fourier = sd['B'].shape[0]
            is_rwf = 'enc_U.V' in sd
            if is_rwf:
                units = sd['enc_U.V'].shape[0]
            else:
                units = sd['enc_U.0.weight'].shape[0]
                
            block_ids = [int(k.split('.')[1]) for k in sd.keys() if k.startswith('blocks.')]
            n_blocks = max(block_ids) + 1 if block_ids else 0
            sigma = sd['B'].std().item()
            
            params.update({
                "N Blocks": n_blocks,
                "Units": units,
                "N Fourier": n_fourier,
                "Sigma": round(sigma, 2),
                "RWF": is_rwf
            })
            
    except Exception as e:
        params["Error"] = str(e)
        
    return params

def main():
    if not os.path.exists(MODEL_DIR):
        print(f"Directory {MODEL_DIR} not found.")
        return

    files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
    all_params = []
    for f in files:
        if "homogeneous" in f.lower():
            continue
            
        path = os.path.join(MODEL_DIR, f)
        params = get_hyperparams(path)
        
        if params.get("Architecture") == "vanilla":
            continue
            
        all_params.append(params)
    
    # Generate Markdown Table
    if not all_params:
        print("No models found.")
        return
        
    # Sort by experiment and architecture
    all_params.sort(key=lambda x: (x['Experiment'], x['Architecture']))
    
    with open("model_hyperparameters.md", "w") as f:
        f.write("# Model Hyperparameters Summary\n\n")
        f.write("This file contains the architectural hyperparameters extracted from the trained model files.\n\n")
        
        # Collect all unique keys for columns
        keys = set()
        for p in all_params:
            keys.update(p.keys())
        
        # Fixed order for common keys
        header_keys = ["Experiment", "Architecture", "File", "Hidden Layers", "Hidden Units", "N Blocks", "Units", "N Fourier", "Sigma", "Estimated Sigma", "RWF"]
        # Only keep keys that exist in any param
        header_keys = [k for k in header_keys if k in keys]
        # Add remaining keys
        remaining_keys = sorted(list(keys - set(header_keys)))
        header_keys.extend(remaining_keys)
        
        # Write table header
        f.write("| " + " | ".join(header_keys) + " |\n")
        f.write("| " + " | ".join(["---"] * len(header_keys)) + " |\n")
        
        for p in all_params:
            row = []
            for k in header_keys:
                row.append(str(p.get(k, "-")))
            f.write("| " + " | ".join(row) + " |\n")

    print("Created model_hyperparameters.md")

if __name__ == "__main__":
    main()

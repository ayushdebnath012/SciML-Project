import os
import torch
import torch.nn as nn
import numpy as np

try:
    from kan import KAN
except ImportError as e:
    print(f"  [DEBUG] Failed to import KAN: {e}")
    KAN = None

try:
    from training_code_piann import PIANN
except ImportError:
    PIANN = None

from src.models import (
    VanillaPINN, FourierFeaturePINN, PirateNet, KANWrapper, FNOWrapper
)

# ─────────────────────────────────────────────
# 1. Robust Model Loader from Checkpoint Files
# ─────────────────────────────────────────────

def load_models(prefix, model_dir="Models", run_archs=None, device=None):
    """
    Robust model loader that infers architecture params from checkpoint keys.
    """
    if run_archs is None:
        run_archs = ["fourier", "pirate", "pikan"]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    models = {}
    
    # Check for possible architectures
    for arch in run_archs:
        # Try different extensions and suffixes
        found_path = None
        for ext in [".pt", ".pkl"]:
            for suffix in ["_model", "_best", ""]:
                path = os.path.join(model_dir, f"{prefix}_{arch}{suffix}{ext}")
                if os.path.exists(path):
                    found_path = path
                    break
            if found_path: 
                break
        
        if not found_path:
            continue

        sd = torch.load(found_path, map_location=device, weights_only=False)
        
        try:
            if arch == "vanilla":
                units = sd['net.0.weight'].shape[0]
                n_layers = sum(1 for k in sd.keys() if 'net.' in k and '.weight' in k) - 1
                model = VanillaPINN(hidden_layers=n_layers, hidden_units=units).to(device)
            
            elif arch == "fourier":
                n_fourier = sd['B'].shape[0]
                units = sd['net.0.weight'].shape[0]
                n_layers = sum(1 for k in sd.keys() if 'net.' in k and '.weight' in k) - 1
                model = FourierFeaturePINN(hidden_layers=n_layers, hidden_units=units, n_fourier=n_fourier).to(device)
            
            elif arch == "pirate":
                n_fourier = sd['B'].shape[0]
                is_rwf = 'enc_U.V' in sd
                units = sd['enc_U.V'].shape[0] if is_rwf else sd['enc_U.0.weight'].shape[0]
                block_ids = [int(k.split('.')[1]) for k in sd.keys() if k.startswith('blocks.')]
                n_blocks = max(block_ids) + 1 if block_ids else 0
                model = PirateNet(n_blocks=n_blocks, units=units, n_fourier=n_fourier).to(device)
                
                # Conversion logic for legacy standard Linear weights to RWFLinear
                if not is_rwf:
                    print(f"  [PirateNet] Converting legacy weights for {prefix}...")
                    new_sd = {}
                    for k, v in sd.items():
                        if any(x in k for x in ['enc_U', 'enc_V', 'proj', 'blocks', 'out_layer']) and '.weight' in k:
                            prefix_key = k.replace('.weight', '')
                            new_sd[f"{prefix_key}.V"] = v
                            new_sd[f"{prefix_key}.s"] = torch.ones(v.shape[0], device=device)
                        else:
                            new_sd[k] = v
                    sd = new_sd
            
            elif arch == "pikan":
                if KAN is None:
                    print(f"  [Skip] pykan library not found. Cannot load {arch}")
                    continue
                
                # Infer grid and k from state dict to avoid size mismatch
                try:
                    grid_key = next(k for k in sd.keys() if 'grid' in k)
                    coef_key = next(k for k in sd.keys() if 'coef' in k)
                    g_len = sd[grid_key].shape[-1]
                    c_len = sd[coef_key].shape[-1]
                    k_val = g_len - c_len - 1
                    grid_val = c_len - k_val
                    print(f"  [pikan] Inferred grid={grid_val}, k={k_val}")
                except Exception:
                    grid_val, k_val = 5, 3 # Fallback to defaults
                
                # Architecture specified by user: 2-5-5-5-5-1
                kan_model = KAN(width=[2, 5, 5, 5, 5, 1], grid=grid_val, k=k_val).to(device)
                
                # Robust loading: handles state_dict or full model object
                if isinstance(sd, dict):
                    kan_model.load_state_dict(sd)
                elif hasattr(sd, 'state_dict'):
                    kan_model.load_state_dict(sd.state_dict())
                else:
                    print(f"  [Warn] Unknown model format for {arch}")
                    kan_model = sd
                
                model = KANWrapper(kan_model)
                model.eval()
                models[arch] = model
                print(f"  [OK] Loaded {arch} model (pykan 2-5-5-5-1)")
                continue
            
            elif arch == "fno":
                width = sd['fno.fc0.weight'].shape[0]
                modes = sd['fno.spectral_layers.0.weight_real'].shape[-1]
                layer_ids = [int(k.split('.')[2]) for k in sd.keys()
                             if k.startswith('fno.spectral_layers.') and k.endswith('.weight_real')]
                n_layers = max(layer_ids) + 1
                nx = sd['x_grid'].shape[0]
                model = FNOWrapper(nx=nx, modes=modes, width=width, n_layers=n_layers).to(device)

            elif "wavekan" in arch.lower():
                from src.models import WavKAN
                # Demention reconstruction from state dict
                scale_keys = [k for k in sd.keys() if k.endswith('.scale')]
                scale_keys.sort(key=lambda x: int(x.split('.')[1]))
                
                layers_hidden = []
                first_key = scale_keys[0]
                in_features = sd[first_key].shape[1]
                layers_hidden.append(in_features)
                
                for k in scale_keys:
                    out_features = sd[k].shape[0]
                    layers_hidden.append(out_features)
                
                kan_model = WavKAN(layers_hidden=layers_hidden, wavelet_type='mexican_hat').to(device)
                
                class WavKANWrapper(nn.Module):
                    def __init__(self, w_model):
                        super().__init__()
                        self.w_model = w_model
                    def forward(self, x, t):
                        return self.w_model(torch.cat([x, t], dim=-1))
                
                model = WavKANWrapper(kan_model)
                model.eval()
                kan_model.load_state_dict(sd)
                models[arch] = model
                print(f"  [OK] Loaded {arch} model (WavKAN {layers_hidden})")
                continue

            elif arch.startswith("piann"):
                if PIANN is None:
                    print(f"  [Skip] training_code_piann library not found. Cannot load {arch}")
                    continue
                model = PIANN(x_min=-1.0, x_max=1.0, N=128, hidden_size=128).to(device)

            model.load_state_dict(sd)
            model.eval()
            models[arch] = model
            print(f"  [OK] Loaded {arch} model")
            
        except Exception as e:
            print(f"  ! Failed to load {arch} from {found_path or path}: {e}")

    return models


# ─────────────────────────────────────────────
# 2. Hyperparameter Extractor for Summaries
# ─────────────────────────────────────────────

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

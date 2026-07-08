"""
Kaggle T4 entrypoint for the 1D elastic-wave FNO baseline.

Typical Kaggle notebook usage:

    !python /kaggle/working/SciML-Project/kaggle/run_fno_t4.py --preset full

Use --preset smoke first to verify the GPU/runtime, then --preset medium or
--preset full for real runs. Extra arguments after the preset override the
underlying wave/run_fno_baseline.py options.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_SCRIPT = REPO_ROOT / "wave" / "run_fno_baseline.py"


PRESETS = {
    "smoke": {
        "model": "all",
        "num_samples": 32,
        "nx": 64,
        "nt": 64,
        "eval_nx": 96,
        "eval_nt": 96,
        "epochs": 10,
        "batch_size": 4,
        "width": 16,
        "modes_x": 8,
        "modes_t": 8,
        "layers": 3,
        "lr": 1e-3,
        "min_lr": 2e-5,
        "save_every": 5,
        "print_every": 1,
        "random_ic": True,
    },
    "medium": {
        "model": "fno",
        "num_samples": 256,
        "nx": 128,
        "nt": 128,
        "eval_nx": 192,
        "eval_nt": 192,
        "epochs": 150,
        "batch_size": 8,
        "width": 48,
        "modes_x": 16,
        "modes_t": 16,
        "layers": 4,
        "lr": 8e-4,
        "min_lr": 1e-5,
        "save_every": 10,
        "print_every": 5,
        "random_ic": True,
    },
    "full": {
        "model": "fno",
        "num_samples": 768,
        "nx": 128,
        "nt": 128,
        "eval_nx": 256,
        "eval_nt": 256,
        "epochs": 350,
        "batch_size": 8,
        "width": 64,
        "modes_x": 20,
        "modes_t": 20,
        "layers": 4,
        "lr": 8e-4,
        "min_lr": 1e-5,
        "save_every": 10,
        "print_every": 5,
        "random_ic": True,
    },
}


def kaggle_working_dir() -> Path:
    path = Path("/kaggle/working")
    if path.exists():
        return path
    return REPO_ROOT / "kaggle_outputs"


def add_option(cmd, name, value):
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
    else:
        cmd.extend([flag, str(value)])


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FNO baseline with Kaggle T4 presets.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dataset-cache", default=None)
    parser.add_argument("--compare-cnn", action="store_true",
                        help="Train both FNO and CNN comparator for the selected preset.")
    parser.add_argument("--amp", action="store_true",
                        help="Forward --amp to the trainer. Leave off if CUDA FFT autocast is unstable.")
    parser.add_argument("--compile", action="store_true",
                        help="Forward --compile to the trainer.")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-zip", action="store_true",
                        help="Do not zip the output directory at the end.")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def main():
    args, passthrough = parse_args()
    preset = dict(PRESETS[args.preset])
    if args.compare_cnn:
        preset["model"] = "all"

    workdir = kaggle_working_dir()
    output_dir = Path(args.output_dir) if args.output_dir else workdir / f"fno_t4_{args.preset}"
    dataset_cache = (
        Path(args.dataset_cache)
        if args.dataset_cache
        else workdir / "operator_data" / f"wave_operator_{args.preset}_seed42.npz"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_cache.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(BASELINE_SCRIPT),
        "--device", "cuda",
        "--output-dir", str(output_dir),
        "--dataset-cache", str(dataset_cache),
        "--num-workers", str(args.num_workers),
        "--pin-memory",
    ]
    for key, value in preset.items():
        add_option(cmd, key, value)
    if args.amp:
        cmd.append("--amp")
    if args.compile:
        cmd.append("--compile")
    cmd.extend(passthrough)

    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    print("Running:")
    print(" ".join(f'"{part}"' if " " in part else part for part in cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    if not args.no_zip:
        archive_base = output_dir.with_suffix("")
        archive_path = shutil.make_archive(str(archive_base), "zip", output_dir)
        print(f"Zipped results: {archive_path}")


if __name__ == "__main__":
    main()

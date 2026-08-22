"""
run_vs.py — Main entry point

Changes from 2.53% run:
  adam_steps default: 60000 → 120000
  n_snaps_interface: 15 (new, passed to train_vs)
"""
import os, argparse
import jax

print(f"JAX devices : {jax.devices()}")
print(f"JAX backend : {jax.default_backend()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke",            action="store_true")
    p.add_argument("--eval",             action="store_true")
    p.add_argument("--windows",          type=int, default=4)
    p.add_argument("--adam_steps",       type=int, default=120_000)
    p.add_argument("--lbfgs_iter",       type=int, default=20_000)
    p.add_argument("--n_bulk",           type=int, default=20_000)
    p.add_argument("--n_snaps",          type=int, default=15)
    p.add_argument("--n_snaps_interface",type=int, default=15)
    p.add_argument("--fdm_path",         type=str, default="fdm_data.npz")
    p.add_argument("--ckpt_dir",         type=str, default="checkpoints_v3")
    p.add_argument("--eval_dir",         type=str, default="eval_output_v3")
    args = p.parse_args()

    if not os.path.exists(args.fdm_path) or not _has_snaps(args.fdm_path):
        print("Generating FDM reference data...")
        from fdm_reference import save_reference
        save_reference(args.fdm_path)

    if args.eval:
        from evaluate_vs import evaluate
        r = evaluate(args.ckpt_dir, args.fdm_path, args.eval_dir)
        print(f"\nVelocity L2 : {r['l2_velocity']*100:.2f}%")
        print(f"Stress   L2 : {r['l2_stress']*100:.2f}%")
        return

    from train_vs import train

    if args.smoke:
        config = {
            "n_windows": 2, "n_bulk": 3_000, "n_interface": 200,
            "n_source": 200, "n_snaps": 5, "n_snaps_interface": 5,
            "n_pts_snap": 100, "adam_steps": 500, "lbfgs_iter": 500,
            "checkpoint_dir": args.ckpt_dir + "_smoke", "fdm_path": args.fdm_path,
        }
        print("[SMOKE TEST]")
    else:
        config = {
            "n_windows":          args.windows,
            "n_bulk":             args.n_bulk,
            "n_interface":        2_000,
            "n_source":           2_000,
            "n_snaps":            args.n_snaps,
            "n_snaps_interface":  args.n_snaps_interface,
            "n_pts_snap":         800,
            "adam_steps":         args.adam_steps,
            "lbfgs_iter":         args.lbfgs_iter,
            "lr":                 5e-4,
            "causal_eps":         1.0,
            "checkpoint_dir":     args.ckpt_dir,
            "fdm_path":           args.fdm_path,
        }

    all_models, all_params, windows = train(config)

    print("\nRunning evaluation...")
    from evaluate_vs import evaluate
    r = evaluate(config.get("checkpoint_dir", args.ckpt_dir), args.fdm_path, args.eval_dir)

    print("\n" + "═"*50)
    print(f"  Velocity L2 : {r['l2_velocity']*100:.2f}%  ← PRIMARY")
    print(f"  Stress   L2 : {r['l2_stress']*100:.2f}%")
    print(f"  Figures     : {args.eval_dir}/")
    print("═"*50)


def _has_snaps(path):
    try:
        import numpy as np
        d = np.load(path)
        return "v_snaps" in d and "t_snaps" in d
    except Exception:
        return False


if __name__ == "__main__":
    main()
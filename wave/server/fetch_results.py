"""Pull sweep results off the training box into the local repository.

Only the small artefacts are copied by default -- `l2_errors.json` and
`causal_convergence.json` are what the paper's tables are built from, and the
`.eqx` checkpoints are large and not needed to regenerate a number. Corrected
causal runs are fetched last and overlay the matching original run directories
locally, while the two remote result trees remain separate for audit.

    python wave/server/fetch_results.py --host 10.71.9.40 --user trishita
    python wave/server/fetch_results.py --include-plots      # also the PNGs
    python wave/server/fetch_results.py --archive-original results/pinn_pre_contfix

Set the password in RPASS rather than passing it on the command line.
"""
import argparse
import os
import posixpath
import stat
import sys

import paramiko

SMALL_FILES = ("l2_errors.json", "causal_convergence.json")
PLOT_FILES = ("loss_curve.png", "gradnorm_weights.png", "causal_wmin.png",
              "causal_heatmap.png", "solution_comparison.png")

REMOTE_LOCAL = [
    ("/home/{user}/sciml_pinn_neurips2026/results", "results/pinn"),
    ("/home/{user}/sciml_pinn_neurips2026/results_ablation", "results/pinn_ablation"),
    # Corrected causal runs intentionally overlay their matching baseline
    # directories locally. Keep this entry after `results`: configurations not
    # affected by the continuation fix still come from the original sweep.
    ("/home/{user}/sciml_pinn_neurips2026/results_corrected", "results/pinn"),
]


def connect(host, user, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username=user, password=password, timeout=30,
                   allow_agent=False, look_for_keys=False)
    return client


def fetch_tree(sftp, remote, local, wanted):
    copied = 0
    try:
        sftp.stat(remote)
    except IOError:
        print(f"  (nothing at {remote})")
        return 0
    stack = [(remote, local)]
    while stack:
        rdir, ldir = stack.pop()
        for entry in sftp.listdir_attr(rdir):
            rpath = posixpath.join(rdir, entry.filename)
            lpath = os.path.join(ldir, entry.filename)
            if stat.S_ISDIR(entry.st_mode):
                stack.append((rpath, lpath))
            elif entry.filename in wanted:
                os.makedirs(ldir, exist_ok=True)
                sftp.get(rpath, lpath)
                copied += 1
    return copied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default=os.environ.get("RHOST", "10.71.9.40"))
    p.add_argument("--user", default=os.environ.get("RUSER", "trishita"))
    p.add_argument("--include-plots", action="store_true")
    p.add_argument(
        "--archive-original",
        metavar="DIR",
        help=("also copy the uncorrected remote sweep into DIR before the "
              "corrected results overlay the working result tree"),
    )
    a = p.parse_args()

    password = os.environ.get("RPASS")
    if not password:
        print("set RPASS in the environment", file=sys.stderr)
        return 2

    wanted = set(SMALL_FILES) | (set(PLOT_FILES) if a.include_plots else set())
    client = connect(a.host, a.user, password)
    try:
        sftp = client.open_sftp()
        if a.archive_original:
            remote = REMOTE_LOCAL[0][0].format(user=a.user)
            print(f"{remote} -> {a.archive_original} (uncorrected archive)")
            n = fetch_tree(sftp, remote, a.archive_original, wanted)
            print(f"  {n} files")
        for remote_tmpl, local in REMOTE_LOCAL:
            remote = remote_tmpl.format(user=a.user)
            print(f"{remote} -> {local}")
            n = fetch_tree(sftp, remote, local, wanted)
            print(f"  {n} files")
        fetch_timings(sftp, a.user)
        sftp.close()
    finally:
        client.close()
    return 0


def fetch_timings(sftp, user):
    """Summarise the pool's per-run wall clock into results/pinn/run_seconds.json.

    The figure needs one number for where to put the PINN points on the time
    axis; the spread goes in the text. Taken from the pool's own `done.txt`
    rather than timed separately, so it is the real end-to-end cost of a run
    including compilation and evaluation.
    """
    import json
    remote = f"/home/{user}/sciml_pinn_neurips2026/sweep_state/done.txt"
    try:
        with sftp.open(remote) as fh:
            text = fh.read().decode()
    except IOError:
        print("  (no done.txt yet; skipping timings)")
        return
    secs = []
    for line in text.splitlines():
        for field in line.split():
            if field.startswith("secs="):
                secs.append(float(field.split("=", 1)[1]))
    if not secs:
        return
    secs.sort()
    summary = {
        "n": len(secs),
        "median_seconds": secs[len(secs) // 2],
        "min_seconds": secs[0],
        "max_seconds": secs[-1],
    }
    out = os.path.join("results", "pinn", "run_seconds.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  timings: n={summary['n']} median={summary['median_seconds']:.0f}s "
          f"range {summary['min_seconds']:.0f}-{summary['max_seconds']:.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())

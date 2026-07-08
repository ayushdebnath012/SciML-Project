# Kaggle T4 FNO Baseline

This folder contains a Kaggle-ready launcher for the supervised FNO baseline
for the 1D conservative elastic wave equation.

## Notebook Setup

1. Create a Kaggle notebook.
2. Enable GPU acceleration and choose a T4 GPU runtime.
3. Upload or clone this repository into `/kaggle/working`.
4. Run a smoke test:

```bash
python kaggle/run_fno_t4.py --preset smoke
```

## Real Runs

Medium run, usually a good first T4 training job:

```bash
python kaggle/run_fno_t4.py --preset medium
```

Full run:

```bash
python kaggle/run_fno_t4.py --preset full
```

Full run with the CNN comparator:

```bash
python kaggle/run_fno_t4.py --preset full --compare-cnn
```

The launcher writes outputs to `/kaggle/working/fno_t4_<preset>/` and caches
FD-generated data in `/kaggle/working/operator_data/`. At the end it zips the
result folder so it is easy to download from Kaggle outputs.

## Presets

| Preset | Samples | Train Grid | Eval Grid | Epochs | Width | Modes |
| --- | ---: | --- | --- | ---: | ---: | --- |
| `smoke` | 32 | 64 x 64 | 96 x 96 | 10 | 16 | 8 x 8 |
| `medium` | 256 | 128 x 128 | 192 x 192 | 150 | 48 | 16 x 16 |
| `full` | 768 | 128 x 128 | 256 x 256 | 350 | 64 | 20 x 20 |

## Useful Overrides

All extra arguments are forwarded to `wave/run_fno_baseline.py`, so you can
override any training option:

```bash
python kaggle/run_fno_t4.py --preset full --epochs 500 --batch-size 6
```

If CUDA FFT autocast is stable in your runtime, you can try:

```bash
python kaggle/run_fno_t4.py --preset full --amp
```

Keep `--amp` off if you see FFT/autocast errors.

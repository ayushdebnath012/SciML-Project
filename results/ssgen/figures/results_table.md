# Neural-operator forward benchmark

Task: velocity model -> shot gathers, per run. Errors on physical amplitudes; a zero prediction scores ~100 %.

## ssgen

600 train / 100 val, velocity (309 x 500) -> 5 shots x 572 steps x 1000 receivers, 80 epochs on NVIDIA H100 NVL. Normalization: zscore.

| model | real params | rel L2 % | out-of-dist % | late-time % | median % | MAE | RMSE | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FNO | 5,404,115 | **46.06** | 47.74 | 100.0 | 54.32 | 0.0025 | 0.0084 | 10.7 |
| DeepONet | 39,766,273 | **50.79** | 51.41 | 107.8 | 57.30 | 0.0031 | 0.0092 | 23.1 |
| PFNO | 2,194,280 | **58.95** | 58.83 | 146.9 | 63.47 | 0.0031 | 0.0105 | 150.5 |
| GNO | 109,139 | **71.54** | 71.73 | 100.3 | 74.20 | 0.0027 | 0.0127 | 158.6 |

**late-time %** is relative L2 after the first 20 % of the record, where the reflections live. A zero prediction scores ~100 %, so a value at or above 100 means the model reproduced the direct arrival and nothing else.

Representation scales: PFNO band limit (220 bins) **3.435 %** - a hard floor, bins above the band are zeroed. Time latent (286 pts) 65.236 % for a single-channel resample - a reference, not a bound: the latent is multi-channel.

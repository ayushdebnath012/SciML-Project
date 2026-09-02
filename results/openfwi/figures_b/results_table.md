# Neural-operator forward benchmark

Task: velocity model -> shot gathers, per run. Errors on physical amplitudes; a zero prediction scores ~100 %.

## flatvel_b

2000 train / 500 val, velocity (70 x 70) -> 5 shots x 1000 steps x 70 receivers, 100 epochs on Tesla P100-PCIE-16GB. Normalization: minmax.

| model | real params | rel L2 % | late-time % | median % | MAE | RMSE | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| FNO | 9,467,521 | **5.79** | 13.8 | 4.84 | 0.0612 | 0.1017 | 9.3 |
| GNO | 174,113 | **14.66** | 39.9 | 14.12 | 0.1291 | 0.2561 | 242.4 |
| PFNO | 3,411,200 | **23.40** | 62.4 | 20.78 | 0.2483 | 0.3998 | 47.8 |
| DeepONet | 1,468,673 | **48.24** | 110.7 | 46.48 | 0.4446 | 0.8300 | 18.1 |

**late-time %** is relative L2 after the first 20 % of the record, where the reflections live. A zero prediction scores ~100 %, so a value at or above 100 means the model reproduced the direct arrival and nothing else.

Representation scales: PFNO band limit (64 bins) **0.066 %** - a hard floor, bins above the band are zeroed. Time latent (250 pts) 1.747 % for a single-channel resample - a reference, not a bound: the latent is multi-channel.

## curvevel_b

2000 train / 500 val, velocity (70 x 70) -> 5 shots x 1000 steps x 70 receivers, 100 epochs on Tesla P100-PCIE-16GB. Normalization: minmax.

| model | real params | rel L2 % | late-time % | median % | MAE | RMSE | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| FNO | 9,467,521 | **14.84** | 65.2 | 13.97 | 0.1472 | 0.2538 | 9.3 |
| GNO | 174,113 | **19.20** | 67.6 | 18.07 | 0.1651 | 0.3296 | 242.5 |
| PFNO | 3,411,200 | **23.50** | 78.6 | 21.71 | 0.2524 | 0.3981 | 47.7 |
| DeepONet | 1,468,673 | **46.82** | 108.2 | 45.75 | 0.4332 | 0.7978 | 18.2 |

**late-time %** is relative L2 after the first 20 % of the record, where the reflections live. A zero prediction scores ~100 %, so a value at or above 100 means the model reproduced the direct arrival and nothing else.

Representation scales: PFNO band limit (64 bins) **0.098 %** - a hard floor, bins above the band are zeroed. Time latent (250 pts) 1.748 % for a single-channel resample - a reference, not a bound: the latent is multi-channel.

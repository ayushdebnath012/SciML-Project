# Neural-operator forward benchmark

Task: velocity model -> shot gathers, per run. Errors on physical amplitudes; a zero prediction scores ~100 %.

## flatvel_a

2000 train / 500 val, velocity (70 x 70) -> 5 shots x 1000 steps x 70 receivers, 100 epochs on NVIDIA H100 NVL. Normalization: minmax.

| model | real params | rel L2 % | late-time % | median % | MAE | RMSE | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| FNO | 9,467,521 | **3.14** | 10.5 | 2.94 | 0.0266 | 0.0461 | 7.4 |
| GNO | 174,113 | **10.85** | 25.2 | 10.16 | 0.0706 | 0.1574 | 168.5 |
| PFNO | 3,411,200 | **19.41** | 57.3 | 17.86 | 0.1781 | 0.2848 | 38.3 |
| DeepONet | 1,468,673 | **53.44** | 122.9 | 52.80 | 0.3766 | 0.7880 | 5.4 |

**late-time %** is relative L2 after the first 20 % of the record, where the reflections live. A zero prediction scores ~100 %, so a value at or above 100 means the model reproduced the direct arrival and nothing else.

Representation scales: PFNO band limit (64 bins) **0.010 %** - a hard floor, bins above the band are zeroed. Time latent (250 pts) 1.742 % for a single-channel resample - a reference, not a bound: the latent is multi-channel.

## curvevel_a

2000 train / 500 val, velocity (70 x 70) -> 5 shots x 1000 steps x 70 receivers, 100 epochs on NVIDIA H100 NVL. Normalization: minmax.

| model | real params | rel L2 % | late-time % | median % | MAE | RMSE | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| FNO | 9,467,521 | **9.25** | 18.3 | 8.61 | 0.0720 | 0.1353 | 4.5 |
| GNO | 174,113 | **14.81** | 34.6 | 12.77 | 0.0948 | 0.2177 | 93.7 |
| PFNO | 3,411,200 | **20.31** | 47.7 | 18.56 | 0.1911 | 0.3001 | 23.0 |
| DeepONet | 1,468,673 | **54.26** | 102.9 | 53.96 | 0.3941 | 0.8063 | 3.3 |

**late-time %** is relative L2 after the first 20 % of the record, where the reflections live. A zero prediction scores ~100 %, so a value at or above 100 means the model reproduced the direct arrival and nothing else.

Representation scales: PFNO band limit (64 bins) **0.053 %** - a hard floor, bins above the band are zeroed. Time latent (250 pts) 1.741 % for a single-channel resample - a reference, not a bound: the latent is multi-channel.

## curvevel_a_matched

2000 train / 500 val, velocity (70 x 70) -> 5 shots x 1000 steps x 70 receivers, 100 epochs on NVIDIA H100 NVL. Normalization: minmax.

| model | real params | rel L2 % | late-time % | median % | MAE | RMSE | s/epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| GNO | 200,955 | **14.90** | 36.3 | 12.64 | 0.0944 | 0.2190 | 207.2 |
| PFNO | 147,200 | **50.10** | 103.9 | 49.38 | 0.4128 | 0.7442 | 29.0 |
| FNO | 165,221 | **52.43** | 70.9 | 51.62 | 0.3013 | 0.7772 | 6.0 |
| DeepONet | 198,315 | **55.89** | 103.8 | 55.72 | 0.4021 | 0.8301 | 2.7 |

**late-time %** is relative L2 after the first 20 % of the record, where the reflections live. A zero prediction scores ~100 %, so a value at or above 100 means the model reproduced the direct arrival and nothing else.

Representation scales: PFNO band limit (64 bins) **0.053 %** - a hard floor, bins above the band are zeroed. Time latent (250 pts) 1.741 % for a single-channel resample - a reference, not a bound: the latent is multi-channel.

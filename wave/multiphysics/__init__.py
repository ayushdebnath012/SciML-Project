"""Multi-physics subsurface dataset: one geology, three forward problems.

    geology.py          latent layered geology -> v, density, sigma, rho maps
    seismic_solver.py   (v, p)  2-D acoustic FD, OpenFWI acquisition geometry
    mt_solver.py        (c, m)  2-D magnetotellurics, TE + TM, quasi-static Maxwell
    dc_solver.py        (r, i)  2.5-D DC resistivity, pole-pole apparent resistivity
    build_dataset.py    CLI: chunked .npy files in the OpenFWI layout + meta.json

Pure NumPy + SciPy so dataset generation runs on any box, GPU or not.
"""

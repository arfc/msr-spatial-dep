# Overview
These scripts are motivated by the work of Shayan Shahbazi [1].
In general, these scripts give a high-level view of the effects of including a more refined spatial mesh on a traditional depletion simulation.
These scripts assume the existence of two neutronicly distinct regions: in-core and ex-core.
Realistically, there would be a power profile in-core, but this is generally ignored for depletion simulations.
The `morty_pde_ode_compare` script has an option to include a sinusoidal source and loss profile in-core, which alters results by roughly <1%.

# `morty_pde_ode_compare.py`
This script offers a comparison between a depletion ODE (in time only) and a depletion PDE (1D in space and in time).

# `parallel_morty_pde.py`
This script includes an example MORTY solve for the MSRE.


[1] https://link.springer.com/article/10.1007/s10967-022-08535-3

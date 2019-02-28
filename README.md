# ir_ldr

The package `ir_ldr` is for measuring the spectral line depth of the APOGEE and WINERED spectra, calculating the line depth ratio (LDR) and finally deriving the effective temperature (T_LDR).

The LDR-Teff relations inside this package are from Jian+19a, Taniguchi+18 and Jian+19b. Please also refer to Fukue+15.

This package relys on `numpy`, `pandas`, `matplotlib` and `scipy.optimize.curvefit`; it is based on python 3.

# Installation

- Clone this repo.
- cd to the first `ir_ldr` file
- `pip install .`
- Then you can import this package in python environment

# Usage

The main functions are listed here:

- `load_yj_linelist`, `load_h_linelist`
- `depth_measure`
- `cal_ldr`
- `LDR2TLDR_WINERED`, `LDR2TLDR_APOGEE`
- `l_type_classify_WINERED`

Please refer to the help information of each function for their usage.

# Author

Mingjie Jian (ssaajianmingjie@gmail.com)

PhD student, Department of Astronomy, the University of Tokyo

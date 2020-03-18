# ir_ldr

The package `ir_ldr` is for measuring the spectral line depth of the APOGEE and WINERED spectra, calculating the line depth ratio (LDR) and finally deriving the effective temperature (T_LDR).

The LDR-Teff relations inside this package are from Jian+19a, Taniguchi+18 and Jian+19b. Please also refer to Fukue+15.

This package relys on `numpy`, `pandas`, `matplotlib` and `scipy.optimize.curvefit`; it is based on python 3.

# Installation

- Clone this repo.
- cd to the first `ir_ldr` file
- `pip install .`
- Then you can import this package in python environment

# Turorial

The synthetic spectra of a dwarf (Teff=5000 K, logg=4.0 dex and feh=0 dex) and supergiant (Teff=5000 K, logg=2.0 dex and feh=0 dex) are in `ir_ldr/file/` for an example of T_LDR calculation.

```py
# Load the linelist.
linelist = ir_ldr.load_linelist('yj', 'dwarf-j20a')

# Here we use all the orders of synthetic spectra.
for order in [43, 44, 45, 46, 47, 48, 52, 53, 54, 55, 56, 57]:
    # Load the synthetic spectra
    spec = pd.read_csv(ir_ldr.__path__[0] + '/file/example_spectra/dwarf/order{}.txt'.format(order),
                       sep=' +', skiprows=2, engine='python', names=['wav', 'residual'])
    wav = spec['wav'].values
    residual = spec['residual'].values

    # Select the line pairs for a specific order
    linelist_sub = linelist[linelist['order'] == order]
    if len(linelist_sub) == 0:
        continue
    linelist_sub.reset_index(drop=True, inplace=True)

    # Measure the line depth of low(1)- and high(2)-EP line
    d1 = ir_ldr.depth_measure(wav, residual, linelist_sub['linewav1'], suffix=1)
    d2 = ir_ldr.depth_measure(wav, residual, linelist_sub['linewav2'], suffix=2)

    # Calculate the logLDR value.
    lgLDR = ir_ldr.cal_ldr(d1, d2, type='lgLDR')
    # Combine the Dataframes.
    record = ir_ldr.combine_df([linelist_sub, d1, d2, lgLDR])
    if order == 43:
        record_all = record
    else:
        record_all = pd.concat([record_all, record], sort=False)
# Calculate T_LDR
ir_ldr.LDR2TLDR_WINERED(record)
```

# Author

Mingjie Jian (ssaajianmingjie@gmail.com)

PhD student, Department of Astronomy, the University of Tokyo

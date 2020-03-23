# ir_ldr

The package `ir_ldr` is for measuring the spectral line depth of the APOGEE and WINERED spectra, calculating the line depth ratio (LDR) and finally deriving the effective temperature (T_LDR).

The LDR-Teff relations inside this package are from [Jian+19](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1310J/abstract), [Taniguchi+18](https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.4993T/abstract) and Jian+20a (in prep.). Please also refer to [Fukue+15](https://ui.adsabs.harvard.edu/abs/2015ApJ...812...64F/abstract).

This package relys on `numpy`, `pandas`, `matplotlib` and `scipy`; it is based on python 3.

# Installation

`pip install ir_ldr`

# Tutorial

The synthetic spectra of a dwarf star (Teff=5000 K, logg=4.5 dex and feh=0 dex; generated from MOOG) in `ir_ldr/file/dwarf` for an example of T_LDR calculation.

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

    # Measure the line depth of low(1)- and high(2)-EP line.
    # Here the signal to noise ratio for the target star and telluric standard are
    # manually set as 300, but the S_N of synthetic spectra is much higher than that.
    d1 = ir_ldr.depth_measure(wav, residual, linelist_sub['linewav1'], suffix=1, S_N=[300, 300])
    d2 = ir_ldr.depth_measure(wav, residual, linelist_sub['linewav2'], suffix=2, S_N=[300, 300])

    # Calculate the logLDR value.
    lgLDR = ir_ldr.cal_ldr(d1, d2, type='logLDR')
    # Combine the Dataframes of one order.
    record = ir_ldr.combine_df([linelist_sub, d1, d2, lgLDR])

    if order == 43:
        record_all = record
    else:
        record_all = pd.concat([record_all, record], sort=False)

# Calculate T_LDR
LDR = ir_ldr.ldr2tldr_winered_solar(record_all, df_output=True)
```

And the result `(T_LDR, T_LDR_err)` is:
```py
LDR[0:2]
>>> (5009.857201559249, 22.35966233607925)
# Note the T_LDR_err is not an accurate estimation here since the S_N is manually set.
```

# Author

Mingjie Jian (ssaajianmingjie@gmail.com)

PhD student, Department of Astronomy, the University of Tokyo

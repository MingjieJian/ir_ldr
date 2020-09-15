# ir_ldr

The package `ir_ldr` is for measuring the spectral line depth of the APOGEE and WINERED spectra, calculating the line depth ratio (LDR) and finally deriving the effective temperature (T_LDR).

The LDR-Teff relations inside this package are from [Jian+19](https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.1310J/abstract), [Taniguchi+18](https://ui.adsabs.harvard.edu/abs/2018MNRAS.473.4993T/abstract) and [Jian+20a](http://adsabs.harvard.edu/abs/2020arXiv200310641J). Please also refer to [Fukue+15](https://ui.adsabs.harvard.edu/abs/2015ApJ...812...64F/abstract).

This package relys on `numpy`, `pandas`, `matplotlib` and `scipy`; it is based on python 3.

# Installation

`pip install ir_ldr`

# Tutorial

The synthetic spectra of a dwarf star (Teff=5000 K, logg=4.5 dex and feh=0 dex; generated from MOOG) in `ir_ldr/file/dwarf` for an example of T_LDR calculation.

~~~py
import ir_ldr
import pandas as pd

# Load the linelist.
linelist = ir_ldr.load_linelist('yj', 'dwarf-j20a')

# Here we use all the orders of synthetic spectra.
for order in [43, 44, 45, 46, 47, 48, 52, 53, 54, 55, 56, 57]:
    # Load the synthetic spectra
    spec = pd.read_csv(ir_ldr.__path__[0] + '/file/example_spectra/dwarf/order{}.txt'.format(order), sep=' +', skiprows=2, engine='python', names=['wav', 'residual'])
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

record_all.reset_index(drop=True, inplace=True)

# Calculate T_LDR
LDR = ir_ldr.ldr2tldr_winered_solar(record_all, df_output=True)
~~~

And the result `(T_LDR, T_LDR_err)` is:
~~~py
LDR[0:2]
>>> (5009.857201559249, 22.35966233607925)
# Note the T_LDR_err is not an accurate estimation here since the S_N is manually set.
~~~

## Update for the Bayesian approach

The use of Bayesian approach method is similar to the process described above: first measure the line depth and calculate the LDR.
Then refer to the example below:

~~~py

# An example for giant star measurement:
log_LDR = [0.454, -0.076, 0.428, 0.637, -0.166, 0.479, 0.023, 0.147, 0.012, -0.078, -0.135, -0.127, 0.287, 0.334, 0.120, 0.867, np.nan, -0.135, 0.448, 0.788, np.nan, 0.009, -0.247, 0.225, 0.073, -0.126, 0.550, 0.357, 0.126, -0.030, 0.024, 0.603, 0.253, -0.364, 0.133, 0.469, np.nan, 0.227, -0.072, 0.354, 0.134, 0.143, 0.109, 0.226, 0.880, -0.012, 0.068, 0.508, np.nan, 0.216, 0.288, -0.018, 0.002, 0.397, 0.735, 0.791, 0.144, 0.238, 0.409, 0.662, -0.079, 0.330, 0.550, 0.130, 0.223, 0.494, 0.496, 0.064]
log_LDR_err = [0.032, 0.033, 0.038, 0.060, 0.027, 0.035, 0.032, 0.029, 0.023, 0.026, 0.027, 0.028, 0.082, 0.030, 0.020, 0.052, np.nan, 0.027, 0.041, 0.043, np.nan, 0.035, 0.036, 0.021, 0.037, 0.023, 0.038, 0.017, 0.014, 0.031, 0.025, 0.049, 0.047, 0.022, 0.025, 0.056, np.nan, 0.046, 0.015, 0.027, 0.011, 0.013, 0.013, 0.045, 0.132, 0.035, 0.028, 0.045, np.nan, 0.018, 0.024, 0.022, 0.020, 0.018, 0.042, 0.047, 0.024, 0.043, 0.039, 0.072, 0.030, 0.050, 0.052, 0.040, 0.038, 0.075, 0.033, 0.026]

record = ir_ldr.load_linelist('yj', 'giant-j20b')
record['log_LDR'] = lg_LDR
record['log_LDR_err'] = lg_LDR_err
res = ir_ldr.cal_posterior(record, 'giant', plot=False, likelihood_out=False)

~~~

The result should be (3777+11-13) K and (0.08+0.05-0.04) dex. 
When set to `True`, the keyword `plot` plots the contour of likelihood, and `likelihood_out` output the array of likelihood also.

# Author

Mingjie Jian (ssaajianmingjie@gmail.com)

PhD student, Department of Astronomy, the University of Tokyo

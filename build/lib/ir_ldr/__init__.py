from . import private

def load_linelist(band, l_type):

    '''
    Load the YJ-band LDR line list and LDR-Teff relation coefficients.

    Parameters
    ----------
    band : string
        Specify the band of relation set to load. Have to be one of the following: "h" or "yj".
    l_type : string
        Specify the type of relation set to load. If band is "h", then have to be "giant"; if band is "yj", then have to be one of the following: "dwarf-j20a", "giant-t18" or "supergiant-j20a"

    Returns
    ----------
    df : pandas.DataFrame
        DataFrame containing LDR linelist and LDR-Teff relations.

    '''
    l_type_dict = {'dwarf-j20a':'dwarf_j20a', 'giant-t18':'giant_t18', 'supergiant-j20a':'spg_j20a'}
    band = band.lower()

    if band == 'h':
        if l_type == 'giant':
            df = private.pd.read_csv(__path__[0] + '/file/h-ldr/lineratio_giant.csv')
        else:
            raise ValueError('Band or l_type incorrect.')
    elif band == 'yj':
        if l_type in ['dwarf-j20a', 'giant-t18', 'supergiant-j20a']:
            df = private.pd.read_csv(__path__[0] + '/file/yj-ldr/lineratio_all_{}.csv'.format(l_type_dict[l_type]))
        else:
            raise ValueError('Band or l_type incorrect.')
    return df

def lineele2ele(lineelement):

    '''
    Function to output the element name from "lineelement".
    Example: "Fe1" -> "Fe"

    Parameters
    ----------
    lineelement : string
        line element name in the output of load_linelist.

    Returns
    ----------
    lineelement_name : string
        line element name without ionization state.
    '''

    lineelement_name = lineelement.split('1')[0]

    return lineelement_name

def cal_delta_chi(df):

    '''
    Function to calculate Delta_chi of the line pair.

    Parameters
    ----------
    df : pandas.Dataframe
        Output of load_linelist.

    Returns
    ----------
    df : pandas.Dataframe
        Dataframe containing delta chi value.
    '''

    chi_df = private.pd.read_csv(__path__[0] + '/file/chi_table.csv')

    chi1 = private.pd.merge(df, chi_df,
         left_on='lineelement1', right_on='atomic_number', how='left')
    chi1 = chi1.loc[:,'chi']

    chi2 = private.pd.merge(df, chi_df,
         left_on='lineelement2', right_on='atomic_number', how='left')
    chi2 = chi2.loc[:,'chi']

    df = df.assign(chi1=chi1)
    df = df.assign(chi2=chi2)
    df = df.assign(del_chi=chi1 - chi2)

    return df

def depth_measure(wav, flux, line_input, suffix=False, S_N=False, func='parabola', del_wav_lim=0.2, plot=False):
    '''
    Measure the line depth of a spectrum. Provides the Gaussian, parabola and Gaussian-Hermite function for fitting the line. Require signal to noise ratio (SNR) to calculate the error of line depth; if not given, then only the fitting error will be included. There is no error estimation for Guassian-Hermite fitting now.

    Parameters
    ----------
    wav : numpy.array
        The wavelength of the in put spectrum. Should be in rest wavelength scale.

    flux : numpy.array
        The flux of the in put spectrum. Have to be telluric corrected and in the same length with wav.

    line_input : float or list like object
        Linelist to be measured.

    suffix : int or str, optional
        Suffix of columns of the output pandas DataFrame. 1 for low EP line and 2 for high EP line. If set to False, no suffix will be added, but it cannot be used to calculate the LDR in cal_ldr.

    S_N : int, float or list with length 2, optional
        Signal to noise ratio of the spectrum.

    func : string, default 'parabola'
        Function to be used in fitting. Can be 'parabola', 'Gauss' or 'GH' (Gaussian-Hermite).

    del_wav_lim : float, default 0.2
        The limit of measured wavelength and specified wavelength in Angstrom; default is 0.2 AA.

    plot : bool, default false
        To control plot the points used for fitting and the fitted function or not.

    Return
    -------
    depth_measure_pd : pandas.DataFrame, [depth, del_wav, error, flag]
        A DataFrame containing depth of each line in the linelist. Flag 0 means normal, 1 means fitting is ok but no smallest value found in the four pixels used for the fitting; 2 means the fitting is bad and should be rejected. The result will be NaN if the flag is not 0. Flag -1 means the line is outside the spectral range.
    '''

    # Convert the type of line into list (if only one number was input)
    if type(line_input) == int or type(line_input) == float or type(line_input) == private.np.float64:
        line_input = [line_input]

    # Create an empty list to store the result.
    depth_measure_list = []

    # Calculate the error of depth if S_N is provided.
    if type(S_N) in [int, float]:
        d_err = 1 / S_N
    elif type(S_N) == list and len(S_N) == 2:
        d_err = (1/S_N[0]**2 + 1/S_N[1]**2)**0.5
    elif S_N == False:
        pass
    else:
        raise TypeError('The type or length of S_N is incorrect.')

    # Do a loop for all the lines inside linelist.
    for line in line_input:

        # Judge if the line is inside wavelength range.
        if line < min(wav) or line > max(wav):
            depth_measure_list.append([line, private.np.nan, private.np.nan, private.np.nan, -1])
            continue

        # Extract 4 or 5 (9 or 10) pixels near the line wavelength.
        ff_there = private.np.min(private.np.abs(wav - line))
        del_wav = abs(wav[private.np.argsort(private.np.abs(wav - line))[0]] - wav[private.np.argsort(private.np.abs(wav - line))[1]]) / 4
        if func in ['parabola', 'Gauss']:
            pixel_num = 4
        elif func == 'GH':
            pixel_num = 9

        if ff_there < del_wav and func != 'GH':
            sub_wav = wav[private.np.argsort(private.np.abs(wav - line))[0:pixel_num+1]]
            sub_flux = flux[private.np.argsort(private.np.abs(wav - line))[0:pixel_num+1]]
            sub_flux = sub_flux[private.np.argsort(sub_wav)]
            sub_wav = sub_wav[private.np.argsort(sub_wav)]
        else:
            sub_wav = wav[private.np.argsort(private.np.abs(wav - line))[0:pixel_num]]
            sub_flux = flux[private.np.argsort(private.np.abs(wav - line))[0:pixel_num]]
            sub_flux = sub_flux[private.np.argsort(sub_wav)]
            sub_wav = sub_wav[private.np.argsort(sub_wav)]

        # Do the fitting (poly, Gaussian or Gaussian-Hermite)
        if func == 'parabola':
            poly_res = private.curve_fit(private.parabola2_func, sub_wav-line, sub_flux)
            poly_depth = 1 - (poly_res[0][2] - poly_res[0][1]**2/(4*poly_res[0][0]))
            poly_del_wav = - poly_res[0][1] / (2*poly_res[0][0])
            # Calculate the error of minimum.
            a = poly_res[0][0]; b = poly_res[0][1]; c = poly_res[0][2]
            poly_err = ((b**2/(4*a**2))**2*poly_res[1][0,0] + (b/(2*a))**2*poly_res[1][1,1] + poly_res[1][2,2]
                   - b**3/(8*a**3)*poly_res[1][1,0] + b**2/(4*a**2)*poly_res[1][2,0] - b/(2*a)*poly_res[1][2,1])**0.5
            if S_N != False:
                poly_err = d_err
            poly_flag = 0
            if poly_res[0][0] < 0 or poly_depth < 0:
                poly_flag = 2
            elif abs(poly_del_wav) > del_wav_lim:
                poly_flag = 1
            if plot:
                private.plt.scatter(sub_wav, sub_flux, c='red')
                # x = private.np.arange(sub_wav[0], sub_wav[-1],0.001)
                x = private.np.arange(line-1.5, line+1.5, 0.001)
                y = private.parabola2_func(x-line, a, b, c)
                private.plt.plot(x[y<=1], y[y<=1], c='C1', label='Parabola fitting')

        if func == 'Gauss':
            try:
                gauss_res = private.curve_fit(private.Gauss_func, sub_wav, sub_flux, p0=[0.5, line, 1])
            except:
                gauss_flag = 2
            else:
                gauss_depth = gauss_res[0][0]
                gauss_del_wav = gauss_res[0][1] - line
                if gauss_res[1][0,0] == private.np.inf or gauss_res[1][0,0] == private.np.nan or gauss_res[1][0,0] < 0 or gauss_depth < 0:
                    gauss_flag = 2
                elif abs(gauss_del_wav) > del_wav_lim:
                    gauss_flag = 1
                else:
                    gauss_err = (gauss_res[1][0,0])**0.5
                    if S_N != False:
                        gauss_err = d_err
                    gauss_flag = 0
                    if plot:
                        x = private.np.arange(line-1.5, line+1.5, 0.001)
                        y = private.Gauss_func(x, gauss_res[0][0], gauss_res[0][1], gauss_res[0][2])
                        private.plt.plot(x, y, c='C2', label='Gaussian fitting')

        if func == 'GH':
            try:
                GH_res = private.curve_fit(private.GH_func, sub_wav, sub_flux, p0=[private.np.min(sub_flux), line, 0.8, 0, 0])
                GH_res = GH_res[0]
            except:
                GH_flag = 2
            else:
                # x = private.np.arange(private.np.min(sub_wav), private.np.max(sub_wav), 0.01)
                x = private.np.arange(line-1.5, line+1.5, 0.001)
                y = []
                for i in x:
                    y.append(private.GH_func(i, GH_res[0], GH_res[1], GH_res[2], GH_res[3], GH_res[4]))
                GH_depth = 1 - min(y)
                GH_del_wav = x[private.np.argmin(y)] - line
                GH_err = d_err
                GH_flag = 0

                if plot:
                    private.plt.scatter(sub_wav, sub_flux, c='C4')
                    private.plt.plot(x, y, c='C0', label='Gauss-Hermit fitting')

        if func == 'parabola':
            depth_measure_list.append([line, poly_depth, poly_del_wav, poly_err, poly_flag])
        elif func == 'Gauss':
            depth_measure_list.append([line, gauss_depth, gauss_del_wav, gauss_err, gauss_flag])
        elif func == 'GH':
            depth_measure_list.append([line, GH_depth, GH_del_wav, GH_err, GH_flag])

    # Store the result into pd.Dataframe
    if suffix == False:
        depth_measure_pd = private.pd.DataFrame(depth_measure_list, columns=['line_wav', 'depth', 'del_wav', 'depth_err', 'flag'])
    else:
        suffix = str(suffix)
        depth_measure_pd = private.pd.DataFrame(depth_measure_list, columns=['line_wav', 'depth'+suffix, 'del_wav'+suffix, 'depth_err'+suffix, 'flag'+suffix])
    depth_measure_pd.set_index('line_wav', inplace=True)

    return depth_measure_pd

def cal_ldr(depth_pd_1, depth_pd_2, type='LDR', flag=True):

    '''
    Function to calculate LDR or lg(LDR) values.

    Parameters
    ----------
    depth_pd_1 : pandas.DataFrame
        The depth measurement (output) of depth_measure with suffix 1. They act as divisors in LDR.

    depth_pd_2 : pandas.DataFrame
        The depth measurement (output) of depth_measure with suffix 2. They act as dividends in LDR.

    type : str, default 'LDR'
        The type of LDR to be calculated. Have to be 'ldr' or 'logldr' (log10).

    flag : bool, default True
        When True the logLDR and error values of the line pairs with flag1 or flag2 not equal 0 will be set as NaN.

    Return
    ----------
    LDR_pd : pandas.DataFrame
        A DataFrame containing the LDR/logLDR values and their errors.
    '''

    d1 = depth_pd_1['depth1'].values
    d2 = depth_pd_2['depth2'].values
    d1_err = depth_pd_1['depth_err1'].values
    d2_err = depth_pd_2['depth_err2'].values

    LDR = d1 / d2
    err_LDR = (d1_err**2/d2**2 + d2_err**2 * d1**2/d2**4)**0.5

    if type.lower() == 'ldr':
        LDR_pd = private.np.column_stack([LDR, err_LDR])
        LDR_pd = private.pd.DataFrame(LDR_pd, columns=['LDR', 'LDR_error'])
    elif type.lower() == 'logldr':
        LDR[LDR < 0] = private.np.nan
        logLDR = private.np.log10(LDR)
        err_logLDR = 1/(LDR*private.np.log(10))*err_LDR
        LDR_pd = private.np.column_stack([logLDR, err_logLDR])
        LDR_pd = private.pd.DataFrame(LDR_pd, columns=['logLDR', 'logLDR_error'])
    if flag:
        LDR_pd.at[(depth_pd_1['flag1'] != 0).values | (depth_pd_2['flag2'] != 0).values,:] = private.np.nan
    return LDR_pd

def combine_df(df_list, remove_line_wav=True):

    '''
    Function to combine the DataFrame of linelist, depth measurement and LDR. Set remove_line_wav to False to keep the line_wav in each DataFrame.

    Parameters
    ----------
    df_list : list, contain DataFrame to be combined.

    remove_line_wav : bool, optional
        Whether to remove the "line_wav" column or not.

    Return
    ----------
    output_df : pandas.DataFrame
        A DataFrame containing the combined values.
    '''
    for i in range(len(df_list)):
        if df_list[i].index[0] != 0:
            df_list[i] = df_list[i].reset_index()

    output_df = private.pd.concat(df_list, axis=1)
    if remove_line_wav:
        output_df.drop('line_wav', axis=1, inplace=True)
    return output_df

def ldr2tldr_apogee(df, metal_term=False, df_output=False, fe_h=0, fe_h_err=False, abun=False, abun_err=False):

    '''
    Function to calculate the temperature derived by each H-band LDR-Teff relation (T_LDRi) and their weighted mean (T_LDR).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame of the output of combine_df. Must contain linelist information from load_linelist and LDR information.

    metal_term : bool, optional
        Choose which set of relations (without or with metallicity/abundance terms) to be used. If set to True, then please note the setting of fe_h, fe_h_err, abun and abun_err. These two set of relations correspond to Table2 (without metal-terms) and Table3 (with metal-terms) of Jian+19.

    df_output : bool, optional
        Set to True to output the DataFrame containing T_LDRi.

    fe_h : float, optional
        Metallicity value of equation 2 in Jian+19.

    fe_h_err : float, optional
        The error of the metallicity value. If not specify, will assume no error in metallicity.

    abun : list like, default 0 for all used line pairs
        Abundance value of equation 2 in Jian+19. Have to be in the same length of the rows in df. If not specify, will assume 0 for all the used line pairs.

    abun_err : list like, optional
        The error of the abundance value. Use 0 if assume no error in abundance. If not specify, will assume no error in abundance.

    Returns
    --------
    T_LDR : float
        The weighted averaged temperature derived from LDR relations.

    T_LDR_err : float
        The error of T_LDR

    df : pandas.DataFrame, optional
        The DataFrame containing T_LDRi.
    '''

    # Calculate T_LDRi
    LDR0 = df['LDR']-df['r0']
    if metal_term == False:
        df['T_LDRi'] = (LDR0)**3*df['ap_wom'] + (LDR0)**2*df['a_wom'] + (LDR0)*df['b_wom'] + df['c_wom']
        T_err_r = df['LDR_error'] * (3*df['ap_wom']*(LDR0)**2 + 2*df['a_wom']*(LDR0) + df['b_wom'])
        df['T_LDRi_error'] = (T_err_r**2 + df['sigma_wm']**2)**0.5
    else:
        if abun == False:
            abun = private.np.array([0]*len(df))
        elif len(df) != len(abun):
            raise IndexError("Please specify abun and the length of abun have to be the same as df.")
        else:
            abun = private.np.array(abun)
        df['T_LDRi'] = (LDR0)**3*df['ap_wm'] + (LDR0)**2*df['a_wm'] + (LDR0)*df['b_wm'] + df['c_wm'] + df['d_wm']*fe_h + df['e_wm']*fe_h*LDR0 + df['f_wm']*private.np.array(abun)
        T_err_r = df['LDR_error'] * (3*df['ap_wm']*(LDR0)**2 + 2*df['a_wm']*(LDR0) + df['b_wm'])
        if fe_h_err == False:
            T_err_fe_h = 0
        else:
            T_err_fe_h = fe_h_err * (df['d_wm'] + df['e_wm']*LDR0)

        if abun_err == False:
            T_err_abun = 0
        elif len(df) != len(abun):
            raise IndexError("The length of abun_err have to be the same as df.")
        else:
            T_err_abun = private.np.array(abun_err) * df['f_wm']
        df['T_LDRi_error'] = (T_err_r**2 + T_err_fe_h**2 + T_err_abun**2 + df['sigma_wm']**2)**0.5

    # Calculate T_LDR
    pointer = ~private.np.isnan(df['T_LDRi'])
    if len(df[pointer]['T_LDRi']) <= 0:
        T_LDR, T_LDR_err = private.np.nan, private.np.nan
    else:
        weights = 1/df[pointer]['T_LDRi_error']**2
        T_LDR = private.np.average(df[pointer]['T_LDRi'], weights=weights)
        T_LDR_err = (private.np.sum(weights*(df[pointer]['T_LDRi']-T_LDR)**2) / (len(df)-1) / private.np.sum(weights))**0.5

    if df_output:
        return T_LDR, T_LDR_err, df
    else:
        return T_LDR, T_LDR_err

def ldr2tldr_winered_solar(df, sigma_clip=0, df_output=False):

    '''
    Function to calculate the temperature derived by each YJ-band LDR-Teff relation (T_LDRi) and their weighted mean (T_LDR).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame of the output of combine_df. Must contain linelist information from load_linelist and LDR information.

    sigma_clip : float, optional
        Sigma clipping criteria; the T_LDRi not within sigma_clip*sigma of the mean T_LDR will be excluded.
        If sigma_clip = 0 then no clipping is done.

    df_output : bool, optional
        Set to True to output the DataFrame containing T_LDRi.

    Returns
    --------
    T_LDR : float
        The weighted averaged temperature derived from LDR relations.

    T_LDR_err : float
        The error of T_LDR.

    df : pandas.DataFrame, optional
        The DataFrame containing T_LDRi.
    '''

    # Calculate the T_LDR and _TLDR_error. For T18 line set only std_res is
    # used, while for others the confidence interval is calculated.
    df['T_LDRi'] = (df['logLDR']) * df['slope'] + df['intercept']
    T_err_r = df['logLDR_error'] * df['slope']
    try:
        T_err_fit = private.t.ppf(1-0.32/2, df['Npoints']-2) * df['std_res'] * (1 + 1 / df['Npoints'] + (df['logLDR']-df['mean_logLDR'])**2 / (df['Npoints']-1) / df['std_logLDR']**2)**0.5
        df['T_LDRi_error'] = (T_err_r**2 + T_err_fit**2)**0.5
    except KeyError:
        df['T_LDRi_error'] = (T_err_r**2 + df['std_res']**2)**0.5

    # Exclude those outside the logLDR range.
    try:
        pointer = (df['logLDR'] > df['max_logLDR']) | (df['logLDR'] < df['min_logLDR'])
        df.at[pointer, 'T_LDRi'] = private.np.nan
        df.at[pointer, 'T_LDRi_error'] = private.np.nan
    except KeyError:
        pass

    pointer = ~private.np.isnan(df['T_LDRi'])
    if len(df[pointer]) == 0:
        T_LDR, T_LDR_err = private.np.nan, private.np.nan
    else:
        weights = 1/df[pointer]['T_LDRi_error']**2
        T_LDR = private.np.average(df[pointer]['T_LDRi'], weights=weights)

        # Do sigma clipping if necessary
        if sigma_clip > 0:
            V1 = private.np.sum(weights)
            V2 = private.np.sum(weights**2)
            # Unbiased sample std
            dis = private.np.sum(weights * (df[pointer]['T_LDRi'] - T_LDR)**2) / (V1 - V2/V1) / len(df[pointer]['T_LDRi'])
            dis = private.np.sqrt(dis)
            # dis = private.np.std(df[pointer]['T_LDRi'])
            pointer = pointer & (df['T_LDRi'] < T_LDR + sigma_clip*dis) & (df['T_LDRi'] > T_LDR - sigma_clip*dis)
            weights = 1/df[pointer]['T_LDRi_error']**2
            T_LDR = private.np.average(df[pointer]['T_LDRi'], weights=weights)
        elif sigma_clip < 0:
            raise ValueError('Sigma_clip have to be >= 0.')

        T_LDR_err = 1 / private.np.sum(1 / df[pointer]['T_LDRi_error']**2)**0.5

    if df_output:
        return T_LDR, T_LDR_err, df
    else:
        return T_LDR, T_LDR_err

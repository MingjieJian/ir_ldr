from . import private
from . import tools

def load_linelist(band, l_type):

    '''
    Function to load the YJ-band LDR line list and LDR-Teff relations.

    Parameters
    ----------
    band : string
        Specify the band of relation set to load. Have to be one of the following: "h" or "yj".
    l_type : string
        Specify the type of relation set to load. If band is "h", then have to be "giant"; if band is "yj", then have to be one of the following: "dwarf", "giant-j19", "giant-t18" or "supergiant"

    Returns
    ----------
    df : pandas.DataFrame
        DataFrame containing LDR linelist and LDR-Teff relations.

    '''
    l_type_dict = {'dwarf':'dwarf', 'giant-j19':'giant_j19', 'giant-t18':'giant_t18', 'supergiant':'spg'}
    if band == 'h':
        if l_type == 'giant':
            df = private.pd.read_csv(__path__[0] + '/file/h-ldr/lineratio_giant.csv')
        else:
            raise ValueError('Band or l_type incorrect.')
    elif band == 'yj':
        if l_type in ['dwarf', 'giant-j19', 'giant-t18', 'supergiant']:
            df = private.pd.read_csv(__path__[0] + '/file/yj-ldr/lineratio_all_{}.csv'.format(l_type_dict[l_type]))
        else:
            raise ValueError('Band or l_type incorrect.')
    return df

def lineele2ele(string):

    '''
    Function to output the element name from "lineelement".
    '''

    string = string.split('1')[0]

    return string

def cal_delta_chi(df):

    '''
    Function to calculate Delta_chi of the line pair.
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

def depth_measure(wav, flux, line_input, suffix=False, SNR=False, func='parabola', plot=False):
    '''
    Function to measure the line depth of a spectrum. Provides the Gaussian, parabola and Gaussian-Hermite function for fitting the line. Require signal to noise ratio (SNR) to calculate the error of line depth; if not given, then only the fitting error will be included. There is no error estimation for Guassian-Hermite fitting now.

    Parameters
    ----------
    wav : numpy.array
        The wavelength of the in put spectrum. Should be in rest wavelength scale.

    flux : numpy.array
        The flux of the in put spectrum. Have to be telluric corrected and in the same length with wav.

    line_input : float or list like object
        Linelist to be measured.

    suffix : int or str, optional
        Suffix of columns of the output pandas DataFrame. 1 for low EP line and 2 for high EP line. If set to False, no suffix will be added, but it cannot be used to calculate the LDR in cal_LDR.

    SNR : float, optional
        Signal to noise ratio of the spectrum.

    func : string, default 'parabola'
        Function to be used in fitting. Can be 'parabola', 'Gauss' or 'GH' (Gaussian-Hermite).

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
            if SNR != False:
                # poly_err = (poly_err**2+1/SNR**2/len(sub_wav))**0.5
                poly_err = 1/SNR
            poly_flag = 0
            if poly_res[0][0] < 0 or poly_depth < 0:
                poly_depth = private.np.nan; poly_del_wav = private.np.nan; poly_err = private.np.nan; poly_flag = 2
            elif abs(poly_del_wav) > private.np.max(sub_wav-line):
                poly_depth = private.np.nan; poly_del_wav = private.np.nan; poly_err = private.np.nan; poly_flag = 1
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
                gauss_depth = private.np.nan; gauss_del_wav = private.np.nan; gauss_err = private.np.nan; gauss_flag = 2
            else:
                gauss_depth = gauss_res[0][0]
                gauss_del_wav = abs(gauss_res[0][1] - line)
                if gauss_res[1][0,0] == private.np.inf or gauss_res[1][0,0] == private.np.nan or gauss_res[1][0,0] < 0 or gauss_depth < 0:
                    gauss_err = private.np.nan
                    gauss_flag = 2
                else:
                    gauss_err = (gauss_res[1][0,0])**0.5
                    if SNR != False:
                        # gauss_err = (gauss_err**2+1/SNR**2/len(sub_wav))**0.5
                        gauss_err = 1/SNR
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
                GH_depth = private.np.nan; GH_del_wav = private.np.nan; GH_err = private.np.nan; GH_flag = 2
            else:
                # x = private.np.arange(private.np.min(sub_wav), private.np.max(sub_wav), 0.01)
                x = private.np.arange(line-1.5, line+1.5, 0.001)
                y = []
                for i in x:
                    y.append(private.GH_func(i, GH_res[0], GH_res[1], GH_res[2], GH_res[3], GH_res[4]))
                GH_depth = 1 - min(y)
                GH_del_wav = x[private.np.argmin(y)] - line
                GH_err = 1/SNR
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

def cal_ldr(depth_pd_1, depth_pd_2, type='LDR'):

    '''
    Function to calculate LDR or lg(LDR) values.

    Parameters
    ----------
    depth_pd_1 : pandas.DataFrame
        The depth measurement (output) of depth_measure with suffix 1. They act as divisors in LDR.

    depth_pd_2 : pandas.DataFrame
        The depth measurement (output) of depth_measure with suffix 2. They act as dividends in LDR.

    type : str, optional
        The type of LDR to be calculated. Have to be 'LDR' or 'lgLDR' (log10).

    Return
    ----------
    LDR_pd : pandas.DataFrame
        A DataFrame containing the LDR/lgLDR values and their errors.
    '''

    d1 = depth_pd_1['depth1'].values
    d2 = depth_pd_2['depth2'].values
    d1_err = depth_pd_1['depth_err1'].values
    d2_err = depth_pd_2['depth_err2'].values

    LDR = d1 / d2
    err_LDR = (d1_err**2/d2**2 + d2_err**2 * d1**2/d2**4)**0.5
    lgLDR = private.np.log10(LDR)
    err_lgLDR = 1/(d1/d2*private.np.log(10))*err_LDR

    if type == 'LDR':
        LDR_pd = private.np.column_stack([LDR, err_LDR])
        LDR_pd = private.pd.DataFrame(LDR_pd, columns=['LDR', 'LDR_error'])
    elif type == 'lgLDR':
        LDR_pd = private.np.column_stack([lgLDR, err_lgLDR])
        LDR_pd = private.pd.DataFrame(LDR_pd, columns=['lgLDR', 'lgLDR_error'])
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

def LDR2TLDR_APOGEE(df, metal_term=False, df_output=False, fe_h=0, fe_h_err=False, abun=False, abun_err=False):

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

    abun : list like, optional
        Abundance value of equation 2 in Jian+19. Have to be in the same length of the rows in df. If not specify, will assume no error in metallicity.

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
    T_LDR = private.np.average(df[pointer]['T_LDRi'], weights=df[pointer]['T_LDRi_error'])
    weights = 1/df[pointer]['T_LDRi_error']**2
    T_LDR_err = (private.np.sum(weights*(df[pointer]['T_LDRi']-T_LDR)**2) / (len(df)-1) / private.np.sum(weights))**0.5

    if df_output:
        return T_LDR, T_LDR_err, df
    else:
        return T_LDR, T_LDR_err

def LDR2TLDR_WINERED(df, df_output=False):

    '''
    Function to calculate the temperature derived by each YJ-band LDR-Teff relation (T_LDRi) and their weighted mean (T_LDR).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame of the output of combine_df. Must contain linelist information from load_linelist and LDR information.

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
    df['T_LDRi'] = (df['lgLDR']) * df['slope'] + df['intercept']
    T_err_r = df['lgLDR_error'] * df['slope']
    try:
        T_err_fit = private.t.ppf(1-0.025, df['Npoints']-2) * df['std_res'] * (1 + 1 / df['Npoints'] + (df['lgLDR']-df['mean_lgLDR'])**2 / (df['Npoints']-1) / df['std_lgLDR']**2)**0.5
        df['T_LDRi_error'] = (T_err_r**2 + T_err_fit**2)**0.5
    except KeyError:
        df['T_LDRi_error'] = (T_err_r**2 + df['std_res']**2)**0.5

    # Exclude those outside the lgLDR range.
    try:
        pointer = (df['lgLDR'] > df['max_lgLDR']) | (df['lgLDR'] < df['min_lgLDR'])
        df.at[pointer, 'T_LDRi'] = 0
    except KeyError:
        pass

    pointer = ~private.np.isnan(df['T_LDRi'])
    if len(df[pointer]) == 0 and df_output:
        return private.np.nan, private.np.nan, df
    T_LDR = private.np.average(df[pointer]['T_LDRi'], weights=df[pointer]['T_LDRi_error'])
    weights = 1/df[pointer]['T_LDRi_error']**2
    T_LDR_err = 1 / private.np.sum(1 / df[pointer]['T_LDRi_error']**2)**0.5

    if df_output:
        return T_LDR, T_LDR_err, df
    else:
        return T_LDR, T_LDR_err

def Teff2LDR(df, teff):
    '''
    Calculate LDR from a given Teff.
    '''
    LDR_T = (teff - df['intercept']) / df['slope']
    df = df.assign(lgLDR_T=LDR_T)
    return df

def return_min(reduc_list):

    '''
    Function to find the location of minimum and second minimum of a list.
    '''

    reduc_list_use = private.copy.copy(reduc_list)
    min1 = private.np.argmin(reduc_list_use)
    reduc_list_use[min1] = private.np.inf
    min2 = private.np.argmin(reduc_list_use)
    return [min1, min2]

def _l_type_classify_WINERED(spectra_dict, SNR_dict, df_output=False):

    '''
    (Deprecated)
    Function to classify stars with WINERED spectra into dwarf, giant and supergiant. At least the spectra of order 54 or 56 have to be provided.

    Parameters
    ----------
    spectra_dict : dict
        A set of spectras grouped into dict format using orders to indicate their keys.
        Example: {43:[wav1, flux1], 43:[wav2, flux2], ...}; wav and flux should be numpy.array.

    SNR_dict : dict
        A dict containing SNR of the spectra, use orders to indicate their keys. Should have the same length with spectra_dict.

    df_output : bool, optional
        Set to True to output the DataFrame containing T_LDRi of the selected type.

    Returns
    -----------
    type : str
        The type among dwarf, giant and supergiant selected.

    T_LDR : float
        The weighted averaged temperature derived from the set of selected LDR relations .

    T_LDR_err : float
        The error of T_LDR.

    df : pandas.DataFrame, optional
        The DataFrame containing T_LDRi of the selected set.
    '''

    raise NotImplementedError("This function is deprecated, please do not use it.")
    l_type_dict = {0:'dwarf', 1:'giant', 2:'supergiant'}

    order_list = list(range(43,49)) + list(range(52,58))
    order_list = order_list[::-1]

    T_LDR_all = []
    T_LDR_error_all = []
    n_all = []
    redu_chi2_all = []
    yj_df_all = []

    for l_type in ['dwarf', 'giant-j19', 'supergiant']:
        yj_line_df = load_linelist('yj', l_type)
        yj_line_df = cal_delta_chi(yj_line_df)

        order_list = list(spectra_dict.keys())[::-1]
        for order in order_list:
            yj_linre_df_order = yj_line_df[yj_line_df['order'] == order]
            yj1 = depth_measure(spectra_dict[order][0], spectra_dict[order][1],
                                       yj_line_df[yj_line_df['order'] == order]['linewav1'].values, SNR=SNR_dict[order], suffix=1)
            yj2 = depth_measure(spectra_dict[order][0], spectra_dict[order][1],
                                       yj_line_df[yj_line_df['order'] == order]['linewav2'].values, SNR=SNR_dict[order], suffix=2)
            yj_ldr = cal_ldr(yj1, yj2, type='lgLDR')

            if order == order_list[0]:
                yj_line_df_final = yj_linre_df_order
                yj1_all = yj1
                yj2_all = yj2
                yj_ldr_all = yj_ldr
            else:
                yj_line_df_final = private.pd.concat([yj_line_df_final, yj_linre_df_order])
                yj1_all = private.pd.concat([yj1_all, yj1])
                yj2_all = private.pd.concat([yj2_all, yj2])
                yj_ldr_all = private.pd.concat([yj_ldr_all, yj_ldr])

        yj_ldr_all.reset_index(drop=True, inplace=True)
        yj_df = combine_df([yj_line_df_final, yj1_all, yj2_all, yj_ldr_all])
        # if renew:
        #     yj_df.loc[yj_df['del_chi'] < 0.9, ['lgLDR', 'lgLDR_error']] = private.np.nan
        T_LDR, T_LDR_error, yj_df = LDR2TLDR_WINERED(yj_df, df_output=True)

        yj_df = Teff2LDR(yj_df, T_LDR)
        pointer = ~private.np.isnan(yj_df['lgLDR'])
        if len(yj_df[pointer]) <= 1:
            return 'unknown', private.np.nan, private.np.nan
        redu_chi2 = private.np.average((yj_df[pointer]['lgLDR']-yj_df[pointer]['lgLDR_T'])**2 * 1/yj_df[pointer]['lgLDR_error']**2)

        redu_chi2_all.append(redu_chi2)
        n_all.append(len(yj_df[pointer]))
        T_LDR_all.append(T_LDR)
        T_LDR_error_all.append(T_LDR_error)
        yj_df_all.append(yj_df)
    # print(redu_chi2_all, n_all)

    # Run the F test
    min_list = return_min(redu_chi2_all)
    p_value = private.f.cdf(redu_chi2_all[min_list[0]]/redu_chi2_all[min_list[1]], n_all[min_list[0]]-1, n_all[min_list[1]]-1)
    # print(p_value)

    if min_list[0] == 0 and p_value <= 0.05:
        return l_type_dict[min_list[0]], T_LDR_all[min_list[0]], T_LDR_error_all[min_list[0]]
    elif min_list[0] == 0 and p_value > 0.05:
        return 'unknown', private.np.nan, private.np.nan

    if private.np.argmin(redu_chi2_all) == 0 and df_output:
        return 'dwarf', T_LDR_all[0], T_LDR_error_all[0], yj_df_all[0]
    elif private.np.argmin(redu_chi2_all) == 0:
        return 'dwarf', T_LDR_all[0], T_LDR_error_all[0]

    if not(56 in spectra_dict.keys()) and not(54 in spectra_dict.keys()):
        raise IndexError("Order 56 and 54 not included in spectra_dict, cannot judge l_type between giant and supergiant.")
    class_span = {56:[10036.653, [10043.3, 10052.8]],
                  54:[[10458, 10461.5]]}
    depth_giant = []
    for order in [56, 54]:
        try:
            wav = spectra_dict[order][0]
        except KeyError:
            continue
        flux = spectra_dict[order][1]
        for line in class_span[order]:
            if type(line) != list:
                depth_giant.append(depth_measure(wav, flux, line, suffix=1)['depth1'].values[0])
            else:
                EW = private.np.sum(1-flux[(wav > line[0]) & (wav < line[1])]) * (private.np.min(wav[wav>line[0]]) - private.np.max(wav[wav<line[0]]))
                depth_giant.append(EW)
    count = 0
    cut_dict = {0:0.3, 1:0.5, 2:0.2}
    for j in range(len(depth_giant)):
        if private.np.isnan(depth_giant[j]):
            count += 0
        elif depth_giant[j] > cut_dict[j]:
            count += 1
        elif depth_giant[j] <= cut_dict[j]:
            count -= 1
    if count > 0 and df_output:
        return 'supergiant', T_LDR_all[2], T_LDR_error_all[2], yj_df_all[2]
    elif count > 0:
        return 'supergiant', T_LDR_all[2], T_LDR_error_all[2]
    elif count < 0 and df_output:
        return 'giant', T_LDR_all[1], T_LDR_error_all[1], yj_df_all[1]
    elif count < 0:
        return 'giant', T_LDR_all[1], T_LDR_error_all[1]
    else:
        return 'unknown', private.np.nan, private.np.nan

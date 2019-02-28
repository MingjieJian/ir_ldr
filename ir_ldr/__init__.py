from . import private
from . import tools

def load_yj_linelist(l_type):
    l_type_dict = {'dwarf':'dwarf', 'giant':'giant', 'supergiant':'spg'}
    df = private.pd.read_csv(__path__[0] + '/file/yj-ldr/lineratio_all_{}.csv'.format(l_type_dict[l_type]))
    return df

def load_h_linelist():
    df = private.pd.read_csv(__path__[0] + '/file/h-ldr/lineratio_giant.csv')
    return df

def depth_measure(wav, flux, line_input, suffix, SNR=False, func='parabola', plot=False, plot_shift=0):
    '''
    Function to measure the line depth of a spectrum. Provides the Gaussian, parabola and Gaussian-Hermite function for fitting the line, the depth as well as the wavelength of the minimum. Require SNR to calculate the error of line depth; if not given, then only the fitting error will be included. There is no error estimation for Guassian-Hermite fitting now.

    Parameters
    ----------
    wav : numpy.array
        The wavelength of the in put spectrum. Should be in rest wavelength scale and telluric corrected.

    flux : numpy.array
        The flux of the in put spectrum. Have to be in the same length with wav.

    line_input : float or list like object
        Linelist to be measured.

    suffix : int or str
        Suffix of columns of the output pandas DataFrame. 1 for low EP line and 2 for high EP line. If set to False, no suffix will be added, but it cannot be used to calculate the LDR in cal_LDR.

    SNR : float, optional
        Signal to noise ratio of the spectrum.

    func : string, default 'parabola'
        Function to be used in fitting. Can be 'parabola', 'Gauss' or 'GH' (Gaussian-Hermite).

    plot : bool, default false
        To control plot the points used for fitting and the fitted function or not.

    Returns
    -------
    depth : list, [depth, del_wav, error, flag]
        The depth of each line in the linelist. Flag 0 means normal, 1 means fitting is ok but no smallest value found in the four pixels used for the fitting; 2 means the fitting is bad and should be rejected. The result will be NaN if the flag is not 0. Flag -1 means the line is outside the spectral range.
    '''

    # Convert the type of line into list (if only one number was input)
    if type(line_input) == int or type(line_input) == float:
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
                poly_err = (poly_err**2+1/SNR**2/len(sub_wav))**0.5
            poly_flag = 0
            if poly_res[0][0] < 0:
                poly_depth = private.np.nan; poly_del_wav = private.np.nan; poly_err = private.np.nan; poly_flag = 2
            elif abs(poly_del_wav) > private.np.max(sub_wav-line):
                poly_depth = private.np.nan; poly_del_wav = private.np.nan; poly_err = private.np.nan; poly_flag = 1
            if plot:
                private.plt.scatter(sub_wav, sub_flux+plot_shift, c='red')
                # x = private.np.arange(sub_wav[0], sub_wav[-1],0.001)
                x = private.np.arange(line-1.5, line+1.5, 0.001)
                y = private.parabola2_func(x-line, a, b, c)+plot_shift
                private.plt.plot(x[y<=1], y[y<=1], c='C1', label='Parabola fitting')

        if func == 'Gauss':
            try:
                gauss_res = private.curve_fit(private.Gauss_func, sub_wav, sub_flux, p0=[0.5, line, 1])
            except:
                gauss_depth = private.np.nan; gauss_del_wav = private.np.nan; gauss_err = private.np.nan; gauss_flag = 2
            else:
                gauss_depth = gauss_res[0][0]
                gauss_del_wav = abs(gauss_res[0][1] - line)
                if gauss_res[1][0,0] == private.np.inf or gauss_res[1][0,0] == private.np.nan or gauss_res[1][0,0] < 0:
                    gauss_err = private.np.nan
                    gauss_flag = 2
                else:
                    gauss_err = (gauss_res[1][0,0])**0.5
                    if SNR != False:
                        gauss_err = (gauss_err**2+1/SNR**2/len(sub_wav))**0.5
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
                GH_err = -99
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

    df_list : list, contain DataFrame to be combined.

    remove_line_wav : bool, optional
        Whether to remove the "line_wav" column or not.
    '''
    for i in range(len(df_list)):
        if df_list[i].index[0] != 0:
            df_list[i] = df_list[i].reset_index()

    output_df = private.pd.concat(df_list, axis=1)
    if remove_line_wav:
        output_df.drop('line_wav', axis=1, inplace=True)
    return output_df

def LDR2TLDR_APOGEE(df, metal_term=False, df_output=False, fe_h=0, fe_h_err=False, abun=False, abun_err=False):

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
        print(T_err_abun)
        df['T_LDRi_error'] = (T_err_r**2 + T_err_fe_h**2 + T_err_abun**2 + df['sigma_wm']**2)**0.5
        print(df['T_LDRi_error'])

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
    df['T_LDRi'] = (df['lgLDR']) * df['slope'] + df['intercept']
    T_err_r = df['lgLDR_error'] * df['slope']
    df['T_LDRi_error'] = (T_err_r**2 + df['weighted_resid']**2)**0.5

    pointer = ~private.np.isnan(df['T_LDRi'])
    T_LDR = private.np.average(df[pointer]['T_LDRi'], weights=df[pointer]['T_LDRi_error'])
    weights = 1/df[pointer]['T_LDRi_error']**2
    T_LDR_err = (private.np.sum(weights*(df[pointer]['T_LDRi']-T_LDR)**2) / (len(df)-1) / private.np.sum(weights))**0.5

    if df_output:
        return T_LDR, T_LDR_err, df
    else:
        return T_LDR, T_LDR_err

def l_type_classify_WINERED(spectra_dict, SNR_dict, df_output=False):

    order_list = list(range(43,49)) + list(range(52,58))
    order_list = order_list[::-1]

    T_LDR_scatter = []
    T_LDR_all = []
    T_LDR_error_all = []
    yj_df_all = []

    for l_type in ['dwarf', 'giant', 'supergiant']:
        yj_line_pd = load_yj_linelist(l_type)

        for order in order_list:
            yj1 = depth_measure(spectra_dict[order][0], spectra_dict[order][1],
                                       yj_line_pd[yj_line_pd['order'] == order]['linewav1'].values, SNR=SNR_dict[order], suffix=1)
            yj2 = depth_measure(spectra_dict[order][0], spectra_dict[order][1],
                                       yj_line_pd[yj_line_pd['order'] == order]['linewav2'].values, SNR=SNR_dict[order], suffix=2)
            yj_ldr = cal_ldr(yj1, yj2, type='lgLDR')

            if order == 57:
                yj1_all = yj1
                yj2_all = yj2
                yj_ldr_all = yj_ldr
            else:
                yj1_all = private.pd.concat([yj1_all, yj1])
                yj2_all = private.pd.concat([yj2_all, yj2])
                yj_ldr_all = private.pd.concat([yj_ldr_all, yj_ldr])

        yj_ldr_all.reset_index(drop=True, inplace=True)
        yj_df = combine_df([yj_line_pd, yj1_all, yj2_all, yj_ldr_all])
        T_LDR, T_LDR_error, yj_df = LDR2TLDR_WINERED(yj_df, df_output=True)
        T_LDR_scatter.append(private.np.std(yj_df['T_LDRi']))
        T_LDR_all.append(T_LDR)
        T_LDR_error_all.append(T_LDR_error)
        yj_df_all.append(yj_df)
    if private.np.argmin(T_LDR_scatter) == 0 and df_output:
        return 'dwarf', T_LDR_all[0], T_LDR_error_all[0], yj_df_all[0]
    elif private.np.argmin(T_LDR_scatter) == 0:
        return 'dwarf', T_LDR_all[0], T_LDR_error_all[0]

    class_span = {56:[10036.7, [10043.3, 10052.8]],
                  54:[[10458, 10461.5]]}
    depth_giant = []
    for order in [56, 54]:
        wav = spectra_dict[order][0]
        flux = spectra_dict[order][1]
        for line in class_span[order]:
            if type(line) != list:
                depth_giant.append(depth_measure(wav, flux, line, suffix=1)['depth1'].values[0])
            else:
                EW = private.np.sum(1-flux[(wav > line[0]) & (wav < line[1])]) * (private.np.min(wav[wav>line[0]]) - private.np.max(wav[wav<line[0]]))
                depth_giant.append(EW)
    count = 0
    cut_dict = {0:0.3, 1:0.5, 2:0.2}
    for j in range(3):
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

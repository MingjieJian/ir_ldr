from . import private

def find_continuum(wav, flux, thres=100, flat_pixels=3, mean_thres=0.03):

    '''
    Function to find the continuum range of a spectrum. The continuum is defined when the (maximum - minimum) of a subset of the spectrum with a length of flat_pixels (default 3) smaller than 1/thres (default 100) and the mean depth of those pixels are smaller than mean_thres (default 0.03).

    Parameters
    ----------
    wav : numpy.array
        The wavelength of the in put spectrum. Should be in rest wavelength scale and telluric corrected.

    flux : numpy.array
        The flux of the in put spectrum. Have to be in the same length with wav.

    thres : float, optional

    flat_pixels : int, optional
        The number of flat pixel

    mean_thres : float, optional
        The threshold of mean flux to be judged as continuum.


    Returns
    -------
    position : list
        A list of position indicating the index of continuum pixels.
    '''

    # Raise error if flag_pixels <= 1 or >= len(wav)
    if flat_pixels <= 1 or flat_pixels >= len(wav):
        raise IndexError("The value fo flat_pixels is too small or too large.")

    spec_range = []
    spec_range_all = []
    spec_value = []
    spec_value_all = []
    position = []
    flat_thres = 1./thres

    for i in private.np.arange(0,len(wav)-flat_pixels-1):
        sub_wav = wav[i:i+flat_pixels+1]
        sub_flux = flux[i:i+flat_pixels+1]
        judge = private.np.ptp(sub_flux)
        spec_range_all.append(judge)
        spec_value_all.append(private.np.mean(sub_flux))
        if judge <= flat_thres and private.np.mean(sub_flux) >= 1 - mean_thres:
            spec_range.append(judge)
            spec_value.append(private.np.mean(sub_flux))
            position.extend(range(i, i+flat_pixels+1))

    position = list(set(position))
    position.sort()
    return position

def return_min(reduc_list):

    '''
    Function to find the location of minimum and second minimum of a list.
    '''

    reduc_list_use = private.copy.copy(reduc_list)
    min1 = private.np.argmin(reduc_list_use)
    reduc_list_use[min1] = private.np.inf
    min2 = private.np.argmin(reduc_list_use)
    return [min1, min2]

def Teff2LDR(df, teff):
    '''
    Calculate LDR from a given Teff.
    '''
    LDR_T = (teff - df['intercept']) / df['slope']
    df = df.assign(lgLDR_T=LDR_T)
    return df

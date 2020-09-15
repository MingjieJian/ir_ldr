import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f, t
import scipy.stats as stats

def Gauss_func(x, a, x0, sigma):
    '''
    Function for defining the Guassian function.
    '''
    res = 1 - a * np.exp(-(x - x0)**2 / (2*sigma**2))
    return res

def parabola2_func(x, a, b, c):
    '''
    Function for defining the second parabola function.
    '''
    return a*x**2 + b*x + c

def GH_func(x, a, x_center, sigma, h2, h4):
    omega = (x-x_center)/sigma
    H0 = 1
    H2 = h2 * (omega**2-1)
    H4 = h4 * (4*omega**4-12*omega**2+3)
    gh = 1 - a * np.exp(-(x-x_center)**2/2/sigma**2) * (H0 + H2 + H4)
    return gh

def derive_fitting_interval(n, std_res, mean_x, std_x, n_para, plot_x, i_type='confidence', conf=0.68):
    '''
    Function to calculate confidence/prediction interval.
    '''
    alpha = 1 - conf
    t = stats.t.ppf(1-alpha/2, n-n_para)
    if i_type == 'confidence':
        interval = t * std_res * (1/n + (plot_x - mean_x)**2 / (n-1) / std_x**2)**0.5
    elif i_type == 'prediction':
        interval = t * std_res * (1 + 1/n + (plot_x - mean_x)**2 / (n-1) / std_x**2)**0.5
    return interval

def derive_fitting_interval_2d(n, std_res, mean_x, std_x, mean_y, std_y, n_para, plot_x, plot_y, i_type='confidence', conf=0.68):
    '''
    Function to calculate confidence/prediction interval.
    '''
    alpha = 1 - conf
    t = stats.t.ppf(1-alpha/2, n-n_para)
    if i_type == 'confidence':
        interval = t * std_res * (1/n + (plot_x - mean_x)**2 / (n-1) / std_x**2 + (plot_y - mean_y)**2 / (n-1) / std_y**2)**0.5
    elif i_type == 'prediction':
        interval = t * std_res * (1 + 1/n + (plot_x - mean_x)**2 / (n-1) / std_x**2 + (plot_y - mean_y)**2 / (n-1) / std_y**2)**0.5
    return interval

def cal_accu_likelihood(likelihood):
    accu_likelihood = []
    for i in range(len(likelihood)-1):
        accu_likelihood.append(np.sum(likelihood[:i+1]))
    accu_likelihood.append(np.sum(likelihood))
    accu_likelihood = np.array(accu_likelihood)
    return accu_likelihood

def credi_interval(likelihood, X, Tldr):
    accu_likelihood = cal_accu_likelihood(likelihood)
    T_low_all = []
    T_high_all = []
    for p_low in np.arange(0.05, 0.3, 0.01):
        T_low = X[np.argmin(np.abs(accu_likelihood-p_low))]
        T_high = X[np.argmin(np.abs(accu_likelihood-(p_low+0.68)))]
        T_low_all.append(T_low)
        T_high_all.append(T_high)
    T_low_all = np.array(T_low_all)
    T_high_all = np.array(T_high_all)
    del_T = T_high_all - T_low_all

    # Find the minimum del_T range and then take the average.
    arg = np.where(del_T == np.min(del_T))
    T_low_final = np.mean(T_low_all[arg]) - Tldr
    T_high_final = np.mean(T_high_all[arg]) - Tldr
    
    return T_low_final, T_high_final
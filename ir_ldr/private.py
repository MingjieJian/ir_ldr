import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import f, t

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

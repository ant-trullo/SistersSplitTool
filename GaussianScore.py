"""This function calculates the Gaussian fitting and its score on a 3D intnesity matrix"""

import numpy as np
from scipy.optimize import curve_fit
# from scipy.stats import chisquare


def gaus_func(x_dt, A, mz, mx, my, sz, sx, sy, bkg):
    """Definition f the function"""
    return A * np.exp(-(x_dt[0] - mz) ** 2 / (sz ** 2) - (x_dt[1] - mx) ** 2 / (sx ** 2) - (x_dt[2] - my) ** 2 / (sy ** 2)) + bkg


class GaussianScore:
    """Only function, does all the job"""
    def __init__(self, raw_spot):

        zlen, xlen, ylen     =  raw_spot.shape                                                                 # matrix size
        [z_mx, x_mx, y_mx]   =  np.where(raw_spot == raw_spot.max())                                           # coordinateof the highest point for the fitting initialization
        initial_guess        =  [raw_spot.max(), z_mx[0], x_mx[0], y_mx[0], 0, 0, 0, 0]                        # find something smarter for the sigma' initialization

        z_lin             =  raw_spot[:, x_mx, y_mx]                                                           # initialization of std values: taken the slice with the maximum value, we take the half number of point above the avarage to roughly estimate
        x_lin             =  raw_spot[z_mx, :, y_mx][0]
        y_lin             =  raw_spot[z_mx, x_mx, :][0]
        initial_guess[4]  =  np.where(z_lin > z_lin.mean())[0].size / 2
        initial_guess[5]  =  np.where(x_lin > x_lin.mean())[0].size / 2
        initial_guess[6]  =  np.where(y_lin > y_lin.mean())[0].size / 2

        z, x, y   =  np.meshgrid(np.arange(zlen), np.arange(xlen), np.arange(ylen), indexing='ij')              # preparing grid of coordinates for the fitting

        size   =  x.shape
        z_1d   =  z.reshape((1, np.prod(size)))
        x_1d   =  x.reshape((1, np.prod(size)))
        y_1d   =  y.reshape((1, np.prod(size)))
        xdata  =  np.vstack((z_1d, x_1d, y_1d))
        # print(initial_guess)
        try:
            popt, pcov  =  curve_fit(gaus_func, xdata, raw_spot.ravel(), p0=initial_guess, bounds=([0, 0, 0, 0, 0, 0, 0, 0], [2 * raw_spot.max(), 1000, 10000, 5000, 500000, 5000000, 5000000, 500000000]), maxfev=3000)    # fitting of the 3D intensity distribution
            fin         =  gaus_func(np.meshgrid(np.arange(zlen), np.arange(xlen), np.arange(ylen), indexing='ij'), *popt)                                     # 3D matrix of theoretical data
            r_sqr       =  1 - np.sum((raw_spot - fin) ** 2) / np.sum((raw_spot - raw_spot.mean()) ** 2)
            self.r_sqr  =  r_sqr
        except ValueError:
            self.r_sqr  =  1
        except RuntimeError:
            self.r_sqr  =  1




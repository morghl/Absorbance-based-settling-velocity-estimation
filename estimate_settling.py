# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:48:19 2024

@author: daniel.espinoza
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import curve_fit
import pandas as pd
from conv_disp_1d_fem import empty_conv_diff_model
import time

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"

def fem_model(t, y, C_inv, K_conv, K_disp):
    
    
    dcdt = C_inv @ - (K_disp - K_conv) @ y
    
    return dcdt

def response_multi(t_data, v1, D1, v2, D2, large_particle_fraction, model, ivp_pars, measured_span = .005, return_all = False):
    
    small_particle_fraction = 1 - large_particle_fraction
    
    sol1 = solve_ivp(lambda t, y: fem_model(t, y, model['C_inv'], v1*model['K_conv'], D1*model['K_disp']), 
                    ivp_pars['t_span'], 
                    ivp_pars['c_init']*large_particle_fraction, 
                    t_eval = t_data,
                    method = 'BDF',
                    )
    
    sol2 = solve_ivp(lambda t, y: fem_model(t, y, model['C_inv'], v2*model['K_conv'], D2*model['K_disp']), 
                    ivp_pars['t_span'], 
                    ivp_pars['c_init']*small_particle_fraction, 
                    t_eval = t_data,
                    method = 'BDF',
                    )
    
    dx = model['L']/(model['n_dofs']-1)
        
    i_point = int(measured_span/dx)
    c1 = sol1.y[i_point, :]
    c2 = sol2.y[i_point, :]
    c_sum = c1 + c2
    
    if return_all:
        return c_sum, c1, c2, sol1, sol2
    else:
        return c_sum

def response_single(t_data, v, D, model, ivp_pars, measured_span = .005):
    
    sol = solve_ivp(lambda t, y: fem_model(t, y, model['C_inv'], v*model['K_conv'], D*model['K_disp']), 
                    ivp_pars['t_span'], 
                    ivp_pars['c_init'], 
                    t_eval = t_data,
                    method = 'BDF',
                    )
    
    dx = model['L']/(model['n_dofs']-1)
        
    i_point = int(measured_span/dx)
    c_measured = sol.y[i_point, :]
    
    
    return c_measured


def estimate_settling(data_dict, cuvette_parameters, plot = True, ivp_pars=None, v_guess=1e-7, D_guess=1e-7, gamma_guess=.5):
    """
    

    Parameters
    ----------
    data_dict : dict
        Dictionary containing information regarding the raw data used for
        estimation. Must contain the following fields:
            t: the time series corresponding to absorbance measurements
            c: the absorbance measurement series
            n_comps: the number of differently sized component groups to model.
                     Currently supports 1 or 2 groups.
    cuvette_parameters : dict
        Dictionary containing information regarding the cuvette. Must contain
        the following fields:
            length: the total length of the cuvette used
            measured point: the point, measured from the top, at which
                            absorbance was measured
            degrees of freedom: number of degrees of freedoms to use in the
                                finite element approximation.
    ivp_pars : dict, optional
        Parameters for the diffential equation solver. Must contain the
        following fields:
            c_init: The initial concentration along the cuvette. Must be of
                    equal length to the number of degrees of freedom in the
                    finite element approximation.
            t_span: The time span for the simulation.
    v_guess : float, optional
        Initial guess for settling velocities. The default is 1e-7.
    D_guess : float, optional
        Initial guess for the dispersion coefficients. The default is 1e-7.
    gamma_guess : float, optional
        Initial guess for the fraction of the distribution when using two
        particle groups. The default is .5.

    Returns
    -------
    res_dict : dict
        Results from the curve fit. Contains the following fields:
            c_sim: The simulated concentration in the measured point over time.
                    If two particle groups were used, represents the sum of the
                    two concentrations at the measured point.
            popt: The optimal set of parameters that solve the curve fitting
                  problem.
            pcov: The parameter covariance.
            MSE: The mean squared error of the model fit.
            R2: The coefficient of determination of the model fit.
        If two particle groups were used, contains the following:
            c1_sim: The simulated concentration of particle group 1.
            c2_sim: The simulated concentration of particle group 2.

    """
    res_dict = {}
    
    t_data = data_dict['t']
    c_data = data_dict['c']
    
    if ivp_pars is None:
        ivp_pars = {}
        # This assumes that the concentration is uniform in the entire domain at t=0
        ivp_pars['c_init'] = np.ones((cuvette_parameters['degrees of freedom'],))*c_data[0]
        ivp_pars['t_span'] = [0, t_data.max()]
    
    start = time.time()
    
    n_comps = data_dict['n_comps']
    
    model = empty_conv_diff_model(cuvette_parameters['length'],
                                  cuvette_parameters['degrees of freedom'])
    
    
    if n_comps == 1:
        popt, pcov = curve_fit(lambda t_data, v, D: response_single(t_data, v, D, model, ivp_pars, measured_span = cuvette_parameters['measured point']),
                                t_data,
                                c_data,
                                p0 = [v_guess, D_guess],
                                bounds = ((0, 0),(np.inf, np.inf)),
                                x_scale = [1e-6, 1e-6]
                                )
        c_sim = response_single(t_data, popt[0], popt[1], model, ivp_pars, measured_span = cuvette_parameters['measured point'])
    elif n_comps > 1:
        if n_comps > 2: 
            print('Too many components: use either 1 or 2. Using 2.')
        popt, pcov = curve_fit(lambda t_data, v1, D1, v2, D2, large_particle_fraction: response_multi(t_data, v1, D1, v2, D2, large_particle_fraction, model, ivp_pars, measured_span = cuvette_parameters['measured point']),
                                t_data,
                                c_data,
                                p0 = [v_guess, D_guess, v_guess, D_guess, gamma_guess],
                                bounds = ((0, 0, 0, 0, 0),(np.inf, np.inf, np.inf, np.inf, 1)),
                                ftol = 1e-10,
                                x_scale = [1e-6, 1e-6, 1e-6, 1e-6, 1]
                                )
        c_sim, c1, c2, sol1, sol2 = response_multi(t_data, *popt, model, ivp_pars, measured_span = cuvette_parameters['measured point'], return_all = True)
        res_dict['c1_sim'] = c1
        res_dict['c2_sim'] = c2
    
    res_dict['pcov'] = pcov
    
    print(f'Calibration complete. Time: {(time.time() - start):.4} seconds.')
    print(f'Optimal parameters: {popt}')
    
    residuals = c_data - c_sim
    
    RSS = np.sum(residuals**2) # Residual sum-of-squares
    
    TSS = np.sum((c_data - np.mean(c_data))**2) # Total sum-of-squares
    
    res_dict['MSE'] = np.mean((c_data - c_sim)**2) # Mean squared error
    R2 = 1 - RSS/TSS
    res_dict['R2'] = R2 # Coefficient of determination
    
    res_dict['c_sim'] = c_sim
    res_dict['popt'] = popt
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t_data/60/60, c_data, '-.', label = 'Experimental data')
        if n_comps > 1:
            ax.plot(t_data/60/60, c1, lw = 1, ls = '--', label = f'Particle group 1, v = {popt[0]:.3} [m/s], D = {popt[1]:.3} [m$^2$/s]', alpha=.5)
            ax.plot(t_data/60/60, c2, lw = 1, ls = '--', label = f'Particle group 2, v = {popt[2]:.3} [m/s], D = {popt[3]:.3} [m$^2$/s]', alpha=.5)
            ax.plot(t_data/60/60, c_sim, label = f'Concentration sum, $\gamma$ = {popt[4]:.3}', color = 'k', alpha=.5)
        else:
            ax.plot(t_data/60/60, c_sim, label = 'Concentration', color = 'k', alpha=.5)
        ax.set_ylabel('Concentration [g/l]')
        ax.set_xlabel('Time [h]')
        ax.set_title(f'R$^2$ = {R2:.4}')
        ax.legend()
        # fig.set_size_inches(4.8*2.5, 6.4*1.5)
        
        fig.tight_layout()
    
    return res_dict

if __name__ == '__main__':
#%% Load data
    
    sheet_names = ['Zon1_processed',
                    'Zon2_processed',
                    'Zon3_processed',
                    'Zon4_processed',
                    'Zon5_processed',
                    'Zon6_processed'
                    ]
    
    n_comps = 2
    
    df = pd.read_excel('data/Sedimenteringshastigheter.xlsx', sheet_name = sheet_names)
    
    exp_data = {}
    
    for key in df:
        
        t_end = 24*60*60
        
        t_raw = df[key]['t (s)'].to_numpy(dtype = 'float64')
        c_raw = df[key]['c (g/l)'].to_numpy(dtype = 'float64')
        
        c = c_raw[t_raw <= t_end]
        t = t_raw[t_raw <= t_end]
        
        exp_data[key[:4]] = {'t': t,
                             'c': c,
                             'n_comps': n_comps
                             }
    L = .037 # mm cuvette
    n_dofs = 100 # Number of nodes
    cuvette_parameters = {'length': L,
                          'degrees of freedom': n_dofs,
                          'measured point': .005}
    
    plot = True

    #%% Calibration
    
    results = []
    
    for i, file_name in enumerate(sheet_names):
        
        file_name = file_name[:4]
        res = estimate_settling(exp_data[file_name], cuvette_parameters, plot=plot, v_guess=1e-7, D_guess=1e-7, gamma_guess=.5)
        results.append(res)
    
    
    plt.show(block = False)
    

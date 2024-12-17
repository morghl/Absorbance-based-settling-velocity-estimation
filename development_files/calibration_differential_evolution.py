# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 13:45:30 2023

@author: DanielEspinoza
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import curve_fit, differential_evolution
import pandas as pd
from conv_disp_1d_fem import empty_conv_diff_model
from calibrate_settling_2_comps_5_pars import response
import time

def objective_function(x, y_data, t_data, model, ivp_pars, c_max, plot = False):
    
    v1, D1, v2, D2, large_particle_fraction = x
    
    if plot:
        y_sim, c1, c2, sol1, sol2 = response(t_data, *popt, model, ivp_pars, c_max, measured_span = measured_span, return_all = True)
    else:
        y_sim = response(t_data, v1, D1, v2, D2, large_particle_fraction, model, ivp_pars, c_max, measured_span = [.005], return_all = False)
    
    SE = np.square(y_data - y_sim)
    SSE = np.sum(SE)
    
    if plot:
        residuals = y_data - y_sim
        
        RSS = np.sum(residuals**2) # Residual sum-of-squares
        
        TSS = np.sum((y_data - np.mean(y_data))**2) # Total sum-of-squares
        
        MSE = np.mean((y_data - y_sim)**2)
        R_squared = 1 - RSS/TSS
        
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax.plot(t_data/60/60, y_data, '-.', label = 'Experimental data')
        ax.plot(t_data/60/60, c1, lw = 1, ls = '--', label = f'Particle group 1, v = {popt[0]:.3} [m/s], D = {popt[1]:.3} [m$^2$/s]')
        ax.plot(t_data/60/60, c2, lw = 1, ls = '--', label = f'Particle group 2, v = {popt[2]:.3} [m/s], D = {popt[3]:.3} [m$^2$/s]')
        ax.plot(t_data/60/60, y_sim, label = f'Concentration sum, $\gamma$ = {popt[4]:.3}', color = 'k')
        ax.set_ylabel('Concentration [g/l]')
        ax.set_xlabel('Time [h]')
        ax.set_title(f'Zone {i+1}, R$^2$ = {R_squared:.4}')
        ax.legend()
        
        fig.set_size_inches(6.4, 4.8)
        
        fig.tight_layout()
        
    return SSE

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"


if __name__ == '__main__':
#%% Load data
    
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    sheet_names = ['Zon1_processed',
                    'Zon2_processed',
                    'Zon3_processed',
                    'Zon4_processed',
                    'Zon5_processed',
                    'Zon6_processed'
                    ]
    
    df = pd.read_excel('data/Sedimenteringshastigheter.xlsx', sheet_name = sheet_names)
    
    exp_data = {}
    
    for key in df:
        
        t_end = 24*60*60
        
        t_raw = df[key]['t (s)'].to_numpy(dtype = 'float64')
        c_raw = df[key]['c (g/l)'].to_numpy(dtype = 'float64')
        c_max = df[key]['c_max (g/l)'][0]
        
        c = c_raw[t_raw <= t_end]
        t = t_raw[t_raw <= t_end]
        
        exp_data[key[:4]] = {'t_raw': t_raw,
                             'c_raw': c_raw,
                             'c_max': c_max,
                             't': t,
                             'c': c,
                             'c_norm': c/c_max
                             }
    #     plt.plot(t, c, '-.', label = file_name)
    
    span = False
    
    large_particle_fraction = .8 # Fraction of the concentration made up of large particles
    
    v_guess = 6.25e-6 # m/s
    D_guess = 4.58e-08 # m^2/s
    
    L = .037 # mm cuvette
    n_dofs = 100 # Number of nodes
    
    model = empty_conv_diff_model(L, n_dofs)
    
    if span:
        ivp_pars = {'t_span': [0, 90000],
                    'c_init': 1/(.025-.005)*np.ones((n_dofs)),
                    }
        
        measured_span = [.005, .025]
        
    else:
        
        
        ivp_pars = {'t_span': [0, 90000],
                    'c_init': np.ones((n_dofs)),
                    }
        
        measured_span = [.005]
        # measured_span = [.015]
    
    #%% Calibration
    
    calibrate = 0
    
    if calibrate:
        i = 0
        file_name = sheet_names[i]
            
        file_name = file_name[:4]
        
        t_data = exp_data[file_name]['t']
        c_data = exp_data[file_name]['c']
        c_max = exp_data[file_name]['c_max']
        # c_max = 5
        # c_max = exp_data[file_name]['c_max']/(.025-.005)
        
        solutions = []
        
        # This assumes that the concentration is uniform in the entire domain at t=0
        if span:
            ivp_pars['c_init'] = np.ones((n_dofs,))*c_data[0]/(measured_span[1] - measured_span[0])
            c_max = exp_data[file_name]['c_max']/(.025-.005)
        else:
            ivp_pars['c_init'] = np.ones((n_dofs,))*c_data[0]
        
        t_eval = t_data
        
        bounds = ((0, 1e-4), (0, 1e-6), (0, 1e-4), (0, 1e-6), (0, 1))
        
        total_start_time = time.time()
        for j, seed in enumerate(seeds):
            start = time.time()
            print(f'Starting calibration with seed number {j + 1}')
            res = differential_evolution(lambda x: objective_function(x, c_data, t_data, model, ivp_pars, c_max), 
                                         bounds = bounds,
                                         seed = seed,
                                         disp = True,
                                            # workers = 3
                                         )
            
            popt = res.x
            
            print(f'Calibration of zone {i+1} complete. Time: {(time.time() - start):.4} seconds.')
            print(f'Optimal parameters: {popt}')
            
            solutions.append(popt)
        total_time = time.time() - total_start_time
        print(f'Total analysis time for {len(seeds)} seeds: {total_time/60:.4} minutes')
        
        solutions = np.array(solutions)
    
    else:
        seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        bounds = ((0, 1e-4), (0, 1e-6), (0, 1e-4), (0, 1e-6), (0, 1))
        solutions = np.array([[9.29615183e-07, 4.68182449e-09, 1.96124285e-05, 1.38635891e-07, 2.68585828e-01],
                               [1.93772651e-05, 1.36411278e-07, 9.25999366e-07, 4.73938006e-09, 7.31666959e-01],
                               [9.19917048e-07, 4.53144248e-09, 1.95024839e-05, 1.38013665e-07, 2.67161010e-01],
                               [9.18657827e-07, 4.53946169e-09, 1.94119062e-05, 1.37237582e-07, 2.67201128e-01],
                               [1.97483804e-05, 1.39556847e-07, 9.40749801e-07, 4.75877737e-09, 7.28097791e-01],
                               [1.95297671e-05, 1.37894395e-07, 9.31415402e-07, 4.71094934e-09, 7.30665866e-01],
                               [9.30968226e-07, 4.74309617e-09, 1.94690075e-05, 1.37287011e-07, 2.68366185e-01],
                               [9.18831962e-07, 4.40775016e-09, 1.97008958e-05, 1.40066067e-07, 2.66677012e-01],
                               [9.11398383e-07, 4.40966944e-09, 1.93347588e-05, 1.37066114e-07, 2.64812644e-01],
                               [9.22073674e-07, 4.55931954e-09, 1.94309242e-05, 1.37516893e-07, 2.67115749e-01]])
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    
    ax1.set_title('v$_1$')
    ax2.set_title('D$_1$')
    ax3.set_title('v$_2$')
    ax4.set_title('D$_2$')
    ax5.set_title('$\gamma$')
    
    ax1.set_ylabel('[m/s]')
    ax3.set_ylabel('[m/s]')
    
    ax2.set_ylabel('[m$^2$/s]')
    ax4.set_ylabel('[m$^2$/s]')
    
    ax5.set_ylabel('[-]')
    
    n_tests = np.arange(0, len(seeds)) + 1
    
    for i, axes in enumerate([ax1, ax2, ax3, ax4, ax5]):
        axes.set_xlabel('Test number')
        axes.scatter(n_tests, solutions[:, i])
        axes.set_ylim(bottom = bounds[i][0], top = bounds[i][1])
    
    fig.set_size_inches(6.4*1.3, 4.8)
    
    fig.tight_layout()
    
    plt.show(block = False)
    

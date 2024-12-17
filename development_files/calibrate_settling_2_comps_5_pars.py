# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 08:10:29 2023

@author: DanielEspinoza
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

def fem_model(t, y, C_inv, K_conv, K_disp, c_max):
    
    
    dcdt = C_inv @ - (K_disp - K_conv) @ y
    
    # dcdt = C_inv @ - (K_disp @ y - (1 - y/c_max)**2*K_conv @ y )
    
    # beta = 1
    
    # dcdt = C_inv @ - (K_disp @ y - (1 - beta*(y/c_max)**2)*K_conv @ y )
    
    # dcdt = C_inv @ - (K_disp @ y - (1 - y/c_max)*(1 - beta*y/c_max)*K_conv @ y )
    
    return dcdt

def response(t_data, v1, D1, v2, D2, large_particle_fraction, model, ivp_pars, c_max, measured_span = [.005], return_all = False):
    
    small_particle_fraction = 1 - large_particle_fraction
    
    sol1 = solve_ivp(lambda t, y: fem_model(t, y, model['C_inv'], v1*model['K_conv'], D1*model['K_disp'], c_max), 
                    ivp_pars['t_span'], 
                    ivp_pars['c_init']*large_particle_fraction, 
                    t_eval = t_data,
                    method = 'BDF',
                    )
    
    sol2 = solve_ivp(lambda t, y: fem_model(t, y, model['C_inv'], v2*model['K_conv'], D2*model['K_disp'], c_max), 
                    ivp_pars['t_span'], 
                    ivp_pars['c_init']*small_particle_fraction, 
                    t_eval = t_data,
                    method = 'BDF',
                    )
    
    dx = model['L']/(model['n_dofs']-1)
        
    if len(measured_span) == 1:
        
        i_point = int(measured_span[0]/dx)
        c1 = sol1.y[i_point, :]
        c2 = sol2.y[i_point, :]
        c_sum = c1 + c2
    
    else:
        
        i_start = int(measured_span[0]/dx)
        i_end = int(measured_span[1]/dx)
        
        x_window = model['mesh'][i_start:i_end+1]
        c1_window = sol1.y[i_start:i_end+1, :]
        c2_window = sol2.y[i_start:i_end+1, :]

        c1 = trapz(c1_window, x=x_window, axis = 0)
        c2 = trapz(c2_window, x=x_window, axis = 0)
        
        c_sum = c1 + c2
    
    if return_all:
        return c_sum, c1, c2, sol1, sol2
    else:
        return c_sum

if __name__ == '__main__':
#%% Load data
    
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
    
    fig, ax = plt.subplots(nrows = 3, ncols = 2)
    ax_flat = ax.flatten()
    MSE = np.zeros(6)
    R_squared = np.zeros(6)
    pars = np.zeros((6, 5))
    
    for i, file_name in enumerate(sheet_names):
        
        file_name = file_name[:4]
        
        t_data = exp_data[file_name]['t']
        c_data = exp_data[file_name]['c']
        c_max = exp_data[file_name]['c_max']
        # c_max = 5
        # c_max = exp_data[file_name]['c_max']/(.025-.005)
        
        # This assumes that the concentration is uniform in the entire domain at t=0
        if span:
            ivp_pars['c_init'] = np.ones((n_dofs,))*c_data[0]/(measured_span[1] - measured_span[0])
            c_max = exp_data[file_name]['c_max']/(.025-.005)
        else:
            ivp_pars['c_init'] = np.ones((n_dofs,))*c_data[0]
        
        t_eval = t_data
        
        start = time.time()
        
        popt, pcov = curve_fit(lambda t_data, v1, D1, v2, D2, large_particle_fraction: response(t_data, v1, D1, v2, D2, large_particle_fraction, model, ivp_pars, c_max, measured_span = measured_span),
                                t_data,
                                c_data,
                                p0 = [1e-7, 1e-7, 1e-7, 1e-7, .5],
                                bounds = ((0, 0, 0, 0, 0),(np.inf, np.inf, np.inf, np.inf, 1)),
                                ftol = 1e-10
                                )
        
        print(f'Calibration of zone {i+1} complete. Time: {(time.time() - start):.4} seconds.')
        print(f'Optimal parameters: {popt}')
        
        pars[i, :] = popt
        
        c_sim, c1, c2, sol1, sol2 = response(t_data, *popt, model, ivp_pars, c_max, measured_span = measured_span, return_all = True)
        
        residuals = c_data - c_sim
        
        RSS = np.sum(residuals**2) # Residual sum-of-squares
        
        TSS = np.sum((c_data - np.mean(c_data))**2) # Total sum-of-squares
        
        MSE[i] = np.mean((c_data - c_sim)**2)
        R_squared[i] = 1 - RSS/TSS
        
        exp_data[file_name]['c1_sim'] = c1
        exp_data[file_name]['c2_sim'] = c2
        exp_data[file_name]['c_sim'] = c_sim
        exp_data[file_name]['popt'] = popt
        
        ax_flat[i].plot(t_data/60/60, c_data, '-.', label = 'Experimental data')
        ax_flat[i].plot(t_data/60/60, c1, lw = 1, ls = '--', label = f'Particle group 1, v = {popt[0]:.3} [m/s], D = {popt[1]:.3} [m$^2$/s]')
        ax_flat[i].plot(t_data/60/60, c2, lw = 1, ls = '--', label = f'Particle group 2, v = {popt[2]:.3} [m/s], D = {popt[3]:.3} [m$^2$/s]')
        ax_flat[i].plot(t_data/60/60, c_sim, label = f'Concentration sum, $\gamma$ = {popt[4]:.3}', color = 'k')
        ax_flat[i].set_ylabel('Concentration [g/l]')
        ax_flat[i].set_xlabel('Time [h]')
        ax_flat[i].set_title(f'Zone {i+1}, R$^2$ = {R_squared[i]:.4}')
        ax_flat[i].legend()
    
    # import pickle
    # with open('calibration_5_pars.pkl', 'wb') as f:
    #     pickle.dump(exp_data, f)
    
    fig.set_size_inches(4.8*2.5, 6.4*1.5)
    
    fig.tight_layout()
    
    
    # fig.savefig('figures/'+'calibrated_run_5_to_25_mm_10_h_500dpi.png',
    #             dpi=500,
    #             edgecolor='w',
    #             bbox_inches='tight')
    
    #%% Simulate full cuvette
    
    # v_guess = 8.79e-6
    # D_guess = 6.7e-8
    
    # zone = 1
    
    # v_guess, D_guess = pars[zone-1, :]
    
    # if span:
    #     ivp_pars['c_init'] = np.ones((n_dofs,))*exp_data['Zon' + str(zone)]['c'][0]/(measured_span[1] - measured_span[0])
    #     c_max = exp_data['Zon' + str(zone)]['c_max']#/(.025-.005)
    # else:
    #     ivp_pars['c_init'] = np.ones((n_dofs,))*exp_data['Zon' + str(zone)]['c'][0]
    #     # c_max = 5
    
    # sol = solve_ivp(lambda t, y: fem_model(t, y,
    #                                        model['C_inv'],
    #                                        v_guess*model['K_conv'],
    #                                        D_guess*model['K_disp'],
    #                                        c_max,
    #                                        ), 
    #                 ivp_pars['t_span'], 
    #                 ivp_pars['c_init'], 
    #                 method = 'BDF',
    #                 )
    
    # plt.figure()
    # plt.plot(model['mesh'], sol.y)
    # plt.xlabel('Cuvette length [m]')
    # plt.ylabel('Concentration [g/l]')
    # plt.grid()
    
    #%% Testing new calibration
    
    # t_data = exp_data['Zon2']['t']
    # c_data = exp_data['Zon2']['c']
    
    # ivp_pars['c_init'] = np.ones((n_dofs,))*c_data[0]
    # size_fraction = .5
    # settling_weight = .5
    
    # v1 = 1.29e-5
    # D1 = 8.01e-8
    # v2 = 7.55e-7
    # D2 = 4.71e-9
    # gamma = .854
    
    # # c_sum, c1, c2 = response(t_data, v1, D1, v2, D2, gamma, model, ivp_pars, c_max, measured_span = measured_span, return_all = True)
    # # popt = np.array([v1, D1, v2, D2, gamma])
    
    # popt, pcov = curve_fit(lambda t_data, v1, D1, v2, D2, large_particle_fraction: response(t_data, v1, D1, v2, D2, large_particle_fraction, model, ivp_pars, c_max, measured_span = measured_span),
    #                         t_data,
    #                         c_data,
    #                         p0 = [1e-7, 1e-7, 1e-7, 1e-7, .5],
    #                         # bounds = (1e-10, np.inf),
    #                         bounds = ((0, 0, 0, 0, 0),(np.inf, np.inf, np.inf, np.inf, 1)),
    #                         )
    # c_sum, c1, c2 = response(t_data, *popt, model, ivp_pars, c_max, measured_span = measured_span, return_all = True)
    
    
    # fig, ax = plt.subplots()
    
    # ax.plot(t_data, c_data, ls = '-.', label = 'Data')
    # ax.plot(t_data, c1, lw = 1, label = f'Particle 1, v = {popt[0]:.3}, D = {popt[1]:.3}')
    # ax.plot(t_data, c2, lw = 1, label = f'Particle 2, v = {popt[2]:.3}, D = {popt[3]:.3}')
    # ax.plot(t_data, c_sum, lw = 1, label = 'Sum of particles')
    # size_fraction = popt[2]
    # settling_weight = popt[3]
    # ax.set_title(f'Size fraction = {popt[4]:.3}')
    # ax.legend()
    
    
    plt.show(block = False)
    

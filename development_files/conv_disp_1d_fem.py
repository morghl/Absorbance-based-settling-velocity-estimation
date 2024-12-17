# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 14:22:56 2023

@author: DanielEspinoza
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, trapz
from scipy.optimize import curve_fit
import pandas as pd

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"

def K_conv_element():
    
    return 1/2 * np.array([[-1, -1],
                           [1, 1]])

def K_disp_element(x1, x2):
    
    return 1/(x2 - x1) * np.array([[1, -1],
                                   [-1, 1]])

def C_element(x1, x2):
    
    return (x2 - x1) * np.array([[1/3, 1/6],
                                 [1/6, 1/3]])

def empty_conv_diff_model(L, n_dofs):
    
    dofs = np.arange(n_dofs)
    
    mesh = np.linspace(0, L, n_dofs)
    
    elements = dofs[:-1]
    edofs = np.vstack((elements, dofs[:-1], dofs[1:])).T

    K_conv = np.zeros((n_dofs, n_dofs))
    K_disp= np.zeros((n_dofs, n_dofs))
    C = np.zeros((n_dofs, n_dofs))

    for element in edofs:
        
        i, j = element[1], element[2]
        
        x1 = mesh[i]
        x2 = mesh[j]
        
        Ke_conv = K_conv_element()
        Ke_disp = K_disp_element(x1, x2)
        Ce = C_element(x1, x2)
        
        K_conv[np.ix_((i, j), (i, j))] += Ke_conv
        K_disp[np.ix_((i, j), (i, j))] += Ke_disp
        C[np.ix_((i, j), (i, j))] += Ce
    
    C_inv = np.linalg.inv(C)
    
    return {'K_conv': K_conv,
            'K_disp': K_disp,
            'C': C,
            'C_inv': C_inv,
            'dofs': dofs,
            'mesh': mesh,
            'elements': elements,
            'edofs': edofs,
            'n_dofs': n_dofs,
            'L': L,
            }

def fem_model(t, y, C_inv, K_conv, K_disp):

    dcdt = C_inv @ - (K_disp - K_conv) @ y
    
    return dcdt

def response(t_data, v, D, model, ivp_pars, measured_span = [.005]):
    
    sol = solve_ivp(lambda t, y: fem_model(t, y, model['C_inv'], v*model['K_conv'], D*model['K_disp']), 
                    ivp_pars['t_span'], 
                    ivp_pars['c_init'], 
                    t_eval = t_data,
                    method = 'BDF',
                    )
    
    dx = model['L']/(model['n_dofs']-1)
    
    if len(measured_span) == 1:
        
        i_point = int(.005/dx)
        c_measured = sol.y[i_point, :]
    
    else:
        
        i_start = int(measured_span[0]/dx)
        i_end = int(measured_span[1]/dx)
        
        x_window = model['mesh'][i_start:i_end+1]
        c_window = sol.y[i_start:i_end+1, :]

        c_measured = trapz(c_window, x=x_window, axis = 0)
    
    return c_measured


if __name__ == "__main__":

    #%% Load data
    
    file_names = ['Zon1_konc',
                  'Zon2_konc',
                  'Zon3_konc',
                  'Zon4_konc',
                  'Zon5_konc',
                  'Zon6_konc'
                  ]
    
    exp_data = {}
    
    for file_name in file_names:
        
        t_end = 1
        
        df = pd.read_csv(f'data/{file_name}.csv', sep = ';')
        
        t_raw = df.iloc[:, 0].to_numpy(dtype = 'float64')
        c_raw = df.iloc[:, 1].to_numpy(dtype = 'float64')
        
        c = c_raw[t_raw <= t_end]
        t = t_raw[t_raw <= t_end]*60*60 
        
        exp_data[file_name] = np.vstack((t, c))
        exp_data[file_name + '_raw'] = np.vstack((t_raw*60*60, c_raw))
    #     plt.plot(t, c, '-.', label = file_name)
    
    span = False
    
    v_guess = 6.5E-6 # m/s
    D_guess = 4.8E-8 # m^2/s
    
    L = .037 # mm cuvette
    n_dofs = 40 # Number of nodes
    
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
    
    #%% Calibration
    
    fig, ax = plt.subplots(nrows = 3, ncols = 2)
    ax_flat = ax.flatten()
    MSE = np.zeros(6)
    pars = np.zeros((6, 2))
    
    for i, file_name in enumerate(file_names):
    
        t_data = exp_data[file_name][0]
        c_data = exp_data[file_name][1]
        t_data_raw = exp_data[file_name + '_raw'][0]
        c_data_raw = exp_data[file_name + '_raw'][1]
        
        t_eval = t_data
        
        popt, pcov = curve_fit(lambda t_data, v, D: response(t_data, v, D, model, ivp_pars, measured_span = measured_span),
                                t_data,
                                c_data,
                                p0 = [v_guess, D_guess],
                                # bounds = (0, np.inf),
                                )
        
        pars[i, :] = popt
        
        c_sim = response(t_data, popt[0], popt[1], model, ivp_pars, measured_span = measured_span)
        # c_sim = response(t_data_raw, popt[0], popt[1], model, ivp_pars, measured_span = measured_span)
        
        MSE[i] = np.mean((c_data - c_sim)**2)
        
        ax_flat[i].plot(t_data, c_data, '-.', label = 'Experimental data')
        # ax_flat[i].plot(t_data_raw, c_data_raw, '-.', label = 'Experimental data')
        ax_flat[i].plot(t_data, c_sim, label = f'Model, v = {popt[0]:.3} [m/s], D = {popt[1]:.3} [m^2/s]', color = 'k')
        # ax_flat[i].plot(t_data_raw, c_sim, label = f'Model, v = {popt[0]:.3} [m/s], D = {popt[1]:.3} [m^2/s]', color = 'k')
        ax_flat[i].set_ylabel('Normalized concentration [-]')
        ax_flat[i].set_xlabel('Time [s]')
        ax_flat[i].set_title(f'Zone {i+1}')
        ax_flat[i].legend()
    
    fig.set_size_inches(4.8*2.5, 6.4*1.5)
    
    fig.tight_layout()
    
    # fig.savefig('figures/'+'calibrated_run_5_to_25_mm_10_h_500dpi.png',
    #             dpi=500,
    #             edgecolor='w',
    #             bbox_inches='tight')
    
    
    #%% Model response
    
    # t_data = exp_data['Zon3_konc'][0]
    # c_data = exp_data['Zon3_konc'][1]
    
    # v_guess = 8.79e-6
    # D_guess = 6.7e-8
    
    # c_sim = response(t_data, v_guess, D_guess, model, ivp_pars, measured_span = measured_span)
    
    # plt.figure()
    # plt.plot(t_data, c_sim, label = f'Model, D = {D_guess}, v = {v_guess}', color = 'k')
    # plt.ylabel('Normalized concentration at x = 0.005 m [-]')
    # plt.xlabel('Time [s]')
    # plt.legend()
    # plt.plot(t_data, c_data)
    
    #%% Time response, full domain
    
    # # v_guess = 8.79e-6
    # # D_guess = 6.7e-8
    
    # sol = solve_ivp(lambda t, y: fem_model(t, y, model['C_inv'], v_guess*model['K_conv'], D_guess*model['K_disp']), 
    #                 ivp_pars['t_span'], 
    #                 ivp_pars['c_init'], 
    #                 method = 'BDF',
    #                 )
    
    # plt.figure()
    # plt.plot(model['mesh'], sol.y)
    # plt.xlabel('Cuvette length [m]')
    # plt.ylabel('Normalized concentration [-]')
    # plt.grid()
    
    
    #%% Show plots
    plt.show(block = False)
    
    # plt.figure()
    # plt.plot(x_window, c_window)
    
    # plt.plot(mesh, np.ones(mesh.shape), '-o')


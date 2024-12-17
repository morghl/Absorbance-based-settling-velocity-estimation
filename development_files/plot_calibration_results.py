# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:59:51 2023

@author: DanielEspinoza
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"

data_file_name = 'calibration_model2'
# data_file_name = 'calibration_model1'

with open(f'{data_file_name}.pkl', 'rb') as f:
    exp_data = pickle.load(f)
    
sheet_names = list(exp_data.keys())

fig, ax = plt.subplots(nrows = 3, ncols = 2)
ax_flat = ax.flatten()

R_squared = np.zeros(6)

v_conv_factor = 1000*3600

data_colors = list(mcolors.TABLEAU_COLORS.values())

for i, file_name in enumerate(sheet_names):
    
    t_data = exp_data[file_name]['t']
    c_data = exp_data[file_name]['c']
    c_max = exp_data[file_name]['c_max']
    c_norm = 1
    c_norm = c_data[0]
    
    t_eval = t_data
    
    popt = exp_data[file_name]['popt']
    
    if 'c1_sim' in exp_data['Zon1'].keys():
        c_sim, c1, c2 = exp_data[file_name]['c_sim'], exp_data[file_name]['c1_sim'], exp_data[file_name]['c2_sim']
        
        if popt[0] < popt[2]:
            c1, c2 = c2, c1
            
            v1 = popt[2]
            D1 = popt[3]
            v2 = popt[0]
            D2 = popt[1]
            
            v1_base, v1_exp = f'{popt[2]:.3}'.split('e')
            D1_base, D1_exp = f'{popt[3]:.3}'.split('e')
            v2_base, v2_exp = f'{popt[0]:.3}'.split('e')
            D2_base, D2_exp = f'{popt[1]:.3}'.split('e')
            popt[4] = 1 - popt[4]
            
        else:
            v1 = popt[0]
            D1 = popt[1]
            v2 = popt[2]
            D2 = popt[3]
            
            v1_base, v1_exp = f'{popt[0]:.3}'.split('e')
            D1_base, D1_exp = f'{popt[1]:.3}'.split('e')
            v2_base, v2_exp = f'{popt[2]:.3}'.split('e')
            D2_base, D2_exp = f'{popt[3]:.3}'.split('e')
        
        if v_conv_factor == 1:
            label1 = rf'Particle group 1, $v_{{s,1}}$ = {v1_base}$\cdot$10$^{{-{v1_exp[2]}}}$ [m/s], $D_{{1}}$ = {D1_base}$\cdot$10$^{{-{D1_exp[2]}}}$ [m$^2$/s]'
            label2 = rf'Particle group 2, $v_{{s,2}}$ = {v2_base}$\cdot$10$^{{-{v2_exp[2]}}}$ [m/s], $D_{{2}}$ = {D2_base}$\cdot$10$^{{-{D2_exp[2]}}}$ [m$^2$/s]'
        else:
            label1 = rf'Particle group 1, $v_{{s,1}}$ = {v1*v_conv_factor:.2f} [mm/h], $D_{{1}}$ = {D1_base}$\cdot$10$^{{-{D1_exp[2]}}}$ [m$^2$/s]'
            label2 = rf'Particle group 2, $v_{{s,2}}$ = {v2*v_conv_factor:.2f} [mm/h], $D_{{2}}$ = {D2_base}$\cdot$10$^{{-{D2_exp[2]}}}$ [m$^2$/s]'
        label_c_sim = f'Concentration sum, $\gamma$ = {popt[4]:.3}'
        
        ax_flat[i].plot(t_data/60/60, c_data/c_norm, label = 'Experimental data', color = data_colors[i], ls = '', marker = 'o', markersize = 3)
        ax_flat[i].plot(t_data/60/60, c1/c_norm, ls = '--', label = label1, color = 'grey', lw = 2)
        ax_flat[i].plot(t_data/60/60, c2/c_norm, ls = '-.', label = label2, color = 'grey', lw = 2)
        ax_flat[i].plot(t_data/60/60, c_sim/c_norm, label = label_c_sim, color = 'k', lw = 2, alpha = 1)
        
    else:
        c_sim = exp_data[file_name]['c_sim']
        v, D = popt
        v_base, v_exp = f'{popt[0]:.3}'.split('e')
        D_base, D_exp = f'{popt[1]:.3}'.split('e')
        
        if v_conv_factor == 1:
            label_c_sim = rf'Model, $v_{{s}}$ = {v_base}$\cdot$10$^{{-{v_exp[2]}}}$ [m/s], $D$ = {D_base}$\cdot$10$^{{-{D_exp[2]}}}$ [m$^2$/s]'
        else:
            label_c_sim = rf'Model, $v_{{s}}$ = {v*v_conv_factor:.2f} [mm/h], $D$ = {D_base}$\cdot$10$^{{-{D_exp[2]}}}$ [m$^2$/s]'
        
        ax_flat[i].plot(t_data/60/60, c_data/c_norm, label = 'Experimental data', color = data_colors[i], ls = '', marker = 'o', markersize = 3)
        
        ax_flat[i].plot(t_data/60/60, c_sim/c_norm, label = label_c_sim, color = 'k', lw = 2, alpha = 1)
    
    residuals = c_data - c_sim
    
    RSS = np.sum(residuals**2) # Residual sum-of-squares
    
    TSS = np.sum((c_data - np.mean(c_data))**2) # Total sum-of-squares
    
    R_squared[i] = 1 - RSS/TSS
    
    ax_flat[i].set_ylabel('Relative concentration [c/c$_0$]')
    ax_flat[i].set_xlabel('Time [h]')
    ax_flat[i].set_title(f'Zone {i+1}, R$^2$ = {R_squared[i]:.4}')
    ax_flat[i].legend()

fig.set_size_inches(4.8*2.5, 6.4*1.5)

fig.tight_layout()

# fig.savefig(f'figures_review/{data_file_name}_relative.svg',
#             dpi=500,
#             edgecolor='w',
#             bbox_inches='tight')

plt.show(block = False)

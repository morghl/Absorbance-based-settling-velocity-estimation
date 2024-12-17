# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:02:47 2023

@author: DanielEspinoza
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "serif"

data_file_name = 'calibration_model2'

with open(f'{data_file_name}.pkl', 'rb') as f:
    exp_data = pickle.load(f)
    
sheet_names = list(exp_data.keys())

fig, ax = plt.subplots(nrows = 1, ncols = 1)

for i, file_name in enumerate(sheet_names):
    
    t_data = exp_data[file_name]['t']
    c_data = exp_data[file_name]['c']
    c_norm = c_data[0]
    
    ax.plot(t_data/60/60, c_data/c_norm, '-.', label = f'Zone {i + 1}')
    
    ax.set_ylabel(r'Normalized absorbance [Abs$_{t}$/Abs$_{0}$]')
    ax.set_xlabel('Time [h]')
    ax.legend()

# fig.set_size_inches(4.8*2.5, 6.4*1.5)

fig.tight_layout()

fig.savefig(f'figures/abs_measurements.svg',
            dpi=500,
            edgecolor='w',
            bbox_inches='tight')

plt.show(block = False)
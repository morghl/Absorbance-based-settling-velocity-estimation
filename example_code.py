# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:17:23 2024

@author: daniel.espinoza
"""

from estimate_settling import estimate_settling
import pandas as pd
import matplotlib.pyplot as plt

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
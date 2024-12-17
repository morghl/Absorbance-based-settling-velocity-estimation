# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:59:38 2023

@author: DanielEspinoza
"""

from calibrate_settling_2_comps_5_pars import response
from conv_disp_1d_fem import empty_conv_diff_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib as mpl

df = pd.read_excel('data/Sedimenteringshastigheter.xlsx', sheet_name = 'Zon1_processed')
t_data = df['t (s)'].to_numpy(dtype = 'float64')
c_data = df['c (g/l)'].to_numpy(dtype = 'float64')#/df['c (g/l)'][0]
c_data = c_data/c_data[0]

L = .037 # mm cuvette
n_dofs = 100 # Number of nodes
width = .010

model = empty_conv_diff_model(L, n_dofs)

ivp_pars = {}

ivp_pars['c_init'] = np.ones((n_dofs,))
ivp_pars['t_span'] = [0, 24*60*60]
size_fraction = .5
settling_weight = .5

v1 = 1.96e-5
D1 = 1.38e-7
v2 = 9.33e-7
D2 = 4.7e-9
gamma = .731

pars = np.array([v1, D1, v2, D2, gamma])

c_sum, c1, c2, sol1, sol2 = response(t_data, *pars, model, ivp_pars, c_max = 1, measured_span = [.005], return_all = True)

c_total = sol1.y + sol2.y

# fig, ax = plt.subplots()

# ax.plot(t_data/60/60, c1, lw = 1, label = f'Particle 1, v = {pars[0]:.3}, D = {pars[1]:.3}')
# ax.plot(t_data/60/60, c2, lw = 1, label = f'Particle 2, v = {pars[2]:.3}, D = {pars[3]:.3}')
# ax.plot(t_data/60/60, c_sum, lw = 1, label = 'Sum of particles')
# size_fraction = pars[2]
# settling_weight = pars[3]
# ax.set_title(f'Size fraction = {pars[4]:.3}')
# ax.legend()

# fig1, ax1 = plt.subplots()

# ax1.plot(model['mesh'], c_total)

# t_plots = np.array([0, .25, .5, 1, 2, 23])*60*60
# t_plots = np.linspace(0, .3*60*60, 6)
# i_plots = [np.where(t_data >= t_plot)[0][0] for t_plot in t_plots]
# c_total_snapshots = c_total[:, i_plots]

# c_min = np.min(c_total_snapshots)
# c_max = np.max(c_total_snapshots)
# norm = mpl.colors.Normalize(vmin = c_min, vmax = c_max)

# fig2, ax2 = plt.subplots(nrows = 1, ncols = t_plots.shape[0])

# for i, t_plot in enumerate(t_plots):
    
#     snapshot_index = np.where(t_data >= t_plot)[0][0]
#     c_total_snapshot = c_total_snapshots[:, i]
#     Z_total = np.vstack((c_total_snapshot, c_total_snapshot)).T
#     water = Rectangle((0, 0), width, L, color = '#4a94e8', alpha = .3)
#     ax2.flat[i].plot([0, 0, width, width], [(1+.1)*L, 0, 0, (1+.1)*L], color = 'k', lw = 2)
#     ax2.flat[i].add_patch(water)
#     # ax2.flat[i].imshow(Z_total, interpolation='bicubic', cmap=plt.cm.Reds, extent=(0, width, 0, L), norm = norm, alpha=1)
#     ax2.flat[i].imshow(Z_total, cmap=plt.cm.Reds, extent=(0, width, 0, L), norm = norm, alpha=1)
#     ax2.flat[i].set_title(f't = {t_plot/60:.3} min')
#     ax2.flat[i].set_xlim(0-.2*width, (1+.2)*width)
#     ax2.flat[i].set_ylim(0-.2*L, (1+.2)*L)
#     ax2.flat[i].set_aspect('equal')

# fig2.tight_layout()

# fig3, ax3 = plt.subplots()

# water = Rectangle((0, 0), width, L, color = '#4a94e8', alpha = .1)
# ax3.plot([0, 0, width, width], [(1+.1)*L, 0, 0, (1+.1)*L], color = 'k', lw = 2)
# ax3.add_patch(water)
    
# t_start = 0
# t_end = t_data[-1]#1*60*60

# t_animate = t_data[(t_data >= t_start)*(t_data <= t_end)]
# c_animate = c_total[:, (t_data >= t_start)*(t_data <= t_end)]

# c_min = np.min(c_animate)
# c_max = np.max(c_animate)
# norm = mpl.colors.Normalize(vmin = c_min, vmax = c_max)

# c_init = c_total[:, t_data == t_start]
# Z_init = np.hstack((c_init, c_init))

# image = ax3.imshow(Z_init, cmap=plt.cm.Reds, extent=(0, width, 0, L), norm = norm, alpha=1)

# def update1(i):
#     snapshot = c_animate[:, i]
#     Z_snapshot = np.vstack((snapshot, snapshot)).T
#     image.set_data(Z_snapshot)
#     return image

# ani1 = animation.FuncAnimation(fig = fig3, func = update1, frames = t_animate.shape[0], interval = 10)

title_padding = 20

fig4, ax_list = plt.subplots(1, 2)

ax4, ax5 = ax_list

water = Rectangle((0, 0), width, L, color = '#4a94e8', alpha = .3)
ax4.plot([0, 0, width, width], [(1+.1)*L, 0, 0, (1+.1)*L], color = 'k', lw = 2)
ax4.plot([0, width], [L - .005, L - .005], ls = '--', lw = 1, color = 'k')
# ax4.add_patch(water)
ax4.set_xticks([])
ax4.set_yticks([0, L-.005, L], labels = ['z = 37 mm', 'z = 5 mm','z = 0 mm'])
ax4.set_title('Stormwater sample', pad = title_padding)

t_start = 0
t_end = 2*60*60

t_animate = t_data[(t_data >= t_start)*(t_data <= t_end)]
c_animate = c_total[:, (t_data >= t_start)*(t_data <= t_end)]

c1_animate = sol1.y[:, (t_data >= t_start)*(t_data <= t_end)]
c2_animate = sol2.y[:, (t_data >= t_start)*(t_data <= t_end)]

c1_min = np.min(c1_animate)
c1_max = np.max(c1_animate)
c2_min = np.min(c2_animate)
c2_max = np.max(c2_animate)
norm1 = mpl.colors.Normalize(vmin = c1_min, vmax = c1_max)

c1_init = sol1.y[:, t_data == t_start]
Z1_init = np.hstack((c1_init, c1_init))
c2_init = sol2.y[:, t_data == t_start]
Z2_init = np.hstack((c2_init, c2_init))

image1 = ax4.imshow(Z1_init, cmap=plt.cm.Reds, extent=(0, width, 0, L), norm = norm1, alpha=.8)
norm2 = mpl.colors.Normalize(vmin = c2_min, vmax = c2_max)
image2 = ax4.imshow(Z2_init, cmap=plt.cm.Greens, extent=(0, width, 0, L), norm = norm2, alpha=.5)

def update2(i):
    snapshot1 = sol1.y[:, i]
    snapshot2 = sol2.y[:, i]
    Z1_snapshot = np.vstack((snapshot1, snapshot1)).T
    Z2_snapshot = np.vstack((snapshot2, snapshot2)).T
    image1.set_data(Z1_snapshot)
    image2.set_data(Z2_snapshot)
    
    x = t_animate[:i]/60/60
    y = c_5mm[:i]
    y1 = c1_5mm[:i]
    y2 = c2_5mm[:i]
    
    line[0].set_xdata(x)
    line[0].set_ydata(y)
    line1[0].set_xdata(x)
    line1[0].set_ydata(y1)
    line2[0].set_xdata(x)
    line2[0].set_ydata(y2)
    
    return (image1, image2, line, line1, line2)

c_5mm = c_animate[model['mesh'] >= .005, :][0]
c1_5mm = c1_animate[model['mesh'] >= .005, :][0]
c2_5mm = c2_animate[model['mesh'] >= .005, :][0]

ax5.plot(t_data/60/60, c_data, label = "Measured concentration", color = "whitesmoke", marker = 'o', markersize = .3)
line = ax5.plot(t_animate[0]/60/60, c_5mm[0], label = "Concentration sum, model", color = 'k')
line1 = ax5.plot(t_animate[0]/60/60, c1_5mm[0], label = "Fast-settling particles, model", color = 'red', ls = '--')
line2 = ax5.plot(t_animate[0]/60/60, c2_5mm[0], label = "Slow-settling particles, model", color = 'green', ls = '--')
ax5.set_ylim((0-.1)*(np.min(c_5mm)), (1+.1)*(np.max(c_5mm)))
ax5.set_xlim((0-.1)*t_start/60/60, (1+.1)*t_end/60/60)
ax5.set_xlabel('Time [h]')
ax5.set_ylabel(r'Normalized concentration [c/c$_0$]')
ax5.set_title('Concentration measured at z = 5 mm', pad = title_padding)
ax5.legend()

fig4.set_size_inches(6.4*1.2, 4.8)
fig4.tight_layout()

ani2 = animation.FuncAnimation(fig = fig4, func = update2, frames = t_animate.shape[0], interval = 1)
# ani2.save(filename="settling_zone1.gif", writer="pillow") # Spara som .gif

plt.show()
ani2.save(filename="settling_zone1_15ms.mp4", writer="ffmpeg") # Spara som .mp4, men har problem med ffmpeg
# ani2.save(filename="settling_zone1_1ms.svg", writer="imagemagick")
# ani2.save(filename="settling_zone1_1ms.gif", writer="pillow")
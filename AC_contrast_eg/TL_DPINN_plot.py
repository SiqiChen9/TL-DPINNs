# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:23:48 2023

@author: Ye Li
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

data = scipy.io.loadmat('AC.mat')
usol = data['uu']
t_star = data['tt'][0]
x_star = data['x'][0]
TT, XX = np.meshgrid(t_star, x_star)

plt.figure(1)
plt.pcolor(TT, XX, usol, cmap='jet')
plt.colorbar()
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#ax = plt.gca()
#ax.xaxis.set_major_locator(MultipleLocator(0.5))
#ax.yaxis.set_major_locator(MultipleLocator(0.5))
plt.title(r'Exact solution $u(t,x)$')
plt.show()

data = np.load('AC_pred_pinn.npy')
plt.figure(2)
plt.pcolor(TT, XX, data, cmap='jet')
plt.colorbar()
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
plt.title(r'PINN prediction $u(t,x)$')
plt.show()

data = np.load('AC_pred_our.npy')
plt.figure(3)
plt.pcolor(TT, XX, data, cmap='jet')
plt.colorbar()
plt.xlabel(r'$t$')
plt.ylabel(r'$x$')
plt.title(r'TL-DPINN prediction $u(t,x)$')
plt.show()

data = np.load('AC_residual_pinn.npy')
plt.figure(4)
plt.plot(np.linspace(0,1,len(data)-2),data[2:],'d-')
plt.xlabel(r'$t$')
plt.ylabel(r'$\mathcal{L}(t_i,\theta)$')
plt.yscale('log')
plt.show()

data = np.load('AC_residual_our.npy')
plt.figure(5)
plt.plot(np.linspace(0,1,len(data)),data,'d-')
plt.xlabel(r'$t$')
plt.ylabel(r'$\mathcal{L}(t_i,\theta)$')
plt.show()

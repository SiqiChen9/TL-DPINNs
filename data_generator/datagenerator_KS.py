# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:08:00 2023
A Kuramoto–Sivashinsky equation numerical solver.
Solving 1D u_t + alpha*u*u_x + beta*u_xx + gamma*u_xxxxx = 0 with periodic boundary condition and initial condition u0 in domain (x,t)\in [-L/2,L/2]\times[0,1].

@author: Ye Li
"""

import numpy as np
import matplotlib.pyplot as plt

#方程参数，可修改
#regular solution
alpha = 5
beta = 0.5
gamma = 0.005
L = 2
def u_initial(x): #初始函数
    y = -np.sin(np.pi*x)
    return y

# =============================================================================
# #chaotic solution
# alpha = 100/16
# beta = 100/16**2
# gamma = 100/16**4
# L = 2*np.pi
# def u_initial(x): #初始函数
#     y = - np.cos(x)*(1 - np.sin(x)) #chaotic solution on [-pi,pi]
#     return y
# =============================================================================

Nx_dim = 511
Nt_dim = 1000001
X = np.linspace(-L/2, L/2, Nx_dim, endpoint=False)
kk = ((np.arange(Nx_dim) + Nx_dim//2)%Nx_dim - Nx_dim//2)*2*np.pi/L
U0 = u_initial(X)
U = np.zeros((Nx_dim,Nt_dim))
U[:,0] = U0
tau = 1/(Nt_dim-1)

for i in range(Nt_dim-1):
    Un = U[:,i]
    Unk = np.fft.fft(Un)
    #correction
    Unk_half = (Unk/(tau/2) - alpha*1j*kk*np.fft.fft(Un**2/2)) / (1/(tau/2) - beta*kk**2 + gamma*kk**4)
    Un_half = np.fft.ifft(Unk_half)
    #prediction
    Vnk = ((1/tau + beta*kk**2/2 -gamma*kk**4/2)*Unk -  alpha*1j*kk*np.fft.fft(Un_half**2/2))/(1/tau - beta*kk**2/2 + gamma*kk**4/2)
    Vn = np.fft.ifft(Vnk)#.real
    U[:,i+1] = Vn

Us = U[:,::4000]
Us = np.vstack((Us,Us[0,:]))
np.save('data_KS.npy',Us)

Xs = np.linspace(-L/2,L/2,Nx_dim+1)
Ts = np.linspace(0,1,251)
TT, XX = np.meshgrid(Ts, Xs)

plt.figure(1)
plt.plot(Xs,Us)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('solution curves at different time point')
plt.show()

plt.figure(2)
plt.pcolor(TT, XX, Us, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title(r'exact $u(x,t)$')
plt.show()

# 与论文中给的数据集的误差
# =============================================================================
# import scipy
# import numpy as np
# data = scipy.io.loadmat('ks_chaotic.mat')
# usol = data['usol']
# err = np.abs(usol - Us)
# plt.figure(3)
# plt.pcolor(TT, XX, err, cmap='jet')
# plt.colorbar()
# plt.title('error')
# plt.show()
# =============================================================================

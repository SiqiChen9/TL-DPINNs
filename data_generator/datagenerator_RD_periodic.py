# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 16:27:58 2023
A diffusion-reaction numerical solver.
Solving 1D u_t - d*u_xx - k*u^2 = 0 with periodic boundary condition and initial condition u0 in domain (x,t)\in [-L/2,L/2]\times[0,1].
@author: Ye Li
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#方程参数，可修改
d = 0.01
k = 0.01
L = 2
def u_initial(x): #初始函数
    return np.sin(2*np.pi*x)*(1 + np.cos(2*np.pi*x))

Nx_dim = 512
Nt_dim = 1000001
X = np.linspace(-L/2, L/2, Nx_dim, endpoint=False)
kk = ((np.arange(Nx_dim) + Nx_dim//2)%Nx_dim - Nx_dim//2)*2*np.pi/L
U0 = u_initial(X)
U = np.zeros((Nx_dim,Nt_dim))
U[:,0] = U0
tau = 1/(Nt_dim-1)

for i in tqdm(range(Nt_dim-1)):
    Un = U[:,i]
    Unk = np.fft.fft(Un)
    #correction
    Unk_half = (Unk/(tau/2) + k*np.fft.fft(Un**2)) / (1/(tau/2) + d*kk**2)
    Un_half = np.fft.ifft(Unk_half)
    #prediction
    Vnk = ((1/tau - d*kk**2/2)*Unk + k*np.fft.fft(Un_half**2))/(1/tau + d*kk**2/2)
    Vn = np.fft.ifft(Vnk)#.real
    U[:,i+1] = Vn

usol = U[::1,::5000]
np.savez('data_RD_periodic.npz',X=X, usol=usol)

Xs = np.linspace(-L/2,L/2,Nx_dim)
# Xs = np.linspace(0,L,Nx_dim)
# Xs = np.linspace(-L/2,L/2,len(usol))
Ts = np.linspace(0,1,201)
TT, XX = np.meshgrid(Ts, Xs)

plt.figure(1)
plt.plot(Xs,usol)
plt.xlabel('$x$')
plt.ylabel('$u(x,t)$')
plt.title('solution curves at different time point')
plt.show()

plt.figure(2)
plt.pcolor(TT, XX, usol, cmap='jet')
plt.colorbar()
plt.xlabel('$t$')
plt.ylabel('$x$')
plt.title(r'exact $u(x,t)$')
plt.show()
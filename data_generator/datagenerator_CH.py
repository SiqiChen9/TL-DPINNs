# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 17:59:45 2023
A Cahn-Hilliard equation numerical solver.
Solving 1D u_t - (r2*(u**3-r1*u_xx)_xx = 0 with periodic boundary condition and initial condition u0 in domain (x,t)\in [-L/2,L/2]\times[0,1].
@author: Ye Li
"""


import numpy as np
import matplotlib.pyplot as plt

#方程参数，可修改
r1 = 1e-6
r2 = 0.01
L = 2
def u_initial(x): #初始函数
    y = -np.cos(2*np.pi*x)
    return y

Nx_dim = 512
Nt_dim = 100001
#Nt_dim = 201
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
    Unk_half = (Unk/(tau/2) - r2*kk**2*np.fft.fft(Un**3)) / (1/(tau/2) + r1*kk**4 - r2*kk**2)
    Un_half = np.fft.ifft(Unk_half)
    #prediction
    Vnk = ((1/tau - r1*kk**4/2 + r2*kk**2/2)*Unk - r2*kk**2*np.fft.fft(Un_half**3))/(1/tau + r1*kk**4/2 - r2*kk**2/2)
    Vn = np.fft.ifft(Vnk)#.real
    U[:,i+1] = Vn

Us = U[:,::500]
#Us = U[:,::1]
np.save('data_CH.npy',Us)

Xs = np.linspace(-L/2,L/2,Nx_dim)
Ts = np.linspace(0,1,201)
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


# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:26:22 2023
A diffusion-reaction numerical solver.
Solving 1D u_t - d*u_xx - k*u^2 = 0 with zero boundary condition and initial condition u0 in domain (x,t)\in [0,1]\times[0,1].
 
@author: Ye Li
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

#方程参数，可修改
d = 0.01
k = 0.01
def u_initial(x): #初始函数
    y = np.sin(2*np.pi*x)*(1 + np.cos(2*np.pi*x))
    return y

Nx_dim = 1001
Nt_dim = 200001
X = np.linspace(0,1,Nx_dim)
U0 = u_initial(X)
U = np.zeros((Nx_dim,Nt_dim))
U[:,0] = U0
tau = 1/(Nt_dim-1)
h = 1/(Nx_dim-1)

#有限差分格式用到的系数
p1 = -d/h**2
r1 = 1/(tau/2) + 2*d/h**2
q1 = -d/h**2

p = -d/(2*h**2)
r = 1/tau + d/h**2
q = -d/(2*h**2)

pn = d/(2*h**2)
rn = 1/tau-d/h**2
qn = d/(2*h**2)

A1 = np.zeros((Nx_dim-2,Nx_dim-2))
A1[0,:2] = np.array([r1,q1])
A1[-1,-2:] = np.array([p1,r1])
for j in range(1,Nx_dim-3):
    A1[j,j-1:j+2] = np.array([p1,r1,q1])

A = np.zeros((Nx_dim-2,Nx_dim-2))
A[0,:2] = np.array([r,q])
A[-1,-2:] = np.array([p,r])
for j in range(1,Nx_dim-3):
    A[j,j-1:j+2] = np.array([p,r,q])

# A second-order correction-prediction scheme, similar to Crank-Nicolson scheme.    
for i in tqdm(range(Nt_dim-1)):
# for i in pbar:
    #correct
    Un = U[:,i]
    b1 = np.zeros(Nx_dim-2)
    for j in range(Nx_dim-2):
        b1[j] = Un[j+1]/(tau/2) + k*Un[j+1]**2
    Un_half = np.zeros(Nx_dim)
    Un_half[1:-1] = np.linalg.solve(A1,b1.T).flatten()
    #predict
    b = np.zeros(Nx_dim-2)
    for j in range(Nx_dim-2):
        b[j] = pn*Un[j] + rn*Un[j+1] + qn*Un[j+2] + k*Un_half[j+1]**2
    U[1:-1,i+1] = np.linalg.solve(A,b.T).flatten()

#缩减分辨率后保存，缩减后时间步长为1/200，空间步长为1/100
Us = U[::10,::1000]
np.save('data_RD.npy',Us)

Xs = np.linspace(0,1,101)
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


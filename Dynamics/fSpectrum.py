import scipy.integrate as si
import scipy.fft as sf
import numpy as np
import matplotlib.pyplot as plt
'''
plot f spectrum using fft.
Author: 陈长, 3120104099
Date: Dec 14, 2020
'''

def f(x):
    return np.tanh(x)

def DX(t, x):
    A = np.array([-1, -1, -1], dtype=np.float64)
    A1 = np.array([3.8, -1.9, 0.7], dtype=np.float64)
    A2 = np.array([2.5, 0.06, 1], dtype=np.float64)
    A3 = np.array([-6.6, 1.3, 0.07], dtype=np.float64)
    I = np.array([0, 0, 0], dtype=np.float64)
    dx = [A[0] * x[0] + A1[0] * f(x[0]) + A1[1] * f(x[1]) + A1[2] * f(x[2]) + I[0],
          A[1] * x[1] + A2[0] * f(x[0]) + A2[1] * f(x[1]) + A2[2] * f(x[2]) + I[1],
          A[2] * x[2] + A3[0] * f(x[0]) + A3[1] * f(x[1]) + A3[2] * f(x[2]) + I[2]]
    return dx

Fs = 1000 # sampling frequency
T = 1/Fs # period
L = 200*Fs
tend = 7000
sol = si.solve_ivp(DX, [0, tend], [5, -2, -5], t_eval=np.arange(0, tend, T))
X = sol.y[0][-L:] # choose a time period to analyze
Y = sf.fft(X)
P2 = np.abs(Y/L)
P1 = P2[1:(L//2+2)]
P1[2:-1] = 2*P1[2:-1]
f = Fs*np.arange(L//2+1)/L

# 绘图模块
fig, ax = plt.subplots(1,2)
ax[0].plot(range(X.size),X)
ax[0].set_title('time waveform being analyzed')
ax[0].set_xlabel('1000*t')
ax[0].set_ylabel('X')
ax[0].grid(True)
ax[0].set_xlim([0,200000])
ax[1].plot(f,P1)
ax[1].set_xlim([0,1.25])
ax[1].set_title('frequency spectrum(fs = 1kHz)')
ax[1].set_xlabel('f')
ax[1].set_ylabel('X(f)')
ax[1].grid(True)
plt.show()
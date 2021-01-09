import scipy.integrate as si
import scipy.fft as sf
import numpy as np
import matplotlib.pyplot as plt
'''
plot waveform in time domain.
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

tend = 7000
stp = 0.001  # int step
sol = si.solve_ivp(DX, [0, tend], [5, -2, -5], t_eval=np.arange(0, tend, stp))
xp, yp, zp = sol.y[0], sol.y[1], sol.y[2]  # components of the solution
print('hek')
# 绘图模块：时域图
timedisp = 200  # 时域波形图绘制最后200s
t = np.int32(timedisp / stp)
print(np.arange(t)[-20:])
fig, ax = plt.subplots(1)
ax.plot(np.arange(t), xp[-t:], label='x')
ax.plot(np.arange(t), yp[-t:], label='y')
ax.plot(np.arange(t), zp[-t:], label='z')
ax.legend()
ax.grid(True)
ax.set_title('waveform in time domain')
ax.set_xlim([0,200000])
plt.xticks(ticks=range(0, 200001, 200000 // 5), labels=[0, 40, 80, 120, 160, 200])
plt.show()

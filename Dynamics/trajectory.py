import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return np.tanh(x)

def DX(t,x):
    A = np.array([-1,-1,-1],dtype = np.float64)
    A1 = np.array([3.8, -1.9, 0.7],dtype = np.float64)
    A2 = np.array([2.5,0.06,1],dtype = np.float64)
    A3 = np.array([-6.6,1.3,0.07],dtype = np.float64)
    I =  np.array([0,0,0],dtype = np.float64)
    dx = [A[0]*x[0] + A1[0]*f(x[0]) + A1[1]*f(x[1]) + A1[2]*f(x[2]) + I[0],
          A[1]*x[1] + A2[0]*f(x[0]) + A2[1]*f(x[1]) + A2[2]*f(x[2]) + I[1],
          A[2]*x[2] + A3[0]*f(x[0]) + A3[1]*f(x[1]) + A3[2]*f(x[2]) + I[2]]
    return dx

tend = 700
stp = 0.001  # int step
sta = np.int32(tend / stp * 0.8)  # use stable points (the latter 80%)
sol = si.solve_ivp(DX, [0, tend], [5, -2, -5], t_eval=np.arange(0, tend, stp))
xp,yp,zp = sol.y[0][-sta:], sol.y[1][-sta:], sol.y[2][-sta:] # components of the solution

# 绘图模块1 绘制空间Trajectory
fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot(xp,yp,zp,linewidth = 0.5)
ax.set_title('trajectory')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

# # 绘图模块2 在三个投影面上的图片
# fig, ax = plt.subplots(1,3)
# ax[0].plot(xp,yp)
# ax[0].set_xlabel('x')
# ax[0].set_ylabel('y')
# ax[0].grid(True)
# ax[1].plot(xp,zp)
# ax[1].set_xlabel('x')
# ax[1].set_ylabel('z')
# ax[1].grid(True)
# ax[2].plot(yp,zp)
# ax[2].set_xlabel('y')
# ax[2].set_ylabel('z')
# ax[2].grid(True)
# plt.show()


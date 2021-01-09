import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt

'''
double side Poincare Map graph.
Author: 陈长, 3120104099
Date: Dec 14, 2020
'''

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

tval = np.arange(0,7000,0.001)
# print(tval.shape)
sol = si.solve_ivp(DX, [0,7000], [5,-2,-5],t_eval=tval)
data = np.vstack((sol.y[0][-1000001:-1], sol.y[1][-1000001:-1], sol.y[2][-1000001:-1])) # 注意：这里需要是一个矩阵外面套一个方括号
ind = np.array([],dtype=np.int32)

reg = np.sign(data[2,0] - 1.3)
print('起始点的z符号为：',reg)

for i,e in enumerate(data[2] - 1.3):
    if np.sign(e) * reg < 0:
        ind = np.append(ind, i)
        reg = np.sign(e)
print('绘制散点图的点数为：',ind.shape)

# 绘图模块2: 双边Poincare Map
fig, ax = plt.subplots(1)
ax.scatter(data[0,ind],data[1,ind])
ax.set_title('dual-side Poincare(PoinPlane: z = 1.3)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
plt.show()


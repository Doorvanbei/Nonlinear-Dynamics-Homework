import numpy as np
import scipy.optimize as op
'''
Solve a nonlinear system to find the fixed point (if there is any).
Author: 陈长, 3120104099
Date: Dec 14, 2020
'''
def f(x):
    return np.tanh(x)

def DX(x):
    A = np.array([-1, -1, -1], dtype=np.float64)
    A1 = np.array([3.8, -1.9, 0.7], dtype=np.float64)
    A2 = np.array([2.5, 0.06, 1], dtype=np.float64)
    A3 = np.array([-6.6, 1.3, 0.07], dtype=np.float64)
    I = np.array([0, 0, 0], dtype=np.float64)
    dx = [A[0] * x[0] + A1[0] * f(x[0]) + A1[1] * f(x[1]) + A1[2] * f(x[2]) + I[0],
          A[1] * x[1] + A2[0] * f(x[0]) + A2[1] * f(x[1]) + A2[2] * f(x[2]) + I[1],
          A[2] * x[2] + A3[0] * f(x[0]) + A3[1] * f(x[1]) + A3[2] * f(x[2]) + I[2]]
    return dx


r = np.random.random(3)*100 - 50
print(r)
sol = op.root(DX,r)
print(sol.x)
print(DX(sol.x))
print(sol.fun) # err


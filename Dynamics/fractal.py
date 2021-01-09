import scipy.integrate as si
import numpy as np
import matplotlib.pyplot as plt

'''
fractal ('Capacity') calculation of strange attractor.
Author: 陈长, 3120104099
Date: Dec 14, 2020
'''
def mkp(x, xyzrange, tik):
    '''
    MarkPoint
    :param x: A point in 3-D space, like [1,2,3]
    :param xyzrange: A 6-element list
    :param tik: coordinate tik
    :return: index shown the specific cubic
    '''
    xmin, xmax, ymin, ymax, zmin, zmax = xyzrange
    ckx = np.arange(xmin, xmax + tik, tik) - x[0] > 0
    cky = np.arange(ymin, ymax + tik, tik) - x[1] > 0
    ckz = np.arange(zmin, zmax + tik, tik) - x[2] > 0
    return np.where(ckx)[0][0] - 1, np.where(cky)[0][0] - 1, np.where(ckz)[0][0] - 1

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

tend = 700
stp = 0.001  # int step
sta = np.int32(tend / stp * 0.8)  # use stable points (the latter 80%)
sol = si.solve_ivp(DX, [0, tend], [5, -2, -5], t_eval=np.arange(0, tend, stp))
# xp,yp,zp = sol.y[0][-sta:], sol.y[1][-sta:], sol.y[2][-sta:] # components of the solution
xyzrange = (xmin, xmax, ymin, ymax, zmin, zmax) = -2, 1, -1.5, 1, -2.5, 5  # boundary for the strange attractor
tik = 1e-5 # diameter
PSet = set()
Dlist = []  # record the 'capacity' dimension of each diameter
while 1:
    print('tik:', tik)
    tx, ty, tz = np.arange(xmin, xmax, tik), np.arange(ymin, ymax, tik), np.arange(zmin, zmax, tik)  # divide the space
    for i in range(-sta, 0): # time spent here is long
        PSet.add(mkp(sol.y[:, i], xyzrange, tik))
    Dlist.append(np.log(len(PSet)) / np.log(1 / tik))
    print('dim:', Dlist[-1])
    tik /= 2
    PSet = set()
    if tik < 1e-9:  # for a tik too small, the RAM may be full.
        print('break')
        break
# # calculation results
0.1             & 3.424554976606713 \\
0.05            & 3.016921777277306 \\
0.025           & 2.725691763975478 \\
0.0125          & 2.5072957970153618\\
0.00625         & 2.3261459624292953\\
0.003125        & 2.163497109261765 \\
0.0015625       & 1.995470160247151 \\
0.00078125      & 1.831411660683802 \\
0.000390625     & 1.6816519925967714\\
0.0001953125    & 1.5490158457696181\\
9.765625e-05    & 1.433260750195553 \\
4.8828125e-05   & 1.333273824415717 \\
1e-05           & 1.14963760540124  \\
5e-06           & 1.0843530468056162\\
2.5e-06         & 1.0260846966837993\\
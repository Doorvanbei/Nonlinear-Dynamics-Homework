import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as si
'''
show a 3d-trajectory using animation form.
Author: 陈长, 3120104099
Date: Dec 14, 2020
'''
def f(x):
    return np.tanh(x)

def DX(t,x): # 即导函数向量
    A = np.array([-1,-1,-1],dtype = np.float64)
    A1 = np.array([3.8, -1.9, 0.7],dtype = np.float64)
    A2 = np.array([2.5,0.06,1],dtype = np.float64)
    A3 = np.array([-6.6,1.3,0.07],dtype = np.float64)
    I =  np.array([0,0,0],dtype = np.float64)
    dx = [A[0]*x[0] + A1[0]*f(x[0]) + A1[1]*f(x[1]) + A1[2]*f(x[2]) + I[0],
          A[1]*x[1] + A2[0]*f(x[0]) + A2[1]*f(x[1]) + A2[2]*f(x[2]) + I[1],
          A[2]*x[2] + A3[0]*f(x[0]) + A3[1]*f(x[1]) + A3[2]*f(x[2]) + I[2]]
    return dx

def update_lines(num, data_lines, lines): # 无需修改这个函数
    for line, data in zip(lines, data_lines):
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines


tval = np.arange(0,700,0.1) # 积分时间采样点
sol = si.solve_ivp(DX, [0,800], [5,-2,-5],t_eval=tval) # 其中第二个参数是积分时间间段，但是这里t_eval赋了值，所以主要取决于后者
data = [np.vstack((sol.y[0][-5001:-1], sol.y[1][-5001:-1], sol.y[2][-5001:-1]))] # 注意：这里需要是一个矩阵外面套一个方括号

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# 注意：这里的lines外部必须套有中括号，否则它是一个2D_line对象。
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([-1.5, 1.0])
ax.set_xlabel('X')
ax.set_ylim3d([-1.5, 1.0])
ax.set_ylabel('Y')
ax.set_zlim3d([-2.0, 4.0])
ax.set_zlabel('Z')
ax.set_title('animation of the trajectory')

# 注意：这里的5000是动画里有5000帧，对应曲线上5000个点，最后的interval参数越大，动画越慢
line_ani = animation.FuncAnimation(fig, update_lines, 5000, fargs=(data, lines), interval=1)

plt.show()
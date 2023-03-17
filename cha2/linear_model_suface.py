import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w_list = np.arange(0.0,4.1,0.1)
b_list = np.arange(-2.0,2.1,0.1)

W,B = np.meshgrid(w_list,b_list)

def forward(x):
    #这里进行的实际是矩阵运算，将W中的每个w*x
    return x * W + B


def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred -y)

w_list = []
b_list = []
mse_list = []
l_sum = 0
for x_val,y_val in zip(x_data,y_data):
    y_pred_val = forward(x_val)
    loss_val = loss(x_val,y_val)
    l_sum += loss_val
print('MSE=',l_sum /len(x_data))

#使用surface函数画出图像
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(W,B, l_sum / len(x_data), cmap=cm.coolwarm)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
#add a colorbar with map
fig.colorbar(surf,shrink = 0.5,aspect = 5)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

W = np.arange(0.0,4.1,0.1)
B = np.arange(-2.0,2.1,0.1)

def forward(x):
    return x * w + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred -y)

w_list = []
b_list = []
mse_list = []
for w in W:
    for b in B:
        print('w=',w)
        print('b=',b)
        l_sum = 0
        for x_val,y_val in zip(x_data,y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val,y_val)
            l_sum += loss_val
            print('\t',x_val,y_val,y_pred_val,loss_val)
    #除以样本总数转变为MSE
        print('MSE=',l_sum /len(x_data))
        b_list.append(b)
        mse_list.append(l_sum / len(x_data))
        w_list.append(w)
#2D的训练图像
# plt.plot(w_list,mse_list)
# plt.title('The process of Loss changing')
# plt.ylabel('Loss')
# plt.xlabel('w')
# plt.show()

#3D的训练图像
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_trisurf(w_list,b_list,mse_list,linewidth=0,antialiased=True)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('mse')
plt.show()


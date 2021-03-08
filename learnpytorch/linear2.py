
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred-y)

w_list = np.arange(0.0, 4.0, 0.1)
b_list = np.arange(-2.0, 2.0, 0.1)
mse_list = []

for w in np.arange(0.0, 4.0, 0.1):
    for b in np.arange(-2, 2.0, 0.1):
        print('w = %.1f' %w,'\tb = %.1f' %b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE = ',l_sum / 3)
        mse_list.append(l_sum / 3)

fig = plt.figure()
ax = Axes3D(fig)
print('mes_len = ',min(mse_list))
X,Y = np.meshgrid(w_list, b_list)
Z = np.array(mse_list).reshape(40, 40)
ax.plot_surface(X, Y, Z,
            rstride=1,
            cstride=1,
            cmap = plt.get_cmap('rainbow'))
# ax.set_zlim(-2, 2)
plt.title("3D graph")
plt.xlabel("w_list")
plt.ylabel("b_list")
plt.show()
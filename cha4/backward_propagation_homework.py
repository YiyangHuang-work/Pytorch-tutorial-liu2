import torch
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0,4.0]
y_data = [2.0,4,0,6.0,8.0]

w1 = torch.tensor([1.0],requires_grad=True)
w2 = torch.tensor([-4.0],requires_grad=True)
b = torch.tensor([5.0],requires_grad=True)

def forward(x):
    return w1 * x**2 + w2 * x + b

def loss(x,y):
    y_pred = forward(x)
    return (y - y_pred)**2

print("Predict (before trainning)",5,forward(5).item())
loss_list = []
for epoch in range(1000):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        w1.data -= 0.0001 * w1.grad.data
        w2.data -= 0.0001 * w2.grad.data
        b.data -= 0.0001 * b.grad.data

        print('\tgrad:', x, y, w1.grad.item(),w2.grad.item(),b.grad.item())  # 避免创造计算图

        w1.grad.zero_()
        w2.grad.zero_()
        b.grad.zero_()
    print("Progess ",epoch,l.item(),w1.item(),w2.item(),b.item())
    loss_list.append(l.item())
print("Predict (after trainning)",5,forward(5).item())

plt.plot(loss_list)
plt.title("Loss in epoch")
plt.xlabel(epoch)
plt.ylabel("loss")
plt.show(block=True)

import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x * w

def cost(xs,ys):
    res = 0
    for x,y in zip(xs,ys):
        res += (y - forward(x))**2
    return res / len(xs)

def grad(xs,ys):
    res = 0
    for x,y in zip(xs,ys):
        res += 2 * x * (forward(x) - y)
    return res / len(xs)

print("Predict (before trainning)",4,forward(4))
cost_list = []
for epoch in range(100):
    cost_val = cost(x_data,y_data)
    cost_list.append(cost_val)
    grad_val = grad(x_data,y_data)
    w -= 0.01 * grad_val
    print('Epoch:',epoch,'w=',w,'loss=',cost_val)
print("Predict (after trainning)",4,forward(4))

plt.title("Cost in each epoch")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.plot(cost_list)
plt.grid(linestyle='-')
plt.show()
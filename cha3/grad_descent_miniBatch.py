import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x * w

def cost_mini(x,y):
    return (y - forward(x))**2

def cost(xs,ys):
    res = 0
    for x,y in zip(xs,ys):
        res += (y - forward(x))**2
    return res / len(xs)

def grad(xs,ys):
    return 2 * x * (forward(x) - y)

def smooth(x,beta):
    for i in range(len(x) - 1):
        x[i + 1] = x[i] * beta + (1 - beta) * x[i + 1]

print("Predict (before trainning)",4,forward(4))
cost_list = []
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        grad_val = grad(x,y)
        w -= 0.01 * grad_val
        print("\tgrad:",x,y,grad)
        l = cost_mini(x,y)
        cost_list.append(l)
    print('Epoch:',epoch,'w=',w,'loss=',cost_mini)
print("Predict (after trainning)",4,forward(4))

smooth(cost_list,0.8)
plt.plot(cost_list)
plt.title("Cost in each epoch")
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.grid(linestyle='-')
plt.show()
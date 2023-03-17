import torch
import matplotlib.pyplot as plt

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = LinearModel()

criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

loss_list = []
for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    loss_list.append(loss.item())
    print(epoch,loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w= ',model.linear.weight.item())
print('b= ',model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ',y_test.data)

plt.plot(loss_list)
plt.title('The cost of Adam')
plt.xlabel("epoch")
plt.ylabel("cost")
plt.show(block=True)
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

x_data = torch.tensor([[1.0],[2.0],[3.0]])
y_data = torch.tensor([[0.0],[0.0],[1.0]])

class LogisticModel(torch.nn.Module):
    def __init__(self):
        super (LogisticModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

model = LogisticModel()

criterion = torch.nn.BCELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0,10,200,dtype=float)
x_t = torch.tensor(x).view((200,1))
x_t = x_t.type(torch.float32)
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show(block=True)


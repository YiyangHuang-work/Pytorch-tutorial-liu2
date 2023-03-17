import torch
import numpy as np
import matplotlib.pyplot as plt

xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype = np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])#保证得到的是矩阵，否则得到的会是一个向量

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.activate = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return x

model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

loss_list = []
for epoch in range(1000):
    #Forward
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    loss_list.append(loss.item())
    print(epoch, loss.item())
    #Backward
    optimizer.zero_grad()
    loss.backward()

    #Update
    optimizer.step()

plt.title('The Loss of result')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss_list)
plt.show(block=True)
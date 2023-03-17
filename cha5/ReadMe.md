## 用Pytorch实现一个简单的$wx + b$

### Pytorch API

​		值得介绍的是pytorch中的核心就是去对Module模块去进行适当的重写。主要override的是\__init__方法和forward()方法，在init阶段主要是进行计算图的构建过程，而在foward中，实际上进行的是对类使用对应的call函数去使用对应的方法：

```python
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
```

这里linear(x)的使用如下代码所示：是对当前类的call方法的使用

```python
class Foobar:
    def __init__(self):
        pass

    def __call__(self,*args,**kwargs):
        print("Hello" + str(args[0]))

def func(*args,**kwargs):
    print(args)
    print(kwargs)

func(1,2,3,4,x=3,y=5)
foobar = Foobar()
foobar(1,2,3)
```

代码运行结果如下：

```python
(1, 2, 3, 4)
{'x': 3, 'y': 5}
Hello1
```

这里面的foobar实际上就是Foobar类的一个实例，后面的括号就是对应调用了call方法，这里相当于对其进行了计算，也就是前向传播，这里没有显式的更新操作，而是使用optimizer优化器来完成参数的更新：

```python
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)
```

详细情况见代码
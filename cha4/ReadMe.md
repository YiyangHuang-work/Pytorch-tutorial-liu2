### 反向传播算法

#### 课程代码

​		在课程代码中使用pytorch实现的线性回归问题，下面的图片体现了我们为什么要使用非线性结点。因为如果仍旧使用线性，我们跟先前的方法是没有区别的，$W_{1}$和$W_2$总是能合成一个新的$W$。我们可以做这样的化简，但是如果加了非线性结点就不可以这么做。

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20230304215947675.png" alt="image-20230304215947675" style="zoom:50%;" />

​		值得注意的是这次课程中对于$pytorch$中$tensor$结构的讲述，$tensor$结构中包括$data$和$grad$两个部分，一个负责在$forward$中存值，一个负责在$backward$​中建立引用。使用时要标明

```python
w1 = torch.tensor([1.0],requires_grad=True)
```

<img src="C:\Users\86156\AppData\Roaming\Typora\typora-user-images\image-20230304233741455.png" alt="image-20230304233741455" style="zoom:50%;" />

​		除此之外，要理解我们在构造$forward$和$loss$函数时我们构造的实际是一种计算图，因为w是一个$tensor$,所以不能当成一个简单的数字去看，如果要使用数值的时候可以使用$.item()$把表项提取出来，例如在计算总的损失时，一定不能使用$tensor$不然会构造一个十分巨大的计算图导致程序无法储存并运行。

```python
def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y - y_pred)**2
```

#### 作业代码

​		在作业的代码中整体来讲和上述的过程是类似的，只不过这里出现了学习率和过拟合的问题，要选择好合适的迭代次数，并不能一味的追求$loss$最小而不断增加$epoch$的数目，那样可能最终的预测值反而会不好，这里要选择合适的值和学习率进行运算，调参的快乐~
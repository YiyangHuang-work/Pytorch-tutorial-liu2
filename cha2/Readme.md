## 学习总结

### matplotlib画三维图像

**官方文件链接**

https://matplotlib.org/stable/tutorials/toolkits/mplot3d.ht

#### 使用surface plots

##### 这里有一个要点是meshgrid函数的使用：

https://blog.csdn.net/lllxxq141592654/article/details/81532855

主要是将当前的一维数据改写为二维的形式，因为surface的输入要求二维模式

```python
def forward(x):
    #这里进行的实际是矩阵运算，将W中的每个w*x
    return x * W + B
```

这里做的是矩阵运算，x是数值做一个二维的乘积再与b累加

```python
def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred -y)
```

这里的y也是做了矩阵运算，但是这里要强调：==np中的 * 和dot()是不同的==，*是对应位置的元素相乘，而dot则是矩阵乘法，乘号要求矩阵的维数是相同的，而dot则是通常意义上的矩阵乘法，两者有着本质上的区别，使用时候需要注意。

**list和numpy.array的差别**

list获得序列长度的方式是len，没有shape选项。而array有shape选项，可以reshape，list可以在使用时使用np.array()进行转化，就可以改变维度。



详细使用代码请见linear_model_surface.py，来自https://www.bilibili.com/video/BV1Y7411d7Ys?p=2

#### 使用 trisurf plots

这个就相对好一些，它要求一维即可，所以直接将得到的对应输入放入即可，详见代码linear_model.py


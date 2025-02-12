None
# 线性回归
:label:`sec_linear_regression`

*回归*（regression）是能为一个或多个自变量与因变量之间关系建模的一类方法。
在自然科学和社会科学领域，回归经常用来表示输入和输出之间的关系。

在机器学习领域中的大多数任务通常都与*预测*（prediction）有关。
当我们想预测一个数值时，就会涉及到回归问题。
常见的例子包括：预测价格（房屋、股票等）、预测住院时间（针对住院病人等）、
预测需求（零售销量等）。
但不是所有的*预测*都是回归问题。
在后面的章节中，我们将介绍分类问题。分类问题的目标是预测数据属于一组类别中的哪一个。

## 线性回归的基本元素

*线性回归*（linear regression）可以追溯到19世纪初，
它在回归的各种标准工具中最简单而且最流行。
线性回归基于几个简单的假设：
首先，假设自变量$\mathbf{x}$和因变量$y$之间的关系是线性的，
即$y$可以表示为$\mathbf{x}$中元素的加权和，这里通常允许包含观测值的一些噪声；
其次，我们假设任何噪声都比较正常，如噪声遵循正态分布。

为了解释*线性回归*，我们举一个实际的例子：
我们希望根据房屋的面积（平方英尺）和房龄（年）来估算房屋价格（美元）。
为了开发一个能预测房价的模型，我们需要收集一个真实的数据集。
这个数据集包括了房屋的销售价格、面积和房龄。
在机器学习的术语中，该数据集称为*训练数据集*（training data set）
或*训练集*（training set）。
每行数据（比如一次房屋交易相对应的数据）称为*样本*（sample），
也可以称为*数据点*（data point）或*数据样本*（data instance）。
我们把试图预测的目标（比如预测房屋价格）称为*标签*（label）或*目标*（target）。
预测所依据的自变量（面积和房龄）称为*特征*（feature）或*协变量*（covariate）。

通常，我们使用$n$来表示数据集中的样本数。
对索引为$i$的样本，其输入表示为$\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$，
其对应的标签是$y^{(i)}$。

### 线性模型
:label:`subsec_linear_model`

线性假设是指目标（房屋价格）可以表示为特征（面积和房龄）的加权和，如下面的式子：

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

 :eqref:`eq_price-area`中的$w_{\mathrm{area}}$和$w_{\mathrm{age}}$
称为*权重*（weight），权重决定了每个特征对我们预测值的影响。
$b$称为*偏置*（bias）、*偏移量*（offset）或*截距*（intercept）。
偏置是指当所有特征都取值为0时，预测值应该为多少。
即使现实中不会有任何房子的面积是0或房龄正好是0年，我们仍然需要偏置项。
如果没有偏置项，我们模型的表达能力将受到限制。
严格来说， :eqref:`eq_price-area`是输入特征的一个
*仿射变换*（affine transformation）。
仿射变换的特点是通过加权和对特征进行*线性变换*（linear transformation），
并通过偏置项来进行*平移*（translation）。

给定一个数据集，我们的目标是寻找模型的权重$\mathbf{w}$和偏置$b$，
使得根据模型做出的预测大体符合数据里的真实价格。
输出的预测值由输入特征通过*线性模型*的仿射变换决定，仿射变换由所选权重和偏置确定。

而在机器学习领域，我们通常使用的是高维数据集，建模时采用线性代数表示法会比较方便。
当我们的输入包含$d$个特征时，我们将预测结果$\hat{y}$
（通常使用“尖角”符号表示$y$的估计值）表示为：

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

将所有特征放到向量$\mathbf{x} \in \mathbb{R}^d$中，
并将所有权重放到向量$\mathbf{w} \in \mathbb{R}^d$中，
我们可以用点积形式来简洁地表达模型：

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

在 :eqref:`eq_linreg-y`中，
向量$\mathbf{x}$对应于单个数据样本的特征。
用符号表示的矩阵$\mathbf{X} \in \mathbb{R}^{n \times d}$
可以很方便地引用我们整个数据集的$n$个样本。
其中，$\mathbf{X}$的每一行是一个样本，每一列是一种特征。

对于特征集合$\mathbf{X}$，预测值$\hat{\mathbf{y}} \in \mathbb{R}^n$
可以通过矩阵-向量乘法表示为：

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b$$

这个过程中的求和将使用广播机制
（广播机制在 :numref:`subsec_broadcasting`中有详细介绍）。
给定训练数据特征$\mathbf{X}$和对应的已知标签$\mathbf{y}$，
线性回归的目标是找到一组权重向量$\mathbf{w}$和偏置$b$：
当给定从$\mathbf{X}$的同分布中取样的新样本特征时，
这组权重向量和偏置能够使得新样本预测标签的误差尽可能小。

虽然我们相信给定$\mathbf{x}$预测$y$的最佳模型会是线性的，
但我们很难找到一个有$n$个样本的真实数据集，其中对于所有的$1 \leq i \leq n$，$y^{(i)}$完全等于$\mathbf{w}^\top \mathbf{x}^{(i)}+b$。
无论我们使用什么手段来观察特征$\mathbf{X}$和标签$\mathbf{y}$，
都可能会出现少量的观测误差。
因此，即使确信特征与标签的潜在关系是线性的，
我们也会加入一个噪声项来考虑观测误差带来的影响。

在开始寻找最好的*模型参数*（model parameters）$\mathbf{w}$和$b$之前，
我们还需要两个东西：
（1）一种模型质量的度量方式；
（2）一种能够更新模型以提高模型预测质量的方法。

### 损失函数

在我们开始考虑如何用模型*拟合*（fit）数据之前，我们需要确定一个拟合程度的度量。
*损失函数*（loss function）能够量化目标的*实际*值与*预测*值之间的差距。
通常我们会选择非负数作为损失，且数值越小表示损失越小，完美预测时的损失为0。
回归问题中最常用的损失函数是平方误差函数。
当样本$i$的预测值为$\hat{y}^{(i)}$，其相应的真实标签为$y^{(i)}$时，
平方误差可以定义为以下公式：

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

常数$\frac{1}{2}$不会带来本质的差别，但这样在形式上稍微简单一些
（因为当我们对损失函数求导后常数系数为1）。
由于训练数据集并不受我们控制，所以经验误差只是关于模型参数的函数。
为了进一步说明，来看下面的例子。
我们为一维情况下的回归问题绘制图像，如 :numref:`fig_fit_linreg`所示。

![用线性模型拟合数据。](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

由于平方误差函数中的二次方项，
估计值$\hat{y}^{(i)}$和观测值$y^{(i)}$之间较大的差异将导致更大的损失。
为了度量模型在整个数据集上的质量，我们需计算在训练集$n$个样本上的损失均值（也等价于求和）。

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

在训练模型时，我们希望寻找一组参数（$\mathbf{w}^*, b^*$），
这组参数能最小化在所有训练样本上的总损失。如下式：

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### 解析解

线性回归刚好是一个很简单的优化问题。
与我们将在本书中所讲到的其他大部分模型不同，线性回归的解可以用一个公式简单地表达出来，
这类解叫作解析解（analytical solution）。
首先，我们将偏置$b$合并到参数$\mathbf{w}$中，合并方法是在包含所有参数的矩阵中附加一列。
我们的预测问题是最小化$\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$。
这在损失平面上只有一个临界点，这个临界点对应于整个区域的损失极小点。
将损失关于$\mathbf{w}$的导数设为0，得到解析解：

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

像线性回归这样的简单问题存在解析解，但并不是所有的问题都存在解析解。
解析解可以进行很好的数学分析，但解析解对问题的限制很严格，导致它无法广泛应用在深度学习里。

### 随机梯度下降

即使在我们无法得到解析解的情况下，我们仍然可以有效地训练模型。
在许多任务上，那些难以优化的模型效果要更好。
因此，弄清楚如何训练这些难以优化的模型是非常重要的。

本书中我们用到一种名为*梯度下降*（gradient descent）的方法，
这种方法几乎可以优化所有深度学习模型。
它通过不断地在损失函数递减的方向上更新参数来降低误差。

梯度下降最简单的用法是计算损失函数（数据集中所有样本的损失均值）
关于模型参数的导数（在这里也可以称为梯度）。
但实际中的执行可能会非常慢：因为在每一次更新参数之前，我们必须遍历整个数据集。
因此，我们通常会在每次需要计算更新的时候随机抽取一小批样本，
这种变体叫做*小批量随机梯度下降*（minibatch stochastic gradient descent）。

在每次迭代中，我们首先随机抽样一个小批量$\mathcal{B}$，
它是由固定数量的训练样本组成的。
然后，我们计算小批量的平均损失关于模型参数的导数（也可以称为梯度）。
最后，我们将梯度乘以一个预先确定的正数$\eta$，并从当前参数的值中减掉。

我们用下面的数学公式来表示这一更新过程（$\partial$表示偏导数）：

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

总结一下，算法的步骤如下：
（1）初始化模型参数的值，如随机初始化；
（2）从数据集中随机抽取小批量样本且在负梯度的方向上更新参数，并不断迭代这一步骤。
对于平方损失和仿射变换，我们可以明确地写成如下形式:

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

公式 :eqref:`eq_linreg_batch_update`中的$\mathbf{w}$和$\mathbf{x}$都是向量。
在这里，更优雅的向量表示法比系数表示法（如$w_1, w_2, \ldots, w_d$）更具可读性。
$|\mathcal{B}|$表示每个小批量中的样本数，这也称为*批量大小*（batch size）。
$\eta$表示*学习率*（learning rate）。
批量大小和学习率的值通常是手动预先指定，而不是通过模型训练得到的。
这些可以调整但不在训练过程中更新的参数称为*超参数*（hyperparameter）。
*调参*（hyperparameter tuning）是选择超参数的过程。
超参数通常是我们根据训练迭代结果来调整的，
而训练迭代结果是在独立的*验证数据集*（validation dataset）上评估得到的。

在训练了预先确定的若干迭代次数后（或者直到满足某些其他停止条件后），
我们记录下模型参数的估计值，表示为$\hat{\mathbf{w}}, \hat{b}$。
但是，即使我们的函数确实是线性的且无噪声，这些估计值也不会使损失函数真正地达到最小值。
因为算法会使得损失向最小值缓慢收敛，但却不能在有限的步数内非常精确地达到最小值。

线性回归恰好是一个在整个域中只有一个最小值的学习问题。
但是对于像深度神经网络这样复杂的模型来说，损失平面上通常包含多个最小值。
深度学习实践者很少会去花费大力气寻找这样一组参数，使得在*训练集*上的损失达到最小。
事实上，更难做到的是找到一组参数，这组参数能够在我们从未见过的数据上实现较低的损失，
这一挑战被称为*泛化*（generalization）。

### 用模型进行预测

给定“已学习”的线性回归模型$\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$，
现在我们可以通过房屋面积$x_1$和房龄$x_2$来估计一个（未包含在训练数据中的）新房屋价格。
给定特征估计目标的过程通常称为*预测*（prediction）或*推断*（inference）。

本书将尝试坚持使用*预测*这个词。
虽然*推断*这个词已经成为深度学习的标准术语，但其实*推断*这个词有些用词不当。
在统计学中，*推断*更多地表示基于数据集估计参数。
当深度学习从业者与统计学家交谈时，术语的误用经常导致一些误解。

## 矢量化加速

在训练我们的模型时，我们经常希望能够同时处理整个小批量的样本。
为了实现这一点，需要(**我们对计算进行矢量化，
从而利用线性代数库，而不是在Python中编写开销高昂的for循环**)。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

```{.python .input  n=3}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
import math
import paddle
import numpy as np
import time
```

为了说明矢量化为什么如此重要，我们考虑(**对向量相加的两种方法**)。
我们实例化两个全为1的10000维向量。
在一种方法中，我们将使用Python的for循环遍历向量；
在另一种方法中，我们将依赖对`+`的调用。

```{.python .input  n=5}
#@tab mxnet, pytorch, tensorflow
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

```{.json .output n=5}
[
 {
  "ename": "AttributeError",
  "evalue": "module 'd2l.paddle' has no attribute 'ones'",
  "output_type": "error",
  "traceback": [
   "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[1;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
   "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_5836/682604306.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[1;31m#@tab all\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      2\u001b[0m \u001b[0mn\u001b[0m \u001b[1;33m=\u001b[0m \u001b[1;36m10000\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 3\u001b[1;33m \u001b[0ma\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mones\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      4\u001b[0m \u001b[0mb\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mones\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mn\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
   "\u001b[1;31mAttributeError\u001b[0m: module 'd2l.paddle' has no attribute 'ones'"
  ]
 }
]
```

```{.python .input  n=7}
#@tab paddle
n=10000
a=paddle.ones([n])
b=paddle.ones([n])
```

由于在本书中我们将频繁地进行运行时间的基准测试，所以[**我们定义一个计时器**]：

```{.python .input  n=8}
#@tab all
class Timer:  #@save
    """记录多次运行时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()
```

现在我们可以对工作负载进行基准测试。

首先，[**我们使用for循环，每次执行一位的加法**]。

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

```{.python .input  n=10}
#@tab paddle
c = paddle.zeros([n])
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "'0.82600 sec'"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

(**或者，我们使用重载的`+`运算符来计算按元素的和**)。

```{.python .input  n=11}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "'0.02699 sec'"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

结果很明显，第二种方法比第一种方法快得多。
矢量化代码通常会带来数量级的加速。
另外，我们将更多的数学运算放到库中，而无须自己编写那么多的计算，从而减少了出错的可能性。

## 正态分布与平方损失
:label:`subsec_normal_distribution_and_squared_loss`

接下来，我们通过对噪声分布的假设来解读平方损失目标函数。

正态分布和线性回归之间的关系很密切。
正态分布（normal distribution），也称为*高斯分布*（Gaussian distribution），
最早由德国数学家高斯（Gauss）应用于天文学研究。
简单的说，若随机变量$x$具有均值$\mu$和方差$\sigma^2$（标准差$\sigma$），其正态分布概率密度函数如下：

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

下面[**我们定义一个Python函数来计算正态分布**]。

```{.python .input  n=12}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

我们现在(**可视化正态分布**)。

```{.python .input  n=13}
#@tab all
# 再次使用numpy进行可视化
x = np.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

```{.json .output n=13}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "D:\\Anaconda3\\envs\\d2l\\lib\\site-packages\\d2l\\paddle.py:35: DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23, directly use `matplotlib_inline.backend_inline.set_matplotlib_formats()`\n  display.set_matplotlib_formats('svg')\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Created with matplotlib (https://matplotlib.org/) -->\r\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 302.08125 180.65625\" width=\"302.08125pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n <metadata>\r\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\r\n   <cc:Work>\r\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\r\n    <dc:date>2022-01-20T17:24:01.327485</dc:date>\r\n    <dc:format>image/svg+xml</dc:format>\r\n    <dc:creator>\r\n     <cc:Agent>\r\n      <dc:title>Matplotlib v3.3.3, https://matplotlib.org/</dc:title>\r\n     </cc:Agent>\r\n    </dc:creator>\r\n   </cc:Work>\r\n  </rdf:RDF>\r\n </metadata>\r\n <defs>\r\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\r\n </defs>\r\n <g id=\"figure_1\">\r\n  <g id=\"patch_1\">\r\n   <path d=\"M 0 180.65625 \r\nL 302.08125 180.65625 \r\nL 302.08125 0 \r\nL 0 0 \r\nz\r\n\" style=\"fill:none;\"/>\r\n  </g>\r\n  <g id=\"axes_1\">\r\n   <g id=\"patch_2\">\r\n    <path d=\"M 43.78125 143.1 \r\nL 294.88125 143.1 \r\nL 294.88125 7.2 \r\nL 43.78125 7.2 \r\nz\r\n\" style=\"fill:#ffffff;\"/>\r\n   </g>\r\n   <g id=\"matplotlib.axis_1\">\r\n    <g id=\"xtick_1\">\r\n     <g id=\"line2d_1\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 71.511736 143.1 \r\nL 71.511736 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_2\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL 0 3.5 \r\n\" id=\"m3a9a4da124\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"71.511736\" xlink:href=\"#m3a9a4da124\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_1\">\r\n      <!-- \u22126 -->\r\n      <g transform=\"translate(64.140642 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.59375 35.5 \r\nL 73.1875 35.5 \r\nL 73.1875 27.203125 \r\nL 10.59375 27.203125 \r\nz\r\n\" id=\"DejaVuSans-8722\"/>\r\n        <path d=\"M 33.015625 40.375 \r\nQ 26.375 40.375 22.484375 35.828125 \r\nQ 18.609375 31.296875 18.609375 23.390625 \r\nQ 18.609375 15.53125 22.484375 10.953125 \r\nQ 26.375 6.390625 33.015625 6.390625 \r\nQ 39.65625 6.390625 43.53125 10.953125 \r\nQ 47.40625 15.53125 47.40625 23.390625 \r\nQ 47.40625 31.296875 43.53125 35.828125 \r\nQ 39.65625 40.375 33.015625 40.375 \r\nz\r\nM 52.59375 71.296875 \r\nL 52.59375 62.3125 \r\nQ 48.875 64.0625 45.09375 64.984375 \r\nQ 41.3125 65.921875 37.59375 65.921875 \r\nQ 27.828125 65.921875 22.671875 59.328125 \r\nQ 17.53125 52.734375 16.796875 39.40625 \r\nQ 19.671875 43.65625 24.015625 45.921875 \r\nQ 28.375 48.1875 33.59375 48.1875 \r\nQ 44.578125 48.1875 50.953125 41.515625 \r\nQ 57.328125 34.859375 57.328125 23.390625 \r\nQ 57.328125 12.15625 50.6875 5.359375 \r\nQ 44.046875 -1.421875 33.015625 -1.421875 \r\nQ 20.359375 -1.421875 13.671875 8.265625 \r\nQ 6.984375 17.96875 6.984375 36.375 \r\nQ 6.984375 53.65625 15.1875 63.9375 \r\nQ 23.390625 74.21875 37.203125 74.21875 \r\nQ 40.921875 74.21875 44.703125 73.484375 \r\nQ 48.484375 72.75 52.59375 71.296875 \r\nz\r\n\" id=\"DejaVuSans-54\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-8722\"/>\r\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-54\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_2\">\r\n     <g id=\"line2d_3\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 104.145435 143.1 \r\nL 104.145435 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_4\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"104.145435\" xlink:href=\"#m3a9a4da124\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_2\">\r\n      <!-- \u22124 -->\r\n      <g transform=\"translate(96.774342 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 37.796875 64.3125 \r\nL 12.890625 25.390625 \r\nL 37.796875 25.390625 \r\nz\r\nM 35.203125 72.90625 \r\nL 47.609375 72.90625 \r\nL 47.609375 25.390625 \r\nL 58.015625 25.390625 \r\nL 58.015625 17.1875 \r\nL 47.609375 17.1875 \r\nL 47.609375 0 \r\nL 37.796875 0 \r\nL 37.796875 17.1875 \r\nL 4.890625 17.1875 \r\nL 4.890625 26.703125 \r\nz\r\n\" id=\"DejaVuSans-52\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-8722\"/>\r\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-52\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_3\">\r\n     <g id=\"line2d_5\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 136.779135 143.1 \r\nL 136.779135 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_6\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"136.779135\" xlink:href=\"#m3a9a4da124\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_3\">\r\n      <!-- \u22122 -->\r\n      <g transform=\"translate(129.408041 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 19.1875 8.296875 \r\nL 53.609375 8.296875 \r\nL 53.609375 0 \r\nL 7.328125 0 \r\nL 7.328125 8.296875 \r\nQ 12.9375 14.109375 22.625 23.890625 \r\nQ 32.328125 33.6875 34.8125 36.53125 \r\nQ 39.546875 41.84375 41.421875 45.53125 \r\nQ 43.3125 49.21875 43.3125 52.78125 \r\nQ 43.3125 58.59375 39.234375 62.25 \r\nQ 35.15625 65.921875 28.609375 65.921875 \r\nQ 23.96875 65.921875 18.8125 64.3125 \r\nQ 13.671875 62.703125 7.8125 59.421875 \r\nL 7.8125 69.390625 \r\nQ 13.765625 71.78125 18.9375 73 \r\nQ 24.125 74.21875 28.421875 74.21875 \r\nQ 39.75 74.21875 46.484375 68.546875 \r\nQ 53.21875 62.890625 53.21875 53.421875 \r\nQ 53.21875 48.921875 51.53125 44.890625 \r\nQ 49.859375 40.875 45.40625 35.40625 \r\nQ 44.1875 33.984375 37.640625 27.21875 \r\nQ 31.109375 20.453125 19.1875 8.296875 \r\nz\r\n\" id=\"DejaVuSans-50\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-8722\"/>\r\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-50\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_4\">\r\n     <g id=\"line2d_7\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 169.412834 143.1 \r\nL 169.412834 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_8\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"169.412834\" xlink:href=\"#m3a9a4da124\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_4\">\r\n      <!-- 0 -->\r\n      <g transform=\"translate(166.231584 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 31.78125 66.40625 \r\nQ 24.171875 66.40625 20.328125 58.90625 \r\nQ 16.5 51.421875 16.5 36.375 \r\nQ 16.5 21.390625 20.328125 13.890625 \r\nQ 24.171875 6.390625 31.78125 6.390625 \r\nQ 39.453125 6.390625 43.28125 13.890625 \r\nQ 47.125 21.390625 47.125 36.375 \r\nQ 47.125 51.421875 43.28125 58.90625 \r\nQ 39.453125 66.40625 31.78125 66.40625 \r\nz\r\nM 31.78125 74.21875 \r\nQ 44.046875 74.21875 50.515625 64.515625 \r\nQ 56.984375 54.828125 56.984375 36.375 \r\nQ 56.984375 17.96875 50.515625 8.265625 \r\nQ 44.046875 -1.421875 31.78125 -1.421875 \r\nQ 19.53125 -1.421875 13.0625 8.265625 \r\nQ 6.59375 17.96875 6.59375 36.375 \r\nQ 6.59375 54.828125 13.0625 64.515625 \r\nQ 19.53125 74.21875 31.78125 74.21875 \r\nz\r\n\" id=\"DejaVuSans-48\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_5\">\r\n     <g id=\"line2d_9\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 202.046534 143.1 \r\nL 202.046534 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_10\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"202.046534\" xlink:href=\"#m3a9a4da124\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_5\">\r\n      <!-- 2 -->\r\n      <g transform=\"translate(198.865284 157.698438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_6\">\r\n     <g id=\"line2d_11\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 234.680233 143.1 \r\nL 234.680233 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_12\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"234.680233\" xlink:href=\"#m3a9a4da124\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_6\">\r\n      <!-- 4 -->\r\n      <g transform=\"translate(231.498983 157.698438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-52\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_7\">\r\n     <g id=\"line2d_13\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 267.313932 143.1 \r\nL 267.313932 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_14\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"267.313932\" xlink:href=\"#m3a9a4da124\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_7\">\r\n      <!-- 6 -->\r\n      <g transform=\"translate(264.132682 157.698438)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-54\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_8\">\r\n     <!-- x -->\r\n     <g transform=\"translate(166.371875 171.376563)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 54.890625 54.6875 \r\nL 35.109375 28.078125 \r\nL 55.90625 0 \r\nL 45.3125 0 \r\nL 29.390625 21.484375 \r\nL 13.484375 0 \r\nL 2.875 0 \r\nL 24.125 28.609375 \r\nL 4.6875 54.6875 \r\nL 15.28125 54.6875 \r\nL 29.78125 35.203125 \r\nL 44.28125 54.6875 \r\nz\r\n\" id=\"DejaVuSans-120\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-120\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"matplotlib.axis_2\">\r\n    <g id=\"ytick_1\">\r\n     <g id=\"line2d_15\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 43.78125 136.922727 \r\nL 294.88125 136.922727 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_16\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL -3.5 0 \r\n\" id=\"m47fcfc1e47\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m47fcfc1e47\" y=\"136.922727\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_9\">\r\n      <!-- 0.0 -->\r\n      <g transform=\"translate(20.878125 140.721946)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.6875 12.40625 \r\nL 21 12.40625 \r\nL 21 0 \r\nL 10.6875 0 \r\nz\r\n\" id=\"DejaVuSans-46\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_2\">\r\n     <g id=\"line2d_17\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 43.78125 105.954474 \r\nL 294.88125 105.954474 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_18\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m47fcfc1e47\" y=\"105.954474\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_10\">\r\n      <!-- 0.1 -->\r\n      <g transform=\"translate(20.878125 109.753693)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 12.40625 8.296875 \r\nL 28.515625 8.296875 \r\nL 28.515625 63.921875 \r\nL 10.984375 60.40625 \r\nL 10.984375 69.390625 \r\nL 28.421875 72.90625 \r\nL 38.28125 72.90625 \r\nL 38.28125 8.296875 \r\nL 54.390625 8.296875 \r\nL 54.390625 0 \r\nL 12.40625 0 \r\nz\r\n\" id=\"DejaVuSans-49\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-49\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_3\">\r\n     <g id=\"line2d_19\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 43.78125 74.986221 \r\nL 294.88125 74.986221 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_20\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m47fcfc1e47\" y=\"74.986221\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_11\">\r\n      <!-- 0.2 -->\r\n      <g transform=\"translate(20.878125 78.78544)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_4\">\r\n     <g id=\"line2d_21\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 43.78125 44.017968 \r\nL 294.88125 44.017968 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_22\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m47fcfc1e47\" y=\"44.017968\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_12\">\r\n      <!-- 0.3 -->\r\n      <g transform=\"translate(20.878125 47.817187)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 40.578125 39.3125 \r\nQ 47.65625 37.796875 51.625 33 \r\nQ 55.609375 28.21875 55.609375 21.1875 \r\nQ 55.609375 10.40625 48.1875 4.484375 \r\nQ 40.765625 -1.421875 27.09375 -1.421875 \r\nQ 22.515625 -1.421875 17.65625 -0.515625 \r\nQ 12.796875 0.390625 7.625 2.203125 \r\nL 7.625 11.71875 \r\nQ 11.71875 9.328125 16.59375 8.109375 \r\nQ 21.484375 6.890625 26.8125 6.890625 \r\nQ 36.078125 6.890625 40.9375 10.546875 \r\nQ 45.796875 14.203125 45.796875 21.1875 \r\nQ 45.796875 27.640625 41.28125 31.265625 \r\nQ 36.765625 34.90625 28.71875 34.90625 \r\nL 20.21875 34.90625 \r\nL 20.21875 43.015625 \r\nL 29.109375 43.015625 \r\nQ 36.375 43.015625 40.234375 45.921875 \r\nQ 44.09375 48.828125 44.09375 54.296875 \r\nQ 44.09375 59.90625 40.109375 62.90625 \r\nQ 36.140625 65.921875 28.71875 65.921875 \r\nQ 24.65625 65.921875 20.015625 65.03125 \r\nQ 15.375 64.15625 9.8125 62.3125 \r\nL 9.8125 71.09375 \r\nQ 15.4375 72.65625 20.34375 73.4375 \r\nQ 25.25 74.21875 29.59375 74.21875 \r\nQ 40.828125 74.21875 47.359375 69.109375 \r\nQ 53.90625 64.015625 53.90625 55.328125 \r\nQ 53.90625 49.265625 50.4375 45.09375 \r\nQ 46.96875 40.921875 40.578125 39.3125 \r\nz\r\n\" id=\"DejaVuSans-51\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-51\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_5\">\r\n     <g id=\"line2d_23\">\r\n      <path clip-path=\"url(#p7438949e74)\" d=\"M 43.78125 13.049715 \r\nL 294.88125 13.049715 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_24\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m47fcfc1e47\" y=\"13.049715\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_13\">\r\n      <!-- 0.4 -->\r\n      <g transform=\"translate(20.878125 16.848934)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_14\">\r\n     <!-- p(x) -->\r\n     <g transform=\"translate(14.798438 85.185156)rotate(-90)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 18.109375 8.203125 \r\nL 18.109375 -20.796875 \r\nL 9.078125 -20.796875 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nz\r\nM 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-112\"/>\r\n       <path d=\"M 31 75.875 \r\nQ 24.46875 64.65625 21.28125 53.65625 \r\nQ 18.109375 42.671875 18.109375 31.390625 \r\nQ 18.109375 20.125 21.3125 9.0625 \r\nQ 24.515625 -2 31 -13.1875 \r\nL 23.1875 -13.1875 \r\nQ 15.875 -1.703125 12.234375 9.375 \r\nQ 8.59375 20.453125 8.59375 31.390625 \r\nQ 8.59375 42.28125 12.203125 53.3125 \r\nQ 15.828125 64.359375 23.1875 75.875 \r\nz\r\n\" id=\"DejaVuSans-40\"/>\r\n       <path d=\"M 8.015625 75.875 \r\nL 15.828125 75.875 \r\nQ 23.140625 64.359375 26.78125 53.3125 \r\nQ 30.421875 42.28125 30.421875 31.390625 \r\nQ 30.421875 20.453125 26.78125 9.375 \r\nQ 23.140625 -1.703125 15.828125 -13.1875 \r\nL 8.015625 -13.1875 \r\nQ 14.5 -2 17.703125 9.0625 \r\nQ 20.90625 20.125 20.90625 31.390625 \r\nQ 20.90625 42.671875 17.703125 53.65625 \r\nQ 14.5 64.65625 8.015625 75.875 \r\nz\r\n\" id=\"DejaVuSans-41\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"63.476562\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"102.490234\" xlink:href=\"#DejaVuSans-120\"/>\r\n      <use x=\"161.669922\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"line2d_25\">\r\n    <path clip-path=\"url(#p7438949e74)\" d=\"M 55.194886 136.922727 \r\nL 108.224648 136.813535 \r\nL 113.609208 136.566288 \r\nL 117.198915 136.184417 \r\nL 119.97278 135.668952 \r\nL 122.257139 135.025186 \r\nL 124.215161 134.257847 \r\nL 126.010014 133.330368 \r\nL 127.804868 132.138334 \r\nL 129.436552 130.779445 \r\nL 131.068237 129.113085 \r\nL 132.699922 127.093512 \r\nL 134.331607 124.67477 \r\nL 135.963292 121.812692 \r\nL 137.594977 118.467288 \r\nL 139.226662 114.605495 \r\nL 141.021516 109.733818 \r\nL 142.816369 104.197078 \r\nL 144.774391 97.412516 \r\nL 146.895582 89.247632 \r\nL 149.506278 78.224523 \r\nL 153.259153 61.239305 \r\nL 157.501534 42.275258 \r\nL 159.785893 33.113056 \r\nL 161.580746 26.820515 \r\nL 163.049263 22.424523 \r\nL 164.354611 19.173268 \r\nL 165.49679 16.884633 \r\nL 166.475801 15.362585 \r\nL 167.454812 14.263605 \r\nL 168.270655 13.679589 \r\nL 169.086497 13.401979 \r\nL 169.739171 13.401979 \r\nL 170.391845 13.599455 \r\nL 171.207688 14.122466 \r\nL 172.02353 14.948577 \r\nL 173.002541 16.331186 \r\nL 173.981552 18.12656 \r\nL 175.123732 20.717347 \r\nL 176.42908 24.286979 \r\nL 177.897596 29.000684 \r\nL 179.69245 35.615359 \r\nL 181.81364 44.367165 \r\nL 184.913841 58.245076 \r\nL 190.46157 83.161144 \r\nL 192.909098 93.115047 \r\nL 195.030288 100.899673 \r\nL 196.98831 107.299489 \r\nL 198.783164 112.473243 \r\nL 200.578017 116.986089 \r\nL 202.209702 120.534567 \r\nL 203.841387 123.5855 \r\nL 205.473072 126.176453 \r\nL 207.104757 128.35023 \r\nL 208.736442 130.152334 \r\nL 210.368127 131.628806 \r\nL 211.999812 132.824481 \r\nL 213.794665 133.865803 \r\nL 215.752687 134.73293 \r\nL 217.873878 135.421686 \r\nL 220.321405 135.97206 \r\nL 223.258438 136.389278 \r\nL 227.011314 136.67952 \r\nL 232.395874 136.850878 \r\nL 243.001826 136.917995 \r\nL 283.467614 136.922727 \r\nL 283.467614 136.922727 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_26\">\r\n    <path clip-path=\"url(#p7438949e74)\" d=\"M 55.194886 136.7876 \r\nL 65.800839 136.522951 \r\nL 73.143421 136.126434 \r\nL 79.017487 135.590287 \r\nL 83.912542 134.926494 \r\nL 88.318091 134.105229 \r\nL 92.234135 133.153577 \r\nL 95.823842 132.06316 \r\nL 99.250381 130.798749 \r\nL 102.51375 129.367709 \r\nL 105.77712 127.695008 \r\nL 109.04049 125.764111 \r\nL 112.30386 123.563438 \r\nL 115.56723 121.087897 \r\nL 118.993769 118.196043 \r\nL 122.583476 114.860886 \r\nL 126.49952 110.903144 \r\nL 131.068237 105.948686 \r\nL 138.247651 97.770807 \r\nL 144.611223 90.644946 \r\nL 148.527267 86.589675 \r\nL 151.790637 83.530672 \r\nL 154.564501 81.224436 \r\nL 157.175197 79.344214 \r\nL 159.459556 77.957408 \r\nL 161.743915 76.832365 \r\nL 163.865105 76.036198 \r\nL 165.823127 75.522597 \r\nL 167.781149 75.227168 \r\nL 169.739171 75.153089 \r\nL 171.697193 75.301158 \r\nL 173.655215 75.66978 \r\nL 175.613237 76.254995 \r\nL 177.734428 77.126088 \r\nL 179.855618 78.233161 \r\nL 182.139977 79.673625 \r\nL 184.587504 81.48006 \r\nL 187.361369 83.820867 \r\nL 190.46157 86.751111 \r\nL 194.051277 90.469337 \r\nL 198.783164 95.721751 \r\nL 211.673475 110.215093 \r\nL 215.752687 114.383393 \r\nL 219.505563 117.90542 \r\nL 222.932101 120.82526 \r\nL 226.195471 123.328273 \r\nL 229.458841 125.556324 \r\nL 232.722211 127.513772 \r\nL 235.985581 129.211619 \r\nL 239.248951 130.665971 \r\nL 242.675489 131.95258 \r\nL 246.265196 133.063568 \r\nL 250.18124 134.034481 \r\nL 254.423621 134.846714 \r\nL 259.155508 135.514669 \r\nL 264.540068 136.040363 \r\nL 270.903639 136.432354 \r\nL 279.225233 136.707949 \r\nL 283.467614 136.785216 \r\nL 283.467614 136.785216 \r\n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_27\">\r\n    <path clip-path=\"url(#p7438949e74)\" d=\"M 55.194886 136.922727 \r\nL 157.175197 136.813535 \r\nL 162.559757 136.566288 \r\nL 166.149464 136.184417 \r\nL 168.923329 135.668952 \r\nL 171.207688 135.025186 \r\nL 173.16571 134.257847 \r\nL 174.960563 133.330368 \r\nL 176.755417 132.138334 \r\nL 178.387102 130.779445 \r\nL 180.018787 129.113085 \r\nL 181.650472 127.093512 \r\nL 183.282156 124.67477 \r\nL 184.913841 121.812692 \r\nL 186.545526 118.467288 \r\nL 188.177211 114.605495 \r\nL 189.972065 109.733818 \r\nL 191.766918 104.197078 \r\nL 193.72494 97.412516 \r\nL 195.846131 89.247632 \r\nL 198.456827 78.224523 \r\nL 202.209702 61.239305 \r\nL 206.452083 42.275258 \r\nL 208.736442 33.113056 \r\nL 210.531295 26.820515 \r\nL 211.999812 22.424523 \r\nL 213.30516 19.173268 \r\nL 214.447339 16.884633 \r\nL 215.42635 15.362585 \r\nL 216.405361 14.263605 \r\nL 217.221204 13.679589 \r\nL 218.037046 13.401979 \r\nL 218.68972 13.401979 \r\nL 219.342394 13.599455 \r\nL 220.158237 14.122466 \r\nL 220.974079 14.948577 \r\nL 221.95309 16.331186 \r\nL 222.932101 18.12656 \r\nL 224.074281 20.717347 \r\nL 225.379629 24.286979 \r\nL 226.848145 29.000684 \r\nL 228.642999 35.615359 \r\nL 230.764189 44.367165 \r\nL 233.864391 58.245076 \r\nL 239.412119 83.161144 \r\nL 241.859647 93.115047 \r\nL 243.980837 100.899673 \r\nL 245.938859 107.299489 \r\nL 247.733713 112.473243 \r\nL 249.528566 116.986089 \r\nL 251.160251 120.534567 \r\nL 252.791936 123.5855 \r\nL 254.423621 126.176453 \r\nL 256.055306 128.35023 \r\nL 257.686991 130.152334 \r\nL 259.318676 131.628806 \r\nL 260.950361 132.824481 \r\nL 262.745215 133.865803 \r\nL 264.703236 134.73293 \r\nL 266.824427 135.421686 \r\nL 269.271954 135.97206 \r\nL 272.208987 136.389278 \r\nL 275.961863 136.67952 \r\nL 281.346423 136.850878 \r\nL 283.467614 136.879593 \r\nL 283.467614 136.879593 \r\n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"patch_3\">\r\n    <path d=\"M 43.78125 143.1 \r\nL 43.78125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_4\">\r\n    <path d=\"M 294.88125 143.1 \r\nL 294.88125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_5\">\r\n    <path d=\"M 43.78125 143.1 \r\nL 294.88125 143.1 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_6\">\r\n    <path d=\"M 43.78125 7.2 \r\nL 294.88125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"legend_1\">\r\n    <g id=\"patch_7\">\r\n     <path d=\"M 50.78125 59.234375 \r\nL 152.05625 59.234375 \r\nQ 154.05625 59.234375 154.05625 57.234375 \r\nL 154.05625 14.2 \r\nQ 154.05625 12.2 152.05625 12.2 \r\nL 50.78125 12.2 \r\nQ 48.78125 12.2 48.78125 14.2 \r\nL 48.78125 57.234375 \r\nQ 48.78125 59.234375 50.78125 59.234375 \r\nz\r\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\r\n    </g>\r\n    <g id=\"line2d_28\">\r\n     <path d=\"M 52.78125 20.298438 \r\nL 72.78125 20.298438 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_29\"/>\r\n    <g id=\"text_15\">\r\n     <!-- mean 0, std 1 -->\r\n     <g transform=\"translate(80.78125 23.798438)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 52 44.1875 \r\nQ 55.375 50.25 60.0625 53.125 \r\nQ 64.75 56 71.09375 56 \r\nQ 79.640625 56 84.28125 50.015625 \r\nQ 88.921875 44.046875 88.921875 33.015625 \r\nL 88.921875 0 \r\nL 79.890625 0 \r\nL 79.890625 32.71875 \r\nQ 79.890625 40.578125 77.09375 44.375 \r\nQ 74.3125 48.1875 68.609375 48.1875 \r\nQ 61.625 48.1875 57.5625 43.546875 \r\nQ 53.515625 38.921875 53.515625 30.90625 \r\nL 53.515625 0 \r\nL 44.484375 0 \r\nL 44.484375 32.71875 \r\nQ 44.484375 40.625 41.703125 44.40625 \r\nQ 38.921875 48.1875 33.109375 48.1875 \r\nQ 26.21875 48.1875 22.15625 43.53125 \r\nQ 18.109375 38.875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.1875 51.21875 25.484375 53.609375 \r\nQ 29.78125 56 35.6875 56 \r\nQ 41.65625 56 45.828125 52.96875 \r\nQ 50 49.953125 52 44.1875 \r\nz\r\n\" id=\"DejaVuSans-109\"/>\r\n       <path d=\"M 56.203125 29.59375 \r\nL 56.203125 25.203125 \r\nL 14.890625 25.203125 \r\nQ 15.484375 15.921875 20.484375 11.0625 \r\nQ 25.484375 6.203125 34.421875 6.203125 \r\nQ 39.59375 6.203125 44.453125 7.46875 \r\nQ 49.3125 8.734375 54.109375 11.28125 \r\nL 54.109375 2.78125 \r\nQ 49.265625 0.734375 44.1875 -0.34375 \r\nQ 39.109375 -1.421875 33.890625 -1.421875 \r\nQ 20.796875 -1.421875 13.15625 6.1875 \r\nQ 5.515625 13.8125 5.515625 26.8125 \r\nQ 5.515625 40.234375 12.765625 48.109375 \r\nQ 20.015625 56 32.328125 56 \r\nQ 43.359375 56 49.78125 48.890625 \r\nQ 56.203125 41.796875 56.203125 29.59375 \r\nz\r\nM 47.21875 32.234375 \r\nQ 47.125 39.59375 43.09375 43.984375 \r\nQ 39.0625 48.390625 32.421875 48.390625 \r\nQ 24.90625 48.390625 20.390625 44.140625 \r\nQ 15.875 39.890625 15.1875 32.171875 \r\nz\r\n\" id=\"DejaVuSans-101\"/>\r\n       <path d=\"M 34.28125 27.484375 \r\nQ 23.390625 27.484375 19.1875 25 \r\nQ 14.984375 22.515625 14.984375 16.5 \r\nQ 14.984375 11.71875 18.140625 8.90625 \r\nQ 21.296875 6.109375 26.703125 6.109375 \r\nQ 34.1875 6.109375 38.703125 11.40625 \r\nQ 43.21875 16.703125 43.21875 25.484375 \r\nL 43.21875 27.484375 \r\nz\r\nM 52.203125 31.203125 \r\nL 52.203125 0 \r\nL 43.21875 0 \r\nL 43.21875 8.296875 \r\nQ 40.140625 3.328125 35.546875 0.953125 \r\nQ 30.953125 -1.421875 24.3125 -1.421875 \r\nQ 15.921875 -1.421875 10.953125 3.296875 \r\nQ 6 8.015625 6 15.921875 \r\nQ 6 25.140625 12.171875 29.828125 \r\nQ 18.359375 34.515625 30.609375 34.515625 \r\nL 43.21875 34.515625 \r\nL 43.21875 35.40625 \r\nQ 43.21875 41.609375 39.140625 45 \r\nQ 35.0625 48.390625 27.6875 48.390625 \r\nQ 23 48.390625 18.546875 47.265625 \r\nQ 14.109375 46.140625 10.015625 43.890625 \r\nL 10.015625 52.203125 \r\nQ 14.9375 54.109375 19.578125 55.046875 \r\nQ 24.21875 56 28.609375 56 \r\nQ 40.484375 56 46.34375 49.84375 \r\nQ 52.203125 43.703125 52.203125 31.203125 \r\nz\r\n\" id=\"DejaVuSans-97\"/>\r\n       <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-110\"/>\r\n       <path id=\"DejaVuSans-32\"/>\r\n       <path d=\"M 11.71875 12.40625 \r\nL 22.015625 12.40625 \r\nL 22.015625 4 \r\nL 14.015625 -11.625 \r\nL 7.71875 -11.625 \r\nL 11.71875 4 \r\nz\r\n\" id=\"DejaVuSans-44\"/>\r\n       <path d=\"M 44.28125 53.078125 \r\nL 44.28125 44.578125 \r\nQ 40.484375 46.53125 36.375 47.5 \r\nQ 32.28125 48.484375 27.875 48.484375 \r\nQ 21.1875 48.484375 17.84375 46.4375 \r\nQ 14.5 44.390625 14.5 40.28125 \r\nQ 14.5 37.15625 16.890625 35.375 \r\nQ 19.28125 33.59375 26.515625 31.984375 \r\nL 29.59375 31.296875 \r\nQ 39.15625 29.25 43.1875 25.515625 \r\nQ 47.21875 21.78125 47.21875 15.09375 \r\nQ 47.21875 7.46875 41.1875 3.015625 \r\nQ 35.15625 -1.421875 24.609375 -1.421875 \r\nQ 20.21875 -1.421875 15.453125 -0.5625 \r\nQ 10.6875 0.296875 5.421875 2 \r\nL 5.421875 11.28125 \r\nQ 10.40625 8.6875 15.234375 7.390625 \r\nQ 20.0625 6.109375 24.8125 6.109375 \r\nQ 31.15625 6.109375 34.5625 8.28125 \r\nQ 37.984375 10.453125 37.984375 14.40625 \r\nQ 37.984375 18.0625 35.515625 20.015625 \r\nQ 33.0625 21.96875 24.703125 23.78125 \r\nL 21.578125 24.515625 \r\nQ 13.234375 26.265625 9.515625 29.90625 \r\nQ 5.8125 33.546875 5.8125 39.890625 \r\nQ 5.8125 47.609375 11.28125 51.796875 \r\nQ 16.75 56 26.8125 56 \r\nQ 31.78125 56 36.171875 55.265625 \r\nQ 40.578125 54.546875 44.28125 53.078125 \r\nz\r\n\" id=\"DejaVuSans-115\"/>\r\n       <path d=\"M 18.3125 70.21875 \r\nL 18.3125 54.6875 \r\nL 36.8125 54.6875 \r\nL 36.8125 47.703125 \r\nL 18.3125 47.703125 \r\nL 18.3125 18.015625 \r\nQ 18.3125 11.328125 20.140625 9.421875 \r\nQ 21.96875 7.515625 27.59375 7.515625 \r\nL 36.8125 7.515625 \r\nL 36.8125 0 \r\nL 27.59375 0 \r\nQ 17.1875 0 13.234375 3.875 \r\nQ 9.28125 7.765625 9.28125 18.015625 \r\nL 9.28125 47.703125 \r\nL 2.6875 47.703125 \r\nL 2.6875 54.6875 \r\nL 9.28125 54.6875 \r\nL 9.28125 70.21875 \r\nz\r\n\" id=\"DejaVuSans-116\"/>\r\n       <path d=\"M 45.40625 46.390625 \r\nL 45.40625 75.984375 \r\nL 54.390625 75.984375 \r\nL 54.390625 0 \r\nL 45.40625 0 \r\nL 45.40625 8.203125 \r\nQ 42.578125 3.328125 38.25 0.953125 \r\nQ 33.9375 -1.421875 27.875 -1.421875 \r\nQ 17.96875 -1.421875 11.734375 6.484375 \r\nQ 5.515625 14.40625 5.515625 27.296875 \r\nQ 5.515625 40.1875 11.734375 48.09375 \r\nQ 17.96875 56 27.875 56 \r\nQ 33.9375 56 38.25 53.625 \r\nQ 42.578125 51.265625 45.40625 46.390625 \r\nz\r\nM 14.796875 27.296875 \r\nQ 14.796875 17.390625 18.875 11.75 \r\nQ 22.953125 6.109375 30.078125 6.109375 \r\nQ 37.203125 6.109375 41.296875 11.75 \r\nQ 45.40625 17.390625 45.40625 27.296875 \r\nQ 45.40625 37.203125 41.296875 42.84375 \r\nQ 37.203125 48.484375 30.078125 48.484375 \r\nQ 22.953125 48.484375 18.875 42.84375 \r\nQ 14.796875 37.203125 14.796875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-100\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"97.412109\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"158.935547\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"220.214844\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"283.59375\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"315.380859\" xlink:href=\"#DejaVuSans-48\"/>\r\n      <use x=\"379.003906\" xlink:href=\"#DejaVuSans-44\"/>\r\n      <use x=\"410.791016\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"442.578125\" xlink:href=\"#DejaVuSans-115\"/>\r\n      <use x=\"494.677734\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"533.886719\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"597.363281\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"629.150391\" xlink:href=\"#DejaVuSans-49\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_30\">\r\n     <path d=\"M 52.78125 34.976562 \r\nL 72.78125 34.976562 \r\n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_31\"/>\r\n    <g id=\"text_16\">\r\n     <!-- mean 0, std 2 -->\r\n     <g transform=\"translate(80.78125 38.476562)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"97.412109\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"158.935547\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"220.214844\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"283.59375\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"315.380859\" xlink:href=\"#DejaVuSans-48\"/>\r\n      <use x=\"379.003906\" xlink:href=\"#DejaVuSans-44\"/>\r\n      <use x=\"410.791016\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"442.578125\" xlink:href=\"#DejaVuSans-115\"/>\r\n      <use x=\"494.677734\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"533.886719\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"597.363281\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"629.150391\" xlink:href=\"#DejaVuSans-50\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_32\">\r\n     <path d=\"M 52.78125 49.654688 \r\nL 72.78125 49.654688 \r\n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_33\"/>\r\n    <g id=\"text_17\">\r\n     <!-- mean 3, std 1 -->\r\n     <g transform=\"translate(80.78125 53.154688)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"97.412109\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"158.935547\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"220.214844\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"283.59375\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"315.380859\" xlink:href=\"#DejaVuSans-51\"/>\r\n      <use x=\"379.003906\" xlink:href=\"#DejaVuSans-44\"/>\r\n      <use x=\"410.791016\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"442.578125\" xlink:href=\"#DejaVuSans-115\"/>\r\n      <use x=\"494.677734\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"533.886719\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"597.363281\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"629.150391\" xlink:href=\"#DejaVuSans-49\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n  </g>\r\n </g>\r\n <defs>\r\n  <clipPath id=\"p7438949e74\">\r\n   <rect height=\"135.9\" width=\"251.1\" x=\"43.78125\" y=\"7.2\"/>\r\n  </clipPath>\r\n </defs>\r\n</svg>\r\n",
   "text/plain": "<Figure size 324x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

就像我们所看到的，改变均值会产生沿$x$轴的偏移，增加方差将会分散分布、降低其峰值。

均方误差损失函数（简称均方损失）可以用于线性回归的一个原因是：
我们假设了观测中包含噪声，其中噪声服从正态分布。
噪声正态分布如下式:

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon,$$

其中，$\epsilon \sim \mathcal{N}(0, \sigma^2)$。

因此，我们现在可以写出通过给定的$\mathbf{x}$观测到特定$y$的*似然*（likelihood）：

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

现在，根据极大似然估计法，参数$\mathbf{w}$和$b$的最优值是使整个数据集的*似然*最大的值：

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

根据极大似然估计法选择的估计量称为*极大似然估计量*。
虽然使许多指数函数的乘积最大化看起来很困难，
但是我们可以在不改变目标的前提下，通过最大化似然对数来简化。
由于历史原因，优化通常是说最小化而不是最大化。
我们可以改为*最小化负对数似然*$-\log P(\mathbf y \mid \mathbf X)$。
由此可以得到的数学公式是：

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

现在我们只需要假设$\sigma$是某个固定常数就可以忽略第一项，
因为第一项不依赖于$\mathbf{w}$和$b$。
现在第二项除了常数$\frac{1}{\sigma^2}$外，其余部分和前面介绍的均方误差是一样的。
幸运的是，上面式子的解并不依赖于$\sigma$。
因此，在高斯噪声的假设下，最小化均方误差等价于对线性模型的极大似然估计。

## 从线性回归到深度网络

到目前为止，我们只谈论了线性模型。
尽管神经网络涵盖了更多更为丰富的模型，我们依然可以用描述神经网络的方式来描述线性模型，
从而把线性模型看作一个神经网络。
首先，我们用“层”符号来重写这个模型。

### 神经网络图

深度学习从业者喜欢绘制图表来可视化模型中正在发生的事情。
在 :numref:`fig_single_neuron`中，我们将线性回归模型描述为一个神经网络。
需要注意的是，该图只显示连接模式，即只显示每个输入如何连接到输出，隐去了权重和偏置的值。

![线性回归是一个单层神经网络。](../img/singleneuron.svg)
:label:`fig_single_neuron`

在 :numref:`fig_single_neuron`所示的神经网络中，输入为$x_1, \ldots, x_d$，
因此输入层中的*输入数*（或称为*特征维度*，feature dimensionality）为$d$。
网络的输出为$o_1$，因此输出层中的*输出数*是1。
需要注意的是，输入值都是已经给定的，并且只有一个*计算*神经元。
由于模型重点在发生计算的地方，所以通常我们在计算层数时不考虑输入层。
也就是说， :numref:`fig_single_neuron`中神经网络的*层数*为1。
我们可以将线性回归模型视为仅由单个人工神经元组成的神经网络，或称为单层神经网络。

对于线性回归，每个输入都与每个输出（在本例中只有一个输出）相连，
我们将这种变换（ :numref:`fig_single_neuron`中的输出层）
称为*全连接层*（fully-connected layer）或称为*稠密层*（dense layer）。
下一章将详细讨论由这些层组成的网络。

### 生物学

线性回归发明的时间（1795年）早于计算神经科学，所以将线性回归描述为神经网络似乎不合适。
当控制学家、神经生物学家沃伦·麦库洛奇和沃尔特·皮茨开始开发人工神经元模型时，
他们为什么将线性模型作为一个起点呢？
我们来看一张图片 :numref:`fig_Neuron`：
这是一张由*树突*（dendrites，输入终端）、
*细胞核*（nucleu，CPU）组成的生物神经元图片。
*轴突*（axon，输出线）和*轴突端子*（axon terminal，输出端子）
通过*突触*（synapse）与其他神经元连接。

![真实的神经元。](../img/neuron.svg)
:label:`fig_Neuron`

树突中接收到来自其他神经元（或视网膜等环境传感器）的信息$x_i$。
该信息通过*突触权重*$w_i$来加权，以确定输入的影响（即，通过$x_i w_i$相乘来激活或抑制）。
来自多个源的加权输入以加权和$y = \sum_i x_i w_i + b$的形式汇聚在细胞核中，
然后将这些信息发送到轴突$y$中进一步处理，通常会通过$\sigma(y)$进行一些非线性处理。
之后，它要么到达目的地（例如肌肉），要么通过树突进入另一个神经元。

当然，许多这样的单元可以通过正确连接和正确的学习算法拼凑在一起，
从而产生的行为会比单独一个神经元所产生的行为更有趣、更复杂，
这种想法归功于我们对真实生物神经系统的研究。

当今大多数深度学习的研究几乎没有直接从神经科学中获得灵感。
我们援引斯图尔特·罗素和彼得·诺维格谁，在他们的经典人工智能教科书
*Artificial Intelligence:A Modern Approach* :cite:`Russell.Norvig.2016`
中所说：虽然飞机可能受到鸟类的启发，但几个世纪以来，鸟类学并不是航空创新的主要驱动力。
同样地，如今在深度学习中的灵感同样或更多地来自数学、统计学和计算机科学。

## 小结

* 机器学习模型中的关键要素是训练数据、损失函数、优化算法，还有模型本身。
* 矢量化使数学表达上更简洁，同时运行的更快。
* 最小化目标函数和执行极大似然估计等价。
* 线性回归模型也是一个简单的神经网络。

## 练习

1. 假设我们有一些数据$x_1, \ldots, x_n \in \mathbb{R}$。我们的目标是找到一个常数$b$，使得最小化$\sum_i (x_i - b)^2$。
    1. 找到最优值$b$的解析解。
    1. 这个问题及其解与正态分布有什么关系?
1. 推导出使用平方误差的线性回归优化问题的解析解。为了简化问题，可以忽略偏置$b$（我们可以通过向$\mathbf X$添加所有值为1的一列来做到这一点）。
    1. 用矩阵和向量表示法写出优化问题（将所有数据视为单个矩阵，将所有目标值视为单个向量）。
    1. 计算损失对$w$的梯度。
    1. 通过将梯度设为0、求解矩阵方程来找到解析解。
    1. 什么时候可能比使用随机梯度下降更好？这种方法何时会失效？
1. 假定控制附加噪声$\epsilon$的噪声模型是指数分布。也就是说，$p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$
    1. 写出模型$-\log P(\mathbf y \mid \mathbf X)$下数据的负对数似然。
    1. 你能写出解析解吗？
    1. 提出一种随机梯度下降算法来解决这个问题。哪里可能出错？（提示：当我们不断更新参数时，在驻点附近会发生什么情况）你能解决这个问题吗？

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1774)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1775)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1776)
:end_tab:

None
None
None
None
# 微积分
:label:`sec_calculus`

在2500年前，古希腊人把一个多边形分成三角形，并把它们的面积相加，才找到计算多边形面积的方法。
为了求出曲线形状（比如圆）的面积，古希腊人在这样的形状上刻内接多边形。
如 :numref:`fig_circle_area`所示，内接多边形的等长边越多，就越接近圆。
这个过程也被称为*逼近法*（method of exhaustion）。

![用逼近法求圆的面积](../img/polygon-circle.svg)
:label:`fig_circle_area`

事实上，逼近法就是*积分*（integral calculus）的起源，
我们将在 :numref:`sec_integral_calculus`中详细描述。
2000多年后，微积分的另一支，*微分*（differential calculus）被发明出来。
在微分学最重要的应用是优化问题，即考虑如何把事情做到最好。
正如在 :numref:`subsec_norms_and_objectives`中讨论的那样，
这种问题在深度学习中是无处不在的。

在深度学习中，我们“训练”模型，不断更新它们，使它们在看到越来越多的数据时变得越来越好。
通常情况下，变得更好意味着最小化一个*损失函数*（loss function），
即一个衡量“我们的模型有多糟糕”这个问题的分数。
最终，我们真正关心的是生成一个模型，它能够在从未见过的数据上表现良好。
但“训练”模型只能将模型与我们实际能看到的数据相拟合。
因此，我们可以将拟合模型的任务分解为两个关键问题：

* *优化*（optimization）：用模型拟合观测数据的过程；
* *泛化*（generalization）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

为了帮助你在后面的章节中更好地理解优化问题和方法，
本节提供了一个非常简短的入门教程，帮你快速掌握深度学习中常用的微分知识。

## 导数和微分

我们首先讨论导数的计算，这是几乎所有深度学习优化算法的关键步骤。
在深度学习中，我们通常选择对于模型参数可微的损失函数。
简而言之，对于每个参数，
如果我们把这个参数*增加*或*减少*一个无穷小的量，我们可以知道损失会以多快的速度增加或减少，

假设我们有一个函数$f: \mathbb{R}^n \rightarrow \mathbb{R}$，其输入和输出都是标量。
(**如果$f$的*导数*存在，这个极限被定义为**)

(**$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$**)
:eqlabel:`eq_derivative`

如果$f'(a)$存在，则称$f$在$a$处是*可微*（differentiable）的。
如果$f$在一个区间内的每个数上都是可微的，则此函数在此区间中是可微的。
我们可以将 :eqref:`eq_derivative`中的导数$f'(x)$解释为$f(x)$相对于$x$的*瞬时*（instantaneous）变化率。
所谓的瞬时变化率是基于$x$中的变化$h$，且$h$接近$0$。

为了更好地解释导数，让我们做一个实验。
(**定义$u=f(x)=3x^2-4x$**)如下：

```{.python .input  n=2}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.json .output n=2}
[
 {
  "ename": "ModuleNotFoundError",
  "evalue": "No module named 'd2l'",
  "output_type": "error",
  "traceback": [
   "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
   "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_38124/2120375071.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[0mget_ipython\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mrun_line_magic\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;34m'matplotlib'\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m'inline'\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 2\u001b[1;33m \u001b[1;32mfrom\u001b[0m \u001b[0md2l\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mmxnet\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0md2l\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      3\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mIPython\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mdisplay\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[0mmxnet\u001b[0m \u001b[1;32mimport\u001b[0m \u001b[0mnp\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mnpx\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[0mnpx\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mset_np\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
   "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'd2l'"
  ]
 }
]
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input  n=1}
#@tab paddle
%matplotlib inline
from d2l import paddle as d2l
from IPython import display
import numpy as np

def f(x):
    return 3 * x ** 2 - 4 * x
```

[**通过令$x=1$并让$h$接近$0$，**] :eqref:`eq_derivative`中(**$\frac{f(x+h)-f(x)}{h}$的数值结果接近$2$**)。
虽然这个实验不是一个数学证明，但我们稍后会看到，当$x=1$时，导数$u'$是$2$。

```{.python .input  n=4}
#@tab all
def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "h=0.10000, numerical limit=2.30000\nh=0.01000, numerical limit=2.03000\nh=0.00100, numerical limit=2.00300\nh=0.00010, numerical limit=2.00030\nh=0.00001, numerical limit=2.00003\n"
 }
]
```

让我们熟悉一下导数的几个等价符号。
给定$y=f(x)$，其中$x$和$y$分别是函数$f$的自变量和因变量。以下表达式是等价的：

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

其中符号$\frac{d}{dx}$和$D$是*微分运算符*，表示*微分*操作。
我们可以使用以下规则来对常见函数求微分：

* $DC = 0$（$C$是一个常数）
* $Dx^n = nx^{n-1}$（*幂律*（power rule），$n$是任意实数）
* $De^x = e^x$
* $D\ln(x) = 1/x$

为了微分一个由一些常见函数组成的函数，下面的一些法则方便使用。
假设函数$f$和$g$都是可微的，$C$是一个常数，则：

*常数相乘法则*
$$\frac{d}{dx} [Cf(x)] = C \frac{d}{dx} f(x),$$

*加法法则*

$$\frac{d}{dx} [f(x) + g(x)] = \frac{d}{dx} f(x) + \frac{d}{dx} g(x),$$

*乘法法则*

$$\frac{d}{dx} [f(x)g(x)] = f(x) \frac{d}{dx} [g(x)] + g(x) \frac{d}{dx} [f(x)],$$

*除法法则*

$$\frac{d}{dx} \left[\frac{f(x)}{g(x)}\right] = \frac{g(x) \frac{d}{dx} [f(x)] - f(x) \frac{d}{dx} [g(x)]}{[g(x)]^2}.$$

现在我们可以应用上述几个法则来计算$u'=f'(x)=3\frac{d}{dx}x^2-4\frac{d}{dx}x=6x-4$。
令$x=1$，我们有$u'=2$：在这个实验中，数值结果接近$2$，
这一点得到了我们在本节前面的实验的支持。
当$x=1$时，此导数也是曲线$u=f(x)$切线的斜率。

[**为了对导数的这种解释进行可视化，我们将使用`matplotlib`**]，
这是一个Python中流行的绘图库。
要配置`matplotlib`生成图形的属性，我们需要(**定义几个函数**)。
在下面，`use_svg_display`函数指定`matplotlib`软件包输出svg图表以获得更清晰的图像。

注意，注释`#@save`是一个特殊的标记，会将对应的函数、类或语句保存在`d2l`包中。
因此，以后无须重新定义就可以直接调用它们（例如，`d2l.use_svg_display()`）。

```{.python .input  n=5}
#@tab all
def use_svg_display():  #@save
    """使用svg格式在Jupyter中显示绘图"""
    display.set_matplotlib_formats('svg')
```

我们定义`set_figsize`函数来设置图表大小。
注意，这里我们直接使用`d2l.plt`，因为导入语句
`from matplotlib import pyplot as plt`已标记为保存到`d2l`包中。

```{.python .input  n=6}
#@tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """设置matplotlib的图表大小"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

下面的`set_axes`函数用于设置由`matplotlib`生成图表的轴的属性。

```{.python .input  n=7}
#@tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

通过这三个用于图形配置的函数，我们定义了`plot`函数来简洁地绘制多条曲线，
因为我们需要在整个书中可视化许多曲线。

```{.python .input  n=8}
#@tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

现在我们可以[**绘制函数$u=f(x)$及其在$x=1$处的切线$y=2x-3$**]，
其中系数$2$是切线的斜率。

```{.python .input  n=9}
#@tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

```{.json .output n=9}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "C:\\Users\\Wan\\AppData\\Local\\Temp/ipykernel_31672/1830532362.py:4: DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23, directly use `matplotlib_inline.backend_inline.set_matplotlib_formats()`\n  display.set_matplotlib_formats('svg')\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Created with matplotlib (https://matplotlib.org/) -->\r\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 243.529359 180.65625\" width=\"243.529359pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n <metadata>\r\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\r\n   <cc:Work>\r\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\r\n    <dc:date>2022-01-20T14:31:37.070421</dc:date>\r\n    <dc:format>image/svg+xml</dc:format>\r\n    <dc:creator>\r\n     <cc:Agent>\r\n      <dc:title>Matplotlib v3.3.3, https://matplotlib.org/</dc:title>\r\n     </cc:Agent>\r\n    </dc:creator>\r\n   </cc:Work>\r\n  </rdf:RDF>\r\n </metadata>\r\n <defs>\r\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\r\n </defs>\r\n <g id=\"figure_1\">\r\n  <g id=\"patch_1\">\r\n   <path d=\"M 0 180.65625 \r\nL 243.529359 180.65625 \r\nL 243.529359 0 \r\nL 0 0 \r\nz\r\n\" style=\"fill:none;\"/>\r\n  </g>\r\n  <g id=\"axes_1\">\r\n   <g id=\"patch_2\">\r\n    <path d=\"M 40.603125 143.1 \r\nL 235.903125 143.1 \r\nL 235.903125 7.2 \r\nL 40.603125 7.2 \r\nz\r\n\" style=\"fill:#ffffff;\"/>\r\n   </g>\r\n   <g id=\"matplotlib.axis_1\">\r\n    <g id=\"xtick_1\">\r\n     <g id=\"line2d_1\">\r\n      <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 49.480398 143.1 \r\nL 49.480398 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_2\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL 0 3.5 \r\n\" id=\"m738971c2fc\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"49.480398\" xlink:href=\"#m738971c2fc\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_1\">\r\n      <!-- 0 -->\r\n      <g transform=\"translate(46.299148 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 31.78125 66.40625 \r\nQ 24.171875 66.40625 20.328125 58.90625 \r\nQ 16.5 51.421875 16.5 36.375 \r\nQ 16.5 21.390625 20.328125 13.890625 \r\nQ 24.171875 6.390625 31.78125 6.390625 \r\nQ 39.453125 6.390625 43.28125 13.890625 \r\nQ 47.125 21.390625 47.125 36.375 \r\nQ 47.125 51.421875 43.28125 58.90625 \r\nQ 39.453125 66.40625 31.78125 66.40625 \r\nz\r\nM 31.78125 74.21875 \r\nQ 44.046875 74.21875 50.515625 64.515625 \r\nQ 56.984375 54.828125 56.984375 36.375 \r\nQ 56.984375 17.96875 50.515625 8.265625 \r\nQ 44.046875 -1.421875 31.78125 -1.421875 \r\nQ 19.53125 -1.421875 13.0625 8.265625 \r\nQ 6.59375 17.96875 6.59375 36.375 \r\nQ 6.59375 54.828125 13.0625 64.515625 \r\nQ 19.53125 74.21875 31.78125 74.21875 \r\nz\r\n\" id=\"DejaVuSans-48\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_2\">\r\n     <g id=\"line2d_3\">\r\n      <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 110.702968 143.1 \r\nL 110.702968 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_4\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"110.702968\" xlink:href=\"#m738971c2fc\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_2\">\r\n      <!-- 1 -->\r\n      <g transform=\"translate(107.521718 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 12.40625 8.296875 \r\nL 28.515625 8.296875 \r\nL 28.515625 63.921875 \r\nL 10.984375 60.40625 \r\nL 10.984375 69.390625 \r\nL 28.421875 72.90625 \r\nL 38.28125 72.90625 \r\nL 38.28125 8.296875 \r\nL 54.390625 8.296875 \r\nL 54.390625 0 \r\nL 12.40625 0 \r\nz\r\n\" id=\"DejaVuSans-49\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_3\">\r\n     <g id=\"line2d_5\">\r\n      <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 171.925539 143.1 \r\nL 171.925539 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_6\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"171.925539\" xlink:href=\"#m738971c2fc\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_3\">\r\n      <!-- 2 -->\r\n      <g transform=\"translate(168.744289 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 19.1875 8.296875 \r\nL 53.609375 8.296875 \r\nL 53.609375 0 \r\nL 7.328125 0 \r\nL 7.328125 8.296875 \r\nQ 12.9375 14.109375 22.625 23.890625 \r\nQ 32.328125 33.6875 34.8125 36.53125 \r\nQ 39.546875 41.84375 41.421875 45.53125 \r\nQ 43.3125 49.21875 43.3125 52.78125 \r\nQ 43.3125 58.59375 39.234375 62.25 \r\nQ 35.15625 65.921875 28.609375 65.921875 \r\nQ 23.96875 65.921875 18.8125 64.3125 \r\nQ 13.671875 62.703125 7.8125 59.421875 \r\nL 7.8125 69.390625 \r\nQ 13.765625 71.78125 18.9375 73 \r\nQ 24.125 74.21875 28.421875 74.21875 \r\nQ 39.75 74.21875 46.484375 68.546875 \r\nQ 53.21875 62.890625 53.21875 53.421875 \r\nQ 53.21875 48.921875 51.53125 44.890625 \r\nQ 49.859375 40.875 45.40625 35.40625 \r\nQ 44.1875 33.984375 37.640625 27.21875 \r\nQ 31.109375 20.453125 19.1875 8.296875 \r\nz\r\n\" id=\"DejaVuSans-50\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_4\">\r\n     <g id=\"line2d_7\">\r\n      <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 233.148109 143.1 \r\nL 233.148109 7.2 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_8\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"233.148109\" xlink:href=\"#m738971c2fc\" y=\"143.1\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_4\">\r\n      <!-- 3 -->\r\n      <g transform=\"translate(229.966859 157.698438)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 40.578125 39.3125 \r\nQ 47.65625 37.796875 51.625 33 \r\nQ 55.609375 28.21875 55.609375 21.1875 \r\nQ 55.609375 10.40625 48.1875 4.484375 \r\nQ 40.765625 -1.421875 27.09375 -1.421875 \r\nQ 22.515625 -1.421875 17.65625 -0.515625 \r\nQ 12.796875 0.390625 7.625 2.203125 \r\nL 7.625 11.71875 \r\nQ 11.71875 9.328125 16.59375 8.109375 \r\nQ 21.484375 6.890625 26.8125 6.890625 \r\nQ 36.078125 6.890625 40.9375 10.546875 \r\nQ 45.796875 14.203125 45.796875 21.1875 \r\nQ 45.796875 27.640625 41.28125 31.265625 \r\nQ 36.765625 34.90625 28.71875 34.90625 \r\nL 20.21875 34.90625 \r\nL 20.21875 43.015625 \r\nL 29.109375 43.015625 \r\nQ 36.375 43.015625 40.234375 45.921875 \r\nQ 44.09375 48.828125 44.09375 54.296875 \r\nQ 44.09375 59.90625 40.109375 62.90625 \r\nQ 36.140625 65.921875 28.71875 65.921875 \r\nQ 24.65625 65.921875 20.015625 65.03125 \r\nQ 15.375 64.15625 9.8125 62.3125 \r\nL 9.8125 71.09375 \r\nQ 15.4375 72.65625 20.34375 73.4375 \r\nQ 25.25 74.21875 29.59375 74.21875 \r\nQ 40.828125 74.21875 47.359375 69.109375 \r\nQ 53.90625 64.015625 53.90625 55.328125 \r\nQ 53.90625 49.265625 50.4375 45.09375 \r\nQ 46.96875 40.921875 40.578125 39.3125 \r\nz\r\n\" id=\"DejaVuSans-51\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-51\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_5\">\r\n     <!-- x -->\r\n     <g transform=\"translate(135.29375 171.376563)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 54.890625 54.6875 \r\nL 35.109375 28.078125 \r\nL 55.90625 0 \r\nL 45.3125 0 \r\nL 29.390625 21.484375 \r\nL 13.484375 0 \r\nL 2.875 0 \r\nL 24.125 28.609375 \r\nL 4.6875 54.6875 \r\nL 15.28125 54.6875 \r\nL 29.78125 35.203125 \r\nL 44.28125 54.6875 \r\nz\r\n\" id=\"DejaVuSans-120\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-120\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"matplotlib.axis_2\">\r\n    <g id=\"ytick_1\">\r\n     <g id=\"line2d_9\">\r\n      <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 40.603125 114.635514 \r\nL 235.903125 114.635514 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_10\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL -3.5 0 \r\n\" id=\"md3e307c096\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"40.603125\" xlink:href=\"#md3e307c096\" y=\"114.635514\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_6\">\r\n      <!-- 0 -->\r\n      <g transform=\"translate(27.240625 118.434732)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_2\">\r\n     <g id=\"line2d_11\">\r\n      <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 40.603125 77.490157 \r\nL 235.903125 77.490157 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_12\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"40.603125\" xlink:href=\"#md3e307c096\" y=\"77.490157\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_7\">\r\n      <!-- 5 -->\r\n      <g transform=\"translate(27.240625 81.289376)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.796875 72.90625 \r\nL 49.515625 72.90625 \r\nL 49.515625 64.59375 \r\nL 19.828125 64.59375 \r\nL 19.828125 46.734375 \r\nQ 21.96875 47.46875 24.109375 47.828125 \r\nQ 26.265625 48.1875 28.421875 48.1875 \r\nQ 40.625 48.1875 47.75 41.5 \r\nQ 54.890625 34.8125 54.890625 23.390625 \r\nQ 54.890625 11.625 47.5625 5.09375 \r\nQ 40.234375 -1.421875 26.90625 -1.421875 \r\nQ 22.3125 -1.421875 17.546875 -0.640625 \r\nQ 12.796875 0.140625 7.71875 1.703125 \r\nL 7.71875 11.625 \r\nQ 12.109375 9.234375 16.796875 8.0625 \r\nQ 21.484375 6.890625 26.703125 6.890625 \r\nQ 35.15625 6.890625 40.078125 11.328125 \r\nQ 45.015625 15.765625 45.015625 23.390625 \r\nQ 45.015625 31 40.078125 35.4375 \r\nQ 35.15625 39.890625 26.703125 39.890625 \r\nQ 22.75 39.890625 18.8125 39.015625 \r\nQ 14.890625 38.140625 10.796875 36.28125 \r\nz\r\n\" id=\"DejaVuSans-53\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_3\">\r\n     <g id=\"line2d_13\">\r\n      <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 40.603125 40.344801 \r\nL 235.903125 40.344801 \r\n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\r\n     </g>\r\n     <g id=\"line2d_14\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"40.603125\" xlink:href=\"#md3e307c096\" y=\"40.344801\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_8\">\r\n      <!-- 10 -->\r\n      <g transform=\"translate(20.878125 44.14402)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_9\">\r\n     <!-- f(x) -->\r\n     <g transform=\"translate(14.798437 83.771094)rotate(-90)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 37.109375 75.984375 \r\nL 37.109375 68.5 \r\nL 28.515625 68.5 \r\nQ 23.6875 68.5 21.796875 66.546875 \r\nQ 19.921875 64.59375 19.921875 59.515625 \r\nL 19.921875 54.6875 \r\nL 34.71875 54.6875 \r\nL 34.71875 47.703125 \r\nL 19.921875 47.703125 \r\nL 19.921875 0 \r\nL 10.890625 0 \r\nL 10.890625 47.703125 \r\nL 2.296875 47.703125 \r\nL 2.296875 54.6875 \r\nL 10.890625 54.6875 \r\nL 10.890625 58.5 \r\nQ 10.890625 67.625 15.140625 71.796875 \r\nQ 19.390625 75.984375 28.609375 75.984375 \r\nz\r\n\" id=\"DejaVuSans-102\"/>\r\n       <path d=\"M 31 75.875 \r\nQ 24.46875 64.65625 21.28125 53.65625 \r\nQ 18.109375 42.671875 18.109375 31.390625 \r\nQ 18.109375 20.125 21.3125 9.0625 \r\nQ 24.515625 -2 31 -13.1875 \r\nL 23.1875 -13.1875 \r\nQ 15.875 -1.703125 12.234375 9.375 \r\nQ 8.59375 20.453125 8.59375 31.390625 \r\nQ 8.59375 42.28125 12.203125 53.3125 \r\nQ 15.828125 64.359375 23.1875 75.875 \r\nz\r\n\" id=\"DejaVuSans-40\"/>\r\n       <path d=\"M 8.015625 75.875 \r\nL 15.828125 75.875 \r\nQ 23.140625 64.359375 26.78125 53.3125 \r\nQ 30.421875 42.28125 30.421875 31.390625 \r\nQ 30.421875 20.453125 26.78125 9.375 \r\nQ 23.140625 -1.703125 15.828125 -13.1875 \r\nL 8.015625 -13.1875 \r\nQ 14.5 -2 17.703125 9.0625 \r\nQ 20.90625 20.125 20.90625 31.390625 \r\nQ 20.90625 42.671875 17.703125 53.65625 \r\nQ 14.5 64.65625 8.015625 75.875 \r\nz\r\n\" id=\"DejaVuSans-41\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-102\"/>\r\n      <use x=\"35.205078\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"74.21875\" xlink:href=\"#DejaVuSans-120\"/>\r\n      <use x=\"133.398438\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"line2d_15\">\r\n    <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 49.480398 114.635514 \r\nL 55.602655 117.38427 \r\nL 61.724912 119.687282 \r\nL 67.847169 121.54455 \r\nL 73.969426 122.956073 \r\nL 80.091683 123.921853 \r\nL 86.21394 124.441888 \r\nL 92.336197 124.516178 \r\nL 98.458454 124.144725 \r\nL 104.580711 123.327527 \r\nL 110.702968 122.064585 \r\nL 116.825225 120.355898 \r\nL 122.947482 118.201468 \r\nL 129.069739 115.601293 \r\nL 135.191996 112.555374 \r\nL 141.314254 109.06371 \r\nL 147.436511 105.126302 \r\nL 153.558768 100.74315 \r\nL 159.681025 95.914254 \r\nL 165.803282 90.639614 \r\nL 171.925539 84.919229 \r\nL 178.047796 78.7531 \r\nL 184.170053 72.141226 \r\nL 190.29231 65.083608 \r\nL 196.414567 57.580247 \r\nL 202.536824 49.63114 \r\nL 208.659081 41.23629 \r\nL 214.781338 32.395695 \r\nL 220.903595 23.109356 \r\nL 227.025852 13.377273 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_16\">\r\n    <path clip-path=\"url(#p5dbbe82ad7)\" d=\"M 49.480398 136.922727 \r\nL 55.602655 135.436913 \r\nL 61.724912 133.951099 \r\nL 67.847169 132.465285 \r\nL 73.969426 130.97947 \r\nL 80.091683 129.493656 \r\nL 86.21394 128.007842 \r\nL 92.336197 126.522028 \r\nL 98.458454 125.036213 \r\nL 104.580711 123.550399 \r\nL 110.702968 122.064585 \r\nL 116.825225 120.578771 \r\nL 122.947482 119.092956 \r\nL 129.069739 117.607142 \r\nL 135.191996 116.121328 \r\nL 141.314254 114.635514 \r\nL 147.436511 113.149699 \r\nL 153.558768 111.663885 \r\nL 159.681025 110.178071 \r\nL 165.803282 108.692257 \r\nL 171.925539 107.206442 \r\nL 178.047796 105.720628 \r\nL 184.170053 104.234814 \r\nL 190.29231 102.749 \r\nL 196.414567 101.263185 \r\nL 202.536824 99.777371 \r\nL 208.659081 98.291557 \r\nL 214.781338 96.805743 \r\nL 220.903595 95.319928 \r\nL 227.025852 93.834114 \r\n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"patch_3\">\r\n    <path d=\"M 40.603125 143.1 \r\nL 40.603125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_4\">\r\n    <path d=\"M 235.903125 143.1 \r\nL 235.903125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_5\">\r\n    <path d=\"M 40.603125 143.1 \r\nL 235.903125 143.1 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_6\">\r\n    <path d=\"M 40.603125 7.2 \r\nL 235.903125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"legend_1\">\r\n    <g id=\"patch_7\">\r\n     <path d=\"M 47.603125 44.55625 \r\nL 172.153125 44.55625 \r\nQ 174.153125 44.55625 174.153125 42.55625 \r\nL 174.153125 14.2 \r\nQ 174.153125 12.2 172.153125 12.2 \r\nL 47.603125 12.2 \r\nQ 45.603125 12.2 45.603125 14.2 \r\nL 45.603125 42.55625 \r\nQ 45.603125 44.55625 47.603125 44.55625 \r\nz\r\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\r\n    </g>\r\n    <g id=\"line2d_17\">\r\n     <path d=\"M 49.603125 20.298437 \r\nL 69.603125 20.298437 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_18\"/>\r\n    <g id=\"text_10\">\r\n     <!-- f(x) -->\r\n     <g transform=\"translate(77.603125 23.798437)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-102\"/>\r\n      <use x=\"35.205078\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"74.21875\" xlink:href=\"#DejaVuSans-120\"/>\r\n      <use x=\"133.398438\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_19\">\r\n     <path d=\"M 49.603125 34.976562 \r\nL 69.603125 34.976562 \r\n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_20\"/>\r\n    <g id=\"text_11\">\r\n     <!-- Tangent line (x=1) -->\r\n     <g transform=\"translate(77.603125 38.476562)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M -0.296875 72.90625 \r\nL 61.375 72.90625 \r\nL 61.375 64.59375 \r\nL 35.5 64.59375 \r\nL 35.5 0 \r\nL 25.59375 0 \r\nL 25.59375 64.59375 \r\nL -0.296875 64.59375 \r\nz\r\n\" id=\"DejaVuSans-84\"/>\r\n       <path d=\"M 34.28125 27.484375 \r\nQ 23.390625 27.484375 19.1875 25 \r\nQ 14.984375 22.515625 14.984375 16.5 \r\nQ 14.984375 11.71875 18.140625 8.90625 \r\nQ 21.296875 6.109375 26.703125 6.109375 \r\nQ 34.1875 6.109375 38.703125 11.40625 \r\nQ 43.21875 16.703125 43.21875 25.484375 \r\nL 43.21875 27.484375 \r\nz\r\nM 52.203125 31.203125 \r\nL 52.203125 0 \r\nL 43.21875 0 \r\nL 43.21875 8.296875 \r\nQ 40.140625 3.328125 35.546875 0.953125 \r\nQ 30.953125 -1.421875 24.3125 -1.421875 \r\nQ 15.921875 -1.421875 10.953125 3.296875 \r\nQ 6 8.015625 6 15.921875 \r\nQ 6 25.140625 12.171875 29.828125 \r\nQ 18.359375 34.515625 30.609375 34.515625 \r\nL 43.21875 34.515625 \r\nL 43.21875 35.40625 \r\nQ 43.21875 41.609375 39.140625 45 \r\nQ 35.0625 48.390625 27.6875 48.390625 \r\nQ 23 48.390625 18.546875 47.265625 \r\nQ 14.109375 46.140625 10.015625 43.890625 \r\nL 10.015625 52.203125 \r\nQ 14.9375 54.109375 19.578125 55.046875 \r\nQ 24.21875 56 28.609375 56 \r\nQ 40.484375 56 46.34375 49.84375 \r\nQ 52.203125 43.703125 52.203125 31.203125 \r\nz\r\n\" id=\"DejaVuSans-97\"/>\r\n       <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-110\"/>\r\n       <path d=\"M 45.40625 27.984375 \r\nQ 45.40625 37.75 41.375 43.109375 \r\nQ 37.359375 48.484375 30.078125 48.484375 \r\nQ 22.859375 48.484375 18.828125 43.109375 \r\nQ 14.796875 37.75 14.796875 27.984375 \r\nQ 14.796875 18.265625 18.828125 12.890625 \r\nQ 22.859375 7.515625 30.078125 7.515625 \r\nQ 37.359375 7.515625 41.375 12.890625 \r\nQ 45.40625 18.265625 45.40625 27.984375 \r\nz\r\nM 54.390625 6.78125 \r\nQ 54.390625 -7.171875 48.1875 -13.984375 \r\nQ 42 -20.796875 29.203125 -20.796875 \r\nQ 24.46875 -20.796875 20.265625 -20.09375 \r\nQ 16.0625 -19.390625 12.109375 -17.921875 \r\nL 12.109375 -9.1875 \r\nQ 16.0625 -11.328125 19.921875 -12.34375 \r\nQ 23.78125 -13.375 27.78125 -13.375 \r\nQ 36.625 -13.375 41.015625 -8.765625 \r\nQ 45.40625 -4.15625 45.40625 5.171875 \r\nL 45.40625 9.625 \r\nQ 42.625 4.78125 38.28125 2.390625 \r\nQ 33.9375 0 27.875 0 \r\nQ 17.828125 0 11.671875 7.65625 \r\nQ 5.515625 15.328125 5.515625 27.984375 \r\nQ 5.515625 40.671875 11.671875 48.328125 \r\nQ 17.828125 56 27.875 56 \r\nQ 33.9375 56 38.28125 53.609375 \r\nQ 42.625 51.21875 45.40625 46.390625 \r\nL 45.40625 54.6875 \r\nL 54.390625 54.6875 \r\nz\r\n\" id=\"DejaVuSans-103\"/>\r\n       <path d=\"M 56.203125 29.59375 \r\nL 56.203125 25.203125 \r\nL 14.890625 25.203125 \r\nQ 15.484375 15.921875 20.484375 11.0625 \r\nQ 25.484375 6.203125 34.421875 6.203125 \r\nQ 39.59375 6.203125 44.453125 7.46875 \r\nQ 49.3125 8.734375 54.109375 11.28125 \r\nL 54.109375 2.78125 \r\nQ 49.265625 0.734375 44.1875 -0.34375 \r\nQ 39.109375 -1.421875 33.890625 -1.421875 \r\nQ 20.796875 -1.421875 13.15625 6.1875 \r\nQ 5.515625 13.8125 5.515625 26.8125 \r\nQ 5.515625 40.234375 12.765625 48.109375 \r\nQ 20.015625 56 32.328125 56 \r\nQ 43.359375 56 49.78125 48.890625 \r\nQ 56.203125 41.796875 56.203125 29.59375 \r\nz\r\nM 47.21875 32.234375 \r\nQ 47.125 39.59375 43.09375 43.984375 \r\nQ 39.0625 48.390625 32.421875 48.390625 \r\nQ 24.90625 48.390625 20.390625 44.140625 \r\nQ 15.875 39.890625 15.1875 32.171875 \r\nz\r\n\" id=\"DejaVuSans-101\"/>\r\n       <path d=\"M 18.3125 70.21875 \r\nL 18.3125 54.6875 \r\nL 36.8125 54.6875 \r\nL 36.8125 47.703125 \r\nL 18.3125 47.703125 \r\nL 18.3125 18.015625 \r\nQ 18.3125 11.328125 20.140625 9.421875 \r\nQ 21.96875 7.515625 27.59375 7.515625 \r\nL 36.8125 7.515625 \r\nL 36.8125 0 \r\nL 27.59375 0 \r\nQ 17.1875 0 13.234375 3.875 \r\nQ 9.28125 7.765625 9.28125 18.015625 \r\nL 9.28125 47.703125 \r\nL 2.6875 47.703125 \r\nL 2.6875 54.6875 \r\nL 9.28125 54.6875 \r\nL 9.28125 70.21875 \r\nz\r\n\" id=\"DejaVuSans-116\"/>\r\n       <path id=\"DejaVuSans-32\"/>\r\n       <path d=\"M 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\n\" id=\"DejaVuSans-108\"/>\r\n       <path d=\"M 9.421875 54.6875 \r\nL 18.40625 54.6875 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\nM 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 64.59375 \r\nL 9.421875 64.59375 \r\nz\r\n\" id=\"DejaVuSans-105\"/>\r\n       <path d=\"M 10.59375 45.40625 \r\nL 73.1875 45.40625 \r\nL 73.1875 37.203125 \r\nL 10.59375 37.203125 \r\nz\r\nM 10.59375 25.484375 \r\nL 73.1875 25.484375 \r\nL 73.1875 17.1875 \r\nL 10.59375 17.1875 \r\nz\r\n\" id=\"DejaVuSans-61\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-84\"/>\r\n      <use x=\"44.583984\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"105.863281\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"169.242188\" xlink:href=\"#DejaVuSans-103\"/>\r\n      <use x=\"232.71875\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"294.242188\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"357.621094\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"396.830078\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"428.617188\" xlink:href=\"#DejaVuSans-108\"/>\r\n      <use x=\"456.400391\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"484.183594\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"547.5625\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"609.085938\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"640.873047\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"679.886719\" xlink:href=\"#DejaVuSans-120\"/>\r\n      <use x=\"739.066406\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"822.855469\" xlink:href=\"#DejaVuSans-49\"/>\r\n      <use x=\"886.478516\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n  </g>\r\n </g>\r\n <defs>\r\n  <clipPath id=\"p5dbbe82ad7\">\r\n   <rect height=\"135.9\" width=\"195.3\" x=\"40.603125\" y=\"7.2\"/>\r\n  </clipPath>\r\n </defs>\r\n</svg>\r\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## 偏导数

到目前为止，我们只讨论了仅含一个变量的函数的微分。
在深度学习中，函数通常依赖于许多变量。
因此，我们需要将微分的思想推广到*多元函数*（multivariate function）上。

设$y = f(x_1, x_2, \ldots, x_n)$是一个具有$n$个变量的函数。
$y$关于第$i$个参数$x_i$的*偏导数*（partial derivative）为：

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$

为了计算$\frac{\partial y}{\partial x_i}$，
我们可以简单地将$x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$看作常数，
并计算$y$关于$x_i$的导数。
对于偏导数的表示，以下是等价的：

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

## 梯度
:label:`subsec_calculus-grad`

我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的*梯度*（gradient）向量。
具体而言，设函数$f:\mathbb{R}^n\rightarrow\mathbb{R}$的输入是
一个$n$维向量$\mathbf{x}=[x_1,x_2,\ldots,x_n]^\top$，并且输出是一个标量。
函数$f(\mathbf{x})$相对于$\mathbf{x}$的梯度是一个包含$n$个偏导数的向量:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_n}\bigg]^\top,$$

其中$\nabla_{\mathbf{x}} f(\mathbf{x})$通常在没有歧义时被$\nabla f(\mathbf{x})$取代。

假设$\mathbf{x}$为$n$维向量，在微分多元函数时经常使用以下规则:

* 对于所有$\mathbf{A} \in \mathbb{R}^{m \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times m}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$
* 对于所有$\mathbf{A} \in \mathbb{R}^{n \times n}$，都有$\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$
* $\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$

同样，对于任何矩阵$\mathbf{X}$，都有$\nabla_{\mathbf{X}} \|\mathbf{X} \|_F^2 = 2\mathbf{X}$。
正如我们之后将看到的，梯度对于设计深度学习中的优化算法有很大用处。

## 链式法则

然而，上面方法可能很难找到梯度。
这是因为在深度学习中，多元函数通常是*复合*（composite）的，
所以我们可能没法应用上述任何规则来微分这些函数。
幸运的是，链式法则使我们能够微分复合函数。

让我们先考虑单变量函数。假设函数$y=f(u)$和$u=g(x)$都是可微的，根据链式法则：

$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$

现在让我们把注意力转向一个更一般的场景，即函数具有任意数量的变量的情况。
假设可微分函数$y$有变量$u_1, u_2, \ldots, u_m$，其中每个可微分函数$u_i$都有变量$x_1, x_2, \ldots, x_n$。
注意，$y$是$x_1, x_2， \ldots, x_n$的函数。
对于任意$i = 1, 2, \ldots, n$，链式法则给出：

$$\frac{dy}{dx_i} = \frac{dy}{du_1} \frac{du_1}{dx_i} + \frac{dy}{du_2} \frac{du_2}{dx_i} + \cdots + \frac{dy}{du_m} \frac{du_m}{dx_i}$$

## 小结

* 微分和积分是微积分的两个分支，前者可以应用于深度学习中的优化问题。
* 导数可以被解释为函数相对于其变量的瞬时变化率，它也是函数曲线的切线的斜率。
* 梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数。
* 链式法则使我们能够微分复合函数。

## 练习

1. 绘制函数$y = f(x) = x^3 - \frac{1}{x}$和其在$x = 1$处切线的图像。
1. 求函数$f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$的梯度。
1. 函数$f(\mathbf{x}) = \|\mathbf{x}\|_2$的梯度是什么？
1. 你可以写出函数$u = f(x, y, z)$，其中$x = x(a, b)$，$y = y(a, b)$，$z = z(a, b)$的链式法则吗?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1755)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1756)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1754)
:end_tab:

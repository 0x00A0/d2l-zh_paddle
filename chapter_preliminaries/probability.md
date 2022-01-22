None
None
None
None
# 概率
:label:`sec_prob`

简单地说，机器学习就是做出预测。

根据病人的临床病史，我们可能想预测他们在下一年心脏病发作的*概率*。
在飞机喷气发动机的异常检测中，我们想要评估一组发动机读数为正常运行情况的概率有多大。
在强化学习中，我们希望智能体（agent）能在一个环境中智能地行动。
这意味着我们需要考虑在每种可行的行为下获得高奖励的概率。
当我们建立推荐系统时，我们也需要考虑概率。
例如，假设我们为一家大型在线书店工作，我们可能希望估计某些用户购买特定图书的概率。
为此，我们需要使用概率学。
有完整的课程、专业、论文、职业、甚至院系，都致力于概率学的工作。
所以很自然地，我们在这部分的目标不是教授你整个科目。
相反，我们希望教给你在基础的概率知识，使你能够开始构建你的第一个深度学习模型，
以便你可以开始自己探索它。

现在让我们更认真地考虑第一个例子：根据照片区分猫和狗。
这听起来可能很简单，但对于机器却可能是一个艰巨的挑战。
首先，问题的难度可能取决于图像的分辨率。

![不同分辨率的图像 ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, 和 $160 \times 160$ pixels)](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

如 :numref:`fig_cat_dog`所示，虽然人类很容易以$160 \times 160$像素的分辨率识别猫和狗，
但它在$40\times40$像素上变得具有挑战性，而且在$10 \times 10$像素下几乎是不可能的。
换句话说，我们在很远的距离（从而降低分辨率）区分猫和狗的能力可能会变为猜测。
概率给了我们一种正式的途径来说明我们的确定性水平。
如果我们完全肯定图像是一只猫，我们说标签$y$是"猫"的*概率*，表示为$P(y=$"猫"$)$等于$1$。
如果我们没有证据表明$y=$“猫”或$y=$“狗”，那么我们可以说这两种可能性是相等的，
即$P(y=$"猫"$)=P(y=$"狗"$)=0.5$。
如果我们不十分确定图像描绘的是一只猫，我们可以将概率赋值为$0.5<P(y=$"猫"$)<1$。

现在考虑第二个例子：给出一些天气监测数据，我们想预测明天北京下雨的概率。
如果是夏天，下雨的概率是0.5。

在这两种情况下，我们都不确定结果，但这两种情况之间有一个关键区别。
在第一种情况中，图像实际上是狗或猫二选一。
在第二种情况下，结果实际上是一个随机的事件。
因此，概率是一种灵活的语言，用于说明我们的确定程度，并且它可以有效地应用于广泛的领域中。

## 基本概率论

假设我们掷骰子，想知道看到1的几率有多大，而不是看到另一个数字。
如果骰子是公平的，那么所有六个结果$\{1, \ldots, 6\}$都有相同的可能发生，
因此我们可以说$1$发生的概率为$\frac{1}{6}$。

然而现实生活中，对于我们从工厂收到的真实骰子，我们需要检查它是否有瑕疵。
检查骰子的唯一方法是多次投掷并记录结果。
对于每个骰子，我们将观察到$\{1, \ldots, 6\}$中的一个值。
对于每个值，一种自然的方法是将它出现的次数除以投掷的总次数，
即此*事件*（event）概率的*估计值*。
*大数定律*（law of large numbers）告诉我们：
随着投掷次数的增加，这个估计值会越来越接近真实的潜在概率。
让我们用代码试一试！

首先，我们导入必要的软件包。

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

```{.python .input  n=3}
#@tab paddle
%matplotlib inline
import sys
from d2l import paddle as d2l
import paddle
import numpy as np
```

在统计学中，我们把从概率分布中抽取样本的过程称为*抽样*（sampling）。
笼统来说，可以把*分布*（distribution）看作是对事件的概率分配，
稍后我们将给出的更正式定义。
将概率分配给一些离散选择的分布称为*多项分布*（multinomial distribution）。

为了抽取一个样本，即掷骰子，我们只需传入一个概率向量。
输出是另一个相同长度的向量：它在索引$i$处的值是采样结果中$i$出现的次数。

```{.python .input  n=23}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.json .output n=23}
[
 {
  "data": {
   "text/plain": "array([0, 0, 0, 0, 0, 1])"
  },
  "execution_count": 23,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

```{.python .input  n=2}
#@tab paddle
fair_probs = [1.0 / 6] * 6
paddle.to_tensor(np.random.multinomial(1, fair_probs))
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "Tensor(shape=[6], dtype=int32, place=CPUPlace, stop_gradient=True,\n       [0, 0, 0, 0, 1, 0])"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

在估计一个骰子的公平性时，我们希望从同一分布中生成多个样本。
如果用Python的for循环来完成这个任务，速度会慢得惊人。
因此我们使用深度学习框架的函数同时抽取多个样本，得到我们想要的任意形状的独立样本数组。

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

```{.python .input  n=25}
#@tab paddle
paddle.to_tensor(np.random.multinomial(10, fair_probs))
```

```{.json .output n=25}
[
 {
  "data": {
   "text/plain": "Tensor(shape=[6], dtype=int32, place=CPUPlace, stop_gradient=True,\n       [3, 2, 2, 1, 1, 1])"
  },
  "execution_count": 25,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

现在我们知道如何对骰子进行采样，我们可以模拟1000次投掷。
然后，我们可以统计1000次投掷后，每个数字被投中了多少次。
具体来说，我们计算相对频率，以作为真实概率的估计。

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# 将结果存储为32位浮点数以进行除法
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # 相对频率作为估计值
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

```{.python .input  n=28}
#@tab paddle
counts = paddle.to_tensor(np.random.multinomial(1000, fair_probs).astype(np.float32))
counts / 1000
```

```{.json .output n=28}
[
 {
  "data": {
   "text/plain": "Tensor(shape=[6], dtype=float32, place=CPUPlace, stop_gradient=True,\n       [0.16300000, 0.18900001, 0.16300000, 0.13900001, 0.18000001, 0.16600001])"
  },
  "execution_count": 28,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

因为我们是从一个公平的骰子中生成的数据，我们知道每个结果都有真实的概率$\frac{1}{6}$，
大约是$0.167$，所以上面输出的估计值看起来不错。

我们也可以看到这些概率如何随着时间的推移收敛到真实概率。
让我们进行500组实验，每组抽取10个样本。

```{.python .input}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input  n=35}
#@tab paddle
counts =paddle.to_tensor(np.random.multinomial(10, fair_probs, size=500),dtype="float32")
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdim=True)
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.json .output n=35}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "E:\\Codes\\d2l-zh_paddle\\d2l\\paddle.py:35: DeprecationWarning: `set_matplotlib_formats` is deprecated since IPython 7.23, directly use `matplotlib_inline.backend_inline.set_matplotlib_formats()`\n  display.set_matplotlib_formats('svg')\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\r\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\r\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\r\n<!-- Created with matplotlib (https://matplotlib.org/) -->\r\n<svg height=\"289.37625pt\" version=\"1.1\" viewBox=\"0 0 385.78125 289.37625\" width=\"385.78125pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\r\n <metadata>\r\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\r\n   <cc:Work>\r\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\r\n    <dc:date>2022-01-20T15:04:55.880281</dc:date>\r\n    <dc:format>image/svg+xml</dc:format>\r\n    <dc:creator>\r\n     <cc:Agent>\r\n      <dc:title>Matplotlib v3.3.3, https://matplotlib.org/</dc:title>\r\n     </cc:Agent>\r\n    </dc:creator>\r\n   </cc:Work>\r\n  </rdf:RDF>\r\n </metadata>\r\n <defs>\r\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\r\n </defs>\r\n <g id=\"figure_1\">\r\n  <g id=\"patch_1\">\r\n   <path d=\"M 0 289.37625 \r\nL 385.78125 289.37625 \r\nL 385.78125 0 \r\nL 0 0 \r\nz\r\n\" style=\"fill:none;\"/>\r\n  </g>\r\n  <g id=\"axes_1\">\r\n   <g id=\"patch_2\">\r\n    <path d=\"M 43.78125 251.82 \r\nL 378.58125 251.82 \r\nL 378.58125 7.2 \r\nL 43.78125 7.2 \r\nz\r\n\" style=\"fill:#ffffff;\"/>\r\n   </g>\r\n   <g id=\"matplotlib.axis_1\">\r\n    <g id=\"xtick_1\">\r\n     <g id=\"line2d_1\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL 0 3.5 \r\n\" id=\"m1b73e62cf5\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"58.999432\" xlink:href=\"#m1b73e62cf5\" y=\"251.82\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_1\">\r\n      <!-- 0 -->\r\n      <g transform=\"translate(55.818182 266.418437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 31.78125 66.40625 \r\nQ 24.171875 66.40625 20.328125 58.90625 \r\nQ 16.5 51.421875 16.5 36.375 \r\nQ 16.5 21.390625 20.328125 13.890625 \r\nQ 24.171875 6.390625 31.78125 6.390625 \r\nQ 39.453125 6.390625 43.28125 13.890625 \r\nQ 47.125 21.390625 47.125 36.375 \r\nQ 47.125 51.421875 43.28125 58.90625 \r\nQ 39.453125 66.40625 31.78125 66.40625 \r\nz\r\nM 31.78125 74.21875 \r\nQ 44.046875 74.21875 50.515625 64.515625 \r\nQ 56.984375 54.828125 56.984375 36.375 \r\nQ 56.984375 17.96875 50.515625 8.265625 \r\nQ 44.046875 -1.421875 31.78125 -1.421875 \r\nQ 19.53125 -1.421875 13.0625 8.265625 \r\nQ 6.59375 17.96875 6.59375 36.375 \r\nQ 6.59375 54.828125 13.0625 64.515625 \r\nQ 19.53125 74.21875 31.78125 74.21875 \r\nz\r\n\" id=\"DejaVuSans-48\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_2\">\r\n     <g id=\"line2d_2\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"119.994149\" xlink:href=\"#m1b73e62cf5\" y=\"251.82\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_2\">\r\n      <!-- 100 -->\r\n      <g transform=\"translate(110.450399 266.418437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 12.40625 8.296875 \r\nL 28.515625 8.296875 \r\nL 28.515625 63.921875 \r\nL 10.984375 60.40625 \r\nL 10.984375 69.390625 \r\nL 28.421875 72.90625 \r\nL 38.28125 72.90625 \r\nL 38.28125 8.296875 \r\nL 54.390625 8.296875 \r\nL 54.390625 0 \r\nL 12.40625 0 \r\nz\r\n\" id=\"DejaVuSans-49\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-49\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_3\">\r\n     <g id=\"line2d_3\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"180.988865\" xlink:href=\"#m1b73e62cf5\" y=\"251.82\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_3\">\r\n      <!-- 200 -->\r\n      <g transform=\"translate(171.445115 266.418437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 19.1875 8.296875 \r\nL 53.609375 8.296875 \r\nL 53.609375 0 \r\nL 7.328125 0 \r\nL 7.328125 8.296875 \r\nQ 12.9375 14.109375 22.625 23.890625 \r\nQ 32.328125 33.6875 34.8125 36.53125 \r\nQ 39.546875 41.84375 41.421875 45.53125 \r\nQ 43.3125 49.21875 43.3125 52.78125 \r\nQ 43.3125 58.59375 39.234375 62.25 \r\nQ 35.15625 65.921875 28.609375 65.921875 \r\nQ 23.96875 65.921875 18.8125 64.3125 \r\nQ 13.671875 62.703125 7.8125 59.421875 \r\nL 7.8125 69.390625 \r\nQ 13.765625 71.78125 18.9375 73 \r\nQ 24.125 74.21875 28.421875 74.21875 \r\nQ 39.75 74.21875 46.484375 68.546875 \r\nQ 53.21875 62.890625 53.21875 53.421875 \r\nQ 53.21875 48.921875 51.53125 44.890625 \r\nQ 49.859375 40.875 45.40625 35.40625 \r\nQ 44.1875 33.984375 37.640625 27.21875 \r\nQ 31.109375 20.453125 19.1875 8.296875 \r\nz\r\n\" id=\"DejaVuSans-50\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-50\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_4\">\r\n     <g id=\"line2d_4\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"241.983582\" xlink:href=\"#m1b73e62cf5\" y=\"251.82\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_4\">\r\n      <!-- 300 -->\r\n      <g transform=\"translate(232.439832 266.418437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 40.578125 39.3125 \r\nQ 47.65625 37.796875 51.625 33 \r\nQ 55.609375 28.21875 55.609375 21.1875 \r\nQ 55.609375 10.40625 48.1875 4.484375 \r\nQ 40.765625 -1.421875 27.09375 -1.421875 \r\nQ 22.515625 -1.421875 17.65625 -0.515625 \r\nQ 12.796875 0.390625 7.625 2.203125 \r\nL 7.625 11.71875 \r\nQ 11.71875 9.328125 16.59375 8.109375 \r\nQ 21.484375 6.890625 26.8125 6.890625 \r\nQ 36.078125 6.890625 40.9375 10.546875 \r\nQ 45.796875 14.203125 45.796875 21.1875 \r\nQ 45.796875 27.640625 41.28125 31.265625 \r\nQ 36.765625 34.90625 28.71875 34.90625 \r\nL 20.21875 34.90625 \r\nL 20.21875 43.015625 \r\nL 29.109375 43.015625 \r\nQ 36.375 43.015625 40.234375 45.921875 \r\nQ 44.09375 48.828125 44.09375 54.296875 \r\nQ 44.09375 59.90625 40.109375 62.90625 \r\nQ 36.140625 65.921875 28.71875 65.921875 \r\nQ 24.65625 65.921875 20.015625 65.03125 \r\nQ 15.375 64.15625 9.8125 62.3125 \r\nL 9.8125 71.09375 \r\nQ 15.4375 72.65625 20.34375 73.4375 \r\nQ 25.25 74.21875 29.59375 74.21875 \r\nQ 40.828125 74.21875 47.359375 69.109375 \r\nQ 53.90625 64.015625 53.90625 55.328125 \r\nQ 53.90625 49.265625 50.4375 45.09375 \r\nQ 46.96875 40.921875 40.578125 39.3125 \r\nz\r\n\" id=\"DejaVuSans-51\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-51\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_5\">\r\n     <g id=\"line2d_5\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"302.978299\" xlink:href=\"#m1b73e62cf5\" y=\"251.82\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_5\">\r\n      <!-- 400 -->\r\n      <g transform=\"translate(293.434549 266.418437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 37.796875 64.3125 \r\nL 12.890625 25.390625 \r\nL 37.796875 25.390625 \r\nz\r\nM 35.203125 72.90625 \r\nL 47.609375 72.90625 \r\nL 47.609375 25.390625 \r\nL 58.015625 25.390625 \r\nL 58.015625 17.1875 \r\nL 47.609375 17.1875 \r\nL 47.609375 0 \r\nL 37.796875 0 \r\nL 37.796875 17.1875 \r\nL 4.890625 17.1875 \r\nL 4.890625 26.703125 \r\nz\r\n\" id=\"DejaVuSans-52\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-52\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"xtick_6\">\r\n     <g id=\"line2d_6\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"363.973015\" xlink:href=\"#m1b73e62cf5\" y=\"251.82\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_6\">\r\n      <!-- 500 -->\r\n      <g transform=\"translate(354.429265 266.418437)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.796875 72.90625 \r\nL 49.515625 72.90625 \r\nL 49.515625 64.59375 \r\nL 19.828125 64.59375 \r\nL 19.828125 46.734375 \r\nQ 21.96875 47.46875 24.109375 47.828125 \r\nQ 26.265625 48.1875 28.421875 48.1875 \r\nQ 40.625 48.1875 47.75 41.5 \r\nQ 54.890625 34.8125 54.890625 23.390625 \r\nQ 54.890625 11.625 47.5625 5.09375 \r\nQ 40.234375 -1.421875 26.90625 -1.421875 \r\nQ 22.3125 -1.421875 17.546875 -0.640625 \r\nQ 12.796875 0.140625 7.71875 1.703125 \r\nL 7.71875 11.625 \r\nQ 12.109375 9.234375 16.796875 8.0625 \r\nQ 21.484375 6.890625 26.703125 6.890625 \r\nQ 35.15625 6.890625 40.078125 11.328125 \r\nQ 45.015625 15.765625 45.015625 23.390625 \r\nQ 45.015625 31 40.078125 35.4375 \r\nQ 35.15625 39.890625 26.703125 39.890625 \r\nQ 22.75 39.890625 18.8125 39.015625 \r\nQ 14.890625 38.140625 10.796875 36.28125 \r\nz\r\n\" id=\"DejaVuSans-53\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-53\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_7\">\r\n     <!-- Groups of experiments -->\r\n     <g transform=\"translate(154.035156 280.096562)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 59.515625 10.40625 \r\nL 59.515625 29.984375 \r\nL 43.40625 29.984375 \r\nL 43.40625 38.09375 \r\nL 69.28125 38.09375 \r\nL 69.28125 6.78125 \r\nQ 63.578125 2.734375 56.6875 0.65625 \r\nQ 49.8125 -1.421875 42 -1.421875 \r\nQ 24.90625 -1.421875 15.25 8.5625 \r\nQ 5.609375 18.5625 5.609375 36.375 \r\nQ 5.609375 54.25 15.25 64.234375 \r\nQ 24.90625 74.21875 42 74.21875 \r\nQ 49.125 74.21875 55.546875 72.453125 \r\nQ 61.96875 70.703125 67.390625 67.28125 \r\nL 67.390625 56.78125 \r\nQ 61.921875 61.421875 55.765625 63.765625 \r\nQ 49.609375 66.109375 42.828125 66.109375 \r\nQ 29.4375 66.109375 22.71875 58.640625 \r\nQ 16.015625 51.171875 16.015625 36.375 \r\nQ 16.015625 21.625 22.71875 14.15625 \r\nQ 29.4375 6.6875 42.828125 6.6875 \r\nQ 48.046875 6.6875 52.140625 7.59375 \r\nQ 56.25 8.5 59.515625 10.40625 \r\nz\r\n\" id=\"DejaVuSans-71\"/>\r\n       <path d=\"M 41.109375 46.296875 \r\nQ 39.59375 47.171875 37.8125 47.578125 \r\nQ 36.03125 48 33.890625 48 \r\nQ 26.265625 48 22.1875 43.046875 \r\nQ 18.109375 38.09375 18.109375 28.8125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 20.953125 51.171875 25.484375 53.578125 \r\nQ 30.03125 56 36.53125 56 \r\nQ 37.453125 56 38.578125 55.875 \r\nQ 39.703125 55.765625 41.0625 55.515625 \r\nz\r\n\" id=\"DejaVuSans-114\"/>\r\n       <path d=\"M 30.609375 48.390625 \r\nQ 23.390625 48.390625 19.1875 42.75 \r\nQ 14.984375 37.109375 14.984375 27.296875 \r\nQ 14.984375 17.484375 19.15625 11.84375 \r\nQ 23.34375 6.203125 30.609375 6.203125 \r\nQ 37.796875 6.203125 41.984375 11.859375 \r\nQ 46.1875 17.53125 46.1875 27.296875 \r\nQ 46.1875 37.015625 41.984375 42.703125 \r\nQ 37.796875 48.390625 30.609375 48.390625 \r\nz\r\nM 30.609375 56 \r\nQ 42.328125 56 49.015625 48.375 \r\nQ 55.71875 40.765625 55.71875 27.296875 \r\nQ 55.71875 13.875 49.015625 6.21875 \r\nQ 42.328125 -1.421875 30.609375 -1.421875 \r\nQ 18.84375 -1.421875 12.171875 6.21875 \r\nQ 5.515625 13.875 5.515625 27.296875 \r\nQ 5.515625 40.765625 12.171875 48.375 \r\nQ 18.84375 56 30.609375 56 \r\nz\r\n\" id=\"DejaVuSans-111\"/>\r\n       <path d=\"M 8.5 21.578125 \r\nL 8.5 54.6875 \r\nL 17.484375 54.6875 \r\nL 17.484375 21.921875 \r\nQ 17.484375 14.15625 20.5 10.265625 \r\nQ 23.53125 6.390625 29.59375 6.390625 \r\nQ 36.859375 6.390625 41.078125 11.03125 \r\nQ 45.3125 15.671875 45.3125 23.6875 \r\nL 45.3125 54.6875 \r\nL 54.296875 54.6875 \r\nL 54.296875 0 \r\nL 45.3125 0 \r\nL 45.3125 8.40625 \r\nQ 42.046875 3.421875 37.71875 1 \r\nQ 33.40625 -1.421875 27.6875 -1.421875 \r\nQ 18.265625 -1.421875 13.375 4.4375 \r\nQ 8.5 10.296875 8.5 21.578125 \r\nz\r\nM 31.109375 56 \r\nz\r\n\" id=\"DejaVuSans-117\"/>\r\n       <path d=\"M 18.109375 8.203125 \r\nL 18.109375 -20.796875 \r\nL 9.078125 -20.796875 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nz\r\nM 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-112\"/>\r\n       <path d=\"M 44.28125 53.078125 \r\nL 44.28125 44.578125 \r\nQ 40.484375 46.53125 36.375 47.5 \r\nQ 32.28125 48.484375 27.875 48.484375 \r\nQ 21.1875 48.484375 17.84375 46.4375 \r\nQ 14.5 44.390625 14.5 40.28125 \r\nQ 14.5 37.15625 16.890625 35.375 \r\nQ 19.28125 33.59375 26.515625 31.984375 \r\nL 29.59375 31.296875 \r\nQ 39.15625 29.25 43.1875 25.515625 \r\nQ 47.21875 21.78125 47.21875 15.09375 \r\nQ 47.21875 7.46875 41.1875 3.015625 \r\nQ 35.15625 -1.421875 24.609375 -1.421875 \r\nQ 20.21875 -1.421875 15.453125 -0.5625 \r\nQ 10.6875 0.296875 5.421875 2 \r\nL 5.421875 11.28125 \r\nQ 10.40625 8.6875 15.234375 7.390625 \r\nQ 20.0625 6.109375 24.8125 6.109375 \r\nQ 31.15625 6.109375 34.5625 8.28125 \r\nQ 37.984375 10.453125 37.984375 14.40625 \r\nQ 37.984375 18.0625 35.515625 20.015625 \r\nQ 33.0625 21.96875 24.703125 23.78125 \r\nL 21.578125 24.515625 \r\nQ 13.234375 26.265625 9.515625 29.90625 \r\nQ 5.8125 33.546875 5.8125 39.890625 \r\nQ 5.8125 47.609375 11.28125 51.796875 \r\nQ 16.75 56 26.8125 56 \r\nQ 31.78125 56 36.171875 55.265625 \r\nQ 40.578125 54.546875 44.28125 53.078125 \r\nz\r\n\" id=\"DejaVuSans-115\"/>\r\n       <path id=\"DejaVuSans-32\"/>\r\n       <path d=\"M 37.109375 75.984375 \r\nL 37.109375 68.5 \r\nL 28.515625 68.5 \r\nQ 23.6875 68.5 21.796875 66.546875 \r\nQ 19.921875 64.59375 19.921875 59.515625 \r\nL 19.921875 54.6875 \r\nL 34.71875 54.6875 \r\nL 34.71875 47.703125 \r\nL 19.921875 47.703125 \r\nL 19.921875 0 \r\nL 10.890625 0 \r\nL 10.890625 47.703125 \r\nL 2.296875 47.703125 \r\nL 2.296875 54.6875 \r\nL 10.890625 54.6875 \r\nL 10.890625 58.5 \r\nQ 10.890625 67.625 15.140625 71.796875 \r\nQ 19.390625 75.984375 28.609375 75.984375 \r\nz\r\n\" id=\"DejaVuSans-102\"/>\r\n       <path d=\"M 56.203125 29.59375 \r\nL 56.203125 25.203125 \r\nL 14.890625 25.203125 \r\nQ 15.484375 15.921875 20.484375 11.0625 \r\nQ 25.484375 6.203125 34.421875 6.203125 \r\nQ 39.59375 6.203125 44.453125 7.46875 \r\nQ 49.3125 8.734375 54.109375 11.28125 \r\nL 54.109375 2.78125 \r\nQ 49.265625 0.734375 44.1875 -0.34375 \r\nQ 39.109375 -1.421875 33.890625 -1.421875 \r\nQ 20.796875 -1.421875 13.15625 6.1875 \r\nQ 5.515625 13.8125 5.515625 26.8125 \r\nQ 5.515625 40.234375 12.765625 48.109375 \r\nQ 20.015625 56 32.328125 56 \r\nQ 43.359375 56 49.78125 48.890625 \r\nQ 56.203125 41.796875 56.203125 29.59375 \r\nz\r\nM 47.21875 32.234375 \r\nQ 47.125 39.59375 43.09375 43.984375 \r\nQ 39.0625 48.390625 32.421875 48.390625 \r\nQ 24.90625 48.390625 20.390625 44.140625 \r\nQ 15.875 39.890625 15.1875 32.171875 \r\nz\r\n\" id=\"DejaVuSans-101\"/>\r\n       <path d=\"M 54.890625 54.6875 \r\nL 35.109375 28.078125 \r\nL 55.90625 0 \r\nL 45.3125 0 \r\nL 29.390625 21.484375 \r\nL 13.484375 0 \r\nL 2.875 0 \r\nL 24.125 28.609375 \r\nL 4.6875 54.6875 \r\nL 15.28125 54.6875 \r\nL 29.78125 35.203125 \r\nL 44.28125 54.6875 \r\nz\r\n\" id=\"DejaVuSans-120\"/>\r\n       <path d=\"M 9.421875 54.6875 \r\nL 18.40625 54.6875 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\nM 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 64.59375 \r\nL 9.421875 64.59375 \r\nz\r\n\" id=\"DejaVuSans-105\"/>\r\n       <path d=\"M 52 44.1875 \r\nQ 55.375 50.25 60.0625 53.125 \r\nQ 64.75 56 71.09375 56 \r\nQ 79.640625 56 84.28125 50.015625 \r\nQ 88.921875 44.046875 88.921875 33.015625 \r\nL 88.921875 0 \r\nL 79.890625 0 \r\nL 79.890625 32.71875 \r\nQ 79.890625 40.578125 77.09375 44.375 \r\nQ 74.3125 48.1875 68.609375 48.1875 \r\nQ 61.625 48.1875 57.5625 43.546875 \r\nQ 53.515625 38.921875 53.515625 30.90625 \r\nL 53.515625 0 \r\nL 44.484375 0 \r\nL 44.484375 32.71875 \r\nQ 44.484375 40.625 41.703125 44.40625 \r\nQ 38.921875 48.1875 33.109375 48.1875 \r\nQ 26.21875 48.1875 22.15625 43.53125 \r\nQ 18.109375 38.875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.1875 51.21875 25.484375 53.609375 \r\nQ 29.78125 56 35.6875 56 \r\nQ 41.65625 56 45.828125 52.96875 \r\nQ 50 49.953125 52 44.1875 \r\nz\r\n\" id=\"DejaVuSans-109\"/>\r\n       <path d=\"M 54.890625 33.015625 \r\nL 54.890625 0 \r\nL 45.90625 0 \r\nL 45.90625 32.71875 \r\nQ 45.90625 40.484375 42.875 44.328125 \r\nQ 39.84375 48.1875 33.796875 48.1875 \r\nQ 26.515625 48.1875 22.3125 43.546875 \r\nQ 18.109375 38.921875 18.109375 30.90625 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 54.6875 \r\nL 18.109375 54.6875 \r\nL 18.109375 46.1875 \r\nQ 21.34375 51.125 25.703125 53.5625 \r\nQ 30.078125 56 35.796875 56 \r\nQ 45.21875 56 50.046875 50.171875 \r\nQ 54.890625 44.34375 54.890625 33.015625 \r\nz\r\n\" id=\"DejaVuSans-110\"/>\r\n       <path d=\"M 18.3125 70.21875 \r\nL 18.3125 54.6875 \r\nL 36.8125 54.6875 \r\nL 36.8125 47.703125 \r\nL 18.3125 47.703125 \r\nL 18.3125 18.015625 \r\nQ 18.3125 11.328125 20.140625 9.421875 \r\nQ 21.96875 7.515625 27.59375 7.515625 \r\nL 36.8125 7.515625 \r\nL 36.8125 0 \r\nL 27.59375 0 \r\nQ 17.1875 0 13.234375 3.875 \r\nQ 9.28125 7.765625 9.28125 18.015625 \r\nL 9.28125 47.703125 \r\nL 2.6875 47.703125 \r\nL 2.6875 54.6875 \r\nL 9.28125 54.6875 \r\nL 9.28125 70.21875 \r\nz\r\n\" id=\"DejaVuSans-116\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-71\"/>\r\n      <use x=\"77.490234\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"116.353516\" xlink:href=\"#DejaVuSans-111\"/>\r\n      <use x=\"177.535156\" xlink:href=\"#DejaVuSans-117\"/>\r\n      <use x=\"240.914062\" xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"304.390625\" xlink:href=\"#DejaVuSans-115\"/>\r\n      <use x=\"356.490234\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"388.277344\" xlink:href=\"#DejaVuSans-111\"/>\r\n      <use x=\"449.458984\" xlink:href=\"#DejaVuSans-102\"/>\r\n      <use x=\"484.664062\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"516.451172\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"576.224609\" xlink:href=\"#DejaVuSans-120\"/>\r\n      <use x=\"635.404297\" xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"698.880859\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"760.404297\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"801.517578\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"829.300781\" xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"926.712891\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"988.236328\" xlink:href=\"#DejaVuSans-110\"/>\r\n      <use x=\"1051.615234\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"1090.824219\" xlink:href=\"#DejaVuSans-115\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"matplotlib.axis_2\">\r\n    <g id=\"ytick_1\">\r\n     <g id=\"line2d_7\">\r\n      <defs>\r\n       <path d=\"M 0 0 \r\nL -3.5 0 \r\n\" id=\"m4f17183b9e\" style=\"stroke:#000000;stroke-width:0.8;\"/>\r\n      </defs>\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m4f17183b9e\" y=\"240.700909\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_8\">\r\n      <!-- 0.0 -->\r\n      <g transform=\"translate(20.878125 244.500128)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 10.6875 12.40625 \r\nL 21 12.40625 \r\nL 21 0 \r\nL 10.6875 0 \r\nz\r\n\" id=\"DejaVuSans-46\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_2\">\r\n     <g id=\"line2d_8\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m4f17183b9e\" y=\"203.637274\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_9\">\r\n      <!-- 0.1 -->\r\n      <g transform=\"translate(20.878125 207.436493)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-49\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_3\">\r\n     <g id=\"line2d_9\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m4f17183b9e\" y=\"166.573639\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_10\">\r\n      <!-- 0.2 -->\r\n      <g transform=\"translate(20.878125 170.372858)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_4\">\r\n     <g id=\"line2d_10\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m4f17183b9e\" y=\"129.510004\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_11\">\r\n      <!-- 0.3 -->\r\n      <g transform=\"translate(20.878125 133.309223)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-51\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_5\">\r\n     <g id=\"line2d_11\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m4f17183b9e\" y=\"92.44637\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_12\">\r\n      <!-- 0.4 -->\r\n      <g transform=\"translate(20.878125 96.245588)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_6\">\r\n     <g id=\"line2d_12\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m4f17183b9e\" y=\"55.382735\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_13\">\r\n      <!-- 0.5 -->\r\n      <g transform=\"translate(20.878125 59.181953)scale(0.1 -0.1)\">\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-53\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"ytick_7\">\r\n     <g id=\"line2d_13\">\r\n      <g>\r\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m4f17183b9e\" y=\"18.3191\"/>\r\n      </g>\r\n     </g>\r\n     <g id=\"text_14\">\r\n      <!-- 0.6 -->\r\n      <g transform=\"translate(20.878125 22.118318)scale(0.1 -0.1)\">\r\n       <defs>\r\n        <path d=\"M 33.015625 40.375 \r\nQ 26.375 40.375 22.484375 35.828125 \r\nQ 18.609375 31.296875 18.609375 23.390625 \r\nQ 18.609375 15.53125 22.484375 10.953125 \r\nQ 26.375 6.390625 33.015625 6.390625 \r\nQ 39.65625 6.390625 43.53125 10.953125 \r\nQ 47.40625 15.53125 47.40625 23.390625 \r\nQ 47.40625 31.296875 43.53125 35.828125 \r\nQ 39.65625 40.375 33.015625 40.375 \r\nz\r\nM 52.59375 71.296875 \r\nL 52.59375 62.3125 \r\nQ 48.875 64.0625 45.09375 64.984375 \r\nQ 41.3125 65.921875 37.59375 65.921875 \r\nQ 27.828125 65.921875 22.671875 59.328125 \r\nQ 17.53125 52.734375 16.796875 39.40625 \r\nQ 19.671875 43.65625 24.015625 45.921875 \r\nQ 28.375 48.1875 33.59375 48.1875 \r\nQ 44.578125 48.1875 50.953125 41.515625 \r\nQ 57.328125 34.859375 57.328125 23.390625 \r\nQ 57.328125 12.15625 50.6875 5.359375 \r\nQ 44.046875 -1.421875 33.015625 -1.421875 \r\nQ 20.359375 -1.421875 13.671875 8.265625 \r\nQ 6.984375 17.96875 6.984375 36.375 \r\nQ 6.984375 53.65625 15.1875 63.9375 \r\nQ 23.390625 74.21875 37.203125 74.21875 \r\nQ 40.921875 74.21875 44.703125 73.484375 \r\nQ 48.484375 72.75 52.59375 71.296875 \r\nz\r\n\" id=\"DejaVuSans-54\"/>\r\n       </defs>\r\n       <use xlink:href=\"#DejaVuSans-48\"/>\r\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\r\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\r\n      </g>\r\n     </g>\r\n    </g>\r\n    <g id=\"text_15\">\r\n     <!-- Estimated probability -->\r\n     <g transform=\"translate(14.798438 183.033437)rotate(-90)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 9.8125 72.90625 \r\nL 55.90625 72.90625 \r\nL 55.90625 64.59375 \r\nL 19.671875 64.59375 \r\nL 19.671875 43.015625 \r\nL 54.390625 43.015625 \r\nL 54.390625 34.71875 \r\nL 19.671875 34.71875 \r\nL 19.671875 8.296875 \r\nL 56.78125 8.296875 \r\nL 56.78125 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-69\"/>\r\n       <path d=\"M 34.28125 27.484375 \r\nQ 23.390625 27.484375 19.1875 25 \r\nQ 14.984375 22.515625 14.984375 16.5 \r\nQ 14.984375 11.71875 18.140625 8.90625 \r\nQ 21.296875 6.109375 26.703125 6.109375 \r\nQ 34.1875 6.109375 38.703125 11.40625 \r\nQ 43.21875 16.703125 43.21875 25.484375 \r\nL 43.21875 27.484375 \r\nz\r\nM 52.203125 31.203125 \r\nL 52.203125 0 \r\nL 43.21875 0 \r\nL 43.21875 8.296875 \r\nQ 40.140625 3.328125 35.546875 0.953125 \r\nQ 30.953125 -1.421875 24.3125 -1.421875 \r\nQ 15.921875 -1.421875 10.953125 3.296875 \r\nQ 6 8.015625 6 15.921875 \r\nQ 6 25.140625 12.171875 29.828125 \r\nQ 18.359375 34.515625 30.609375 34.515625 \r\nL 43.21875 34.515625 \r\nL 43.21875 35.40625 \r\nQ 43.21875 41.609375 39.140625 45 \r\nQ 35.0625 48.390625 27.6875 48.390625 \r\nQ 23 48.390625 18.546875 47.265625 \r\nQ 14.109375 46.140625 10.015625 43.890625 \r\nL 10.015625 52.203125 \r\nQ 14.9375 54.109375 19.578125 55.046875 \r\nQ 24.21875 56 28.609375 56 \r\nQ 40.484375 56 46.34375 49.84375 \r\nQ 52.203125 43.703125 52.203125 31.203125 \r\nz\r\n\" id=\"DejaVuSans-97\"/>\r\n       <path d=\"M 45.40625 46.390625 \r\nL 45.40625 75.984375 \r\nL 54.390625 75.984375 \r\nL 54.390625 0 \r\nL 45.40625 0 \r\nL 45.40625 8.203125 \r\nQ 42.578125 3.328125 38.25 0.953125 \r\nQ 33.9375 -1.421875 27.875 -1.421875 \r\nQ 17.96875 -1.421875 11.734375 6.484375 \r\nQ 5.515625 14.40625 5.515625 27.296875 \r\nQ 5.515625 40.1875 11.734375 48.09375 \r\nQ 17.96875 56 27.875 56 \r\nQ 33.9375 56 38.25 53.625 \r\nQ 42.578125 51.265625 45.40625 46.390625 \r\nz\r\nM 14.796875 27.296875 \r\nQ 14.796875 17.390625 18.875 11.75 \r\nQ 22.953125 6.109375 30.078125 6.109375 \r\nQ 37.203125 6.109375 41.296875 11.75 \r\nQ 45.40625 17.390625 45.40625 27.296875 \r\nQ 45.40625 37.203125 41.296875 42.84375 \r\nQ 37.203125 48.484375 30.078125 48.484375 \r\nQ 22.953125 48.484375 18.875 42.84375 \r\nQ 14.796875 37.203125 14.796875 27.296875 \r\nz\r\n\" id=\"DejaVuSans-100\"/>\r\n       <path d=\"M 48.6875 27.296875 \r\nQ 48.6875 37.203125 44.609375 42.84375 \r\nQ 40.53125 48.484375 33.40625 48.484375 \r\nQ 26.265625 48.484375 22.1875 42.84375 \r\nQ 18.109375 37.203125 18.109375 27.296875 \r\nQ 18.109375 17.390625 22.1875 11.75 \r\nQ 26.265625 6.109375 33.40625 6.109375 \r\nQ 40.53125 6.109375 44.609375 11.75 \r\nQ 48.6875 17.390625 48.6875 27.296875 \r\nz\r\nM 18.109375 46.390625 \r\nQ 20.953125 51.265625 25.265625 53.625 \r\nQ 29.59375 56 35.59375 56 \r\nQ 45.5625 56 51.78125 48.09375 \r\nQ 58.015625 40.1875 58.015625 27.296875 \r\nQ 58.015625 14.40625 51.78125 6.484375 \r\nQ 45.5625 -1.421875 35.59375 -1.421875 \r\nQ 29.59375 -1.421875 25.265625 0.953125 \r\nQ 20.953125 3.328125 18.109375 8.203125 \r\nL 18.109375 0 \r\nL 9.078125 0 \r\nL 9.078125 75.984375 \r\nL 18.109375 75.984375 \r\nz\r\n\" id=\"DejaVuSans-98\"/>\r\n       <path d=\"M 9.421875 75.984375 \r\nL 18.40625 75.984375 \r\nL 18.40625 0 \r\nL 9.421875 0 \r\nz\r\n\" id=\"DejaVuSans-108\"/>\r\n       <path d=\"M 32.171875 -5.078125 \r\nQ 28.375 -14.84375 24.75 -17.8125 \r\nQ 21.140625 -20.796875 15.09375 -20.796875 \r\nL 7.90625 -20.796875 \r\nL 7.90625 -13.28125 \r\nL 13.1875 -13.28125 \r\nQ 16.890625 -13.28125 18.9375 -11.515625 \r\nQ 21 -9.765625 23.484375 -3.21875 \r\nL 25.09375 0.875 \r\nL 2.984375 54.6875 \r\nL 12.5 54.6875 \r\nL 29.59375 11.921875 \r\nL 46.6875 54.6875 \r\nL 56.203125 54.6875 \r\nz\r\n\" id=\"DejaVuSans-121\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-69\"/>\r\n      <use x=\"63.183594\" xlink:href=\"#DejaVuSans-115\"/>\r\n      <use x=\"115.283203\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"154.492188\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"182.275391\" xlink:href=\"#DejaVuSans-109\"/>\r\n      <use x=\"279.6875\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"340.966797\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"380.175781\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"441.699219\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"505.175781\" xlink:href=\"#DejaVuSans-32\"/>\r\n      <use x=\"536.962891\" xlink:href=\"#DejaVuSans-112\"/>\r\n      <use x=\"600.439453\" xlink:href=\"#DejaVuSans-114\"/>\r\n      <use x=\"639.302734\" xlink:href=\"#DejaVuSans-111\"/>\r\n      <use x=\"700.484375\" xlink:href=\"#DejaVuSans-98\"/>\r\n      <use x=\"763.960938\" xlink:href=\"#DejaVuSans-97\"/>\r\n      <use x=\"825.240234\" xlink:href=\"#DejaVuSans-98\"/>\r\n      <use x=\"888.716797\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"916.5\" xlink:href=\"#DejaVuSans-108\"/>\r\n      <use x=\"944.283203\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"972.066406\" xlink:href=\"#DejaVuSans-116\"/>\r\n      <use x=\"1011.275391\" xlink:href=\"#DejaVuSans-121\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n   <g id=\"line2d_14\">\r\n    <path clip-path=\"url(#p1b07345985)\" d=\"M 58.999432 203.637274 \r\nL 59.609379 185.105455 \r\nL 60.219326 191.282727 \r\nL 60.829273 185.105455 \r\nL 61.43922 188.81182 \r\nL 62.049168 178.928182 \r\nL 62.659115 182.458052 \r\nL 63.269062 175.839549 \r\nL 63.879009 166.573638 \r\nL 65.098903 166.573638 \r\nL 65.708851 169.662277 \r\nL 66.318798 163.72259 \r\nL 67.538692 173.986364 \r\nL 68.148639 175.839549 \r\nL 69.368534 170.69182 \r\nL 69.978481 170.475073 \r\nL 70.588428 172.133184 \r\nL 71.198375 170.103508 \r\nL 71.808322 169.943062 \r\nL 73.028217 172.75091 \r\nL 74.248111 169.424687 \r\nL 74.858058 172.06455 \r\nL 75.468005 173.192147 \r\nL 76.077952 172.963924 \r\nL 76.6879 173.986364 \r\nL 77.907794 178.156024 \r\nL 78.517741 177.805043 \r\nL 79.127688 179.65492 \r\nL 80.347583 176.869092 \r\nL 81.567477 178.277942 \r\nL 82.787371 177.692729 \r\nL 83.397319 176.517542 \r\nL 84.007266 176.28078 \r\nL 85.22716 179.208967 \r\nL 85.837107 179.75182 \r\nL 86.447054 178.659609 \r\nL 87.057002 178.402456 \r\nL 87.666949 178.928182 \r\nL 88.276896 180.188853 \r\nL 88.886843 180.65782 \r\nL 89.49679 181.835134 \r\nL 90.106737 180.828882 \r\nL 91.326632 180.300913 \r\nL 91.936579 178.703555 \r\nL 92.546526 179.148801 \r\nL 93.156473 178.928182 \r\nL 93.76642 179.993232 \r\nL 94.376368 179.765782 \r\nL 94.986315 178.928182 \r\nL 96.206209 179.725249 \r\nL 96.816156 179.516494 \r\nL 98.036051 181.399095 \r\nL 99.255945 179.850166 \r\nL 101.085786 179.281174 \r\nL 101.695734 179.624213 \r\nL 102.305681 180.472505 \r\nL 102.915628 179.774386 \r\nL 103.525575 180.59772 \r\nL 104.135522 180.90491 \r\nL 105.355417 179.569978 \r\nL 105.965364 179.878532 \r\nL 107.185258 179.545909 \r\nL 107.795205 179.843334 \r\nL 108.405152 179.681508 \r\nL 109.625047 178.486951 \r\nL 110.234994 178.346797 \r\nL 110.844941 177.778925 \r\nL 111.454888 178.502167 \r\nL 112.674783 179.067001 \r\nL 113.28473 179.75182 \r\nL 113.894677 179.607004 \r\nL 114.504624 179.868204 \r\nL 115.114571 178.928182 \r\nL 115.724518 179.585343 \r\nL 116.944413 180.086425 \r\nL 117.55436 180.711315 \r\nL 118.164307 179.432452 \r\nL 119.994149 180.151408 \r\nL 120.604096 179.65492 \r\nL 121.214043 178.808236 \r\nL 121.82399 179.403357 \r\nL 123.043884 178.461977 \r\nL 123.653832 179.04365 \r\nL 124.873726 179.494905 \r\nL 126.703567 179.148801 \r\nL 127.313515 178.381525 \r\nL 127.923462 178.928182 \r\nL 128.533409 178.498461 \r\nL 129.143356 178.715175 \r\nL 130.36325 178.509386 \r\nL 130.973198 179.032002 \r\nL 132.193092 179.438704 \r\nL 132.803039 178.725652 \r\nL 133.412986 178.928182 \r\nL 134.022933 178.529649 \r\nL 134.632881 178.434003 \r\nL 135.242828 178.928182 \r\nL 136.462722 179.314262 \r\nL 137.072669 178.928182 \r\nL 138.292564 178.739564 \r\nL 138.902511 179.208967 \r\nL 140.732352 179.75182 \r\nL 141.342299 179.10987 \r\nL 142.562194 180.002494 \r\nL 143.782088 179.81065 \r\nL 144.392035 179.453908 \r\nL 145.001982 179.363206 \r\nL 145.61193 179.014577 \r\nL 146.221877 178.928182 \r\nL 146.831824 179.354203 \r\nL 147.441771 179.266665 \r\nL 148.051718 178.423918 \r\nL 148.661665 178.093416 \r\nL 149.88156 178.434003 \r\nL 151.101454 178.277942 \r\nL 151.711401 178.685936 \r\nL 152.321348 178.847962 \r\nL 152.931296 178.290529 \r\nL 153.541243 178.453008 \r\nL 155.371084 178.228871 \r\nL 155.981031 178.619319 \r\nL 158.42082 178.325522 \r\nL 159.030767 178.703555 \r\nL 159.640714 177.960662 \r\nL 160.250662 178.336347 \r\nL 160.860609 178.266333 \r\nL 161.470556 178.635766 \r\nL 163.910345 178.356876 \r\nL 166.350133 178.928182 \r\nL 166.96008 179.27522 \r\nL 167.570028 179.204262 \r\nL 168.179975 179.340004 \r\nL 168.789922 179.679011 \r\nL 170.009816 179.940853 \r\nL 170.619763 179.666772 \r\nL 171.229711 179.595996 \r\nL 171.839658 179.924521 \r\nL 173.059552 180.17678 \r\nL 173.669499 180.104806 \r\nL 174.889394 180.351221 \r\nL 176.109288 180.208449 \r\nL 176.719235 180.329213 \r\nL 177.329182 180.638816 \r\nL 177.939129 180.75615 \r\nL 178.549077 181.060441 \r\nL 179.768971 180.914846 \r\nL 180.378918 181.213773 \r\nL 180.988865 181.140937 \r\nL 182.20876 180.63226 \r\nL 184.038601 181.507045 \r\nL 184.648548 181.613952 \r\nL 185.258495 181.363455 \r\nL 185.868443 181.29269 \r\nL 187.088337 181.504488 \r\nL 187.698284 181.084405 \r\nL 188.918178 181.641567 \r\nL 189.528126 181.571481 \r\nL 190.138073 181.673639 \r\nL 190.74802 181.433254 \r\nL 191.967914 181.636028 \r\nL 192.577861 181.399095 \r\nL 193.187809 181.332008 \r\nL 193.797756 180.764672 \r\nL 194.407703 180.867238 \r\nL 195.01765 180.803426 \r\nL 195.627597 180.575457 \r\nL 196.237544 180.841501 \r\nL 196.847492 180.288818 \r\nL 197.457439 180.066101 \r\nL 198.067386 180.330886 \r\nL 199.28728 180.532671 \r\nL 199.897227 180.152988 \r\nL 200.507175 180.253781 \r\nL 201.117122 180.195315 \r\nL 201.727069 180.452788 \r\nL 202.337016 180.39398 \r\nL 202.946963 180.648437 \r\nL 203.55691 180.745027 \r\nL 204.166858 180.530655 \r\nL 205.996699 180.357634 \r\nL 206.606646 180.453434 \r\nL 207.216593 180.244651 \r\nL 207.826541 180.188853 \r\nL 208.436488 180.434838 \r\nL 209.046435 180.37872 \r\nL 209.656382 180.472505 \r\nL 210.876276 180.954329 \r\nL 211.486224 180.897034 \r\nL 212.096171 180.398962 \r\nL 212.706118 180.344318 \r\nL 214.535959 180.617282 \r\nL 215.145907 180.418419 \r\nL 215.755854 180.364758 \r\nL 216.365801 180.454616 \r\nL 216.975748 180.401226 \r\nL 217.585695 180.490255 \r\nL 218.195642 180.437136 \r\nL 218.80559 180.243497 \r\nL 219.415537 180.332112 \r\nL 220.635431 179.949987 \r\nL 222.465273 179.800808 \r\nL 223.07522 179.889092 \r\nL 223.685167 179.83996 \r\nL 224.295114 179.927448 \r\nL 224.905061 180.15006 \r\nL 225.515008 180.235782 \r\nL 226.734903 179.868204 \r\nL 227.954797 179.772558 \r\nL 228.564744 179.99094 \r\nL 229.174691 179.81065 \r\nL 229.784639 179.763545 \r\nL 230.394586 179.979634 \r\nL 231.004533 179.932264 \r\nL 233.444322 180.262645 \r\nL 234.664216 180.167916 \r\nL 235.274163 179.993232 \r\nL 235.88411 179.565018 \r\nL 236.494057 179.647453 \r\nL 237.104005 179.602834 \r\nL 237.713952 179.432452 \r\nL 238.323899 179.5145 \r\nL 238.933846 179.470781 \r\nL 240.763688 180.085133 \r\nL 241.373635 180.163641 \r\nL 241.983582 180.364758 \r\nL 242.593529 180.319095 \r\nL 244.423371 180.54845 \r\nL 245.033318 180.018289 \r\nL 245.643265 180.095223 \r\nL 246.253212 180.051327 \r\nL 246.863159 180.247601 \r\nL 247.473106 180.083935 \r\nL 248.083054 180.159665 \r\nL 248.693001 180.353707 \r\nL 249.912895 180.502014 \r\nL 250.522842 180.693117 \r\nL 251.132789 180.765727 \r\nL 251.742737 180.604038 \r\nL 252.962631 180.516075 \r\nL 253.572578 180.356678 \r\nL 254.182525 180.313743 \r\nL 254.792472 180.386176 \r\nL 255.40242 180.228663 \r\nL 256.012367 180.300913 \r\nL 256.622314 180.258674 \r\nL 257.842208 179.948286 \r\nL 259.062103 180.092291 \r\nL 259.67205 179.826694 \r\nL 260.281997 180.010607 \r\nL 261.501891 179.818603 \r\nL 262.721786 179.850166 \r\nL 263.94168 180.101315 \r\nL 266.991416 180.011916 \r\nL 267.601363 180.080798 \r\nL 268.21131 179.933789 \r\nL 268.821257 180.109925 \r\nL 270.651099 179.993232 \r\nL 271.261046 179.636181 \r\nL 273.090887 179.419556 \r\nL 274.310782 179.242281 \r\nL 276.140623 179.032002 \r\nL 277.360518 178.756116 \r\nL 277.970465 178.413414 \r\nL 280.410253 178.588772 \r\nL 282.850042 178.156024 \r\nL 284.069936 178.193591 \r\nL 284.679884 178.062367 \r\nL 285.289831 177.732582 \r\nL 287.119672 177.544478 \r\nL 287.729619 177.318149 \r\nL 288.339567 177.387964 \r\nL 289.559461 177.135307 \r\nL 291.389302 177.246412 \r\nL 293.219144 177.259517 \r\nL 294.439038 177.300067 \r\nL 295.658933 177.244915 \r\nL 296.878827 177.569502 \r\nL 301.148457 177.376106 \r\nL 301.758404 177.163248 \r\nL 304.198193 177.150109 \r\nL 307.857876 177.448053 \r\nL 310.297665 177.252989 \r\nL 310.907612 177.048145 \r\nL 312.127506 177.086882 \r\nL 312.737453 177.15055 \r\nL 313.3474 176.947909 \r\nL 313.957348 176.92315 \r\nL 314.567295 177.075002 \r\nL 315.177242 177.050061 \r\nL 316.397136 176.562421 \r\nL 317.007083 176.364032 \r\nL 317.617031 176.428203 \r\nL 318.836925 176.29525 \r\nL 320.056819 176.336324 \r\nL 321.886661 176.525909 \r\nL 322.496608 176.331734 \r\nL 324.936397 176.666484 \r\nL 326.156291 176.873786 \r\nL 326.766238 176.93461 \r\nL 329.206027 176.590836 \r\nL 329.815974 176.734907 \r\nL 333.475657 176.599723 \r\nL 334.085604 176.41354 \r\nL 334.695551 176.47364 \r\nL 335.915446 176.348662 \r\nL 336.525393 176.489789 \r\nL 338.355234 176.424978 \r\nL 339.575129 176.623433 \r\nL 340.185076 176.601678 \r\nL 341.40497 176.238939 \r\nL 342.624865 176.356532 \r\nL 343.844759 176.235525 \r\nL 345.6746 176.567441 \r\nL 346.894495 176.525186 \r\nL 347.504442 176.66058 \r\nL 351.164125 176.611707 \r\nL 352.993966 176.70283 \r\nL 354.213861 176.737481 \r\nL 354.823808 176.487779 \r\nL 356.653649 176.502751 \r\nL 358.483491 176.818872 \r\nL 360.313332 176.60703 \r\nL 361.533227 176.641227 \r\nL 362.143174 176.47216 \r\nL 363.363068 176.729075 \r\nL 363.363068 176.729075 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_15\">\r\n    <path clip-path=\"url(#p1b07345985)\" d=\"M 58.999432 240.700909 \r\nL 59.609379 222.169091 \r\nL 60.219326 215.991818 \r\nL 60.829273 194.371365 \r\nL 61.43922 188.81182 \r\nL 62.049168 197.460002 \r\nL 62.659115 198.342469 \r\nL 63.269062 199.004321 \r\nL 63.879009 203.637274 \r\nL 64.488956 199.930911 \r\nL 65.098903 203.637274 \r\nL 65.708851 203.637274 \r\nL 66.318798 200.786225 \r\nL 67.538692 191.282727 \r\nL 68.148639 194.371365 \r\nL 69.368534 191.282727 \r\nL 69.978481 193.883687 \r\nL 70.588428 188.81182 \r\nL 71.198375 187.752857 \r\nL 71.808322 190.159587 \r\nL 73.028217 191.282727 \r\nL 74.248111 186.530979 \r\nL 74.858058 185.79182 \r\nL 75.468005 183.781756 \r\nL 76.077952 183.188374 \r\nL 76.6879 183.870001 \r\nL 77.297847 183.312055 \r\nL 77.907794 183.947217 \r\nL 78.517741 183.420745 \r\nL 79.127688 181.835134 \r\nL 79.737635 183.517015 \r\nL 80.347583 184.075912 \r\nL 80.95753 183.602875 \r\nL 81.567477 182.179383 \r\nL 82.177424 181.779231 \r\nL 82.787371 182.325685 \r\nL 83.397319 181.037498 \r\nL 84.007266 181.575585 \r\nL 85.22716 184.2631 \r\nL 87.057002 185.499752 \r\nL 87.666949 182.78898 \r\nL 88.276896 181.701656 \r\nL 88.886843 182.140363 \r\nL 89.49679 180.381658 \r\nL 90.716685 182.65786 \r\nL 91.326632 180.987273 \r\nL 93.76642 182.549346 \r\nL 94.376368 182.278569 \r\nL 95.596262 182.978857 \r\nL 96.206209 181.518655 \r\nL 96.816156 180.693117 \r\nL 97.426103 179.314262 \r\nL 98.036051 178.548046 \r\nL 98.645998 178.366613 \r\nL 99.255945 178.743789 \r\nL 99.865892 179.65492 \r\nL 100.475839 179.465341 \r\nL 101.085786 180.340131 \r\nL 101.695734 179.624213 \r\nL 102.915628 181.29755 \r\nL 103.525575 181.098576 \r\nL 104.135522 179.916546 \r\nL 105.355417 180.532671 \r\nL 105.965364 180.353707 \r\nL 106.575311 179.710116 \r\nL 107.795205 179.385761 \r\nL 108.405152 180.133508 \r\nL 109.625047 179.81065 \r\nL 110.234994 178.346797 \r\nL 110.844941 178.209895 \r\nL 111.454888 177.22411 \r\nL 112.674783 177.817663 \r\nL 114.504624 177.451008 \r\nL 118.164307 179.054248 \r\nL 118.774254 178.553806 \r\nL 119.384201 178.80464 \r\nL 119.994149 178.68354 \r\nL 120.604096 179.291551 \r\nL 121.214043 179.527921 \r\nL 121.82399 179.04698 \r\nL 122.433937 179.281174 \r\nL 123.043884 179.860605 \r\nL 123.653832 179.04365 \r\nL 124.263779 178.928182 \r\nL 124.873726 179.154876 \r\nL 126.703567 178.817873 \r\nL 127.313515 178.053524 \r\nL 127.923462 176.977468 \r\nL 128.533409 177.531582 \r\nL 129.143356 177.437118 \r\nL 130.36325 177.881187 \r\nL 130.973198 176.851788 \r\nL 132.193092 176.06928 \r\nL 132.803039 176.29525 \r\nL 133.412986 176.216212 \r\nL 134.022933 176.437349 \r\nL 134.632881 176.951455 \r\nL 135.242828 176.574936 \r\nL 137.072669 177.204294 \r\nL 138.292564 176.476137 \r\nL 140.122405 176.254441 \r\nL 140.732352 175.359094 \r\nL 141.342299 175.294493 \r\nL 141.952247 175.501376 \r\nL 143.172141 175.372918 \r\nL 143.782088 175.839549 \r\nL 145.001982 176.231063 \r\nL 145.61193 175.904344 \r\nL 146.221877 175.839549 \r\nL 146.831824 176.031255 \r\nL 147.441771 175.966477 \r\nL 148.661665 176.340405 \r\nL 149.88156 176.210186 \r\nL 150.491507 176.391818 \r\nL 151.101454 176.083386 \r\nL 151.711401 176.021231 \r\nL 152.321348 176.441232 \r\nL 152.931296 176.616689 \r\nL 154.15119 176.488751 \r\nL 154.761137 176.66058 \r\nL 155.371084 177.063349 \r\nL 156.590979 177.393459 \r\nL 157.200926 177.326671 \r\nL 157.810873 177.715467 \r\nL 158.42082 177.421532 \r\nL 159.640714 177.290832 \r\nL 160.860609 176.722017 \r\nL 161.470556 175.784718 \r\nL 163.300397 176.270519 \r\nL 163.910345 176.000227 \r\nL 164.520292 176.159061 \r\nL 165.130239 176.10429 \r\nL 165.740186 175.418372 \r\nL 167.570028 175.891315 \r\nL 168.179975 175.427728 \r\nL 168.789922 175.378811 \r\nL 169.399869 175.126784 \r\nL 170.009816 175.485116 \r\nL 170.619763 175.638117 \r\nL 171.229711 175.989805 \r\nL 172.449605 175.492694 \r\nL 173.059552 175.839549 \r\nL 173.669499 175.986624 \r\nL 174.279446 176.327228 \r\nL 174.889394 176.47021 \r\nL 175.499341 176.804744 \r\nL 176.109288 176.751735 \r\nL 176.719235 176.508219 \r\nL 177.939129 176.406846 \r\nL 178.549077 176.168792 \r\nL 179.159024 176.120334 \r\nL 179.768971 176.44486 \r\nL 180.378918 176.580823 \r\nL 180.988865 176.899827 \r\nL 182.818707 177.293025 \r\nL 183.428654 177.240734 \r\nL 184.038601 177.548791 \r\nL 184.648548 177.495771 \r\nL 185.868443 178.100607 \r\nL 186.47839 178.222211 \r\nL 187.088337 178.167004 \r\nL 187.698284 178.287149 \r\nL 188.308231 178.058147 \r\nL 188.918178 178.004481 \r\nL 189.528126 177.778925 \r\nL 191.357967 177.624725 \r\nL 191.967914 177.405019 \r\nL 192.577861 177.355787 \r\nL 195.627597 177.939819 \r\nL 196.237544 177.889527 \r\nL 197.457439 178.115387 \r\nL 198.067386 178.388682 \r\nL 198.677333 178.498461 \r\nL 199.28728 178.767736 \r\nL 199.897227 178.874931 \r\nL 200.507175 178.663066 \r\nL 201.117122 178.611399 \r\nL 201.727069 178.717892 \r\nL 202.337016 178.666435 \r\nL 202.946963 178.459022 \r\nL 203.55691 178.564813 \r\nL 204.166858 178.824799 \r\nL 205.386752 179.03071 \r\nL 205.996699 178.82608 \r\nL 207.216593 179.029451 \r\nL 207.826541 178.978612 \r\nL 208.436488 179.078847 \r\nL 209.656382 178.977999 \r\nL 210.266329 178.779334 \r\nL 210.876276 178.730512 \r\nL 211.486224 178.239089 \r\nL 212.096171 178.192796 \r\nL 212.706118 178.439863 \r\nL 213.316065 178.393145 \r\nL 213.926012 178.637489 \r\nL 214.535959 178.590362 \r\nL 215.145907 178.83204 \r\nL 215.755854 178.928182 \r\nL 216.975748 178.548046 \r\nL 218.195642 178.739564 \r\nL 219.415537 178.366613 \r\nL 220.635431 178.556617 \r\nL 221.245378 178.372925 \r\nL 221.855325 178.328897 \r\nL 222.465273 178.147414 \r\nL 223.07522 177.830001 \r\nL 223.685167 178.061997 \r\nL 224.295114 178.156024 \r\nL 224.905061 177.977833 \r\nL 228.564744 177.732582 \r\nL 229.174691 177.95747 \r\nL 229.784639 178.048858 \r\nL 230.394586 177.745302 \r\nL 231.004533 177.836794 \r\nL 231.61448 177.666628 \r\nL 232.224427 177.367612 \r\nL 233.444322 177.034105 \r\nL 234.054269 177.126482 \r\nL 234.664216 176.961716 \r\nL 235.88411 177.399788 \r\nL 236.494057 177.108852 \r\nL 238.323899 177.001713 \r\nL 238.933846 176.716052 \r\nL 240.15374 176.647982 \r\nL 241.373635 176.827913 \r\nL 241.983582 176.793842 \r\nL 242.593529 176.882728 \r\nL 243.203476 176.726385 \r\nL 243.813423 176.936825 \r\nL 245.643265 177.197744 \r\nL 246.863159 177.128978 \r\nL 248.083054 176.822749 \r\nL 248.693001 176.552309 \r\nL 249.302948 176.52043 \r\nL 249.912895 176.606787 \r\nL 250.522842 176.457276 \r\nL 251.132789 176.426 \r\nL 251.742737 176.511837 \r\nL 252.352684 176.480588 \r\nL 254.792472 176.933042 \r\nL 255.40242 176.78622 \r\nL 256.012367 176.754701 \r\nL 256.622314 176.951455 \r\nL 257.232261 176.805932 \r\nL 257.842208 177.001327 \r\nL 259.062103 176.599971 \r\nL 259.67205 176.794218 \r\nL 260.281997 176.651367 \r\nL 262.111838 177.11569 \r\nL 262.721786 177.194861 \r\nL 264.551627 176.771623 \r\nL 265.771521 176.820645 \r\nL 266.381469 176.464522 \r\nL 267.601363 176.514907 \r\nL 268.821257 176.349844 \r\nL 269.431204 176.10735 \r\nL 270.651099 176.372074 \r\nL 271.261046 176.556395 \r\nL 274.310782 176.520093 \r\nL 275.530676 176.568325 \r\nL 277.360518 176.588047 \r\nL 279.190359 176.709828 \r\nL 279.800306 176.27349 \r\nL 281.020201 176.423426 \r\nL 281.630148 176.193982 \r\nL 282.850042 176.443844 \r\nL 284.069936 176.290324 \r\nL 285.289831 176.636616 \r\nL 290.779355 176.496185 \r\nL 291.389302 176.664263 \r\nL 292.609197 176.515189 \r\nL 295.048985 176.890322 \r\nL 295.658933 176.673244 \r\nL 296.26888 176.74238 \r\nL 297.488774 176.406846 \r\nL 298.098721 176.476137 \r\nL 298.708668 176.074726 \r\nL 299.318616 175.956839 \r\nL 305.418087 176.54879 \r\nL 307.857876 176.360614 \r\nL 308.467823 176.155946 \r\nL 309.07777 176.31299 \r\nL 310.297665 176.355566 \r\nL 310.907612 176.421465 \r\nL 312.127506 176.285022 \r\nL 313.957348 176.303949 \r\nL 315.177242 176.433798 \r\nL 315.787189 176.322605 \r\nL 317.007083 176.53886 \r\nL 320.056819 176.422719 \r\nL 321.886661 176.783299 \r\nL 325.546344 176.728059 \r\nL 327.376185 176.827073 \r\nL 327.986132 176.720017 \r\nL 328.59608 176.864447 \r\nL 329.815974 176.818193 \r\nL 330.425921 176.961429 \r\nL 333.475657 176.846266 \r\nL 334.695551 177.046366 \r\nL 335.915446 177.08179 \r\nL 337.13534 176.954702 \r\nL 339.575129 177.10582 \r\nL 340.795023 177.140377 \r\nL 341.40497 177.197484 \r\nL 342.014917 176.935516 \r\nL 343.234812 176.811775 \r\nL 348.114389 177.185544 \r\nL 349.334283 177.141051 \r\nL 349.944231 176.886323 \r\nL 351.164125 176.843355 \r\nL 351.774072 176.667892 \r\nL 352.993966 176.856306 \r\nL 354.213861 176.890322 \r\nL 355.433755 176.771844 \r\nL 356.653649 177.033316 \r\nL 357.263597 176.78505 \r\nL 357.873544 176.76425 \r\nL 358.483491 176.442209 \r\nL 360.313332 176.681904 \r\nL 361.533227 176.4175 \r\nL 363.363068 176.358438 \r\nL 363.363068 176.358438 \r\n\" style=\"fill:none;stroke:#ff7f0e;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_16\">\r\n    <path clip-path=\"url(#p1b07345985)\" d=\"M 58.999432 18.319091 \r\nL 59.609379 92.446367 \r\nL 60.219326 117.155456 \r\nL 60.829273 129.51 \r\nL 61.43922 136.922731 \r\nL 62.049168 135.687278 \r\nL 62.659115 150.689224 \r\nL 63.269062 157.307733 \r\nL 63.879009 162.455457 \r\nL 64.488956 170.280004 \r\nL 65.708851 175.839549 \r\nL 66.928745 174.515845 \r\nL 67.538692 176.457276 \r\nL 68.758586 175.294493 \r\nL 69.368534 178.928182 \r\nL 69.978481 176.327228 \r\nL 71.198375 178.928182 \r\nL 71.808322 180.051327 \r\nL 72.418269 182.688264 \r\nL 73.028217 182.016821 \r\nL 73.638164 179.916546 \r\nL 74.248111 179.403357 \r\nL 74.858058 176.182732 \r\nL 75.468005 177.163248 \r\nL 76.077952 175.520032 \r\nL 76.6879 172.75091 \r\nL 77.297847 173.747244 \r\nL 77.907794 173.523069 \r\nL 78.517741 174.435625 \r\nL 79.127688 174.204386 \r\nL 79.737635 175.045327 \r\nL 80.347583 174.810001 \r\nL 80.95753 175.589118 \r\nL 82.177424 175.126784 \r\nL 83.397319 176.517542 \r\nL 84.007266 176.28078 \r\nL 84.617213 176.916981 \r\nL 85.22716 174.15484 \r\nL 85.837107 173.986364 \r\nL 87.666949 175.839549 \r\nL 88.276896 174.894049 \r\nL 89.49679 174.567755 \r\nL 90.106737 175.839549 \r\nL 90.716685 176.364032 \r\nL 91.326632 175.496366 \r\nL 91.936579 176.00802 \r\nL 92.546526 175.839549 \r\nL 93.76642 174.241975 \r\nL 94.986315 173.986364 \r\nL 95.596262 173.257245 \r\nL 96.206209 174.345049 \r\nL 97.426103 174.10219 \r\nL 98.036051 174.556577 \r\nL 98.645998 175.558764 \r\nL 99.255945 175.424657 \r\nL 100.475839 176.242412 \r\nL 102.305681 175.839549 \r\nL 102.915628 176.220337 \r\nL 103.525575 175.589118 \r\nL 104.135522 175.468912 \r\nL 104.745469 175.839549 \r\nL 105.355417 176.681904 \r\nL 105.965364 176.552309 \r\nL 106.575311 177.36432 \r\nL 107.185258 177.692729 \r\nL 107.795205 177.097884 \r\nL 108.405152 177.873527 \r\nL 109.0151 177.737387 \r\nL 110.844941 178.64087 \r\nL 111.454888 178.502167 \r\nL 112.674783 177.401219 \r\nL 113.28473 178.104551 \r\nL 113.894677 177.57054 \r\nL 114.504624 177.853877 \r\nL 115.724518 176.825284 \r\nL 116.334466 176.717372 \r\nL 116.944413 176.225628 \r\nL 117.55436 176.508219 \r\nL 118.164307 177.163248 \r\nL 118.774254 177.430661 \r\nL 119.384201 176.951455 \r\nL 121.82399 176.552309 \r\nL 123.043884 175.664721 \r\nL 123.653832 176.272534 \r\nL 124.263779 176.182732 \r\nL 126.09362 176.924746 \r\nL 126.703567 177.494175 \r\nL 127.923462 177.952825 \r\nL 128.533409 177.853877 \r\nL 129.143356 178.076146 \r\nL 129.753303 177.977833 \r\nL 130.36325 177.252989 \r\nL 130.973198 177.163248 \r\nL 131.583145 176.766139 \r\nL 133.412986 176.517542 \r\nL 134.022933 177.035149 \r\nL 134.632881 176.951455 \r\nL 135.242828 176.574936 \r\nL 137.072669 177.204294 \r\nL 137.682616 177.692729 \r\nL 138.292564 177.607853 \r\nL 138.902511 177.243473 \r\nL 139.512458 176.605903 \r\nL 140.122405 177.08422 \r\nL 141.342299 177.474707 \r\nL 143.172141 178.839302 \r\nL 144.392035 178.665319 \r\nL 145.001982 177.536121 \r\nL 145.61193 177.459463 \r\nL 146.221877 177.64125 \r\nL 146.831824 177.309312 \r\nL 148.661665 177.842985 \r\nL 149.271613 177.269856 \r\nL 149.88156 177.19855 \r\nL 150.491507 177.373637 \r\nL 151.101454 177.058749 \r\nL 151.711401 176.505723 \r\nL 152.931296 176.377569 \r\nL 154.761137 175.487679 \r\nL 155.371084 175.431616 \r\nL 155.981031 175.144607 \r\nL 156.590979 175.551789 \r\nL 157.810873 175.441623 \r\nL 158.42082 175.613552 \r\nL 159.030767 175.334136 \r\nL 159.640714 175.727909 \r\nL 160.250662 175.673094 \r\nL 161.470556 176.004033 \r\nL 162.080503 175.730538 \r\nL 162.69045 175.243494 \r\nL 163.910345 175.571749 \r\nL 164.520292 175.520032 \r\nL 165.130239 175.680705 \r\nL 165.740186 176.050138 \r\nL 166.96008 176.360106 \r\nL 167.570028 176.305434 \r\nL 168.179975 176.663187 \r\nL 168.789922 176.607444 \r\nL 170.009816 176.902848 \r\nL 170.619763 177.249576 \r\nL 171.839658 177.533316 \r\nL 172.449605 177.276506 \r\nL 173.669499 177.163248 \r\nL 174.279446 177.49766 \r\nL 174.889394 177.440465 \r\nL 175.499341 177.190829 \r\nL 176.719235 177.463468 \r\nL 177.329182 177.787767 \r\nL 177.939129 177.541451 \r\nL 179.159024 176.681904 \r\nL 179.768971 176.631109 \r\nL 180.378918 176.766139 \r\nL 180.988865 176.715428 \r\nL 181.598812 176.848707 \r\nL 182.20876 176.798089 \r\nL 184.648548 177.316724 \r\nL 185.258495 177.086882 \r\nL 185.868443 177.03658 \r\nL 186.47839 177.339743 \r\nL 187.088337 176.937405 \r\nL 188.308231 176.492076 \r\nL 188.918178 176.445727 \r\nL 190.138073 176.6975 \r\nL 190.74802 176.480047 \r\nL 191.357967 176.434604 \r\nL 191.967914 176.55882 \r\nL 192.577861 176.176491 \r\nL 193.187809 175.965328 \r\nL 193.797756 176.256932 \r\nL 194.407703 176.213511 \r\nL 195.627597 176.457276 \r\nL 196.237544 176.249543 \r\nL 196.847492 176.370196 \r\nL 198.677333 176.242412 \r\nL 200.507175 176.595139 \r\nL 201.727069 176.194413 \r\nL 203.55691 176.073141 \r\nL 204.166858 176.188476 \r\nL 205.386752 175.801099 \r\nL 205.996699 175.762968 \r\nL 206.606646 175.420106 \r\nL 207.216593 175.687647 \r\nL 207.826541 175.65045 \r\nL 208.436488 175.312222 \r\nL 209.046435 175.576951 \r\nL 209.656382 175.690099 \r\nL 210.266329 175.504634 \r\nL 210.876276 175.468912 \r\nL 212.096171 175.692468 \r\nL 212.706118 175.949422 \r\nL 213.316065 176.058428 \r\nL 213.926012 176.021231 \r\nL 214.535959 176.12911 \r\nL 215.145907 176.091924 \r\nL 215.755854 175.767718 \r\nL 216.365801 175.732222 \r\nL 216.975748 175.982101 \r\nL 217.585695 175.946053 \r\nL 218.80559 176.438487 \r\nL 221.245378 176.290699 \r\nL 221.855325 176.392735 \r\nL 222.465273 176.356234 \r\nL 223.685167 176.831105 \r\nL 224.295114 176.929656 \r\nL 224.905061 176.891719 \r\nL 225.515008 176.989331 \r\nL 226.734903 176.913855 \r\nL 227.34485 177.010329 \r\nL 227.954797 177.239436 \r\nL 228.564744 177.201207 \r\nL 229.174691 177.295621 \r\nL 229.784639 176.993667 \r\nL 231.61448 177.275109 \r\nL 232.224427 177.49766 \r\nL 232.834374 177.459463 \r\nL 233.444322 177.550674 \r\nL 234.054269 177.383866 \r\nL 234.664216 177.474707 \r\nL 235.274163 177.309312 \r\nL 238.323899 178.383745 \r\nL 238.933846 178.469063 \r\nL 239.543793 178.304215 \r\nL 240.15374 178.389229 \r\nL 241.373635 178.310456 \r\nL 241.983582 178.148331 \r\nL 242.593529 178.110002 \r\nL 243.203476 177.949605 \r\nL 243.813423 178.034106 \r\nL 244.423371 177.996528 \r\nL 245.033318 178.080322 \r\nL 245.643265 178.284299 \r\nL 246.863159 177.96861 \r\nL 247.473106 177.931849 \r\nL 248.693001 178.096631 \r\nL 249.302948 178.296643 \r\nL 249.912895 178.259308 \r\nL 250.522842 178.457536 \r\nL 251.742737 178.616397 \r\nL 252.352684 178.578527 \r\nL 252.962631 178.657079 \r\nL 254.182525 178.581796 \r\nL 254.792472 178.659609 \r\nL 256.012367 178.585 \r\nL 259.67205 179.040496 \r\nL 260.891944 178.853761 \r\nL 262.111838 178.780224 \r\nL 262.721786 178.522514 \r\nL 264.551627 178.855081 \r\nL 265.161574 178.709519 \r\nL 266.381469 178.747031 \r\nL 267.601363 178.784106 \r\nL 268.21131 178.964098 \r\nL 268.821257 178.928182 \r\nL 270.041152 179.177409 \r\nL 271.870993 179.069381 \r\nL 272.48094 179.244966 \r\nL 273.700835 179.278169 \r\nL 274.920729 179.624213 \r\nL 275.530676 179.483445 \r\nL 276.75057 179.82544 \r\nL 277.360518 179.582046 \r\nL 278.580412 179.817984 \r\nL 284.069936 179.595996 \r\nL 285.289831 179.725249 \r\nL 286.509725 179.555823 \r\nL 287.729619 179.881061 \r\nL 289.559461 179.873517 \r\nL 291.99925 180.024933 \r\nL 292.609197 179.796865 \r\nL 293.829091 179.92039 \r\nL 295.048985 179.565018 \r\nL 296.26888 179.783499 \r\nL 299.928563 180.051327 \r\nL 300.53851 179.7373 \r\nL 301.148457 179.611102 \r\nL 306.028034 179.993232 \r\nL 307.857876 179.985417 \r\nL 314.567295 180.516627 \r\nL 315.177242 180.483506 \r\nL 316.397136 180.680602 \r\nL 320.056819 181.001677 \r\nL 320.666766 180.881924 \r\nL 322.496608 180.953987 \r\nL 323.106555 180.835454 \r\nL 323.716502 180.887872 \r\nL 324.936397 180.737544 \r\nL 327.376185 180.861207 \r\nL 329.815974 180.816068 \r\nL 330.425921 180.950341 \r\nL 332.86571 180.90491 \r\nL 335.305498 181.186833 \r\nL 336.525393 181.366581 \r\nL 352.384019 182.567903 \r\nL 353.603914 182.578391 \r\nL 354.823808 182.741317 \r\nL 356.043702 182.902906 \r\nL 357.873544 182.878622 \r\nL 360.92328 183.013155 \r\nL 361.533227 182.905502 \r\nL 362.143174 183.021555 \r\nL 362.753121 182.840044 \r\nL 363.363068 182.807509 \r\nL 363.363068 182.807509 \r\n\" style=\"fill:none;stroke:#2ca02c;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_17\">\r\n    <path clip-path=\"url(#p1b07345985)\" d=\"M 58.999432 203.637274 \r\nL 59.609379 222.169091 \r\nL 60.219326 166.573638 \r\nL 60.829273 185.105455 \r\nL 61.43922 173.986364 \r\nL 62.049168 172.75091 \r\nL 62.659115 171.868443 \r\nL 63.269062 166.573638 \r\nL 63.879009 170.69182 \r\nL 65.098903 169.943062 \r\nL 65.708851 172.75091 \r\nL 66.318798 177.977833 \r\nL 66.928745 179.81065 \r\nL 67.538692 178.928182 \r\nL 68.758586 181.835134 \r\nL 69.978481 180.228663 \r\nL 71.198375 185.987922 \r\nL 71.808322 183.420745 \r\nL 73.028217 182.016821 \r\nL 73.638164 184.364186 \r\nL 74.248111 183.67993 \r\nL 75.468005 187.752857 \r\nL 76.077952 185.744483 \r\nL 76.6879 185.105455 \r\nL 77.297847 182.116455 \r\nL 77.907794 180.472505 \r\nL 78.517741 182.297606 \r\nL 79.737635 183.517015 \r\nL 80.347583 182.016821 \r\nL 80.95753 182.601156 \r\nL 81.567477 182.179383 \r\nL 82.177424 182.729581 \r\nL 82.787371 178.619319 \r\nL 84.617213 180.364758 \r\nL 85.22716 179.208967 \r\nL 85.837107 180.575457 \r\nL 88.276896 182.458052 \r\nL 89.49679 180.381658 \r\nL 90.106737 180.116122 \r\nL 91.326632 180.987273 \r\nL 92.546526 180.472505 \r\nL 93.156473 180.878903 \r\nL 94.376368 182.906767 \r\nL 94.986315 183.252275 \r\nL 95.596262 182.978857 \r\nL 96.816156 183.634675 \r\nL 97.426103 183.368101 \r\nL 98.645998 181.736036 \r\nL 99.255945 182.062921 \r\nL 99.865892 181.290083 \r\nL 100.475839 181.613952 \r\nL 101.085786 182.458052 \r\nL 101.695734 182.756356 \r\nL 102.305681 182.016821 \r\nL 103.525575 181.599438 \r\nL 104.135522 181.893274 \r\nL 104.745469 182.667061 \r\nL 105.355417 181.495364 \r\nL 105.965364 182.254406 \r\nL 106.575311 182.525078 \r\nL 107.185258 181.86239 \r\nL 107.795205 182.131212 \r\nL 108.405152 180.585503 \r\nL 109.0151 181.309784 \r\nL 110.234994 181.835134 \r\nL 110.844941 182.519622 \r\nL 111.454888 182.762354 \r\nL 112.064835 181.736036 \r\nL 112.674783 181.982115 \r\nL 113.28473 180.575457 \r\nL 113.894677 180.421589 \r\nL 114.504624 179.465341 \r\nL 115.114571 179.725249 \r\nL 115.724518 180.373926 \r\nL 116.334466 180.228663 \r\nL 116.944413 180.858584 \r\nL 117.55436 179.565018 \r\nL 118.164307 179.432452 \r\nL 118.774254 178.928182 \r\nL 119.384201 179.175278 \r\nL 121.214043 178.808236 \r\nL 122.433937 179.987146 \r\nL 123.653832 179.736427 \r\nL 124.263779 179.957731 \r\nL 125.483673 179.040496 \r\nL 126.09362 178.928182 \r\nL 126.703567 178.156024 \r\nL 127.923462 178.603065 \r\nL 128.533409 179.143046 \r\nL 129.143356 179.034686 \r\nL 129.753303 179.561749 \r\nL 130.36325 179.451683 \r\nL 130.973198 179.65492 \r\nL 131.583145 180.163641 \r\nL 132.193092 180.051327 \r\nL 132.803039 180.244651 \r\nL 135.242828 179.81065 \r\nL 135.852775 180.290105 \r\nL 136.462722 180.182944 \r\nL 137.682616 179.403357 \r\nL 139.512458 180.786013 \r\nL 140.122405 180.679951 \r\nL 140.732352 181.124545 \r\nL 141.342299 181.017555 \r\nL 142.562194 180.271067 \r\nL 143.172141 180.172528 \r\nL 143.782088 180.340131 \r\nL 144.392035 179.979634 \r\nL 145.61193 180.310512 \r\nL 146.221877 180.729889 \r\nL 148.661665 181.349007 \r\nL 149.271613 181.747342 \r\nL 149.88156 181.646184 \r\nL 150.491507 181.300914 \r\nL 151.101454 181.20402 \r\nL 151.711401 181.592888 \r\nL 152.321348 181.495364 \r\nL 152.931296 181.877335 \r\nL 154.15119 182.15453 \r\nL 155.371084 181.958543 \r\nL 155.981031 182.094037 \r\nL 156.590979 181.537217 \r\nL 157.810873 181.353619 \r\nL 158.42082 181.489494 \r\nL 159.030767 181.84835 \r\nL 159.640714 181.756333 \r\nL 160.860609 182.458052 \r\nL 161.470556 182.583373 \r\nL 162.69045 183.263111 \r\nL 163.300397 183.166078 \r\nL 163.910345 183.284412 \r\nL 164.520292 183.188374 \r\nL 165.740186 183.420745 \r\nL 168.179975 183.046364 \r\nL 168.789922 183.16013 \r\nL 169.399869 183.068991 \r\nL 170.619763 182.486832 \r\nL 171.229711 182.400813 \r\nL 171.839658 182.116455 \r\nL 172.449605 182.429742 \r\nL 173.059552 182.345401 \r\nL 173.669499 182.065848 \r\nL 174.279446 181.984308 \r\nL 174.889394 182.097677 \r\nL 175.499341 181.823779 \r\nL 176.109288 181.744763 \r\nL 177.329182 181.209024 \r\nL 177.939129 181.323453 \r\nL 178.549077 181.62486 \r\nL 179.159024 181.736036 \r\nL 179.768971 181.287344 \r\nL 180.378918 181.213773 \r\nL 180.988865 180.956544 \r\nL 182.20876 181.179995 \r\nL 182.818707 181.108396 \r\nL 183.428654 181.218297 \r\nL 184.038601 180.787366 \r\nL 184.648548 180.718699 \r\nL 185.258495 180.828882 \r\nL 185.868443 180.760679 \r\nL 186.47839 180.340131 \r\nL 187.088337 180.626202 \r\nL 188.308231 180.842269 \r\nL 188.918178 181.121982 \r\nL 190.74802 180.920855 \r\nL 191.967914 181.128306 \r\nL 192.577861 181.399095 \r\nL 193.797756 181.599438 \r\nL 194.407703 181.532059 \r\nL 195.01765 181.630742 \r\nL 195.627597 181.399095 \r\nL 196.847492 181.595025 \r\nL 197.457439 181.366581 \r\nL 198.067386 181.463834 \r\nL 199.28728 181.014015 \r\nL 199.897227 181.111533 \r\nL 200.507175 181.367277 \r\nL 201.117122 181.462448 \r\nL 201.727069 181.399095 \r\nL 202.946963 180.961211 \r\nL 203.55691 181.056486 \r\nL 204.166858 180.840811 \r\nL 204.776805 180.781368 \r\nL 205.386752 180.56862 \r\nL 206.606646 181.063539 \r\nL 207.216593 180.54845 \r\nL 207.826541 180.642693 \r\nL 209.046435 180.528772 \r\nL 209.656382 180.323055 \r\nL 210.266329 180.416685 \r\nL 210.876276 180.361312 \r\nL 211.486224 180.454047 \r\nL 212.096171 180.693117 \r\nL 212.706118 180.783809 \r\nL 213.316065 180.727862 \r\nL 213.926012 180.52701 \r\nL 214.535959 180.182944 \r\nL 215.755854 180.077445 \r\nL 216.365801 180.168408 \r\nL 216.975748 180.116122 \r\nL 217.585695 180.206239 \r\nL 218.195642 179.729817 \r\nL 218.80559 179.961646 \r\nL 219.415537 179.770537 \r\nL 220.025484 180.000467 \r\nL 220.635431 180.089325 \r\nL 221.245378 180.316333 \r\nL 221.855325 180.403352 \r\nL 222.465273 180.627511 \r\nL 223.07522 180.300913 \r\nL 223.685167 180.250257 \r\nL 224.295114 180.336238 \r\nL 224.905061 180.285825 \r\nL 225.515008 179.965247 \r\nL 226.124956 180.186103 \r\nL 226.734903 180.271067 \r\nL 227.954797 180.172528 \r\nL 228.564744 179.99094 \r\nL 229.174691 179.678282 \r\nL 229.784639 179.895443 \r\nL 231.61448 179.363206 \r\nL 234.054269 179.700341 \r\nL 234.664216 179.65492 \r\nL 235.274163 179.73762 \r\nL 235.88411 179.565018 \r\nL 236.494057 179.647453 \r\nL 237.104005 179.476337 \r\nL 239.543793 179.801736 \r\nL 240.15374 179.757348 \r\nL 240.763688 179.589297 \r\nL 241.373635 179.29882 \r\nL 241.983582 179.133408 \r\nL 242.593529 179.091821 \r\nL 243.203476 179.295152 \r\nL 244.423371 179.211729 \r\nL 245.033318 179.412674 \r\nL 246.253212 179.329306 \r\nL 249.302948 179.717611 \r\nL 249.912895 179.557717 \r\nL 250.522842 179.516494 \r\nL 251.132789 179.710116 \r\nL 251.742737 179.785598 \r\nL 252.352684 179.627499 \r\nL 252.962631 179.586574 \r\nL 253.572578 179.777557 \r\nL 254.182525 179.736427 \r\nL 254.792472 179.235129 \r\nL 255.40242 179.425427 \r\nL 256.012367 179.385761 \r\nL 256.622314 179.460381 \r\nL 257.842208 179.154876 \r\nL 258.452155 179.116514 \r\nL 259.67205 179.265124 \r\nL 262.721786 179.075699 \r\nL 263.94168 178.67156 \r\nL 264.551627 178.635766 \r\nL 266.381469 179.073104 \r\nL 266.991416 178.928182 \r\nL 270.041152 179.391036 \r\nL 270.651099 179.034686 \r\nL 271.870993 179.281174 \r\nL 273.700835 179.488162 \r\nL 274.310782 179.242281 \r\nL 274.920729 179.311003 \r\nL 275.530676 179.171113 \r\nL 281.020201 179.875931 \r\nL 281.630148 180.042121 \r\nL 284.069936 179.896515 \r\nL 285.289831 180.123788 \r\nL 287.119672 180.213055 \r\nL 287.729619 180.17678 \r\nL 288.949514 180.398962 \r\nL 289.559461 180.460277 \r\nL 291.389302 180.060147 \r\nL 293.219144 180.147592 \r\nL 295.048985 179.851591 \r\nL 295.658933 180.008017 \r\nL 299.318616 179.897779 \r\nL 300.53851 179.924019 \r\nL 301.148457 179.797351 \r\nL 302.368351 179.823888 \r\nL 304.198193 179.541314 \r\nL 305.418087 179.568791 \r\nL 306.028034 179.536785 \r\nL 306.637982 179.3228 \r\nL 307.247929 179.473239 \r\nL 308.467823 179.50071 \r\nL 310.297665 179.855524 \r\nL 313.3474 180.051327 \r\nL 314.567295 179.8989 \r\nL 315.177242 179.779208 \r\nL 317.007083 180.122844 \r\nL 319.446872 179.90962 \r\nL 320.056819 179.792137 \r\nL 320.666766 179.847593 \r\nL 321.276714 179.7308 \r\nL 322.496608 179.841225 \r\nL 323.106555 179.725249 \r\nL 324.326449 179.83494 \r\nL 324.936397 179.550151 \r\nL 326.766238 179.293203 \r\nL 328.59608 179.458062 \r\nL 329.815974 179.233577 \r\nL 332.86571 179.50473 \r\nL 335.915446 178.955339 \r\nL 337.745287 179.036084 \r\nL 339.575129 178.954985 \r\nL 340.795023 179.141654 \r\nL 342.014917 179.167302 \r\nL 343.234812 179.192735 \r\nL 343.844759 179.244966 \r\nL 345.064653 178.954471 \r\nL 345.6746 178.849492 \r\nL 346.284548 178.902009 \r\nL 347.504442 178.615409 \r\nL 351.164125 178.850967 \r\nL 352.384019 178.723128 \r\nL 352.993966 178.621241 \r\nL 354.823808 178.775662 \r\nL 355.433755 178.598393 \r\nL 356.043702 178.649701 \r\nL 357.263597 178.524772 \r\nL 362.143174 178.853761 \r\nL 363.363068 178.80464 \r\nL 363.363068 178.80464 \r\n\" style=\"fill:none;stroke:#d62728;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_18\">\r\n    <path clip-path=\"url(#p1b07345985)\" d=\"M 58.999432 166.573638 \r\nL 59.609379 129.51 \r\nL 60.219326 166.573638 \r\nL 60.829273 175.839549 \r\nL 61.43922 188.81182 \r\nL 62.049168 191.282727 \r\nL 62.659115 177.163248 \r\nL 63.269062 185.105455 \r\nL 63.879009 187.164545 \r\nL 64.488956 181.399095 \r\nL 65.708851 178.928182 \r\nL 66.928745 182.458052 \r\nL 67.538692 178.928182 \r\nL 68.148639 178.156024 \r\nL 68.758586 179.65492 \r\nL 69.368534 176.869092 \r\nL 69.978481 176.327228 \r\nL 70.588428 177.692729 \r\nL 71.198375 175.398313 \r\nL 71.808322 176.681904 \r\nL 72.418269 176.242412 \r\nL 73.028217 174.295232 \r\nL 74.248111 179.403357 \r\nL 74.858058 178.928182 \r\nL 75.468005 177.163248 \r\nL 76.077952 179.354203 \r\nL 76.6879 180.163641 \r\nL 77.297847 178.529649 \r\nL 77.907794 178.156024 \r\nL 78.517741 175.558764 \r\nL 79.737635 172.927406 \r\nL 80.95753 174.5874 \r\nL 81.567477 174.376508 \r\nL 82.787371 177.692729 \r\nL 84.617213 176.916981 \r\nL 85.22716 177.524258 \r\nL 85.837107 176.457276 \r\nL 86.447054 177.048145 \r\nL 87.057002 176.825284 \r\nL 87.666949 177.383866 \r\nL 88.886843 176.951455 \r\nL 89.49679 177.474707 \r\nL 90.106737 176.552309 \r\nL 90.716685 176.364032 \r\nL 91.326632 177.555457 \r\nL 92.546526 177.163248 \r\nL 93.156473 177.627708 \r\nL 93.76642 176.798089 \r\nL 94.376368 176.624797 \r\nL 94.986315 177.075002 \r\nL 95.596262 176.902848 \r\nL 96.206209 176.138449 \r\nL 96.816156 176.574936 \r\nL 97.426103 177.576908 \r\nL 98.645998 177.243473 \r\nL 99.255945 177.637412 \r\nL 100.475839 177.316724 \r\nL 101.085786 176.633771 \r\nL 101.695734 177.014101 \r\nL 102.305681 176.869092 \r\nL 102.915628 176.220337 \r\nL 103.525575 176.08998 \r\nL 104.135522 176.457276 \r\nL 104.745469 175.839549 \r\nL 105.355417 176.20056 \r\nL 105.965364 176.077134 \r\nL 106.575311 176.89516 \r\nL 109.0151 178.183937 \r\nL 109.625047 178.045715 \r\nL 110.234994 178.782836 \r\nL 111.454888 178.502167 \r\nL 112.064835 179.208967 \r\nL 112.674783 178.234107 \r\nL 114.504624 179.062472 \r\nL 115.114571 179.725249 \r\nL 115.724518 179.191045 \r\nL 116.334466 178.277942 \r\nL 117.55436 177.272419 \r\nL 118.164307 177.163248 \r\nL 118.774254 177.430661 \r\nL 120.604096 177.111338 \r\nL 121.214043 177.368871 \r\nL 121.82399 177.265073 \r\nL 123.043884 177.76266 \r\nL 123.653832 177.311703 \r\nL 124.263779 177.212274 \r\nL 124.873726 177.794737 \r\nL 126.703567 177.494175 \r\nL 127.923462 178.603065 \r\nL 129.753303 177.344267 \r\nL 130.36325 177.881187 \r\nL 130.973198 177.786166 \r\nL 132.193092 178.213457 \r\nL 133.412986 179.229513 \r\nL 134.632881 179.02702 \r\nL 135.242828 179.516494 \r\nL 135.852775 178.830902 \r\nL 136.462722 178.735146 \r\nL 137.072669 178.928182 \r\nL 137.682616 179.403357 \r\nL 138.292564 179.305425 \r\nL 138.902511 178.928182 \r\nL 139.512458 179.113968 \r\nL 140.122405 178.467196 \r\nL 141.342299 178.837342 \r\nL 141.952247 178.206752 \r\nL 142.562194 178.39103 \r\nL 143.172141 178.839302 \r\nL 143.782088 178.751692 \r\nL 145.61193 180.051327 \r\nL 146.831824 179.354203 \r\nL 147.441771 179.520526 \r\nL 148.051718 179.432452 \r\nL 148.661665 179.846427 \r\nL 149.271613 180.0061 \r\nL 150.491507 179.828185 \r\nL 151.101454 180.228663 \r\nL 153.541243 179.878532 \r\nL 154.15119 180.265937 \r\nL 154.761137 180.413857 \r\nL 155.371084 180.793021 \r\nL 155.981031 180.472505 \r\nL 157.810873 180.216694 \r\nL 158.42082 180.359506 \r\nL 159.030767 179.826694 \r\nL 159.640714 179.970135 \r\nL 160.250662 179.2241 \r\nL 160.860609 178.928182 \r\nL 162.69045 179.361676 \r\nL 163.910345 179.213838 \r\nL 164.520292 178.928182 \r\nL 165.130239 178.434003 \r\nL 165.740186 178.366613 \r\nL 166.350133 178.090589 \r\nL 166.96008 178.025888 \r\nL 168.179975 178.310456 \r\nL 168.789922 178.040844 \r\nL 169.399869 178.181479 \r\nL 170.009816 178.118049 \r\nL 170.619763 178.25674 \r\nL 171.229711 177.993247 \r\nL 171.839658 178.131116 \r\nL 172.449605 178.069314 \r\nL 173.059552 177.811019 \r\nL 173.669499 178.143769 \r\nL 174.279446 177.49766 \r\nL 174.889394 177.246412 \r\nL 175.499341 177.576908 \r\nL 176.109288 177.327852 \r\nL 176.719235 177.463468 \r\nL 177.329182 177.027483 \r\nL 178.549077 176.921355 \r\nL 179.159024 177.243473 \r\nL 179.768971 177.376106 \r\nL 180.378918 177.136776 \r\nL 181.598812 177.399159 \r\nL 182.20876 177.163248 \r\nL 182.818707 176.747969 \r\nL 183.428654 176.155946 \r\nL 184.038601 175.929506 \r\nL 184.648548 176.242412 \r\nL 185.258495 176.374117 \r\nL 186.47839 176.28078 \r\nL 187.088337 176.410436 \r\nL 187.698284 176.713688 \r\nL 188.308231 176.666086 \r\nL 188.918178 176.272534 \r\nL 189.528126 176.572202 \r\nL 190.138073 176.354323 \r\nL 190.74802 176.650848 \r\nL 191.967914 176.897303 \r\nL 192.577861 177.187316 \r\nL 193.797756 177.091698 \r\nL 194.407703 177.37694 \r\nL 195.01765 177.163248 \r\nL 195.627597 177.116187 \r\nL 196.847492 177.34985 \r\nL 197.457439 177.627708 \r\nL 198.067386 177.417583 \r\nL 198.677333 177.692729 \r\nL 199.28728 177.644597 \r\nL 199.897227 177.91639 \r\nL 200.507175 178.026782 \r\nL 201.727069 177.929309 \r\nL 202.946963 178.146249 \r\nL 203.55691 177.786166 \r\nL 204.166858 177.894332 \r\nL 204.776805 177.847161 \r\nL 205.386752 178.107964 \r\nL 205.996699 178.213457 \r\nL 206.606646 177.86051 \r\nL 208.436488 178.174857 \r\nL 210.266329 178.035083 \r\nL 211.486224 178.239089 \r\nL 212.096171 178.192796 \r\nL 212.706118 177.853877 \r\nL 213.316065 177.955388 \r\nL 213.926012 177.910752 \r\nL 214.535959 178.011246 \r\nL 215.145907 178.255171 \r\nL 216.975748 178.548046 \r\nL 217.585695 178.218151 \r\nL 218.195642 178.173708 \r\nL 218.80559 178.270531 \r\nL 219.415537 178.507005 \r\nL 220.025484 178.461977 \r\nL 220.635431 178.277942 \r\nL 221.855325 178.467196 \r\nL 222.465273 178.42298 \r\nL 223.07522 178.516367 \r\nL 223.685167 178.061997 \r\nL 224.295114 177.883496 \r\nL 225.515008 178.071485 \r\nL 226.124956 178.029671 \r\nL 226.734903 178.12245 \r\nL 228.564744 177.998273 \r\nL 229.174691 178.222211 \r\nL 229.784639 178.180756 \r\nL 230.394586 178.271028 \r\nL 231.004533 178.229694 \r\nL 231.61448 178.31916 \r\nL 232.224427 178.017852 \r\nL 232.834374 177.977833 \r\nL 233.444322 177.808959 \r\nL 234.054269 178.027335 \r\nL 234.664216 177.987702 \r\nL 235.274163 178.203952 \r\nL 235.88411 178.291352 \r\nL 236.494057 178.251223 \r\nL 237.104005 178.464363 \r\nL 238.323899 178.383745 \r\nL 238.933846 178.594278 \r\nL 240.15374 178.5136 \r\nL 240.763688 178.597631 \r\nL 241.373635 178.557545 \r\nL 242.593529 178.723641 \r\nL 243.203476 178.561218 \r\nL 243.813423 178.521784 \r\nL 244.423371 178.604131 \r\nL 245.033318 178.564813 \r\nL 245.643265 178.40503 \r\nL 246.253212 178.607285 \r\nL 247.473106 178.768769 \r\nL 251.742737 178.499477 \r\nL 252.352684 178.695082 \r\nL 252.962631 178.77327 \r\nL 253.572578 178.619319 \r\nL 254.182525 178.697258 \r\nL 254.792472 178.659609 \r\nL 256.012367 179.042579 \r\nL 256.622314 179.118253 \r\nL 257.232261 179.307159 \r\nL 257.842208 179.268217 \r\nL 259.062103 179.416358 \r\nL 259.67205 179.377438 \r\nL 260.281997 179.450733 \r\nL 260.891944 179.411945 \r\nL 261.501891 179.595996 \r\nL 263.331733 179.479728 \r\nL 265.771521 179.763932 \r\nL 266.381469 179.725249 \r\nL 266.991416 179.903545 \r\nL 268.21131 179.933789 \r\nL 271.870993 180.234235 \r\nL 273.090887 180.36721 \r\nL 273.700835 180.223134 \r\nL 274.310782 180.289276 \r\nL 274.920729 180.146239 \r\nL 277.360518 180.82094 \r\nL 277.970465 180.67841 \r\nL 278.580412 180.742012 \r\nL 279.190359 180.498098 \r\nL 280.410253 180.828882 \r\nL 281.020201 180.789829 \r\nL 281.630148 180.953517 \r\nL 282.240095 180.813346 \r\nL 283.459989 180.937053 \r\nL 284.679884 180.759718 \r\nL 285.899778 180.882388 \r\nL 286.509725 180.745027 \r\nL 290.169408 181.20402 \r\nL 290.779355 180.971063 \r\nL 292.609197 181.244663 \r\nL 293.829091 180.880587 \r\nL 295.658933 180.960807 \r\nL 296.26888 180.828882 \r\nL 299.318616 181.023763 \r\nL 299.928563 180.987273 \r\nL 301.148457 181.194217 \r\nL 303.588246 181.325336 \r\nL 304.80814 181.160566 \r\nL 306.028034 180.997419 \r\nL 307.247929 181.199242 \r\nL 307.857876 181.072862 \r\nL 308.467823 181.127898 \r\nL 309.687717 180.967286 \r\nL 311.517559 181.131162 \r\nL 312.127506 180.828882 \r\nL 317.007083 180.472505 \r\nL 318.226978 180.320243 \r\nL 320.056819 180.396907 \r\nL 322.496608 180.269206 \r\nL 323.716502 180.461851 \r\nL 325.546344 180.282108 \r\nL 327.376185 180.020764 \r\nL 327.986132 180.158047 \r\nL 329.206027 180.180332 \r\nL 329.815974 180.233048 \r\nL 330.425921 180.11932 \r\nL 337.13534 180.604292 \r\nL 338.965181 180.512789 \r\nL 340.185076 180.211773 \r\nL 340.795023 180.342418 \r\nL 342.624865 180.333316 \r\nL 343.234812 180.46258 \r\nL 346.284548 180.23693 \r\nL 348.724336 179.888517 \r\nL 349.334283 179.860605 \r\nL 349.944231 179.987886 \r\nL 351.164125 180.00921 \r\nL 351.774072 180.05833 \r\nL 353.603914 179.821591 \r\nL 354.823808 179.843334 \r\nL 355.433755 179.968301 \r\nL 360.92328 179.725249 \r\nL 362.753121 180.01756 \r\nL 363.363068 179.990675 \r\nL 363.363068 179.990675 \r\n\" style=\"fill:none;stroke:#9467bd;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_19\">\r\n    <path clip-path=\"url(#p1b07345985)\" d=\"M 58.999432 240.700909 \r\nL 59.609379 222.169091 \r\nL 60.219326 215.991818 \r\nL 60.829273 203.637274 \r\nL 61.43922 196.224548 \r\nL 62.049168 197.460002 \r\nL 62.659115 193.047662 \r\nL 63.269062 189.73841 \r\nL 63.879009 183.046364 \r\nL 64.488956 185.105455 \r\nL 65.098903 180.051327 \r\nL 65.708851 172.75091 \r\nL 66.318798 175.126784 \r\nL 66.928745 171.868443 \r\nL 67.538692 173.986364 \r\nL 68.148639 168.890119 \r\nL 68.758586 170.934066 \r\nL 69.368534 174.810001 \r\nL 69.978481 176.327228 \r\nL 70.588428 173.986364 \r\nL 71.198375 175.398313 \r\nL 71.808322 173.31248 \r\nL 72.418269 169.796567 \r\nL 73.638164 172.503821 \r\nL 74.248111 175.126784 \r\nL 75.468005 174.515845 \r\nL 76.077952 176.798089 \r\nL 76.6879 177.692729 \r\nL 77.297847 179.725249 \r\nL 77.907794 179.314262 \r\nL 79.127688 180.745027 \r\nL 79.737635 180.340131 \r\nL 80.347583 182.016821 \r\nL 80.95753 179.595996 \r\nL 81.567477 181.20402 \r\nL 82.177424 179.878532 \r\nL 83.397319 182.845479 \r\nL 84.007266 182.458052 \r\nL 84.617213 178.64087 \r\nL 85.22716 179.208967 \r\nL 87.057002 176.036695 \r\nL 88.886843 177.692729 \r\nL 89.49679 178.928182 \r\nL 90.106737 178.690598 \r\nL 90.716685 177.063349 \r\nL 91.936579 179.377438 \r\nL 93.156473 178.928182 \r\nL 93.76642 178.076146 \r\nL 94.986315 177.692729 \r\nL 96.816156 178.928182 \r\nL 97.426103 178.735146 \r\nL 98.036051 179.118253 \r\nL 98.645998 180.051327 \r\nL 100.475839 179.465341 \r\nL 101.085786 178.751692 \r\nL 101.695734 178.580167 \r\nL 102.305681 177.89864 \r\nL 104.135522 178.928182 \r\nL 104.745469 178.765627 \r\nL 105.355417 179.088634 \r\nL 105.965364 178.453008 \r\nL 106.575311 177.36432 \r\nL 107.185258 177.692729 \r\nL 109.0151 177.290832 \r\nL 110.234994 177.910752 \r\nL 110.844941 177.778925 \r\nL 112.064835 178.366613 \r\nL 112.674783 179.067001 \r\nL 113.28473 178.928182 \r\nL 113.894677 179.607004 \r\nL 115.114571 180.123788 \r\nL 115.724518 179.585343 \r\nL 116.334466 180.228663 \r\nL 116.944413 180.086425 \r\nL 118.164307 181.323453 \r\nL 118.774254 181.548843 \r\nL 119.384201 181.399095 \r\nL 120.604096 181.835134 \r\nL 121.214043 182.40665 \r\nL 121.82399 181.898029 \r\nL 122.433937 181.75208 \r\nL 123.653832 182.161152 \r\nL 124.263779 182.016821 \r\nL 124.873726 181.195073 \r\nL 125.483673 181.736036 \r\nL 126.09362 181.933342 \r\nL 127.313515 182.973478 \r\nL 127.923462 182.5045 \r\nL 128.533409 182.365969 \r\nL 129.143356 182.549346 \r\nL 129.753303 182.412797 \r\nL 130.36325 182.592668 \r\nL 130.973198 183.08097 \r\nL 131.583145 182.943411 \r\nL 132.193092 183.114434 \r\nL 132.803039 182.978857 \r\nL 133.412986 182.544149 \r\nL 135.242828 182.163896 \r\nL 136.462722 181.341181 \r\nL 137.072669 181.514021 \r\nL 137.682616 181.399095 \r\nL 138.292564 181.568852 \r\nL 139.512458 181.343357 \r\nL 140.122405 181.50973 \r\nL 140.732352 181.399095 \r\nL 141.342299 181.835134 \r\nL 141.952247 181.723737 \r\nL 143.172141 180.439174 \r\nL 143.782088 180.075391 \r\nL 144.392035 180.242497 \r\nL 145.001982 180.668259 \r\nL 145.61193 180.828882 \r\nL 146.831824 180.63226 \r\nL 147.441771 180.282108 \r\nL 148.051718 180.693117 \r\nL 148.661665 180.096858 \r\nL 149.271613 180.0061 \r\nL 150.491507 180.319095 \r\nL 151.101454 180.716341 \r\nL 151.711401 180.623904 \r\nL 152.321348 180.291999 \r\nL 153.541243 180.591297 \r\nL 154.15119 180.502014 \r\nL 154.761137 180.648437 \r\nL 155.371084 180.093705 \r\nL 155.981031 180.00921 \r\nL 156.590979 180.155964 \r\nL 157.200926 180.5297 \r\nL 158.42082 180.359506 \r\nL 159.030767 180.500583 \r\nL 160.250662 181.221544 \r\nL 161.470556 181.486821 \r\nL 162.080503 181.181072 \r\nL 163.300397 181.01122 \r\nL 165.740186 181.525448 \r\nL 166.350133 181.440975 \r\nL 166.96008 180.941002 \r\nL 169.399869 180.625235 \r\nL 170.009816 180.345919 \r\nL 171.839658 180.123788 \r\nL 172.449605 180.249528 \r\nL 175.499341 179.893383 \r\nL 176.109288 180.208449 \r\nL 179.768971 180.914846 \r\nL 180.378918 180.65782 \r\nL 180.988865 180.587751 \r\nL 181.598812 180.33489 \r\nL 182.20876 180.63226 \r\nL 182.818707 180.563345 \r\nL 183.428654 180.675902 \r\nL 184.038601 180.60744 \r\nL 184.648548 180.181547 \r\nL 187.088337 179.923577 \r\nL 188.308231 180.146239 \r\nL 189.528126 180.019979 \r\nL 190.138073 180.129322 \r\nL 190.74802 180.40845 \r\nL 191.967914 179.943625 \r\nL 193.797756 180.26381 \r\nL 194.407703 179.87001 \r\nL 195.01765 179.81065 \r\nL 195.627597 180.081273 \r\nL 196.237544 179.857506 \r\nL 197.457439 180.066101 \r\nL 198.067386 179.683485 \r\nL 199.897227 179.034686 \r\nL 200.507175 178.663066 \r\nL 201.117122 178.928182 \r\nL 201.727069 178.87561 \r\nL 202.337016 179.137584 \r\nL 203.55691 179.343461 \r\nL 204.166858 179.290033 \r\nL 204.776805 179.545909 \r\nL 205.386752 179.645874 \r\nL 205.996699 179.59186 \r\nL 207.216593 180.092755 \r\nL 209.046435 179.928553 \r\nL 210.266329 180.118983 \r\nL 210.876276 179.916546 \r\nL 212.096171 180.398962 \r\nL 213.316065 179.998263 \r\nL 213.926012 179.945619 \r\nL 214.535959 180.038161 \r\nL 215.145907 179.841556 \r\nL 215.755854 180.077445 \r\nL 216.975748 179.97357 \r\nL 217.585695 180.064234 \r\nL 218.195642 180.295672 \r\nL 218.80559 180.102569 \r\nL 219.415537 180.19172 \r\nL 220.025484 180.140329 \r\nL 220.635431 180.368 \r\nL 221.245378 180.316333 \r\nL 221.855325 180.126759 \r\nL 222.465273 180.214159 \r\nL 223.07522 180.438185 \r\nL 223.685167 180.52379 \r\nL 224.295114 180.336238 \r\nL 224.905061 180.285825 \r\nL 227.34485 180.623032 \r\nL 227.954797 180.572497 \r\nL 228.564744 180.655164 \r\nL 229.174691 180.604872 \r\nL 231.61448 180.929272 \r\nL 232.224427 181.138999 \r\nL 233.444322 181.295777 \r\nL 234.054269 181.115968 \r\nL 234.664216 181.322144 \r\nL 235.274163 181.143483 \r\nL 235.88411 181.220782 \r\nL 236.494057 181.170617 \r\nL 237.104005 180.994304 \r\nL 237.713952 180.945255 \r\nL 238.323899 180.645256 \r\nL 239.543793 180.550499 \r\nL 240.763688 180.209089 \r\nL 241.373635 180.410731 \r\nL 241.983582 180.487897 \r\nL 242.593529 180.441819 \r\nL 243.203476 180.640694 \r\nL 243.813423 180.350581 \r\nL 244.423371 180.183894 \r\nL 245.033318 180.381658 \r\nL 245.643265 180.215954 \r\nL 246.253212 180.291999 \r\nL 246.863159 180.127654 \r\nL 247.473106 180.323055 \r\nL 248.693001 180.234914 \r\nL 249.302948 179.954439 \r\nL 249.912895 180.029865 \r\nL 250.522842 179.869485 \r\nL 251.132789 179.592826 \r\nL 251.742737 179.551758 \r\nL 252.352684 179.627499 \r\nL 252.962631 179.470389 \r\nL 253.572578 179.545909 \r\nL 254.182525 179.505498 \r\nL 254.792472 179.695547 \r\nL 255.40242 179.65492 \r\nL 256.012367 179.500152 \r\nL 256.622314 179.118253 \r\nL 257.842208 179.381564 \r\nL 258.452155 179.229513 \r\nL 259.062103 179.303702 \r\nL 259.67205 179.265124 \r\nL 260.891944 179.411945 \r\nL 261.501891 179.262092 \r\nL 263.94168 179.55141 \r\nL 265.771521 179.436903 \r\nL 266.381469 179.616558 \r\nL 268.821257 178.928182 \r\nL 269.431204 179.106716 \r\nL 270.041152 178.856976 \r\nL 273.700835 178.753189 \r\nL 274.310782 178.823485 \r\nL 274.920729 178.684572 \r\nL 275.530676 178.754664 \r\nL 276.75057 178.583083 \r\nL 277.360518 178.549631 \r\nL 277.970465 178.722277 \r\nL 278.580412 178.380613 \r\nL 279.800306 178.519769 \r\nL 280.410253 178.283305 \r\nL 284.069936 178.794622 \r\nL 284.679884 178.561875 \r\nL 285.899778 178.596962 \r\nL 286.509725 178.763014 \r\nL 287.119672 178.631674 \r\nL 287.729619 178.796754 \r\nL 288.949514 178.437925 \r\nL 289.559461 178.406621 \r\nL 290.779355 178.733627 \r\nL 291.99925 178.573352 \r\nL 295.658933 178.959945 \r\nL 296.26888 178.83315 \r\nL 302.368351 179.360593 \r\nL 302.978299 179.236278 \r\nL 303.588246 179.296975 \r\nL 304.80814 179.050504 \r\nL 305.418087 179.202732 \r\nL 307.247929 179.200711 \r\nL 308.467823 179.319912 \r\nL 309.07777 179.108544 \r\nL 309.687717 179.258038 \r\nL 310.297665 178.9581 \r\nL 310.907612 178.838656 \r\nL 312.127506 179.04698 \r\nL 312.737453 178.839302 \r\nL 313.957348 179.04613 \r\nL 314.567295 179.016433 \r\nL 316.397136 179.366287 \r\nL 317.617031 179.306088 \r\nL 318.836925 179.506851 \r\nL 320.056819 179.619347 \r\nL 321.886661 179.271365 \r\nL 323.106555 179.469052 \r\nL 323.716502 179.269001 \r\nL 324.936397 179.550151 \r\nL 325.546344 179.520526 \r\nL 326.766238 179.71438 \r\nL 328.59608 179.625395 \r\nL 329.206027 179.762948 \r\nL 329.815974 179.733313 \r\nL 330.425921 179.537603 \r\nL 331.645815 179.562456 \r\nL 332.255763 179.450981 \r\nL 332.86571 179.50473 \r\nL 333.475657 179.393874 \r\nL 334.085604 179.447513 \r\nL 335.305498 179.309164 \r\nL 335.915446 179.362631 \r\nL 336.525393 179.090744 \r\nL 342.014917 178.928182 \r\nL 343.234812 178.954637 \r\nL 345.6746 178.770802 \r\nL 346.284548 178.666435 \r\nL 347.504442 178.849989 \r\nL 348.114389 178.902175 \r\nL 348.724336 178.798411 \r\nL 349.944231 178.902335 \r\nL 351.164125 178.773751 \r\nL 352.384019 178.800023 \r\nL 352.993966 178.928182 \r\nL 357.263597 178.827329 \r\nL 359.093438 178.903125 \r\nL 361.533227 178.878465 \r\nL 362.143174 178.928182 \r\nL 362.753121 178.829151 \r\nL 363.363068 178.878763 \r\nL 363.363068 178.878763 \r\n\" style=\"fill:none;stroke:#8c564b;stroke-linecap:square;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"line2d_20\">\r\n    <path clip-path=\"url(#p1b07345985)\" d=\"M 43.78125 178.804639 \r\nL 378.58125 178.804639 \r\n\" style=\"fill:none;stroke:#000000;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\r\n   </g>\r\n   <g id=\"patch_3\">\r\n    <path d=\"M 43.78125 251.82 \r\nL 43.78125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_4\">\r\n    <path d=\"M 378.58125 251.82 \r\nL 378.58125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_5\">\r\n    <path d=\"M 43.78125 251.82 \r\nL 378.58125 251.82 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"patch_6\">\r\n    <path d=\"M 43.78125 7.2 \r\nL 378.58125 7.2 \r\n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\r\n   </g>\r\n   <g id=\"legend_1\">\r\n    <g id=\"patch_7\">\r\n     <path d=\"M 295.726562 103.26875 \r\nL 371.58125 103.26875 \r\nQ 373.58125 103.26875 373.58125 101.26875 \r\nL 373.58125 14.2 \r\nQ 373.58125 12.2 371.58125 12.2 \r\nL 295.726562 12.2 \r\nQ 293.726562 12.2 293.726562 14.2 \r\nL 293.726562 101.26875 \r\nQ 293.726562 103.26875 295.726562 103.26875 \r\nz\r\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\r\n    </g>\r\n    <g id=\"line2d_21\">\r\n     <path d=\"M 297.726562 20.298437 \r\nL 317.726562 20.298437 \r\n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_22\"/>\r\n    <g id=\"text_16\">\r\n     <!-- P(die=1) -->\r\n     <g transform=\"translate(325.726562 23.798437)scale(0.1 -0.1)\">\r\n      <defs>\r\n       <path d=\"M 19.671875 64.796875 \r\nL 19.671875 37.40625 \r\nL 32.078125 37.40625 \r\nQ 38.96875 37.40625 42.71875 40.96875 \r\nQ 46.484375 44.53125 46.484375 51.125 \r\nQ 46.484375 57.671875 42.71875 61.234375 \r\nQ 38.96875 64.796875 32.078125 64.796875 \r\nz\r\nM 9.8125 72.90625 \r\nL 32.078125 72.90625 \r\nQ 44.34375 72.90625 50.609375 67.359375 \r\nQ 56.890625 61.8125 56.890625 51.125 \r\nQ 56.890625 40.328125 50.609375 34.8125 \r\nQ 44.34375 29.296875 32.078125 29.296875 \r\nL 19.671875 29.296875 \r\nL 19.671875 0 \r\nL 9.8125 0 \r\nz\r\n\" id=\"DejaVuSans-80\"/>\r\n       <path d=\"M 31 75.875 \r\nQ 24.46875 64.65625 21.28125 53.65625 \r\nQ 18.109375 42.671875 18.109375 31.390625 \r\nQ 18.109375 20.125 21.3125 9.0625 \r\nQ 24.515625 -2 31 -13.1875 \r\nL 23.1875 -13.1875 \r\nQ 15.875 -1.703125 12.234375 9.375 \r\nQ 8.59375 20.453125 8.59375 31.390625 \r\nQ 8.59375 42.28125 12.203125 53.3125 \r\nQ 15.828125 64.359375 23.1875 75.875 \r\nz\r\n\" id=\"DejaVuSans-40\"/>\r\n       <path d=\"M 10.59375 45.40625 \r\nL 73.1875 45.40625 \r\nL 73.1875 37.203125 \r\nL 10.59375 37.203125 \r\nz\r\nM 10.59375 25.484375 \r\nL 73.1875 25.484375 \r\nL 73.1875 17.1875 \r\nL 10.59375 17.1875 \r\nz\r\n\" id=\"DejaVuSans-61\"/>\r\n       <path d=\"M 8.015625 75.875 \r\nL 15.828125 75.875 \r\nQ 23.140625 64.359375 26.78125 53.3125 \r\nQ 30.421875 42.28125 30.421875 31.390625 \r\nQ 30.421875 20.453125 26.78125 9.375 \r\nQ 23.140625 -1.703125 15.828125 -13.1875 \r\nL 8.015625 -13.1875 \r\nQ 14.5 -2 17.703125 9.0625 \r\nQ 20.90625 20.125 20.90625 31.390625 \r\nQ 20.90625 42.671875 17.703125 53.65625 \r\nQ 14.5 64.65625 8.015625 75.875 \r\nz\r\n\" id=\"DejaVuSans-41\"/>\r\n      </defs>\r\n      <use xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"60.302734\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"99.316406\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"162.792969\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"190.576172\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"252.099609\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"335.888672\" xlink:href=\"#DejaVuSans-49\"/>\r\n      <use x=\"399.511719\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_23\">\r\n     <path d=\"M 297.726562 34.976562 \r\nL 317.726562 34.976562 \r\n\" style=\"fill:none;stroke:#ff7f0e;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_24\"/>\r\n    <g id=\"text_17\">\r\n     <!-- P(die=2) -->\r\n     <g transform=\"translate(325.726562 38.476562)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"60.302734\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"99.316406\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"162.792969\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"190.576172\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"252.099609\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"335.888672\" xlink:href=\"#DejaVuSans-50\"/>\r\n      <use x=\"399.511719\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_25\">\r\n     <path d=\"M 297.726562 49.654687 \r\nL 317.726562 49.654687 \r\n\" style=\"fill:none;stroke:#2ca02c;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_26\"/>\r\n    <g id=\"text_18\">\r\n     <!-- P(die=3) -->\r\n     <g transform=\"translate(325.726562 53.154687)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"60.302734\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"99.316406\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"162.792969\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"190.576172\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"252.099609\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"335.888672\" xlink:href=\"#DejaVuSans-51\"/>\r\n      <use x=\"399.511719\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_27\">\r\n     <path d=\"M 297.726562 64.332812 \r\nL 317.726562 64.332812 \r\n\" style=\"fill:none;stroke:#d62728;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_28\"/>\r\n    <g id=\"text_19\">\r\n     <!-- P(die=4) -->\r\n     <g transform=\"translate(325.726562 67.832812)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"60.302734\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"99.316406\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"162.792969\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"190.576172\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"252.099609\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"335.888672\" xlink:href=\"#DejaVuSans-52\"/>\r\n      <use x=\"399.511719\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_29\">\r\n     <path d=\"M 297.726562 79.010937 \r\nL 317.726562 79.010937 \r\n\" style=\"fill:none;stroke:#9467bd;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_30\"/>\r\n    <g id=\"text_20\">\r\n     <!-- P(die=5) -->\r\n     <g transform=\"translate(325.726562 82.510937)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"60.302734\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"99.316406\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"162.792969\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"190.576172\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"252.099609\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"335.888672\" xlink:href=\"#DejaVuSans-53\"/>\r\n      <use x=\"399.511719\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n    <g id=\"line2d_31\">\r\n     <path d=\"M 297.726562 93.689062 \r\nL 317.726562 93.689062 \r\n\" style=\"fill:none;stroke:#8c564b;stroke-linecap:square;stroke-width:1.5;\"/>\r\n    </g>\r\n    <g id=\"line2d_32\"/>\r\n    <g id=\"text_21\">\r\n     <!-- P(die=6) -->\r\n     <g transform=\"translate(325.726562 97.189062)scale(0.1 -0.1)\">\r\n      <use xlink:href=\"#DejaVuSans-80\"/>\r\n      <use x=\"60.302734\" xlink:href=\"#DejaVuSans-40\"/>\r\n      <use x=\"99.316406\" xlink:href=\"#DejaVuSans-100\"/>\r\n      <use x=\"162.792969\" xlink:href=\"#DejaVuSans-105\"/>\r\n      <use x=\"190.576172\" xlink:href=\"#DejaVuSans-101\"/>\r\n      <use x=\"252.099609\" xlink:href=\"#DejaVuSans-61\"/>\r\n      <use x=\"335.888672\" xlink:href=\"#DejaVuSans-54\"/>\r\n      <use x=\"399.511719\" xlink:href=\"#DejaVuSans-41\"/>\r\n     </g>\r\n    </g>\r\n   </g>\r\n  </g>\r\n </g>\r\n <defs>\r\n  <clipPath id=\"p1b07345985\">\r\n   <rect height=\"244.62\" width=\"334.8\" x=\"43.78125\" y=\"7.2\"/>\r\n  </clipPath>\r\n </defs>\r\n</svg>\r\n",
   "text/plain": "<Figure size 432x324 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

每条实线对应于骰子的6个值中的一个，并给出骰子在每组实验后出现值的估计概率。
当我们通过更多的实验获得更多的数据时，这$6$条实体曲线向真实概率收敛。

### 概率论公理

在处理骰子掷出时，我们将集合$\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$
称为*样本空间*（sample space）或*结果空间*（outcome space），
其中每个元素都是*结果*（outcome）。
*事件*（event）是一组给定样本空间的随机结果。
例如，“看到$5$”（$\{5\}$）和“看到奇数”（$\{1, 3, 5\}$）都是掷出骰子的有效事件。
注意，如果一个随机实验的结果在$\mathcal{A}$中，则事件$\mathcal{A}$已经发生。
也就是说，如果投掷出$3$点，因为$3 \in \{1, 3, 5\}$，我们可以说，“看到奇数”的事件发生了。

*概率*（probability）可以被认为是将集合映射到真实值的函数。
在给定的样本空间$\mathcal{S}$中，事件$\mathcal{A}$的概率，
表示为$P(\mathcal{A})$，满足以下属性：

* 对于任意事件$\mathcal{A}$，其概率从不会是负数，即$P(\mathcal{A}) \geq 0$；
* 整个样本空间的概率为$1$，即$P(\mathcal{S}) = 1$；
* 对于*互斥*（mutually exclusive）事件（对于所有$i \neq j$都有$\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$）的任意一个可数序列$\mathcal{A}_1, \mathcal{A}_2, \ldots$，序列中任意一个事件发生的概率等于它们各自发生的概率之和，即$P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$。

以上也是概率论的公理，由科尔莫戈罗夫于1933年提出。
有了这个公理系统，我们可以避免任何关于随机性的哲学争论；
相反，我们可以用数学语言严格地推理。
例如，假设事件$\mathcal{A}_1$为整个样本空间，
且当所有$i > 1$时的$\mathcal{A}_i = \emptyset$，
那么我们可以证明$P(\emptyset) = 0$，即不可能发生事件的概率是$0$。

### 随机变量

在我们掷骰子的随机实验中，我们引入了*随机变量*（random variable）的概念。
随机变量几乎可以是任何数量，并且它可以在随机实验的一组可能性中取一个值。
考虑一个随机变量$X$，其值在掷骰子的样本空间$\mathcal{S}=\{1,2,3,4,5,6\}$中。
我们可以将事件“看到一个$5$”表示为$\{X=5\}$或$X=5$，
其概率表示为$P(\{X=5\})$或$P(X=5)$。
通过$P(X=a)$，我们区分了随机变量$X$和$X$可以采取的值（例如$a$）。
然而，这可能会导致繁琐的表示。
为了简化符号，一方面，我们可以将$P(X)$表示为随机变量$X$上的*分布*（distribution）：
分布告诉我们$X$获得某一值的概率。
另一方面，我们可以简单用$P(a)$表示随机变量取值$a$的概率。
由于概率论中的事件是来自样本空间的一组结果，因此我们可以为随机变量指定值的可取范围。
例如，$P(1 \leq X \leq 3)$表示事件$\{1 \leq X \leq 3\}$，
即$\{X = 1, 2, \text{or}, 3\}$的概率。
等价地，$P(1 \leq X \leq 3)$表示随机变量$X$从$\{1, 2, 3\}$中取值的概率。

请注意，*离散*（discrete）随机变量（如骰子的每一面）
和*连续*（continuous）随机变量（如人的体重和身高）之间存在微妙的区别。
现实生活中，测量两个人是否具有完全相同的身高没有太大意义。
如果我们进行足够精确的测量，你会发现这个星球上没有两个人具有完全相同的身高。
在这种情况下，询问某人的身高是否落入给定的区间，比如是否在1.79米和1.81米之间更有意义。
在这些情况下，我们将这个看到某个数值的可能性量化为*密度*（density）。
高度恰好为1.80米的概率为0，但密度不是0。
在任何两个不同高度之间的区间，我们都有非零的概率。
在本节的其余部分中，我们将考虑离散空间中的概率。
对于连续随机变量的概率，你可以参考深度学习数学附录中[随机变量](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/random-variables.html)
的一节。

## 处理多个随机变量

很多时候，我们会考虑多个随机变量。
比如，我们可能需要对疾病和症状之间的关系进行建模。
给定一个疾病和一个症状，比如“流感”和“咳嗽”，以某个概率存在或不存在于某个患者身上。
我们需要估计这些概率以及概率之间的关系，以便我们可以运用我们的推断来实现更好的医疗服务。

再举一个更复杂的例子：图像包含数百万像素，因此有数百万个随机变量。
在许多情况下，图像会附带一个*标签*（label），标识图像中的对象。
我们也可以将标签视为一个随机变量。
我们甚至可以将所有元数据视为随机变量，例如位置、时间、光圈、焦距、ISO、对焦距离和相机类型。
所有这些都是联合发生的随机变量。
当我们处理多个随机变量时，会有若干个变量是我们感兴趣的。

### 联合概率

第一个被称为*联合概率*（joint probability）$P(A=a,B=b)$。
给定任意值$a$和$b$，联合概率可以回答：$A=a$和$B=b$同时满足的概率是多少？
请注意，对于任何$a$和$b$的取值，$P(A = a, B=b) \leq P(A=a)$。
这点是确定的，因为要同时发生$A=a$和$B=b$，$A=a$就必须发生，$B=b$也必须发生（反之亦然）。因此，$A=a$和$B=b$同时发生的可能性不大于$A=a$或是$B=b$单独发生的可能性。

### 条件概率

联合概率的不等式带给我们一个有趣的比率：
$0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$。
我们称这个比率为*条件概率*（conditional probability），
并用$P(B=b \mid A=a)$表示它：它是$B=b$的概率，前提是$A=a$已发生。

### 贝叶斯定理

使用条件概率的定义，我们可以得出统计学中最有用的方程之一：
*Bayes定理*（Bayes' theorem）。
根据*乘法法则*（multiplication rule ）可得到$P(A, B) = P(B \mid A) P(A)$。
根据对称性，可得到$P(A, B) = P(A \mid B) P(B)$。
假设$P(B)>0$，求解其中一个条件变量，我们得到

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

请注意，这里我们使用紧凑的表示法：
其中$P(A, B)$是一个*联合分布*（joint distribution），
$P(A \mid B)$是一个*条件分布*（conditional distribution）。
这种分布可以在给定值$A = a, B=b$上进行求值。

### 边际化

为了能进行事件概率求和，我们需要*求和法则*（sum rule），
即$B$的概率相当于计算$A$的所有可能选择，并将所有选择的联合概率聚合在一起：

$$P(B) = \sum_{A} P(A, B),$$

这也称为*边际化*（marginalization）。
边际化结果的概率或分布称为*边际概率*（marginal probability）
或*边际分布*（marginal distribution）。

### 独立性

另一个有用属性是*依赖*（dependence）与*独立*（independence）。
如果两个随机变量$A$和$B$是独立的，意味着事件$A$的发生跟$B$事件的发生无关。
在这种情况下，统计学家通常将这一点表述为$A \perp  B$。
根据贝叶斯定理，马上就能同样得到$P(A \mid B) = P(A)$。
在所有其他情况下，我们称$A$和$B$依赖。
比如，两次连续抛出一个骰子的事件是相互独立的。
相比之下，灯开关的位置和房间的亮度并不是（因为可能存在灯泡坏掉、电源故障，或者开关故障）。

由于$P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$等价于$P(A, B) = P(A)P(B)$，
因此两个随机变量是独立的，当且仅当两个随机变量的联合分布是其各自分布的乘积。
同样地，给定另一个随机变量$C$时，两个随机变量$A$和$B$是*条件独立的*（conditionally independent），
当且仅当$P(A, B \mid C) = P(A \mid C)P(B \mid C)$。
这个情况表示为$A \perp B \mid C$。

### 应用
:label:`subsec_probability_hiv_app`

我们实战演练一下！
假设一个医生对患者进行艾滋病病毒（HIV）测试。
这个测试是相当准确的，如果患者健康但测试显示他患病，这个概率只有1%；
如果患者真正感染HIV，它永远不会检测不出。
我们使用$D_1$来表示诊断结果（如果阳性，则为$1$，如果阴性，则为$0$），
$H$来表示感染艾滋病病毒的状态（如果阳性，则为$1$，如果阴性，则为$0$）。
在 :numref:`conditional_prob_D1`中列出了这样的条件概率。

:条件概率为$P(D_1 \mid H)$

| 条件概率 | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

请注意，每列的加和都是1（但每行的加和不是），因为条件概率需要总和为1，就像概率一样。
让我们计算如果测试出来呈阳性，患者感染HIV的概率，即$P(H = 1 \mid D_1 = 1)$。
显然，这将取决于疾病有多常见，因为它会影响错误警报的数量。
假设人口总体是相当健康的，例如，$P(H=1) = 0.0015$。
为了应用贝叶斯定理，我们需要运用边际化和乘法法则来确定

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

因此，我们得到

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

换句话说，尽管使用了非常准确的测试，患者实际上患有艾滋病的几率只有13.06%。
正如我们所看到的，概率可能是违反直觉的。

患者在收到这样可怕的消息后应该怎么办？
很可能，患者会要求医生进行另一次测试来确定病情。
第二个测试具有不同的特性，它不如第一个测试那么精确，
如 :numref:`conditional_prob_D2`所示。

:条件概率为$P(D_2 \mid H)$

| 条件概率 | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

不幸的是，第二次测试也显示阳性。让我们通过假设条件独立性来计算出应用Bayes定理的必要概率：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

现在我们可以应用边际化和乘法规则：

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

最后，鉴于存在两次阳性检测，患者患有艾滋病的概率为

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

也就是说，第二次测试使我们能够对患病的情况获得更高的信心。
尽管第二次检验比第一次检验的准确性要低得多，但它仍然显著提高我们的预测概率。

## 期望和方差

为了概括概率分布的关键特征，我们需要一些测量方法。
一个随机变量$X$的*期望*（expectation，或平均值（average））表示为

$$E[X] = \sum_{x} x P(X = x).$$

当函数$f(x)$的输入是从分布$P$中抽取的随机变量时，$f(x)$的期望值为

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$

在许多情况下，我们希望衡量随机变量$X$与其期望值的偏置。这可以通过方差来量化

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

方差的平方根被称为*标准差*（standard deviation）。
随机变量函数的方差衡量的是：当从该随机变量分布中采样不同值$x$时，
函数值偏离该函数的期望的程度：

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$

## 小结

* 我们可以从概率分布中采样。
* 我们可以使用联合分布、条件分布、Bayes定理、边缘化和独立性假设来分析多个随机变量。
* 期望和方差为概率分布的关键特征的概括提供了实用的度量形式。

## 练习

1. 进行$m=500$组实验，每组抽取$n=10$个样本。改变$m$和$n$，观察和分析实验结果。
2. 给定两个概率为$P(\mathcal{A})$和$P(\mathcal{B})$的事件，计算$P(\mathcal{A} \cup \mathcal{B})$和$P(\mathcal{A} \cap \mathcal{B})$的上限和下限。（提示：使用[友元图](https://en.wikipedia.org/wiki/Venn_diagram)来展示这些情况。)
3. 假设我们有一系列随机变量，例如$A$、$B$和$C$，其中$B$只依赖于$A$，而$C$只依赖于$B$，你能简化联合概率$P(A, B, C)$吗？（提示：这是一个[马尔可夫链](https://en.wikipedia.org/wiki/Markov_chain)。)
4. 在 :numref:`subsec_probability_hiv_app`中，第一个测试更准确。为什么不运行第一个测试两次，而是同时运行第一个和第二个测试?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1761)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1762)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1760)
:end_tab:

#################   WARNING   ################
# The below part is generated automatically through:
#    d2lbook build lib
# Don't edit it directly

def use_svg_display():
    """ʹ��svg��ʽ��Jupyter����ʾ��ͼ

    Defined in :numref:`sec_calculus`"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """����matplotlib��ͼ���С

    Defined in :numref:`sec_calculus`"""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """����matplotlib����

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """�������ݵ�

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # ���X��һ���ᣬ���True
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

class Timer:
    """��¼�������ʱ��"""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """������ʱ��"""
        self.tik = time.time()

    def stop(self):
        """ֹͣ��ʱ������ʱ���¼���б���"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """����ƽ��ʱ��"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """����ʱ���ܺ�"""
        return sum(self.times)

    def cumsum(self):
        """�����ۼ�ʱ��"""
        return np.array(self.times).cumsum().tolist()

def synthetic_data(w, b, num_examples):
    """����y=Xw+b+����

    Defined in :numref:`sec_linear_scratch`"""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))

def linreg(X, w, b):
    """���Իع�ģ��

    Defined in :numref:`sec_linear_scratch`"""
    return d2l.matmul(X, w) + b

def squared_loss(y_hat, y):
    """������ʧ

    Defined in :numref:`sec_linear_scratch`"""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    """С��������ݶ��½�

    Defined in :numref:`sec_linear_scratch`"""
    with paddle.no_grad():  # ����Paddle��ܵ�����,��ʹ��no_grad��Ҳ�����ֶ��޸�stop_gradient�����ƴ��ݶȲ�����inplace����,��bug���ύIssue:https://github.com/PaddlePaddle/Paddle/issues/38016
        for param in params:
            param.stop_gradient=True
            param.subtract_(lr * param.grad / batch_size)
            param.clear_grad()

def load_array(data_arrays, batch_size, is_train=True):
    """����һ��PyTorch���ݵ�����

    Defined in :numref:`sec_linear_concise`"""
    dataset = io.TensorDataset(data_arrays)
    return io.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

def get_fashion_mnist_labels(labels):
    """����Fashion-MNIST���ݼ����ı���ǩ

    Defined in :numref:`sec_fashion_mnist`"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """����ͼ���б�

    Defined in :numref:`sec_fashion_mnist`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(d2l.numpy(img))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def get_dataloader_workers():
    """ʹ��4����������ȡ����

    Defined in :numref:`sec_fashion_mnist`"""
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    """����Fashion-MNIST���ݼ���Ȼ������ص��ڴ���

    Defined in :numref:`sec_fashion_mnist`"""
    trans = [paddle.vision.transforms.ToTensor()]
    if resize:
        trans.append(paddle.vision.transforms.Resize(resize))
    trans = paddle.vision.transforms.transforms.Compose(trans)

    mnist_train = paddle.vision.datasets.FashionMNIST(mode='train', backend='cv2', transform=trans)
    mnist_test = paddle.vision.datasets.FashionMNIST(mode='test', backend='cv2', transform=trans)

    return (io.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                             num_workers=get_dataloader_workers()),
            io.DataLoader(mnist_test, batch_size=batch_size, shuffle=True,
                             num_workers=get_dataloader_workers()))

def accuracy(y_hat, y):
    """����Ԥ����ȷ������

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

def evaluate_accuracy(net, data_iter):
    """������ָ�����ݼ���ģ�͵ľ���

    Defined in :numref:`sec_softmax_scratch`"""
    metric = Accumulator(2)  # ��ȷԤ������Ԥ������
    with paddle.no_grad():
        for X, y in data_iter:
            X.stop_gradient=False
            y.stop_gradient=False
            metric.add(accuracy(net(X), y.T), y.size) #��ΪPaddle�����ݼ���ǩΪ������,��������Ҫת�ó�������
    return metric[0] / metric[1]

class Accumulator:
    """��n���������ۼ�"""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train_epoch_ch3(net, train_iter, loss, updater):
    """ѵ��ģ��һ���������ڣ��������3�£�

    Defined in :numref:`sec_softmax_scratch`"""
    # ѵ����ʧ�ܺ͡�ѵ��׼ȷ���ܺ͡�������
    metric = Accumulator(3)
    for X, y in train_iter:
        # �����ݶȲ����²���
        y_hat = net(X)
        W.stop_gradient=False
        b.stop_gradient=False
        l = loss(y_hat, y.cast('int32').T)
        if isinstance(updater, paddle.optimizer.Optimizer):
            # ʹ��Paddle���õ��Ż�������ʧ����
            updater.clear_grad()
            l.sum().backward()
            updater.step()
        else:
            # ʹ�ö��Ƶ��Ż�������ʧ����
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y.T), y.size)
    # ����ѵ����ʧ��ѵ������
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator:
    """�ڶ����л�������"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # �����ػ��ƶ�����
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # ʹ��lambda�����������
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # ��ͼ������Ӷ�����ݵ�
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """ѵ��ģ�ͣ��������3�£�

    Defined in :numref:`sec_softmax_scratch`"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):
    """Ԥ���ǩ���������3�£�

    Defined in :numref:`sec_softmax_scratch`"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])# Alias defined in config.ini

ones = paddle.ones
zeros = paddle.zeros
tensor = paddle.to_tensor
arange = paddle.arange
meshgrid = paddle.meshgrid
sin = paddle.sin
sinh = paddle.sinh
cos = paddle.cos
cosh = paddle.cosh
tanh = paddle.tanh
linspace = paddle.linspace
exp = paddle.exp
log = paddle.log
normal = paddle.normal
rand = paddle.rand
randn = paddle.randn
matmul = paddle.matmul
int32 = paddle.int32
float32 = paddle.float32
concat = paddle.concat
stack = paddle.stack
abs = paddle.abs
eye = paddle.eye
numpy = lambda x, *args, **kwargs: x.detach().numpy(*args, **kwargs)
size = lambda x, *args, **kwargs: x.shape(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.cast(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)


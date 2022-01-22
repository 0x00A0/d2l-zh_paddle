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
    trans = [vision.transforms.ToTensor()]
    if resize:
        trans.insert(0, vision.transforms.Resize(resize))
    trans = vision.transforms.transforms.Compose(trans)

    mnist_train = vision.datasets.FashionMNIST(mode='train', backend='cv2')
    mnist_test = vision.datasets.FashionMNIST(mode='test', backend='cv2')

    return (io.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                             num_workers=get_dataloader_workers()),
            io.DataLoader(mnist_test, batch_size=batch_size, shuffle=True,
                             num_workers=get_dataloader_workers()))# Alias defined in config.ini
nn_Module = nn.Module

ones = paddle.ones
zeros = paddle.zeros
to_tensor = paddle.to_tensor
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
size = lambda x, *args, **kwargs: x.size(*args, **kwargs)
reshape = lambda x, *args, **kwargs: x.reshape(*args, **kwargs)
to = lambda x, *args, **kwargs: x.to(*args, **kwargs)
reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
argmax = lambda x, *args, **kwargs: x.argmax(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
transpose = lambda x, *args, **kwargs: x.t(*args, **kwargs)
reduce_mean = lambda x, *args, **kwargs: x.mean(*args, **kwargs)


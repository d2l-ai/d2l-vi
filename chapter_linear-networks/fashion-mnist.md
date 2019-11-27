<<<<<<< HEAD
# Image Classification Data (Fashion-MNIST)

Before we implement softmax regression ourselves, let's pick a real dataset to work with. To make things visually compelling, we will pick an image classification dataset. The most commonly used image classification data set is the [MNIST](http://yann.lecun.com/exdb/mnist/) handwritten digit recognition data set, proposed by LeCun, Cortes and Burges in the 1990s. However, even simple models achieve classification accuracy over 95% on MNIST, so it is hard to spot the differences between better models and weaker ones. In order to get a better intuition, we will use the qualitatively similar, but comparatively complex [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset, proposed by [Xiao, Rasul and Vollgraf](https://arxiv.org/abs/1708.07747) in 2017.

## Getting the Data

First, import the packages or modules required in this section.
=======
# The Image Classification Dataset (Fashion-MNIST)
:label:`sec_fashion_mnist`

In :numref:`sec_naive_bayes`, we trained a naive Bayes classifier,
using the MNIST dataset introduced in 1998 :cite:`LeCun.Bottou.Bengio.ea.1998`. 
While MNIST had a good run as a benchmark dataset, 
even simple models by today's standards achieve classification accuracy over 95%.
making it unsuitable for distinguishing between stronger models and weaker ones. 
Today, MNIST serves as more of sanity checks than as a benchmark.
To up the ante just a bit, we will focus our discussion in the coming sections
on the qualitatively similar, but comparatively complex Fashion-MNIST 
dataset :cite:`Xiao.Rasul.Vollgraf.2017`, which was released in 2017.
>>>>>>> 1ec5c63... copy from d2l-en (#16)

```{.python .input}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
<<<<<<< HEAD
from mxnet.gluon import data as gdata
import sys
import time
```

Conveniently, Gluon's `data` package provides easy access 
to a number of benchmark datasets for testing our models. 
The first time we invoke `data.vision.FashionMNIST(train=True)` 
to collect the training data, 
Gluon will automatically retrieve the dataset via our Internet connection.
Subsequently, Gluon will use the already-downloaded local copy.
We specify whether we are requesting the training set or the test set 
by setting the value of the parameter `train` to `True` or `False`, respectively. 
Recall that we will only be using the training data for training,
holding out the test set for a final evaluation of our model.
=======
from mxnet import gluon
import sys

d2l.use_svg_display()
```

## Getting the Dataset

Just as with MNIST, Gluon makes it easy to download and load the FashionMNIST dataset into memory via the `FashionMNIST` class contained in `gluon.data.vision`.
We briefly work through the mechanics of loading and exploring the dataset below. 
Please refer to :numref:`sec_naive_bayes` for more details on loading data.
>>>>>>> 1ec5c63... copy from d2l-en (#16)

```{.python .input  n=23}
mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
```

FashionMNIST consists of images from 10 categories, each represented 
by 6k images in the training set and by 1k in the test set. 
Consequently the training set and the test set 
contain 60k and 10k images, respectively.

```{.python .input}
len(mnist_train), len(mnist_test)
```

<<<<<<< HEAD
We can access any example by indexing into the dataset using square brackets `[]`. In the following code, we access the image and label corresponding to the first example.

```{.python .input  n=24}
feature, label = mnist_train[0]
```

Our example, stored here in the variable `feature` corresponds to an image with a height and width of 28 pixels. Each pixel is an 8-bit unsigned integer (uint8) with values between 0 and 255. It is stored in a 3D NDArray. Its last dimension is the number of channels. Since the data set is a grayscale image, the number of channels is 1. When we encounter color, images, we'll have 3 channels for red, green, and blue. To keep things simple, we will record the shape of the image with the height and width of $h$ and $w$ pixels, respectively, as $h \times w$ or `(h, w)`.

```{.python .input}
feature.shape, feature.dtype
```

The label of each image is represented as a scalar in NumPy. Its type is a 32-bit integer.

```{.python .input}
label, type(label), label.dtype
```

There are 10 categories in Fashion-MNIST: t-shirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. The following function can convert a numeric label into a corresponding text label.

```{.python .input  n=25}
# This function has been saved in the d2l package for future use
=======
The images in Fashion-MNIST are associated with the following categories: 
t-shirt, trousers, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. 
The following function converts between numeric label indices and their names in text.

```{.python .input  n=25}
# Saved in the d2l package for later use
>>>>>>> 1ec5c63... copy from d2l-en (#16)
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
```

<<<<<<< HEAD
The following defines a function that can draw multiple images and corresponding labels in a single line.

```{.python .input}
# This function has been saved in the d2l package for future use
def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # Here _ means that we ignore (not use) variables
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
```

Next, let's take a look at the image contents and text labels for the first nine examples in the training data set.

```{.python .input  n=27}
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))
=======
We can now create a function to visualize these examples.

```{.python .input}
# Saved in the d2l package for later use
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
```

Here are the images and their corresponding labels (in text)
for the first few examples in the training dataset.

```{.python .input}
X, y = mnist_train[:18]
show_images(X.squeeze(axis=-1), 2, 9, titles=get_fashion_mnist_labels(y));
>>>>>>> 1ec5c63... copy from d2l-en (#16)
```

## Reading a Minibatch

<<<<<<< HEAD
To make our life easier when reading from the training and test sets we use a `DataLoader` rather than creating one from scratch, as we did in the section on ["Linear Regression Implementation Starting from Scratch"](linear-regression-scratch.md). Recall that a data loader reads a mini-batch of data with an example number of `batch_size` each time.

In practice, reading data can often be a significant performance bottleneck for training, especially when the model is simple or when the computer is fast. A handy feature of Gluon's `DataLoader` is the ability to use multiple processes to speed up data reading (not currently supported on Windows). For instance, we can set aside 4 processes to read the data (via `num_workers`).

In addition, we convert the image data from uint8 to 32-bit floating point numbers using the `ToTensor` class. Beyond that we divide all numbers by 255 so that all pixels have values between 0 and 1. The `ToTensor` class also moves the image channel from the last dimension to the first dimension to facilitate the convolutional neural network calculations introduced later. Through the `transform_first` function of the data set, we apply the transformation of `ToTensor` to the first element of each data example (image and label), i.e., the image.
=======
To make our life easier when reading from the training and test sets,
we use a `DataLoader` rather than creating one from scratch, 
as we did in :numref:`sec_linear_scratch`. 
Recall that at each iteration, a `DataLoader` 
reads a minibatch of data with size `batch_size` each time.

During training, reading data can be a significant performance bottleneck, 
especially when our model is simple or when our computer is fast. 
A handy feature of Gluon's `DataLoader` is the ability 
to use multiple processes to speed up data reading.
For instance, we can set aside 4 processes to read the data (via `num_workers`).
Because this feature is not currently supported on Windows
the following code checks the platform to make sure
that we do not saddle our Windows-using friends 
with error messages later on.

```{.python .input}
# Saved in the d2l package for later use
def get_dataloader_workers(num_workers=4):
    # 0 means no additional process is used to speed up the reading of data.
    if sys.platform.startswith('win'):
        return 0
    else:
        return num_workers
```

Below, we convert the image data from uint8 to 32-bit 
floating point numbers using the `ToTensor` class.
Additionally, the transformer will divide all numbers by 255 
so that all pixels have values between 0 and 1. 
The `ToTensor` class also moves the image channel 
from the last dimension to the first dimension 
to facilitate the convolutional neural network calculations introduced later. 
Through the `transform_first` function of the dataset, 
we apply the transformation of `ToTensor` 
to the first element of each instance (image and label).
>>>>>>> 1ec5c63... copy from d2l-en (#16)

```{.python .input  n=28}
batch_size = 256
transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    # 0 means no additional processes are needed to speed up the reading of
    # data
    num_workers = 0
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)
```

The logic that we will use to obtain and read the Fashion-MNIST data set is encapsulated in the `d2l.load_data_fashion_mnist` function, which we will use in later chapters. This function will return two variables, `train_iter` and `test_iter`. As the content of this book continues to deepen, we will further improve this function. Its full implementation will be described in the section ["Deep Convolutional Neural Networks (AlexNet)"](../chapter_convolutional-neural-networks/alexnet.md).

Let's look at the time it takes to read the training data.

```{.python .input}
start = time.time()
for X, y in train_iter:
    continue
<<<<<<< HEAD
'%.2f sec' % (time.time() - start)
=======
'%.2f sec' % timer.stop()
```

## Putting All Things Together 

Now we define the `load_data_fashion_mnist` function 
that obtains and reads the Fashion-MNIST dataset. 
It returns the data iterators for both the training set and validation set. 
In addition, it accepts an optional argument to resize images to another shape.

```{.python .input  n=4}
# Saved in the d2l package for later use
def load_data_fashion_mnist(batch_size, resize=None):
    """Download the Fashion-MNIST dataset and then load into memory."""
    dataset = gluon.data.vision
    trans = [dataset.transforms.Resize(resize)] if resize else []
    trans.append(dataset.transforms.ToTensor())
    trans = dataset.transforms.Compose(trans)
    mnist_train = dataset.FashionMNIST(train=True).transform_first(trans)
    mnist_test = dataset.FashionMNIST(train=False).transform_first(trans)
    return (gluon.data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=get_dataloader_workers()),
            gluon.data.DataLoader(mnist_test, batch_size, shuffle=False,
                                  num_workers=get_dataloader_workers()))
```

Below, we verify that image resizing works.

```{.python .input  n=5}
train_iter, test_iter = load_data_fashion_mnist(32, (64, 64))
for X, y in train_iter:
    print(X.shape)
    break
>>>>>>> 1ec5c63... copy from d2l-en (#16)
```

We are now ready to work with the FashionMNIST dataset in the sections that follow.

## Summary

* Fashion-MNIST is an apparel classification dataset consisting of images representing 10 categories. 
 * We will use this dataset in subsequent sections and chapters to evaluate various classification algorithms.
* We store the shape of each image with height $h$ width $w$ pixels as $h \times w$ or `(h, w)`.
* Data iterators are a key component for efficient performance. Rely on well-implemented iterators that exploit multi-threading to avoid slowing down your training loop.

## Exercises

1. Does reducing the `batch_size` (for instance, to 1) affect read performance?
1. For non-Windows users, try modifying `num_workers` to see how it affects read performance. Plot the performance against the number of works employed.
1. Use the MXNet documentation to see which other datasets are available in `mxnet.gluon.data.vision`.
1. Use the MXNet documentation to see which other transformations are available in `mxnet.gluon.data.vision.transforms`.

## [Discussions](https://discuss.mxnet.io/t/2335)

![](../img/qr_fashion-mnist.svg)

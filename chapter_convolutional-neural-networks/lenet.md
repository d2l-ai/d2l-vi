# Convolutional Neural Networks (LeNet)
<<<<<<< HEAD

In our first encounter with image data we applied a [Multilayer Perceptron](../chapter_deep-learning-basics/mlp-scratch.md) to pictures of clothing in the Fashion-MNIST data set. Both the height and width of each image were 28 pixels. We expanded the pixels in the image line by line to get a vector of length 784, and then used them as inputs to the fully connected layer. However, this classification method has certain limitations:

1. The adjacent pixels in the same column of an image may be far apart in this vector. The patterns they create may be difficult for the model to recognize. In fact, the vectorial representation ignores position entirely - we could have permuted all $28 \times 28$ pixels at random and obtained the same results.
2. For large input images, using a fully connected layer can easily cause the model to become too large, as we discussed previously.

As discussed in the previous sections, the convolutional layer attempts to solve both problems. On the one hand, the convolutional layer retains the input shape, so that the correlation of image pixels in the directions of both height and width can be recognized effectively. On the other hand, the convolutional layer repeatedly calculates the same kernel and the input of different positions through the sliding window, thereby avoiding excessively large parameter sizes.

A convolutional neural network is a network with convolutional layers. In this section, we will introduce an early convolutional neural network used to recognize handwritten digits in images - [LeNet5](http://yann.lecun.com/exdb/lenet/). Convolutional networks were invented by Yann LeCun and coworkers at AT&T Bell Labs in the early 90s. LeNet showed that it was possible to use gradient descent to train the convolutional neural network for handwritten digit recognition. It achieved outstanding results at the time (only matched by Support Vector Machines at the time).
=======
:label:`sec_lenet`

We are now ready to put all of the tools together
to deploy your first fully-functional convolutional neural network.
In our first encounter with image data we applied a multilayer perceptron (:numref:`sec_mlp_scratch`)
to pictures of clothing in the Fashion-MNIST dataset.
Each image in Fashion-MNIST consisted of
a two-dimensional $28 \times 28$ matrix.
To make this data amenable to multilayer perceptrons
which anticipate receiving inputs as one-dimensional fixed-length vectors,
we first flattened each image, yielding vectors of length 784,
before processing them with a series of fully-connected layers.

Now that we have introduced convolutional layers,
we can keep the image in its original spatially-organized grid,
processing it with a series of successive convolutional layers.
Moreover, because we are using convolutional layers,
we can enjoy a considerable savings in the number of parameters required.

In this section, we will introduce one of the first
published convolutional neural networks
whose benefit was first demonstrated by Yann Lecun,
then a researcher at AT&T Bell Labs,
for the purpose of recognizing handwritten digits in images—[LeNet5](http://yann.lecun.com/exdb/lenet/).
In the 90s, their experiments with LeNet gave the first compelling evidence
that it was possible to train convolutional neural networks
by backpropagation.
Their model achieved outstanding results at the time
(only matched by Support Vector Machines at the time)
and was adopted to recognize digits for processing deposits in ATM machines.
Some ATMs still run the code
that Yann and his colleague Leon Bottou wrote in the 1990s!
>>>>>>> 1ec5c63... copy from d2l-en (#16)

## LeNet

LeNet is divided into two parts: a block of convolutional layers and one of fully connected ones. Below, we will introduce these two modules separately. Before going into details, let's briefly review the model in pictures. To illustrate the issue of channels and the specific layers we will use a rather description (later we will see how to convey the same information more concisely).

![Data flow in LeNet 5. The input is a handwritten digit, the output a probabilitiy over 10 possible outcomes.](../img/lenet.svg)
<<<<<<< HEAD

The basic units in the convolutional block are a convolutional layer and a subsequent average pooling layer (note that max-pooling works better, but it had not been invented in the 90s yet). The convolutional layer is used to recognize the spatial patterns in the image, such as lines and the parts of objects, and the subsequent average pooling layer is used to reduce the dimensionality. The convolutional layer block is composed of repeated stacks of these two basic units. In the convolutional layer block, each convolutional layer uses a $5\times 5$ window and a sigmoid activation function for the output (note that ReLU works better, but it had not been invented in the 90s yet). The number of output channels for the first convolutional layer is 6, and the number of output channels for the second convolutional layer is increased to 16. This is because the height and width of the input of the second convolutional layer is smaller than that of the first convolutional layer. Therefore, increasing the number of output channels makes the parameter sizes of the two convolutional layers similar. The window shape for the two average pooling layers of the convolutional layer block is $2\times 2$ and the stride is 2. Because the pooling window has the same shape as the stride, the areas covered by the pooling window sliding on each input do not overlap. In other words, the pooling layer performs downsampling.

The output shape of the convolutional layer block is (batch size, channel, height, width). When the output of the convolutional layer block is passed into the fully connected layer block, the fully connected layer block flattens each example in the mini-batch. That is to say, the input shape of the fully connected layer will become two dimensional: the first dimension is the example in the mini-batch, the second dimension is the vector representation after each example is flattened, and the vector length is the product of channel, height, and width.  The fully connected layer block has three fully connected layers. They have 120, 84, and 10 outputs, respectively. Here, 10 is the number of output classes.

Next, we implement the LeNet model through the Sequential class.
=======
:label:`img_lenet`

The basic units in the convolutional block are a convolutional layer
and a subsequent average pooling layer
(note that max-pooling works better,
but it had not been invented in the 90s yet).
The convolutional layer is used to recognize
the spatial patterns in the image,
such as lines and the parts of objects,
and the subsequent average pooling layer
is used to reduce the dimensionality.
The convolutional layer block is composed of
repeated stacks of these two basic units.
Each convolutional layer uses a $5\times 5$ kernel
and processes each output with a sigmoid activation function
(again, note that ReLUs are now known to work more reliably,
but had not been invented yet).
The first convolutional layer has 6 output channels,
and second convolutional layer increases channel depth further to 16.

However, coinciding with this increase in the number of channels,
the height and width are shrunk considerably.
Therefore, increasing the number of output channels
makes the parameter sizes of the two convolutional layers similar.
The two average pooling layers are of size $2\times 2$ and take stride 2
(note that this means they are non-overlapping).
In other words, the pooling layer downsamples the representation
to be precisely *one quarter* the pre-pooling size.

The convolutional block emits an output with size given by
(batch size, channel, height, width).
Before we can pass the convolutional block's output
to the fully-connected block, we must flatten
each example in the minibatch.
In other words, we take this 4D input and transform it into the 2D
input expected by fully-connected layers:
as a reminder, the first dimension indexes the examples in the minibatch
and the second gives the flat vector representation of each example.
LeNet's fully-connected layer block has three fully-connected layers,
with 120, 84, and 10 outputs, respectively.
Because we are still performing classification,
the 10 dimensional output layer corresponds
to the number of possible output classes.

While getting to the point
where you truly understand
what is going on inside LeNet
may have taken a bit of work,
you can see below that implementing it
in a modern deep learning library
is remarkably simple.
Again, we will rely on the Sequential class.
>>>>>>> 1ec5c63... copy from d2l-en (#16)

```{.python .input}
import sys
sys.path.insert(0, '..')

import d2l
<<<<<<< HEAD
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time
=======
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
>>>>>>> 1ec5c63... copy from d2l-en (#16)

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        # Dense will transform the input of the shape (batch size, channel,
        # height, width) into the input of the shape (batch size,
        # channel * height * width) automatically by default
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))
```

<<<<<<< HEAD
We took the liberty of replacing the Gaussian activation in the last layer by a regular dense network since this is rather much more convenient to train. Other than that the network matches the historical definition of LeNet5. Next, we feed a single-channel example of size $28 \times 28$ into the network and perform a forward computation layer by layer to see the output shape of each layer.
=======
As compared to the original network,
we took the liberty of replacing
the Gaussian activation in the last layer
by a regular dense layer, which tends to be
significantly more convenient to train.
Other than that, this network matches
the historical definition of LeNet5.

Next, let's take a look of an example.
As shown in :numref:`img_lenet_vert`, we feed 
a single-channel example
of size $28 \times 28$ into the network
and perform a forward computation layer by layer
printing the output shape at each layer
to make sure we understand what is happening here.
>>>>>>> 1ec5c63... copy from d2l-en (#16)

```{.python .input}
X = np.random.uniform(size=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

We can see that the height and width of the input in the convolutional layer block is reduced, layer by layer. The convolutional layer uses a kernel with a height and width of 5 to reduce the height and width by 4, while the pooling layer halves the height and width, but the number of channels increases from 1 to 16. The fully connected layer reduces the number of outputs layer by layer, until the number of image classes becomes 10.

![Compressed notation for LeNet5](../img/lenet-vert.svg)
:label:`img_lenet_vert`

## Data Acquisition and Training

<<<<<<< HEAD
Now, we will experiment with the LeNet model. We still use Fashion-MNIST as the training data set since the problem is rather more difficult than OCR (even in the 1990s the error rates were in the 1% range).
=======
Now that we have implemented the model,
we might as well run some experiments
to see what we can accomplish with the LeNet model.
While it might serve nostalgia
to train LeNet on the original MNIST dataset,
that dataset has become too easy,
with MLPs getting over 98% accuracy,
so it would be hard to see the benefits of convolutional networks.
Thus we will stick with Fashion-MNIST as our dataset
because while it has the same shape ($28\times28$ images),
this dataset is notably more challenging.
>>>>>>> 1ec5c63... copy from d2l-en (#16)

```{.python .input}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
```

Since convolutional networks are significantly more expensive to compute than multilayer perceptrons we recommend using GPUs to speed up training. Time to introduce a convenience function that allows us to detect whether we have a GPU: it works by trying to allocate an NDArray on `gpu(0)`, and use `gpu(0)` if this is successful. Otherwise, we catch the resulting exception and we stick with the CPU.

```{.python .input}
# This function has been saved in the d2l package for future use
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()
ctx
```

<<<<<<< HEAD
Accordingly, we slightly modify the `evaluate_accuracy` function described when [implementing the SoftMax from scratch](../chapter_deep-learning-basics/softmax-regression-scratch.md).  Since the data arrives in the CPU when loading we need to copy it to the GPU before any computation can occur. This is accomplished via the `as_in_context` function described in the [GPU Computing](../chapter_deep-learning-computation/use-gpu.md) section. Note that we accumulate the errors on the same device as where the data eventually lives (in `acc`). This avoids intermediate copy operations that would destroy performance.

```{.python .input}
# This function has been saved in the d2l package for future use. The function
# will be gradually improved. Its complete implementation will be discussed in
# the "Image Augmentation" section
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
=======
For evaluation, we need to make a slight modification
to the `evaluate_accuracy` function that we described
in :numref:`sec_softmax_scratch`.
Since the full dataset lives on the CPU,
we need to copy it to the GPU before we can compute our models.
This is accomplished via the `as_in_context` function
described in :numref:`sec_use_gpu`.

```{.python .input}
# Saved in the d2l package for later use
def evaluate_accuracy_gpu(net, data_iter, ctx=None):
    if not ctx:  # Query the first device the first parameter is on
        ctx = list(net.collect_params().values())[0].list_ctx()[0]
    metric = d2l.Accumulator(2)  # num_corrected_examples, num_examples
>>>>>>> 1ec5c63... copy from d2l-en (#16)
    for X, y in data_iter:
        # If ctx is the GPU, copy the data to the GPU.
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n
```

<<<<<<< HEAD
Just like the data loader we need to update the training function to deal with GPUs. Unlike [`train_ch3`](../chapter_deep-learning-basics/softmax-regression-scratch.md) we now move data prior to computation.

```{.python .input}
# This function has been saved in the d2l package for future use
def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
=======
We also need to update our training function to deal with GPUs.
Unlike the `train_epoch_ch3` defined in :numref:`sec_softmax_scratch`, we now need to move each batch of data to our designated context (hopefully, the GPU)
prior to making the forward and backward passes.

The training function `train_ch5` is also very similar to `train_ch3` defined in :numref:`sec_softmax_scratch`. Since we will deal with networks with tens of layers now, the function will only support Gluon models. We initialize the model parameters on the device indicated by `ctx`,
this time using the Xavier initializer.
The loss function and the training algorithm
still use the cross-entropy loss function
and minibatch stochastic gradient descent. Since each epoch takes tens of second to run, we visualize the training loss in a finer granularity.

```{.python .input}
# Saved in the d2l package for later use
def train_ch5(net, train_iter, test_iter, num_epochs, lr, ctx=d2l.try_gpu()):
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(),
                            'sgd', {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer = d2l.Timer()
>>>>>>> 1ec5c63... copy from d2l-en (#16)
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
<<<<<<< HEAD
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))
=======
            trainer.step(X.shape[0])
            metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_loss, train_acc = metric[0]/metric[2], metric[1]/metric[2]
            if (i+1) % 50 == 0:
                animator.add(epoch + i/len(train_iter),
                             (train_loss, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch+1, (None, None, test_acc))
    print('loss %.3f, train acc %.3f, test acc %.3f' % (
        train_loss, train_acc, test_acc))
    print('%.1f exampes/sec on %s' % (metric[2]*num_epochs/timer.sum(), ctx))
>>>>>>> 1ec5c63... copy from d2l-en (#16)
```

We initialize the model parameters on the device indicated by `ctx`, this time using Xavier. The loss function and the training algorithm still use the cross-entropy loss function and mini-batch stochastic gradient descent.

```{.python .input}
lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* A convolutional neural network (in short, ConvNet) is a network using convolutional layers.
* In a ConvNet we alternate between convolutions, nonlinearities and often also pooling operations.
* Ultimately the resolution is reduced prior to emitting an output via one (or more) dense layers.
* LeNet was the first successful deployment of such a network.

## Exercises

1. Replace the average pooling with max pooling. What happens?
1. Try to construct a more complex network based on LeNet to improve its accuracy.
    * Adjust the convolution window size.
    * Adjust the number of output channels.
    * Adjust the activation function (ReLU?).
    * Adjust the number of convolution layers.
    * Adjust the number of fully connected layers.
    * Adjust the learning rates and other training details (initialization, epochs, etc.)
1. Try out the improved network on the original MNIST dataset.
1. Display the activations of the first and second layer of LeNet for different inputs (e.g., sweaters, coats).


<<<<<<< HEAD
## References

[1] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2353)
=======
## [Discussions](https://discuss.mxnet.io/t/2353)
>>>>>>> 1ec5c63... copy from d2l-en (#16)

![](../img/qr_lenet.svg)

<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Implementation of Multilayer Perceptron from Scratch
-->

# *dịch tiêu đề phía trên*
:label:`sec_mlp_scratch`

<!--
Now that we have characterized multilayer perceptrons (MLPs) mathematically, let's try to implement one ourselves.
-->

*dịch đoạn phía trên*

```{.python .input  n=9}
import d2l
from mxnet import gluon, np, npx
npx.set_np()
```

<!--
To compare against our previous results achieved with (linear) softmax regression (:numref:`sec_softmax_scratch`), 
we will continue work with the Fashion-MNIST image classification dataset (:numref:`sec_fashion_mnist`).
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

<!--
## Initializing Model Parameters
-->

## *dịch tiêu đề phía trên*

<!--
Recall that Fashion-MNIST contains $10$ classes, and that each image consists of a $28 \times 28 = 784$ grid of (black and white) pixel values.
Again, we will disregard the spatial structure among the pixels (for now), so we can think of this as simply a classification dataset with $784$ input features and $10$ classes.
To begin, we will implement an MLP with one hidden layer and $256$ hidden units.
Note that we can regard both of these quantities as *hyperparameters* and ought in general to set them based on performance on validation data.
Typically, we choose layer widths in powers of $2$ which tends to be computationally efficient because of how memory is alotted and addressed in hardware.
-->

*dịch đoạn phía trên*

<!--
Again, we will represent our parameters with several `ndarray`s.
Note that *for every layer*, we must keep track of one weight matrix and one bias vector.
As always, we call `attach_grad` to allocate memory for the gradients (of the loss) with respect to these parameters.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Activation Function
-->

## *dịch tiêu đề phía trên*

<!--
To make sure we know how everything works, we will implement the ReLU activation ourselves using the `maximum` function rather than invoking `npx.relu` directly.
-->

*dịch đoạn phía trên*

```{.python .input  n=4}
def relu(X):
    return np.maximum(X, 0)
```

<!--
## The model
-->

## *dịch tiêu đề phía trên*

<!--
Because we are disregarding spatial structure, we `reshape` each 2D image into a flat vector of length  `num_inputs`.
Finally, we implement our model with just a few lines of code.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## The Loss Function
-->

## *dịch tiêu đề phía trên*

<!--
To ensure numerical stability (and because we already implemented the softmax function from scratch (:numref:`sec_softmax_scratch`), 
we leverage Gluon's integrated function for calculating the softmax and cross-entropy loss.
Recall our easlier discussion of these intricacies (:numref:`sec_mlp`).
We encourage the interested reader to examine the source code for `mxnet.gluon.loss.SoftmaxCrossEntropyLoss` to deepen their knowledge of implementation details.
-->

*dịch đoạn phía trên*

```{.python .input  n=6}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Training
-->

## *dịch tiêu đề phía trên*

<!--
Fortunately, the training loop for MLPs is exactly the same as for softmax regression.
Leveraging the `d2l` package again, we call the `train_ch3` function (see :numref:`sec_softmax_scratch`), setting the number of epochs to $10$ and the learning rate to $0.5$.
-->

*dịch đoạn phía trên*

```{.python .input  n=7}
num_epochs, lr = 10, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

<!--
To evaluate the learned model, we apply it on some test data.
-->

*dịch đoạn phía trên*

```{.python .input}
d2l.predict_ch3(net, test_iter)
```

<!--
This looks a bit better than our previous result, using simple linear models and gives us some signal that we are on the right path.
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## Tóm tắt

<!--
We saw that implementing a simple MLP is easy, even when done manually.
That said, with a large number of layers, this can still get messy (e.g., naming and keeping track of our model's parameters, etc).
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Change the value of the hyperparameter `num_hiddens` and see how this hyperparameter influences your results. Determine the best value of this hyperparameter, keeping all others constant.
2. Try adding an additional hidden layer to see how it affects the results.
3. How does changing the learning rate alter your results? Fixing the model architecture and other hyperparameters (including number of epochs), what learning rate gives you the best results?
4. What is the best result you can get by optimizing over all the parameters (learning rate, iterations, number of hidden layers, number of hidden units per layer) jointly?
5. Describe why it is much more challenging to deal with multiple hyperparameters.
6. What is the smartest strategy you can think of for structuring a search over multiple hyperparameters?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2339)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2339)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

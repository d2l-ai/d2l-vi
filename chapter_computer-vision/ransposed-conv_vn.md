<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Transposed Convolution
-->

# *dịch tiêu đề phía trên*
:label:`sec_transposed_conv`


<!--
The layers we introduced so far for convolutional neural networks, 
including convolutional layers (:numref:`sec_conv_layer`) and pooling layers (:numref:`sec_pooling`), 
often reduce the input width and height, or keep them unchanged.
Applications such as semantic segmentation (:numref:`sec_semantic_segmentation`) and generative adversarial networks (:numref:`sec_dcgan`), 
however, require to predict values for each pixel and therefore needs to increase input width and height. 
Transposed convolution, also named fractionally-strided convolution :cite:`Dumoulin.Visin.2016` or deconvolution :cite:`Long.Shelhamer.Darrell.2015`, serves this purpose.
-->

*dịch đoạn phía trên*



```{.python .input  n=13}
from mxnet import np, npx, init
from mxnet.gluon import nn
from d2l import mxnet as d2l

npx.set_np()
```


<!--
## Basic 2D Transposed Convolution
-->

## *dịch tiêu đề phía trên*


<!--
Let us consider a basic case that both input and output channels are 1, with 0 padding and 1 stride.
:numref:`fig_trans_conv` illustrates how transposed convolution with a $2\times 2$ kernel is computed on the $2\times 2$ input matrix.
-->

*dịch đoạn phía trên*


<!--
![Transposed convolution layer with a $2\times 2$ kernel.](../img/trans_conv.svg)
-->


![*dịch mô tả phía trên*](../img/trans_conv.svg)
:label:`fig_trans_conv`


<!--
We can implement this operation by giving matrix kernel $K$ and matrix input $X$.
-->

*dịch đoạn phía trên*



```{.python .input}
def trans_conv(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y
```


<!--
Remember the convolution computes results by `Y[i, j] = (X[i: i + h, j: j + w] * K).sum()` (refer to `corr2d` in :numref:`sec_conv_layer`), which summarizes input values through the kernel.
While the transposed convolution broadcasts input values through the kernel, which results in a larger output shape.
-->

*dịch đoạn phía trên*


<!--
Verify the results in :numref:`fig_trans_conv`.
-->

*dịch đoạn phía trên*


```{.python .input}
X = np.array([[0, 1], [2, 3]])
K = np.array([[0, 1], [2, 3]])
trans_conv(X, K)
```


<!--
Or we can use `nn.Conv2DTranspose` to obtain the same results.
As `nn.Conv2D`, both input and kernel should be 4-D tensors.
-->

*dịch đoạn phía trên*


```{.python .input  n=17}
X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.Conv2DTranspose(1, kernel_size=2)
tconv.initialize(init.Constant(K))
tconv(X)
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Padding, Strides, and Channels
-->

# *dịch tiêu đề phía trên*


<!--
We apply padding elements to the input in convolution, while they are applied to the output in transposed convolution.
A $1\times 1$ padding means we first compute the output as normal, then remove the first/last rows and columns.
-->

*dịch đoạn phía trên*


```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, padding=1)
tconv.initialize(init.Constant(K))
tconv(X)
```


<!--
Similarly, strides are applied to outputs as well.
-->

*dịch đoạn phía trên*


```{.python .input}
tconv = nn.Conv2DTranspose(1, kernel_size=2, strides=2)
tconv.initialize(init.Constant(K))
tconv(X)
```


<!--
The multi-channel extension of the transposed convolution is the same as the convolution.
When the input has multiple channels, denoted by $c_i$, the transposed convolution assigns a $k_h\times k_w$ kernel matrix to each input channel.
If the output has a channel size $c_o$, then we have a $c_i\times k_h\times k_w$ kernel for each output channel.
-->

*dịch đoạn phía trên*



<!--
As a result, if we feed $X$ into a convolutional layer $f$ to compute $Y=f(X)$ and create a transposed convolution layer $g$ with 
the same hyperparameters as $f$ except for the output channel set to be the channel size of $X$, then $g(Y)$ should has the same shape as $X$. 
Let us verify this statement.
-->

*dịch đoạn phía trên*


```{.python .input}
X = np.random.uniform(size=(1, 10, 16, 16))
conv = nn.Conv2D(20, kernel_size=5, padding=2, strides=3)
tconv = nn.Conv2DTranspose(10, kernel_size=5, padding=2, strides=3)
conv.initialize()
tconv.initialize()
tconv(conv(X)).shape == X.shape
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Analogy to Matrix Transposition
-->

## *dịch tiêu đề phía trên*


<!--
The transposed convolution takes its name from the matrix transposition.
In fact, convolution operations can also be achieved by matrix multiplication.
In the example below, we define a $3\times$ input $X$ with a $2\times 2$ kernel $K$, and then use `corr2d` to compute the convolution output.
-->

*dịch đoạn phía trên*


```{.python .input}
X = np.arange(9).reshape(3, 3)
K = np.array([[0, 1], [2, 3]])
Y = d2l.corr2d(X, K)
Y
```


<!--
Next, we rewrite convolution kernel $K$ as a matrix $W$.
Its shape will be $(4, 9)$, where the $i^\mathrm{th}$ row present applying the kernel to the input to generate the $i^\mathrm{th}$ output element.
-->

*dịch đoạn phía trên*


```{.python .input}
def kernel2matrix(K):
    k, W = np.zeros(5), np.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W

W = kernel2matrix(K)
W
```


<!--
Then the convolution operator can be implemented by matrix multiplication with proper reshaping.
-->

*dịch đoạn phía trên*


```{.python .input}
Y == np.dot(W, X.reshape(-1)).reshape(2, 2)
```


<!--
We can implement transposed convolution as a matrix multiplication as well by reusing `kernel2matrix`.
To reuse the generated $W$, we construct a $2\times 2$ input, so the corresponding weight matrix will have a shape $(9, 4)$, which is $W^\top$. Let us verify the results.
-->

*dịch đoạn phía trên*


```{.python .input}
X = np.array([[0, 1], [2, 3]])
Y = trans_conv(X, K)
Y == np.dot(W.T, X.reshape(-1)).reshape(3, 3)
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

## Tóm tắt


<!--
* Compared to convolutions that reduce inputs through kernels, transposed convolutions broadcast inputs.
* If a convolution layer reduces the input width and height by $n_w$ and $h_h$ time, respectively.
Then a transposed convolution layer with the same kernel sizes, padding and strides will increase the input width and height by $n_w$ and $n_h$, respectively.
* We can implement convolution operations by the matrix multiplication, the corresponding transposed convolutions can be done by transposed matrix multiplication.
-->

*dịch đoạn phía trên*


## Bài tập


<!--
Is it efficient to use matrix multiplication to implement convolution operations? Why?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/376)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* 

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 



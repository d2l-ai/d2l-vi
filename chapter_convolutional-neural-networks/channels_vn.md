<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Multiple Input and Output Channels
-->

# *dịch tiêu đề phía trên*
:label:`sec_channels`

<!--
While we have described the multiple channels that comprise each image (e.g., color images have the standard RGB channels to indicate the amount of red, green and blue), 
until now, we simplified all of our numerical examples by working with just a single input and a single output channel.
This has allowed us to think of our inputs, convolutional kernels, and outputs each as two-dimensional arrays.
-->

*dịch đoạn phía trên*

<!--
When we add channels into the mix, our inputs and hidden representations both become three-dimensional arrays.
For example, each RGB input image has shape $3\times h\times w$.
We refer to this axis, with a size of 3, as the channel dimension.
In this section, we will take a deeper look at convolution kernels with multiple input and multiple output channels.
-->

*dịch đoạn phía trên*

<!--
## Multiple Input Channels
-->

## *dịch tiêu đề phía trên*

<!--
When the input data contains multiple channels, we need to construct a convolution kernel with the same number of input channels as the input data, so that it can perform cross-correlation with the input data.
Assuming that the number of channels for the input data is $c_i$, the number of input channels of the convolution kernel also needs to be $c_i$. 
If our convolution kernel's window shape is $k_h\times k_w$, then when $c_i=1$, we can think of our convolution kernel as just a two-dimensional array of shape $k_h\times k_w$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
However, when $c_i>1$, we need a kernel that contains an array of shape $k_h\times k_w$ *for each input channel*. 
Concatenating these $c_i$ arrays together yields a convolution kernel of shape $c_i\times k_h\times k_w$.
Since the input and convolution kernel each have $c_i$ channels, we can perform a cross-correlation operation 
on the two-dimensional array of the input and the two-dimensional kernel array of the convolution kernel for each channel, 
adding the $c_i$ results together (summing over the channels) to yield a two-dimensional array.
This is the result of a two-dimensional cross-correlation between multi-channel input data and a *multi-input channel* convolution kernel.
-->

*dịch đoạn phía trên*

<!--
In :numref:`fig_conv_multi_in`, we demonstrate an example of a two-dimensional cross-correlation with two input channels.
The shaded portions are the first output element as well as the input and kernel array elements used in its computation:
$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$.
-->

*dịch đoạn phía trên*

<!--
![Cross-correlation computation with 2 input channels. The shaded portions are the first output element as well as the input and kernel array elements used in its computation: $(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$. ](../img/conv-multi-in.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/conv-multi-in.svg)
:label:`fig_conv_multi_in`


<!--
To make sure we really understand what is going on here, we can implement cross-correlation operations with multiple input channels ourselves.
Notice that all we are doing is performing one cross-correlation operation per channel and then adding up the results using the `add_n` function.
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
import d2l
from mxnet import np, npx
npx.set_np()

def corr2d_multi_in(X, K):
    # First, traverse along the 0th dimension (channel dimension) of X and K.
    # Then, add them together by using * to turn the result list into a
    # positional argument of the add_n function
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))
```

<!--
We can construct the input array `X` and the kernel array `K` corresponding to the values in the above diagram to validate the output of the cross-correlation operation.
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
X = np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = np.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

corr2d_multi_in(X, K)
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Multiple Output Channels
-->

## *dịch tiêu đề phía trên*

<!--
Regardless of the number of input channels, so far we always ended up with one output channel.
However, as we discussed earlier, it turns out to be essential to have multiple channels at each layer.
In the most popular neural network architectures, we actually increase the channel dimension as we go higher up in the neural network, 
typically downsampling to trade off spatial resolution for greater *channel depth*.
Intuitively, you could think of each channel as responding to some different set of features.
Reality is a bit more complicated than the most naive interpretations of this intuition since representations are not learned independent but are rather optimized to be jointly useful.
So it may not be that a single channel learns an edge detector but rather that some direction in channel space corresponds to detecting edges.
-->

*dịch đoạn phía trên*


<!--
Denote by $c_i$ and $c_o$ the number of input and output channels, respectively, and let $k_h$ and $k_w$ be the height and width of the kernel.
To get an output with multiple channels, we can create a kernel array of shape $c_i\times k_h\times k_w$ for each output channel.
We concatenate them on the output channel dimension, so that the shape of the convolution kernel is $c_o\times c_i\times k_h\times k_w$.
In cross-correlation operations, the result on each output channel is calculated from the convolution kernel corresponding to that output channel and takes input from all channels in the input array.
-->

*dịch đoạn phía trên*

<!--
We implement a cross-correlation function to calculate the output of multiple channels as shown below.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
def corr2d_multi_in_out(X, K):
    # Traverse along the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are merged
    # together using the stack function
    return np.stack([corr2d_multi_in(X, k) for k in K])
```

<!--
We construct a convolution kernel with 3 output channels by concatenating the kernel array `K` with `K+1` (plus one for each element in `K`) and `K+2`.
-->

*dịch đoạn phía trên*

```{.python .input  n=4}
K = np.stack((K, K + 1, K + 2))
K.shape
```

<!--
Below, we perform cross-correlation operations on the input array `X` with the kernel array `K`.
Now the output contains 3 channels.
The result of the first channel is consistent with the result of the previous input array `X` and the multi-input channel, single-output channel kernel.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
corr2d_multi_in_out(X, K)
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## $1\times 1$ Convolutional Layer
-->

## *dịch tiêu đề phía trên*

<!--
At first, a $1 \times 1$ convolution, i.e., $k_h = k_w = 1$, does not seem to make much sense.
After all, a convolution correlates adjacent pixels.
A $1 \times 1$ convolution obviously does not.
Nonetheless, they are popular operations that are sometimes included in the designs of complex deep networks.
Let us see in some detail what it actually does.
-->

*dịch đoạn phía trên*

<!--
Because the minimum window is used, the $1\times 1$ convolution loses the ability of larger convolutional layers to recognize patterns 
consisting of interactions among adjacent elements in the height and width dimensions.
The only computation of the $1\times 1$ convolution occurs on the channel dimension.
-->

*dịch đoạn phía trên*

<!--
:numref:`fig_conv_1x1` shows the cross-correlation computation using the $1\times 1$ convolution kernel with 3 input channels and 2 output channels.
Note that the inputs and outputs have the same height and width.
Each element in the output is derived from a linear combination of elements *at the same position* in the input image.
You could think of the $1\times 1$ convolutional layer as constituting a fully-connected layer applied at every single pixel location 
to transform the $c_i$ corresponding input values into $c_o$ output values.
Because this is still a convolutional layer, the weights are tied across pixel location
Thus the $1\times 1$ convolutional layer requires $c_o\times c_i$ weights (plus the bias terms).
-->

*dịch đoạn phía trên*


<!--
![The cross-correlation computation uses the $1\times 1$ convolution kernel with 3 input channels and 2 output channels. The inputs and outputs have the same height and width. ](../img/conv-1x1.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/conv-1x1.svg)
:label:`fig_conv_1x1`

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
Let us check whether this works in practice: we implement the $1 \times 1$ convolution using a fully-connected layer.
The only thing is that we need to make some adjustments to the data shape before and after the matrix multiplication.
-->

Hãy kiểm tra xem liệu nó có hoạt động trong thực tế: Ta sẽ lập trình một phép tích chập $1 \times 1$ sử dụng một tầng kết nối đầy đủ.
Vấn đề duy nhất là ta cần phải điều chỉnh kích thước dữ liệu trước và sau phép nhân ma trận.

```{.python .input  n=6}
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape(c_i, h * w)
    K = K.reshape(c_o, c_i)
    Y = np.dot(K, X)  # Matrix multiplication in the fully connected layer
    return Y.reshape(c_o, h, w)
```

<!--
When performing $1\times 1$ convolution, the above function is equivalent to the previously implemented cross-correlation function `corr2d_multi_in_out`.
Let us check this with some reference data.
-->

Khi thực hiện phép tích chập $1\times 1$, hàm bên trên tương đương với hàm tương quan chéo đã được lập trình ở `corr2d_multi_in_out`.

```{.python .input  n=7}
X = np.random.uniform(size=(3, 3, 3))
K = np.random.uniform(size=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

np.abs(Y1 - Y2).sum() < 1e-6
```

<!--
## Summary
-->

## Tóm tắt

<!--
* Multiple channels can be used to extend the model parameters of the convolutional layer.
* The $1\times 1$ convolutional layer is equivalent to the fully-connected layer, when applied on a per pixel basis.
* The $1\times 1$ convolutional layer is typically used to adjust the number of channels between network layers and to control model complexity.
-->

* Ta có thể sử dụng nhiều kênh để mở rộng các tham số mô hình của tầng tích chập.
* Tầng tích chập $1\times 1$ khi được áp dụng lên từng điểm ảnh tương đương với tầng kết nối đầy đủ.
* Tầng tích chập $1\times 1$ thường được sử dụng để điều chỉnh số lượng kênh giữa các tầng của mạng và để kiểm soát độ phức tạp của mô hình.


<!--
## Exercises
-->

## Bài tập

<!--
1. Assume that we have two convolutional kernels of size $k_1$ and $k_2$ respectively (with no nonlinearity in between).
    * Prove that the result of the operation can be expressed by a single convolution.
    * What is the dimensionality of the equivalent single convolution?
    * Is the converse true?
2. Assume an input shape of $c_i\times h\times w$ and a convolution kernel with the shape $c_o\times c_i\times k_h\times k_w$, padding of $(p_h, p_w)$, and stride of $(s_h, s_w)$.
    * What is the computational cost (multiplications and additions) for the forward computation?
    * What is the memory footprint?
    * What is the memory footprint for the backward computation?
    * What is the computational cost for the backward computation?
3. By what factor does the number of calculations increase if we double the number of input channels $c_i$ and the number of output channels $c_o$? What happens if we double the padding?
4. If the height and width of the convolution kernel is $k_h=k_w=1$, what is the complexity of the forward computation?
5. Are the variables `Y1` and `Y2` in the last example of this section exactly the same? Why?
6. How would you implement convolutions using matrix multiplication when the convolution window is not $1\times 1$?
-->

1. Giả sử rằng ta có hai bộ lọc tích chập có kích thước tương ứng là $k_1$ và $k_2$ (không có tính phi tuyến ở giữa).
    * Chứng minh rằng kết quả của phép tính có thể được biểu diễn bằng chỉ một phép tích chập.
    * Phép tích chập tương đương này có kích thước là bao nhiêu?
    * Điều ngược lại có đúng không?
2. Giả sử kích thước của đầu vào là $c_i\times h\times w$ và áp dụng một bộ lọc tích chập có kích thước $c_o\times c_i\times k_h\times k_w$, đồng thời sử dụng đệm $(p_h, p_w)$ và sải bước $(s_h, s_w)$.
    * Chi phí tính toán (phép nhân và phép cộng) cho tính toán truyền xuôi là bao nhiêu?
    * Độ phức tạp bộ nhớ cho tính toán truyền xuôi là bao nhiêu?
    * Độ phức tạp bộ nhớ cho tính toán truyền ngược là bao nhiêu?
    * Chi phí tính toán cho tính toán truyền nguược là bao nhiên?
3. Số lượng tính toán sẽ tăng lên bao nhiêu lần nếu ta nhân đôi số lượng kênh đầu vào $c_i$ và số lượng kênh đầu ra $c_o$? Điều gì xảy ra nếu ta nhân đôi phần đệm?
4. Nếu chiều cao và chiều rộng của bộ lọc tích chập là $k_h =k_w=1$, thì độ phức tạp của tính toán truyền xuôi là bao nhiêu?
5. Các biến `Y1` và` Y2` trong ví dụ cuối cùng của mục này có giống nhau không? Tại sao?
6. Khi cửa sổ tích chập không phải là $1\times 1$, bạn sẽ lập trình các phép tích chập sử dụng phép nhân ma trận như thế nào?

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2351)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2351)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
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

<!-- Phần 5 -->
*

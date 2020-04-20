<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Convolutions for Images
-->

# Phép tích chập cho ảnh
:label:`sec_conv_layer`

<!--
Now that we understand how convolutional layers work in theory, we are ready to see how this works in practice.
Since we have motivated convolutional neural networks by their applicability to image data, 
we will stick with image data in our examples, and begin by revisiting the convolutional layer that we introduced in the previous section.
We note that strictly speaking, *convolutional* layers are a slight misnomer, since the operations are typically expressed as cross correlations.
-->

Giờ chúng ta đã hiểu cách các tầng tích chập hoạt động trên lý thuyết, hãy xem chúng hoạt động trong thực tế như thế nào.
Với cảm hứng từ khả năng ứng dụng các mạng nơ-ron tích chập với dữ liệu hình ảnh, chúng ta vẫn sẽ sử dụng loại dữ liệu này trong các ví dụ, và bắt đầu với tầng tích chập được giới thiệu ở phần trước.
Chú ý rằng, một cách chặt chẽ, việc đặt tên các tầng là *tích chập* là không chính xác, vì các phép toán thường được biểu diễn dưới dạng tương quan chéo.

<!--
## The Cross-Correlation Operator
-->

## Toán tử tương quan chéo

<!--
In a convolutional layer, an input array and a correlation kernel array are combined to produce an output array through a cross-correlation operation.
Let us see how this works for two dimensions.
In :numref:`fig_correlation`, the input is a two-dimensional array with a height of 3 and width of 3.
We mark the shape of the array as $3 \times 3$ or (3, 3).
The height and width of the kernel array are both 2.
Common names for this array in the deep learning research community include *kernel* and *filter*.
The shape of the kernel window (also known as the convolution window) is given precisely by the height and width of the kernel (here it is $2 \times 2$).
-->

Trong một tầng tích chập, một mảng đầu vào và một mảng bộ lọc tương quan được kết hợp để tạo ra mảng đầu ra bằng phép toán tương quan chéo.
Hãy xem phép toán này hoạt động như thế nào với mảng hai chiều.
Trong :numref:`fig_correlation`, đầu vào là một mảng hai chiều với chiều dài 3 và chiều rộng 3.
Ta kí hiệu kích thước của mảng là $3 \times 3$ hoặc (3, 3).
Chiều dài và chiều rộng của mảng bộ lọc đều là 2.
Trong cộng đồng nghiên cứu học sâu, các tên thường gặp của mảng này gồm có *kernel* và *filter*.
Kích thước của cửa sổ bộ lọc (còn gọi là cửa sổ tích chập) được định nghĩa từ chiều dài và chiều rộng của bộ lọc (ở đây là $2 \times 2$).


<!--
![Two-dimensional cross-correlation operation. The shaded portions are the first output element and the input and kernel array elements used in its computation: $0\times0+1\times1+3\times2+4\times3=19$. ](../img/correlation.svg)
-->

![Phép tương quan chéo hai chiều. Các phần in đậm là phần tử đầu tiên của đầu ra, các phần tử của đầu vào và bộ lọc được sử dụng trong phép toán: $0\times0+1\times1+3\times2+4\times3=19$. ](../img/correlation.svg)
:label:`fig_correlation`

<!--
In the two-dimensional cross-correlation operation, we begin with the convolution window positioned at the top-left corner of the input array 
and slide it across the input array, both from left to right and top to bottom.
When the convolution window slides to a certain position, the input subarray contained in that window and the kernel array are multiplied (elementwise) 
and the resulting array is summed up yielding a single scalar value.
This result if precisely the value of the output array at the corresponding location.
Here, the output array has a height of 2 and width of 2 and the four elements are derived from the two-dimensional cross-correlation operation:
-->

Trong phép tương quan chéo hai chiều, ta bắt đầu với cửa sổ tích chập đặt tại vị trí góc trên bên trái của mảng đầu vào và di chuyển cửa sổ này từ trái sang phải và từ trên xuống dưới.
Khi cửa sổ tích chập ở một vị trí nào đó, mảng con của đầu vào chứa trong cửa sổ đó và mảng bộ lọc được nhân theo từng phần tử
và cộng các kết quả với nhau tạo thành một giá trị số vô hướng duy nhất.
Giá trị này được ghi vào mảng đầu ra tại vị trí tương ứng.
Ở đây, mảng đầu ra có chiều dài 2 và chiều rộng 2, với bốn phần tử được tính từ phép tương quan chéo hai chiều.

$$
0\times0+1\times1+3\times2+4\times3=19,\\
1\times0+2\times1+4\times2+5\times3=25,\\
3\times0+4\times1+6\times2+7\times3=37,\\
4\times0+5\times1+7\times2+8\times3=43.
$$

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
Note that along each axis, the output is slightly *smaller* than the input.
Because the kernel has a width greater than one, and we can only computer the cross-correlation for locations where the kernel fits wholly within the image, 
the output size is given by the input size $H \times W$ minus the size of the convolutional kernel $h \times w$ via $(H-h+1) \times (W-w+1)$.
This is the case since we need enough space to 'shift' the convolutional kernel across the image 
(later we will see how to keep the size unchanged by padding the image with zeros around its boundary such that there is enough space to shift the kernel).
Next, we implement the above process in the `corr2d` function.
It accepts the input array `X` with the kernel array `K` and outputs the array `Y`.
-->

*dịch đoạn phía trên*

```{.python .input}
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()

# Saved in the d2l package for later use
def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y
```

<!--
We can construct the input array `X` and the kernel array `K` from the figure above
to validate the output of the above implementations of the two-dimensional cross-correlation operation.
-->

*dịch đoạn phía trên*

```{.python .input}
X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = np.array([[0, 1], [2, 3]])
corr2d(X, K)
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Convolutional Layers
-->

## *dịch tiêu đề phía trên*

<!--
A convolutional layer cross-correlates the input and kernels and adds a scalar bias to produce an output.
The parameters of the convolutional layer are precisely the values that constitute the kernel and the scalar bias.
When training the models based on convolutional layers, we typically initialize the kernels randomly, just as we would with a fully-connected layer.
-->

*dịch đoạn phía trên*

<!--
We are now ready to implement a two-dimensional convolutional layer based on the `corr2d` function defined above.
In the `__init__` constructor function, we declare `weight` and `bias` as the two model parameters.
The forward computation function `forward` calls the `corr2d` function and adds the bias.
As with $h \times w$ cross-correlation we also refer to convolutional layers as $h \times w$ convolutions.
-->

*dịch đoạn phía trên*

```{.python .input  n=70}
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Object Edge Detection in Images
-->

## *dịch tiêu đề phía trên*

<!--
Let us look at a simple application of a convolutional layer: detecting the edge of an object in an image by finding the location of the pixel change.
First, we construct an 'image' of $6\times 8$ pixels.
The middle four columns are black (0) and the rest are white (1).
-->

*dịch đoạn phía trên*

```{.python .input  n=66}
X = np.ones((6, 8))
X[:, 2:6] = 0
X
```

<!--
Next, we construct a kernel `K` with a height of 1 and width of 2.
When we perform the cross-correlation operation with the input, if the horizontally adjacent elements are the same, the output is 0. Otherwise, the output is non-zero.
-->

*dịch đoạn phía trên*

```{.python .input  n=67}
K = np.array([[1, -1]])
```

<!--
Enter `X` and our designed kernel `K` to perform the cross-correlation operations.
As you can see, we will detect 1 for the edge from white to black and -1 for the edge from black to white.
The rest of the outputs are 0.
-->

*dịch đoạn phía trên*

```{.python .input  n=69}
Y = corr2d(X, K)
Y
```

<!--
Let us apply the kernel to the transposed image.
As expected, it vanishes. The kernel `K` only detects vertical edges.
-->

*dịch đoạn phía trên*

```{.python .input}
corr2d(X.T, K)
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Learning a Kernel
-->

## *dịch tiêu đề phía trên*

<!--
Designing an edge detector by finite differences `[1, -1]` is neat if we know this is precisely what we are looking for.
However, as we look at larger kernels, and consider successive layers of convolutions, it might be impossible to specify precisely what each filter should be doing manually.
-->

*dịch đoạn phía trên*

<!--
Now let us see whether we can learn the kernel that generated `Y` from `X` by looking at the (input, output) pairs only.
We first construct a convolutional layer and initialize its kernel as a random array.
Next, in each iteration, we will use the squared error to compare `Y` and the output of the convolutional layer, then calculate the gradient to update the weight.
For the sake of simplicity, in this convolutional layer, we will ignore the bias.
-->

*dịch đoạn phía trên*

<!--
We previously constructed the `Conv2D` class.
However, since we used single-element assignments,
Gluon has some trouble finding the gradient.
Instead, we use the built-in `Conv2D` class provided by Gluon below.
-->

*dịch đoạn phía trên*

```{.python .input  n=83}
# Construct a convolutional layer with 1 output channel
# (channels will be introduced in the following section)
# and a kernel array shape of (1, 2)
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example channel, height, width), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = X.reshape(1, 1, 6, 8)
Y = Y.reshape(1, 1, 6, 7)

for i in range(10):
    with autograd.record():
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    # For the sake of simplicity, we ignore the bias here
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum()))
```

<!--
As you can see, the error has dropped to a small value after 10 iterations.
Now we will take a look at the kernel array we learned.
-->

*dịch đoạn phía trên*

```{.python .input}
conv2d.weight.data().reshape(1, 2)
```

<!--
Indeed, the learned kernel array is remarkably close to the kernel array `K` we defined earlier.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Cross-Correlation and Convolution
-->

## *dịch tiêu đề phía trên*

<!--
Recall the observation from the previous section that cross-correlation and convolution are equivalent.
In the figure above it is easy to see this correspondence.
Simply flip the kernel from the bottom left to the top right.
In this case the indexing in the sum is reverted, yet the same result can be obtained.
In keeping with standard terminology with deep learning literature, 
we will continue to refer to the cross-correlation operation as a convolution even though, strictly-speaking, it is slightly different.
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## Tóm tắt

<!--
* The core computation of a two-dimensional convolutional layer is a two-dimensional cross-correlation operation. 
In its simplest form, this performs a cross-correlation operation on the two-dimensional input data and the kernel, and then adds a bias.
* We can design a kernel to detect edges in images.
* We can learn the kernel through data.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Construct an image `X` with diagonal edges.
    * What happens if you apply the kernel `K` to it?
    * What happens if you transpose `X`?
    * What happens if you transpose `K`?
2. When you try to automatically find the gradient for the `Conv2D` class we created, what kind of error message do you see?
3. How do you represent a cross-correlation operation as a matrix multiplication by changing the input and kernel arrays?
4. Design some kernels manually.
    * What is the form of a kernel for the second derivative?
    * What is the kernel for the Laplace operator?
    * What is the kernel for an integral?
    * What is the minimum size of a kernel to obtain a derivative of degree $d$?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->


<!--
## [Discussions](https://discuss.mxnet.io/t/2349)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2349)
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
* Nguyễn Văn Cường

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

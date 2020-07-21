<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Implementation of Softmax Regression from Scratch
-->

# Lập trình Hồi quy Sofmax từ đầu
:label:`sec_softmax_scratch`

<!--
Just as we implemented linear regression from scratch, we believe that multiclass logistic (softmax) regression is similarly fundamental and you ought to know the gory details of how to implement it yourself.
As with linear regression, after doing things by hand we will breeze through an implementation in Gluon for comparison.
To begin, let's import the familiar packages.
-->

Giống như khi ta lập trình hồi quy tuyến tính từ đầu, hồi quy (softmax) logistic đa lớp cũng là một kĩ thuật căn bản mà bạn nên hiểu biết tường tận các chi tiết để có thể tự xây dựng lại.
Sau khi tự lập trình lại mọi thứ thì ta cũng sẽ dùng Gluon để so sánh như ở phần hồi quy tuyến tính.
Để bắt đầu, chúng ta nhập các thư viện quen thuộc vào.

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
from IPython import display
npx.set_np()
```

<!--
We will work with the Fashion-MNIST dataset, just introduced in :numref:`sec_fashion_mnist`, setting up an iterator with batch size $256$.
-->

Ta sẽ làm việc trên tập dữ liệu Fashion-MNIST, vừa được giới thiệu trong :numref:`sec_fashion_mnist`, thiết lập một iterator với kích thước batch là $256$.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

<!--
## Initializing Model Parameters
-->

## Khởi tạo các Tham số của Mô hình

<!--
As in our linear regression example, each example here will be represented by a fixed-length vector.
Each example in the raw data is a $28 \times 28$ image.
In this section, we will flatten each image, treating them as $784$ 1D vectors.
In the future, we will talk about more sophisticated strategies for exploiting the spatial structure in images, but for now we treat each pixel location as just another feature.
-->

Giống như ví dụ về hồi quy tuyến tính, mỗi mẫu sẽ được biểu diễn bằng một vector có chiều dài cố định.
Mỗi mẫu trong tập dữ liệu thô là một ảnh $28 \times 28$.
Trong phần này, chúng ta sẽ trải phẳng mỗi tấm ảnh thành một vector có kích thước là $784$.
Sau này ta sẽ bàn về các chiến lược công phu hơn có khả năng khai thác cấu trúc không gian của ảnh, còn bây giờ ta hãy xem mỗi điểm ảnh là một đặc trưng. 

<!--
Recall that in softmax regression, we have as many outputs as there are categories.
Because our dataset has $10$ categories, our network will have an output dimension of $10$.
Consequently, our weights will constitute a $784 \times 10$ matrix and the biases will constitute a $1 \times 10$ vector.
As with linear regression, we will initialize our weights $W$ with Gaussian noise and our biases to take the initial value $0$.
-->

Nhắc lại trong hồi quy softmax, mỗi lớp sẽ có một đầu ra.
Vì tập dữ liệu của chúng ta có $10$ lớp, mạng của chúng ta sẽ có $10$ đầu ra.
Do đó, các trọng số sẽ tạo thành một ma trận $784 \times 10$ và các hệ số điều chỉnh sẽ tạo thành một vector $1 \times 10$.
Cũng như hồi quy tuyến tính, ta sẽ khởi tạo các trọng số $W$ bằng nhiễu Gauss và các hệ số điều chỉnh sẽ được khởi tạo bằng $0$.

```{.python .input  n=3}
num_inputs = 784
num_outputs = 10

W = np.random.normal(0, 0.01, (num_inputs, num_outputs))
b = np.zeros(num_outputs)
```

<!--
Recall that we need to *attach gradients* to the model parameters.
More literally, we are allocating memory for future gradients to be stored and notifiying MXNet that we will want to calculate gradients with respect to these parameters in the future.
-->

Hãy nhớ rằng ta cần *đính kèm gradient* vào các tham số của mô hình.
Cụ thể hơn, ta cho MXNet biết rằng ta sẽ muốn tính các gradient theo các tham số này và cần phân bổ bộ nhớ để lưu trữ chúng trong tương lai.

```{.python .input  n=4}
W.attach_grad()
b.attach_grad()
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## The Softmax
-->

## Softmax

<!--
Before implementing the softmax regression model, let's briefly review how operators such as `sum` work along specific dimensions in an `ndarray`.
Given a matrix `X` we can sum over all elements (default) or only over elements in the same axis, *i.e.*, the column (`axis=0`) or the same row (`axis=1`).
Note that if `X` is an array with shape `(2, 3)` and we sum over the columns (`X.sum(axis=0`), the result will be a (1D) vector with shape `(3,)`.
If we want to keep the number of axes in the original array (resulting in a 2D array with shape `(1, 3)`), 
rather than collapsing out the dimension that we summed over we can specify `keepdims=True` when invoking `sum`.
-->

Trước khi xây dựng mô hình hồi quy softmax, hãy ôn nhanh tác dụng của các toán tử như `sum` trên những chiều cụ thể của một `ndarray`.
Cho một ma trận `X`, chúng ta có thể tính tổng tất cả các phần tử (mặc định) hoặc chỉ trên các phần tử trong cùng một trục, *ví dụ*, cột (`axis=0`) hoặc cùng một hàng (`axis=1`).
Lưu ý rằng nếu `X` là một mảng có kích thước `(2, 3)`, tính tổng các cột (`X.sum (axis=0)`) sẽ trả về một vector (một chiều) có kích thước là `(3,)`.
Nếu chúng ta muốn giữ số lượng trục giống với mảng ban đầu thay vì thu gọn kích thước trên các trục đã tính toán (dẫn đến một mảng 2 chiều có kích thước `(1, 3)`), ta có thể chỉ định `keepdims=True` khi gọi hàm `sum`.

```{.python .input  n=5}
X = np.array([[1, 2, 3], [4, 5, 6]])
print(X.sum(axis=0, keepdims=True), '\n', X.sum(axis=1, keepdims=True))
```

<!--
We are now ready to implement the softmax function.
Recall that softmax consists of two steps:
First, we exponentiate each term (using `exp`).
Then, we sum over each row (we have one row per example in the batch) to get the normalization constants for each example.
Finally, we divide each row by its normalization constant, ensuring that the result sums to $1$.
Before looking at the code, let's recall what this looks expressed as an equation:
-->

Bây giờ chúng ta đã có thể bắt đầu xây dựng hàm softmax.
Lưu ý rằng việc thực thi hàm softmax bao gồm hai bước:
Đầu tiên, chúng ta lũy thừa từng giá trị (sử dụng `exp`).
Sau đó tính tổng trên mỗi hàng để lấy các hằng số chuẩn hóa cho mỗi mẫu (vì mỗi mẫu là một hàng).
Cuối cùng, chia mỗi hàng theo hằng số chuẩn hóa của nó để đảm bảo rằng kết quả có tổng bằng $1$.
Trước khi xem đoạn mã, chúng ta hãy nhớ lại các bước này được thể hiện trong phương trình sau:

$$
\mathrm{softmax}(\mathbf{X})_{ij} = \frac{\exp(X_{ij})}{\sum_k \exp(X_{ik})}.
$$

<!--
The denominator, or normalization constant, is also sometimes called the partition function (and its logarithm is called the log-partition function).
The origins of that name are in [statistical physics](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)) where a related equation models the distribution over an ensemble of particles).
-->

Mẫu số hoặc hằng số chuẩn hóa đôi khi cũng được gọi là hàm phân hoạch (*partition function*) và logarit của nó được gọi là hàm log phân hoạch (*log-partition function*).
Tên gốc của hàm được định nghĩa trong [cơ học thống kê](https://en.wikipedia.org/wiki/Partition_function_(statistical_mechanics)) với phương trình liên quan đến mô hình hóa phân phối trên một tập hợp các phần tử.

```{.python .input  n=6}
def softmax(X):
    X_exp = np.exp(X)
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # The broadcast mechanism is applied here
```

<!--
As you can see, for any random input, we turn each element into a non-negative number.
Moreover, each row sums up to 1, as is required for a probability.
Note that while this looks correct mathematically, we were a bit sloppy in our implementation 
because failed to take precautions against numerical overflow or underflow due to large (or very small) elements of the matrix, as we did in :numref:`sec_naive_bayes`.
-->

Có thể thấy rằng với bất kỳ đầu vào ngẫu nhiên nào thì mỗi phần tử đều được biến đổi thành một số không âm.
Hơn nữa, mỗi hàng đều có tổng là 1 nên thoả mãn yêu cầu làm một giá trị xác suất.
Chú ý rằng đoạn mã trên tuy đúng về mặt toán học nhưng chúng tôi đã hơi cẩu thả về mặt lập trình vì đã không kiểm tra vấn đề tràn số trên và dưới gây ra bởi các giá trị vô cùng lớn hoặc vô cùng nhỏ trong ma trận, không giống với cách đã thực hiện tại :numref:`sec_naive_bayes`.

```{.python .input  n=7}
X = np.random.normal(size=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## The Model
-->

## Mô hình

<!--
Now that we have defined the softmax operation, we can implement the softmax regression model.
The below code defines the forward pass through the network.
Note that we flatten each original image in the batch into a vector with length `num_inputs` with the `reshape` function before passing the data through our model.
-->

Sau khi đã định nghĩa hàm softmax, chúng ta có thể bắt đầu lập trình mô hình hồi quy softmax.
Đoạn mã sau định nghĩa lượt truyền xuôi qua mạng.
Chú ý rằng chúng ta sẽ làm phẳng mỗi ảnh gốc trên batch thành một vector có độ dài `num_inputs` bằng hàm `reshape`, trước khi truyền dữ liệu sang mô hình đã khởi tạo.

```{.python .input  n=8}
def net(X):
    return softmax(np.dot(X.reshape(-1, num_inputs), W) + b)
```

<!--
## The Loss Function
-->

## Hàm mất mát

<!--
Next, we need to implement the cross-entropy loss function, introduced in :numref:`sec_softmax`.
This may be the most common loss function in all of deep learning because, at the moment, classification problems far outnumber regression problems.
-->

Tiếp đến chúng ta cần lập trình hàm mất mát entropy chéo đã được giới thiệu ở :numref:`sec_softmax`.
Đây có lẽ là hàm mất mát thông dụng nhất trong phần lớn nghiên cứu về học sâu vì hiện nay số lượng bài toán phân loại đã vượt xa hơn số lượng bài toán hồi quy.

<!--
Recall that cross-entropy takes the negative log likelihood of the predicted probability assigned to the true label $-\log P(y \mid x)$.
Rather than iterating over the predictions with a Python `for` loop (which tends to be inefficient), 
we can use the `pick` function which allows us to easily select the appropriate terms from the matrix of softmax entries.
Below, we illustrate the `pick` function on a toy example, with $3$ categories and $2$ examples.
-->

Nhắc lại rằng hàm entropy chéo lấy đầu vào là đối log hợp lý của xác suất dự đoán được gán cho nhãn thật $-\log P(y \mid x)$.
Thay vì lặp qua các dự đoán của mô hình bằng vòng lặp `for` trong Python (có xu hướng kém hiệu quả),
chúng ta có thể sử dụng hàm `pick` để dễ dàng chọn các phần tử thích hợp từ ma trận của các giá trị softmax.
Dưới đây, ta minh hoạ cách sử dụng hàm `pick` trong một ví dụ đơn giản với ma trận có $3$ lớp và $2$ mẫu.

```{.python .input  n=9}
y_hat = np.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], [0, 2]]
```

<!--
Now we can implement the cross-entropy loss function efficiently with just one line of code.
-->

Bây giờ chúng ta có thể lập trình hàm mất mát entropy chéo hiệu quả hơn chỉ với một dòng lệnh.

```{.python .input  n=10}
def cross_entropy(y_hat, y):
    return - np.log(y_hat[range(len(y_hat)), y])
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Classification Accuracy
-->

## Độ chính xác cho bài toán Phân loại

<!--
Given the predicted probability distribution `y_hat`, we typically choose the class with highest predicted probability whenever we must output a *hard* prediction.
Indeed, many applications require that we make a choice.
Gmail must categorize an email into Primary, Social, Updates, or Forums.
It might estimate probabilities internally, but at the end of the day it has to choose one among the categories.
-->

Với phân phối xác suất dự đoán `y_hat`, ta thường chọn lớp có xác suất dự đoán cao nhất khi phải đưa ra một dự đoán *cụ thể* vì nhiều ứng dụng trong thực tế có yêu cầu như vậy.
Ví dụ, Gmail cần phải phân loại một email vào trong các danh mục sau: Email chính (Primary), Mạng xã hội (Social), Nội dung cập nhật (Updates) hoặc Diễn đàn (Forums).
Có thể các xác suất được tính toán bên trong nội bộ hệ thống, nhưng cuối cùng kết quả vẫn chỉ là một trong các danh mục.

<!--
When predictions are consistent with the actual category `y`, they are correct.
The classification accuracy is the fraction of all predictions that are correct.
Although it can be difficult optimize accuracy directly (it is not differentiable), 
it is often the performance metric that we care most about, and we will nearly always report it when training classifiers.
-->

Các dự đoán được coi là chính xác khi chúng khớp với lớp thực tế `y`.
Độ chính xác phân loại được tính bởi tỉ lệ các dự đoán chính xác trên tất cả các dự đoán đã đưa ra.
Dù ta có thể gặp khó khăn khi tối ưu hóa trực tiếp độ chính xác (chúng không khả vi),
đây thường là phép đo chất lượng được quan tâm nhiều nhất và sẽ luôn được tính khi huấn luyện các bộ phân loại.

<!--
To compute accuracy we do the following:
First, we execute `y_hat.argmax(axis=1)` to gather the predicted classes(given by the indices for the largest entires each row).
The result has the same shape as the variable `y`.
Now we just need to check how frequently the two match.
Since the equality operator `==` is datatype-sensitive(e.g., an `int` and a `float32` are never equal),we also need to convert both to the same type (we pick `float32`).
The result is an `ndarray` containing entries of 0 (false) and 1 (true).
Taking the mean yields the desired result.
-->

Độ chính xác được tính toán như sau:
Đầu tiên, dùng lệnh `y_hat.argmax(axis=1)` nhằm lấy ra các lớp được dự đoán (được cho bởi chỉ số của phần tử lớn nhất trên mỗi hàng).
Kết quả trả về sẽ có cùng kích thước với biến `y` và bây giờ ta chỉ cần so sánh hai vector này.
Vì toán tử `==` so khớp cả kiểu dữ liệu của biến (ví dụ một biến `int` và một biến `float32` không thể bằng nhau), ta cần đưa chúng về cùng một kiểu dữ liệu (ở đây ta chọn kiểu `float32`).
Kết quả sẽ là một `ndarray` chứa các giá trị 0 (false) và 1 (true), giá trị trung bình này mang lại kết quả mà ta mong muốn.


```{.python .input  n=11}
# Saved in the d2l package for later use
def accuracy(y_hat, y):
    if y_hat.shape[1] > 1:
        return float((y_hat.argmax(axis=1) == y.astype('float32')).sum())
    else:
        return float((y_hat.astype('int32') == y.astype('int32')).sum())
```

<!--
We will continue to use the variables `y_hat` and `y` defined in the `pick` function, as the predicted probability distribution and label, respectively.
We can see that the first example's prediction category is $2$ (the largest element of the row is 0.6 with an index of $2$), which is inconsistent with the actual label, $0$.
The second example's prediction category is $2$ (the largest element of the row is $0.5$ with an index of $2$), which is consistent with the actual label, $2$.
Therefore, the classification accuracy rate for these two examples is $0.5$.
-->

Ta sẽ tiếp tục sử dụng biến `y_hat` và `y` đã được định nghĩa trong hàm `pick`, lần lượt tương ứng với phân phối xác suất được dự đoán và nhãn.
Có thể thấy rằng kết quả dự đoán lớp từ ví dụ đầu tiên là $2$ (phần từ lớn nhất trong hàng là $0.6$ với chỉ số tương ứng là $2$) không khớp với nhãn thực tế là $0$. 
Dự đoán lớp ở ví dụ thứ hai là $2$ (phần tử lớn nhất hàng là $0.5$ với chỉ số tương ứng là $2$) khớp với nhãn thực tế là $2$.
Do đó, ta có độ chính xác phân loại cho hai ví dụ này là $0.5$.

```{.python .input  n=12}
y = np.array([0, 2])
accuracy(y_hat, y) / len(y)
```

<!--
Similarly, we can evaluate the accuracy for model `net` on the dataset (accessed via `data_iter`).
-->

Tương tự như trên, ta có thể đánh giá độ chính xác của mô hình `net` trên tập dữ liệu (được truy xuất thông qua `data_iter`).

```{.python .input  n=13}
# Saved in the d2l package for later use
def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)  # num_corrected_examples, num_examples
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.size)
    return metric[0] / metric[1]
```

<!--
Here `Accumulator` is a utility class to accumulated sum over multiple numbers.
-->

`Accumulator` ở đây là một lớp đa tiện ích, có tác dụng tính tổng tích lũy của nhiều số.

```{.python .input}
# Saved in the d2l package for later use
class Accumulator(object):
    """Sum a list of numbers over time."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a+float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0] * len(self.data)

    def __getitem__(self, i):
        return self.data[i]
```

<!--
Because we initialized the `net` model with random weights, the accuracy of this model should be close to random guessing, i.e., $0.1$ for $10$ classes.
-->

Vì ta đã khởi tạo mô hình `net` với trọng số ngẫu nhiên nên độ chính xác của mô hình lúc này sẽ gần với việc phỏng đoán ngẫu nhiên, tức độ chính xác bằng $0.1$ với $10$ lớp.

```{.python .input  n=14}
evaluate_accuracy(net, test_iter)
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Model Training
-->

## Huấn luyện Mô hình

<!--
The training loop for softmax regression should look strikingly familiar if you read through our implementation of linear regression in :numref:`sec_linear_scratch`.
Here we refactor the implementation to make it reusable.
First, we define a function to train for one data epoch.
Note that `updater` is general function to update the model parameters, which accepts the batch size as an argument.
It can be either a wrapper of `d2l.sgd` or a Gluon trainer.
-->

Vòng lặp huấn luyện cho hồi quy softmax trông khá quen thuộc nếu bạn đã xem cách lập trình cho hồi quy tuyến tính tại :numref:`sec_linear_scratch`. 
Ở đây, chúng ta tái cấu trúc (*refactor*) lại đoạn mã để sau này có thể tái sử dụng. 
Đầu tiên, chúng ta định nghĩa một hàm để huấn luyện với một epoch dữ liệu. 
Lưu ý rằng `updater` là một hàm tổng quát để cập nhật các tham số của mô hình và sẽ nhận kích thước batch làm đối số. 
Nó có thể là một wrapper của `d2l.sgd` hoặc là một đối tượng huấn luyện Gluon. 

```{.python .input  n=15}
# Saved in the d2l package for later use
def train_epoch_ch3(net, train_iter, loss, updater):
    metric = Accumulator(3)  # train_loss_sum, train_acc_sum, num_examples
    if isinstance(updater, gluon.Trainer):
        updater = updater.step
    for X, y in train_iter:
        # Compute gradients and update parameters
        with autograd.record():
            y_hat = net(X)
            l = loss(y_hat, y)
        l.backward()
        updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.size)
    # Return training loss and training accuracy
    return metric[0]/metric[2], metric[1]/metric[2]
```

<!--
Before showing the implementation of the training function, we define a utility class that draw data in animation.
Again, it aims to simplify the codes in later chapters.
-->

Trước khi xem đoạn mã thực hiện hàm huấn luyện, ta định nghĩa một lớp phụ trợ để minh họa dữ liệu.
Mục đích của nó là đơn giản hoá các đoạn mã sẽ xuất hiện trong những chương sau. 

```{.python .input  n=16}
# Saved in the d2l package for later use
class Animator(object):
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear', fmts=None,
                 nrows=1, ncols=1, figsize=(3.5, 2.5)):
        """Incrementally plot multiple lines."""
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda to capture arguments
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        """Add multiple data points into the figure."""
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        if not self.fmts:
            self.fmts = ['-'] * n
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
```

<!--
The training function then runs multiple epochs and visualize the training progress.
-->

Hàm huấn luyện sau đó sẽ chạy qua nhiều epoch và trực quan hoá quá trình huấn luyện. 

```{.python .input  n=17}
# Saved in the d2l package for later use
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs],
                        ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
```

<!--
Again, we use the minibatch stochastic gradient descent to optimize the loss function of the model.
Note that the number of epochs (`num_epochs`), and learning rate (`lr`) are both adjustable hyper-parameters.
By changing their values, we may be able to increase the classification accuracy of the model.
In practice we will want to split our data three ways into training, validation, and test data, using the validation data to choose the best values of our hyperparameters.
-->

Nhắc lại, chúng ta sử dụng giải thuật hạ gradient ngẫu nhiên theo minibatch để tối ưu hàm mất mát của mô hình.
Lưu ý rằng số lượng epoch (`num_epochs`), và hệ số học (`lr`) là hai siêu tham số được hiệu chỉnh. 
Bằng cách thay đổi các giá trị này, chúng ta có thể tăng độ chính xác khi phân loại của mô hình. 
Trong thực tế, chúng ta sẽ chia dữ liệu của mình theo ba hướng tiếp cận khác nhau gồm: dữ liệu huấn luyện, dữ liệu kiểm định và dữ liệu kiểm tra; sử dụng dữ liệu kiểm định để chọn ra những giá trị tốt nhất cho các siêu tham số.

```{.python .input  n=18}
num_epochs, lr = 10, 0.1

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
## Prediction
-->

## Dự đoán

<!--
Now that training is complete, our model is ready to classify some images.
Given a series of images, we will compare their actual labels (first line of text output) and the model predictions (second line of text output).
-->

Giờ thì việc huấn luyện đã hoàn thành, mô hình của chúng ta đã sẵn sàng để phân loại ảnh.
Cho một loạt các ảnh, chúng ta sẽ so sánh các nhãn thật của chúng (dòng đầu tiên của văn bản đầu ra) với những dự đoán của mô hình (dòng thứ hai của văn bản đầu ra).

```{.python .input  n=19}
# Saved in the d2l package for later use
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true+'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```

<!--
## Summary
-->

## Tóm tắt

<!--
With softmax regression, we can train models for multi-category classification.
The training loop is very similar to that in linear regression: retrieve and read data, define models and loss functions, then train models using optimization algorithms.
As you will soon find out, most common deep learning models have similar training procedures.
-->

Với hồi quy softmax, chúng ta có thể huấn luyện các mô hình cho bài toán phân loại đa lớp.
Vòng lặp huấn luyện rất giống với vòng lặp huấn luyện của hồi quy tuyến tính: truy xuất và đọc dữ liệu, định nghĩa mô hình và hàm mất mát, và rồi huấn luyện mô hình sử dụng các giải thuật tối ưu.
Rồi bạn sẽ sớm thấy rằng hầu hết các mô hình học sâu phổ biến đều có quy trình huấn luyện tương tự như vậy.

<!--
## Exercises
-->

## Bài tập

<!--
1. In this section, we directly implemented the softmax function based on the mathematical definition of the softmax operation. 
What problems might this cause (hint: try to calculate the size of $\exp(50)$)?
2. The function `cross_entropy` in this section is implemented according to the definition of the cross-entropy loss function. 
What could be the problem with this implementation (hint: consider the domain of the logarithm)?
3. What solutions you can think of to fix the two problems above?
4. Is it always a good idea to return the most likely label. E.g. would you do this for medical diagnosis?
5. Assume that we want to use softmax regression to predict the next word based on some features. What are some problems that might arise from a large vocabulary?
-->

1. Trong mục này, chúng ta đã lập trình hàm softmax dựa vào định nghĩa toán học của phép toán softmax. 
Điều này có thể gây ra những vấn đề gì (gợi ý: thử tính $\exp(50)$)?
2. Hàm `cross_entropy` trong mục này được lập trình dựa vào định nghĩa của hàm mất mát entropy chéo. 
Vấn đề gì có thể xảy ra với cách lập trình như vậy (gợi ý: xem xét miền của hàm log)?
3. Bạn có thể tìm ra giải pháp cho hai vấn đề trên không?
4. Việc trả về nhãn có khả năng nhất có phải lúc nào cũng là ý tưởng tốt không? Ví dụ, bạn có dùng phương pháp này cho chẩn đoán bệnh hay không?
5. Giả sử rằng chúng ta muốn sử dụng hồi quy softmax để dự đoán từ tiếp theo dựa vào một số đặc trưng. Những vấn đề gì có thể xảy ra nếu dùng một tập từ vựng lớn?

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2336)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2336)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

<!--
![](../img/qr_softmax-regression-scratch.svg)
-->

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
* Bùi Nhật Quân
* Lý Phi Long
* Phạm Hồng Vinh
* Lâm Ngọc Tâm
* Vũ Hữu Tiệp
* Lê Khắc Hồng Phúc
* Nguyễn Cảnh Thướng
* Phạm Minh Đức
* Nguyễn Quang Hải
* Đinh Minh Tân
* Lê Cao Thăng

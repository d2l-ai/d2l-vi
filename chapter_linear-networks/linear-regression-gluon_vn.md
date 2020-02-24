<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Concise Implementation of Linear Regression
-->

# Triển khai xúc tích của Hồi quy Tuyến tính
:label:`sec_linear_gluon`

<!--
Broad and intense interest in deep learning for the past several years has inspired both companies, academics, 
and hobbyists to develop a variety of mature open source frameworks for automating the repetitive work of implementing gradient-based learning algorithms.
In the previous section, we relied only on (i) `ndarray` for data storage and linear algebra; and (ii) `autograd` for calculating derivatives.
In practice, because data iterators, loss functions, optimizers, and neural network layers (and some whole architectures) are so common, modern libraries implement these components for us as well.
-->


*dịch đoạn phía trên*

Sự quan tâm nhiệt thành và rộng khắp tới học sâu trong những năm gần đây đã tạo cảm hứng cho các công ty, học viện và những người đam mê phát triển nhiều framework nguồn mở tới giai đoạn hoàn thiện, giúp tự động hóa các công việc lặp đi lặp lại trong quá trình triển khai các thuật toán học dựa trên gradient.
Trong chương trước, chúng ta chỉ dựa vào (i) `ndarray` để lưu dữ liệu và thực hiện tính toán đại số tuyến tính; và (ii) `autograd` để thực hiện tính đạo hàm.
Trên thực tế, do các bộ duyệt dữ liệu, các hàm mất mát, các bộ tối ưu, và các tầng của mạng nơ-ron (thậm chí là toàn bộ kiến trúc) rất phổ biển, các thư viện hiện đại đã triển khai sẵn những thành phần này cho chúng ta.


<!--
In this section, we will show you how to implement the linear regression model from :numref:`sec_linear_scratch` concisely by using Gluon.
-->


Mục này sẽ hướng dẫn bạn làm cách nào để xây dựng mô hình hồi quy tuyến tính trong phần :numref:`sec_linear_scratch` một cách xúc tích với Gluon.


<!--
## Generating the Dataset
-->

## Tạo Tập dữ liệu

<!--
To start, we will generate the same dataset as in the previous section.
-->

Chúng ta bắt đầu bằng việc tạo một tập dữ liệu như mục trước.


```{.python .input  n=2}
import d2l
from mxnet import autograd, gluon, np, npx
npx.set_np()

true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Reading the Dataset
-->

## Đọc tập dữ liệu

<!--
Rather than rolling our own iterator, we can call upon Gluon's `data` module to read data.
The first step will be to instantiate an `ArrayDataset`.
This object's constructor takes one or more `ndarray`s as arguments.
Here, we pass in `features` and `labels` as arguments.
Next, we will use the `ArrayDataset` to instantiate a `DataLoader`, which also requires that we specify a `batch_size` 
and specify a Boolean value `shuffle` indicating whether or not we want the `DataLoader` to shuffle the data on each epoch (pass through the dataset).
-->

Thay vì tự viết vòng lặp để đọc dữ liệu thì ta có thể gọi module `data` của Gluon. 
Bước đầu tiên sẽ là khởi tạo một `ArrayDataset`.
Hàm tạo của đối tượng này sẽ lấy một hoặc nhiều `ndarray` làm đối số.
Tại đây, ta truyền vào hàm hai đối số là `features` và `labels`.
Kế tiếp, ta sử dụng `ArrayDataset` để khởi tạo một` DataLoader`, điều này yêu cầu ta truyền vào đó một giá trị `batch_size`
và giá trị Boolean `shuffle` để cho biết chúng ta có muốn `DataLoader` xáo trộn dữ liệu trên mỗi epoch (mỗi lần duyệt qua toàn bộ tập dữ liệu) hay không.

```{.python .input  n=3}
# Saved in the d2l package for later use
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a Gluon data loader"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)
```

<!--
Now we can use `data_iter` in much the same way as we called the `data_iter` function in the previous section.
To verify that it is working, we can read and print the first minibatch of instances.
-->

Bây giờ, ta có thể sử dụng `data_iter` theo cách tương tự như cách ta gọi hàm `data_iter` trong phần trước.
Để biết rằng nó có hoạt động được hay không, ta có thể thử đọc và in ra minibatch đầu tiên.

```{.python .input  n=5}
for X, y in data_iter:
    print(X, '\n', y)
    break
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Defining the Model
-->

## Định nghĩa Mô hình

<!--
When we implemented linear regression from scratch (in :numref`sec_linear_scratch`), 
we defined our model parameters explicitly and coded up the calculations to produce output using basic linear algebra operations.
You *should* know how to do this.
But once your models get more complex, and once you have to do this nearly every day, you will be glad for the assistance.
The situation is similar to coding up your own blog from scratch.
Doing it once or twice is rewarding and instructive, but you would be a lousy web developer if every time you needed a blog you spent a month reinventing the wheel.
-->

Khi ta lập trình hồi quy tuyến tính từ đầu (in :numref`sec_linear_scratch`),
ta đã định nghĩa rõ ràng các tham số của mô hình và lập trình để tạo đầu ra từ các phép toán đại số tuyến tính cơ bản.
Bạn *nên* biết cách để làm được điều này.
Nhưng một khi mô hình trở nên phức tạp hơn, đồng thời bạn phải làm điều này gần như hàng ngày, bạn sẽ thấy vui mừng khi có sự hỗ trợ từ các thư viện.
Tình huống này tương tự như việc lập trình blog của riêng bạn lại từ đầu.
Làm điều này một hoặc hai lần thì sẽ bổ ích và mang tính hướng dẫn, nhưng bạn sẽ trở thành một nhà phát triển web tồi nếu mỗi lần viết blog bạn lại dành ra cả một tháng chỉ để phát triển lại từ đầu.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
For standard operations, we can use Gluon's predefined layers, which allow us to focus especially on the layers used to construct the model rather than having to focus on the implementation.
To define a linear model, we first import the `nn` module, which defines a large number of neural network layers (note that "nn" is an abbreviation for neural networks).
We will first define a model variable `net`, which will refer to an instance of the `Sequential` class.
In Gluon, `Sequential` defines a container for several layers that will be chained together.
Given input data, a `Sequential` passes it through the first layer, in turn passing the output as the second layer's input and so forth.
In the following example, our model consists of only one layer, so we do not really need `Sequential`.
But since nearly all of our future models will involve multiple layers, we will use it anyway just to familiarize you with the most standard workflow.
-->

Đối với những phép toán chuẩn, chúng ta có thể sử dụng các tầng đã được định nghĩa trước trong Gluon, điều này cho phép chúng ta tập trung vào những tầng được dùng để xây dựng mô hình hơn là việc ta phải tập trung vào việc triển khai.
Để định nghĩa một mô hình tuyến tính, đầu tiên chúng ta cần nạp vào module 'nn', giúp ta định nghĩa được một lượng lớn các tầng trong mạng nơ-ron (lưu ý rằng "nn" là một chữ viết tắt của neural network)
Đầu tiên ta sẽ định nghĩa một biến mẫu là 'net', tham chiếu đến thể hiện của class "Sequential'
Trong Gluon, 'Sequential' định nghĩa một container chứa nhiều tầng liên kết với nhau
Khi nhận được dữ liệu đầu vào, 'Sequential' sẽ truyền nó vào tầng đầu, từ đó lần lượt xuất ra và trở thành dữ liệu đầu vào của tầng thứ hai và cứ tiếp tục như thế ở các tầng kế tiếp
Trong ví dụ mẫu trên, mô hình chúng ta chỉ có duy nhất một tầng, vì vậy không nhất thiết sử dụng 'Sequential'
Tuy nhiên vì hầu hết các mô hình chúng ta gặp phải trong tương lai đều liên quan đến nhiều tầng, do đó dù sao cũng nên dùng để làm quen với quy trình công việc tiêu chuẩn nhất


```{.python .input  n=5}
from mxnet.gluon import nn
net = nn.Sequential()
```

<!--
Recall the architecture of a single-layer network as shown in :numref:`fig_singleneuron`. 
The layer is said to be *fully-connected* because each of its inputs are connected to each of its outputs by means of a matrix-vector multiplication.
In Gluon, the fully-connected layer is defined in the `Dense` class.
Since we only want to generate a single scalar output, we set that number to $1$.
-->

Hãy cùng nhớ lại kiến trúc của mạng đơn tầng như đã trình bày tại :numref:`fig_singleneuron`
Tầng được gọi là *kết nối đầy đủ* do mỗi đầu vào (inputs) được kết nối lần lượt với từng đầu ra (outputs) bằng một phép nhân ma trận với vector
Trong Gluon, tầng có kết nối đầy đủ được định nghĩa trong class 'Dense'
Bởi vì chúng ta chỉ mong muốn xuất gia một giá trị vô hướng, nên chúng ta gán cho con số đó giá trị $1$.
<!--
![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
-->

![Hồi quy tuyến tính là một mạng nơ ron đơn tầng.](../img/singleneuron.svg)
:label:`fig_singleneuron`

```{.python .input  n=6}
net.add(nn.Dense(1))
```

<!--
It is worth noting that, for convenience,
Gluon does not require us to specify the input shape for each layer.
So here, we do not need to tell Gluon how many inputs go into this linear layer.
When we first try to pass data through our model, e.g., when we execute `net(X)` later, Gluon will automatically infer the number of inputs to each layer.
We will describe how this works in more detail in the chapter "Deep Learning Computation".
-->

Để thuận tiện, điều đáng chú ý là
Gluon không yêu cầu chúng ta định hình kích thước đầu vào (input) trong mỗi tầng
Nên tại đây, chúng ta không cần thiết cho Gluon biết có bao nhiêu đầu vào (inputs) cho mỗi tầng tuyến tính
Khi chúng ta cố gắng truyền dữ liệu qua mô hình của mình lần đầu tiên, ví dụ: khi chúng ta thực hiện `net (X)` sau đó, Gluon sẽ tự động suy ra số lượng đầu vào cho mỗi lớp.
Chúng ta sẽ mô tả làm thế nào điều này hoạt động chi tiết hơn trong chương "Tính toán học tập sâu" sau.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Initializing Model Parameters
-->

## *dịch tiêu đề phía trên*

<!--
Before using `net`, we need to initialize the model parameters, such as the weights and biases in the linear regression model.
We will import the `initializer` module from MXNet.
This module provides various methods for model parameter initialization.
Gluon makes `init` available as a shortcut (abbreviation) to access the `initializer` package.
By calling `init.Normal(sigma=0.01)`, we specify that each *weight* parameter should be randomly sampled from a normal distribution with mean $0$ and standard deviation $0.01$.
The *bias* parameter will be initialized to zero by default.
Both the weight vector and bias will have attached gradients.
-->

*dịch đoạn phía trên*

```{.python .input  n=7}
from mxnet import init
net.initialize(init.Normal(sigma=0.01))
```

<!--
The code above may look straightforward but you should note that something strange is happening here.
We are initializing parameters for a network even though Gluon does not yet know how many dimensions the input will have!
It might be $2$ as in our example or it might be $2000$.
Gluon lets us get away with this because behind the scenes, the initialization is actually *deferred*.
The real initialization will take place only when we for the first time attempt to pass data through the network.
Just be careful to remember that since the parameters have not been initialized yet, we cannot access or manipulate them.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Defining the Loss Function
-->

## *dịch tiêu đề phía trên*

<!--
In Gluon, the `loss` module defines various loss functions.
We will use the imported module `loss` with the pseudonym `gloss`, to avoid confusing it for the variable holding our chosen loss function.
In this example, we will use the Gluon implementation of squared loss (`L2Loss`).
-->

*dịch đoạn phía trên*

```{.python .input  n=8}
from mxnet.gluon import loss as gloss
loss = gloss.L2Loss()  # The squared loss is also known as the L2 norm loss
```

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## Defining the Optimization Algorithm
-->

## *dịch tiêu đề phía trên*

<!--
Minibatch SGD and related variants are standard tools for optimizing neural networks and thus Gluon supports SGD alongside a number of variations on this algorithm through its `Trainer` class.
When we instantiate the `Trainer`, we will specify the parameters to optimize over (obtainable from our net via `net.collect_params()`), 
the optimization algorithm we wish to use (`sgd`), and a dictionary of hyper-parameters required by our optimization algorithm.
SGD just requires that we set the value `learning_rate`, (here we set it to 0.03).
-->

*dịch đoạn phía trên*

```{.python .input  n=9}
from mxnet import gluon
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
## Training
-->

## *dịch tiêu đề phía trên*

<!--
You might have noticed that expressing our model through Gluon requires comparatively few lines of code.
We did not have to individually allocate parameters, define our loss function, or implement stochastic gradient descent.
Once we start working with much more complex models, Gluon's advantages will grow considerably.
However, once we have all the basic pieces in place, the training loop itself is strikingly similar to what we did when implementing everything from scratch.
-->

*dịch đoạn phía trên*

<!--
To refresh your memory: for some number of epochs, we will make a complete pass over the dataset (train_data), iteratively grabbing one minibatch of inputs and the corresponding ground-truth labels.
For each minibatch, we go through the following ritual:
-->

*dịch đoạn phía trên*

<!--
* Generate predictions by calling `net(X)` and calculate the loss `l` (the forward pass).
* Calculate gradients by calling `l.backward()` (the backward pass).
* Update the model parameters by invoking our SGD optimizer (note that `trainer` already knows which parameters to optimize over, so we just need to pass in the minibatch size.
-->

*dịch đoạn phía trên*

<!--
For good measure, we compute the loss after each epoch and print it to monitor progress.
-->

*dịch đoạn phía trên*

```{.python .input  n=10}
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))
```

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
Below, we compare the model parameters learned by training on finite data and the actual parameters that generated our dataset.
To access parameters with Gluon, we first access the layer that we need from `net` and then access that layer's weight (`weight`) and bias (`bias`).
To access each parameter's values as an `ndarray`, we invoke its `data` method.
As in our from-scratch implementation, note that our estimated parameters are close to their ground truth counterparts.
-->

*dịch đoạn phía trên*

```{.python .input  n=12}
w = net[0].weight.data()
print('Error in estimating w', true_w.reshape(w.shape) - w)
b = net[0].bias.data()
print('Error in estimating b', true_b - b)
```

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
## Summary
-->

## *dịch tiêu đề phía trên*
<!--
* Using Gluon, we can implement models much more succinctly.
* In Gluon, the `data` module provides tools for data processing, the `nn` module defines a large number of neural network layers, and the `loss` module defines many common loss functions.
* MXNet's module `initializer` provides various methods for model parameter initialization.
* Dimensionality and storage are automatically inferred (but be careful not to attempt to access parameters before they have been initialized).
-->


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*
<!--
1. If we replace `l = loss(output, y)` with `l = loss(output, y).mean()`, we need to change `trainer.step(batch_size)` to `trainer.step(1)` for the code to behave identically. Why?
2. Review the MXNet documentation to see what loss functions and initialization methods are provided in the modules `gluon.loss` and `init`. Replace the loss by Huber's loss.
3. How do you access the gradient of `dense.weight`?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2333)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2333)
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
* Nguyễn Văn Tâm
* Phạm Hồng Vinh
* Vũ Hữu Tiệp 

<!-- Phần 2 -->
* Lý Phi Long

<!-- Phần 3 -->
* Phạm Đăng Khoa

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*

<!-- Phần 7 -->
*

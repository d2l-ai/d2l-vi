<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Parameter Management
-->

# Quản lý Tham số

<!--
The ultimate goal of training deep networks is to find good parameter values for a given architecture.
When everything is standard, the `nn.Sequential` class is a perfectly good tool for it.
However, very few models are entirely standard and most scientists want to build things that are novel.
This section shows how to manipulate parameters. In particular we will cover the following aspects:
-->

<!-- UPDATE
Once we have chosen an architecture and set our hyperparameters, we proceed to the training loop, where our goal is to find parameter values that minimize our objective function. 
After training, we will need these parameters in order to make future predictions.
Additionally, we will sometimes wish to extract the parameters either to reuse them in some other context,
to save our model to disk so that it may be exectuted in other software,or for examination in the hopes of gaining scientific understanding.
-->

Mục tiêu cuối cùng của việc huấn luyện mạng học sâu là tìm các giá trị tham số tốt cho một kiến trúc có sẵn.
Thông thường, lớp `nn.Sequential` là một công cụ tối ưu cho việc huấn luyện.
Tuy nhiên, rất ít mô hình có cấu trúc hoàn toàn theo tiêu chuẩn, các nhà khoa học luôn muốn xây dựng các kiến trúc mạng mới lạ.
Phần này trình bày cách thức thao tác với tham số. Cụ thể, các khía cạnh sau sẽ được đề cập:

<!--
Most of the time, we will be able to ignore the nitty-gritty details of how parameters are declared and manipulated, relying on Gluon to do the heavy lifting.
However, when we move away from stacked architectures with standard layers, we will sometimes need to get into the weeds of declaring and manipulate parameters. 
In this section, we cover the following:
-->

*dịch nội dung trên*

<!--
* Accessing parameters for debugging, diagnostics, to visualize them or to save them is the first step to understanding how to work with custom models.
* Second, we want to set them in specific ways, e.g., for initialization purposes. We discuss the structure of parameter initializers.
* Last, we show how this knowledge can be put to good use by building networks that share some parameters.
-->

<!-- UPDATE
* Accessing parameters for debugging, diagnostics, and visualiziations.
* Parameter initialization.
* Sharing parameters across different model components.
-->

* Truy cập các tham số cho việc tìm lỗi, gỡ lỗi, để lưu lại hoặc biểu diễn trực quan là bước đầu tiên để hiểu cách làm việc với các mô hình được tuỳ chỉnh.
* Thứ hai là cách gán giá trị cụ thể cho chúng, ví dụ như lúc khởi tạo. Cấu trúc của các bộ khởi tạo tham số cũng sẽ được thảo luận thêm.
* Cuối cùng, chúng ta sẽ trình bày cách áp dụng những kiến thức này để xây dựng các mạng có chung một vài tham số.

<!--
As always, we start from our trusty Multilayer Perceptron with a hidden layer. This will serve as our choice for demonstrating the various features.
-->

<!-- UPDATE
We start by focusing on an MLP with one hidden layer.
-->

Như thường lệ, chúng ta bắt đầu từ mạng Perceptron đa tầng với một tầng ẩn, để minh hoạ số lượng lớn các đặc trưng.

```{.python .input  n=1}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # Use the default initialization method

x = np.random.uniform(size=(2, 20))
net(x)  # Forward computation
```

<!--
## Parameter Access
-->

## Truy cập Tham số

<!--
In the case of a Sequential class we can access the parameters with ease, simply by indexing each of the layers in the network.
The `params` variable then contains the required data. Let's try this out in practice by inspecting the parameters of the first layer.
-->

<!-- UPDATE
Let us start with how to access parameters from the models that you already know.
When a model is defined via the Sequential class, we can first access any layer by indexing into the model as though it were a list.
Each layer's parameters are conveniently located in its `params` attribute. 
We can inspect the parameters of the `net` defined above.
-->

Trong trường hợp lớp Tuần tự (*Sequential*) chúng ta có thể dễ dàng truy cập các tham số bằng chỉ số của các tầng trong mạng.
Biến `params` khi đó chứa dữ liệu các tham số. Ví dụ sau biểu diễn cách truy cập các tham số của tầng thứ nhất.


```{.python .input  n=2}
print(net[0].params)
print(net[1].params)
```

<!--
The output tells us a number of things. 
First, the layer consists of two sets of parameters: `dense0_weight` and `dense0_bias`, as we would expect. 
They are both single precision and they have the necessary shapes that we would expect from the first layer, given that the input dimension is 20 and the output dimension 256. 
In particular the names of the parameters are very useful since they allow us to identify parameters *uniquely* even in a network of hundreds of layers and with nontrivial structure. 
The second layer is structured accordingly.
-->

<!-- UPDATE
The output tells us a few important things.
First, each fully-connected layer contains two parameters, e.g., `dense0_weight` and `dense0_bias`, corresponding to that layer's weights and biases, respectively.
Both are stored as single precision floats.
Note that the names of the parameters are allow us to *uniquely* identifyeach layer's parameters, even in a network contains hundreds of layers.
-->

Kết quả từ đoạn mã này cho ta một vài thông tin.
Đầu tiên, tầng này có hai tập tham số: `dense0_weight` và `dense0_bias` như kỳ vọng.
Chúng đều ở dạng số thực dấu phẩy động độ chính xác đơn và có kích thước cần thiết ở tầng đầu tiên như kỳ vọng, với số chiều của đầu vào là 20 và số chiều của đầu ra là 256.
Tên của các tham số rất hữu ích vì chúng cho phép xác định các tham số *một cách độc nhất* ngay cả trong mạng với hàng trăm tầng với cấu trúc phức tạp.
Tầng thứ hai cũng được cấu trúc theo cách như vậy.
<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
### Targeted Parameters
-->

### Các tham số Mục tiêu

<!--
In order to do something useful with the parameters we need to access them, though. 
There are several ways to do this, ranging from simple to general. 
Let's look at some of them.
-->

<!-- UPDATE
Note that each parameters is represented as an instance of the `Parameter` class.
To do anything useful with the parameters, we first need to access the underlying numerical values. 
There are several ways to do this.
Some are simpler while others are more general.
To begin, given a layer, we can access one of its parameters via the `bias` or `weight` attributes, and further access that parameter's value via its `data()` method.
The following code extracts the bias from the second neural network layer.
-->

Tuy nhiên, để làm việc với các tham số, ta cần truy cập chúng.
Có một vài cách để làm việc này, từ đơn giản đến tổng quát.
Hãy xem qua một số ví dụ.

```{.python .input  n=3}
print(net[1].bias)
print(net[1].bias.data())
```

<!--
The first returns the bias of the second layer. 
Since this is an object containing data, gradients, and additional information, we need to request the data explicitly. 
Note that the bias is all 0 since we initialized the bias to contain all zeros. 
Note that we can also access the parameters by name, such as `dense0_weight`. 
This is possible since each layer comes with its own parameter dictionary that can be accessed directly. 
Both methods are entirely equivalent but the first method leads to much more readable code.
-->

<!-- UPDATE
Parameters are complex objects, containing data, gradients, and additional information.
That's why we need to request the data explicitly.
Note that the bias vector consists of zeroes because we have not updated the network since it was initialized.
We can also access each parameter by name, e.g., `dense0_weight` as follows. 
Under the hood this is possible because each layer contains a parameter dictionary. 
-->

Ví dụ đầu tiên trả về hệ số điều chỉnh của tầng thứ hai.
Vì đây là một đối tượng chứa dữ liệu, các gradient, và các thông tin bổ sung khác, ta cần phải truy vấn dữ liệu một cách tường minh.
Chú ý rằng các hệ số điều chỉnh đều bằng 0 vì ta đã khởi tạo chúng bằng như thế.
Ta cũng có thể truy xuất các tham số theo tên của chúng, chẳng hạn như `dense0_weight`.
Điều này có thể thực hiện được vì mỗi tầng đều có một bộ từ điển tham số kèm theo cho phép ta truy xuất một cách trực tiếp.
Cả hai phương pháp hoàn toàn tương đương với nhau nhưng phương pháp đầu tiên sẽ giúp mã nguồn dễ đọc hơn.

```{.python .input  n=4}
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
```

<!--
Note that the weights are nonzero. 
This is by design since they were randomly initialized when we constructed the network. 
`data` is not the only function that we can invoke. 
For instance, we can compute the gradient with respect to the parameters. 
It has the same shape as the weight. However, since we did not invoke backpropagation yet, the values are all 0.
-->

<!-- UPDATE
Note that unlike the biases, the weights are nonzero. 
This is because unlike biases, weights are initialized randomly. 
In addition to `data`, each `Parameter` also provides a `grad()` method for accessing the gradient. 
It has the same shape as the weight. 
Because we have not invoked backpropagation for this network yet, its values are all 0.
-->

Chú ý rằng các trọng số là khác không.
Điều này là có chủ ý vì chúng được khởi tạo ngẫu nhiên khi ta xây dựng mạng.
`data` không phải là hàm duy nhất mà ta có thể gọi.
Chẳng hạn, ta có thể gọi hàm truy cập gradient theo các tham số.
Các gradient này có cùng kích thước với trọng số, tuy nhiên tất cả các giá trị hiện tại đều bằng 0 bởi vì ta chưa gọi hàm lan truyền ngược.

```{.python .input  n=5}
net[0].weight.grad()
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### All Parameters at Once
-->

### Tất cả các Tham số cùng lúc

<!--
Accessing parameters as described above can be a bit tedious, 
in particular if we have more complex blocks, or blocks of blocks (or even blocks of blocks of blocks), 
since we need to walk through the entire tree in reverse order to how the blocks were constructed. 
To avoid this, blocks come with a method `collect_params` which grabs all parameters of a network in one dictionary such that we can traverse it with ease. 
It does so by iterating over all constituents of a block and calls `collect_params` on subblocks as needed. 
To see the difference consider the following:
-->

<!-- UPDATE
When we need to perform operations on all parameters, accessing them one-by-one can grow tedious.
The situation can grow especially unwieldy when we work with more complex Blocks, (e.g., nested Blocks), since we would need to recurse through the entire tree in to extact each sub-Block's parameters.
To avoid this, each Block comes with a `collect_params`  method that returns all Parameters in a single dictionary.
We can invoke `collect_params` on a single layer  or a whole network as follows:
-->

Khi ta cần phải thực hiện các phép toán với tất cả tham số, việc truy cập lần lượt từng tham số sẽ trở nên khá khó chịu.
Việc này sẽ càng chậm chạp khi ta làm việc với các khối phức tạp hơn, ví dụ như các khối lồng nhau vì lúc đó ta sẽ phải duyệt toàn bộ cây bằng đệ quy để có thể trích xuất tham số của từng khối con.
Để tránh vấn đề này, mỗi khối có thêm một phương thức `collect_params` để trả về một từ điển duy nhất chứa tất cả tham số.
Ta có thể gọi `collect_params` với một tầng duy nhất hoặc với toàn bộ mạng nơ-ron như sau:

```{.python .input  n=6}
# parameters only for the first layer
print(net[0].collect_params())
# parameters of the entire network
print(net.collect_params())
```

<!--
This provides us with a third way of accessing the parameters of the network. 
If we wanted to get the value of the bias term of the second layer we could simply use this:
-->

<!-- UPDATE
This provides us with a third way of accessing the parameters of the network:
-->

Từ đó, ta có cách thứ ba để truy cập các tham số của mạng:

```{.python .input  n=7}
net.collect_params()['dense1_bias'].data()
```

<!--
Throughout the book we will see how various blocks name their subblocks (Sequential simply numbers them). 
This makes it very convenient to use regular expressions to filter out the required parameters.
-->

<!-- UPDATE
Throughout the book we encounter Blocks that name their sub-Blocks in various ways. 
Sequential simply numbers them.
We can exploit this naming convention by leveraging one clever feature of `collect_params`: it allows us to filter the parameters returned by using regular expressions.
-->

Xuyên suốt cuốn sách này ta sẽ thấy các khối đặt tên cho khối con theo nhiều cách khác nhau . 
Khối Sequential chỉ đơn thuần đánh số chúng. 
Ta có thể tận dụng quy ước định danh này cùng với một tính năng thông minh của `collect_params` để lọc ra các tham số được trả về bằng các biểu thức chính quy.

```{.python .input  n=8}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

<!--
### Rube Goldberg Striking Again
-->

<!-- UPDATE
### Collecting Parameters from Nested Blocks
-->

### Thu thập Tham số từ Khối lồng nhau 

<!--
Let's see how the parameter naming conventions work if we nest multiple blocks inside each other. 
For that we first define a function that produces blocks (a block factory, so to speak) and then we combine these inside yet larger blocks.
-->

<!-- UPDATE
Let us see how the parameter naming conventions work if we nest multiple blocks inside each other. 
For that we first define a function that produces Blocks (a Block factory, so to speak) and then combine these inside yet larger Blocks.
-->

Hãy cùng xem cách hoạt động của các quy ước định danh tham số khi ta lồng nhiều khối vào nhau.
Trước hết ta định nghĩa một hàm tạo khối (có thể gọi là một nhà máy khối) và rồi kết hợp chúng trong các khối lớn hơn.

```{.python .input  n=20}
def block1():
    net = nn.Sequential()
    net.add(nn.Dense(32, activation='relu'))
    net.add(nn.Dense(16, activation='relu'))
    return net

def block2():
    net = nn.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = nn.Sequential()
rgnet.add(block2())
rgnet.add(nn.Dense(10))
rgnet.initialize()
rgnet(x)
```

<!--
Now that we are done designing the network, let's see how it is organized. 
`collect_params` provides us with this information, both in terms of naming and in terms of logical structure.
-->

<!-- UPDATE
Now that we have designed the network, let us see how it is organized.
Notice below that while `collect_params()` produces a list of named parameters, invoking `collect_params` as an attribute reveals our network's structure.
-->

Bây giờ ta đã xong phần thiết kế mạng, hãy cùng xem cách nó được tổ chức.
Hãy để ý ở dưới rằng dù `collect_params()` tạo ra một danh sách các tham số được định danh, việc gọi `collect_params` như một thuộc tính sẽ tiết lộ cấu trúc của mạng.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

<!--
Since the layers are hierarchically generated, we can also access them accordingly. 
For instance, to access the first major block, within it the second subblock and then within it, in turn the bias of the first layer, we perform the following.
-->

<!-- UPDATE
Since the layers are hierarchically nested, we can also access them as though indexing through nested lists. 
For instance, we can access the first major block, within it the second subblock, and within that the bias of the first layer, with as follows:
-->

Bởi vì các tầng được sinh ra theo cơ chế phân cấp, ta cũng có thể truy cập chúng theo cách này.
Chẳng hạn,  ta có thể truy cập khối chính đầu tiên, khối con thứ hai bên trong nó và hệ số điều chỉnh của tầng đầu tiên bên trong nữa như sau:

```{.python .input}
rgnet[0][1][0].bias.data()
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Parameter Initialization
-->

## Khởi tạo Tham số

<!--
Now that we know how to access the parameters, let's look at how to initialize them properly. 
We discussed the need for initialization in :numref:`sec_numerical_stability`. 
By default, MXNet initializes the weight matrices uniformly by drawing from $U[-0.07, 0.07]$ and the bias parameters are all set to $0$. 
However, we often need to use other methods to initialize the weights. 
MXNet's `init` module provides a variety of preset initialization methods, but if we want something out of the ordinary, we need a bit of extra work.
-->

<!-- UPDATE
Now that we know how to access the parameters, let us look at how to initialize them properly.
We discussed the need for initialization in :numref:`sec_numerical_stability`. 
By default, MXNet initializes weight matrices uniformly by drawing from $U[-0.07, 0.07]$ and the bias parameters are all set to $0$.
However, we will often want to initialize our weights according to various other protocols. 
MXNet's `init` module provides a variety of preset initialization methods.
If we want to create a custom initializer, we need to do some extra work.
-->

Bây giờ khi đã biết cách truy cập tham số, hãy cùng xem xét việc khởi tạo chúng đúng cách.
Ta đã thảo luận về sự cần thiết của việc khởi tạo tham số trong :numref:`sec_numerical_stability`.
Theo mặc định, MXNet khởi tạo các ma trận trọng số bằng cách lấy mẫu từ phân phối đều $U[-0,07, 0,07]$ và đặt tất cả các hệ số điều chỉnh bằng $0$.
Tuy nhiên, thường ta sẽ muốn khởi tạo trọng số theo nhiều phương pháp khác.
Mô-đun `init` của MXNet cung cấp sẵn nhiều phương thức khởi tạo.
Nếu ta muốn tạo một bộ khởi tạo tùy chỉnh, ta sẽ cần làm việc thêm một chút.


<!--
### Built-in Initialization
-->

### Phương thức Khởi tạo có sẵn

<!--
Let's begin with the built-in initializers. 
The code below initializes all parameters with Gaussian random variables.
-->

<!-- UPDATE
Let us begin by calling on built-in initializers. 
The code below initializes all parameters as Gaussian random variables with standard deviation $0.01$.
-->

Ta sẽ bắt đầu với việc gọi các bộ khởi tạo có sẵn.
Đoạn mã dưới đây khởi tạo tất cả các tham số với các biến ngẫu nhiên Gauss có độ lệch chuẩn bằng $0.01$. 

```{.python .input  n=9}
# force_reinit ensures that variables are freshly initialized
# even if they were already initialized previously
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
net[0].weight.data()[0]
```

<!--
If we wanted to initialize all parameters to 1, we could do this simply by changing the initializer to `Constant(1)`.
-->

<!-- UPDATE
We can also initialize all parameters to a given constant value (say, $1$), by using the initializer `Constant(1)`.
-->

Ta cũng có thể khởi tạo tất cả tham số với một hằng số (ví dụ như $1$) bằng cách sử dụng bộ khởi tạo `Constant(1)`.

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

<!--
If we want to initialize only a specific parameter in a different manner, we can simply set the initializer only for the appropriate subblock (or parameter) for that matter. 
For instance, below we initialize the second layer to a constant value of 42 and we use the `Xavier` initializer for the weights of the first layer.
-->

<!-- UPDATE
We can also apply different initialziers for certain Blocks.
For example, below we initialize the first layer with the `Xavier` initializer and initialize the second layer to a constant value of 42.
-->

Ta còn có thể áp dụng các bộ khởi tạo khác nhau cho các khối khác nhau.
Ví dụ, trong đoạn mã nguồn bên dưới, ta khởi tạo tầng đầu tiên bằng cách sử dụng bộ khởi tạo `Xavier` và khởi tạo tầng thứ hai với một hằng số là 42.

```{.python .input  n=11}
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
net[1].initialize(init=init.Constant(42), force_reinit=True)
print(net[1].weight.data()[0, 0])
```

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Custom Initialization
-->

### Phương thức Khởi tạo tùy chỉnh

<!--
Sometimes, the initialization methods we need are not provided in the `init` module. 
At this point, we can implement a subclass of the `Initializer` class so that we can use it like any other initialization method. 
Usually, we only need to implement the `_init_weight` function and modify the incoming `ndarray` according to the initial result. 
In the example below, we  pick a decidedly bizarre and nontrivial distribution, just to prove the point. 
We draw the coefficients from the following distribution:
-->

<!-- UPDATE
Sometimes, the initialization methods we need are not provided in the `init` module. 
In these cases, we can define a subclass of `Initializer`. 
Usually, we only need to implement the `_init_weight` function which takes an `ndarray` argument (`data`) and assigns to it the desired initialized values. 
In the example below, we define an initializer for the following strange distribution:
-->

Đôi khi, các phương thức khởi tạo mà ta cần không có sẵn trong mô-đun `init`.
Trong trường hợp đó, ta có thể lập trình một lớp con của lớp `Initializer` và sử dụng nó như bất kỳ phương thức khởi tạo nào khác.
Thông thường, ta chỉ cần lập trình hàm `_init_weight` để thay đổi đối số `ndarray` đầu vào bằng giá trị khởi tạo mong muốn.
Trong ví dụ bên dưới, ta sẽ tự tạo một phân phối để chứng minh luận điểm trên.
Ta sẽ lấy các hệ số từ phân phối sau:

$$
\begin{aligned}
    w \sim \begin{cases}
        U[5, 10] & \text{ với xác suất } \frac{1}{4} \\
            0    & \text{ với xác suất } \frac{1}{2} \\
        U[-10, -5] & \text{ với xác suất } \frac{1}{4}
    \end{cases}
\end{aligned}
$$

```{.python .input  n=12}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        data[:] = np.random.uniform(-10, 10, data.shape)
        data *= np.abs(data) >= 5

net.initialize(MyInit(), force_reinit=True)
net[0].weight.data()[0]
```

<!--
If even this functionality is insufficient, we can set parameters directly. 
Since `data()` returns an `ndarray` we can access it just like any other matrix. 
A note for advanced users: if you want to adjust parameters within an `autograd` scope you need to use `set_data` to avoid confusing the automatic differentiation mechanics.
-->

<!-- UPDATE
Note that we always have the option of setting parameters directly by calling `data()` to access the underlying `ndarray`. 
A note for advanced users: if you want to adjust parameters within an `autograd` scope, you need to use `set_data` to avoid confusing the automatic differentiation mechanics.
-->

Nếu thậm chí tính năng này vẫn là chưa đủ thì ta có thể đặt các tham số một cách trực tiếp.
Do hàm `data()` trả về một mảng `ndarray` nên ta có thể truy cập nó giống như bất kỳ ma trận nào khác.
Một lưu ý cho người dùng nâng cao: nếu muốn điều chỉnh các tham số trong phạm vi `autograd`, bạn cần sử dụng `set_data` để tránh làm rối loạn các cơ chế tính vi phân tự động.

```{.python .input  n=13}
net[0].weight.data()[:] += 1
net[0].weight.data()[0, 0] = 42
net[0].weight.data()[0]
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Tied Parameters
-->

## Các Tham số bị Trói buộc

<!--
In some cases, we want to share model parameters across multiple layers. 
For instance when we want to find good word embeddings we may decide to use the same parameters both for encoding and decoding of words. 
We discussed one such case when we introduced :numref:`sec_model_construction`. 
Let's see how to do this a bit more elegantly. In the following we allocate a dense layer and then use its parameters specifically to set those of another layer.
-->

<!-- UPDATE
Often, we want to share parameters across multiple layers.
Later we will see that when learning word embeddings, it might be sensible to use the same parameters both for encoding and decoding words. 
We discussed one such case when we introduced :numref:`sec_model_construction`. 
Let us see how to do this a bit more elegantly. In the following we allocate a dense layer and then use its parameters specifically to set those of another layer.
-->

Trong một số trường hợp, ta sẽ muốn chia sẻ các tham số mô hình trên nhiều tầng.
Ví dụ, trong quá trình huấn luyện embedding từ, ta có thể quyết định sử dụng cùng một bộ tham số để mã hóa và giải mã các từ.
Ta đã thảo luận về một trường hợp như vậy trong :numref:`sec_model_construction`.
Hãy xem làm thế nào để thực hiện việc này một cách tinh tế hơn. Sau đây ta sẽ tạo một tầng kết nối đầy đủ và sử dụng chính tham số của nó làm tham số cho một tầng khác.


```{.python .input  n=14}
net = nn.Sequential()
# We need to give the shared layer a name such that we can reference its
# parameters
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

x = np.random.uniform(size=(2, 20))
net(x)

# Check whether the parameters are the same
print(net[1].weight.data()[0] == net[2].weight.data()[0])
net[1].weight.data()[0, 0] = 100
# Make sure that they are actually the same object rather than just having the
# same value
print(net[1].weight.data()[0] == net[2].weight.data()[0])
```

<!--
The above example shows that the parameters of the second and third layer are tied. 
They are identical rather than just being equal. 
That is, by changing one of the parameters the other one changes, too. 
What happens to the gradients is quite ingenious. 
Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are accumulated in the `shared.params.grad( )` during backpropagation.
-->

<!-- UPDATE
This example shows that the parameters  of the second and third layer are tied. 
They are not just equal, they are  represented by the same exact `ndarray`. 
Thus, if we change one of the parameters, the other one changes, too. 
You might wonder, *when parameters are tied what happens to the gradients?*
Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are added together in `shared.params.grad( )` during backpropagation.
-->

Ví dụ trên cho thấy các tham số của tầng thứ hai và thứ ba đã bị trói buộc với nhau.
Thay vì chỉ có giá trị bằng nhau, chúng là cùng một đối tượng.
Tức là nếu thay đổi các tham số của tầng này này thì các tham số của tầng kia cũng sẽ thay đổi theo.
Cách xử lý gradient ở đây là khá tài tình.
Vì các tham số mô hình chứa gradient nên gradient của tầng ẩn thứ hai và tầng ẩn thứ ba được cộng lại trong `shared.params.grad( )` trong quá trình lan truyền ngược.

<!--
## Summary
-->

## Tóm tắt

<!--
* We have several ways to access, initialize, and tie model parameters.
* We can use custom initialization.
* Gluon has a sophisticated mechanism for accessing parameters in a unique and hierarchical manner.
-->

* Ta có một số cách để truy cập, khởi tạo và trói buộc các tham số mô hình.
* Ta có thể sử dụng các bộ khởi tạo tùy chỉnh.
* Gluon có một cơ chế tinh vi để truy cập các tham số theo phân cấp một cách độc nhất.


<!--
## Exercises
-->

## Bài tập

<!--
1. Use the FancyMLP defined in :numref:`sec_model_construction` and access the parameters of the various layers.
2. Look at the [MXNet documentation](http://beta.mxnet.io/api/gluon-related/mxnet.initializer.html) and explore different initializers.
3. Try accessing the model parameters after `net.initialize()` and before `net(x)` to observe the shape of the model parameters. What changes? Why?
4. Construct a multilayer perceptron containing a shared parameter layer and train it. During the training process, observe the model parameters and gradients of each layer.
5. Why is sharing parameters a good idea?
-->

1. Sử dụng FixedHiddenMLP được định nghĩa trong :numref:`sec_model_construction` và truy cập tham số của các tầng khác nhau. <!-- Trong `sec_model_construction` mình chỉ thấy có hàm FixedHiddenMLP chứ không có hàm FancyMLP, hình như FancyMLP là trong bản cũ của sách thì phải -->
2. Xem [tài liệu của MXNet](http://beta.mxnet.io/api/gluon-related/mxnet.initializer.html) và nghiên cứu các bộ khởi tạo khác nhau.
3. Thử truy cập các tham số mô hình sau khi gọi `net.initialize()` và trước khi gọi `net(x)` và quan sát kích thước của chúng. Điều gì đã thay đổi? Tại sao?
4. Xây dựng và huấn luyện một perceptron đa tầng trong đó có một tầng sử dụng tham số được chia sẻ. Trong quá trình huấn luyện, hãy quan sát các tham số mô hình và gradient của từng tầng.
5. Tại sao việc chia sẻ tham số lại là là một ý tưởng hay?

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2326)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2326)
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
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
* Lê Cao Thăng
* Nguyễn Duy Du
* Phạm Hồng Vinh
* Phạm Minh Đức

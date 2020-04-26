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

Một khi ta đã chọn được kiến trúc mạng và các giá trị siêu tham số, ta sẽ bắt đầu với vòng lặp huấn luyện với mục tiêu là tìm các giá trị tham số để tối thiểu hóa hàm mục tiêu.
Sau khi huấn luyện xong, ta sẽ cần các tham số đó để đưa ra dự đoán trong tương lai.
Hơn nữa, thi thoảng ta sẽ muốn trích xuất tham số để sử dụng lại trong một hoàn cảnh khác, có thể lưu trữ mô hình để thực thi trong một phần mềm khác hoặc để rút ra hiểu biết khoa học bằng việc phân tích mô hình.

<!--
Most of the time, we will be able to ignore the nitty-gritty details of how parameters are declared and manipulated, relying on Gluon to do the heavy lifting.
However, when we move away from stacked architectures with standard layers, we will sometimes need to get into the weeds of declaring and manipulate parameters. 
In this section, we cover the following:
-->

Thông thường, ta có thể bỏ qua những chi tiết chuyên sâu về việc khai báo và xử lý tham số bởi Gluon sẽ đảm nhiệm công việc nặng nhọc này.
Tuy nhiên, khi ta bắt đầu tiến xa hơn những kiến trúc chỉ gồm các tầng tiêu chuẩn được xếp chồng lên nhau, đôi khi ta sẽ phải tự đi sâu vào việc khai báo và xử lý tham số.
Trong mục này, chúng tôi sẽ đề cập những việc sau:

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

* Truy cập các tham số để gỡ lỗi, chẩn đoán mô hình và biểu diễn trực quan.
* Khởi tạo tham số.
* Chia sẻ tham số giữa các thành phần khác nhau của mô hình.

<!--
As always, we start from our trusty Multilayer Perceptron with a hidden layer. This will serve as our choice for demonstrating the various features.
-->

<!-- UPDATE
We start by focusing on an MLP with one hidden layer.
-->

Chúng ta sẽ bắt đầu từ mạng Perceptron đa tầng với một tầng ẩn.

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

Hãy bắt đầu với việc truy cập tham số của những mô hình mà bạn đã biết.
Khi một mô hình được định nghĩa bằng lớp Tuần tự (*Sequential*), ta có thể truy cập bất kỳ tầng nào bằng chỉ số, như thể nó là một danh sách.
Thuộc tính `params` của mỗi tầng chứa tham số của chúng.
Ta có thể quan sát các tham số của mạng `net` định nghĩa ở trên.


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

Kết quả của đoạn mã này cho ta một vài thông tin quan trọng.
Đầu tiên, mỗi tầng kết nối đầy đủ đều có hai tập tham số, ví dụ như `dense0_weight` và `dense0_bias` tương ứng với trọng số và hệ số điều chỉnh của tầng đó.
Chúng đều được lưu trữ ở dạng số thực dấu phẩy động độ chính xác đơn.
Lưu ý rằng tên của các tham số cho phép ta xác định tham số của từng tầng *một cách độc nhất*, kể cả khi mạng nơ-ron chứa hàng trăm tầng.
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

Lưu ý rằng mỗi tham số được biểu diễn bằng một thực thể của lớp `Parameter`.
Để làm việc với các tham số, trước hết ta phải truy cập được các giá trị số của chúng.
Có một vài cách để làm việc này, một số cách đơn giản hơn trong khi các cách khác lại tổng quát hơn.
Để bắt đầu, ta có thể truy cập tham số của một tầng thông qua thuộc tính `bias` hoặc `weight` rồi sau đó truy cập giá trị số của chúng thông qua phương thức `data()`.
Đoạn mã sau trích xuất hệ số điều chỉnh của tầng thứ hai trong mạng nơ-ron.

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

Tham số là các đối tượng khá phức tạp bởi chúng chứa dữ liệu, gradient và một vài thông tin khác.
Đó là lí do tại sao ta cần yêu cầu dữ liệu một cách tường minh.
Lưu ý rằng vector hệ số điều chỉnh chứa các giá trị không vì ta chưa hề cập nhật mô hình kể từ khi nó được khởi tạo.
Ta cũng có thể truy cập các tham số theo tên của chúng, chẳng hạn như `dense0_weight` ở dưới.
Điều này khả thi vì thực ra mỗi tầng đều chứa một từ điển tham số.

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

Chú ý rằng khác với hệ số điều chỉnh, trọng số chứa các giá trị khác không bởi chúng được khởi tạo ngẫu nhiên.
Ngoài `data`, mỗi `Parameter` còn cung cấp phương thức `grad()` để truy cập gradient.
Gradient sẽ có cùng kích thước với trọng số. <!-- thế còn bias thì sao ._. -->
Vì ta chưa thực hiện lan truyền ngược với mạng nơ-ron này, các giá trị của gradient đều là 0.

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

Truy cập các tham số như mô tả phía trên có thể hơi dài dòng với các khối phức tạp, chẳng hạn như khi ta có khối của các khối (hoặc thậm chí nhiều khối của các khối của các khối), vì ta cần phải duyệt qua toàn bộ cây theo thứ tự ngược với cách các khối được xây dựng.
Để tránh rắc rối này, các khối có thêm một phương thức `collect_params` giúp tập hợp tất cả các tham số có trong mạng thành một từ điển để ta có thể dễ dàng duyệt qua.
Nó thực hiện điều này bằng cách lặp qua các thành phần của một khối và gọi `collect_params` trên các khối con khi cần thiết.
Để thấy được sự khác nhau ta hãy xem ví dụ sau:

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

Đây là cách thứ ba để truy cập các tham số của mạng.
Nếu muốn lấy giá trị của hệ số điều chỉnh của tầng thứ hai, đơn giản ta có thể dùng: 

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

Xuyên suốt cuốn sách này ta sẽ thấy cách các loại khối khác nhau định danh khối con của chúng (khối Sequential đơn giản là đánh số các khối con).
Điều này làm cho việc sử dụng các biểu thức chính quy (_regular expression_) để lọc ra các tham số cần thiết thuận tiện rất nhiều.

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

### Rube Goldberg Lại Nổi lên

<!--
Let's see how the parameter naming conventions work if we nest multiple blocks inside each other. 
For that we first define a function that produces blocks (a block factory, so to speak) and then we combine these inside yet larger blocks.
-->

<!-- UPDATE
Let us see how the parameter naming conventions work if we nest multiple blocks inside each other. 
For that we first define a function that produces Blocks (a Block factory, so to speak) and then combine these inside yet larger Blocks.
-->

Hãy cùng xem cách hoạt động của các quy ước định danh tham số khi ta lồng nhiều khối vào nhau.
Trước hết ta định nghĩa một hàm tạo khối (có thể tạm gọi là một nhà máy khối) và sau đó ta kết hợp chúng vào bên trong các khối còn lớn hơn.

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

Bây giờ ta đã xong phần thiết kế mạng, hãy xem cách nó được tổ chức.
`collect_params` cung cấp chúng ta thông tin này, cả về cách định danh lẫn cấu trúc logic.

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

Bởi vì các tầng được sinh ra theo cơ chế phân cấp, ta có thể truy cập chúng theo cách này.
Chẳng hạn, để truy cập khối chính đầu tiên, bên trong nó là khối con thứ hai và tiếp theo bên trong nó, trong trường hợp này là hệ số điều chỉnh của tầng đầu tiên, ta thực hiện như sau.

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

Bây giờ khi đã biết cách truy cập tham số, ta sẽ xem làm thế nào để khởi tạo chúng đúng cách.
Ta đã thảo luận về sự cần thiết của việc khởi tạo tham số trong :numref:`sec_numerical_stability`.
Theo mặc định, MXNet khởi tạo các ma trận trọng số bằng cách lấy mẫu từ phân phối đều $U[-0,07, 0,07]$ và đặt tất cả các hệ số điều chỉnh bằng $0$.
Tuy nhiên, ta thường cần sử dụng các phương pháp khác để khởi tạo trọng số.
Mô-đun `init` của MXNet đã cung cấp sẵn nhiều phương thức khởi tạo, nhưng nếu muốn một cái gì đó khác thường, ta sẽ cần làm việc thêm một chút.


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

Ta sẽ bắt đầu với các bộ khởi tạo có sẵn.
Mã nguồn dưới đây khởi tạo tất cả các tham số với các biến ngẫu nhiên từ phân phối Gauss.

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

Nếu muốn khởi tạo tất cả các tham số bằng 1, ta có thể đơn thuần thay bộ khởi tạo thành `Constant(1)`.

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

Nếu muốn khởi tạo một tham số cụ thể theo một cách riêng biệt, ta có thể đơn thuần sử dụng một bộ khởi tạo riêng cho khối con (hay tham số) tương ứng.
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

### Phương thức Khởi tạo Tùy chỉnh

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
Trong trường hợp đó, ta có thể khai báo một lớp con của lớp `Initializer`.
Thông thường, ta chỉ cần lập trình hàm `_init_weight` nhận một đối số `ndarray` (`data`) và gán cho nó giá trị khởi tạo mong muốn.
Trong ví dụ bên dưới, ta sẽ khai báo một bộ khởi tạo cho phân phối kì lạ sau:

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

Lưu ý rằng ta luôn có thể đặt trực tiếp giá trị của các tham số bằng cách gọi hàm `data()` để truy cập `ndarray` của tham số đó.
Một lưu ý khác cho người dùng nâng cao: nếu muốn điều chỉnh các tham số trong phạm vi của `autograd`, bạn cần sử dụng hàm `set_data` để tránh làm rối loạn cơ chế tính vi phân tự động.

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

Thông thường, ta sẽ muốn chia sẻ các tham số mô hình cho nhiều tầng.
Sau này ta sẽ thấy trong quá trình huấn luyện embedding từ, việc sử dụng cùng một bộ tham số để mã hóa và giải mã các từ có thể khá hợp lý.
Ta đã thảo luận về một trường hợp như vậy trong :numref:`sec_model_construction`.
Hãy cùng xem làm thế nào để thực hiện việc này một cách tinh tế hơn. Sau đây ta sẽ tạo một tầng kết nối đầy đủ và sử dụng chính tham số của nó làm tham số cho một tầng khác.


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

Ví dụ này cho thấy các tham số của tầng thứ hai và thứ ba đã bị trói buộc với nhau.
Chúng không chỉ có giá trị bằng nhau, chúng được biểu diễn bởi cùng một `ndarray`. 
Vì vậy, nếu ta thay đổi các tham số của tầng này này thì các tham số của tầng kia cũng sẽ thay đổi theo.
Bạn có thể tự hỏi rằng *chuyện gì sẽ xảy ra với gradient khi các tham số bị trói buộc?*.
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

* Ta có vài cách để truy cập, khởi tạo và trói buộc các tham số mô hình.
* Ta có thể sử dụng các phương thức khởi tạo tùy chỉnh.
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

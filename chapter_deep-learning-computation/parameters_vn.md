<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Parameter Management
-->

# Quản lý Tham số

<!--
Once we have chosen an architecture and set our hyperparameters, we proceed to the training loop, where our goal is to find parameter values that minimize our objective function. 
After training, we will need these parameters in order to make future predictions.
Additionally, we will sometimes wish to extract the parameters either to reuse them in some other context,
to save our model to disk so that it may be exectuted in other software,or for examination in the hopes of gaining scientific understanding.
-->

Một khi ta đã chọn được kiến trúc mạng và các giá trị siêu tham số, ta sẽ bắt đầu với vòng lặp huấn luyện với mục tiêu là tìm các giá trị tham số để cực tiểu hóa hàm mục tiêu.
Sau khi huấn luyện xong, ta sẽ cần các tham số đó để đưa ra dự đoán trong tương lai.
Hơn nữa, thi thoảng ta sẽ muốn trích xuất tham số để sử dụng lại trong một hoàn cảnh khác, có thể lưu trữ mô hình để thực thi trong một phần mềm khác hoặc để rút ra hiểu biết khoa học bằng việc phân tích mô hình.

<!--
Most of the time, we will be able to ignore the nitty-gritty details of how parameters are declared and manipulated, relying on Gluon to do the heavy lifting.
However, when we move away from stacked architectures with standard layers, we will sometimes need to get into the weeds of declaring and manipulate parameters. 
In this section, we cover the following:
-->

Thông thường, ta có thể bỏ qua những chi tiết chuyên sâu về việc khai báo và xử lý tham số bởi Gluon sẽ đảm nhiệm công việc nặng nhọc này.
Tuy nhiên, khi ta bắt đầu tiến xa hơn những kiến trúc chỉ gồm các tầng tiêu chuẩn được xếp chồng lên nhau, đôi khi ta sẽ phải tự đi sâu vào việc khai báo và xử lý tham số.
Trong mục này, chúng tôi sẽ đề cập đến những việc sau:

<!--
* Accessing parameters for debugging, diagnostics, and visualiziations.
* Parameter initialization.
* Sharing parameters across different model components.
-->

* Truy cập các tham số để gỡ lỗi, chẩn đoán mô hình và biểu diễn trực quan.
* Khởi tạo tham số.
* Chia sẻ tham số giữa các thành phần khác nhau của mô hình.

<!--
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
Parameters are complex objects, containing data, gradients, and additional information.
That's why we need to request the data explicitly.
Note that the bias vector consists of zeroes because we have not updated the network since it was initialized.
We can also access each parameter by name, e.g., `dense0_weight` as follows. 
Under the hood this is possible because each layer contains a parameter dictionary. 
-->

Tham số là các đối tượng khá phức tạp bởi chúng chứa dữ liệu, gradient và một vài thông tin khác.
Đó là lý do tại sao ta cần yêu cầu dữ liệu một cách tường minh.
Lưu ý rằng vector hệ số điều chỉnh chứa các giá trị không vì ta chưa hề cập nhật mô hình kể từ khi nó được khởi tạo.
Ta cũng có thể truy cập các tham số theo tên của chúng, chẳng hạn như `dense0_weight` ở dưới.
Điều này khả thi vì thực ra mỗi tầng đều chứa một từ điển tham số.

```{.python .input  n=4}
print(net[0].params['dense0_weight'])
print(net[0].params['dense0_weight'].data())
```

<!--
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
This provides us with a third way of accessing the parameters of the network:
-->

Từ đó, ta có cách thứ ba để truy cập các tham số của mạng:

```{.python .input  n=7}
net.collect_params()['dense1_bias'].data()
```

<!--
Throughout the book we encounter Blocks that name their sub-Blocks in various ways. 
Sequential simply numbers them.
We can exploit this naming convention by leveraging one clever feature of `collect_params`: it allows us to filter the parameters returned by using regular expressions.
-->

Xuyên suốt cuốn sách này ta sẽ thấy các khối đặt tên cho khối con theo nhiều cách khác nhau. 
Khối Sequential chỉ đơn thuần đánh số chúng. 
Ta có thể tận dụng quy ước định danh này cùng với một tính năng thông minh của `collect_params` để lọc ra các tham số được trả về bằng các biểu thức chính quy (_regular expression_).

```{.python .input  n=8}
print(net.collect_params('.*weight'))
print(net.collect_params('dense0.*'))
```

<!--
### Collecting Parameters from Nested Blocks
-->

### Thu thập Tham số từ các Khối lồng nhau 

<!--
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
Now that we have designed the network, let us see how it is organized.
Notice below that while `collect_params()` produces a list of named parameters, invoking `collect_params` as an attribute reveals our network's structure.
-->

Bây giờ ta đã xong phần thiết kế mạng, hãy cùng xem cách nó được tổ chức.
Hãy để ý ở dưới rằng dù hàm `collect_params()` trả về một danh sách các tham số được định danh, việc gọi `collect_params` như một thuộc tính sẽ cho ta biết cấu trúc của mạng.

```{.python .input}
print(rgnet.collect_params)
print(rgnet.collect_params())
```

<!--
Since the layers are hierarchically nested, we can also access them as though indexing through nested lists. 
For instance, we can access the first major block, within it the second subblock, and within that the bias of the first layer, with as follows:
-->

Bởi vì các tầng được lồng vào nhau theo cơ chế phân cấp, ta cũng có thể truy cập chúng tương tự như cách ta dùng chỉ số để truy cập các danh sách lồng nhau.
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
Nếu ta muốn một bộ khởi tạo tùy chỉnh, ta sẽ cần làm thêm một chút việc.


<!--
### Built-in Initialization
-->

### Phương thức Khởi tạo có sẵn

<!--
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
We can also initialize all parameters to a given constant value (say, $1$), by using the initializer `Constant(1)`.
-->

Ta cũng có thể khởi tạo tất cả tham số với một hằng số (ví dụ như $1$) bằng cách sử dụng bộ khởi tạo `Constant(1)`.

```{.python .input  n=10}
net.initialize(init=init.Constant(1), force_reinit=True)
net[0].weight.data()[0]
```

<!--
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

### Phương thức Khởi tạo Tùy chỉnh

<!--
Sometimes, the initialization methods we need are not provided in the `init` module. 
In these cases, we can define a subclass of `Initializer`. 
Usually, we only need to implement the `_init_weight` function which takes an `ndarray` argument (`data`) and assigns to it the desired initialized values. 
In the example below, we define an initializer for the following strange distribution:
-->

Đôi khi, các phương thức khởi tạo mà ta cần không có sẵn trong mô-đun `init`.
Trong trường hợp đó, ta có thể khai báo một lớp con của lớp `Initializer`.
Thông thường, ta chỉ cần lập trình hàm `_init_weight` để nhận một đối số `ndarray` (`data`) và gán giá trị khởi tạo mong muốn cho nó.
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
Note that we always have the option of setting parameters directly by calling `data()` to access the underlying `ndarray`. 
A note for advanced users: if you want to adjust parameters within an `autograd` scope, you need to use `set_data` to avoid confusing the automatic differentiation mechanics.
-->

Lưu ý rằng ta luôn có thể trực tiếp đặt giá trị cho tham số bằng cách gọi hàm `data()` để truy cập `ndarray` của tham số đó.
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
This example shows that the parameters  of the second and third layer are tied. 
They are not just equal, they are  represented by the same exact `ndarray`. 
Thus, if we change one of the parameters, the other one changes, too. 
You might wonder, *when parameters are tied what happens to the gradients?*
Since the model parameters contain gradients, the gradients of the second hidden layer and the third hidden layer are added together in `shared.params.grad( )` during backpropagation.
-->

Ví dụ này cho thấy các tham số của tầng thứ hai và thứ ba đã bị trói buộc với nhau.
Chúng không chỉ có giá trị bằng nhau, chúng còn được biểu diễn bởi cùng một `ndarray`. 
Vì vậy, nếu ta thay đổi các tham số của tầng này này thì các tham số của tầng kia cũng sẽ thay đổi theo.
Bạn có thể tự hỏi rằng *chuyện gì sẽ xảy ra với gradient khi các tham số bị trói buộc?*.
Vì các tham số mô hình chứa gradient nên gradient của tầng ẩn thứ hai và tầng ẩn thứ ba được cộng lại tại `shared.params.grad( )` trong quá trình lan truyền ngược.

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


1. Sử dụng FancyMLP được định nghĩa trong :numref:`sec_model_construction` và truy cập tham số của các tầng khác nhau.
2. Xem [tài liệu MXNet](http://beta.mxnet.io/api/gluon-related/mxnet.initializer.html) và nghiên cứu các bộ khởi tạo khác nhau.
3. Thử truy cập các tham số mô hình sau khi gọi `net.initialize()` và trước khi gọi `net(x)` và quan sát kích thước của chúng. Điều gì đã thay đổi? Tại sao?
4. Xây dựng và huấn luyện một perceptron đa tầng mà trong đó có một tầng sử dụng tham số được chia sẻ. Trong quá trình huấn luyện, hãy quan sát các tham số mô hình và gradient của từng tầng.
5. Tại sao việc chia sẻ tham số lại là là một ý tưởng hay?


<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2326)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
* Lê Cao Thăng
* Nguyễn Duy Du
* Phạm Hồng Vinh
* Phạm Minh Đức

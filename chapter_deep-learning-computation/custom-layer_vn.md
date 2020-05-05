<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Custom Layers
-->

# Các tầng Tuỳ chỉnh

<!--
One of factors behind deep learnings success is the availability of a wide range of layers that can be composed in creative ways to design architectures suitable for a wide variety of tasks.
For instance, researchers have invented layers specifically for handling images, text, looping over sequential data, performing dynamic programming, etc.
Sooner or later you will encounter (or invent) a layer that does not exist yet in Gluon,
In these cases, you must build a custom layer.
In this section, we show you how.
-->

Một trong những yếu tố dẫn đến thành công của học sâu là sự đa dạng của các tầng. 
Những tầng này có thể được sắp xếp theo nhiều cách sáng tạo để thiết kế nên những kiến trúc phù hợp với nhiều tác vụ khác nhau. 
Ví dụ, các nhà nghiên cứu đã phát minh ra các tầng chuyên dụng để xử lý ảnh, chữ viết, lặp trên dữ liệu tuần tự, thực thi quy hoạch động, v.v...
Dù sớm hay muộn, bạn cũng sẽ gặp (hoặc sáng tạo) một tầng không có trong Gluon.
Đối với những trường hợp như vậy, bạn cần xây dựng một tầng tuỳ chỉnh. 
Phần này sẽ hướng dẫn bạn cách thực hiện điều đó.

<!--
## Layers without Parameters
-->

## Các tầng không có Tham số

<!--
To start, we construct a custom layer (a Block) that does not have any parameters of its own. 
This should look familiar if you recall our introduction to Gluon's `Block` in :numref:`sec_model_construction`. 
The following `CenteredLayer` class simply subtracts the mean from its input. 
To build it, we simply need to inherit from the Block class and implement the `forward` method.
-->

Để bắt đầu, ta tạo một tầng tùy chỉnh (một Khối) không chứa bất kỳ tham số nào.
Bước này khá quen thuộc nếu bạn còn nhớ phần giới thiệu về `Block` của Gluon tại :numref:`sec_model_construction`.
Lớp `CenteredLayer` chỉ đơn thuần trừ đi giá trị trung bình từ đầu vào của nó.
Để xây dựng nó, chúng ta chỉ cần kế thừa từ lớp `Block` và lập trình phương thức `forward`.

```{.python .input  n=1}
from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()
```

<!--
Let us verify that our layer works as intended by feeding some data through it.
-->

Hãy cùng xác thực rằng tầng này hoạt động như ta mong muốn bằng cách truyền dữ liệu vào nó.

```{.python .input  n=2}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

<!--
We can now incorporate our layer as a component in constructing more complex models.
-->

Chúng ta cũng có thể kết hợp tầng này như là một thành phần để xây dựng các mô hình phức tạp hơn.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

<!--
As an extra sanity check, we can send random data through the network and check that the mean is in fact 0.
Because we are dealing with floating point numbers, we may still see a *very* small nonzero number due to quantization.
-->

Để kiểm tra thêm, chúng ta có thể truyền dữ liệu ngẫu nhiên qua mạng và chứng thực xem giá trị trung bình đã về 0 hay chưa.
Chú ý rằng vì đang làm việc với các số thực dấu phẩy động, chúng ta sẽ thấy một giá trị khác không *rất* nhỏ.

```{.python .input  n=4}
y = net(np.random.uniform(size=(4, 8)))
y.mean()
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Layers with Parameters
-->

## Tầng có Tham số

<!--
Now that we know how to define simple layers let us move on to defining layers with parameters that can be adjusted through training. 
To automate some of the routine work the `Parameter` class and the `ParameterDict` dictionary provide some basic housekeeping functionality.
In particular, they govern access, initialization, sharing, saving and loading model parameters. 
This way, among other benefits, we will not need to write custom serialization routines for every custom layer.
-->

Giờ đây ta đã biết cách định nghĩa các tầng đơn giản, hãy chuyển sang việc định nghĩa các tầng chứa tham số có thể điều chỉnh được trong quá trình huấn luyện. 
Để tự động hóa các công việc lặp lại, lớp `Parameter` và từ điển `ParameterDict` cung cấp một số tính năng quản trị cơ bản. 
Cụ thể, chúng sẽ quản lý việc truy cập, khởi tạo, chia sẻ, lưu và nạp các tham số mô hình. 
Bằng cách này, cùng với nhiều lợi ích khác, ta không cần phải viết lại các thủ tục tuần tự hóa (_serialization_) cho mỗi tầng tùy chỉnh mới.

<!--
The `Block` class contains a `params` variable of the `ParameterDict` type. 
This dictionary maps strings representing parameter names to model parameters (of the `Parameter` type). 
The `ParameterDict` also supplied a `get` function that makes it easy to generate a new parameter with a specified name and shape.
-->

Lớp `Block` chứa biến `params` với kiểu dữ liệu `ParameterDict`.
Từ điển này ánh xạ các xâu kí tự biểu thị tên tham số đến các tham số mô hình (thuộc kiểu `Parameter`).
`ParameterDict` cũng cung cấp hàm `get` giúp việc tạo tham số mới với tên và chiều cụ thể trở nên dễ dàng.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

<!--
We now have all the basic ingredients that we need to implement our own version of Gluon's `Dense` layer. 
Recall that this layer requires two parameters, one to represent the weight and another for the bias. 
In this implementation, we bake in the ReLU activation as a default.
In the `__init__`, function, `in_units` and `units` denote the number of inputs and outputs, respectively.
-->

Giờ đây chúng ta đã có tất cả các thành phần cơ bản cần thiết để tự tạo một phiên bản tùy chỉnh của tầng `Dense` trong Gluon. 
Chú ý rằng tầng này yêu cầu hai tham số: một cho trọng số và một cho hệ số điều chỉnh. 
Trong cách lập trình này, ta sử dụng hàm kích hoạt mặc định là hàm ReLU. 
Trong hàm `__init__`, `in_units` và `units` biểu thị lần lượt số lượng đầu vào và đầu ra. 

```{.python .input  n=19}
class MyDense(nn.Block):
    # units: the number of outputs in this layer; in_units: the number of
    # inputs in this layer
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = np.dot(x, self.weight.data()) + self.bias.data()
        return npx.relu(linear)
```

<!--
Naming our parameters allows us to access them by name through dictionary lookup later.
Generally, you will want to give your variables simple names that make their purpose clear.
Next, we instantiate the `MyDense` class and access its model parameters.
Note that the Block's name is automatically prepended to each Parameter's name.
-->

Việc đặt tên cho các tham số cho phép ta truy cập chúng theo tên thông qua tra cứu từ điển sau này. 
Nhìn chung, bạn sẽ muốn đặt cho các biến những tên đơn giản biểu thị rõ mục đích của chúng.
Tiếp theo, ta sẽ khởi tạo lớp `MyDense` và truy cập các tham số mô hình.
Lưu ý rằng tên của Khối được tự động thêm vào trước tên các tham số.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

<!--
We can directly carry out forward calculations using custom layers.
-->

Ta có thể trực tiếp thực thi các phép tính truyền xuôi có sử dụng các tầng tùy chỉnh.


```{.python .input  n=20}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

<!--
We can also construct models using custom layers.
Once we have that we can use it just like the built-in dense layer.
The only exception is that in our case, shape inference is not automatic. 
If you are interested in these bells and whisteles, please consult the [MXNet documentation](http://www.mxnet.io) for details on how to implement shape inference in custom layers.
-->

Các tầng tùy chỉnh cũng có thể được dùng để xây dựng mô hình. 
Chúng có thể được sử dụng như các tầng kết nối dày đặc được lập trình sẵn.
Ngoại lệ duy nhất là việc suy luận kích thước sẽ không được thực hiện tự động.
Để biết thêm chi tiết về cách thực hiện việc này, vui lòng tham khảo [tài liệu MXNet](http://www.mxnet.io).

```{.python .input  n=19}
net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(np.random.uniform(size=(2, 64)))
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Summary
-->

## Tóm tắt

<!--
* We can design custom layers via the Block class. This allows us to define flexible new layers that behave differently from any existing layers in the library.
* Once defined, custom layers can be invoked in arbitrary contexts and architectures.
* Blocks can have local parameters, which are stored as a `ParameterDict` object in each Blovk's `params` attribute.
-->

* Ta có thể thiết kế các tầng tùy chỉnh thông qua lớp `Block`. Điều này cho phép ta định nghĩa một cách linh hoạt các tầng có cách hoạt động khác với các tầng có sẵn trong thư viện.
* Một khi đã được định nghĩa, các tầng tùy chỉnh có thể được gọi trong những bối cảnh và kiến trúc tùy ý.
* Các khối có thể có các tham số cục bộ, được lưu trữ dưới dạng đối tượng `ParameterDict` trong mỗi thuộc tính `params` của Block.

<!--
## Exercises
-->

## Bài tập

<!--
1. Design a layer that learns an affine transform of the data.
2. Design a layer that takes an input and computes a tensor reduction, i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
3. Design a layer that returns the leading half of the Fourier coefficients of the data. Hint: look up the `fft` function in MXNet.
-->

1. Thiết kế một tầng có khả năng học một phép biến đổi affine của dữ liệu. 
2. Thiết kế một tầng nhận đầu vào và tính toán phép giảm tensor, tức trả về $y_k = \sum_{i, j} W_{ijk} x_i x_j$. 
3. Thiết kế một tầng trả về nửa đầu của các hệ số Fourier của dữ liệu. Gợi ý: hãy tra cứu hàm `fft` trong MXNet. 

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2328)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Lê Quang Nhật
* Nguyễn Văn Cường
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
* Phạm Minh Đức
* Nguyễn Duy Du

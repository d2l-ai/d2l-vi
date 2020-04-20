<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Custom Layers
-->

# Các tầng Tuỳ chỉnh

<!--
One of the reasons for the success of deep learning can be found in the wide range of layers that can be used in a deep network. 
This allows for a tremendous degree of customization and adaptation. 
For instance, scientists have invented layers for images, text, pooling, loops, dynamic programming, even for computer programs. 
Sooner or later you will encounter a layer that does not exist yet in Gluon, or even better, you will eventually invent a new layer that works well for your problem at hand. 
This is when it is time to build a custom layer. This section shows you how.
-->

Một trong những lý do dẫn đến thành công của học sâu là sự đa dạng của các tầng có thể sử dụng trong một mạng.
Điều này cho phép một mức độ tuỳ chỉnh và thích ứng rất lớn.
Ví dụ, các nhà khoa học đã phát minh ra các tầng cho hình ảnh, chữ viết, gộp (*pooling*), vòng lặp, quy hoạch động, thậm chí cho cả các chương trình máy tính.
Dù sớm hay muộn, bạn cũng sẽ gặp một tầng không có trong Gluon, hay thậm chí tuyệt vời hơn, bạn sẽ phát minh ra một tầng mới hoạt động tốt mà có thể sử dụng ngay cho bài toán của mình.
Đó là khi bạn cần xây dựng một tầng tuỳ chỉnh. Phần này sẽ hướng dẫn bạn cách thực hiện việc này.

<!--
## Layers without Parameters
-->

## Các tầng không có Tham số

<!--
Since this is slightly intricate, we start with a custom layer (also known as Block) that does not have any inherent parameters. 
Our first step is very similar to when we introduced blocks in :numref:`sec_model_construction`. 
The following `CenteredLayer` class constructs a layer that subtracts the mean from the input.
We build it by inheriting from the Block class and implementing the `forward` method.
-->

Do việc này hơi phức tạp một chút, chúng ta sẽ bắt đầu với một tầng tuỳ chỉnh (hay còn gọi là Khối) mà tự thân không có bất kì tham số nào.
Bước đầu tiên rất giống với khi chúng ta giới thiệu các khối ở :numref:`sec_model_construction`.
Lớp `CenteredLayer` dưới đây xây dựng một tầng có chức năng trừ đi giá trị trung bình khỏi đầu vào.
Chúng ta xây dựng lớp này bằng việc kế thừa từ lớp `Block` và lập trình phương thức `forward`.

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
To see how it works let's feed some data into the layer.
-->

Để xem tầng này hoạt động thế nào, hãy truyền vào một chút dữ liệu.

```{.python .input  n=2}
layer = CenteredLayer()
layer(np.array([1, 2, 3, 4, 5]))
```

<!--
We can also use it to construct more complex models.
-->

Chúng ta cũng có thể sử dụng tầng này để xây dựng các mô hình phức tạp hơn.

```{.python .input  n=3}
net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())
net.initialize()
```

<!--
Let's see whether the centering layer did its job. 
For that we send random data through the network and check whether the mean vanishes. 
Note that since we are dealing with floating point numbers, we are going to see a very small albeit typically nonzero number.
-->

Hãy xem tầng này có hoạt động không.
Để làm việc đó, chúng ta truyền dữ liệu ngẫu nhiên vào mạng và kiểm tra xem giá trị trung bình đã về 0 chưa.
Chú ý rằng vì đang làm việc với các số thực dấu phẩy động, chúng ta sẽ thấy một giá trị khác không rất nhỏ.

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
Now that we know how to define layers in principle, let's define layers with parameters. 
These can be adjusted through training. 
In order to simplify things for an avid deep learning researcher the `Parameter` class and the `ParameterDict` dictionary provide some basic housekeeping functionality. 
In particular, they govern access, initialization, sharing, saving and loading model parameters. 
For instance, this way we do not need to write custom serialization routines for each new custom layer.
-->

Bây giờ về nguyên tắc ta đã biết cách định nghĩa các tầng, hãy chuyển sang việc định nghĩa các tầng có tham số.
Các tham số có thể được điều chỉnh thông qua quá trình huấn luyện.
Lớp `Parameter` và từ điển `ParameterDict` sẽ cung cấp một số tính năng quản trị cơ bản nhằm đơn giản hóa công việc cho các nhà nghiên cứu học sâu.
Cụ thể, chúng sẽ quản lý việc truy cập, khởi tạo, chia sẻ, lưu và nạp các tham số mô hình.
Bằng cách này, ta không cần phải viết lại các thủ tục tuần tự hóa cho mỗi tầng tùy chỉnh mới.

<!--
For instance, we can use the member variable `params` of the `ParameterDict` type that comes with the Block class. 
It is a dictionary that maps string type parameter names to model parameters in the `Parameter` type.
We can create a `Parameter` instance from `ParameterDict` via the `get` function.
-->

Ví dụ, ta có thể sử dụng biến thành viên `params` với kiểu `ParameterDict` đi kèm với lớp `Block`.
Nó là một từ điển ánh xạ các xâu kí tự là tên tham số sang các tham số mô hình có kiểu `Parameter`.
Ta có thể tạo một thực thể `Parameter` từ `ParameterDict` thông qua hàm `get`.

```{.python .input  n=7}
params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
params
```

<!--
Let's use this to implement our own version of the dense layer. 
It has two parameters: bias and weight. To make it a bit nonstandard, we bake in the ReLU activation as default. 
Next, we implement a fully connected layer with both weight and bias parameters.
It uses ReLU as an activation function, where `in_units` and `units` are the number of inputs and the number of outputs, respectively.
-->

Ta sẽ sử dụng biến `params` này để tự lập trình một phiên bản khác của tầng `Dense`.
Tầng này gồm có hai tham số: hệ số điều chỉnh và trọng số. Để biến tấu một chút, ta sẽ mặc định sử dụng hàm kích hoạt ReLU.
Tiếp theo, ta lập trình một tầng kết nối đầy đủ với cả trọng số và hệ số điều chỉnh.
Tầng này sử dụng hàm kích hoạt ReLU, trong đó `in_units` và `units` lần lượt là số lượng đầu vào và số lượng đầu ra.


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
Naming the parameters allows us to access them by name through dictionary lookup later. 
It is a good idea to give them instructive names. 
Next, we instantiate the `MyDense` class and access its model parameters.
-->

Việc đặt tên cho các tham số sẽ cho phép ta sau này truy cập chúng theo tên thông qua việc tra cứu từ điển.
Vì vậy, việc sử dụng các tên có tính chất gợi nhớ sẽ là một ý tưởng hay.
Tiếp theo, ta sẽ khởi tạo lớp `MyDense` và truy cập các tham số mô hình.

```{.python .input}
dense = MyDense(units=3, in_units=5)
dense.params
```

<!--
We can directly carry out forward calculations using custom layers.
-->

Ta có thể trực tiếp sử dụng các tầng tùy chỉnh để thực thi các phép tính trong lượt truyền xuôi.


```{.python .input  n=20}
dense.initialize()
dense(np.random.uniform(size=(2, 5)))
```

<!--
We can also construct models using custom layers. 
Once we have that we can use it just like the built-in dense layer. 
The only exception is that in our case size inference is not automatic. 
Please consult the [MXNet documentation](http://www.mxnet.io) for details on how to do this.
-->

Các tầng tùy chỉnh cũng có thể được dùng để xây dựng mô hình.
Các tầng này có thể được sử dụng giống như các tầng dày đặc được lập trình sẵn.
Ngoại lệ duy nhất là việc suy luận kích thước sẽ không được thực hiện tự động.
Để biết thêm chi tiết về cách thực hiện việc này, vui lòng tham khảo [tài liệu MXNet] (http://www.mxnet.io).

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
* We can design custom layers via the Block class. This is more powerful than defining a block factory, since it can be invoked in many contexts.
* Blocks can have local parameters.
-->

* Ta có thể thiết kế các tầng tùy chỉnh thông qua lớp Block. Cách này linh hoạt hơn việc định nghĩa một khối vì các tầng tùy chỉnh có thể được gọi trong nhiều ngữ cảnh khác nhau.
* Các khối có thể có các tham số cục bộ.


<!--
## Exercises
-->

## Bài tập

<!--
1. Design a layer that learns an affine transform of the data, i.e., it removes the mean and learns an additive parameter instead.
2. Design a layer that takes an input and computes a tensor reduction, i.e., it returns $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
3. Design a layer that returns the leading half of the Fourier coefficients of the data. Hint: look up the `fft` function in MXNet.
-->

1. Thiết kế một tầng học một phép biến đổi affine của dữ liệu, tức là, loại bỏ giá trị trung bình và thay vào đó học một tham số phụ gia. <!--  -->
2. Thiết kế một tầng nhận đầu vào và tính toán phép giảm tensor, tức là, trả về $y_k = \sum_{i, j} W_{ijk} x_i x_j$.
3. Thiết kế một tầng trả về nửa đầu của các hệ số Fourier của dữ liệu. Gợi ý: tra cứu hàm `fft` trong MXNet.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2328)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2328)
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
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
* Phạm Minh Đức
<!-- Phần 2 -->
* Nguyễn Duy Du
* Lê Khắc Hồng Phúc

<!-- Phần 3 -->
* Nguyễn Duy Du
* Lê Khắc Hồng Phúc

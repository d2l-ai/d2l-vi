<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Deferred Initialization
-->

# Khởi tạo trễ
:label:`sec_deferred_init`

<!--
So far, it might seem that we got away with being sloppy in setting up our networks.
Specifically, we did the following unintuitive things, which not might seem like they should work:
-->

Cho tới nay, có vẻ như ta chưa phải chịu hậu quả của việc thiết lập mạng cẩu thả. Cụ thể, ta đã "giả mù" và làm những điều không trực quan sau: 

<!--
* We defined the network architectures without specifying the input dimensionality.
* We added layers without specifying the output dimension of the previous layer.
* We even "initialized" these parameters before providing enough information to determine how many parameters our models should contain.
-->

* Ta định nghĩa kiến trúc mạng mà không xét đến chiều đầu vào.
* Ta thêm các tầng mà không xét đến chiều đầu ra của tầng trước đó.
* Ta thậm chí còn "khởi tạo" các tham số mà không có đầy đủ thông tin để xác định số lượng các tham số của mô hình. 

<!--
You might be surprised that our code runs at all.
After all, there is no way MXNet  could tell what the input dimensionality of a network would be.
The trick here is that MXNet *defers initialization*, waiting until the first time we pass data through the model, to infer the sizes of each layer *on the fly*.
Later on, when working with convolutional neural networks this technique will become even more convenient, since the input dimensionality 
(i.e., the resolution of an image) will affect the dimensionality of each subsequent layer. 
Hence, the ability to set parameters without the need to know, at the time of writing the code, what the dimensionality is 
can greatly simplify the task of specifying and subsequently modifying our models. 
Next, we go deeper into the mechanics of initialization.
-->

Bạn có thể khá bất ngờ khi thấy mã nguồn của ta vẫn chạy. 
Suy cho cùng, MXNet (hay bất cứ framework nào khác) không thể dự đoán được chiều của đầu vào.
Thủ thuật ở đây đó là MXNet đã "khởi tạo trễ", tức đợi cho đến khi ta truyền dữ liệu qua mô hình lần đầu để suy ra kích thước của mỗi tầng khi chúng "di chuyển".  
Ở các chương sau, khi làm việc với các mạng nơ-ron tích chập, kỹ thuật này sẽ còn trở nên tiện lợi hơn, bởi chiều của đầu vào (tức độ phân giải của một bức ảnh) sẽ tác động đến chiều của các tầng tiếp theo trong mạng.
Do đó, khả năng gán giá trị các tham số mà không cần biết số chiều tại thời điểm viết mã có thể đơn giản hóa việc xác định và sửa đổi mô hình về sau một cách đáng kể.
Tiếp theo, chúng ta sẽ đi sâu hơn vào cơ chế của việc khởi tạo.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Instantiating a Network
-->

## Khởi tạo Mạng

<!--
To begin, let us instantiate an MLP. 
-->

Để bắt đầu, hãy cùng khởi tạo một MLP.

```{.python .input}
from mxnet import init, np, npx
from mxnet.gluon import nn
npx.set_np()

def getnet():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = getnet()
```

<!--
At this point, the network cannot possibly know the dimensions of the input layer's weights because the input dimension remains unknown.
Consequently MXNet has not yet initialized any parameters.
We confirm by attempting to access the parameters below.
-->

Lúc này, mạng nơ-ron không thể biết được chiều của các trọng số ở tầng đầu vào bởi nó còn chưa biết chiều của đầu vào.
Và vì thế MXNet chưa khởi tạo bất kỳ tham số nào cả.
Ta có thể xác thực việc này bằng cách truy cập các tham số như dưới đây.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

<!--
Note that while the Parameter objects exist, the input dimension to each layer to listed as `-1`.
MXNet uses the special value `-1` to indicate that the parameters dimension remains unknown.
At this point attempts to access `net[0].weight.data()` would trigger a runtime error stating that the network must be initialized before the parameters can be accessed.
Now let us see what happens when we attempt to initialze parameters via the `initialize` method.
-->

Chú ý rằng mặc dù đối tượng Parameter có tồn tại, chiều đầu vào của mỗi tầng được liệt kê là `-1`.
MXNet sử dụng giá trị đặc biệt `-1` để ám chỉ việc chưa biết chiều tham số.
Tại thời điểm này, việc thử truy cập `net[0].weight.data()` sẽ gây ra lỗi thực thi báo rằng mạng cần khởi tạo trước khi truy cập tham số. 
Bây giờ hãy cùng xem điều gì sẽ xảy ra khi ta thử khởi tạo các tham số với phương thức `initialize`.

```{.python .input}
net.initialize()
net.collect_params()
```

<!--
As we can see, nothing has changed. 
When input dimensions are known, calls to initialize do not truly initalize the parameters.
Instead, this call registers to MXNet that we wish (and optionally, according to which distribution) to initialize the parameters. 
Only once we pass data through the network will MXNet finally initialize parameters and we will see a difference.
-->

Như ta đã thấy, không có gì thay đổi ở đây cả.
Khi chưa biết chiều của đầu vào, việc gọi phương thức khởi tạo không thực sự khởi tạo các tham số.
Thay vào đó, việc gọi phương thức trên sẽ chỉ đăng ký với MXNet là chúng ta muốn khởi tạo các tham số và phân phối mà ta muốn dùng để khởi tạo (không bắt buộc).
Chỉ khi truyền dữ liệu qua mạng thì MXNet mới khởi tạo các tham số và ta mới thấy được sự khác biệt. 

```{.python .input}
x = np.random.uniform(size=(2, 20))
net(x)  # Forward computation

net.collect_params()
```

<!--
As soon as we knew the input dimensionality, $\mathbf{x} \in \mathbb{R}^{20}$ MXNet can identify the shape of the first layer's weight matrix, i.e., $\mathbf{W}_1 \in \mathbb{R}^{256 \times 20}$.
Having recognized the first layer shape, MXNet proceeds to the second layer, whose dimensionality is $10 \times 256$ and so on through the computational graph until all shapes are known.
Note that in this case, only the first layer required deferred initialization, but MXNet initializes sequentially. 
Once all parameter shapes are known, MXNet can finally initialize the parameters. 
-->

Ngay khi biết được chiều của đầu vào là $\mathbf{x} \in \mathbb{R}^{20}$, MXNet có thể xác định kích thước của ma trận trọng số tầng đầu tiên: $\mathbf{W}_1 \in \mathbb{R}^{256 \times 20}$.
Sau khi biết được kích thước tầng đầu tiên, MXNet tiếp tục tính kích thước tầng thứ hai ($10 \times 256$) và cứ thế đi hết đồ thị tính toán cho đến khi nó biết được kích thước của mọi tầng.
Chú ý rằng trong trường hợp này, chỉ tầng đầu tiên cần được khởi tạo trễ, tuy nhiên MXNet vẫn khởi tạo theo thứ tự. 
Khi mà tất cả kích thước tham số đã được biết, MXNet cuối cùng có thể khởi tạo các tham số. 

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Deferred Initialization in Practice
-->

## Khởi tạo Trễ trong Thực tiễn

<!--
Now that we know how it works in theory, let us see when the initialization is actually triggered.
In order to do so, we mock up an initializer which does nothing but report a debug message stating when it was invoked and with which parameters.
-->

Giờ ta đã biết nó hoạt động như thế nào về mặt lý thuyết, hãy xem thử khi nào thì việc khởi tạo này thực sự diễn ra.
Để làm điều này, chúng ta cần lập trình thử một bộ khởi tạo. Nó sẽ không làm gì ngoài việc gửi một thông điệp gỡ lỗi cho biết khi nào nó được gọi và cùng với các tham số nào.

```{.python .input  n=22}
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # The actual initialization logic is omitted here

net = getnet()
net.initialize(init=MyInit())
```

<!--
Note that, although `MyInit` will print information about the model parameters when it is called, the above `initialize` function does not print any information after it has been executed.  
Therefore there is no real initialization parameter when calling the `initialize` function. 
Next, we define the input and perform a forward calculation.
-->

Lưu ý rằng, mặc dù `MyInit` sẽ in thông tin về các tham số mô hình khi nó được gọi, hàm khởi tạo `initialize` ở trên không xuất bất cứ thông tin nào sau khi được thực thi. 
Do đó, việc khởi tạo tham số không thực sự được thực hiện khi gọi hàm `initialize`.
Kế tiếp, ta định nghĩa đầu vào và thực hiện một lượt phép tính truyền xuôi.

```{.python .input  n=25}
x = np.random.uniform(size=(2, 20))
y = net(x)
```

<!--
At this time, information on the model parameters is printed. 
When performing a forward calculation based on the input `x`, the system can automatically infer the shape of the weight parameters of all layers based on the shape of the input. 
Once the system has created these parameters, it calls the `MyInit` instance to initialize them before proceeding to the forward calculation.
-->

Lúc này, thông tin về các tham số mô hình mới được in ra. 
Khi thực hiện lượt truyền xuôi dựa trên biến đầu vào `x`, hệ thống có thể tự động suy ra kích thước các tham số của tất cả các tầng dựa trên kích thước của biến đầu vào này. 
Một khi hệ thống đã tạo ra các tham số trên, nó sẽ gọi thực thể `MyInit` để khởi tạo chúng trước khi bắt đầu thực hiện lượt truyền xuôi. 

<!--
This initialization will only be called when completing the initial forward calculation. 
After that, we will not re-initialize when we run the forward calculation `net(x)`, so the output of the `MyInit` instance will not be generated again.
-->

Việc khởi tạo này sẽ chỉ được gọi khi lượt truyền xuôi đầu tiên hoàn thành. 
Sau thời điểm này, chúng ta sẽ không khởi tạo lại khi dùng lệnh `net(x)` để thực hiện lượt truyền xuôi, do đó đầu ra của thực thể `MyInit` sẽ không được sinh ra nữa.  

```{.python .input}
y = net(x)
```

<!--
As mentioned at the beginning of this section, deferred initialization can be source of confusion.
Before the first forward calculation, we were unable to directly manipulate the model parameters,
for example, we could not use the `data` and `set_data` functions to get and modify the parameters.
Therefore, we often force initialization by sending a sample observation through the network.
-->

Như đã đề cập ở phần mở đầu của mục này, việc khởi tạo trễ cũng có thể gây ra sự khó hiểu. 
Trước khi lượt truyền xuôi đầu tiên được thực thi, chúng ta không thể thao tác trực tiếp lên các tham số của mô hình. 
Chẳng hạn, chúng ta sẽ không thể dùng các hàm `data` và `set_data` để nhận và thay đổi các tham số. 
Do đó, chúng ta thường ép việc khởi tạo diễn ra bằng cách đưa một mẫu dữ liệu qua mạng này. 

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Forced Initialization
-->

## Khởi tạo Cưỡng chế

<!--
Deferred initialization does not occur if the system knows the shape of all parameters when we call the `initialize` function. 
This can occur in two cases:
-->

Khởi tạo trễ không xảy ra nếu hệ thống đã biết kích thước của tất cả các tham số khi gọi hàm `initialize`.
Việc này có thể xảy ra trong hai trường hợp: 

<!--
* We have already seen some data and we just want to reset the parameters.
* We specified all input and output dimensions of the network when defining it.
-->

* Ta đã truyền dữ liệu vào mạng từ trước và chỉ muốn khởi tạo lại các tham số. 
* Ta đã chỉ rõ cả chiều đầu vào và chiều đầu ra của mạng khi định nghĩa nó. 

<!--
Forced reinitialization works as illustrated below.
-->

Khởi tạo cưỡng chế hoạt động như minh hoạ dưới đây.

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

<!--
The second case requires that we specify all parameters when creating each layer.
For instance, for dense layers we must specify `in_units` at the time that the layer is instantiated.
-->

Trường hợp thứ hai yêu cầu ta chỉ rõ tất cả tham số khi tạo mỗi tầng trong mạng.
Ví dụ, với các tầng kết nối dày đặc thì chúng ta cần chỉ rõ `in_units` tại thời điểm tầng đó được khởi tạo.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

net.initialize(init=MyInit())
```

<!--
## Summary
-->

## Tóm tắt

<!--
* Deferred initialization can be convenient, allowing Gluon to infer parameter shapes automatically, making it easy to modify architectures and eliminating one common source of errors.
* We do not need deferred initialization when we specify all variables explicitly.
* We can forcibly re-initialize a network's parameters by invoking initalize with the `force_reinit=True` flag.
-->

* Khởi tạo trễ có thể khá tiện lợi, cho phép Gluon suy ra kích thước của tham số một cách tự động và nhờ vậy giúp ta dễ dàng sửa đổi các kiến trúc mạng cũng như loại bỏ những nguồn gây lỗi thông dụng. 
* Chúng ta không cần khởi tạo trễ khi đã định nghĩa các biến một cách tường minh.
* Chúng ta có thể cưỡng chế việc khởi tạo lại các tham số mạng bằng cách gọi khởi tạo với `force_reinit=True`. 

<!--
## Exercises
-->

## Bài tập

<!--
1. What happens if you specify the input dimensions to the first laye but not to subsequent layers? Do you get immediate initialization?
2. What happens if you specify mismatching dimensions?
3. What would you need to do if you have input of varying dimensionality? Hint - look at parameter tying.
-->

1. Chuyện gì xảy ra nếu ta chỉ chỉ rõ chiều đầu vào của tầng đầu tiên nhưng không làm vậy với các tầng tiếp theo? Việc khởi tạo có xảy ra ngay lập tức không?
2. Chuyện gì xảy ra nếu ta chỉ định các chiều không khớp nhau? 
3. Bạn cần làm gì nếu đầu vào có chiều biến thiên? Gợi ý - hãy tìm hiểu về cách ràng buộc tham số (*parameter tying*). 

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2327)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Lý Phi Long
* Nguyễn Mai Hoàng Long
* Phạm Minh Đức
* Nguyễn Lê Quang Nhật

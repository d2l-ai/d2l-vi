<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Deferred Initialization
-->

# Khởi tạo trễ
:label:`sec_deferred_init`

<!--
In the previous examples we played fast and loose with setting up our networks. In particular we did the following things that *shouldn't* work:
-->

<!-- UPDATE
So far, it might seem that we got away with being sloppy in setting up our networks.
Specifically, we did the following unintuitive things, which not might seem like they should work:
-->

Ở các ví dụ trên chúng ta chưa chặt chẽ trong việc xây dựng các mạng nơ-ron.
Cụ thể, dưới đây là những công đoạn ta đã thực hiện mà đáng ra sẽ *không* hoạt động:

<!--
* We defined the network architecture with no regard to the input dimensionality.
* We added layers without regard to the output dimension of the previous layer.
* We even "initialized" these parameters without knowing how many parameters were to initialize.
-->

<!-- UPDATE
* We defined the network architectures without specifying the input dimensionality.
* We added layers without specifying the output dimension of the previous layer.
* We even "initialized" these parameters before providing enough information to determine how many parameters our models should contain.
-->

* Ta định nghĩa kiến trúc mạng mà không xét đến chiều đầu vào.
* Ta thêm các tầng mà không xét đến chiều đầu ra của tầng trước đó.
* Ta thậm chí đã "khởi tạo" các tham số mà không biết có bao nhiêu tham số cần khởi tạo.

<!--
All of those things sound impossible and indeed, they are. 
After all, there is no way MXNet (or any other framework for that matter) could predict what the input dimensionality of a network would be. 
Later on, when working with convolutional networks and images this problem will become even more pertinent, since the input dimensionality 
(i.e., the resolution of an image) will affect the dimensionality of subsequent layers at a long range. 
Hence, the ability to set parameters without the need to know at the time of writing the code what the dimensionality is can greatly simplify statistical modeling. 
In what follows, we will discuss how this works using initialization as an example. 
After all, we cannot initialize variables that we do not know exist.
-->

<!-- UPDATE
You might be surprised that our code runs at all.
After all, there is no way MXNet  could tell what the input dimensionality of a network would be.
The trick here is that MXNet *defers initialization*, waiting until the first time we pass data through the model, to infer the sizes of each layer *on the fly*.
Later on, when working with convolutional neural networks this technique will become even more convenient, since the input dimensionality 
(i.e., the resolution of an image) will affect the dimensionality of each subsequent layer. 
Hence, the ability to set parameters without the need to know, at the time of writing the code, what the dimensionality is 
can greatly simplify the task of specifying and subsequently modifying our models. 
Next, we go deeper into the mechanics of initialization.
-->

Tất cả những điều đó nghe bất khả thi và thực sự, đúng là vậy.
Suy cho cùng, MXNet (hay bất cứ framework nào khác) không thể dự đoán được chiều của đầu vào sẽ như thế nào.
Ở các chương sau, khi làm việc với các mạng nơ-ron tích chập và ảnh, vấn đề này còn trở nên rõ ràng hơn, khi chiều của đầu vào (trong trường hợp này là độ phân giải của một bức ảnh) về lâu dài sẽ tác động đến chiều các tầng phía sau của mạng.
Do đó, khả năng gán giá trị các tham số mà không cần biết số chiều tại thời điểm viết mã có thể làm việc mô hình hoá thống kê trở nên đơn giản hơn nhiều.
Dưới đây, chúng ta sẽ thảo luận cơ chế hoạt động của việc này qua một ví dụ về khởi tạo.
Vì dù gì chúng ta cũng không thể khởi tạo các biến mà ta không biết chúng tồn tại.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Instantiating a Network
-->

## Khởi tạo Mạng

<!--
Let's see what happens when we instantiate a network. 
We start with our trusty MLP as before.
-->

<!-- UPDATE
To begin, let us instantiate an MLP. 
-->

Hãy xem điều gì sẽ xảy ra khi ta khởi tạo một mạng nơ-ron nhé!
Ta bắt đầu với mạng MLP đáng tin cậy như trước đây.

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
At this point the network does not really know yet what the dimensionalities of the various parameters should be. 
All one could tell at this point is that each layer needs weights and bias, albeit of unspecified dimensionality. 
If we try accessing the parameters, that is exactly what happens.
-->

<!-- UPDATE
At this point, the network cannot possibly know the dimensions of the input layer's weights because the input dimension remains unknown.
Consequently MXNet has not yet initialized any parameters.
We confirm by attempting to access the parameters below.
-->

Lúc này, mạng nơ-ron chưa biết được số chiều thực sự của các tham số là bao nhiêu.
Điều ta duy nhất biết được tại thời điểm này là mỗi lớp cần có trọng số và hệ số điều chỉnh, mặc dù số chiều vẫn còn chưa xác định.
Nếu ta thử truy cập vào các tham số, đó chính xác là những gì sẽ xảy ra.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

<!--
In particular, trying to access `net[0].weight.data()` at this point would trigger a runtime error stating that the network needs initializing before it can do anything. 
Let's see whether anything changes after we initialize the parameters:
-->

<!--UPDATE
Note that while the Parameter objects exist, the input dimension to each layer to listed as `-1`.
MXNet uses the special value `-1` to indicate that the parameters dimension remains unknown.
At this point attempts to access `net[0].weight.data()` would trigger a runtime error stating that the network must be initialized before the parameters can be accessed.
Now let us see what happens when we attempt to initialze parameters via the `initialize` method.
-->

Cụ thể, thử truy cập `net[0].weight.data()` vào lúc này sẽ gây ra lỗi thực thi báo rằng mạng cần khởi tạo trước khi làm bất cứ điều gì.
Ta hãy xem liệu có điều gì thay đổi sau khi ta khởi tạo các tham số:

```{.python .input}
net.initialize()
net.collect_params()
```

<!--
As we can see, nothing really changed. 
Only once we provide the network with some data do we see a difference. 
Let's try it out.
-->

<!-- UPDATE
As we can see, nothing has changed. 
When input dimensions are known, calls to initialize do not truly initalize the parameters.
Instead, this call registers to MXNet that we wish (and optionally, according to which distribution) to initialize the parameters. 
Only once we pass data through the network will MXNet finally initialize parameters and we will see a difference.
-->

Như ta đã thấy, không có gì thay đổi ở đây cả.
Chỉ khi nào ta cung cấp cho mạng một ít dữ liệu thì ta mới thấy được sự khác biệt.
Hãy thử xem!

```{.python .input}
x = np.random.uniform(size=(2, 20))
net(x)  # Forward computation

net.collect_params()
```

<!--
The main difference to before is that as soon as we knew the input dimensionality, 
$\mathbf{x} \in \mathbb{R}^{20}$ it was possible to define the weight matrix for the first layer, 
i.e., $\mathbf{W}_1 \in \mathbb{R}^{256 \times 20}$. 
With that out of the way, we can progress to the second layer, 
define its dimensionality to be $10 \times 256$ and so on through the computational graph and bind all the dimensions as they become available. 
Once this is known, we can proceed by initializing parameters. 
This is the solution to the three problems outlined above.
-->

<!-- UPDATE
As soon as we knew the input dimensionality, $\mathbf{x} \in \mathbb{R}^{20}$ MXNet can identify the shape of the first layer's weight matrix, i.e., $\mathbf{W}_1 \in \mathbb{R}^{256 \times 20}$.
Having recognized the first layer shape, MXNet proceeds to the second layer, whose dimensionality is $10 \times 256$ and so on through the computational graph until all shapes are known.
Note that in this case, only the first layer required deferred initialization, but MXNet initializes sequentially. 
Once all parameter shapes are known, MXNet can finally initialize the parameters. 
-->

Điểm khác biệt chính so với lúc trước là ngay khi ta biết được số chiều của đầu vào $\mathbf{x} \in \mathbb{R}^{20}$, ta có thể định nghĩa ma trận trọng số cho tầng đầu tiên, tức $\mathbf{W}_1 \in \mathbb{R}^{256 \times 20}$.
Với cách đó, ta có thể chuyển sang tầng thứ hai, định nghĩa số chiều là $10 \times 256$ và cứ thế ta truyền qua đồ thị tính toán rồi liên kết tất cả số chiều lại với nhau.
Một khi ta biết được số chiều, ta có thể tiến hành khởi tạo các tham số.
Đây là lời giải cho ba bài toán được đặt ra ở trên.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Deferred Initialization in Practice
-->

## Khởi tạo Trễ trong Thực tiễn

<!--
Now that we know how it works in theory, let's see when the initialization is actually triggered. 
In order to do so, we mock up an initializer which does nothing but report a debug message stating when it was invoked and with which parameters.
-->

<!-- UPDATE
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
Note that, although `MyInit` will print information about the model parameters when it is called, 
the above `initialize` function does not print any information after it has been executed.
Therefore there is no real initialization parameter when calling the `initialize` function. 
Next, we define the input and perform a forward calculation.
-->

<!-- UPDATE
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

<!-- UPDATE
At this time, information on the model parameters is printed. 
When performing a forward calculation based on the input `x`, the system can automatically infer the shape of the weight parameters of all layers based on the shape of the input. 
Once the system has created these parameters, it calls the `MyInit` instance to initialize them before proceeding to the forward calculation.
-->

Lúc này, thông tin về các tham số mô hình mới được in ra. 
Khi thực hiện lượt truyền xuôi dựa trên biến đầu vào `x`, hệ thống có thể tự động suy ra kích thước các tham số của tất cả các tầng dựa trên kích thước của biến đầu vào này. 
Một khi hệ thống đã tạo ra các tham số trên, nó sẽ gọi thực thể `MyInit` để khởi tạo chúng trước khi bắt đầu thực hiện phép truyền xuôi. 

<!--
Of course, this initialization will only be called when completing the initial forward calculation. 
After that, we will not re-initialize when we run the forward calculation `net(x)`, so the output of the `MyInit` instance will not be generated again.
-->

<!-- UPDATE
This initialization will only be called when completing the initial forward calculation. 
After that, we will not re-initialize when we run the forward calculation `net(x)`, so the output of the `MyInit` instance will not be generated again.
-->

Việc khởi tạo này sẽ chỉ được gọi khi lượt truyền xuôi đầu tiên hoàn thành. 
Sau thời điểm này, chúng ta sẽ không khởi tạo lại khi ta chạy phương thức truyền xuôi `net(x)`, do đó đầu ra của thực thể `MyInit` sẽ không được sinh ra nữa.  

```{.python .input}
y = net(x)
```

<!--
As mentioned at the beginning of this section, deferred initialization can also cause confusion. 
Before the first forward calculation, we were unable to directly manipulate the model parameters, 
for example, we could not use the `data` and `set_data` functions to get and modify the parameters. 
Therefore, we often force initialization by sending a sample observation through the network.
-->

<!-- UPDATE
As mentioned at the beginning of this section, deferred initialization can be source of confusion.
Before the first forward calculation, we were unable to directly manipulate the model parameters,
for example, we could not use the `data` and `set_data` functions to get and modify the parameters.
Therefore, we often force initialization by sending a sample observation through the network.
-->

Như đã đề cập ở phần mở đầu của mục này, việc khởi tạo trễ cũng có thể gây ra sự khó hiểu. 
Trước khi lượt truyền xuôi đầu tiên được thực thi, chúng ta không thể thao tác trực tiếp lên các tham số của mô hình. Chẳng hạn, chúng ta sẽ không thể dùng các hàm `data` và `set_data` để nhận và thay đổi các tham số. 
Do đó, chúng ta thường ép việc khởi tạo diễn ra bằng cách chạy một quan sát mẫu qua mạng này. 

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Forced Initialization
-->

## Khởi tạo Cưỡng chế

<!--
Deferred initialization does not occur if the system knows the shape of all parameters when calling the `initialize` function. 
This can occur in two cases:
-->

<!-- UPDATE
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
* Ta chỉ rõ tất cả chiều đầu vào và đầu ra của mạng khi định nghĩa nó. 

<!--
Forced reinitialization works as illustrated below.
-->

Trường hợp thứ nhất hoạt động tốt, như minh hoạ dưới đây.

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

<!--
The second case requires us to specify the remaining set of parameters when creating the layer. 
For instance, for dense layers we also need to specify the `in_units` so that initialization can occur immediately once `initialize` is called.
-->

<!-- UPDATE
The second case requires that we specify all parameters when creating each layer.
For instance, for dense layers we must specify `in_units` at the time that the layer is instantiated.
-->

Trường hợp thứ hai yêu cầu ta chỉ rõ tất cả tham số khi tạo mỗi tầng trong mạng.
Ví dụ, với các tầng kết nối đầy đủ thì chúng ta cần chỉ rõ `in_units` tại thời điểm tầng đó được khởi tạo.

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
* Deferred initialization is a good thing. It allows Gluon to set many things automatically and it removes a great source of errors from defining novel network architectures.
* We can override this by specifying all implicitly defined variables.
* Initialization can be repeated (or forced) by setting the `force_reinit=True` flag.
-->

<!-- UPDATE
* Deferred initialization can be convenient, allowing Gluon to infer parameter shapes automatically, making it easy to modify architectures and eliminating one common source of errors.
* We do not need deferred initialization when we specify all variables explicitly.
* We can forcibly re-initialize a network's parameters by invoking initalize with the `force_reinit=True` flag.
-->

* Khởi tạo trễ có thể khá tiện lợi, cho phép Gluon gán giá trị một cách tự động và nhờ vậy giúp ta dễ dàng sửa đổi các kiến trúc mạng cũng như loại bỏ những nguồn gây lỗi thông dụng. 
* Chúng ta không cần khởi tạo trễ khi đã định nghĩa các biến một cách tường minh.
* Chúng ta có thể cưỡng chế việc khởi tạo lại các tham số mạng bằng cách gọi khởi tạo với cờ `force_reinit=True`. 

<!--
## Exercises
-->

## Bài tập

<!--
1. What happens if you specify the input dimensions to the first laye but not to subsequent layers? Do you get immediate initialization?
2. What happens if you specify mismatching dimensions?
3. What would you need to do if you have input of varying dimensionality? Hint - look at parameter tying.
-->

1. Chuyện gì xảy ra nếu ta chỉ chỉ rõ chiều đầu vào của tầng đầu tiên nhưng không làm vậy với các tầng tiếp theo? Có thể vẫn khởi tạo ngay lập tức được không?
2. Chuyện gì xảy ra nếu ta truyền vào giá trị chiều không phù hợp? 
3. Bạn cần làm gì nếu đầu vào có chiều biến thiên? Gợi ý - tìm hiểu về ràng buộc tham số (*parameter tying*). 

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2327)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2327)
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
* Phạm Hồng Vinh
* Lý Phi Long
* Nguyễn Mai Hoàng Long
* Phạm Minh Đức

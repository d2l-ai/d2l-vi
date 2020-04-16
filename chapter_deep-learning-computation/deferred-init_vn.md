<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Deferred Initialization
-->

# *dịch tiêu đề phía trên*
:label:`sec_deferred_init`

<!--
In the previous examples we played fast and loose with setting up our networks. In particular we did the following things that *shouldn't* work:
-->

*dịch đoạn phía trên*

<!--
* We defined the network architecture with no regard to the input dimensionality.
* We added layers without regard to the output dimension of the previous layer.
* We even "initialized" these parameters without knowing how many parameters were to initialize.
-->

*dịch đoạn phía trên*

<!--
All of those things sound impossible and indeed, they are. 
After all, there is no way MXNet (or any other framework for that matter) could predict what the input dimensionality of a network would be. 
Later on, when working with convolutional networks and images this problem will become even more pertinent, since the input dimensionality 
(i.e., the resolution of an image) will affect the dimensionality of subsequent layers at a long range. 
Hence, the ability to set parameters without the need to know at the time of writing the code what the dimensionality is can greatly simplify statistical modeling. 
In what follows, we will discuss how this works using initialization as an example. 
After all, we cannot initialize variables that we do not know exist.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Instantiating a Network
-->

## *dịch tiêu đề phía trên*

<!--
Let's see what happens when we instantiate a network. 
We start with our trusty MLP as before.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

<!--
In particular, trying to access `net[0].weight.data()` at this point would trigger a runtime error stating that the network needs initializing before it can do anything. 
Let's see whether anything changes after we initialize the parameters:
-->

*dịch đoạn phía trên*

```{.python .input}
net.initialize()
net.collect_params()
```

<!--
As we can see, nothing really changed. 
Only once we provide the network with some data do we see a difference. 
Let's try it out.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

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

Giờ ta đã biết nó hoạt động như thế nào về mặt lý thuyết, hãy xem thử khi nào thì việc khởi tạo này thực sự diễn ra.
Để làm như vậy, chúng ta cần lên khung một bộ khởi tạo mà nó không làm gì ngoài việc gửi một thông điệp gỡ lỗi cho biết khi nào nó được gọi và cùng với các tham số nào.
*dịch đoạn phía trên*

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

Lưu ý rằng, mặc dù `MyInit` sẽ in thông tin về các tham số mô hình khi nó được gọi, hàm khởi tạo `initialize` ở trên không xuất bất cứ thông tin nào sau khi nó đã thực thi. 
Do đó không có bất cứ tham số khởi tạo thật sự nào khi thực hiện gọi hàm `initialize` này. 
Kế tiếp, ta định nghĩa đầu vào và thực hiện một lượt phép tính truyền xuôi.
*dịch đoạn phía trên*

```{.python .input  n=25}
x = np.random.uniform(size=(2, 20))
y = net(x)
```

<!--
At this time, information on the model parameters is printed. 
When performing a forward calculation based on the input `x`, the system can automatically infer the shape of the weight parameters of all layers based on the shape of the input. 
Once the system has created these parameters, it calls the `MyInit` instance to initialize them before proceeding to the forward calculation.
-->

Ở thời điểm này, thông tin về các tham số mô hình được in ra.
Khi thực hiện tính toán tiếp theo dựa trên biến đầu vào `x`, hệ thống có thể tự động suy ra kích thước các tham số của tất cả các lớp dựa trên kích thước của biến đầu vào này. 
Ngay khi hệ thống đã tạo ra các tham số trên, nó gọi mẫu `MyInit` để khởi tạo chúng trước khi xử lý tính toán kế tiếp.
*dịch đoạn phía trên*

<!--
Of course, this initialization will only be called when completing the initial forward calculation. 
After that, we will not re-initialize when we run the forward calculation `net(x)`, so the output of the `MyInit` instance will not be generated again.
-->

Tất nhiên việc khởi tạo này sẽ chỉ được gọi khi ta hoàn thành lượt truyền xuôi lần đầu tiên.
Sau thời điểm này, chúng ta sẽ không khởi tạo lại khi chúng ta chạy lệnh tính toán kế tiếp `net(x)`, do đó đầu ra của mẫu `MyInit` sẽ không được sinh ra nữa. 
*dịch đoạn phía trên*

```{.python .input}
y = net(x)
```

<!--
As mentioned at the beginning of this section, deferred initialization can also cause confusion. 
Before the first forward calculation, we were unable to directly manipulate the model parameters, 
for example, we could not use the `data` and `set_data` functions to get and modify the parameters. 
Therefore, we often force initialization by sending a sample observation through the network.
-->

Như đã đề cập ở phần mở đầu của mục này, việc trì hoãn khởi tạo cũng có thể gây ra sự khó hiểu.
Trước khi lệnh tính toán kế tiếp đầu tiên được thực thi, chúng ta không thể nào thao tác trực tiếp lên các thông số của mô hình, chẳng hạn như chúng ta sẽ không thể dùng các hàm `data` và `set_data` để nhận và thay đổi các tham số. 
Do đó, chúng ta thường ép việc khởi tạo bằng cách gửi một mẫu quan sát qua mạng này. 
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Forced Initialization
-->

## *dịch tiêu đề phía trên*

<!--
Deferred initialization does not occur if the system knows the shape of all parameters when calling the `initialize` function. 
This can occur in two cases:
-->

*dịch đoạn phía trên*

<!--
* We have already seen some data and we just want to reset the parameters.
* We specified all input and output dimensions of the network when defining it.
-->

*dịch đoạn phía trên*

<!--
The first case works just fine, as illustrated below.
-->

*dịch đoạn phía trên*

```{.python .input}
net.initialize(init=MyInit(), force_reinit=True)
```

<!--
The second case requires us to specify the remaining set of parameters when creating the layer. 
For instance, for dense layers we also need to specify the `in_units` so that initialization can occur immediately once `initialize` is called.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
1. What happens if you specify only parts of the input dimensions. Do you still get immediate initialization?
2. What happens if you specify mismatching dimensions?
3. What would you need to do if you have input of varying dimensionality? Hint - look at parameter tying.
-->

*dịch đoạn phía trên*

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
<!-- Phần 1 -->
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
* Nguyễn Mai Hoàng Long

<!-- Phần 4 -->
*

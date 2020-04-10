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

Ở các ví dụ trên chúng ta chưa chặt chẽ trong việc xây dựng các mạng nơron.
Cụ thể, dưới đây là những công đoạn ta đã thực hiện mà đáng ra sẽ *không* hoạt động:

<!--
* We defined the network architecture with no regard to the input dimensionality.
* We added layers without regard to the output dimension of the previous layer.
* We even "initialized" these parameters without knowing how many parameters were to initialize.
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

Tất cả những điều đó nghe bất khả thi và thực sự, đúng là vậy.
Suy cho cùng, MXNet (hay bất cứ framework nào khác) không thể dự đoán được chiều của đầu vào sẽ như thế nào.
Ở các chương sau, khi làm việc với các mạng nơron tích chập và ảnh, vấn đề này còn trở nên rõ ràng hơn, khi chiều của đầu vào (trong trường hợp này là độ phân giải của một bức ảnh) về lâu dài sẽ tác động đến chiều các tầng phía sau của mạng.
Do đó, khả năng gán giá trị các tham số mà không cần biết chiều tại thời điểm viết mã có thể làm việc mô hình hoá thống kê trở nên đơn giản hơn nhiều.
Dưới đây, chúng ta sẽ thảo luận cơ chế hoạt động của việc này qua một ví dụ về khởi tạo.
Vì dù gì chúng ta cũng không thể khởi tạo các biến mà ta không biết chúng tồn tại.

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

## *dịch tiêu đề phía trên*

<!--
Now that we know how it works in theory, let's see when the initialization is actually triggered. 
In order to do so, we mock up an initializer which does nothing but report a debug message stating when it was invoked and with which parameters.
-->

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

*dịch đoạn phía trên*

<!--
Of course, this initialization will only be called when completing the initial forward calculation. 
After that, we will not re-initialize when we run the forward calculation `net(x)`, so the output of the `MyInit` instance will not be generated again.
-->

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
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

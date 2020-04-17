<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# File I/O
-->

# *dịch tiêu đề phía trên*

<!--
So far we discussed how to process data, how to build, train and test deep learning models. 
However, at some point we are likely happy with what we obtained and we want to save the results for later use and distribution. 
Likewise, when running a long training process it is best practice to save intermediate results (checkpointing) to ensure that 
we do not lose several days worth of computation when tripping over the power cord of our server. 
At the same time, we might want to load a pre-trained model (e.g., we might have word embeddings for English and use it for our fancy spam classifier). 
For all of these cases we need to load and store both individual weight vectors and entire models. 
This section addresses both issues.
-->

*dịch đoạn phía trên*

<!--
## Loading and Saving `ndarray`s
-->

## *dịch tiêu đề phía trên*

<!--
In its simplest form, we can directly use the `load` and `save` functions to store and read `ndarray`s separately. 
This works just as expected.
-->

*dịch đoạn phía trên*

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

<!--
Then, we read the data from the stored file back into memory.
-->

*dịch đoạn phía trên*

```{.python .input}
x2 = npx.load('x-file')
x2
```

<!--
We can also store a list of `ndarray`s and read them back into memory.
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

<!--
We can even write and read a dictionary that maps from a string to an `ndarray`. 
This is convenient, for instance when we want to read or write all the weights in a model.
-->

*dịch đoạn phía trên*

```{.python .input  n=4}
mydict = {'x': x, 'y': y}
npx.save('mydict', mydict)
mydict2 = npx.load('mydict')
mydict2
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Gluon Model Parameters
-->

## Tham số mô hình Gluon

<!--
Saving individual weight vectors (or other `ndarray` tensors) is useful but it gets very tedious if we want to save (and later load) an entire model. 
After all, we might have hundreds of parameter groups sprinkled throughout. 
Writing a script that collects all the terms and matches them to an architecture is quite some work. 
For this reason Gluon provides built-in functionality to load and save entire networks rather than just single weight vectors. 
An important detail to note is that this saves model *parameters* and not the entire model. 
I.e. if we have a 3 layer MLP we need to specify the *architecture* separately. 
The reason for this is that the models themselves can contain arbitrary code, hence they cannot be serialized quite so easily 
(there is a way to do this for compiled models: please refer to the [MXNet documentation](http://www.mxnet.io) for the technical details on it). 
The result is that in order to reinstate a model we need to generate the architecture in code and then load the parameters from disk. 
The deferred initialization (:numref:`sec_deferred_init`) is quite advantageous here since we can simply define a model without the need to put actual values in place. 
Let's start with our favorite MLP.
-->

Khả năng lưu từng vector trọng số đơn lẻ (hoặc các `ndarray` tensor khác) là hữu ích nhưng sẽ mất nhiều thời gian nếu chúng ta muốn lưu (và sau đó tải lại) toàn bộ mô hình.
Dù sao, có thể chúng ta sẽ có hàng trăm nhóm tham số rải rác xuyên suốt mô hình.
Việc viết một đoạn mã thu thập tất cả các nhóm này và khớp chúng với một kiến trúc là tương đối vất vả.
Do đó, Gluon cung cấp sẵn chức năng lưu và tải toàn bộ các mạng thay vì chỉ là các vector trọng số đơn lẻ.
Một chi tiết quan trọng cần lưu ý là chức năng này lưu các *tham số* của mô hình, không phải là toàn bộ mô hình.
Điều đó có nghĩa, nếu ta có Perceptron 3 tầng, ta cần chỉ rõ *kiến trúc* này một cách riêng rẽ.
Lý do là bởi bản thân các mô hình có thể chứa các đoạn mã tuỳ ý, các đoạn mã này khó có thể được đọc ghi vào tệp một cách dễ dàng như các tham số.
(có một cách thực hiện điều này cho các mô hình đã được biên dịch, chi tiết kĩ thuật đọc thêm tại [MXNet documentation](http://www.mxnet.io)).
Kết quả là, để khôi phục lại một mô hình chúng ta cần xây dựng kiến trúc của nó từ mã nguồn rồi tải các tham số từ ổ cứng vào kiến trúc này.
Việc khởi tạo trễ (:numref:`sec_deferred_init`) lúc này rất có lợi vì ta chỉ cần định nghĩa một mô hình mà không cần gán giá trị cụ thể cho tham số.
Như thường lệ, hãy bắt đầu với Perceptron Đa tầng.

```{.python .input  n=6}
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
x = np.random.uniform(size=(2, 20))
y = net(x)
```

<!--
Next, we store the parameters of the model as a file with the name `mlp.params`.
-->

Tiếp theo, chúng ta lưu các tham số của mô hình vào tệp `mlp.params`.

```{.python .input}
net.save_parameters('mlp.params')
```

<!--
To check whether we are able to recover the model we instantiate a clone of the original MLP model. 
Unlike the random initialization of model parameters, here we read the parameters stored in the file directly.
-->

Để kiểm tra xem có thể khôi phục lại mô hình này không, chúng ta tạo một đối tượng khác của mô hình MLP trên.
Khác với quá trình khởi tạo tham số ngẫu nhiên của mô hình, ở đây chúng ta trực tiếp tải các tham số từ tệp được ghi trước đó.

```{.python .input  n=8}
clone = MLP()
clone.load_parameters('mlp.params')
```

<!--
Since both instances have the same model parameters, the computation result of the same input `x` should be the same. 
Let's verify this.
-->

Vì cả hai đối tượng của mô hình có cùng bộ tham số, kết quả tính toán với cùng đầu vào `x` sẽ như nhau.
Hãy kiểm chứng điều này.

```{.python .input}
yclone = clone(x)
yclone == y
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Summary
-->

## Tóm tắt

<!--
* The `save` and `load` functions can be used to perform File I/O for `ndarray` objects.
* The `load_parameters` and `save_parameters` functions allow us to save entire sets of parameters for a network in Gluon.
* Saving the architecture has to be done in code rather than in parameters.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Even if there is no need to deploy trained models to a different device, what are the practical benefits of storing model parameters?
2. Assume that we want to reuse only parts of a network to be incorporated into a network of a *different* architecture. 
How would you go about using, say the first two layers from a previous network in a new network.
3. How would you go about saving network architecture and parameters? What restrictions would you impose on the architecture?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2329)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2329)
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
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
<!-- Phần 3 -->
*

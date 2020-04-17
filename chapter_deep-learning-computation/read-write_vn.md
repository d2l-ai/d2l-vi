<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# File I/O
-->

# Đọc/Ghi tệp

<!--
So far we discussed how to process data, how to build, train and test deep learning models. 
However, at some point we are likely happy with what we obtained and we want to save the results for later use and distribution. 
Likewise, when running a long training process it is best practice to save intermediate results (checkpointing) to ensure that 
we do not lose several days worth of computation when tripping over the power cord of our server. 
At the same time, we might want to load a pre-trained model (e.g., we might have word embeddings for English and use it for our fancy spam classifier). 
For all of these cases we need to load and store both individual weight vectors and entire models. 
This section addresses both issues.
-->

Cho đến giờ ta đã thảo luận về cách xử lý dữ liệu, cách xây dựng, huấn luyện và kiểm tra các mô hình học sâu.
Tuy nhiên, tại một số thời điểm, ta có thể hài lòng với những gì thu được và muốn lưu lại kết quả để sau này sử dụng và phân phối.
Tương tự như vậy, khi thực hiện một quá trình huấn luyện dài, cách tốt nhất là lưu lại các kết quả trung gian (điểm kiểm tra) để đảm bảo rằng ta sẽ không mất nhiều ngày để tính toán lại khi không may vấp phải dây nguồn của máy chủ.
Đồng thời, ta có thể muốn nạp một mô hình đã được huấn luyện sẵn (ví dụ: sử dụng các embedding từ tiếng Anh có sẵn để xây dựng một bộ phân loại thư rác màu mè).
Đối với tất cả các trường hợp này, ta cần đọc và lưu cả các vector trọng số đơn lẻ và toàn bộ mô hình.
Mục này sẽ giải quyết cả hai vấn đề trên.

<!--
## Loading and Saving `ndarray`s
-->

## Đọc và Lưu các `ndarray`

<!--
In its simplest form, we can directly use the `load` and `save` functions to store and read `ndarray`s separately. 
This works just as expected.
-->

Ở dạng đơn giản nhất, ta có thể sử dụng trực tiếp các hàm `load` và `save` để đọc và lưu các `ndarray` riêng rẽ.
Cách này hoạt động đúng như mong đợi.

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

Sau đó, ta đọc dữ liệu từ các tệp được lưu trở lại vào trong bộ nhớ.

```{.python .input}
x2 = npx.load('x-file')
x2
```

<!--
We can also store a list of `ndarray`s and read them back into memory.
-->

Ta cũng có thể lưu một danh sách các `ndarray` và đọc chúng trở lại vào trong bộ nhớ.

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

Ta thậm chí có thể ghi và đọc một từ điển ánh xạ từ một chuỗi sang một `ndarray`.
Cách này khá là thuận tiện, ví dụ khi ta muốn đọc hoặc ghi tất cả các trọng số trong một mô hình.

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

## *dịch tiêu đề phía trên*

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

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

```{.python .input}
net.save_parameters('mlp.params')
```

<!--
To check whether we are able to recover the model we instantiate a clone of the original MLP model. 
Unlike the random initialization of model parameters, here we read the parameters stored in the file directly.
-->

*dịch đoạn phía trên*

```{.python .input  n=8}
clone = MLP()
clone.load_parameters('mlp.params')
```

<!--
Since both instances have the same model parameters, the computation result of the same input `x` should be the same. 
Let's verify this.
-->

*dịch đoạn phía trên*

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
* Nguyễn Duy Du

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

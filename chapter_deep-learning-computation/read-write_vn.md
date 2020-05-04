<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# File I/O
-->

# Đọc/Ghi tệp

<!--
So far we discussed how to process data and how to build, train, and test deep learning models. 
However, at some point, we will hopefully be happy enough with the learned models that 
we will want to save the results for later use in various contexts (perhaps even to make predictions in deployment). 
Additionally, when running a long training process,the best practice is to periodically save intermediate results (checkpointing) 
to ensure that we do not lose several days worth of computation if we trip over the power cord of our server.
Thus it is time we learned how to load and store both individual weight vectors and entire models. 
This section addresses both issues.
-->

Đến nay, ta đã thảo luận về cách xử lý dữ liệu và cách xây dựng, huấn luyện, kiểm tra những mô hình học sâu.
Tuy nhiên, có thể đến một lúc nào đó ta sẽ cảm thấy hài lòng với những gì thu được và muốn lưu lại kết quả để sau này sử dụng trong những bối cảnh khác nhau (thậm chí có thể đưa ra dự đoán khi triển khai).
Ngoài ra, khi vận hành một quá trình huấn luyện dài hơi, cách tốt nhất là lưu kết quả trung gian một cách định kỳ (điểm kiểm tra) nhằm đảm bảo rằng ta sẽ không mất kết quả tính toán sau nhiều ngày nếu chẳng may ta vấp phải dây nguồn của máy chủ.
Vì vậy, đã đến lúc chúng ta học cách đọc và lưu trữ đồng thời các vector trọng số riêng lẻ cùng toàn bộ các mô hình.
Mục này sẽ giải quyết cả hai vấn đề trên.

<!--
## Loading and Saving `ndarray`s
-->

## Đọc và Lưu các `ndarray`

<!--
For individual `ndarray`s, we can directly invoke their `load` and `save` functions to read and write them respectively. 
Both functions require that we supply a name, and `save` requires as input the variable to be saved.
-->

Đối với `ndarray` riêng lẻ, ta có thể sử dụng trực tiếp các hàm `load` và `save` để đọc và ghi tương ứng.
Cả hai hàm đều yêu cầu ta cung cấp tên, và hàm `save` yêu cầu đầu vào với biến đã được lưu.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

x = np.arange(4)
npx.save('x-file', x)
```

<!--
We can now read this data from the stored file back into memory.
-->

Bây giờ, chúng ta có thể đọc lại dữ liệu từ các tệp được lưu vào trong bộ nhớ.

```{.python .input}
x2 = npx.load('x-file')
x2
```

<!--
MXNet also allows us to store a list of `ndarray`s and read them back into memory.
-->

MXNet đồng thời cho phép ta lưu một danh sách các `ndarray` và đọc lại chúng vào trong bộ nhớ.

```{.python .input  n=2}
y = np.zeros(4)
npx.save('x-files', [x, y])
x2, y2 = npx.load('x-files')
(x2, y2)
```

<!--
We can even write and read a dictionary that maps from strings to `ndarray`s. 
This is convenient when we want to read or write all the weights in a model.
-->

Ta còn có thể ghi và đọc một từ điển ánh xạ từ một chuỗi sang một `ndarray`.
Điều này khá là thuận tiện khi chúng ta muốn đọc hoặc ghi tất cả các trọng số của một mô hình.

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
For this reason Gluon provides built-in functionality to load and save entire networks.
An important detail to note is that this saves model *parameters* and not the entire model. 
For example, if we have a 3 layer MLP, we need to specify the *architecture* separately. 
The reason for this is that the models themselves can contain arbitrary code, hence they cannot be serialized as naturally (and there is a way to do this for compiled models: 
please refer to the [MXNet documentation](http://www.mxnet.io) for technical details). 
Thus, in order to reinstate a model, we need to generate the architecture in code and then load the parameters from disk. 
The deferred initialization (:numref:`sec_deferred_init`) is advantageous here since we can simply define a modelwithout the need to put actual values in place. 
Let us start with our familiar MLP.
-->

Khả năng lưu từng vector trọng số đơn lẻ (hoặc các `ndarray` tensor khác) là hữu ích nhưng sẽ mất nhiều thời gian nếu chúng ta muốn lưu (và sau đó nạp lại) toàn bộ mô hình.
Dù sao, chúng ta có thể có hàng trăm nhóm tham số rải rác xuyên suốt mô hình.
Vì lý do đó mà Gluon cung cấp sẵn tính năng lưu và nạp toàn bộ các mạng.
Một chi tiết quan trọng cần lưu ý là chức năng này chỉ lưu các *tham số* của mô hình, không phải là toàn bộ mô hình.
Điều đó có nghĩa là nếu ta có một MLP ba tầng, ta cần chỉ rõ *kiến trúc* này một cách riêng lẻ.
Lý do là vì bản thân các mô hình có thể chứa mã nguồn bất kỳ, chúng không được thêm vào tập tin một cách dễ dàng như các tham số
(có một cách thực hiện điều này cho các mô hình đã được biên dịch, chi tiết kĩ thuật đọc thêm trong [tài liệu MXNet](http://www.mxnet.io)).
Vì vậy, để khôi phục lại một mô hình thì chúng ta cần xây dựng kiến trúc của nó từ mã nguồn rồi nạp các tham số từ ổ cứng vào kiến trúc này.
Việc khởi tạo trễ (:numref:`sec_deferred_init`) lúc này rất có lợi vì ta chỉ cần định nghĩa một mô hình mà không cần gán giá trị cụ thể cho tham số.
Như thường lệ, hãy bắt đầu với một MLP quen thuộc.


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
Gluon Blocks support a `save_parameters` method that writes all parameters to disk given a string for the file name. 
-->

Tiếp theo, chúng ta lưu các tham số của mô hình vào tệp `mlp.params`.
Những khối Gloun hỗ trợ phương thức từ hàm `save_parameter` nhằm ghi tất cả các tham số vào ổ cứng được cung cấp với một chuỗi những tên tệp. 

```{.python .input}
net.save_parameters('mlp.params')
```

<!--
To recover the model, we instantiate a clone of the original MLP model.
Instead of randomly initializing the model parameters, we read the parameters stored in the file directly.
Conveniently we can load parameters into Blocks via their `load_parameters` method. 
-->

Để khôi phục mô hình, chúng ta tạo một đối tượng khác dựa trên mô hình MLP gốc.
Thay vì khởi tạo ngẫu nhiên những tham số mô hình, ta đọc các tham số được lưu trực tiếp trong tập tin.
Và thật thuận tiện, ta đã có thể nạp các tham số vào khối thông qua phương thức từ hàm `load_parameters`.

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

* Hàm `save` và `load` có thể được sử dụng để thực hiện việc xuất nhập tập tin cho các đối tượng `ndarray`.
* Hàm `load_parameters` và `save_parameters` cho phép ta lưu toàn bộ tập tham số của một mạng trong Gluon.
* Việc lưu kiến trúc này phải được hoàn thiện trong chương trình thay vì trong các tham số.

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

1. Nếu không cần phải triển khai các mô hình huấn luyện sang một thiết bị khác, theo bạn thì lợi ích thực tế của việc lưu các tham số mô hình là gì?
2. Giả sử chúng ta muốn sử dụng lại chỉ một phần của một mạng nào đó để phối hợp với một mạng của một kiến trúc *khác*.
Trong trường hợp ta muốn sử dụng hai lớp đầu tiên của mạng trước đó vào trong một mạng mới, bạn sẽ làm thể nào để thực hiện được việc này?
3. Làm thế nào để bạn có thể lưu kiến trúc mạng và các tham số? Có những hạn chế nào khi bạn tận dụng kiến trúc này?

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2329)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Duy Du
* Phạm Minh Đức
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Phạm Hồng Vinh

<!-- ===================== Bắt đầu phần dịch===================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Concise Implementation of Multilayer Perceptron
-->

# Cách lập trình súc tích Perceptron Đa tầng
:label:`sec_mlp_gluon`

<!--
As you might expect, by relying on the Gluon library, we can implement MLPs even more concisely.
-->

Như bạn đã có thể đoán trước, ta có thể dựa vào thư viện Gluon để lập trình MLP một cách súc tích hơn.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

<!--
## The Model
-->

## Mô hình

<!--
As compared to our gluon implementation of softmax regression implementation (:numref:`sec_softmax_gluon`), 
the only difference is that we add *two* `Dense` (fully-connected) layers (previously, we added *one*).
The first is our hidden layer, which contains *256* hidden units and applies the ReLU activation function.
The second, is our output layer.
-->

So với việc dùng gluon để lập trình hồi quy softmax (:numref:`sec_softmax_gluon`), khác biệt duy nhất ở đây là ta thêm *hai* tầng `Dense` (kết nối đầy đủ), trong khi trước đây ta chỉ có *một*.
Tầng đầu tiên là tầng ẩn, chứa *256* nút ẩn và áp dụng hàm kích hoạt ReLU.
Còn tầng thứ hai là tầng đầu ra.

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

<!--
Note that Gluon, as usual, automatically infers the missing input dimensions to each layer.
-->

Lưu ý rằng như thường lệ, Gluon sẽ tự động suy ra chiều đầu vào còn thiếu cho mỗi tầng.

<!--
The training loop is *exactly* the same as when we implemented softmax regression.
This modularity enables us to separate matterns concerning the model architecture from orthogonal considerations.
-->

Vòng lặp huấn luyện ở đây giống *hệt* như lúc ta lập trình hồi quy softmax.
Lập trình hướng mô-đun như vậy cho phép ta tách các chi tiết liên quan đến kiến trúc của mô hình ra khỏi các mối bận tâm khác.

```{.python .input  n=6}
batch_size, num_epochs = 256, 10
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
```

<!--
## Exercises
-->

## Bài tập

<!--
1. Try adding different numbers of hidden layers. What setting (keeping other parameters and hyperparameters constant) works best?
2. Try out different activation functions. Which ones work best?
3. Try different schemes for initializing the weights. What method works best?
-->

1. Bằng việc thử thêm số lượng các tầng ẩn khác nhau, bạn hãy xem thiết lập nào cho kết quả tốt nhất (giữ nguyên giá trị các tham số và siêu tham số khác)?
2. Bằng việc thử thay đổi các hàm kích hoạt khác nhau, bạn hãy chỉ ra hàm nào mang lại kết quả tốt nhất?
3. Bạn hãy thử các cách khác nhau để khởi tạo trọng số. Phương pháp nào là tốt nhất?

<!-- ===================== Kết thúc phần dịch ===================== -->

<!-- ========================================= REVISE - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2340)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2340)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
* Lý Phi Long
* Vũ Hữu Tiệp
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
* Phạm Minh Đức
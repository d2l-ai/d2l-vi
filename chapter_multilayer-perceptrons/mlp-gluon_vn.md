<!-- ===================== Bắt đầu phần dịch===================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Concise Implementation of Multilayer Perceptron
-->

# *dịch tiêu đề phía trên*
:label:`sec_mlp_gluon`

<!--
As you might expect, by relying on the Gluon library, we can implement MLPs even more concisely.
-->

*dịch đoạn phía trên*

```{.python .input}
import d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

<!--
## The Model
-->

## *dịch tiêu đề phía trên*

<!--
As compared to our gluon implementation of softmax regression implementation (:numref:`sec_softmax_gluon`), 
the only difference is that we add *two* `Dense` (fully-connected) layers (previously, we added *one*).
The first is our hidden layer, which contains *256* hidden units and applies the ReLU activation function.
The second, is our output layer.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))
```

<!--
Note that Gluon, as usual, automatically infers the missing input dimensions to each layer.
-->

*dịch đoạn phía trên*

<!--
The training loop is *exactly* the same as when we implemented softmax regression.
This modularity enables us to separate matterns concerning the model architecture from orthogonal considerations.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

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
* 

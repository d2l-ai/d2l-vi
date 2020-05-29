<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE BẮT ĐẦU =================================== -->

<!--
# Concise Implementation of Recurrent Neural Networks
-->

# Cách lập trình súc tích Mạng nơ-ron Truy hồi
:label:`sec_rnn_gluon`

<!--
While :numref:`sec_rnn_scratch` was instructive to see how recurrent neural networks (RNNs) are implemented, this is not convenient or fast.
This section will show how to implement the same language model more efficiently using functions provided by Gluon.
We begin as before by reading the "Time Machine" corpus.
-->

Dù :numref:`sec_rnn_scratch` đã mô tả cách lập trình mạng nơ-ron truy hồi từ đầu một cách chi tiết, tuy nhiên cách làm này không được nhanh và thuận tiện.
Phần này sẽ hướng dẫn cách lập trình cùng một mô hình ngôn ngữ theo cách hiệu quả hơn bằng các hàm của Gluon.
Ta cũng bắt đầu với việc đọc kho ngữ liệu "Cỗ máy Thời gian".

```{.python .input  n=1}
import d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

<!--
## Defining the Model
-->

## Định nghĩa Mô hình

<!--
Gluon's `rnn` module provides a recurrent neural network implementation (beyond many other sequence models).
We construct the recurrent neural network layer `rnn_layer` with a single hidden layer and 256 hidden units, and initialize the weights.
-->

Mô-đun `rnn` của Gluon đã lập trình sẵn mạng nơ-ron truy hồi (cùng với các mô hình chuỗi khác).
Ta xây dựng tầng truy hồi `rnn_layer` với một tầng ẩn và 256 nút ẩn, và khởi tạo các trọng số.


```{.python .input  n=26}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

<!--
Initializing the state is straightforward. We invoke the member function `rnn_layer.begin_state(batch_size)`.
This returns an initial state for each element in the minibatch.
That is, it returns an object of size (hidden layers, batch size, number of hidden units).
The number of hidden layers defaults to be 1.
In fact, we have not even discussed yet what it means to have multiple layers---this will happen in :numref:`sec_deep_rnn`.
For now, suffice it to say that multiple layers simply amount to the output of one RNN being used as the input for the next RNN.
-->

Việc khởi tạo trạng thái cũng khá đơn giản, chỉ cần gọi phương thức `rnn_layer.begin_state(batch_size)`.
Phương thức này trả về trạng thái ban đầu cho mỗi phần tử trong minibatch.
Tức là nó trả về một đối tượng có kích thước (số tầng ẩn, kích thước batch, số nút ẩn).
Số tầng ẩn mặc định là 1.
Trên thực tế, ta chưa thảo luận việc mạng có nhiều tầng sẽ như thế nào -- điều này sẽ được đề cập ở :numref:`sec_deep_rnn`.
Tạm thời, có thể nói rằng trong mạng nhiều tầng, đầu ra của một RNN sẽ là đầu vào của RNN tiếp theo.

```{.python .input  n=37}
batch_size = 1
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

<!--
With a state variable and an input, we can compute the output with the updated state.
-->

Với một biến trạng thái và một đầu vào, ta có thể tính toán đầu ra và cập nhật trạng thái.


```{.python .input  n=38}
num_steps = 1
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
Similar to :numref:`sec_rnn_scratch`, we define an `RNNModel` block by subclassing the `Block` class for a complete recurrent neural network.
Note that `rnn_layer` only contains the hidden recurrent layers, we need to create a separate output layer.
While in the previous section, we have the output layer within the `rnn` block.
-->

*dịch đoạn phía trên*


```{.python .input  n=39}
# Saved in the d2l package for later use
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of Y to
        # (num_steps * batch_size, num_hiddens). Its output shape is
        # (num_steps * batch_size, vocab_size).
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

<!--
## Training and Predicting
-->

## *dịch tiêu đề phía trên*

<!--
Before training the model, let us make a prediction with the a model that has random weights.
-->

*dịch đoạn phía trên*

```{.python .input  n=42}
ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, len(vocab))
model.initialize(force_reinit=True, ctx=ctx)
d2l.predict_ch8('time traveller', 10, model, vocab, ctx)
```

<!--
As is quite obvious, this model does not work at all. Next, we call `train_ch8` with the same hyper-parameters defined in :numref:`sec_rnn_scratch` and train our model with Gluon.
-->

*dịch đoạn phía trên*

```{.python .input  n=19}
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
```

<!--
Compared with the last section, this model achieves comparable perplexity, albeit within a shorter period of time, due to the code being more optimized.
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## Tóm tắt

<!--
* Gluon's `rnn` module provides an implementation at the recurrent neural network layer.
* Gluon's `nn.RNN` instance returns the output and hidden state after forward computation. This forward computation does not involve output layer computation.
* As before, the computational graph needs to be detached from previous steps for reasons of efficiency.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Compare the implementation with the previous section.
    * Why does Gluon's implementation run faster?
    * If you observe a significant difference beyond speed, try to find the reason.
2. Can you make the model overfit?
    * Increase the number of hidden units.
    * Increase the number of iterations.
    * What happens if you adjust the clipping parameter?
3. Implement the autoregressive model of the introduction to the current chapter using an RNN.
4. What happens if you increase the number of hidden layers in the RNN model? Can you make the model work?
5. How well can you compress the text using this model?
    * How many bits do you need?
    * Why does not everyone use this model for text compression? Hint: what about the compressor itself?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->
<!-- ========================================= REVISE KẾT THÚC =================================== -->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2365)
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

<!-- Phần 2 -->
*

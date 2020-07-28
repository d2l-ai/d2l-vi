<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE BẮT ĐẦU =================================== -->

<!--
# Concise Implementation of Recurrent Neural Networks
-->

# Lập trình súc tích Mạng nơ-ron Hồi tiếp
:label:`sec_rnn_gluon`

<!--
While :numref:`sec_rnn_scratch` was instructive to see how recurrent neural networks (RNNs) are implemented, this is not convenient or fast.
This section will show how to implement the same language model more efficiently using functions provided by Gluon.
We begin as before by reading the "Time Machine" corpus.
-->

Dù :numref:`sec_rnn_scratch` đã mô tả cách lập trình mạng nơ-ron hồi tiếp từ đầu một cách chi tiết, tuy nhiên cách làm này không được nhanh và thuận tiện.
Phần này sẽ hướng dẫn cách lập trình cùng một mô hình ngôn ngữ nhưng hiệu quả hơn bằng các hàm của Gluon.
Như trước, ta cũng bắt đầu với việc đọc kho ngữ liệu "Cỗ máy Thời gian".

```{.python .input  n=1}
from d2l import mxnet as d2l
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

Mô-đun `rnn` của Gluon đã lập trình sẵn mạng nơ-ron hồi tiếp (cùng với các mô hình chuỗi khác).
Ta xây dựng tầng hồi tiếp `rnn_layer` với một tầng ẩn có 256 nút rồi khởi tạo các trọng số.


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
Phương thức này trả về một trạng thái ban đầu cho mỗi phần tử trong minibatch, có kích thước là (số tầng ẩn, kích thước batch, số nút ẩn).
Số tầng ẩn mặc định là 1.
Thực ra ta chưa thảo luận việc mạng có nhiều tầng sẽ như thế nào --- điều này sẽ được đề cập ở :numref:`sec_deep_rnn`.
Tạm thời, có thể nói rằng trong mạng nhiều tầng, đầu ra của một RNN sẽ là đầu vào của RNN tiếp theo.

```{.python .input  n=37}
batch_size = 1
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

<!--
With a state variable and an input, we can compute the output with the updated state.
-->

Với một biến trạng thái và một đầu vào, ta có thể tính đầu ra với trạng thái vừa được cập nhật.


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

Tương tự :numref:`sec_rnn_scratch`, ta định nghĩa khối `RNNModel` bằng cách kế thừa lớp `Block` để xây dựng mạng nơ-ron hồi tiếp hoàn chỉnh.
Chú ý rằng `rnn_layer` chỉ chứa các tầng hồi tiếp ẩn và ta cần tạo riêng biệt một tầng đầu ra, trong khi ở phần trước tầng đầu ra được tích hợp sẵn trong khối `rnn`.

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

## Huấn luyện và Dự đoán

<!--
Before training the model, let us make a prediction with the a model that has random weights.
-->

Trước khi huấn luyện, hãy thử dự đoán bằng mô hình có trọng số ngẫu nhiên.

```{.python .input  n=42}
ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, len(vocab))
model.initialize(force_reinit=True, ctx=ctx)
d2l.predict_ch8('time traveller', 10, model, vocab, ctx)
```

<!--
As is quite obvious, this model does not work at all. Next, we call `train_ch8` with the same hyper-parameters defined in :numref:`sec_rnn_scratch` and train our model with Gluon.
-->

Khá rõ ràng, mô hình này không tốt. Tiếp theo, ta gọi hàm `train_ch8` với các siêu tham số định nghĩa trong :numref:`sec_rnn_scratch` để huấn luyện mô hình bằng Gluon.

```{.python .input  n=19}
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
```

<!--
Compared with the last section, this model achieves comparable perplexity, albeit within a shorter period of time, due to the code being more optimized.
-->

So với phần trước, mô hình này đạt được perplexity tương đương, nhưng thời gian huấn luyện tốt hơn do các đoạn mã được tối ưu hơn.

<!--
## Summary
-->

## Tóm tắt

<!--
* Gluon's `rnn` module provides an implementation at the recurrent neural network layer.
* Gluon's `nn.RNN` instance returns the output and hidden state after forward computation. This forward computation does not involve output layer computation.
* As before, the computational graph needs to be detached from previous steps for reasons of efficiency.
-->

* Mô-đun `rnn` của Gluon đã lập trình sẵn tầng mạng nơ-ron hồi tiếp.
* Mỗi thực thể của `nn.RNN` trả về đầu ra và trạng thái ẩn sau lượt truyền xuôi. Lượt truyền xuôi này không bao gồm tính toán tại tầng đầu ra.
* Như trước, đồ thị tính toán cần được tách khỏi các bước trước đó để đảm bảo hiệu năng.

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

1. So sánh với cách lập trình từ đầu ở phần trước.
    * Tại sao lập trình bằng Gluon chạy nhanh hơn?
    * Nếu bạn nhận thấy khác biệt đáng kể nào khác ngoài tốc độ, hãy thử tìm hiểu tại sao.
2. Bạn có thể làm quá khớp mô hình này không? Hãy thử
    * Tăng số nút ẩn.
    * Tăng số vòng lặp.
    * Thay đổi tham số gọt (*clipping*) thì sao?
3. Hãy lập trình mô hình tự hồi quy ở phần giới thiệu của chương này bằng RNN.
4. Nếu tăng số tầng ẩn của mô hình RNN thì sao? Bạn có thể làm mô hình hoạt động không?
5. Có thể nén văn bản bằng cách sử dụng mô hình này không?
    * Nếu có thì cần bao nhiêu bit?
    * Tại sao không ai sử dụng mô hình này để nén văn bản? Gợi ý: bản thân bộ nén thì sao?


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->
<!-- ========================================= REVISE KẾT THÚC =================================== -->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2365)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Cường
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc

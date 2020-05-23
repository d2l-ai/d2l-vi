<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Implementation of Recurrent Neural Networks from Scratch
-->

# Lập trình Mạng nơ-ron Truy hồi từ đầu
:label:`sec_rnn_scratch`

<!--
In this section we implement a language model introduced in :numref:`chap_rnn` from scratch.
It is based on a character-level recurrent neural network trained on H. G. Wells' *The Time Machine*.
As before, we start by reading the dataset first, which is introduced in :numref:`sec_language_model`.
-->

*dịch đoạn phía trên*


```{.python .input  n=14}
%matplotlib inline
import d2l
import math
from mxnet import autograd, np, npx, gluon
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

<!--
## One-hot Encoding
-->

## *dịch tiêu đề phía trên*

<!--
Remember that each token is presented as a numerical index in `train_iter`.
Feeding these indices directly to the neural network might make it hard to learn.
We often present each token as a more expressive feature vector.
The easiest representation is called *one-hot encoding*.
-->

*dịch đoạn phía trên*

<!--
In a nutshell, we map each index to a different unit vector: assume that the number of different tokens in the vocabulary is $N$ (the `len(vocab)`) and the token indices range from 0 to $N-1$.
If the index of a token is the integer $i$, then we create a vector $\mathbf{e}_i$ of all 0s with a length of $N$ and set the element at position $i$ to 1.
This vector is the one-hot vector of the original token.
The one-hot vectors with indices 0 and 2 are shown below.
-->

*dịch đoạn phía trên*


```{.python .input  n=21}
npx.one_hot(np.array([0, 2]), len(vocab))
```

<!--
The shape of the minibatch we sample each time is (batch size, timestep).
The `one_hot` function transforms such a minibatch into a 3-D tensor with the last dimension equals to the vocabulary size.
We often transpose the input so that we will obtain a (timestep, batch size, vocabulary size) output that fits into a sequence model easier.
-->

*dịch đoạn phía trên*


```{.python .input  n=18}
X = np.arange(batch_size * num_steps).reshape(batch_size, num_steps)
npx.one_hot(X.T, len(vocab)).shape
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Initializing the Model Parameters
-->

## *dịch tiêu đề phía trên*

<!--
Next, we initialize the model parameters for a RNN model.
The number of hidden units `num_hiddens` is a tunable parameter.
-->

*dịch đoạn phía trên*


```{.python .input  n=19}
def get_params(vocab_size, num_hiddens, ctx):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=ctx)
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = np.zeros(num_hiddens, ctx=ctx)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = np.zeros(num_outputs, ctx=ctx)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

<!--
## RNN Model
-->

## *dịch tiêu đề phía trên*

<!--
First, we need an `init_rnn_state` function to return the hidden state at initialization.
It returns an `ndarray` filled with 0 and with a shape of (batch size, number of hidden units).
Using tuples makes it easier to handle situations where the hidden state contains multiple variables (e.g., when combining multiple layers in an RNN where each layer requires initializing).
-->

*dịch đoạn phía trên*


```{.python .input  n=20}
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (np.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

<!--
The following `rnn` function defines how to compute the hidden state and output in a timestep.
The activation function here uses the $\tanh$ function.
As described in :numref:`sec_mlp`, the mean value of the $\tanh$ function is 0, when the elements are evenly distributed over the real numbers.
-->

*dịch đoạn phía trên*


```{.python .input  n=6}
def rnn(inputs, state, params):
    # Inputs shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

<!--
Now we have all functions defined, next we create a class to wrap these functions and store parameters.
-->

*dịch đoạn phía trên*


```{.python .input}
# Saved in the d2l package for later use
class RNNModelScratch:
    """A RNN Model based on scratch implementations."""

    def __init__(self, vocab_size, num_hiddens, ctx,
                 get_params, init_state, forward):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, ctx)
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

<!--
Let us do a sanity check whether inputs and outputs have the correct dimensions, e.g., to ensure that the dimensionality of the hidden state has not changed.
-->

*dịch đoạn phía trên*

```{.python .input}
num_hiddens, ctx = 512, d2l.try_gpu()
model = RNNModelScratch(len(vocab), num_hiddens, ctx, get_params,
                        init_rnn_state, rnn)
state = model.begin_state(X.shape[0], ctx)
Y, new_state = model(X.as_in_ctx(ctx), state)
Y.shape, len(new_state), new_state[0].shape
```

<!--
We can see that the output shape is (number steps $\times$ batch size, vocabulary size), while the hidden state shape remains the same, i.e., (batch size, number of hidden units).
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Prediction
-->

## *dịch tiêu đề phía trên*

<!--
We first explain the predicting function so we can regularly check the prediction during training.
This function predicts the next `num_predicts` characters based on the `prefix` (a string containing several characters).
For the beginning of the sequence, we only update the hidden state.
After that we begin generating new characters and emitting them.
-->

*dịch đoạn phía trên*


```{.python .input}
# Saved in the d2l package for later use
def predict_ch8(prefix, num_predicts, model, vocab, ctx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    outputs = [vocab[prefix[0]]]

    def get_input():
        return np.array([outputs[-1]], ctx=ctx).reshape(1, 1)
    for y in prefix[1:]:  # Warmup state with prefix
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

<!--
We test the `predict_rnn` function first.
Given that we did not train the network, it will generate nonsensical predictions.
We initialize it with the sequence `traveller ` and have it generate 10 additional characters.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
predict_ch8('time traveller ', 10, model, vocab, ctx)
```

<!--
## Gradient Clipping
-->

## *dịch tiêu đề phía trên*

<!--
For a sequence of length $T$, we compute the gradients over these $T$ timesteps in an iteration, which results in a chain of matrix-products with length $\mathcal{O}(T)$ during backpropagating.
As mentioned in :numref:`sec_numerical_stability`, it might result in numerical instability, e.g., the gradients may either explode or vanish, when $T$ is large.
Therefore, RNN models often need extra help to stabilize the training.
-->

*dịch đoạn phía trên*

<!--
Recall that when solving an optimization problem, we take update steps for the weights $\mathbf{w}$ in the general direction of the negative gradient $\mathbf{g}_t$ on a minibatch, 
say $\mathbf{w} - \eta \cdot \mathbf{g}_t$. Let us further assume that the objective is well behaved, i.e., it is Lipschitz continuous with constant $L$, i.e.,
-->

*dịch đoạn phía trên*


$$|l(\mathbf{w}) - l(\mathbf{w}')| \leq L \|\mathbf{w} - \mathbf{w}'\|.$$

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
In this case we can safely assume that if we update the weight vector by $\eta \cdot \mathbf{g}_t$, we will not observe a change by more than $L \eta \|\mathbf{g}_t\|$.
This is both a curse and a blessing.
A curse since it limits the speed of making progress, whereas a blessing since it limits the extent to which things can go wrong if we move in the wrong direction.
-->

*dịch đoạn phía trên*

<!--
Sometimes the gradients can be quite large and the optimization algorithm may fail to converge.
We could address this by reducing the learning rate $\eta$ or by some other higher order trick.
But what if we only rarely get large gradients?
In this case such an approach may appear entirely unwarranted.
One alternative is to clip the gradients by projecting them back to a ball of a given radius, say $\theta$ via
-->

*dịch đoạn phía trên*

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

<!--
By doing so we know that the gradient norm never exceeds $\theta$ and that the updated gradient is entirely aligned with the original direction $\mathbf{g}$.
It also has the desirable side-effect of limiting the influence any given minibatch (and within it any given sample) can exert on the weight vectors.
This bestows a certain degree of robustness to the model.
Gradient clipping provides a quick fix to the gradient exploding.
While it does not entirely solve the problem, it is one of the many techniques to alleviate it.
-->

*dịch đoạn phía trên*

<!--
Below we define a function to clip the gradients of a model that is either a `RNNModelScratch` instance or a Gluon model.
Also note that we compute the gradient norm over all parameters.
-->

*dịch đoạn phía trên*


```{.python .input  n=10}
# Saved in the d2l package for later use
def grad_clipping(model, theta):
    if isinstance(model, gluon.Block):
        params = [p.data() for p in model.collect_params().values()]
    else:
        params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Training
-->

## *dịch tiêu đề phía trên*

<!--
Let us first define the function to train the model on one data epoch.
It differs from the models training of :numref:`sec_softmax_scratch` in three places:
-->

*dịch đoạn phía trên*

<!--
1. Different sampling methods for sequential data (independent sampling and sequential partitioning) will result in differences in the initialization of hidden states.
2. We clip the gradients before updating the model parameters. 
This ensures that the model does not diverge even when gradients blow up at some point during the training process, and it effectively reduces the step size automatically.
3. We use perplexity to evaluate the model. This ensures that sequences of different length are comparable.
-->

*dịch đoạn phía trên*


<!--
When the consecutive sampling is used, we initialize the hidden state at the beginning of each epoch.
Since the $i^\mathrm{th}$ example in the next minibatch is adjacent to the current $i^\mathrm{th}$ example, 
so the next minibatch can use the current hidden state directly, we only detach the gradient so that we compute the gradients within a minibatch.
When using the random sampling, we need to re-initialize the hidden state for each iteration since each example is sampled with a random position.
Same as the `train_epoch_ch3` function in :numref:`sec_softmax_scratch`, we use generalized `updater`, which could be either a Gluon trainer or a scratched implementation.
-->

*dịch đoạn phía trên*


```{.python .input}
# Saved in the d2l package for later use
def train_epoch_ch8(model, train_iter, loss, updater, ctx, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # loss_sum, num_examples
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize state when either it is the first iteration or
            # using random sampling.
            state = model.begin_state(batch_size=X.shape[0], ctx=ctx)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(ctx), y.as_in_ctx(ctx)
        with autograd.record():
            py, state = model(X, state)
            l = loss(py, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)  # Since used mean already
        metric.add(l * y.size, y.size)
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()
```

<!--
The training function again supports either we implement the model from scratch or using Gluon.
-->

*dịch đoạn phía trên*


```{.python .input  n=11}
# Saved in the d2l package for later use
def train_ch8(model, train_iter, vocab, lr, num_epochs, ctx,
              use_random_iter=False):
    # Initialize
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[1, num_epochs])
    if isinstance(model, gluon.Block):
        model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
        trainer = gluon.Trainer(model.collect_params(),
                                'sgd', {'learning_rate': lr})

        def updater(batch_size):
            return trainer.step(batch_size)
    else:
        def updater(batch_size):
            return d2l.sgd(model.params, lr, batch_size)

    def predict(prefix):
        return predict_ch8(prefix, 50, model, vocab, ctx)

    # Train and check the progress.
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, ctx, use_random_iter)
        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    print('Perplexity %.1f, %d tokens/sec on %s' % (ppl, speed, ctx))
    print(predict('time traveller'))
    print(predict('traveller'))
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
Now we can train a model.
Since we only use $10,000$ tokens in the dataset, the model needs more epochs to converge.
-->

Bây giờ ta có thể huấn luyện mô hình.
Vì ta chỉ có thể sử dụng $10,000$ token trong tập dữ liệu nên mô hình sẽ cần nhiều epoch hơn để hội tụ.


```{.python .input}
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
```

<!--
Finally let us check the results to use a random sampling iterator.
-->

Cuối cùng, ta sẽ kiểm tra kết quả để sử dụng một bộ lặp lấy mẫu ngẫu nhiên.


```{.python .input}
train_ch8(model, train_iter, vocab, lr, num_epochs, ctx, use_random_iter=True)
```

<!--
While implementing the above RNN model from scratch is instructive, it is not convenient. In the next section we will see how to improve significantly on the current model and how to make it faster and easier to implement.
-->

Mặc dù lập trình mô hình RNN ở trên từ đầu mang tính hướng dẫn, nhưng nó không thực sự thuận tiện. Trong phần tiếp theo, ta sẽ xem cách cải thiện đáng kể mô hình hiện tại và cách làm cho nó nhanh hơn và dễ triển khai hơn.


<!--
## Summary
-->

## Tóm tắt

<!--
* Sequence models need state initialization for training.
* Between sequential models you need to ensure to detach the gradients, to ensure that the automatic differentiation does not propagate effects beyond the current sample.
* A simple RNN language model consists of an encoder, an RNN model, and a decoder.
* Gradient clipping prevents gradient explosion (but it cannot fix vanishing gradients).
* Perplexity calibrates model performance across different sequence length. It is the exponentiated average of the cross-entropy loss.
* Sequential partitioning typically leads to better models.
-->

* Mô hình chuỗi cần khởi tạo trạng thái cho quá trình huấn luyện.
* Giữa các mô hình chuỗi, ta cần đảm bảo tách các gradient, để chắc chắn rằng phép vi phân tự động không lan truyền các hiệu ứng ngoài mẫu hiện tại.
* Mô hình ngôn ngữ RNN đơn giản bao gồm bộ mã hóa, mô hình RNN và bộ giải mã.
* Cắt gradient ngăn chặn sự bùng nổ gradient (nhưng nó không thể khắc phục vấn đề tiêu biến gradient).
* Perplexity điều chỉnh hiệu măng của mô hình trên các độ dài chuỗi khác nhau. Đây là trung bình lũy thừa của mất mát cross-entropy.
* Chia dữ liệu tuần tự thường tạo ra các mô hình tốt hơn.

<!--
## Exercises
-->

## Bài tập

<!--
1. Show that one-hot encoding is equivalent to picking a different embedding for each object.
2. Adjust the hyperparameters to improve the perplexity.
    * How low can you go? Adjust embeddings, hidden units, learning rate, etc.
    * How well will it work on other books by H. G. Wells, e.g., [The War of the Worlds](http://www.gutenberg.org/ebooks/36).
3. Modify the predict function such as to use sampling rather than picking the most likely next character.
    * What happens?
    * Bias the model towards more likely outputs, e.g., by sampling from $q(w_t \mid w_{t-1}, \ldots, w_1) \propto p^\alpha(w_t \mid w_{t-1}, \ldots, w_1)$ for $\alpha > 1$.
4. Run the code in this section without clipping the gradient. What happens?
5. Change adjacent sampling so that it does not separate hidden states from the computational graph. Does the running time change? How about the accuracy?
6. Replace the activation function used in this section with ReLU and repeat the experiments in this section.
7. Prove that the perplexity is the inverse of the harmonic mean of the conditional word probabilities.
-->

1. Chứng minh rằng biễu diễn one-hot tương đương với việc chọn một embedding khác nhau cho từng đối tượng.
2. Điều chỉnh các siêu tham số để cải thiện perplexity.
    * Bạn có thể giảm lỗi xuống bao nhiêu? Điều chỉnh embedding, nút ẩn, tốc độ học, vv
    * Mô hình này sẽ hoạt động như thế nào trên các cuốn sách khác của H. G. Wells, ví dụ như [The War of the Worlds] (http://www.gutenberg.org/ebooks/36).
3. Thay đổi hàm dự đoán như sử dụng lấy mẫu thay vì chọn ký tự tiếp theo có khả năng cao nhất.
    * Điều gì sẽ xảy ra?
    * Điều chỉnh mô hình để thiên vị các đầu ra có khả năng cao hơn, ví dụ: bằng cách lấy mẫu từ $q(w_t \mid w_{t-1}, \ldots, w_1) \propto p^\alpha(w_t \mid w_{t-1}, \ldots, w_1)$ for $\alpha > 1$.
4. Điều gì sẽ xảy ra nếu ta chạy mã nguồn trong phần này mà không cắt gradient?
5. Thay đổi phép lấy mẫu liền kề để nó không tách các trạng thái ẩn khỏi biểu đồ tính toán. Thời gian chạy có thay đổi không? Độ chính xác thì sao?
6. Thay thế hàm kích hoạt được sử dụng trong phần này bằng ReLU và thực hiện lại các thử nghiệm.
7. Chứng minh rằng perplexity là nghịch đảo của harmonic mean của xác suất từ có điều kiện.

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2364)
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
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

<!-- Phần 6 -->
* Nguyễn Duy Du

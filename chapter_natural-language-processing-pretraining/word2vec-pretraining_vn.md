<!--
# Pretraining word2vec
-->

# Tiền huấn luyện word2vec
:label:`sec_word2vec_pretraining`


<!--
In this section, we will train a skip-gram model defined in:numref:`sec_word2vec`.
-->

Trong phần này, ta sẽ huấn luyện một mô hình skip-gram đã được định nghĩa ở :numref:`sec_word2vec`.


<!--
First, import the packages and modules required for the experiment, and load the PTB dataset.
-->

Đầu tiên, ta nhập các gói thư viện và mô-đun cần thiết cho thí nghiệm, và nạp tập dữ liệu PTB.


```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```


<!--
## The Skip-Gram Model
-->

## Mô hình Skip-Gram


<!--
We will implement the skip-gram model by using embedding layers and minibatch multiplication.
These methods are also often used to implement other natural language processing applications.
-->

Ta sẽ lập trình mô hình skip-gram bằng cách sử dụng các tầng embedding và phép nhân minibatch.
Các phương pháp này cũng thường được sử dụng để lập trình các ứng dụng xử lý ngôn ngữ tự nhiên khác.


<!--
### Embedding Layer
-->

### Tầng Embedding


<!--
The layer in which the obtained word is embedded is called the embedding layer, which can be obtained by creating an `nn.Embedding` instance in Gluon.
The weight of the embedding layer is a matrix whose number of rows is the dictionary size (`input_dim`) and whose number of columns is the dimension of each word vector (`output_dim`).
We set the dictionary size to $20$ and the word vector dimension to $4$.
-->

Để thu được các embedding từ, ta sử dụng tầng embedding, có thể được tạo bằng một thực thể `nn.Embedding` trong Gluon.
Trọng số của tầng embedding là một ma trận có số hàng là kích thước từ điển (`input_dim`) và số cột là chiều của mỗi vector từ (`output_dim`).
Ta đặt kích thước từ điển bằng $20$ và chiều vector từ là $4$.


```{.python .input  n=15}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```


<!--
The input of the embedding layer is the index of the word.
When we enter the index $i$ of a word, the embedding layer returns the $i^\mathrm{th}$ row of the weight matrix as its word vector.
Below we enter an index of shape ($2$, $3$) into the embedding layer.
Because the dimension of the word vector is 4, we obtain a word vector of shape ($2$, $3$, $4$).
-->

Đầu vào của tầng embedding là chỉ số của từ.
Khi ta nhập vào chỉ số $i$ của một từ, tầng embedding sẽ trả về vector từ tương ứng là hàng thứ $i$ của ma trận trọng số.
Dưới đây ta nhập vào tầng embedding một chỉ số có kích thước ($2$, $3$).
Vì số chiều vector từ là 4, ta thu được vector từ kích thước ($2$, $3$, $4$).

```{.python .input  n=16}
x = np.array([[1, 2, 3], [4, 5, 6]])
embed(x)
```


<!--
### Minibatch Multiplication
-->

### Phép nhân Minibatch


<!--
We can multiply the matrices in two minibatches one by one, by the minibatch multiplication operation `batch_dot`.
Suppose the first batch contains $n$ matrices $\mathbf{X}_1, \ldots, \mathbf{X}_n$ with a shape of $a\times b$, 
and the second batch contains $n$ matrices $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ with a shape of $b\times c$.
The output of matrix multiplication on these two batches are $n$ matrices $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ with a shape of $a\times c$.
Therefore, given two tensors of shape ($n$, $a$, $b$) and ($n$, $b$, $c$), the shape of the minibatch multiplication output is ($n$, $a$, $c$).
-->

Ta có thể nhân các ma trận trong hai minibatch bằng toán tử nhân minibatch `batch_dot`.
Giả sử batch đầu tiên chứa $n$ ma trận $\mathbf{X}_1, \ldots, \mathbf{X}_n$ có kích thước là $a\times b$, 
và batch thứ hai chứa $n$ ma trận $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ có kích thước là $b\times c$.
Đầu ra của toán tử nhân ma trận trên hai batch đầu vào là $n$ ma trận $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ có kích thước là $a\times c$.   
Do đó, với hai tensor có kích thước là ($n$, $a$, $b$) và ($n$, $b$, $c$), kích thước đầu ra của toán tử nhân minibatch là ($n$, $a$, $c$).


```{.python .input  n=17}
X = np.ones((2, 1, 4))
Y = np.ones((2, 4, 6))
npx.batch_dot(X, Y).shape
```


<!--
### Skip-gram Model Forward Calculation
-->

### Tính toán Truyền xuôi của Mô hình Skip-Gram 


<!--
In forward calculation, the input of the skip-gram model contains the central target word index `center`
and the concatenated context and noise word index `contexts_and_negatives`.
In which, the `center` variable has the shape (batch size, 1),
while the `contexts_and_negatives` variable has the shape (batch size, `max_len`).
These two variables are first transformed from word indexes to word vectors by the word embedding layer, 
and then the output of shape (batch size, 1, `max_len`) is obtained by minibatch multiplication.
Each element in the output is the inner product of the central target word vector and the context word vector or noise word vector.
-->

Ở lượt truyền xuôi, đầu vào của mô hình skip-gram chứa chỉ số `center` của từ đích trung tâm
và chỉ số `contexts_and_negatives` được nối lại từ chỉ số của từ ngữ cảnh và từ nhiễu.
Trong đó, biến `center` có kích thước là (kích thước batch, 1),
và biến `contexts_and_negatives` có kích thước là (kích thước batch, `max_len`).
Đầu tiên hai biến này được biến đổi từ chỉ số từ thành vector từ bởi tầng embedding từ, sau đó đầu ra có kích thước là (kích thước batch, 1, `max_len`) thu được bằng phép nhân minibatch.
Mỗi phần tử của đầu ra là tích vô hướng của vector từ đích trung tâm và vector từ ngữ cảnh hoặc vector từ nhiễu.


```{.python .input  n=18}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```


<!--
Verify that the output shape should be (batch size, 1, `max_len`).
-->

Hãy xác nhận kích thước đầu ra là (kích thước batch, 1, `max_len`).


```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```


<!--
## Training
-->

## Huấn luyện


<!--
Before training the word embedding model, we need to define the loss function of the model.
-->

Trước khi huấn luyện mô hình embedding từ, ta cần định nghĩa hàm mất mát của mô hình.


<!--
### Binary Cross Entropy Loss Function
-->

### Hàm Mất mát Entropy chéo Nhị phân


<!--
According to the definition of the loss function in negative sampling, we can directly use Gluon's binary cross-entropy loss function `SigmoidBinaryCrossEntropyLoss`.
-->

Theo định nghĩa hàm mất mát trong phương pháp lấy mẫu âm, ta có thể sử dụng trực tiếp hàm mất mát entropy chéo nhị phân của Gluon `SigmoidBinaryCrossEntropyLoss`.


```{.python .input  n=19}
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
```


<!--
It is worth mentioning that we can use the mask variable to specify the partial predicted value and label that participate in loss function calculation in the minibatch: 
when the mask is 1, the predicted value and label of the corresponding position will participate in the calculation of the loss function; 
When the mask is 0, the predicted value and label of the corresponding position do not participate in the calculation of the loss function.
As we mentioned earlier, mask variables can be used to avoid the effect of padding on loss function calculations.
-->

Lưu ý là ta có thể sử dụng biến mặt nạ để chỉ định một phần giá trị dự đoán và nhãn được dùng khi tính hàm mất mát trong minibatch:
khi mặt nạ bằng 1, giá trị dự đoán và nhãn của vị trí tương ứng sẽ được dùng trong phép tính hàm mất mát;
khi mặt nạ bằng 0, giá trị dự đoán và nhãn của vị trí tương ứng sẽ không được dùng trong phép tính hàm mất mát.
Như đã đề cập, các biến mặt nạ có thể được sử dụng nhằm tránh ảnh hưởng của vùng đệm lên phép tính hàm mất mát.


<!--
Given two identical examples, different masks lead to different loss values.
-->

Với hai mẫu giống nhau, mặt nạ khác nhau sẽ dẫn đến giá trị mất mát cũng khác nhau.


```{.python .input}
pred = np.array([[.5]*4]*2)
label = np.array([[1, 0, 1, 0]]*2)
mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask)
```


<!--
We can normalize the loss in each example due to various lengths in each example.
-->

Ta có thể chuẩn hóa mất mát trong từng mẫu do các mẫu có độ dài khác nhau.


```{.python .input}
loss(pred, label, mask) / mask.sum(axis=1) * mask.shape[1]
```


<!--
### Initializing Model Parameters
-->

### Khởi tạo Tham số Mô hình


<!--
We construct the embedding layers of the central and context words, respectively, and set the hyperparameter word vector dimension `embed_size` to 100.
-->

Ta khai báo tầng embedding lần lượt của từ trung tâm và từ ngữ cảnh, và đặt siêu tham số số chiều của vector từ `embed_size` bằng 100.


```{.python .input  n=20}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```


<!--
### Training
-->

### Huấn luyện


<!--
The training function is defined below.
Because of the existence of padding, the calculation of the loss function is slightly different compared to the previous training functions.
-->

Hàm huấn luyện được định nghĩa như dưới đây.
Do có phần đệm nên phép tính mất mát có một chút khác biệt so với các hàm huấn luyện trước.


```{.python .input  n=21}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # Sum of losses, no. of tokens
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i+1) % 50 == 0:
                animator.add(epoch+(i+1)/len(data_iter),
                             (metric[0]/metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```


<!--
Now, we can train a skip-gram model using negative sampling.
-->

Giờ ta có thể huấn luyện một mô hình skip-gram sử dụng phương pháp lấy mẫu âm.


```{.python .input  n=22}
lr, num_epochs = 0.01, 5
train(net, data_iter, lr, num_epochs)
```


<!--
## Applying the Word Embedding Model
-->

## Áp dụng Mô hình Embedding Từ


<!--
After training the word embedding model, we can represent similarity in meaning between words based on the cosine similarity of two word vectors.
As we can see, when using the trained word embedding model, the words closest in meaning to the word "chip" are mostly related to chips.
-->

Sau khi huấn luyện mô hình embedding từ, ta có thể biểu diễn sự tương tự về nghĩa giữa các từ dựa trên độ tương tự cô-sin giữa hai vector từ.
Có thể thấy, khi sử dụng mô hình embedding từ đã được huấn luyện, các từ có nghĩa gần nhất với từ "chip" hầu hết là những từ có liên quan đến chip xử lý.


```{.python .input  n=23}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.idx_to_token[i]}')

get_similar_tokens('chip', 3, net[0])
```


## Tóm tắt

<!--
We can pretrain a skip-gram model through negative sampling.
-->

Ta có thể tiền huấn luyện một mô hình skip-gram thông qua phương pháp lấy mẫu âm.


## Bài tập


<!--
1. Set `sparse_grad=True` when creating an instance of `nn.Embedding`.
Does it accelerate training? Look up MXNet documentation to learn the meaning of this argument.
2. Try to find synonyms for other words.
3. Tune the hyperparameters and observe and analyze the experimental results.
4. When the dataset is large, we usually sample the context words and the noise words for the central target word in the current minibatch only when updating the model parameters.
In other words, the same central target word may have different context words or noise words in different epochs.
What are the benefits of this sort of training? Try to implement this training method.
-->

1. Đặt `sparse_grad=True` khi tạo một đối tượng `nn.Embedding`.
Việc này có tăng tốc quá trình huấn luyện không? Hãy tra tài liệu của MXNet để tìm hiểu ý nghĩa của tham số này.
2. Hãy tìm từ đồng nghĩa cho các từ khác.
3. Điều chỉnh các siêu tham số, quan sát và phân tích kết quả thí nghiệm.
4. Khi tập dữ liệu lớn, ta thường lấy mẫu các từ ngữ cảnh và các từ nhiễu cho từ đích trung tâm trong minibatch hiện tại chỉ khi cập nhật tham số mô hình.
Nói cách khác, cùng một từ đích trung tâm có thể có các từ ngữ cảnh và từ nhiễu khác nhau với mỗi epoch khác nhau.
Cách huấn luyện này có lợi ích gì? Hãy thử lập trình phương pháp huấn luyện này.


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/384)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Đỗ Trường Giang
* Phạm Minh Đức
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Nguyễn Văn Cường

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 21/07/2020)*

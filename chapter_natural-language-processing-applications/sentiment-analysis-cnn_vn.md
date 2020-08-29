<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Sentiment Analysis: Using Convolutional Neural Networks
-->

# Phân tích Cảm xúc: Sử dụng Mạng Nơ-ron Tích Chập
:label:`sec_sentiment_cnn`


<!--
In :numref:`chap_cnn`, we explored how to process two-dimensional image data with two-dimensional convolutional neural networks.
In the previous language models and text classification tasks, we treated text data as a time series with only one dimension, and naturally,
we used recurrent neural networks to process such data.
In fact, we can also treat text as a one-dimensional image, so that we can use one-dimensional
convolutional neural networks to capture associations between adjacent words. 
As described in :label:`fig_nlp-map-sa-cnn` This section describes a groundbreaking approach to applying
convolutional neural networks to sentiment analysis: textCNN :cite:`Kim.2014`.
-->

Trong chương :numref:`chap_cnn`, chúng ta đã tìm hiểu cách xử lí dữ liệu ảnh hai chiều với mạng nơ-ron tích chập hai chiều.
Như đề cập về tác vụ mô hình ngôn ngữ và phân loại văn bản ở chương trước, chúng ta coi dữ liệu văn bản như là dữ liệu chuỗi thời gian với chỉ một chiều duy nhất, và vì vậy,
chúng sẽ được xử lí bằng mạng nơ-ron hồi tiếp.
Thực tế, chúng ta cũng có thể coi văn bản như một bức ảnh một chiều, và sử dụng mạng nơ-ron một chiều để tìm ra mối liên kết giữa những từ liền kề nhau.
Theo như :label:`fig_nlp-map-sa-cnn` Chương này sẽ miêu tả một hướng tiếp cận đột phá bằng cách áp dụng
mạng nơ-ron tích chập để phân tích cảm xúc: textCNN :cite:`Kim.2014`.


<!--
![This section feeds pretrained GloVe to a CNN-based architecture for sentiment analysis.](../img/nlp-map-sa-cnn.svg)
-->

![Phần này truyền mô hình tiền huấn luyện GloVe vào một kiến trúc mạng nơ-ron tích chập cho tác vụ phân loại cảm xúc](../img/nlp-map-sa-cnn.svg)
:label:`fig_nlp-map-sa-cnn`


<!--
First, import the packages and modules required for the experiment.
-->

Đầu tiên, nhập những gói thư viện và mô-đun cần thiết cho thử nghiệm


```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size = 64
train_iter, test_iter, vocab = d2l.load_data_imdb(batch_size)
```


<!--
## One-Dimensional Convolutional Layer
-->

## Mạng Nơ-ron Tích Chập Một Chiều


<!--
Before introducing the model, let us explain how a one-dimensional convolutional layer works.
Like a two-dimensional convolutional layer, a one-dimensional convolutional layer uses a one-dimensional cross-correlation operation.
In the one-dimensional cross-correlation operation, the convolution window starts from 
the leftmost side of the input array and slides on the input array from left to right successively.
When the convolution window slides to a certain position, the input subarray in the window and kernel array 
are multiplied and summed by element to get the element at the corresponding location in the output array.
As shown in :numref:`fig_conv1d`, the input is a one-dimensional array with a width of 7 and the width of the kernel array is 2.
As we can see, the output width is $7-2+1=6$ and the first element is obtained by performing multiplication 
by element on the leftmost input subarray with a width of 2 and kernel array and then summing the results.
-->

Trước khi giới thiệu mô hình, chúng ta hãy xem làm cách nào mà một mạng nơ-ron tích chập một chiều hoạt động.
Tương tự như mạng nơ-ron tích chập hai chiều, một mạng nơ-ron tích chập một chiều sử dụng phép tính tương quan chéo một chiều.
Trong phép tính tương quan chéo một chiều, cửa sổ tích chập bắt đầu từ phía ngoài cùng bên trái của mảng đầu vào và trượt liên tiếp từ trái qua phải của mảng đầu vào.
Xét trên một vị trí nhất định của cửa sổ tích chập khi trượt, mảng đầu vào con trong cửa sổ đó và mảng hạt nhân
được thực hiện phép tính nhân và cộng từng phần tử để lấy được từng phần tử ở vị trí tương ứng trong mảng đầu ra.
Như ví dụ ở :numref:`fig_conv1d`, đầu vào là một mảng một chiều với độ rộng là 7 và độ rộng của mảng hạt nhân là 2.
Chúng ta có thể thấy rằng độ rộng của đầu ra là $7-2+1=6$ và phần tử đầu tiên có được bởi phép tính nhân
từng phần tử giữa mảng con 2 phần tử ngoài cùng bên trái của đầu vào và mảng hạt nhân, và được cộng lại với nhau sau đó.


<!--
![One-dimensional cross-correlation operation. The shaded parts are the first output element as well as the input and kernel array elements used in its calculation: $0\times1+1\times2=2$.](../img/conv1d.svg)
-->

![Phép tính tương quan chéo một chiều. Những vùng in đậm là phần tử đầu tiên của đầu ra, phần tử của đầu vào và mảng hạt nhân được dùng trong phép tính: $0\times1+1\times2=2$.](../img/conv1d.svg)
:label:`fig_conv1d`


<!--
Next, we implement one-dimensional cross-correlation in the `corr1d` function.
It accepts the input array `X` and kernel array `K` and outputs the array `Y`.
-->

Tiếp theo, chúng ta sẽ triển khai tương quan chéo một chiều trong hàm `corr1d`.
Hàm này chấp nhận mảng đầu vào `X` và mảng hạt nhân `K` và cho ra đầu ra là mảng `Y`


```{.python .input  n=2}
def corr1d(X, K):
    w = K.shape[0]
    Y = np.zeros((X.shape[0] - w + 1))
    for i in range(Y.shape[0]):
        Y[i] = (X[i: i + w] * K).sum()
    return Y
```


<!--
Now, we will reproduce the results of the one-dimensional cross-correlation operation in :numref:`fig_conv1d`.
-->

Bây giờ chúng ta sẽ tái hiện lại kết quả của phép tính tương quan chéo một chiều ở :numref:`fig_conv1d`.


```{.python .input  n=3}
X, K = np.array([0, 1, 2, 3, 4, 5, 6]), np.array([1, 2])
corr1d(X, K)
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->


<!--
The one-dimensional cross-correlation operation for multiple input channels is also similar to the two-dimensional cross-correlation operation for multiple input channels.
On each channel, it performs the one-dimensional cross-correlation operation on the kernel and its corresponding input and adds the results of the channels to get the output.
:numref:`fig_conv1d_channel` shows a one-dimensional cross-correlation operation with three input channels.
-->

*dịch đoạn phía trên*


<!--
![One-dimensional cross-correlation operation with three input channels. The shaded parts are the first output element as well as the input and kernel array elements used in its calculation: $0\times1+1\times2+1\times3+2\times4+2\times(-1)+3\times(-3)=2$.](../img/conv1d-channel.svg)
-->

![*dịch mô tả phía trên*](../img/conv1d-channel.svg)
:label:`fig_conv1d_channel`


<!--
Now, we reproduce the results of the one-dimensional cross-correlation operation with multi-input channel in :numref:`fig_conv1d_channel`.
-->

*dịch đoạn phía trên*


```{.python .input  n=4}
def corr1d_multi_in(X, K):
    # First, we traverse along the 0th dimension (channel dimension) of `X`
    # and `K`. Then, we add them together by using * to turn the result list
    # into a positional argument of the `add_n` function
    return sum(corr1d(x, k) for x, k in zip(X, K))

X = np.array([[0, 1, 2, 3, 4, 5, 6],
              [1, 2, 3, 4, 5, 6, 7],
              [2, 3, 4, 5, 6, 7, 8]])
K = np.array([[1, 2], [3, 4], [-1, -3]])
corr1d_multi_in(X, K)
```


<!--
The definition of a two-dimensional cross-correlation operation tells us that a one-dimensional cross-correlation operation 
with multiple input channels can be regarded as a two-dimensional cross-correlation operation with a single input channel.
As shown in :numref:`fig_conv1d_2d`, we can also present the one-dimensional cross-correlation operation with 
multiple input channels in :numref:`fig_conv1d_channel` as the equivalent two-dimensional cross-correlation operation with a single input channel.
Here, the height of the kernel is equal to the height of the input.
-->

*dịch đoạn phía trên*


<!--
![Two-dimensional cross-correlation operation with a single input channel. The highlighted parts are the first output element and the input and kernel array elements used in its calculation: $2\times(-1)+3\times(-3)+1\times3+2\times4+0\times1+1\times2=2$.](../img/conv1d-2d.svg)
-->

![*dịch mô tả phía trên*](../img/conv1d-2d.svg)
:label:`fig_conv1d_2d`


<!--
Both the outputs in :numref:`fig_conv1d` and :numref:`fig_conv1d_channel` have only one channel.
We discussed how to specify multiple output channels in a two-dimensional convolutional layer in :numref:`sec_channels`.
Similarly, we can also specify multiple output channels in the one-dimensional
convolutional layer to extend the model parameters in the convolutional layer.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Max-Over-Time Pooling Layer
-->

### *dịch tiêu đề trên*


<!--
Similarly, we have a one-dimensional pooling layer.
The max-over-time pooling layer used in TextCNN actually corresponds to a one-dimensional global maximum pooling layer.
Assuming that the input contains multiple channels, and each channel consists of values on different timesteps, 
the output of each channel will be the largest value of all timesteps in the channel.
Therefore, the input of the max-over-time pooling layer can have different timesteps on each channel.
-->

*dịch đoạn phía trên*


<!--
To improve computing performance, we often combine timing examples of different lengths into a minibatch 
and make the lengths of each timing example in the batch consistent by appending special characters (such as 0) to the end of shorter examples.
Naturally, the added special characters have no intrinsic meaning.
Because the main purpose of the max-over-time pooling layer is to capture the most important features of timing, 
it usually allows the model to be unaffected by the manually added characters.
-->

*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## The TextCNN Model
-->

## *dịch tiêu đề trên*


<!--
TextCNN mainly uses a one-dimensional convolutional layer and max-over-time pooling layer.
Suppose the input text sequence consists of $n$ words, and each word is represented by a $d$-dimension word vector.
Then the input example has a width of $n$, a height of 1, and $d$ input channels.
The calculation of textCNN can be mainly divided into the following steps:
-->

*dịch đoạn phía trên*


<!--
1. Define multiple one-dimensional convolution kernels and use them to perform convolution calculations on the inputs. 
Convolution kernels with different widths may capture the correlation of different numbers of adjacent words.
2. Perform max-over-time pooling on all output channels, and then concatenate the pooling output values of these channels in a vector.
3. The concatenated vector is transformed into the output for each category through the fully connected layer. 
A dropout layer can be used in this step to deal with overfitting.
-->

*dịch đoạn phía trên*


<!--
![TextCNN design.](../img/textcnn.svg)
-->

![*dịch mô tả phía trên*](../img/textcnn.svg)
:label:`fig_conv1d_textcnn`

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
:numref:`fig_conv1d_textcnn` gives an example to illustrate the textCNN.
The input here is a sentence with 11 words, with each word represented by a 6-dimensional word vector.
Therefore, the input sequence has a width of 11 and 6 input channels
We assume there are two one-dimensional convolution kernels with widths of 2 and 4, and 4 and 5 output channels, respectively.
Therefore, after one-dimensional convolution calculation, the width of the four output channels is $11-2+1=10$, 
while the width of the other five channels is $11-4+1=8$.
Even though the width of each channel is different, we can still perform max-over-time pooling 
for each channel and concatenate the pooling outputs of the 9 channels into a 9-dimensional vector.
Finally, we use a fully connected layer to transform the 9-dimensional vector 
into a 2-dimensional output: positive sentiment and negative sentiment predictions.
-->

*dịch đoạn phía trên*


<!--
Next, we will implement a textCNN model.
Compared with the previous section, in addition to replacing the recurrent neural network with a one-dimensional convolutional layer,
here we use two embedding layers, one with a fixed weight and another that participates in training.
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
class TextCNN(nn.Block):
    def __init__(self, vocab_size, embed_size, kernel_sizes, num_channels,
                 **kwargs):
        super(TextCNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # The embedding layer does not participate in training
        self.constant_embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Dense(2)
        # The max-over-time pooling layer has no weight, so it can share an
        # instance
        self.pool = nn.GlobalMaxPool1D()
        # Create multiple one-dimensional convolutional layers
        self.convs = nn.Sequential()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.add(nn.Conv1D(c, k, activation='relu'))

    def forward(self, inputs):
        # Concatenate the output of two embedding layers with shape of
        # (batch size, no. of words, word vector dimension) by word vector
        embeddings = np.concatenate((
            self.embedding(inputs), self.constant_embedding(inputs)), axis=2)
        # According to the input format required by Conv1D, the word vector
        # dimension, that is, the channel dimension of the one-dimensional
        # convolutional layer, is transformed into the previous dimension
        embeddings = embeddings.transpose(0, 2, 1)
        # For each one-dimensional convolutional layer, after max-over-time
        # pooling, an ndarray with the shape of (batch size, channel size, 1)
        # can be obtained. Use the flatten function to remove the last
        # dimension and then concatenate on the channel dimension
        encoding = np.concatenate([
            np.squeeze(self.pool(conv(embeddings)), axis=-1)
            for conv in self.convs], axis=1)
        # After applying the dropout method, use a fully connected layer to
        # obtain the output
        outputs = self.decoder(self.dropout(encoding))
        return outputs
```


<!--
Create a TextCNN instance. It has 3 convolutional layers with kernel widths of 3, 4, and 5, all with 100 output channels.
-->

*dịch đoạn phía trên*


```{.python .input  n=6}
embed_size, kernel_sizes, nums_channels = 100, [3, 4, 5], [100, 100, 100]
devices = d2l.try_all_gpus()
net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
net.initialize(init.Xavier(), ctx=devices)
```


<!--
### Load Pre-trained Word Vectors
-->

### *dịch tiêu đề trên*


<!--
As in the previous section, load pre-trained 100-dimensional GloVe word vectors and initialize the embedding layers `embedding` and `constant_embedding`.
Here, the former participates in training while the latter has a fixed weight.
-->

*dịch đoạn phía trên*


```{.python .input  n=7}
glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
embeds = glove_embedding[vocab.idx_to_token]
net.embedding.weight.set_data(embeds)
net.constant_embedding.weight.set_data(embeds)
net.constant_embedding.collect_params().setattr('grad_req', 'null')
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### Train and Evaluate the Model
-->

### *dịch tiêu đề trên*


<!--
Now we can train the model.
-->

*dịch đoạn phía trên*


```{.python .input  n=8}
lr, num_epochs = 0.001, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```


<!--
Below, we use the trained model to classify sentiments of two simple sentences.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
d2l.predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input  n=10}
d2l.predict_sentiment(net, vocab, 'this movie is so bad')
```

## Tóm tắt

<!--
* We can use one-dimensional convolution to process and analyze timing data.
* A one-dimensional cross-correlation operation with multiple input channels can be regarded as a two-dimensional cross-correlation operation with a single input channel.
* The input of the max-over-time pooling layer can have different numbers of timesteps on each channel.
* TextCNN mainly uses a one-dimensional convolutional layer and max-over-time pooling layer.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. Tune the hyperparameters and compare the two sentiment analysis methods, using recurrent neural networks 
and using convolutional neural networks, as regards accuracy and operational efficiency.
2. Can you further improve the accuracy of the model on the test set by using the three methods introduced in the previous section: 
tuning hyperparameters, using larger pre-trained word vectors, and using the spaCy word tokenization tool?
3. What other natural language processing tasks can you use textCNN for?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/393)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.
Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Trương Lộc Phát

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* 

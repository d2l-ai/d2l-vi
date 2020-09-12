<!--
# The Dataset for Pretraining Word Embedding
-->

# Tập dữ liệu để Tiền Huấn luyện Embedding Từ
:label:`sec_word2vec_data`


<!--
In this section, we will introduce how to preprocess a dataset with negative sampling :numref:`sec_approx_train` and load into minibatches forword2vec training.
The dataset we use is [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/LDC99T42), which is a small but commonly-used corpus.
It takes samples from Wall Street Journal articles and includes training sets, validation sets, and test sets.
-->

Trong phần này, chúng tôi sẽ giới thiệu cách tiền xử lý một tập dữ liệu với phương pháp lấy mẫu âm :numref:`sec_approx_train` và tạo các minibatch để huấn luyện word2vec.
Tập dữ liệu mà ta sẽ sử dụng là [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/LDC99T42), một kho ngữ liệu nhỏ nhưng được sử dụng phổ biến.
Tập dữ liệu này được thu thập từ các bài báo của Wall Street Journal và bao gồm các tập huấn luyện, tập kiểm định và tập kiểm tra.


<!--
First, import the packages and modules required for the experiment.
-->

Đầu tiên, ta nhập các gói và mô-đun cần thiết cho thí nghiệm.


```{.python .input  n=1}
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```



<!--
## Reading and Preprocessing the Dataset
-->

## Đọc và Tiền xử lý Dữ liệu


<!--
This dataset has already been preprocessed.
Each line of the dataset acts as a sentence.
All the words in a sentence are separated by spaces.
In the word embedding task, each word is a token.
-->

Tập dữ liệu này đã được tiền xử lý trước.
Mỗi dòng của tập dữ liệu được xem là một câu. 
Tất cả các từ trong một câu được phân cách bằng dấu cách.
Trong bài toán embedding từ, mỗi từ là một token.


```{.python .input  n=2}
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    data_dir = d2l.download_extract('ptb')
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```


<!--
Next we build a vocabulary with words appeared not greater than 10 times mapped into a "&lt;unk&gt;" token.
Note that the preprocessed PTB data also contains "&lt;unk&gt;" tokens presenting rare words.
-->

Tiếp theo ta sẽ xây dựng bộ từ vựng, trong đó các từ xuất hiện dưới 10 lần sẽ được xem như token "&lt;unk&gt;".
Lưu ý rằng tập dữ liệu PTB đã được tiền xử lý cũng chứa các token "&lt;unk&gt;" đại diện cho các từ hiếm gặp.


```{.python .input  n=3}
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}' 
```


<!--
## Subsampling
-->

## Lấy mẫu con


<!--
In text data, there are generally some words that appear at high frequencies, such "the", "a", and "in" in English.
Generally speaking, in a context window, it is better to train the word embedding model when a word (such as "chip") and 
a lower-frequency word (such as "microprocessor") appear at the same time, rather than when a word appears with a higher-frequency word (such as "the").
Therefore, when training the word embedding model, we can perform subsampling[2] on the words.
Specifically, each indexed word $w_i$ in the dataset will drop out at a certain probability.
The dropout probability is given as:
-->


Trong dữ liệu văn bản, thường có một số từ xuất hiện với tần suất cao, chẳng hạn như các từ "the", "a" và "in" trong tiếng Anh.
Nói chung, trong cửa sổ ngữ cảnh, sẽ tốt hơn nếu ta huấn luyện mô hình embedding từ khi một từ bình thường (chẳng hạn như "chip") và
một từ có tần suất thấp hơn (chẳng hạn như "microprocessor") xuất hiện cùng lúc, hơn là khi một từ bình thường xuất hiện với một từ có tần suất cao hơn (chẳng hạn như "the").
Do đó, khi huấn luyện mô hình embedding từ, ta có thể thực hiện lấy mẫu con [2] trên các từ.
Cụ thể, mỗi từ $w_i$ được gán chỉ số trong tập dữ liệu sẽ bị loại bỏ với một xác suất nhất định.
Xác suất loại bỏ được tính như sau:


$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$


<!--
Here, $f(w_i)$ is the ratio of the instances of word $w_i$ to the total number of words in the dataset, 
and the constant $t$ is a hyperparameter (set to $10^{-4}$ in this experiment).
As we can see, it is only possible to drop out the word $w_i$ in subsampling when $f(w_i) > t$.
The higher the word's frequency, the higher its dropout probability.
-->

Ở đây, $f(w_i)$ là tỷ lệ giữa số lần xuất hiện từ $w_i$ với tổng số từ trong tập dữ liệu,
và hằng số $t$ là một siêu tham số (có giá trị bằng $10^{-4}$ trong thí nghiệm này).
Như ta có thể thấy, từ $w_i$ chỉ có thể được loại bỏ trong lúc lấy mẫu con khi $f(w_i) > t$.
Tần suất của từ càng cao, xác suất loại bỏ càng lớn.

```{.python .input  n=4}
#@save
def subsampling(sentences, vocab):
    # Map low frequency words into <unk>
    sentences = [[vocab.idx_to_token[vocab[tk]] for tk in line]
                 for line in sentences]
    # Count the frequency for each word
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if to keep this token during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    # Now do the subsampling
    return [[tk for tk in line if keep(tk)] for line in sentences]

subsampled = subsampling(sentences, vocab)
```


<!--
Compare the sequence lengths before and after sampling, we can see subsampling significantly reduced the sequence length.
-->

So sánh độ dài chuỗi trước và sau khi lấy mẫu, ta có thể thấy việc lấy mẫu con làm giảm đáng kể độ dài chuỗi. 


```{.python .input  n=5}
d2l.set_figsize()
d2l.plt.hist([[len(line) for line in sentences],
              [len(line) for line in subsampled]])
d2l.plt.xlabel('# tokens per sentence')
d2l.plt.ylabel('count')
d2l.plt.legend(['origin', 'subsampled']);
```


<!--
For individual tokens, the sampling rate of the high-frequency word "the" is less than 1/20.
-->

Với các token riêng lẻ, tỉ lệ lấy mẫu của các từ có tần suất cao như từ "the" nhỏ hơn 1/20.


```{.python .input  n=6}
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([line.count(token) for line in sentences])}, '
            f'after={sum([line.count(token) for line in subsampled])}')

compare_counts('the')
```


<!--
But the low-frequency word "join" is completely preserved.
-->

Nhưng các từ có tần số thấp như từ "join" hoàn toàn được giữ nguyên. 


```{.python .input  n=7}
compare_counts('join')
```


<!--
Last, we map each token into an index to construct the corpus.
-->

Cuối cùng, ta ánh xạ từng token tới một chỉ số tương ứng để xây dựng kho ngữ liệu.


```{.python .input  n=8}
corpus = [vocab[line] for line in subsampled]
corpus[0:3]
```


<!--
## Loading the Dataset
-->

## Nạp Dữ liệu


<!--
Next we read the corpus with token indicies into data batches for training.
-->

Tiếp theo, ta đọc kho ngữ liệu với các chỉ số token thành các batch dữ liệu cho quá trình huấn luyện. 


<!--
### Extracting Central Target Words and Context Words
-->

### Trích xuất từ Đích Trung tâm và Từ Ngữ cảnh


<!--
We use words with a distance from the central target word not exceeding the context window size as the context words of the given center target word.
The following definition function extracts all the central target words and their context words.
It uniformly and randomly samples an integer to be used as the context window size between integer 1 and the `max_window_size` (maximum context window).
-->

Ta sử dụng các từ với khoảng cách tới từ đích trung tâm không quá độ dài cửa sổ ngữ cảnh để làm từ ngữ cảnh cho từ đích trung tâm đó. 
Hàm sau đây trích xuất tất cả từ đích trung tâm và các từ ngữ cảnh của chúng.
Ta chọn kích thước cửa sổ ngữ cảnh là một số nguyên từ 1 tới `max_window_size` (kích thước cửa sổ tối đa), được lấy ngẫu nhiên theo phân phối đều.


```{.python .input  n=9}
#@save
def get_centers_and_contexts(corpus, max_window_size):
    centers, contexts = [], []
    for line in corpus:
        # Each sentence needs at least 2 words to form a "central target word
        # - context word" pair
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the central target word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```


<!--
Next, we create an artificial dataset containing two sentences of 7 and 3 words, respectively.
Assume the maximum context window is 2 and print all the central target words and their context words.
-->

Kế tiếp, ta tạo một tập dữ liệu nhân tạo chứa hai câu có lần lượt 7 và 3 từ.
Hãy giả sử cửa sổ ngữ cảnh cực đại là 2 và in tất cả các từ đích trung tâm và các từ ngữ cảnh của chúng.


```{.python .input  n=10}
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```


<!--
We set the maximum context window size to 5.
The following extracts all the central target words and their context words in the dataset.
-->

Ta thiết lập cửa sổ ngữ cảnh cực đại là 5.
Đoạn mã sau trích xuất tất cả các từ đích trung tâm và các từ ngữ cảnh của chúng trong tập dữ liệu.


```{.python .input  n=11}
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {len(all_centers)}' 
```

<!--
### Negative Sampling
-->

### Lấy mẫu Âm

<!--
We use negative sampling for approximate training.
For a central and context word pair, we randomly sample $K$ noise words ($K=5$ in the experiment).
According to the suggestion in the Word2vec paper, the noise word sampling probability $P(w)$ is the ratio of 
the word frequency of $w$ to the total word frequency raised to the power of 0.75 [2].
-->

Ta thực hiện lấy mẫu âm để huấn luyện gần đúng.
Với mỗi cặp từ đích trung tâm và ngữ cảnh, ta lẫy mẫu ngẫu nhiên $K$ từ nhiễu ($K=5$ trong thử nghiệm này).
Theo đề xuất trong bài báo Word2vec, xác suất lấy mẫu từ nhiễu $P(w)$ là tỷ lệ giữa tần suất xuất hiện của từ $w$ và tổng tần suất xuất hiện của tất cả các từ, lấy mũ 0.75 [2]. 


<!--
We first define a class to draw a candidate according to the sampling weights.
It caches a 10000 size random number bank instead of calling `random.choices` every time.
-->

Trước hết ta sẽ định nghĩa một lớp để lấy ra một ứng cử viên dựa theo các trọng số lấy mẫu.
Lớp này sẽ lưu lại 10000 số ngẫu nhiên một lần thay vì gọi `random.choices` liên tục.


```{.python .input  n=12}
#@save
class RandomGenerator:
    """Draw a random int in [0, n] according to n sampling weights."""
    def __init__(self, sampling_weights):
        self.population = list(range(len(sampling_weights)))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i-1]

generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```



```{.python .input  n=13}
#@save
def get_negatives(all_contexts, corpus, K):
    counter = d2l.count_corpus(corpus)
    sampling_weights = [counter[i]**0.75 for i in range(len(counter))]
    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # Noise words cannot be context words
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

all_negatives = get_negatives(all_contexts, corpus, 5)
```


<!--
### Reading into Batches
-->

### Đọc Dữ liệu thành Batch


<!--
We extract all central target words `all_centers`, and the context words `all_contexts` and noise words `all_negatives` of each central target word from the dataset.
We will read them in random minibatches.
-->

Chúng ta trích xuất tất cả các từ đích trung tâm `all_centers`, cũng như các từ ngữ cảnh `all_contexts` và những từ nhiễu của mỗi từ đích trung tâm trong tập dữ liệu,
rồi đọc chúng thành các minibatch ngẫu nhiên.


<!--
In a minibatch of data, the $i^\mathrm{th}$ example includes a central word and its corresponding $n_i$ context words and $m_i$ noise words.
Since the context window size of each example may be different, the sum of context words and noise words, $n_i+m_i$, will be different.
When constructing a minibatch, we concatenate the context words and noise words of each example, 
and add 0s for padding until the length of the concatenations are the same, that is, the length of all concatenations is $\max_i n_i+m_i$(`max_len`).
In order to avoid the effect of padding on the loss function calculation, we construct the mask variable `masks`, 
each element of which corresponds to an element in the concatenation of context and noise words, `contexts_negatives`.
When an element in the variable `contexts_negatives` is a padding, the element in the mask variable `masks` at the same position will be 0.
Otherwise, it takes the value 1.
In order to distinguish between positive and negative examples, we also need to distinguish the context words from the noise words in the `contexts_negatives` variable.
Based on the construction of the mask variable, we only need to create a label variable `labels` with the same shape 
as the `contexts_negatives` variable and set the elements corresponding to context words (positive examples) to 1, and the rest to 0.
-->

Trong một minibatch dữ liệu, mẫu thứ $i$ bao gồm một từ đích trung tâm cùng $n_i$ từ ngữ cảnh và $m_i$ từ nhiễu tương ứng với từ đích trung tâm đó.
Do kích thước cửa sổ ngữ cảnh của mỗi mẫu có thể khác nhau, nên tổng số từ ngữ cảnh và từ nhiễu, $n_i+m_i$, cũng sẽ khác nhau.
Khi tạo một minibatch, chúng ta nối (*concatenate*) các từ ngữ cảnh và các từ nhiễu của mỗi mẫu,
và đệm thêm các giá trị 0 để độ dài của các đoạn nối bằng nhau, tức bằng $\max_i n_i+m_i$ (`max_len`).
Nhằm tránh ảnh hưởng của phần đệm lên việc tính toán hàm mất mát, chúng ta tạo một biến mặt nạ `masks`,
mỗi phần tử trong đó tương ứng với một phần tử trong phần nối giữa từ ngữ cảnh và từ nhiễu, `contexts_negatives`.
Khi một phần tử trong biến `contexts_negatives` là đệm, thì phần tử trong biến mặt nạ `masks` ở vị trí đó sẽ là 0, còn lại là bằng 1.
Để phân biệt giữa các mẫu dương và âm, chúng ta cũng cần phân biệt các từ ngữ cảnh với các từ nhiễu trong biến `contexts_negatives`.
Dựa trên cấu tạo của biến mặt nạ, chúng ta chỉ cần tạo một biến nhãn `labels` có cùng kích thước
với biến `contexts_negatives` và đặt giá trị các phần tử tương ứng với các từ ngữ cảnh (mẫu dương) bằng 1 và phần còn lại bằng 0.


<!--
Next, we will implement the minibatch reading function `batchify`.
Its minibatch input `data` is a list whose length is the batch size, each element of which contains central target words `center`, context words `context`, and noise words `negative`.
The minibatch data returned by this function conforms to the format we need, for example, it includes the mask variable.
-->

Tiếp đó, chúng ta lập trình chức năng đọc minibatch `batchify`,
với đầu vào minibatch `data` là một danh sách có độ dài là kích thước batch, mỗi phần tử trong đó chứa các từ đích trung tâm `center`, các từ ngữ cảnh `context` và các từ nhiễu `negative`.
Dữ liệu trong minibatch được trả về bởi hàm này đều tuân theo định dạng chúng ta cần, bao gồm biến mặt nạ.


```{.python .input  n=14}
#@save
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (np.array(centers).reshape(-1, 1), np.array(contexts_negatives),
            np.array(masks), np.array(labels))
```


<!--
Construct two simple examples:
-->

Tạo hai ví dụ mẫu đơn giản:


```{.python .input  n=15}
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```


<!--
We use the `batchify` function just defined to specify the minibatch reading method in the `DataLoader` instance.
-->

Chúng ta dùng hàm `batchify` vừa được định nghĩa để chỉ định phương thức đọc minibatch trong thực thể `DataLoader`.


<!--
## Putting All Things Together
-->

## Kết hợp mọi thứ cùng nhau


<!--
Last, we define the `load_data_ptb` function that read the PTB dataset and return the data iterator.
-->

Cuối cùng, chúng ta định nghĩa hàm `load_data_ptb` để đọc tập dữ liệu PTB và trả về iterator dữ liệu.


```{.python .input  n=16}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled = subsampling(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(all_contexts, corpus, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      batchify_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```


<!--
Let us print the first minibatch of the data iterator.
-->

Ta hãy cùng in ra minibatch đầu tiên trong iterator dữ liệu.


```{.python .input  n=17}
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```


## Tóm tắt

<!--
* Subsampling attempts to minimize the impact of high-frequency words on the training of a word embedding model.
* We can pad examples of different lengths to create minibatches with examples of all the same length 
and use mask variables to distinguish between padding and non-padding elements, so that only non-padding elements participate in the calculation of the loss function.
-->

* Việc lấy mẫu con cố gắng giảm thiểu tác động của các từ có tần suất cao đến việc huấn luyện mô hình embedding từ.
* Ta có thể đệm để tạo ra các minibatch với các mẫu có cùng độ dài và sử dụng các biến mặt nạ để phân biệt phần tử đệm, 
vì thế chỉ có những phần tử không phải đệm mới được dùng để tính toán hàm mất mát.


## Bài tập


<!--
We use the `batchify` function to specify the minibatch reading method in the `DataLoader` instance and print the shape of each variable in the first batch read.
How should these shapes be calculated?
-->

Chúng ta sử dụng hàm `batchify` để chỉ định phương thức đọc minibatch trong thực thể `DataLoader` và in ra kích thước của từng biến trong lần đọc batch đầu tiên.
Những kích thước này được tính toán như thế nào?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/383)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Mai Hoàng Long
* Phạm Đăng Khoa
* Phạm Minh Đức
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Phạm Hồng Vinh

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 30/06/2020)*

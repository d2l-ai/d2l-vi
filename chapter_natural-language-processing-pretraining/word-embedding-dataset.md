# Các Dataset cho Pretraining Word Embeddings
:label:`sec_word2vec_data`

Bây giờ chúng ta đã biết các chi tiết kỹ thuật của các mô hình word2vec và các phương pháp đào tạo gần đúng, chúng ta hãy đi qua các triển khai của họ. Cụ thể, chúng tôi sẽ lấy mô hình skip-gram trong :numref:`sec_word2vec` và lấy mẫu âm trong :numref:`sec_approx_train` làm ví dụ. Trong phần này, chúng ta bắt đầu với tập dữ liệu để đào tạo trước mô hình nhúng từ: định dạng ban đầu của dữ liệu sẽ được chuyển thành các minibatches có thể được lặp lại trong quá trình đào tạo.

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import gluon, np
import os
import random
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
import os
import random
```

## Đọc tập dữ liệu

Tập dữ liệu mà chúng tôi sử dụng ở đây là [Penn Tree Bank (PTB)](https://catalog.ldc.upenn.edu/LDC99T42). Cơ sở này được lấy mẫu từ các bài báo của Wall Street Journal, được chia thành các bộ đào tạo, xác nhận và kiểm tra. Ở định dạng ban đầu, mỗi dòng của tệp văn bản đại diện cho một câu của các từ được phân tách bằng dấu cách. Ở đây chúng ta coi từng từ như một mã thông báo.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip',
                       '319d85e578af0cdc590547f26231e4e31cdf1e42')

#@save
def read_ptb():
    """Load the PTB dataset into a list of text lines."""
    data_dir = d2l.download_extract('ptb')
    # Read the training set.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]

sentences = read_ptb()
f'# sentences: {len(sentences)}'
```

Sau khi đọc bộ đào tạo, chúng tôi xây dựng một từ vựng cho corpus, trong đó bất kỳ từ nào xuất hiện dưới 10 lần được thay thế bằng <unk>mã thông báo "". Lưu ý rằng tập dữ liệu gốc cũng chứa "<unk>" token đại diện cho các từ hiếm (không xác định).

```{.python .input}
#@tab all
vocab = d2l.Vocab(sentences, min_freq=10)
f'vocab size: {len(vocab)}'
```

## Lấy mẫu phụ

Dữ liệu văn bản thường có các từ tần số cao như “the”, “a”, và “in”: chúng thậm chí có thể xảy ra hàng tỷ lần trong thể rất lớn. Tuy nhiên, những từ này thường đồng xuất hiện với nhiều từ khác nhau trong các cửa sổ ngữ cảnh, cung cấp ít tín hiệu hữu ích. Ví dụ, hãy xem xét từ “chip” trong một cửa sổ ngữ cảnh: trực giác sự xuất hiện của nó với một từ tần số thấp “intel” hữu ích hơn trong đào tạo hơn là đồng xuất hiện với một từ tần số cao “a”. Hơn nữa, đào tạo với một lượng lớn các từ (tần số cao) là chậm. Do đó, khi đào tạo từ nhúng mô hình, các từ tần số cao có thể là * mẫu con* :cite:`Mikolov.Sutskever.Chen.ea.2013`. Cụ thể, mỗi từ được lập chỉ mục $w_i$ trong tập dữ liệu sẽ bị loại bỏ với xác suất 

$$ P(w_i) = \max\left(1 - \sqrt{\frac{t}{f(w_i)}}, 0\right),$$

trong đó $f(w_i)$ là tỷ lệ của số từ $w_i$ với tổng số từ trong tập dữ liệu và hằng số $t$ là một siêu tham số ($10^{-4}$ trong thí nghiệm). Chúng ta có thể thấy rằng chỉ khi tần số tương đối $f(w_i) > t$, từ $w_i$ mới có thể bị loại bỏ và tần số tương đối của từ càng cao thì xác suất bị loại bỏ càng lớn.

```{.python .input}
#@tab all
#@save
def subsample(sentences, vocab):
    """Subsample high-frequency words."""
    # Exclude unknown tokens '<unk>'
    sentences = [[token for token in line if vocab[token] != vocab.unk]
                 for line in sentences]
    counter = d2l.count_corpus(sentences)
    num_tokens = sum(counter.values())

    # Return True if `token` is kept during subsampling
    def keep(token):
        return(random.uniform(0, 1) <
               math.sqrt(1e-4 / counter[token] * num_tokens))

    return ([[token for token in line if keep(token)] for line in sentences],
            counter)

subsampled, counter = subsample(sentences, vocab)
```

Đoạn mã sau vẽ biểu đồ của số lượng mã thông báo trên mỗi câu trước và sau khi lấy mẫu. Đúng như dự đoán, subsampling rút ngắn đáng kể các câu bằng cách thả các từ tần số cao, điều này sẽ dẫn đến tăng tốc đào tạo.

```{.python .input}
#@tab all
d2l.show_list_len_pair_hist(['origin', 'subsampled'], '# tokens per sentence',
                            'count', sentences, subsampled);
```

Đối với mã thông báo riêng lẻ, tỷ lệ lấy mẫu của từ tần số cao “the” nhỏ hơn 1/20.

```{.python .input}
#@tab all
def compare_counts(token):
    return (f'# of "{token}": '
            f'before={sum([l.count(token) for l in sentences])}, '
            f'after={sum([l.count(token) for l in subsampled])}')

compare_counts('the')
```

Ngược lại, các từ tần số thấp “tham gia” được giữ hoàn toàn.

```{.python .input}
#@tab all
compare_counts('join')
```

Sau khi lấy mẫu đăng ký, chúng tôi ánh xạ mã thông báo đến các chỉ số của họ cho corpus.

```{.python .input}
#@tab all
corpus = [vocab[line] for line in subsampled]
corpus[:3]
```

## Trích xuất từ trung tâm và từ ngữ cảnh

Hàm `get_centers_and_contexts` sau trích xuất tất cả các từ trung tâm và từ ngữ cảnh của chúng từ `corpus`. Nó đồng đều mẫu một số nguyên giữa 1 và `max_window_size` một cách ngẫu nhiên như kích thước cửa sổ ngữ cảnh. Đối với bất kỳ từ trung tâm nào, những từ có khoảng cách từ nó không vượt quá kích thước cửa sổ ngữ cảnh được lấy mẫu là các từ ngữ cảnh của nó.

```{.python .input}
#@tab all
#@save
def get_centers_and_contexts(corpus, max_window_size):
    """Return center words and context words in skip-gram."""
    centers, contexts = [], []
    for line in corpus:
        # To form a "center word--context word" pair, each sentence needs to
        # have at least 2 words
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at `i`
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # Exclude the center word from the context words
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])
    return centers, contexts
```

Tiếp theo, chúng ta tạo ra một tập dữ liệu nhân tạo có chứa hai câu 7 và 3 từ, tương ứng. Hãy để kích thước cửa sổ ngữ cảnh tối đa là 2 và in tất cả các từ trung tâm và các từ ngữ cảnh của chúng.

```{.python .input}
#@tab all
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)
```

Khi đào tạo trên tập dữ liệu PTB, chúng tôi đặt kích thước cửa sổ ngữ cảnh tối đa là 5. Sau đây trích xuất tất cả các từ trung tâm và các từ ngữ cảnh của chúng trong tập dữ liệu.

```{.python .input}
#@tab all
all_centers, all_contexts = get_centers_and_contexts(corpus, 5)
f'# center-context pairs: {sum([len(contexts) for contexts in all_contexts])}'
```

## Lấy mẫu âm

Chúng tôi sử dụng lấy mẫu tiêu cực cho đào tạo gần đúng. Để lấy mẫu các từ nhiễu theo một phân phối được xác định trước, chúng ta xác định lớp `RandomGenerator` sau, trong đó phân phối lấy mẫu (có thể không chuẩn hóa) được truyền qua đối số `sampling_weights`.

```{.python .input}
#@tab all
#@save
class RandomGenerator:
    """Randomly draw among {1, ..., n} according to n sampling weights."""
    def __init__(self, sampling_weights):
        # Exclude 
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0

    def draw(self):
        if self.i == len(self.candidates):
            # Cache `k` random sampling results
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]
```

Ví dụ: chúng ta có thể vẽ 10 biến ngẫu nhiên $X$ trong số các chỉ số 1, 2 và 3 với xác suất lấy mẫu $P(X=1)=2/9, P(X=2)=3/9$ và $P(X=3)=4/9$ như sau.

```{.python .input}
generator = RandomGenerator([2, 3, 4])
[generator.draw() for _ in range(10)]
```

Đối với một cặp từ trung tâm và từ ngữ cảnh, chúng tôi lấy mẫu ngẫu nhiên `K` (5 trong thí nghiệm) các từ tiếng ồn. Theo các đề xuất trong bài báo word2vec, xác suất lấy mẫu $P(w)$ của một từ tiếng ồn $w$ được đặt thành tần số tương đối của nó trong từ điển nâng lên công suất 0,75 :cite:`Mikolov.Sutskever.Chen.ea.2013`.

```{.python .input}
#@tab all
#@save
def get_negatives(all_contexts, vocab, counter, K):
    """Return noise words in negative sampling."""
    # Sampling weights for words with indices 1, 2, ... (index 0 is the
    # excluded unknown token) in the vocabulary
    sampling_weights = [counter[vocab.to_tokens(i)]**0.75
                        for i in range(1, len(vocab))]
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

all_negatives = get_negatives(all_contexts, vocab, counter, 5)
```

## Tải ví dụ đào tạo trong Minibatches
:label:`subsec_word2vec-minibatch-loading`

Sau khi tất cả các từ trung tâm cùng với các từ ngữ cảnh và các từ tiếng ồn được lấy mẫu được trích xuất, chúng sẽ được chuyển thành các ví dụ nhỏ có thể được tải lặp lại trong quá trình đào tạo. 

Trong một minibatch, ví dụ $i^\mathrm{th}$ bao gồm một từ trung tâm và $n_i$ từ ngữ cảnh của nó và $m_i$ từ nhiễu. Do kích thước cửa sổ ngữ cảnh khác nhau, $n_i+m_i$ thay đổi cho $i$ khác nhau. Do đó, đối với mỗi ví dụ, chúng tôi nối các từ ngữ cảnh và các từ nhiễu của nó trong biến `contexts_negatives` và pad số không cho đến khi độ dài nối đạt $\max_i n_i+m_i$ (`max_len`). Để loại trừ các miếng đệm trong tính toán tổn thất, chúng tôi xác định một biến mặt nạ `masks`. Có một sự tương ứng một-một giữa các phần tử trong `masks` và các phần tử trong `contexts_negatives`, trong đó số không (nếu không) trong `masks` tương ứng với các miếng đệm trong `contexts_negatives`. 

Để phân biệt giữa các ví dụ tích cực và tiêu cực, chúng ta tách các từ ngữ cảnh khỏi các từ nhiễu trong `contexts_negatives` thông qua một biến `labels`. Tương tự như `masks`, cũng có sự tương ứng một-một giữa các phần tử trong `labels` và các phần tử trong `contexts_negatives`, trong đó các phần tử (nếu không số không) trong `labels` tương ứng với các từ ngữ cảnh (ví dụ tích cực) trong `contexts_negatives`. 

Ý tưởng trên được thực hiện trong hàm `batchify` sau. Đầu vào của nó `data` là một danh sách có độ dài bằng với kích thước lô, trong đó mỗi phần tử là một ví dụ bao gồm từ trung tâm `center`, các từ ngữ cảnh của nó `context`, và các từ nhiễu của nó `negative`. Hàm này trả về một minibatch có thể được nạp để tính toán trong quá trình đào tạo, chẳng hạn như bao gồm biến mask.

```{.python .input}
#@tab all
#@save
def batchify(data):
    """Return a minibatch of examples for skip-gram with negative sampling."""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (d2l.reshape(d2l.tensor(centers), (-1, 1)), d2l.tensor(
        contexts_negatives), d2l.tensor(masks), d2l.tensor(labels))
```

Hãy để chúng tôi kiểm tra chức năng này bằng cách sử dụng một minibatch gồm hai ví dụ.

```{.python .input}
#@tab all
x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))

names = ['centers', 'contexts_negatives', 'masks', 'labels']
for name, data in zip(names, batch):
    print(name, '=', data)
```

## Đặt tất cả mọi thứ lại với nhau

Cuối cùng, chúng ta định nghĩa hàm `load_data_ptb` đọc tập dữ liệu PTB và trả về bộ lặp dữ liệu và từ vựng.

```{.python .input}
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    data_iter = gluon.data.DataLoader(
        dataset, batch_size, shuffle=True,batchify_fn=batchify,
        num_workers=d2l.get_dataloader_workers())
    return data_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    """Download the PTB dataset and then load it into memory."""
    num_workers = d2l.get_dataloader_workers()
    sentences = read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10)
    subsampled, counter = subsample(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = get_negatives(
        all_contexts, vocab, counter, num_noise_words)

    class PTBDataset(torch.utils.data.Dataset):
        def __init__(self, centers, contexts, negatives):
            assert len(centers) == len(contexts) == len(negatives)
            self.centers = centers
            self.contexts = contexts
            self.negatives = negatives

        def __getitem__(self, index):
            return (self.centers[index], self.contexts[index],
                    self.negatives[index])

        def __len__(self):
            return len(self.centers)

    dataset = PTBDataset(all_centers, all_contexts, all_negatives)

    data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True,
                                      collate_fn=batchify,
                                      num_workers=num_workers)
    return data_iter, vocab
```

Hãy để chúng tôi in minibatch đầu tiên của bộ lặp dữ liệu.

```{.python .input}
#@tab all
data_iter, vocab = load_data_ptb(512, 5, 5)
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Tóm tắt

* Các từ tần số cao có thể không hữu ích trong đào tạo. Chúng tôi có thể subsample chúng để tăng tốc trong đào tạo.
* Đối với hiệu quả tính toán, chúng tôi tải các ví dụ trong minibatches. Chúng ta có thể xác định các biến khác để phân biệt các miếng đệm từ các miếng đệm không và các ví dụ tích cực với các biến tiêu cực.

## Bài tập

1. Làm thế nào để thời gian chạy của mã trong phần này thay đổi nếu không sử dụng subsampling?
1. Các `RandomGenerator` lớp lưu trữ `k` kết quả lấy mẫu ngẫu nhiên. Đặt `k` thành các giá trị khác và xem nó ảnh hưởng đến tốc độ tải dữ liệu như thế nào.
1. Những siêu tham số khác trong mã của phần này có thể ảnh hưởng đến tốc độ tải dữ liệu?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/383)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1330)
:end_tab:

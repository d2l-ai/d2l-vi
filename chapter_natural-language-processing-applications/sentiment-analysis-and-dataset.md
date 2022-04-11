# Phân tích tình cảm và tập dữ liệu
:label:`sec_sentiment`

Với sự gia tăng của mạng xã hội trực tuyến và các nền tảng đánh giá, rất nhiều dữ liệu có ý kiến đã được ghi lại, mang tiềm năng lớn trong việc hỗ trợ các quy trình ra quyết định.
*Phân tích tâm học*
nghiên cứu tình cảm của mọi người trong văn bản được sản xuất của họ, chẳng hạn như đánh giá sản phẩm, bình luận blog và các cuộc thảo luận diễn đàn. Nó thích ứng dụng rộng rãi cho các lĩnh vực đa dạng như chính trị (ví dụ: phân tích tình cảm công cộng đối với các chính sách), tài chính (ví dụ: phân tích tình cảm của thị trường) và tiếp thị (ví dụ: nghiên cứu sản phẩm và quản lý thương hiệu). 

Vì tình cảm có thể được phân loại là phân cực hoặc thang đo rời rạc (ví dụ: dương và âm), chúng ta có thể coi phân tích tình cảm như một nhiệm vụ phân loại văn bản, chuyển đổi một chuỗi văn bản có độ dài khác nhau thành một danh mục văn bản có độ dài cố định. Trong chương này, chúng ta sẽ sử dụng [tập dữ liệu xem xét phim lớn] của Stanford (https://ai.stanford.edu/~amaas/data/sentiment/) để phân tích tình cảm. Nó bao gồm một bộ đào tạo và một bộ thử nghiệm, có chứa 25000 đánh giá phim được tải xuống từ IMDb. Trong cả hai bộ dữ liệu, có số lượng nhãn “dương” và “tiêu cực” bằng nhau, cho thấy các cực tâm lý khác nhau.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
```

##  Đọc tập dữ liệu

Đầu tiên, tải xuống và trích xuất tập dữ liệu xem xét IMDb này trong đường dẫn `../data/aclImdb`.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```

Tiếp theo, đọc các tập dữ liệu đào tạo và kiểm tra. Mỗi ví dụ là một đánh giá và nhãn của nó: 1 cho “tích cực” và 0 cho “tiêu cực”.

```{.python .input}
#@tab all
#@save
def read_imdb(data_dir, is_train):
    """Read the IMDb review dataset text sequences and labels."""
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

## Xử lý trước bộ dữ liệu

Đối xử với từng từ như một mã thông báo và lọc ra các từ xuất hiện dưới 5 lần, chúng tôi tạo ra một từ vựng ra khỏi tập dữ liệu đào tạo.

```{.python .input}
#@tab all
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])
```

Sau khi token hóa, chúng ta hãy vẽ biểu đồ của độ dài xem xét trong các mã thông báo.

```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.xlabel('# tokens per review')
d2l.plt.ylabel('count')
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```

Như chúng tôi mong đợi, các đánh giá có độ dài khác nhau. Để xử lý một minibatch các đánh giá như vậy tại mỗi thời điểm, chúng tôi đặt độ dài của mỗi bài đánh giá là 500 với cắt ngắn và đệm, tương tự như bước tiền xử lý cho bộ dữ liệu dịch máy trong :numref:`sec_machine_translation`.

```{.python .input}
#@tab all
num_steps = 500  # sequence length
train_features = d2l.tensor([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
print(train_features.shape)
```

## Tạo bộ lặp dữ liệu

Bây giờ chúng ta có thể tạo ra các bộ lặp dữ liệu. Tại mỗi lần lặp lại, một minibatch các ví dụ được trả về.

```{.python .input}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

```{.python .input}
#@tab pytorch
train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])), 64)

for X, y in train_iter:
    print('X:', X.shape, ', y:', y.shape)
    break
print('# batches:', len(train_iter))
```

## Đặt tất cả mọi thứ lại với nhau

Cuối cùng, chúng ta kết hợp các bước trên vào hàm `load_data_imdb`. Nó trả về trình lặp dữ liệu đào tạo và kiểm tra và từ vựng của tập dữ liệu xem xét IMDb.

```{.python .input}
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_imdb(batch_size, num_steps=500):
    """Return data iterators and the vocabulary of the IMDb review dataset."""
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
    test_features = torch.tensor([d2l.truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in test_tokens])
    train_iter = d2l.load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = d2l.load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Tóm tắt

* Phân tích tình cảm nghiên cứu tình cảm của mọi người trong văn bản được sản xuất của họ, được coi là một vấn đề phân loại văn bản biến đổi một chuỗi văn bản có độ dài khác nhau
into a fixed-length cố định chiều dài text văn bản category thể loại.
* Sau khi xử lý trước, chúng ta có thể tải tập dữ liệu xem xét phim lớn của Stanford (tập dữ liệu đánh giá IMDb) vào các bộ lặp dữ liệu bằng từ vựng.

## Bài tập

1. Những siêu tham số trong phần này chúng ta có thể sửa đổi để đẩy nhanh các mô hình phân tích tâm lý đào tạo?
1. Bạn có thể triển khai một hàm để tải tập dữ liệu của [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html) vào bộ lặp dữ liệu và nhãn để phân tích tình cảm không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/391)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1387)
:end_tab:

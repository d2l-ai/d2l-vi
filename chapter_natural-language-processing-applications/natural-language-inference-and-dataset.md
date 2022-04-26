# Suy luận ngôn ngữ tự nhiên và tập dữ liệu
:label:`sec_natural-language-inference-and-dataset`

Năm :numref:`sec_sentiment`, chúng tôi đã thảo luận về vấn đề phân tích tình cảm. Nhiệm vụ này nhằm phân loại một chuỗi văn bản đơn lẻ thành các danh mục được xác định trước, chẳng hạn như một tập hợp các phân cực tình cảm. Tuy nhiên, khi có nhu cầu quyết định xem một câu có thể được suy ra dạng khác hay loại bỏ sự dư thừa bằng cách xác định các câu tương đương về mặt ngữ nghĩa, biết cách phân loại một chuỗi văn bản là không đủ. Thay vào đó, chúng ta cần có khả năng lý luận trên các cặp chuỗi văn bản. 

## Suy luận ngôn ngữ tự nhiên

*Suy luận ngôn ngữ tự nhiên* nghiên cứu liệu một giả thuyết **
có thể được suy ra từ một *premise*, trong đó cả hai đều là một chuỗi văn bản. Nói cách khác, suy luận ngôn ngữ tự nhiên xác định mối quan hệ logic giữa một cặp chuỗi văn bản. Các mối quan hệ như vậy thường rơi vào ba loại: 

* *Entailment*: giả thuyết có thể được suy ra từ tiền đề.
* *Contradiction*: sự phủ định của giả thuyết có thể được suy ra từ tiền đề.
* *Trung tính*: tất cả các trường hợp khác.

Suy luận ngôn ngữ tự nhiên còn được gọi là nhiệm vụ liên kết văn bản nhận dạng. Ví dụ, cặp sau sẽ được dán nhãn là *entailment* bởi vì “thể hiện tình cảm” trong giả thuyết có thể được suy ra từ “ôm nhau” trong tiền đề. 

> Tiền đề: Hai phụ nữ đang ôm nhau. 

> Giả thuyết: Hai phụ nữ đang thể hiện tình cảm. 

Sau đây là một ví dụ về *mâu thuẫn * như “chạy ví dụ mã hóa” chỉ ra “không ngủ” chứ không phải là “ngủ”. 

> Tiền đề: Một người đàn ông đang chạy ví dụ mã hóa từ Dive into Deep Learning. 

> Giả thuyết: Người đàn ông đang ngủ. 

Ví dụ thứ ba cho thấy mối quan hệ *trung tính* vì không “nổi tiếng” hay “không nổi tiếng” đều không thể suy ra từ thực tế là “đang biểu diễn cho chúng tôi”.  

> Tiền đề: Các nhạc sĩ đang biểu diễn cho chúng tôi. 

> Giả thuyết: Các nhạc sĩ nổi tiếng. 

Suy luận ngôn ngữ tự nhiên đã là một chủ đề trung tâm để hiểu ngôn ngữ tự nhiên. Nó thích các ứng dụng rộng rãi, từ truy xuất thông tin đến trả lời câu hỏi tên miền mở. Để nghiên cứu vấn đề này, chúng ta sẽ bắt đầu bằng cách điều tra một tập dữ liệu chuẩn chuẩn suy luận ngôn ngữ tự nhiên phổ biến. 

## Bộ dữ liệu Suy luận ngôn ngữ tự nhiên Stanford (SNLI)

Stanford Natural Language Inference (SNLI) Corpus là một bộ sưu tập của hơn 500000 cặp câu tiếng Anh được dán nhãn :cite:`Bowman.Angeli.Potts.ea.2015`. Chúng tôi tải xuống và lưu trữ tập dữ liệu SNLI được trích xuất trong đường dẫn `../data/snli_1.0`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import os
import re

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

### Đọc tập dữ liệu

Tập dữ liệu SNLI ban đầu chứa thông tin phong phú hơn nhiều so với những gì chúng ta thực sự cần trong các thí nghiệm của mình. Do đó, chúng tôi định nghĩa một hàm `read_snli` để chỉ trích xuất một phần của tập dữ liệu, sau đó trả về danh sách các cơ sở, giả thuyết và nhãn của chúng.

```{.python .input}
#@tab all
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

Bây giờ chúng ta hãy in 3 cặp tiền đề và giả thuyết đầu tiên, cũng như nhãn của chúng (“0", “1", và “2" tương ứng với “entailment”, “mâu thuẫn”, và “trung lập”, tương ứng).

```{.python .input}
#@tab all
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

Bộ huấn luyện có khoảng 550000 cặp, và bộ thử nghiệm có khoảng 10000 cặp. Sau đây cho thấy ba nhãn “entailment”, “contradiction”, và “neutral” được cân bằng trong cả bộ tập huấn và bộ thử nghiệm.

```{.python .input}
#@tab all
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### Xác định một lớp để tải tập dữ liệu

Dưới đây chúng ta định nghĩa một class để tải tập dữ liệu SNLI bằng cách kế thừa từ lớp `Dataset` trong Gluon. Đối số `num_steps` trong hàm tạo lớp xác định độ dài của một chuỗi văn bản sao cho mỗi minibatch của chuỗi sẽ có hình dạng giống nhau. Nói cách khác, các token sau `num_steps` đầu tiên theo trình tự dài hơn được cắt tỉa, trong khi các token đặc biệt “<pad>” sẽ được thêm vào các chuỗi ngắn hơn cho đến khi độ dài của chúng trở thành `num_steps`. Bằng cách thực hiện hàm `__getitem__`, chúng ta có thể tùy ý truy cập tiền đề, giả thuyết và nhãn với chỉ số `idx`.

```{.python .input}
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

```{.python .input}
#@tab pytorch
#@save
class SNLIDataset(torch.utils.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = torch.tensor(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return torch.tensor([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### Đặt tất cả mọi thứ lại với nhau

Bây giờ chúng ta có thể gọi hàm `read_snli` và lớp `SNLIDataset` để tải xuống tập dữ liệu SNLI và trả về `DataLoader` phiên bản cho cả bộ đào tạo và thử nghiệm, cùng với từ vựng của bộ đào tạo. Đáng chú ý là chúng ta phải sử dụng từ vựng được xây dựng từ bộ đào tạo như của bộ thử nghiệm. Do đó, bất kỳ mã thông báo mới nào từ bộ thử nghiệm sẽ không được biết đến với mô hình được đào tạo trên bộ đào tạo.

```{.python .input}
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                             shuffle=True,
                                             num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                            shuffle=False,
                                            num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```

Ở đây chúng ta đặt kích thước lô thành 128 và độ dài chuỗi là 50 và gọi hàm `load_data_snli` để lấy các bộ lặp dữ liệu và từ vựng. Sau đó, chúng tôi in kích thước từ vựng.

```{.python .input}
#@tab all
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```

Bây giờ chúng tôi in hình dạng của minibatch đầu tiên. Trái ngược với phân tích tâm lý, chúng tôi có hai đầu vào `X[0]` và `X[1]` đại diện cho các cặp cơ sở và giả thuyết.

```{.python .input}
#@tab all
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Tóm tắt

* Suy luận ngôn ngữ tự nhiên nghiên cứu liệu một giả thuyết có thể được suy ra từ một tiền đề, trong đó cả hai đều là một chuỗi văn bản.
* Trong suy luận ngôn ngữ tự nhiên, các mối quan hệ giữa cơ sở và giả thuyết bao gồm sự đòi hỏi, mâu thuẫn, và trung lập.
* Stanford Natural Language Inference (SNLI) Corpus là một tập dữ liệu chuẩn phổ biến của suy luận ngôn ngữ tự nhiên.

## Bài tập

1. Dịch máy từ lâu đã được đánh giá dựa trên sự phù hợp bề ngoài $n$ gram giữa bản dịch đầu ra và bản dịch chân lý. Bạn có thể thiết kế một biện pháp để đánh giá kết quả dịch máy bằng cách sử dụng suy luận ngôn ngữ tự nhiên không?
1. Làm thế nào chúng ta có thể thay đổi các siêu tham số để giảm kích thước từ vựng?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/394)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1388)
:end_tab:

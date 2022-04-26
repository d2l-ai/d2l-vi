# Xử lý sơ bộ văn bản
:label:`sec_text_preprocessing`

Chúng tôi đã xem xét và đánh giá các công cụ thống kê và các thách thức dự đoán đối với dữ liệu trình tự. Dữ liệu như vậy có thể có nhiều hình thức. Cụ thể, như chúng ta sẽ tập trung vào nhiều chương của cuốn sách, văn bản là một trong những ví dụ phổ biến nhất về dữ liệu trình tự. Ví dụ, một bài viết có thể được xem đơn giản là một chuỗi các từ, hoặc thậm chí là một chuỗi các ký tự. Để tạo điều kiện cho các thí nghiệm trong tương lai của chúng tôi với dữ liệu trình tự, chúng tôi sẽ dành phần này để giải thích các bước xử lý tiền xử lý phổ biến cho văn bản. Thông thường, các bước này là: 

1. Tải văn bản dưới dạng chuỗi vào bộ nhớ.
1. Chia chuỗi thành mã thông báo (ví dụ: từ và ký tự).
1. Xây dựng một bảng từ vựng để ánh xạ các token chia thành các chỉ số số.
1. Chuyển đổi văn bản thành chuỗi các chỉ số số để chúng có thể được thao tác bởi các mô hình một cách dễ dàng.

```{.python .input}
import collections
from d2l import mxnet as d2l
import re
```

```{.python .input}
#@tab pytorch
import collections
from d2l import torch as d2l
import re
```

```{.python .input}
#@tab tensorflow
import collections
from d2l import tensorflow as d2l
import re
```

## Đọc tập dữ liệu

Để bắt đầu, chúng tôi tải văn bản từ H G Wells' [*The Time Machine*](http://www.gutenberg.org/ebooks/35). Đây là một cơ thể khá nhỏ chỉ hơn 30000 từ, nhưng với mục đích của những gì chúng tôi muốn minh họa điều này là tốt. Các bộ sưu tập tài liệu thực tế hơn chứa nhiều hàng tỷ từ. Hàm sau (** đọc tập dữ liệu vào danh sách các dòng văn bản**), trong đó mỗi dòng là một chuỗi. Để đơn giản, ở đây chúng ta bỏ qua dấu câu và viết hoa.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

def read_time_machine():  #@save
    """Load the time machine dataset into a list of text lines."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]

lines = read_time_machine()
print(f'# text lines: {len(lines)}')
print(lines[0])
print(lines[10])
```

## Mã hóa

Hàm `tokenize` sau lấy một danh sách (`lines`) làm đầu vào, trong đó mỗi phần tử là một chuỗi văn bản (ví dụ, một dòng văn bản). [**Mỗi chuỗi văn bản được chia thành danh sách các tokens**]. Một * token* là đơn vị cơ bản trong văn bản. Cuối cùng, một danh sách các danh sách token được trả về, trong đó mỗi mã thông báo là một chuỗi.

```{.python .input}
#@tab all
def tokenize(lines, token='word'):  #@save
    """Split text lines into word or character tokens."""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])
```

## Từ vựng

Loại chuỗi của mã thông báo bất tiện khi được sử dụng bởi các mô hình, lấy đầu vào số. Bây giờ chúng ta hãy [**xây dựng một từ điển, thường được gọi là *vocabulary* là tốt, để ánh xạ chuỗi mã thông báo thành các chỉ số số bắt đầu từ 0**]. Để làm như vậy, trước tiên chúng ta đếm các mã thông báo duy nhất trong tất cả các tài liệu từ bộ đào tạo, cụ thể là một * corpus*, sau đó gán một chỉ số số cho mỗi mã thông báo duy nhất theo tần số của nó. Hiếm khi xuất hiện thẻ thường được loại bỏ để giảm sự phức tạp. Bất kỳ mã thông báo nào không tồn tại trong corpus hoặc đã được gỡ bỏ đều được ánh xạ thành một mã thông báo không xác định đặc biệt “<unk>”. Chúng tôi tùy chọn thêm một danh sách các token dành riêng, chẳng hạn như “<pad>” cho padding, “<bos>” để trình bày phần đầu cho một chuỗi, và “<eos>” cho phần cuối của một chuỗi.

```{.python .input}
#@tab all
class Vocab:  #@save
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

def count_corpus(tokens):  #@save
    """Count token frequencies."""
    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
```

Chúng tôi [**xây dựng một từ vừ**] bằng cách sử dụng tập dữ liệu máy thời gian làm cơ thể. Sau đó, chúng tôi in vài mã thông báo thường xuyên đầu tiên với các chỉ số của chúng.

```{.python .input}
#@tab all
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])
```

Bây giờ chúng ta có thể (** chuyển đổi mỗi dòng văn bản thành một danh sách các chỉ số**).

```{.python .input}
#@tab all
for i in [0, 10]:
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

## Đặt tất cả mọi thứ lại với nhau

Sử dụng các hàm trên, chúng ta [** đóng gói mọi thứ vào hàm `load_corpus_time_machine`**], trả về `corpus`, danh sách các chỉ số token và `vocab`, từ vựng của cơ thể máy thời gian. Các sửa đổi chúng tôi đã làm ở đây là: (i) chúng tôi mã hóa văn bản thành các ký tự, không phải từ ngữ, để đơn giản hóa việc đào tạo trong các phần sau; (ii) `corpus` là một danh sách duy nhất, không phải là danh sách các danh sách mã thông báo, vì mỗi dòng văn bản trong tập dữ liệu máy thời gian không nhất thiết phải là một câu hoặc một đoạn văn.

```{.python .input}
#@tab all
def load_corpus_time_machine(max_tokens=-1):  #@save
    """Return token indices and the vocabulary of the time machine dataset."""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # Since each text line in the time machine dataset is not necessarily a
    # sentence or a paragraph, flatten all the text lines into a single list
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

## Tóm tắt

* Văn bản là một hình thức quan trọng của dữ liệu trình tự.
* Để xử lý trước văn bản, chúng ta thường chia văn bản thành token, xây dựng một từ vựng để ánh xạ chuỗi token thành các chỉ số số, và chuyển đổi dữ liệu văn bản thành các chỉ số token cho các mô hình thao tác.

## Bài tập

1. Token ization là một bước tiền xử lý chính. Nó thay đổi cho các ngôn ngữ khác nhau. Cố gắng tìm ba phương pháp thường được sử dụng khác để mã hóa văn bản.
1. Trong thí nghiệm của phần này, mã hóa văn bản thành các từ và thay đổi các đối số `min_freq` của trường hợp `Vocab`. Điều này ảnh hưởng đến kích thước từ vựng như thế nào?

[Discussions](https://discuss.d2l.ai/t/115)

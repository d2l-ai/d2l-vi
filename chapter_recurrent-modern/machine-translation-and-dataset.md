# Dịch máy và bộ dữ liệu
:label:`sec_machine_translation`

Chúng tôi đã sử dụng RNNs để thiết kế các mô hình ngôn ngữ, đó là chìa khóa để xử lý ngôn ngữ tự nhiên. Một điểm chuẩn hàng đầu khác là * dịch máy*, một miền vấn đề trung tâm cho các mô hình * chuyển đổi chuỗi lự* chuyển đổi chuỗi đầu vào thành chuỗi đầu ra. Đóng một vai trò quan trọng trong các ứng dụng AI hiện đại khác nhau, các mô hình chuyển tải chuỗi sẽ tạo thành trọng tâm của phần còn lại của chương này và :numref:`chap_attention`. Để kết thúc này, phần này giới thiệu vấn đề dịch máy và tập dữ liệu của nó sẽ được sử dụng sau này. 

*Dịch máy* đề cập đến
dịch tự động của một chuỗi từ ngôn ngữ này sang ngôn ngữ khác. Trên thực tế, lĩnh vực này có thể có niên đại từ những năm 1940 ngay sau khi máy tính kỹ thuật số được phát minh, đặc biệt là bằng cách xem xét việc sử dụng máy tính để nứt mã ngôn ngữ trong Thế chiến II. Trong nhiều thập kỷ, các phương pháp thống kê đã chiếm ưu thế trong lĩnh vực này :cite:`Brown.Cocke.Della-Pietra.ea.1988,Brown.Cocke.Della-Pietra.ea.1990` trước khi sự gia tăng của học tập từ đầu đến cuối bằng cách sử dụng mạng thần kinh. Cái sau thường được gọi là
*dịch máy nơ-ron*
to distinguish phân biệt itself chinh no from
*dịch máy thống thống*
liên quan đến việc phân tích thống kê trong các thành phần như mô hình dịch thuật và mô hình ngôn ngữ. 

Nhấn mạnh việc học từ đầu đến cuối, cuốn sách này sẽ tập trung vào các phương pháp dịch máy thần kinh. Khác với vấn đề mô hình ngôn ngữ của chúng tôi trong :numref:`sec_language_model` có corpus là trong một ngôn ngữ duy nhất, các tập dữ liệu dịch máy bao gồm các cặp chuỗi văn bản trong ngôn ngữ nguồn và ngôn ngữ đích, tương ứng. Do đó, thay vì sử dụng lại thói quen tiền xử lý để mô hình hóa ngôn ngữ, chúng ta cần một cách khác để xử lý trước các bộ dữ liệu dịch máy. Trong phần sau, chúng tôi chỉ ra cách tải dữ liệu được xử lý trước vào minibatches để đào tạo.

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
import os
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import os
```

## [**Tải xuống và xử lý sơ bộ dữ liệu**]

Để bắt đầu, chúng tôi tải xuống một tập dữ liệu tiếng Anh-Pháp bao gồm [bilingual sentence pairs from the Tatoeba Project](http://www.manythings.org/anki/). Mỗi dòng trong tập dữ liệu là một cặp phân cách tab của một chuỗi văn bản tiếng Anh và chuỗi văn bản tiếng Pháp được dịch. Lưu ý rằng mỗi chuỗi văn bản có thể chỉ là một câu hoặc một đoạn văn của nhiều câu. Trong vấn đề dịch máy này, nơi tiếng Anh được dịch sang tiếng Pháp, tiếng Anh là ngôn ngữ nguồn * và tiếng Pháp là ngôn ngữ mục tiêu *.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

#@save
def read_data_nmt():
    """Load the English-French dataset."""
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[:75])
```

Sau khi tải xuống tập dữ liệu, chúng tôi [** tiến hành một số bước xử lý**] cho dữ liệu văn bản thô. Ví dụ, chúng ta thay thế không gian không phá vỡ bằng dấu cách, chuyển đổi chữ hoa thành chữ thường và chèn khoảng cách giữa các từ và dấu chấm câu.

```{.python .input}
#@tab all
#@save
def preprocess_nmt(text):
    """Preprocess the English-French dataset."""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # Replace non-breaking space with space, and convert uppercase letters to
    # lowercase ones
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # Insert space between words and punctuation marks
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[:80])
```

## [**Tokenization**]

Khác với mã hóa cấp ký tự trong :numref:`sec_language_model`, đối với dịch máy, chúng tôi thích mã hóa cấp từ ở đây (các mô hình hiện đại có thể sử dụng các kỹ thuật mã hóa tiên tiến hơn). Hàm `tokenize_nmt` sau mã hóa các cặp chuỗi văn bản `num_examples` đầu tiên, trong đó mỗi token là một từ hoặc dấu chấm câu. Hàm này trả về hai danh sách các danh sách token: `source` và `target`. Cụ thể, `source[i]` là danh sách các mã thông báo từ chuỗi văn bản $i^\mathrm{th}$ trong ngôn ngữ nguồn (tiếng Anh ở đây) và `target[i]` là trong ngôn ngữ đích (tiếng Pháp ở đây).

```{.python .input}
#@tab all
#@save
def tokenize_nmt(text, num_examples=None):
    """Tokenize the English-French dataset."""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[:6], target[:6]
```

Hãy để chúng tôi [** vẽ biểu đồ của số lượng mã thông báo trên mỗi chuỗi văn bản.**] Trong tập dữ liệu tiếng Anh-Pháp đơn giản này, hầu hết các chuỗi văn bản có ít hơn 20 mã thông báo.

```{.python .input}
#@tab all
#@save
def show_list_len_pair_hist(legend, xlabel, ylabel, xlist, ylist):
    """Plot the histogram for list length pairs."""
    d2l.set_figsize()
    _, _, patches = d2l.plt.hist(
        [[len(l) for l in xlist], [len(l) for l in ylist]])
    d2l.plt.xlabel(xlabel)
    d2l.plt.ylabel(ylabel)
    for patch in patches[1].patches:
        patch.set_hatch('/')
    d2l.plt.legend(legend)

show_list_len_pair_hist(['source', 'target'], '# tokens per sequence',
                        'count', source, target);
```

## [**Từ vừ**]

Vì tập dữ liệu dịch máy bao gồm các cặp ngôn ngữ, chúng ta có thể xây dựng hai từ vựng cho cả ngôn ngữ nguồn và ngôn ngữ đích riêng biệt. Với tokenization cấp từ, kích thước từ vựng sẽ lớn hơn đáng kể so với việc sử dụng tokenization cấp ký tự. Để giảm bớt điều này, ở đây chúng tôi xử lý các token không thường xuyên xuất hiện ít hơn 2 lần so với cùng một mã thông <unk>báo chưa biết (” “). Bên cạnh đó, chúng tôi chỉ định các token đặc biệt bổ sung như cho <pad>các chuỗi padding (” “) với cùng độ dài trong minibatches và để đánh dấu đầu (” <bos>“) hoặc kết thúc (” <eos>“) của chuỗi. Các token đặc biệt như vậy thường được sử dụng trong các nhiệm vụ xử lý ngôn ngữ tự nhiên.

```{.python .input}
#@tab all
src_vocab = d2l.Vocab(source, min_freq=2,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

## Đọc tập dữ liệu
:label:`subsec_mt_data_loading`

Nhớ lại rằng trong mô hình ngôn ngữ [**mỗi chuỗi ví dụ**], hoặc là một đoạn của một câu hoặc một khoảng trên nhiều câu, (** có chiều dài cố định.**) Điều này được chỉ định bởi đối số `num_steps` (số bước thời gian hoặc mã thông báo) trong :numref:`sec_language_model`. Trong dịch máy, mỗi ví dụ là một cặp chuỗi văn bản nguồn và đích, trong đó mỗi chuỗi văn bản có thể có độ dài khác nhau. 

Đối với hiệu quả tính toán, chúng ta vẫn có thể xử lý một minibatch các chuỗi văn bản cùng một lúc bằng cách *cutcation* và *padding*. Giả sử rằng mọi chuỗi trong cùng một minibatch nên có cùng chiều dài `num_steps`. Nếu chuỗi văn bản có ít hơn `num_steps` token, chúng ta sẽ tiếp tục nối mã thông báo "<pad>" đặc biệt vào cuối cho đến khi độ dài của nó đạt đến `num_steps`. Nếu không, chúng tôi sẽ cắt ngắn chuỗi văn bản bằng cách chỉ lấy mã thông báo `num_steps` đầu tiên của nó và loại bỏ phần còn lại. Bằng cách này, mỗi chuỗi văn bản sẽ có cùng độ dài để được tải trong các minibatches có cùng hình dạng. 

Chức năng `truncate_pad` sau đây (**cắt ngắn hoặc miếng đệm chuỗi văn bản**) như mô tả trước đó.

```{.python .input}
#@tab all
#@save
def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences."""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

Bây giờ chúng ta định nghĩa một hàm để [**transform text sequences thành minibatches để đào tạo.**] Chúng tôi thêm <eos>token “” đặc biệt vào cuối mỗi chuỗi để chỉ ra kết thúc của chuỗi. Khi một mô hình dự đoán bằng cách tạo ra một token chuỗi sau token, việc tạo ra <eos>token “” có thể gợi ý rằng chuỗi đầu ra đã hoàn tất. Bên cạnh đó, chúng tôi cũng ghi lại độ dài của mỗi chuỗi văn bản không bao gồm các token padding. Thông tin này sẽ cần thiết bởi một số mô hình mà chúng tôi sẽ đề cập sau.

```{.python .input}
#@tab all
#@save
def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches."""
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = d2l.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = d2l.reduce_sum(
        d2l.astype(array != vocab['<pad>'], d2l.int32), 1)
    return array, valid_len
```

## [** Putting All Things Together**]

Cuối cùng, chúng ta định nghĩa hàm `load_data_nmt` để trả về bộ lặp dữ liệu, cùng với từ vựng cho cả ngôn ngữ nguồn và ngôn ngữ đích.

```{.python .input}
#@tab all
#@save
def load_data_nmt(batch_size, num_steps, num_examples=600):
    """Return the iterator and the vocabularies of the translation dataset."""
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab
```

Hãy để chúng tôi [** đọc minibatch đầu tiên từ dữ liệu tiếng Anh-Pháp. **]

```{.python .input}
#@tab all
train_iter, src_vocab, tgt_vocab = load_data_nmt(batch_size=2, num_steps=8)
for X, X_valid_len, Y, Y_valid_len in train_iter:
    print('X:', d2l.astype(X, d2l.int32))
    print('valid lengths for X:', X_valid_len)
    print('Y:', d2l.astype(Y, d2l.int32))
    print('valid lengths for Y:', Y_valid_len)
    break
```

## Tóm tắt

* Dịch máy đề cập đến bản dịch tự động của một chuỗi từ ngôn ngữ này sang ngôn ngữ khác.
* Sử dụng tokenization cấp từ, kích thước từ vựng sẽ lớn hơn đáng kể so với việc sử dụng tokenization cấp ký tự. Để giảm bớt điều này, chúng ta có thể coi các token không thường xuyên như cùng một mã thông báo không xác định.
* Chúng ta có thể cắt ngắn và pad chuỗi văn bản để tất cả chúng sẽ có cùng độ dài để được tải trong minibatches.

## Bài tập

1. Hãy thử các giá trị khác nhau của đối số `num_examples` trong hàm `load_data_nmt`. Điều này ảnh hưởng đến kích thước từ vựng của ngôn ngữ nguồn và ngôn ngữ đích như thế nào?
1. Văn bản trong một số ngôn ngữ như tiếng Trung và tiếng Nhật không có chỉ số ranh giới từ (ví dụ: khoảng cách). Token ization cấp từ vẫn là một ý tưởng hay cho những trường hợp như vậy? Tại sao hoặc tại sao không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/344)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1060)
:end_tab:

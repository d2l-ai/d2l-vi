# Tập dữ liệu cho Pretraining BERT
:label:`sec_bert-dataset`

Để chuẩn bị mô hình BERT như được triển khai trong :numref:`sec_bert`, chúng ta cần tạo bộ dữ liệu ở định dạng lý tưởng để tạo điều kiện thuận lợi cho hai nhiệm vụ đào tạo trước: mô hình hóa ngôn ngữ đeo mặt nạ và dự đoán câu tiếp theo. Một mặt, mô hình BERT ban đầu được đào tạo sơ bộ về việc nối hai corpora BookCorpus khổng lồ và Wikipedia tiếng Anh (xem :numref:`subsec_bert_pretraining_tasks`), khiến hầu hết độc giả của cuốn sách này khó chạy. Mặt khác, mô hình BERT được đào tạo sẵn sẵn có thể không phù hợp với các ứng dụng từ các lĩnh vực cụ thể như y học. Do đó, nó đang trở nên phổ biến để pretrain BERT trên một tập dữ liệu tùy chỉnh. Để tạo điều kiện cho việc trình diễn pretraining BERT, chúng tôi sử dụng một cơ thể nhỏ hơn WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`. 

So sánh với tập dữ liệu PTB được sử dụng để đào tạo trước word2vec trong :numref:`sec_word2vec_data`, WikiText-2 (i) giữ lại dấu câu ban đầu, làm cho nó phù hợp với dự đoán câu tiếp theo; (ii) giữ lại trường hợp và số gốc; (iii) lớn hơn hai lần.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import random

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import os
import random
import torch
```

Trong tập dữ liệu WikiText-2, mỗi dòng đại diện cho một đoạn văn mà không gian được chèn giữa bất kỳ dấu chấm câu nào và mã thông báo trước đó. Các đoạn có ít nhất hai câu được giữ lại. Để chia câu, chúng ta chỉ sử dụng dấu chấm làm dấu phân cách để đơn giản. Chúng tôi để lại các cuộc thảo luận về các kỹ thuật tách câu phức tạp hơn trong các bài tập ở cuối phần này.

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/'
    'wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')

#@save
def _read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        lines = f.readlines()
    # Uppercase letters are converted to lowercase ones
    paragraphs = [line.strip().lower().split(' . ')
                  for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs
```

## Xác định hàm trợ giúp cho các nhiệm vụ Pretraining

Sau đây, chúng ta bắt đầu bằng cách thực hiện các hàm trợ giúp cho hai nhiệm vụ pretraining BERT: dự đoán câu tiếp theo và mô hình ngôn ngữ đeo mặt nạ. Các chức năng trợ giúp này sẽ được gọi sau khi chuyển đổi cơ thể văn bản thô thành tập dữ liệu của định dạng lý tưởng để pretrain BERT. 

### Tạo nhiệm vụ dự đoán câu tiếp theo

Theo mô tả của :numref:`subsec_nsp`, hàm `_get_next_sentence` tạo ra một ví dụ đào tạo cho nhiệm vụ phân loại nhị phân.

```{.python .input}
#@tab all
#@save
def _get_next_sentence(sentence, next_sentence, paragraphs):
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next
```

Hàm sau tạo ra các ví dụ đào tạo cho dự đoán câu tiếp theo từ đầu vào `paragraph` bằng cách gọi hàm `_get_next_sentence`. Ở đây `paragraph` là danh sách các câu, trong đó mỗi câu là danh sách các token. Đối số `max_len` chỉ định độ dài tối đa của một chuỗi đầu vào BERT trong quá trình đào tạo trước.

```{.python .input}
#@tab all
#@save
def _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len):
    nsp_data_from_paragraph = []
    for i in range(len(paragraph) - 1):
        tokens_a, tokens_b, is_next = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs)
        # Consider 1 '<cls>' token and 2 '<sep>' tokens
        if len(tokens_a) + len(tokens_b) + 3 > max_len:
            continue
        tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
        nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph
```

### Tạo tác vụ mô hình hóa ngôn ngữ đeo mặt nạ
:label:`subsec_prepare_mlm_data`

Để tạo ra các ví dụ đào tạo cho tác vụ mô hình hóa ngôn ngữ được đeo mặt nạ từ một chuỗi đầu vào BERT, chúng tôi xác định hàm `_replace_mlm_tokens` sau. Trong đầu vào của nó, `tokens` là danh sách các mã thông báo đại diện cho chuỗi đầu vào BERT, `candidate_pred_positions` là danh sách các chỉ số mã thông báo của chuỗi đầu vào BERT không bao gồm các mã thông báo đặc biệt (mã thông báo đặc biệt không được dự đoán trong nhiệm vụ mô hình hóa ngôn ngữ đeo mặt nạ) và `num_mlm_preds` chỉ ra số dự đoán (thu hồi 15% mã thông báo ngẫu nhiên để dự đoán). Theo định nghĩa của nhiệm vụ mô hình hóa ngôn ngữ đeo mặt nạ trong :numref:`subsec_mlm`, tại mỗi vị trí dự đoán, đầu vào có thể được thay thế bằng một mã thông báo “<mask>” đặc biệt hoặc một mã thông báo ngẫu nhiên, hoặc vẫn không thay đổi. Cuối cùng, hàm trả về các token đầu vào sau khi có thể thay thế, các chỉ số mã thông báo nơi dự đoán diễn ra và dán nhãn cho các dự đoán này.

```{.python .input}
#@tab all
#@save
def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):
    # Make a new copy of tokens for the input of a masked language model,
    # where the input may contain replaced '<mask>' or random tokens
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    # Shuffle for getting 15% random tokens for prediction in the masked
    # language modeling task
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # 80% of the time: replace the word with the '<mask>' token
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            # 10% of the time: keep the word unchanged
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # 10% of the time: replace the word with a random word
            else:
                masked_token = random.choice(vocab.idx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```

Bằng cách gọi hàm `_replace_mlm_tokens` đã nói ở trên, hàm sau lấy chuỗi đầu vào BERT (`tokens`) làm đầu vào và trả về các chỉ số của mã thông báo đầu vào (sau khi thay thế token có thể như được mô tả trong :numref:`subsec_mlm`), các chỉ số mã thông báo nơi dự đoán diễn ra và các chỉ số nhãn cho những dự đoán.

```{.python .input}
#@tab all
#@save
def _get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # `tokens` is a list of strings
    for i, token in enumerate(tokens):
        # Special tokens are not predicted in the masked language modeling
        # task
        if token in ['<cls>', '<sep>']:
            continue
        candidate_pred_positions.append(i)
    # 15% of random tokens are predicted in the masked language modeling task
    num_mlm_preds = max(1, round(len(tokens) * 0.15))
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)
    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]
```

## Chuyển văn bản thành tập dữ liệu Pretraining

Bây giờ chúng tôi gần như đã sẵn sàng để tùy chỉnh một lớp `Dataset` cho BERT pretraining. Trước đó, chúng ta vẫn cần định nghĩa một hàm helper `_pad_bert_inputs` để thêm các <mask>token “” đặc biệt vào các đầu vào. Lập luận của nó `examples` chứa các đầu ra từ các chức năng trợ giúp `_get_nsp_data_from_paragraph` và `_get_mlm_data_from_tokens` cho hai nhiệm vụ pretraining.

```{.python .input}
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(np.array(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype='int32'))
        all_segments.append(np.array(segments + [0] * (
            max_len - len(segments)), dtype='int32'))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(np.array(len(token_ids), dtype='float32'))
        all_pred_positions.append(np.array(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype='int32'))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            np.array([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype='float32'))
        all_mlm_labels.append(np.array(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype='int32'))
        nsp_labels.append(np.array(is_next))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

```{.python .input}
#@tab pytorch
#@save
def _pad_bert_inputs(examples, max_len, vocab):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, all_segments, valid_lens,  = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments,
         is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<pad>']] * (
            max_len - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            max_len - len(segments)), dtype=torch.long))
        # `valid_lens` excludes count of '<pad>' tokens
        valid_lens.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Predictions of padded tokens will be filtered out in the loss via
        # multiplication of 0 weights
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
    return (all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels)
```

Đặt các chức năng trợ giúp để tạo ra các ví dụ đào tạo về hai nhiệm vụ đào tạo trước và chức năng trợ giúp cho đầu vào đệm lại với nhau, chúng tôi tùy chỉnh lớp `_WikiTextDataset` sau làm bộ dữ liệu WikiText-2 để đào tạo trước BERT. Bằng cách thực hiện chức năng `__getitem__ `, chúng ta có thể tùy ý truy cập các ví dụ pretraining (mô hình hóa ngôn ngữ đeo mặt nạ và dự đoán câu tiếp theo) được tạo ra từ một cặp câu từ cơ thể WikiText-2. 

Mô hình BERT ban đầu sử dụng nhúng WordPiece có kích thước từ vựng là 30000 :cite:`Wu.Schuster.Chen.ea.2016`. Phương pháp mã hóa của WordPiece là một sửa đổi nhỏ của thuật toán mã hóa cặp byte ban đầu trong :numref:`subsec_Byte_Pair_Encoding`. Để đơn giản, chúng tôi sử dụng hàm `d2l.tokenize` để mã hóa. Các mã thông báo không thường xuyên xuất hiện ít hơn năm lần được lọc ra.

```{.python .input}
#@save
class _WikiTextDataset(gluon.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
#@save
class _WikiTextDataset(torch.utils.data.Dataset):
    def __init__(self, paragraphs, max_len):
        # Input `paragraphs[i]` is a list of sentence strings representing a
        # paragraph; while output `paragraphs[i]` is a list of sentences
        # representing a paragraph, where each sentence is a list of tokens
        paragraphs = [d2l.tokenize(
            paragraph, token='word') for paragraph in paragraphs]
        sentences = [sentence for paragraph in paragraphs
                     for sentence in paragraph]
        self.vocab = d2l.Vocab(sentences, min_freq=5, reserved_tokens=[
            '<pad>', '<mask>', '<cls>', '<sep>'])
        # Get data for the next sentence prediction task
        examples = []
        for paragraph in paragraphs:
            examples.extend(_get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, max_len))
        # Get data for the masked language model task
        examples = [(_get_mlm_data_from_tokens(tokens, self.vocab)
                      + (segments, is_next))
                     for tokens, segments, is_next in examples]
        # Pad inputs
        (self.all_token_ids, self.all_segments, self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.nsp_labels) = _pad_bert_inputs(
            examples, max_len, self.vocab)

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx], self.all_pred_positions[idx],
                self.all_mlm_weights[idx], self.all_mlm_labels[idx],
                self.nsp_labels[idx])

    def __len__(self):
        return len(self.all_token_ids)
```

Bằng cách sử dụng hàm `_read_wiki` và lớp `_WikiTextDataset`, chúng tôi xác định `load_data_wiki` sau đây để tải xuống và tập dữ liệu WikiText-2 và tạo ra các ví dụ đào tạo trước từ nó.

```{.python .input}
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```

```{.python .input}
#@tab pytorch
#@save
def load_data_wiki(batch_size, max_len):
    """Load the WikiText-2 dataset."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=num_workers)
    return train_iter, train_set.vocab
```

Đặt kích thước lô thành 512 và độ dài tối đa của chuỗi đầu vào BERT là 64, chúng tôi in ra các hình dạng của một minibatch các ví dụ pretraining BERT. Lưu ý rằng trong mỗi chuỗi đầu vào BERT, $10$ ($64 \times 0.15$) vị trí được dự đoán cho nhiệm vụ mô hình hóa ngôn ngữ đeo mặt nạ.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```

Cuối cùng, chúng ta hãy nhìn vào kích thước từ vựng. Ngay cả sau khi lọc các mã thông báo không thường xuyên, nó vẫn lớn hơn hai lần so với tập dữ liệu PTB.

```{.python .input}
#@tab all
len(vocab)
```

## Tóm tắt

* So sánh với tập dữ liệu PTB, tập ngày WikiText-2 giữ lại dấu câu ban đầu, chữ hoa và số, và lớn hơn hai lần.
* Chúng ta có thể tùy ý truy cập các ví dụ pretraining (mô hình hóa ngôn ngữ đeo mặt nạ và dự đoán câu tiếp theo) được tạo ra từ một cặp câu từ WikiText-2 corpus.

## Bài tập

1. Để đơn giản, khoảng thời gian được sử dụng làm dấu phân cách duy nhất để tách câu. Hãy thử các kỹ thuật tách câu khác, chẳng hạn như spacy và NLTK. Lấy NLTK làm ví dụ. Bạn cần cài đặt NLTK trước: `pip install nltk`. Trong mã, `import nltk` đầu tiên. Sau đó, tải xuống trình mã hóa câu Punkt: `nltk.download('punkt')`. Để chia các câu như `sentences = 'Điều này thật tuyệt! Tại sao không? ' `, invoking `nltk.tokenize.sent_tokenize (câu) ` will return a list of two sentence strings: ` ['Điều này thật tuyệt! ' , 'Tại sao không? '] `.
1. Kích thước từ vựng nếu chúng ta không lọc ra bất kỳ mã thông báo không thường xuyên nào?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/389)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1496)
:end_tab:

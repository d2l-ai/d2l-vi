<!--
# The Dataset for Pretraining BERT
-->

# Tập dữ liệu để Tiền huấn luyện BERT
:label:`sec_bert-dataset`


<!--
To pretrain the BERT model as implemented in :numref:`sec_bert`, we need to generate the dataset in the ideal format to facilitate the two pretraining tasks:
masked language modeling and next sentence prediction.
On one hand, the original BERT model is pretrained on the concatenation of
two huge corpora BookCorpus and English Wikipedia (see :numref:`subsec_bert_pretraining_tasks`), making it hard to run for most readers of this book.
On the other hand, the off-the-shelf pretrained BERT model may not fit for applications from specific domains like medicine.
Thus, it is getting popular to pretrain BERT on a customized dataset.
To facilitate the demonstration of BERT pretraining, we use a smaller corpus WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`.
-->

Để tiền huấn luyện mô hình BERT như thực hiện trong :numref:`sec_bert`, ta cần sinh tập dữ liệu ở định dạng lý tưởng để thuận tiện cho hai tác vụ tiền huấn luyện: 
mô hình hóa ngôn ngữ có mặt nạ và dự đoán câu tiếp theo.
Một mặt, mô hình BERT gốc được tiền huấn luyện trên kho ngữ liệu được ghép lại từ hai kho ngữ liệu khổng lồ là BookCorpus và Wikipedia Tiếng Anh 
(xem :numref:`subsec_bert_pretraining_tasks`), khiến việc thực hành trở nên khó khăn đối với hầu hết độc giả của cuốn sách này.
Mặt khác, mô hình BERT đã được tiền huấn luyện sẵn có thể không phù hợp với các ứng dụng ở một số lĩnh vực cụ thể như ngành dược. 
Do đó, việc tiền huấn luyện BERT trên một tập dữ liệu tùy chỉnh đang ngày càng trở nên phổ biến hơn.
Để thuận tiện minh họa cho tiền huấn luyện BERT, ta sử dụng một kho ngữ liệu nhỏ hơn là WikiText-2 :cite:`Merity.Xiong.Bradbury.ea.2016`.


<!--
Comparing with the PTB dataset used for pretraining word2vec in :numref:`sec_word2vec_data`,
WikiText-2 i) retains the original punctuation, making it suitable for next sentence prediction; ii) retains the original case and numbers; iii) is over twice larger.
-->

So với tập dữ liệu PTB đã dùng để thực hiện tiền huấn luyện word2vec ở :numref:`sec_word2vec_data`,
WikiText-2 đã i) giữ lại dấu ngắt câu ban đầu, giúp nó phù hợp cho việc dự đoán câu kế tiếp; ii) giữ lại ký tự viết hoa và số; iii) và lớn hơn gấp hai lần. 


```{.python .input  n=1}
import collections
from d2l import mxnet as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, np, npx
import os
import random
import time
import zipfile

npx.set_np()
```


<!--
In the WikiText-2 dataset, each line represents a paragraph where
space is inserted between any punctuation and its preceding token.
Paragraphs with at least two sentences are retained.
To split sentences, we only use the period as the delimiter for simplicity.
We leave discussions of more complex sentence splitting techniques in the exercises at the end of this section.
-->

Trong tập dữ liệu WikiText-2, mỗi dòng biểu diễn một đoạn văn. 
Dấu cách được chèn vào giữa bất cứ dấu ngắt câu nào và token đứng trước nó.
Các đoạn văn có tối thiểu hai câu được giữ lại.
Để tách các câu, ta chỉ sử dụng dấu chấm làm dấu phân cách cho đơn giản.
Ta sẽ dành việc thảo luận về các kỹ thuật tách câu phức tạp hơn ở phần bài tập cuối mục.


```{.python .input  n=2}
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

<!--
## Defining Helper Functions for Pretraining Tasks
-->

## Định nghĩa các Hàm trợ giúp cho các Tác vụ Tiền huấn luyện


<!--
In the following, we begin by implementing helper functions for the two BERT pretraining tasks:
next sentence prediction and masked language modeling.
These helper functions will be invoked later when transforming the raw text corpus
into the dataset of the ideal format to pretrain BERT.
-->

Ở phần này, ta sẽ bắt đầu lập trình các hàm hỗ trợ cho các hai tác vụ tiền huấn luyện BERT:
dự đoán câu tiếp theo và mô hình hóa ngôn ngữ có mặt nạ. 
Các hàm hỗ trợ này sẽ được gọi khi thực hiện chuyển đổi các kho ngữ liệu văn bản thô sang tập dữ liệu có định dạng lý tưởng để tiền huấn luyện BERT.

<!--
### Generating the Next Sentence Prediction Task
-->

### Sinh tác vụ Dự đoán câu tiếp theo


<!--
According to descriptions of :numref:`subsec_nsp`,
the `_get_next_sentence` function generates a training example
for the binary classification task.
-->

Dựa theo mô tả của :numref:`subsec_nsp`,
hàm `_get_next_sentence` sinh một mẫu để huấn luyện cho tác vụ phân loại nhị phân.


```{.python .input  n=3}
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


<!--
The following function generates training examples for next sentence prediction
from the input `paragraph` by invoking the `_get_next_sentence` function.
Here `paragraph` is a list of sentences, where each sentence is a list of tokens.
The argument `max_len` specifies the maximum length of a BERT input sequence during pretraining.
-->

Hàm sau đây sinh các mẫu huấn luyện cho tác vụ dự đoán câu tiếp theo từ đầu vào `paragraph` thông qua hàm `_get_next_sentence`.
`paragraph` ở đây là một danh sách các câu mà mỗi câu là một danh sách các token.
Đối số `max_len` là chiều dài cực đại của chuỗi đầu vào BERT trong suốt quá trình tiền huấn luyện.


```{.python .input  n=4}
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


<!--
### Generating the Masked Language Modeling Task
-->

### Tạo Tác vụ Mô hình hóa Ngôn ngữ có Mặt nạ
:label:`subsec_prepare_mlm_data`


<!--
In order to generate training examples for the masked language modeling task from a BERT input sequence,
we define the following `_replace_mlm_tokens` function.
In its inputs, `tokens` is a list of tokens representing a BERT input sequence,
`candidate_pred_positions` is a list of token indices of the BERT input sequence
excluding those of special tokens (special tokens are not predicted in the masked language modeling task),
and `num_mlm_preds` indicates the number of predictions (recall 15% random tokens to predict).
Following the definition of the masked language modeling task in :numref:`subsec_mlm`,
at each prediction position, the input may be replaced by
a special “&lt;mask&gt;” token or a random token, or remain unchanged.
In the end, the function returns the input tokens after possible replacement,
the token indices where predictions take place and labels for these predictions.
-->

Để tạo dữ liệu huấn luyện cho tác vụ mô hình hóa ngôn ngữ có mặt nạ từ một chuỗi đầu vào BERT,
chúng ta cần định nghĩa hàm `_replace_mlm_tokens`.
Đầu vào của nó, `tokens` là một danh sách các token biểu diễn cho một chuỗi đầu vào BERT,
còn `candidate_pred_positions` là một danh sách chỉ số của các token của chuỗi đầu vào BERT 
ngoại trừ những token đặc biệt (token đặc biệt không được dự đoán trong tác vụ mô hình hóa ngôn ngữ có mặt nạ),
và `num_mlm_preds` chỉ định số lượng token được dự đoán (nhớ lại rằng 15% token ngẫu nhiên được dự đoán). 
Dựa trên định nghĩa của tác vụ mô hình hóa ngôn ngữ có mặt nạ trong :numref:`subsec_mlm`,
tại mỗi vị trí dự đoán, đầu vào có thể bị thay thế bởi token đặc biệt “&lt;mask&gt;” hoặc một token ngẫu nhiên, hoặc không đổi.
Cuối cùng, hàm này trả về những token đầu vào sau khi thực hiện thay thế (nếu có),
những chỉ số token được dự đoán và nhãn cho những dự đoán này.


```{.python .input  n=5}
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
                masked_token = random.randint(0, len(vocab) - 1)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels
```


<!--
By invoking the aforementioned `_replace_mlm_tokens` function, the following function takes a BERT input sequence (`tokens`) as an input and returns indices of the input tokens
(after possible token replacement as described in :numref:`subsec_mlm`), the token indices where predictions take place, and label indices for these predictions.
-->

Bằng cách gọi hàm `_replace_mlm_tokens` ở trên, hàm dưới đây nhận một chuỗi đầu vào BERT (`tokens`) làm đầu vào 
và trả về chỉ số của những token đầu vào (sau khi thay thế token (nếu có) như mô tả ở :numref:`subsec_mlm`), 
những chỉ số của token được dự đoán và chỉ số nhãn cho những dự đoán này. 


```{.python .input  n=6}
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


<!--
## Transforming Text into the Pretraining Dataset
-->

## Biến đổi Văn bản thành bộ Dữ liệu Tiền huấn luyện


<!--
Now we are almost ready to customize a `Dataset` class for pretraining BERT.
Before that,  we still need to define a helper function `_pad_bert_inputs` to append the special “&lt;mask&gt;” tokens to the inputs.
Its argument `examples` contain the outputs from the helper functions `_get_nsp_data_from_paragraph` and `_get_mlm_data_from_tokens` for the two pretraining tasks.
-->

Bây giờ chúng ta gần như đã sẵn sàng để tùy chỉnh một lớp `Dataset` cho việc tiền huấn luyện BERT.
Trước đó, chúng ta vẫn cần định nghĩa một hàm hỗ trợ `_pad_bert_inputs` để giúp nối các token “&lt;mask&gt;” đặc biệt vào đầu vào.
Đối số `examples` của hàm chứa các kết quả đầu ra từ những hàm hỗ trợ `_get_nsp_data_from_paragraph` và `_get_mlm_data_from_tokens` cho hai tác vụ tiền huấn luyện.


```{.python .input  n=7}
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


<!--
Putting the helper functions for generating training examples of the two pretraining tasks,
and the helper function for padding inputs together,
we customize the following `_WikiTextDataset` class as the WikiText-2 dataset for pretraining BERT.
By implementing the `__getitem__ `function,
we can arbitrarily access the pretraining (masked language modeling and next sentence prediction) examples 
generated from a pair of sentences from the WikiText-2 corpus.
-->

Kết hợp những hàm hỗ trợ để tạo dữ liệu huấn luyện cho hai tác vụ tiền huấn luyện và hàm hỗ trợ đệm đầu vào,
ta tùy chỉnh lớp `_WikiTextDataset` sau đây thành bộ dữ liệu WikiText-2 cho tiền huấn luyện BERT.
Bằng cách lập trình hàm `__getitem__`,
ta có thể tùy ý truy cập những mẫu dữ liệu tiền huấn luyện (mô hình hóa ngôn ngữ có mặt nạ và dự đoán câu tiếp theo)
được tạo ra từ một cặp câu trong kho ngữ liệu WikiText-2.


<!--
The original BERT model uses WordPiece embeddings whose vocabulary size is 30,000 :cite:`Wu.Schuster.Chen.ea.2016`.
The tokenization method of WordPiece is a slight modification of the original byte pair encoding algorithm in :numref:`subsec_Byte_Pair_Encoding`.
For simplicity, we use the `d2l.tokenize` function for tokenization.
Infrequent tokens that appear less than five times are filtered out.
-->

Mô hình BERT ban đầu sử dụng embedding WordPiece có kích thước bộ từ vựng là 30,000 :cite:`Wu.Schuster.Chen.ea.2016`.
Phương pháp tách token của WordPiece là một phiên bản của thuật toán mã hóa cặp byte ban đầu :numref:`subsec_Byte_Pair_Encoding` với một chút chỉnh sửa.
Để cho đơn giản, chúng tôi sử dụng hàm `d2l.tokenize` để tách từ.
Những token xuất hiện ít hơn năm lần được loại bỏ.


```{.python .input  n=8}
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


<!--
By using the `_read_wiki` function and the `_WikiTextDataset` class,
we define the following `load_data_wiki` to download and WikiText-2 dataset
and generate pretraining examples from it.
-->

Bằng cách sử dụng hàm `_read_wiki` và lớp `_WikiTextDataset`, 
ta định nghĩa hàm `load_data_wiki` dưới đây để tải xuống bộ dữ liệu WikiText-2 và tạo mẫu dữ liệu tiền huấn luyện.


```{.python .input  n=9}
#@save
def load_data_wiki(batch_size, max_len):
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('wikitext-2', 'wikitext-2')
    paragraphs = _read_wiki(data_dir)
    train_set = _WikiTextDataset(paragraphs, max_len)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    return train_iter, train_set.vocab
```


<!--
Setting the batch size to 512 and the maximum length of a BERT input sequence to be 64, we print out the shapes of a minibatch of BERT pretraining examples.
Note that in each BERT input sequence, $10$ ($64 \times 0.15$) positions are predicted for the masked language modeling task.
-->

Đặt kích thước batch là 512 và chiều dài tối đa của chuỗi đầu vào BERT là 64, ta in ra kích thước một minibatch dữ liệu tiền huấn luyện.
Lưu ý rằng trong mỗi chuỗi đầu vào BERT, $10$ ($64 \times 0.15$) vị trí được dự đoán đối với tác vụ mô hình hóa ngôn ngữ có mặt nạ.


```{.python .input  n=10}
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)

for (tokens_X, segments_X, valid_lens_x, pred_positions_X, mlm_weights_X,
     mlm_Y, nsp_y) in train_iter:
    print(tokens_X.shape, segments_X.shape, valid_lens_x.shape,
          pred_positions_X.shape, mlm_weights_X.shape, mlm_Y.shape,
          nsp_y.shape)
    break
```


<!--
In the end, let us take a look at the vocabulary size.
Even after filtering out infrequent tokens, it is still over twice larger than that of the PTB dataset.
-->

Cuối cùng, hãy nhìn vào kích thước của bộ từ vựng.
Mặc dù những token ít xuất hiện đã bị loại bỏ, kích thước của nó vẫn lớn gấp đôi bộ dữ liệu PTB.


```{.python .input  n=11}
len(vocab)
```


## Tóm tắt

<!--
* Comparing with the PTB dataset, the WikiText-2 dateset retains the original punctuation, case and numbers, and is over twice larger.
* We can arbitrarily access the pretraining (masked language modeling and next sentence prediction) examples generated from a pair of sentences from the WikiText-2 corpus.
-->

* So sánh với tập dữ liệu PTB, tập dữ liệu WikiText-2 vẫn giữ nguyên dấu câu, chữ viết hoa và ký tự số, có kích thước lớn hơn gấp đôi.
* Ta có thể tùy ý truy cập vào các mẫu tiền huấn luyện (tác vụ mô hình hoá ngôn ngữ có mặt nạ và dự đoán câu tiếp theo) được sinh ra từ một cặp câu trong kho ngữ liệu WikiText-2.

## Bài tập

<!--
1. For simplicity, the period is used as the only delimiter for splitting sentences. 
Try other sentence splitting techniques, such as the spaCy and NLTK. Take NLTK as an example. 
You need to install NLTK first: `pip install nltk`. 
In the code, first `import nltk`. Then, download the Punkt sentence tokenizer: `nltk.download('punkt')`. 
To split sentences such as `sentences = 'This is great ! Why not ?'`, 
invoking `nltk.tokenize.sent_tokenize(sentences)` will return a list of two sentence strings: `['This is great !', 'Why not ?']`.
1. What is the vocabulary size if we do not filter out any infrequent token?
-->

1. Để đơn giản, dấu chấm được dùng làm dấu phân cách duy nhất để tách các câu.
Hãy thử các kỹ thuật tách câu khác, ví dụ như công cụ spaCy và NLTK. Lấy NLTK làm ví dụ.
Bạn cần cài đặt NLTK trước: `pip install nltk`.
Trong mã nguồn, đầu tiên hãy `import nltk`. Sau đó, tải xuống bộ token hoá câu Punkt (*Punkt sentence tokenizer*): `nltk.download('punkt')`.
Để tách các câu, ví dụ `sentences = 'This is great ! Why not ?'`,
việc gọi `nltk.tokenize.sent_tokenize(sentences)` sẽ trả về một danh sách gồm hai chuỗi câu là `['This is great !', 'Why not ?']`.
1. Nếu ta không lọc ra những token ít gặp thì kích thước bộ từ vựng là bao nhiêu?


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/389)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Mai Hoàng Long
* Nguyễn Đình Nam
* Nguyễn Văn Quang
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Phạm Minh Đức

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 29/08/2020)*

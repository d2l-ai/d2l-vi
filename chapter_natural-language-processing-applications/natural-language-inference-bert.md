# Suy luận ngôn ngữ tự nhiên: Tinh chỉnh BERT
:label:`sec_natural-language-inference-bert`

Trong các phần trước của chương này, chúng tôi đã thiết kế một kiến trúc dựa trên sự chú ý (trong :numref:`sec_natural-language-inference-attention`) cho nhiệm vụ suy luận ngôn ngữ tự nhiên trên tập dữ liệu SNLI (như được mô tả trong :numref:`sec_natural-language-inference-and-dataset`). Bây giờ chúng tôi xem lại nhiệm vụ này bằng cách tinh chỉnh BERT. Như đã thảo luận trong :numref:`sec_finetuning-bert`, suy luận ngôn ngữ tự nhiên là một bài toán phân loại cặp văn bản cấp trình tự, và tinh chỉnh BERT chỉ đòi hỏi một kiến trúc dựa trên MLP bổ sung, như minh họa trong :numref:`fig_nlp-map-nli-bert`. 

![This section feeds pretrained BERT to an MLP-based architecture for natural language inference.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

Trong phần này, chúng tôi sẽ tải xuống một phiên bản nhỏ được đào tạo trước của BERT, sau đó tinh chỉnh nó để suy luận ngôn ngữ tự nhiên trên bộ dữ liệu SNLI.

```{.python .input}
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import json
import multiprocessing
import torch
from torch import nn
import os
```

## Đang tải BERT Pretrained

Chúng tôi đã giải thích cách chuẩn bị BERT trên bộ dữ liệu WikiText-2 trong :numref:`sec_bert-dataset` và :numref:`sec_bert-pretraining` (lưu ý rằng mô hình BERT ban đầu được đào tạo trước trên corpora lớn hơn nhiều). Như đã thảo luận trong :numref:`sec_bert-pretraining`, mô hình BERT ban đầu có hàng trăm triệu thông số. Trong phần sau, chúng tôi cung cấp hai phiên bản BERT được đào tạo trước: “bert.base” lớn bằng mô hình cơ sở BERT ban đầu đòi hỏi rất nhiều tài nguyên tính toán để tinh chỉnh, trong khi “bert.small” là một phiên bản nhỏ để tạo điều kiện cho trình diễn.

```{.python .input}
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```

```{.python .input}
#@tab pytorch
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.torch.zip',
                             '225d66f04cae318b841a13d32af3acc165f253ac')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.torch.zip',
                              'c72329e68a732bef0452e4b96a1c341c8910f81f')
```

Hoặc mô hình BERT được đào tạo trước chứa một tập tin “vocab.json” xác định tập từ vựng và một tập tin “pretrained.params” của các tham số được đào tạo trước. Chúng tôi thực hiện chức năng `load_pretrained_model` sau đây để tải các thông số BERT được đào tạo trước.

```{.python .input}
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, ffn_num_hiddens, num_heads, 
                         num_layers, dropout, max_len)
    # Load pretrained BERT parameters
    bert.load_parameters(os.path.join(data_dir, 'pretrained.params'),
                         ctx=devices)
    return bert, vocab
```

```{.python .input}
#@tab pytorch
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(data_dir, 'vocab.json')))
    vocab.token_to_idx = {token: idx for idx, token in enumerate(
        vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # Load pretrained BERT parameters
    bert.load_state_dict(torch.load(os.path.join(data_dir,
                                                 'pretrained.params')))
    return bert, vocab
```

Để tạo điều kiện cho trình diễn trên hầu hết các máy móc, chúng tôi sẽ tải và tinh chỉnh phiên bản nhỏ (“bert.small”) của BERT được đào tạo trước trong phần này. Trong bài tập, chúng tôi sẽ chỉ ra cách tinh chỉnh “bert.base” lớn hơn nhiều để cải thiện đáng kể độ chính xác của thử nghiệm.

```{.python .input}
#@tab all
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)
```

## Tập dữ liệu cho tinh chỉnh BERT

Đối với nhiệm vụ hạ lưu suy luận ngôn ngữ tự nhiên trên tập dữ liệu SNLI, chúng tôi xác định một lớp tập dữ liệu tùy chỉnh `SNLIBERTDataset`. Trong mỗi ví dụ, tiền đề và giả thuyết tạo thành một cặp chuỗi văn bản và được đóng gói thành một chuỗi đầu vào BERT như mô tả trong :numref:`fig_bert-two-seqs`. Nhớ lại :numref:`subsec_bert_input_rep` rằng ID phân đoạn được sử dụng để phân biệt tiền đề và giả thuyết trong một chuỗi đầu vào BERT. Với độ dài tối đa được xác định trước của chuỗi đầu vào BERT (`max_len`), token cuối cùng của cặp văn bản đầu vào dài hơn sẽ bị xóa cho đến khi `max_len` được đáp ứng. Để tăng tốc tạo bộ dữ liệu SNLI để tinh chỉnh BERT, chúng tôi sử dụng 4 quy trình công nhân để tạo ra các ví dụ đào tạo hoặc thử nghiệm song song.

```{.python .input}
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = np.array(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (np.array(all_token_ids, dtype='int32'),
                np.array(all_segments, dtype='int32'), 
                np.array(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

```{.python .input}
#@tab pytorch
class SNLIBERTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        all_premise_hypothesis_tokens = [[
            p_tokens, h_tokens] for p_tokens, h_tokens in zip(
            *[d2l.tokenize([s.lower() for s in sentences])
              for sentences in dataset[:2]])]
        
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        pool = multiprocessing.Pool(4)  # Use 4 worker processes
        out = pool.map(self._mp_worker, all_premise_hypothesis_tokens)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long), 
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens
        self._truncate_pair_of_tokens(p_tokens, h_tokens)
        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                             * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):
        # Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        # input
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()
            else:
                h_tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx]), self.labels[idx]

    def __len__(self):
        return len(self.all_token_ids)
```

Sau khi tải xuống tập dữ liệu SNLI, chúng tôi tạo ra các ví dụ đào tạo và thử nghiệm bằng cách khởi tạo lớp `SNLIBERTDataset`. Những ví dụ như vậy sẽ được đọc trong minibatches trong quá trình đào tạo và thử nghiệm suy luận ngôn ngữ tự nhiên.

```{.python .input}
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = gluon.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

```{.python .input}
#@tab pytorch
# Reduce `batch_size` if there is an out of memory error. In the original BERT
# model, `max_len` = 512
batch_size, max_len, num_workers = 512, 128, d2l.get_dataloader_workers()
data_dir = d2l.download_extract('SNLI')
train_set = SNLIBERTDataset(d2l.read_snli(data_dir, True), max_len, vocab)
test_set = SNLIBERTDataset(d2l.read_snli(data_dir, False), max_len, vocab)
train_iter = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True,
                                   num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(test_set, batch_size,
                                  num_workers=num_workers)
```

## Tinh chỉnh BERT

Như :numref:`fig_bert-two-seqs` chỉ ra, tinh chỉnh BERT cho suy luận ngôn ngữ tự nhiên chỉ yêu cầu thêm MLP bao gồm hai lớp được kết nối hoàn toàn (xem `self.hidden` và `self.output` trong lớp `BERTClassifier` sau). MLP này biến đổi đại diện BERT của mã thông <cls>báo “” đặc biệt, mã hóa thông tin của cả tiền đề và giả thuyết, thành ba đầu ra của suy luận ngôn ngữ tự nhiên: entailment, mâu thuẫn, và trung lập.

```{.python .input}
class BERTClassifier(nn.Block):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Dense(3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

```{.python .input}
#@tab pytorch
class BERTClassifier(nn.Module):
    def __init__(self, bert):
        super(BERTClassifier, self).__init__()
        self.encoder = bert.encoder
        self.hidden = bert.hidden
        self.output = nn.Linear(256, 3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_lens_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_lens_x)
        return self.output(self.hidden(encoded_X[:, 0, :]))
```

Sau đây, mô hình BERT được đào tạo trước `bert` được đưa vào phiên bản `BERTClassifier` `net` cho ứng dụng hạ lưu. Trong các triển khai phổ biến của tinh chỉnh BERT, chỉ các tham số của lớp đầu ra của MLP bổ sung (`net.output`) sẽ được học từ đầu. Tất cả các thông số của bộ mã hóa BERT được đào tạo trước (`net.encoder`) và lớp ẩn của MLP bổ sung (`net.hidden`) sẽ được tinh chỉnh.

```{.python .input}
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```

```{.python .input}
#@tab pytorch
net = BERTClassifier(bert)
```

Nhớ lại rằng trong :numref:`sec_bert` cả lớp `MaskLM` và lớp `NextSentencePred` đều có các thông số trong MLP được sử dụng của họ. Các thông số này là một phần của những thông số trong mô hình BERT được đào tạo trước `bert`, và do đó là một phần của các thông số trong `net`. Tuy nhiên, các thông số như vậy chỉ để tính toán mất mô hình hóa ngôn ngữ đeo mặt nạ và mất dự đoán câu tiếp theo trong quá trình đào tạo trước. Hai chức năng mất mát này không liên quan đến việc tinh chỉnh các ứng dụng hạ lưu, do đó các thông số của MLP được sử dụng trong `MaskLM` và `NextSentencePred` không được cập nhật (staled) khi BERT được tinh chỉnh. 

Để cho phép các tham số với gradient cũ, cờ `ignore_stale_grad=True` được đặt trong hàm `step` của `d2l.train_batch_ch13`. Chúng tôi sử dụng chức năng này để đào tạo và đánh giá mô hình `net` bằng cách sử dụng bộ đào tạo (`train_iter`) và bộ thử nghiệm (`test_iter`) của SNLI. Do các nguồn lực tính toán hạn chế, độ chính xác đào tạo và thử nghiệm có thể được cải thiện hơn nữa: chúng tôi để lại các cuộc thảo luận của nó trong các bài tập.

```{.python .input}
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

```{.python .input}
#@tab pytorch
lr, num_epochs = 1e-4, 5
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Tóm tắt

* Chúng ta có thể tinh chỉnh mô hình BERT được đào tạo trước cho các ứng dụng hạ nguồn, chẳng hạn như suy luận ngôn ngữ tự nhiên trên bộ dữ liệu SNLI.
* Trong quá trình tinh chỉnh, mô hình BERT trở thành một phần của mô hình cho ứng dụng hạ lưu. Các thông số chỉ liên quan đến mất sơ bộ sẽ không được cập nhật trong quá trình tinh chỉnh. 

## Bài tập

1. Tinh chỉnh một mô hình BERT được đào tạo trước lớn hơn nhiều như mô hình cơ sở BERT ban đầu nếu tài nguyên tính toán của bạn cho phép. Đặt các đối số trong hàm `load_pretrained_model` là: thay thế 'bert.small' bằng 'bert.base', tăng giá trị lần lượt là `num_hiddens=256`, `ffn_num_hiddens=512`, `num_heads=4` và `num_layers=2` lên 768, 3072, 12 và 12. Bằng cách tăng kỷ nguyên tinh chỉnh (và có thể điều chỉnh các siêu tham số khác), bạn có thể nhận được độ chính xác thử nghiệm cao hơn 0,86 không?
1. Làm thế nào để cắt ngắn một cặp chuỗi theo tỷ lệ chiều dài của chúng? So sánh phương pháp cắt ngắn cặp này và phương pháp được sử dụng trong lớp `SNLIBERTDataset`. Ưu và nhược điểm của họ là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/397)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1526)
:end_tab:

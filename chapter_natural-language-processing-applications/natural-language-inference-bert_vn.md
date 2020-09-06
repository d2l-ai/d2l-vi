<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Natural Language Inference: Fine-Tuning BERT
-->

# Suy diễn ngôn ngữ tự nhiên: Tinh chỉnh BERT
:label:`sec_natural-language-inference-bert`


<!--
In earlier sections of this chapter, we have designed an attention-based architecture
(in :numref:`sec_natural-language-inference-attention`) for the natural language inference task
on the SNLI dataset (as described in :numref:`sec_natural-language-inference-and-dataset`).
Now we revisit this task by fine-tuning BERT.
As discussed in :numref:`sec_finetuning-bert`,
natural language inference is a sequence-level text pair classification problem,
and fine-tuning BERT only requires an additional MLP-based architecture, as illustrated in :numref:`fig_nlp-map-nli-bert`.
-->

Ở các phần đầu của chương này, ta đã thiết kế kiến trúc dựa trên cơ chế tập trung
(trong :numref:`sec_natural-language-inference-attention`) cho tác vụ suy diễn ngôn ngữ tự nhiên 
trên tập dữ liệu SNLI (như được mô tả trong :numref:`sec_natural-language-inference-and-dataset`).
Bây giờ ta trở lại tác vụ này qua thực hiện tinh chỉnh BERT.
Như đã thảo luận trong :numref:`sec_finetuning-bert`,
suy diễn ngôn ngữ tự nhiên là bài toán phân loại cặp văn bản ở mức chuỗi,
và việc tinh chỉnh BERT chỉ đòi hỏi kiến trúc dựa trên MLP bổ trợ, như minh họa trong hình :numref:`fig_nlp-map-nli-bert`.


<!--
![This section feeds pretrained BERT to an MLP-based architecture for natural language inference.](../img/nlp-map-nli-bert.svg)
-->

![Phần này truyền BERT tiền huấn luyện sang một kiến trúc dựa trên MLP để suy diễn ngôn ngữ tự nhiên.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`


<!--
In this section, we will download a pretrained small version of BERT,
then fine-tune it for natural language inference on the SNLI dataset.
-->

Trong phần này, chúng ta sẽ tải một phiên bản nhỏ tiền huấn luyện BERT,
rồi tinh chỉnh nó để suy diễn ngôn ngữ tự nhiên dựa trên tập dữ liệu SNLI.


```{.python .input  n=1}
from d2l import mxnet as d2l
import json
import multiprocessing
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```


<!--
## Loading Pretrained BERT
-->

## Nạp tiền huấn luyện BERT


<!--
We have explained how to pretrain BERT on the WikiText-2 dataset in :numref:`sec_bert-dataset` and :numref:`sec_bert-pretraining`
(note that the original BERT model is pretrained on much bigger corpora).
As discussed in :numref:`sec_bert-pretraining`, the original BERT model has hundreds of millions of parameters.
In the following, we provide two versions of pretrained BERT:
"bert.base" is about as big as the original BERT base model that requires a lot of computational resources to fine-tune,
while "bert.small" is a small version to facilitate demonstration.
-->

Ta đã giải thích việc làm thế nào tiền huấn luyện BERT trên tập dữ liệu WikiText-2 trong :numref:`sec_bert-dataset` và :numref:`sec_bert-pretraining`
(lưu ý rằng mô hình BERT ban đầu được tiền huấn luyện trên kho ngữ liệu lớn hơn nhiều).
Như đã thảo luận trong :numref:`sec_bert-pretraining`, mô hình BERT gốc có hàng trăm triệu tham số.
Trong phần sau đây, chúng tôi cung cấp hai phiên bản BERT tiền huấn luyện:
"bert.base" có kích thước xấp xỉ mô hình cơ sở BERT gốc là mô hình đòi hỏi nhiều tài nguyên tính toán để tinh chỉnh,
trong khi "bert.small" là phiên bản nhỏ để thuận tiện cho việc biểu diễn.


```{.python .input  n=2}
d2l.DATA_HUB['bert.base'] = (d2l.DATA_URL + 'bert.base.zip',
                             '7b3820b35da691042e5d34c0971ac3edbd80d3f4')
d2l.DATA_HUB['bert.small'] = (d2l.DATA_URL + 'bert.small.zip',
                              'a4e718a47137ccd1809c9107ab4f5edd317bae2c')
```


<!--
Either pretrained BERT model contains a "vocab.json" file that defines the vocabulary set
and a "pretrained.params" file of the pretrained parameters.
We implement the following `load_pretrained_model` function to load pretrained BERT parameters.
-->

Cả hai mô hình BERT tiền huấn luyện chứa tập tin "vocab.json" là nơi định nghĩa tập từ vựng 
và tập tin "pretrained.params" chứa các tham số tiền huấn luyện.
Ta thực hiện hàm `load_pretrained_model` sau đây để nạp các tham số tiền huấn luyện BERT.


```{.python .input  n=3}
def load_pretrained_model(pretrained_model, num_hiddens, ffn_num_hiddens,
                          num_heads, num_layers, dropout, max_len, devices):
    data_dir = d2l.download_extract(pretrained_model)
    # Define an empty vocabulary to load the predefined vocabulary
    vocab = d2l.Vocab([])
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


<!--
To facilitate demonstration on most of machines,
we will load and fine-tune the small version ("bert.small") of the pretrained BERT in this section.
In the exercise, we will show how to fine-tune the much larger "bert.base" to significantly improve the testing accuracy.
-->

Để thuận tiện biểu diễ trên hầu hết các máy,
ta sẽ nạp và tinh chỉnh phiên bản nhỏ ("bert-small") của BERT tiền huấn luyện ở mục này.
Trong bài tập, ta sẽ thể hiện cách làm thế nào để tinh chỉnh "bert-base" lớn hơn nhiều để cải thiện đáng kể độ chính xác khi kiểm tra.


```{.python .input  n=4}
devices = d2l.try_all_gpus()
bert, vocab = load_pretrained_model(
    'bert.small', num_hiddens=256, ffn_num_hiddens=512, num_heads=4,
    num_layers=2, dropout=0.1, max_len=512, devices=devices)
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## The Dataset for Fine-Tuning BERT
-->

## *dịch tiêu đề trên*


<!--
For the downstream task natural language inference on the SNLI dataset, we define a customized dataset class `SNLIBERTDataset`.
In each example, the premise and hypothesis form a pair of text sequence
and is packed into one BERT input sequence as depicted in :numref:`fig_bert-two-seqs`.
Recall :numref:`subsec_bert_input_rep` that segment IDs
are used to distinguish the premise and the hypothesis in a BERT input sequence.
With the predefined maximum length of a BERT input sequence (`max_len`),
the last token of the longer of the input text pair keeps getting removed until `max_len` is met.
To accelerate generation of the SNLI dataset for fine-tuning BERT,
we use 4 worker processes to generate training or testing examples in parallel.
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
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


<!--
After downloading the SNLI dataset, we generate training and testing examples
by instantiating the `SNLIBERTDataset` class.
Such examples will be read in minibatches during training and testing
of natural language inference.
-->

*dịch đoạn phía trên*


```{.python .input  n=6}
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


<!--
## Fine-Tuning BERT
-->

## *dịch tiêu đề trên*


<!--
As :numref:`fig_bert-two-seqs` indicates, fine-tuning BERT for natural language inference
requires only an extra MLP consisting of two fully-connected layers
(see `self.hidden` and `self.output` in the following `BERTClassifier` class).
This MLP transforms the BERT representation of the special “&lt;cls&gt;” token,
which encodes the information of both the premise and the hypothesis,
into three outputs of natural language inference:
entailment, contradiction, and neutral.
-->

*dịch đoạn phía trên*


```{.python .input  n=7}
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

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
In the following, the pretrained BERT model `bert` is fed into the `BERTClassifier` instance `net` for the downstream application.
In common implementations of BERT fine-tuning, only the parameters of the output layer of the additional MLP (`net.output`) will be learned from scratch.
All the parameters of the pretrained BERT encoder (`net.encoder`) and the hidden layer of the additional MLP (net.hidden) will be fine-tuned.
-->

*dịch đoạn phía trên*


```{.python .input  n=8}
net = BERTClassifier(bert)
net.output.initialize(ctx=devices)
```


<!--
Recall that in :numref:`sec_bert` both the `MaskLM` class and the `NextSentencePred` class have parameters in their employed MLPs.
These parameters are part of those in the pretrained BERT model `bert`, and thus part of parameters in `net`.
However, such parameters are only for computing the masked language modeling loss and the next sentence prediction loss during pretraining.
These two loss functions are irrelevant to fine-tuning downstream applications, thus the parameters of the employed MLPs in 
`MaskLM` and `NextSentencePred` are not updated (staled) when BERT is fine-tuned.
-->

*dịch đoạn phía trên*


<!--
To allow parameters with stale gradients, the flag `ignore_stale_grad=True` is set in the `step` function of `d2l.train_batch_ch13`.
We use this function to train and evaluate the model `net` using the training set (`train_iter`) and the testing set (`test_iter`) of SNLI.
Due to the limited computational resources, the training and testing accuracy can be further improved: we leave its discussions in the exercises.
-->

*dịch đoạn phía trên*


```{.python .input  n=46}
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices,
               d2l.split_batch_multi_inputs)
```

## Tóm tắt

<!--
* We can fine-tune the pretrained BERT model for downstream applications, such as natural language inference on the SNLI dataset.
* During fine-tuning, the BERT model becomes part of the model for the downstream application.
Parameters that are only related to pretraining loss will not be updated during fine-tuning. 
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. Fine-tune a much larger pretrained BERT model that is about as big as the original BERT base model if your computational resource allows. 
Set arguments in the `load_pretrained_model` function as: replacing 'bert.small' with 'bert.base', 
increasing values of `num_hiddens=256`, `ffn_num_hiddens=512`, `num_heads=4`, `num_layers=2` to `768`, `3072`, `12`, `12`, respectively. 
By increasing fine-tuning epochs (and possibly tuning other hyperparameters), can you get a testing accuracy higher than 0.86?
2. How to truncate a pair of sequences according to their ratio of length? 
Compare this pair truncation method and the one used in the `SNLIBERTDataset` class. What are their pros and cons?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/397)
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
* Nguyễn Mai Hoàng Long

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

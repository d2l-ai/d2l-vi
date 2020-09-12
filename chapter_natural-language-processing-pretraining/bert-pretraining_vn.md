<!--
# Pretraining BERT
-->

# Tiền Huấn luyện BERT
:label:`sec_bert-pretraining`


<!--
With the BERT model implemented in :numref:`sec_bert` and the pretraining examples generated from the WikiText-2 dataset in :numref:`sec_bert-dataset`, 
we will pretrain BERT on the WikiText-2 dataset in this section.
-->

Trong phần này, sử dụng mô hình BERT đã được lập trình trong :numref:`sec_bert` và các mẫu dữ liệu tiền huấn luyện được tạo ra từ tập dữ liệu WikiText-2 trong :numref:`sec_bert-dataset`, 
ta sẽ tiền huấn luyện BERT trên tập dữ liệu này.


```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```


<!--
To start, we load the WikiText-2 dataset as minibatches of pretraining examples for masked language modeling and next sentence prediction.
The batch size is 512 and the maximum length of a BERT input sequence is 64.
Note that in the original BERT model, the maximum length is 512.
-->

Đầu tiên, ta nạp các mẫu dữ liệu của tập dữ liệu WikiText-2 thành các minibatch cho quá trình tiền huấn luyện hai tác vụ: mô hình hóa ngôn ngữ có mặt nạ và dự đoán câu tiếp theo.
Kích thước batch là 512 và độ dài tối đa của chuỗi đầu vào BERT là 64.
Lưu ý rằng trong mô hình BERT gốc, độ dài tối đa này là 512.


```{.python .input  n=12}
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```


<!--
## Pretraining BERT
-->

## Tiền Huấn luyện BERT


<!--
The original BERT has two versions of different model sizes :cite:`Devlin.Chang.Lee.ea.2018`.
The base model ($\text{BERT}_{\text{BASE}}$) uses 12 layers (Transformer encoder blocks) with 768 hidden units (hidden size) and 12 self-attention heads.
The large model ($\text{BERT}_{\text{LARGE}}$) uses 24 layers with 1024 hidden units and 16 self-attention heads.
Notably, the former has 110 million parameters while the latter has 340 million parameters.
For demonstration with ease, we define a small BERT, using 2 layers, 128 hidden units, and 2 self-attention heads.
-->

Mô hình BERT gốc có hai phiên bản với hai kích thước mô hình khác nhau :cite:`Devlin.Chang.Lee.ea.2018`.
Mô hình cơ bản ($\text{BERT}_{\text{BASE}}$) sử dụng 12 tầng (khối mã hóa của Transformer) với 768 nút ẩn (kích thước ẩn) và tầng tự tập trung 12 đầu.
Mô hình lớn ($\text{BERT}_{\text{LARGE}}$) sử dụng 24 tầng với 1024 nút ẩn và tầng tự tập trung 16 đầu.
Đáng chú ý là tổng số lượng tham số trong mô hình đầu tiên là 110 triệu, còn ở mô hình thứ hai là 340 triệu.
Để minh họa thì ta định nghĩa mô hình BERT nhỏ dưới đây, sử dụng 2 tầng với 128 nút ẩn và tầng tự tập trung 2 đầu.


```{.python .input  n=14}
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_layers=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```


<!--
Before defining the training loop, we define a helper function `_get_batch_loss_bert`.
Given the shard of training examples, this function computes the loss for both the masked language modeling and next sentence prediction tasks.
Note that the final loss of BERT pretraining is just the sum of both the masked language modeling loss and the next sentence prediction loss.
-->

Ta sẽ định nghĩa hàm hỗ trợ `_get_batch_loss_bert` trước khi bắt đầu lập trình vòng lặp cho quá trình huấn luyện.
Hàm này nhận đầu vào là một batch các mẫu huấn luyện và tính giá trị mất mát đối với hai tác vụ mô hình hóa ngôn ngữ có mặt nạ và dự đoán câu tiếp theo.
Lưu ý rằng mất mát cuối cùng của tác vụ tiền huấn luyện BERT chỉ là tổng mất mát của cả hai tác vụ nói trên.


```{.python .input  n=16}
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```


<!--
Invoking the two aforementioned helper functions, the following `train_bert` function defines 
the procedure to pretrain BERT (`net`) on the WikiText-2 (`train_iter`) dataset.
Training BERT can take very long.
Instead of specifying the number of epochs for training as in the `train_ch13` function (see :numref:`sec_image_augmentation`), 
the input `num_steps` of the following function specifies the number of iteration steps for training.
-->

Sử dụng hai hàm hỗ trợ được đề cập ở trên, hàm `train_bert` dưới đây sẽ định nghĩa quá trình tiền huấn luyện BERT (`net`) trên tập dữ liệu WikiText-2 (`train_iter`).
Việc huấn luyện BERT có thể mất rất nhiều thời gian.
Do đó, thay vì truyền vào số lượng epoch huấn luyện như trong hàm `train_ch13` (:numref:`sec_image_augmentation`), 
ta sử dụng tham số `num_steps` trong hàm sau để xác định số vòng lặp huấn luyện.


```{.python .input  n=17}
#@save
def train_bert(train_iter, net, loss, vocab_size, devices, log_interval,
               num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 1e-3})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            if (step + 1) % log_interval == 0:
                animator.add(step + 1,
                             (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```


<!--
We can plot both the masked language modeling loss and the next sentence prediction loss during BERT pretraining.
-->

Ta có thể vẽ đồ thị hàm mất mát ứng với hai tác vụ mô hình hóa ngôn ngữ có mặt nạ và dự đoán câu tiếp theo trong quá trình tiền huấn luyện BERT.


```{.python .input  n=18}
train_bert(train_iter, net, loss, len(vocab), devices, 1, 50)
```


<!--
## Representing Text with BERT
-->

## Biểu diễn Văn bản với BERT

<!--
After pretraining BERT, we can use it to represent single text, text pairs, or any token in them.
The following function returns the BERT (`net`) representations for all tokens in `tokens_a` and `tokens_b`.
-->

Ta có thể sử dụng mô hình BERT đã tiền huấn luyện để biểu diễn một văn bản đơn, cặp văn bản hay một token bất kỳ trong văn bản.
Hàm sau sẽ trả về biểu diễn của mô hình BERT (`net`) cho toàn bộ các token trong `tokens_a` và `tokens_b`.


```{.python .input}
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```


<!--
Consider the sentence "a crane is flying".
Recall the input representation of BERT as discussed in :numref:`subsec_bert_input_rep`.
After inserting special tokens “&lt;cls&gt;” (used for classification) and “&lt;sep&gt;” (used for separation), the BERT input sequence has a length of six.
Since zero is the index of the “&lt;cls&gt;” token, `encoded_text[:, 0, :]` is the BERT representation of the entire input sentence.
To evaluate the polysemy token "crane", we also print out the first three elements of the BERT representation of the token.
-->

Xét câu "a crane is flying".
Hãy nhớ lại biểu diễn đầu vào của BERT được thảo luận trong :numref:`subsec_bert_input_rep`,
sau khi thêm các token đặc biệt “&lt;cls&gt;” (dùng cho phân loại) và “&lt;sep&gt;” (dùng để ngăn cách), chiều dài của chuỗi đầu vào BERT là 6.
Vì 0 là chỉ số của token “&lt;cls&gt;”, `encoded_text[:, 0, :]` là biểu diễn BERT của toàn bộ câu đầu vào.
Để đánh giá token đa nghĩa "crane", ta sẽ in cả ba phần tử đầu tiên trong biểu diễn BERT của token này.


```{.python .input}
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```


<!--
Now consider a sentence pair "a crane driver came" and "he just left".
Similarly, `encoded_pair[:, 0, :]` is the encoded result of the entire sentence pair from the pretrained BERT.
Note that the first three elements of the polysemy token "crane" are different from those when the context is different.
This supports that BERT representations are context-sensitive.
-->

Bây giờ, ta sẽ xem xét cặp câu "a crane driver came" và "he just left".
Tương tự như trên, `encoded_pair[:, 0, :]` là kết quả mã hóa của cặp câu này thông qua BERT đã được tiền huấn luyện.
Lưu ý rằng khi token đa nghĩa "crane" xuất hiện trong ngữ cảnh khác nhau, ba phần tử đầu tiên trong biểu diễn BERT token này cũng thay đổi.
Điều này thể hiện rằng biểu diễn BERT có tính nhạy ngữ cảnh.


```{.python .input}
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```


<!--
In :numref:`chap_nlp_app`, we will fine-tune a pretrained BERT model
for downstream natural language processing applications.
-->

Ở :numref:`chap_nlp_app`, ta sẽ tinh chỉnh mô hình BERT đã được tiền huấn luyện với một số tác vụ xuôi dòng trong xử lý ngôn ngữ tự nhiên.


## Tóm tắt

<!--
* The original BERT has two versions, where the base model has 110 million parameters and the large model has 340 million parameters.
* After pretraining BERT, we can use it to represent single text, text pairs, or any token in them.
* In the experiment, the same token has different BERT representation when their contexts are different. This supports that BERT representations are context-sensitive.
-->

* Mô hình BERT gốc có hai phiên bản, trong đó mô hình cơ bản có 110 triệu tham số và mô hình lớn có 340 triệu tham số.
* Ta có thể sử dụng mô hình BERT đã được tiền huấn luyện để biểu diễn một văn bản đơn, cặp văn bản hay một token bất kỳ.
* Trong thí nghiệm trên, ta đã thấy rằng cùng một token có thể có nhiều cách biểu diễn khác nhau với những ngữ cảnh khác nhau.
Điều này thể hiện rằng biểu diễn BERT có tính nhạy ngữ cảnh.


## Bài tập

<!--
1. In the experiment, we can see that the masked language modeling loss is significantly higher than the next sentence prediction loss. Why?
2. Set the maximum length of a BERT input sequence to be 512 (same as the original BERT model). 
Use the configurations of the original BERT model such as $\text{BERT}_{\text{LARGE}}$. 
Do you encounter any error when running this section? Why?
-->

1. Kết quả thí nghiệm trên cho thấy mất mát ứng với tác vụ mô hình hóa ngôn ngữ có mặt nạ cao hơn đáng kể so với tác vụ dự đoán câu tiếp theo. Hãy giải thích hiện tượng này.
2. Thay đổi chiều dài tối đa của chuỗi đầu vào BERT thành 512 (giống với mô hình BERT gốc) và sử dụng cấu hình của mô hình BERT gốc như là $\text{BERT}_{\text{LARGE}}$. 
Bạn có gặp lỗi khi chạy lại thí nghiệm không? Giải thích tại sao.


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/390)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Bùi Thị Cẩm Nhung
* Nguyễn Văn Quang
* Phạm Minh Đức
* Nguyễn Văn Cường

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 21/07/2020)*

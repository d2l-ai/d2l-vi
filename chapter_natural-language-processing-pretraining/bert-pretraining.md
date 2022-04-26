# Pretraining BERT
:label:`sec_bert-pretraining`

Với mô hình BERT được triển khai trong :numref:`sec_bert` và các ví dụ đào tạo trước được tạo ra từ tập dữ liệu WikiText-2 trong :numref:`sec_bert-dataset`, chúng tôi sẽ pretrain BERT trên tập dữ liệu WikiText-2 trong phần này.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

Để bắt đầu, chúng tôi tải tập dữ liệu WikiText-2 dưới dạng các ví dụ đào tạo trước cho mô hình ngôn ngữ được đeo mặt nạ và dự đoán câu tiếp theo. Kích thước lô là 512 và độ dài tối đa của chuỗi đầu vào BERT là 64. Lưu ý rằng trong mô hình BERT ban đầu, chiều dài tối đa là 512.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## Pretraining BERT

BERT ban đầu có hai phiên bản của các kích cỡ mô hình khác nhau :cite:`Devlin.Chang.Lee.ea.2018`. Mô hình cơ sở ($\text{BERT}_{\text{BASE}}$) sử dụng 12 lớp (khối mã hóa biến áp) với 768 đơn vị ẩn (kích thước ẩn) và 12 đầu tự chú ý. Mô hình lớn ($\text{BERT}_{\text{LARGE}}$) sử dụng 24 lớp với 1024 đơn vị ẩn và 16 đầu tự chú ý. Đáng chú ý, trước đây có 110 triệu tham số trong khi sau này có 340 triệu tham số. Để trình diễn một cách dễ dàng, chúng tôi xác định một BERT nhỏ, sử dụng 2 lớp, 128 đơn vị ẩn và 2 đầu tự chú ý.

```{.python .input}
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_layers=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, norm_shape=[128],
                    ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                    num_layers=2, dropout=0.2, key_size=128, query_size=128,
                    value_size=128, hid_in_features=128, mlm_in_features=128,
                    nsp_in_features=128)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

Trước khi xác định vòng đào tạo, chúng tôi xác định một hàm trợ giúp `_get_batch_loss_bert`. Với phần nhỏ của các ví dụ đào tạo, chức năng này tính toán sự mất mát cho cả mô hình ngôn ngữ đeo mặt nạ và nhiệm vụ dự đoán câu tiếp theo. Lưu ý rằng sự mất mát cuối cùng của đào tạo trước BERT chỉ là tổng của cả sự mất mát mô hình hóa ngôn ngữ đeo mặt nạ và mất dự đoán câu tiếp theo.

```{.python .input}
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

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

Gọi hai hàm helper nói trên, hàm `train_bert` sau đây xác định quy trình chuẩn bị BERT (`net`) trên tập dữ liệu WikiText-2 (`train_iter`). Đào tạo BERT có thể mất rất nhiều thời gian. Thay vì chỉ định số epochs để đào tạo như trong hàm `train_ch13` (xem :numref:`sec_image_augmentation`), đầu vào `num_steps` của hàm sau chỉ định số bước lặp lại để đào tạo.

```{.python .input}
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
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

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
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

Chúng ta có thể vẽ cả sự mất mát mô hình hóa ngôn ngữ đeo mặt nạ và mất dự đoán câu tiếp theo trong quá trình đào tạo trước BERT.

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## Đại diện cho văn bản với BERT

Sau khi đào tạo trước BERT, chúng ta có thể sử dụng nó để đại diện cho văn bản duy nhất, cặp văn bản hoặc bất kỳ mã thông báo nào trong đó. Hàm sau trả về các đại diện BERT (`net`) cho tất cả các token trong `tokens_a` và `tokens_b`.

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

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

Hãy xem xét câu “một cần cẩu đang bay”. Nhớ lại đại diện đầu vào của BERT như đã thảo luận trong :numref:`subsec_bert_input_rep`. Sau khi chèn các token đặc biệt “<cls>” (dùng để phân loại) và “<sep>” (dùng để tách), dãy đầu vào BERT có độ dài sáu. Vì số không là chỉ số của <cls>token “”, `encoded_text[:, 0, :]` là đại diện BERT của toàn bộ câu đầu vào. Để đánh giá mã thông báo polysemy “crane”, chúng tôi cũng in ra ba yếu tố đầu tiên của đại diện BERT của mã thông báo.

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

Bây giờ hãy xem xét một cặp câu “một người lái xe cần cẩu đến” và “anh ta vừa rời đi”. Tương tự, `encoded_pair[:, 0, :]` là kết quả được mã hóa của toàn bộ cặp câu từ BERT được đào tạo trước. Lưu ý rằng ba yếu tố đầu tiên của mã thông báo polysemy “crane” khác với những yếu tố khi ngữ cảnh khác nhau. Điều này hỗ trợ rằng các đại diện BERT là nhạy cảm với ngữ cảnh.

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

Trong :numref:`chap_nlp_app`, chúng tôi sẽ tinh chỉnh một mô hình BERT được đào tạo trước cho các ứng dụng xử lý ngôn ngữ tự nhiên ở hạ nguồn. 

## Tóm tắt

* BERT ban đầu có hai phiên bản, trong đó mô hình cơ sở có 110 triệu tham số và model lớn có 340 triệu tham số.
* Sau khi đào tạo trước BERT, chúng ta có thể sử dụng nó để đại diện cho văn bản duy nhất, cặp văn bản hoặc bất kỳ mã thông báo nào trong đó.
* Trong thí nghiệm, cùng một mã thông báo có đại diện BERT khác nhau khi ngữ cảnh của chúng khác nhau. Điều này hỗ trợ rằng các đại diện BERT là nhạy cảm với ngữ cảnh.

## Bài tập

1. Trong thí nghiệm, chúng ta có thể thấy rằng mất mô hình hóa ngôn ngữ đeo mặt nạ cao hơn đáng kể so với mất dự đoán câu tiếp theo. Tại sao?
2. Đặt độ dài tối đa của chuỗi đầu vào BERT là 512 (giống như mô hình BERT ban đầu). Sử dụng các cấu hình của mô hình BERT ban đầu như $\text{BERT}_{\text{LARGE}}$. Bạn có gặp bất kỳ lỗi nào khi chạy phần này không? Tại sao?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:

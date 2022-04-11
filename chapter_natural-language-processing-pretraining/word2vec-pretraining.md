# Pretraining word2vec
:label:`sec_word2vec_pretraining`

Chúng tôi tiếp tục thực hiện mô hình skip-gram được xác định trong :numref:`sec_word2vec`. Sau đó, chúng ta sẽ pretrain word2vec bằng cách sử dụng lấy mẫu âm trên tập dữ liệu PTB. Trước hết, chúng ta hãy lấy bộ lặp dữ liệu và từ vựng cho tập dữ liệu này bằng cách gọi hàm `d2l.load_data_ptb`, được mô tả trong :numref:`sec_word2vec_data`

```{.python .input}
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## Mô hình Skip-Gram

Chúng tôi thực hiện mô hình skip-gram bằng cách sử dụng các lớp nhúng và phép nhân ma trận hàng loạt. Đầu tiên, chúng ta hãy xem lại cách nhúng lớp hoạt động. 

### Lớp nhúng

Như được mô tả trong :numref:`sec_seq2seq`, một lớp nhúng ánh xạ chỉ mục của mã thông báo vào vector tính năng của nó. Trọng lượng của lớp này là một ma trận có số hàng bằng với kích thước từ điển (`input_dim`) và số cột bằng với chiều vectơ cho mỗi mã thông báo (`output_dim`). Sau khi một mô hình nhúng từ được đào tạo, trọng lượng này là những gì chúng ta cần.

```{.python .input}
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

Đầu vào của một lớp nhúng là chỉ mục của một mã thông báo (word). Đối với bất kỳ chỉ số token $i$, biểu diễn vectơ của nó có thể được lấy từ hàng $i^\mathrm{th}$ của ma trận trọng lượng trong lớp nhúng. Vì kích thước vectơ (`output_dim`) được đặt thành 4, lớp nhúng trả về vectơ có hình dạng (2, 3, 4) cho một minibatch các chỉ số token có hình dạng (2, 3).

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Xác định tuyên truyền chuyển tiếp

Trong tuyên truyền chuyển tiếp, đầu vào của mô hình bỏ qua gram bao gồm các chỉ số từ trung tâm `center` của hình dạng (kích thước lô, 1) và ngữ cảnh nối và các chỉ số từ tiếng ồn `contexts_and_negatives` của hình dạng (kích thước lô, `max_len`), trong đó `max_len` được xác định trong :numref:`subsec_word2vec-minibatch-loading`. Hai biến này lần đầu tiên được chuyển đổi từ các chỉ số token thành vectơ thông qua lớp nhúng, sau đó phép nhân ma trận lô của chúng (được mô tả trong :numref:`subsec_batch_dot`) trả về một đầu ra của hình dạng (kích thước lô, 1, `max_len`). Mỗi phần tử trong đầu ra là tích chấm của một vector từ trung tâm và một vector ngữ cảnh hoặc từ nhiễu.

```{.python .input}
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

Chúng ta hãy in hình dạng đầu ra của chức năng `skip_gram` này cho một số đầu vào ví dụ.

```{.python .input}
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Đào tạo

Trước khi đào tạo mô hình skip-gram với lấy mẫu âm, trước tiên chúng ta hãy xác định chức năng mất mát của nó. 

### Nhị phân Cross-Entropy Mất

Theo định nghĩa của hàm mất để lấy mẫu âm trong :numref:`subsec_negative-sampling`, chúng ta sẽ sử dụng mất nhị phân cross-entropy.

```{.python .input}
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

Nhớ lại mô tả của chúng tôi về biến mặt nạ và biến nhãn trong :numref:`subsec_word2vec-minibatch-loading`. Sau đây tính toán mất nhị phân cross-entropy cho các biến đã cho.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Dưới đây cho thấy cách tính toán các kết quả trên (theo cách kém hiệu quả hơn) bằng cách sử dụng chức năng kích hoạt sigmoid trong mất nhị phân cross-entropy. Chúng ta có thể xem xét hai đầu ra là hai tổn thất bình thường được tính trung bình so với các dự đoán không đeo mặt nạ.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Khởi tạo các tham số mô hình

Chúng tôi xác định hai lớp nhúng cho tất cả các từ trong từ vựng khi chúng được sử dụng làm từ trung tâm và từ ngữ cảnh, tương ứng. Kích thước vector từ `embed_size` được đặt thành 100.

```{.python .input}
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### Xác định vòng đào tạo

Vòng đào tạo được định nghĩa dưới đây. Do sự tồn tại của đệm, việc tính toán chức năng mất mát hơi khác so với các chức năng đào tạo trước đó.

```{.python .input}
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(m):
        if type(m) == nn.Embedding:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Bây giờ chúng ta có thể đào tạo một mô hình skip-gram bằng cách sử dụng lấy mẫu âm.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Áp dụng Word Embeddings
:label:`subsec_apply-word-embed`

Sau khi đào tạo mô hình word2vec, chúng ta có thể sử dụng sự tương đồng cosin của các vectơ từ từ từ mô hình được đào tạo để tìm các từ từ điển tương tự về mặt ngữ nghĩa nhất với từ đầu vào.

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Tóm tắt

* Chúng ta có thể đào tạo một mô hình skip-gram với lấy mẫu âm bằng cách sử dụng các lớp nhúng và mất nhị phân cross-entropy.
* Các ứng dụng của nhúng từ bao gồm việc tìm các từ tương tự về mặt ngữ nghĩa cho một từ nhất định dựa trên sự tương đồng cosin của vectơ từ.

## Bài tập

1. Sử dụng mô hình được đào tạo, tìm các từ tương tự về mặt ngữ nghĩa cho các từ đầu vào khác. Bạn có thể cải thiện kết quả bằng cách điều chỉnh các siêu tham số?
1. Khi một cơ thể đào tạo là rất lớn, chúng ta thường lấy mẫu từ ngữ cảnh và các từ tiếng ồn cho các từ trung tâm trong minibatch hiện tại * khi cập nhật tham số mô hình*. Nói cách khác, cùng một từ trung tâm có thể có các từ ngữ cảnh khác nhau hoặc từ tiếng ồn trong các kỷ nguyên đào tạo khác nhau. Lợi ích của phương pháp này là gì? Cố gắng thực hiện phương pháp đào tạo này.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:

# Hệ thống khuyến cáo Sequence-Aware

Trong các phần trước, chúng tôi tóm tắt tác vụ đề xuất như một vấn đề hoàn thành ma trận mà không xem xét hành vi ngắn hạn của người dùng. Trong phần này, chúng tôi sẽ giới thiệu một mô hình khuyến nghị đưa các bản ghi tương tác người dùng theo thứ tự theo thứ tự vào tài khoản. Đây là một đề xuất nhận thức trình tự :cite:`Quadrana.Cremonesi.Jannach.2018` trong đó đầu vào là một danh sách có thứ tự và thường có dấu thời gian của hành động người dùng trong quá khứ. Một số văn học gần đây đã chứng minh tính hữu ích của việc kết hợp thông tin đó trong mô hình hóa các mô hình hành vi thời gian của người dùng và khám phá sự trôi dạt quan tâm của họ. 

Mô hình chúng tôi sẽ giới thiệu, Caser :cite:`Tang.Wang.2018`, viết tắt của mô hình đề xuất nhúng trình tự phức tạp, thông qua các mạng thần kinh phức tạp nắm bắt các ảnh hưởng mô hình động của các hoạt động gần đây của người dùng. Thành phần chính của Caser bao gồm một mạng xã hội ngang và một mạng xã hội theo chiều dọc, nhằm phát hiện ra các mẫu trình tự cấp độ và cấp điểm liên minh, tương ứng. Mẫu cấp điểm chỉ ra tác động của một mục trong trình tự lịch sử trên mục tiêu, trong khi mẫu cấp liên kết ngụ ý ảnh hưởng của một số hành động trước đó đối với mục tiêu tiếp theo. Ví dụ, mua cả sữa và bơ cùng nhau dẫn đến xác suất mua bột cao hơn là chỉ mua một trong số chúng. Hơn nữa, sở thích chung của người dùng hoặc sở thích dài hạn cũng được mô hình hóa trong các lớp kết nối hoàn toàn cuối cùng, dẫn đến mô hình hóa toàn diện hơn về sở thích người dùng. Chi tiết của mô hình được mô tả như sau. 

## Model Architectures

Trong hệ thống đề xuất nhận thức được trình tự, mỗi người dùng được liên kết với một chuỗi một số mục từ bộ mục. Hãy để $S^u = (S_1^u, ... S_{|S_u|}^u)$ biểu thị trình tự được đặt hàng. Mục tiêu của Caser là giới thiệu mặt hàng bằng cách xem xét thị hiếu chung của người dùng cũng như ý định ngắn hạn. Giả sử chúng ta xem xét các mục $L$ trước đó, một ma trận nhúng đại diện cho các tương tác trước đây cho bước thời gian $t$ có thể được xây dựng: 

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

trong đó $\mathbf{Q} \in \mathbb{R}^{n \times k}$ đại diện cho nhúng mục và $\mathbf{q}_i$ biểu thị hàng $i^\mathrm{th}$. $\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$ có thể được sử dụng để suy ra sự quan tâm thoáng qua của người dùng $u$ tại bước thời gian $t$. Chúng ta có thể xem ma trận đầu vào $\mathbf{E}^{(u, t)}$ như một hình ảnh đó là đầu vào của hai thành phần phức tạp tiếp theo. 

Lớp ghép ngang có $d$ bộ lọc ngang $\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$, và lớp ghép dọc có $d'$ bộ lọc dọc $\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$. Sau một loạt các hoạt động phức tạp và pool, chúng tôi nhận được hai đầu ra: 

$$
\mathbf{o} = \text{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \\
\mathbf{o}'= \text{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

trong đó $\mathbf{o} \in \mathbb{R}^d$ là đầu ra của mạng xã hội ngang và $\mathbf{o}' \in \mathbb{R}^{kd'}$ là đầu ra của mạng ghép dọc. Để đơn giản, chúng tôi bỏ qua các chi tiết về các hoạt động phức tạp và hồ bơi. Chúng được nối và đưa vào một lớp mạng thần kinh được kết nối hoàn toàn để có được nhiều biểu diễn cấp cao hơn. 

$$
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

trong đó $\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$ là ma trận trọng lượng và $\mathbf{b} \in \mathbb{R}^k$ là thiên vị. Vector đã học được $\mathbf{z} \in \mathbb{R}^k$ là đại diện cho ý định ngắn hạn của người dùng. 

Cuối cùng, hàm dự đoán kết hợp hương vị ngắn hạn và chung của người dùng với nhau, được định nghĩa là: 

$$
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

trong đó $\mathbf{V} \in \mathbb{R}^{n \times 2k}$ là một ma trận nhúng mục khác. $\mathbf{b}' \in \mathbb{R}^n$ là mục thiên vị cụ thể. $\mathbf{P} \in \mathbb{R}^{m \times k}$ là ma trận nhúng người dùng cho thị hiếu chung của người dùng. $\mathbf{p}_u \in \mathbb{R}^{ k}$ là hàng $u^\mathrm{th}$ của $P$ và $\mathbf{v}_i \in \mathbb{R}^{2k}$ là hàng $i^\mathrm{th}$ của $i^\mathrm{th}$. 

Mô hình có thể được học với mất BPR hoặc Bản lề. Kiến trúc của Caser được hiển thị dưới đây: 

![Illustration of the Caser Model](../img/rec-caser.svg)

Đầu tiên chúng tôi nhập các thư viện cần thiết.

```{.python .input  n=3}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## Model Implementation Code sau đây thực hiện mô hình Caser. Nó bao gồm một lớp phức tạp dọc, một lớp ghép ngang và một lớp kết nối đầy đủ.

```{.python .input  n=4}
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully-connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## Sequential Dataset with Negative Sampling Để xử lý dữ liệu tương tác tuần tự, chúng ta cần thực hiện lại class Dataset. Đoạn code sau đây tạo ra một lớp tập dữ liệu mới có tên là `SeqDataset`. Trong mỗi mẫu, nó xuất ra danh tính người dùng, $L$ trước đó của anh ấy đã tương tác các mục như một chuỗi và mục tiếp theo mà anh ta tương tác như mục tiêu. Hình dưới đây cho thấy quá trình tải dữ liệu cho một người dùng. Giả sử rằng người dùng này thích 9 bộ phim, chúng tôi sắp xếp chín bộ phim này theo thứ tự thời gian. Bộ phim mới nhất bị bỏ lại như là mục thử nghiệm. Đối với tám bộ phim còn lại, chúng ta có thể nhận được ba mẫu đào tạo, với mỗi mẫu chứa một chuỗi năm ($L=5$) phim và mục tiếp theo của nó làm mục tiêu. Các mẫu âm cũng được bao gồm trong tập dữ liệu tùy chỉnh. 

![Illustration of the data generation process](../img/rec-seq-data.svg)

```{.python .input  n=5}
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## Tải tập dữ liệu MovieLens 100K

Sau đó, chúng tôi đọc và chia bộ dữ liệu MovieLens 100K ở chế độ nhận thức được trình tự và tải dữ liệu đào tạo với dataloader tuần tự được thực hiện ở trên.

```{.python .input  n=6}
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

Cấu trúc dữ liệu đào tạo được hiển thị ở trên. Phần tử đầu tiên là danh tính người dùng, danh sách tiếp theo chỉ ra năm mục cuối cùng mà người dùng này thích và phần tử cuối cùng là mục mà người dùng này thích sau năm mục. 

## Đào tạo mô hình ngay bây giờ, chúng ta hãy đào tạo mô hình. Chúng tôi sử dụng cài đặt tương tự như NeuMF, bao gồm tốc độ học tập, trình tối ưu hóa và $k$, trong phần cuối để kết quả có thể so sánh được.

```{.python .input  n=7}
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices,
                  d2l.evaluate_ranking, candidates, eval_step=1)
```

## Tóm tắt * Suy ra lợi ích ngắn hạn và dài hạn của người dùng có thể đưa ra dự đoán về mục tiếp theo mà anh ta ưa thích hiệu quả hơn* Mạng thần kinh phức tạp có thể được sử dụng để thu hút lợi ích ngắn hạn của người dùng từ các tương tác tuần tự. 

## Bài tập

* Tiến hành một nghiên cứu cắt bỏ bằng cách loại bỏ một trong các mạng xã hội ngang và dọc, thành phần nào quan trọng hơn?
* Thay đổi siêu tham số $L$. Tương tác lịch sử lâu hơn có mang lại độ chính xác cao hơn không?
* Ngoài nhiệm vụ đề xuất nhận thức trình tự mà chúng tôi đã giới thiệu ở trên, còn có một loại tác vụ đề xuất nhận thức trình tự khác được gọi là đề xuất dựa trên phiên :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015`. Bạn có thể giải thích sự khác biệt giữa hai nhiệm vụ này không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/404)
:end_tab:

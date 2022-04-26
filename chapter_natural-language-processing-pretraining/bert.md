# Đại diện bộ mã hóa hai chiều từ Transformers (BERT)
:label:`sec_bert`

Chúng tôi đã giới thiệu một số mô hình nhúng từ để hiểu ngôn ngữ tự nhiên. Sau khi đào tạo trước, đầu ra có thể được coi là một ma trận trong đó mỗi hàng là một vectơ đại diện cho một từ của một từ vựng được xác định trước. Trên thực tế, các mô hình nhúng từ này đều là * bối cảnh độc lập*. Hãy để chúng tôi bắt đầu bằng cách minh họa tài sản này. 

## Từ bối cảnh độc lập đến bối cảnh nhạy cảm

Nhớ lại các thí nghiệm trong :numref:`sec_word2vec_pretraining` và :numref:`sec_synonyms`. Ví dụ, word2vec và Glove đều gán cùng một vector được đào tạo trước cho cùng một từ bất kể ngữ cảnh của từ (nếu có). Chính thức, một biểu diễn theo ngữ cảnh độc lập của bất kỳ mã thông báo $x$ là một hàm $f(x)$ chỉ mất $x$ làm đầu vào của nó. Với sự phong phú của ngữ nghĩa polysemy và phức tạp trong các ngôn ngữ tự nhiên, các đại diện độc lập ngữ cảnh có những hạn chế rõ ràng. Ví dụ, từ “crane” trong ngữ cảnh “một cần cẩu đang bay” và “một trình điều khiển cần cẩu đến” có ý nghĩa hoàn toàn khác nhau; do đó, cùng một từ có thể được gán các biểu diễn khác nhau tùy thuộc vào ngữ cảnh. 

Điều này thúc đẩy sự phát triển của biểu diễn từ *context-sensitive*, trong đó biểu diễn của các từ phụ thuộc vào bối cảnh của chúng. Do đó, một biểu diễn nhạy cảm với ngữ cảnh của token $x$ là một hàm $f(x, c(x))$ tùy thuộc vào cả $x$ và bối cảnh của nó $c(x)$. Các biểu diễn nhạy cảm với ngữ cảnh phổ biến bao gồm TagLM (tagger trình tự tăng cường mô hình ngôn ngữ) :cite:`Peters.Ammar.Bhagavatula.ea.2017`, cove (Context Vectors) :cite:`McCann.Bradbury.Xiong.ea.2017` và ELMo (Embeddings from Language Models) :cite:`Peters.Neumann.Iyyer.ea.2018`. 

Ví dụ, bằng cách lấy toàn bộ chuỗi làm đầu vào, ELMo là một hàm gán một biểu diễn cho mỗi từ từ chuỗi đầu vào. Cụ thể, ELMo kết hợp tất cả các biểu diễn lớp trung gian từ LSTM hai chiều được đào tạo trước làm đại diện đầu ra. Sau đó, đại diện ELMo sẽ được thêm vào mô hình giám sát hiện có của tác vụ hạ nguồn dưới dạng các tính năng bổ sung, chẳng hạn như bằng cách nối đại diện ELMo và biểu diễn ban đầu (ví dụ: Glove) của các mã thông báo trong mô hình hiện có. Một mặt, tất cả các trọng lượng trong mô hình LSTM hai chiều được đào tạo trước được đóng băng sau khi biểu diễn ELMo được thêm vào. Mặt khác, mô hình giám sát hiện có được tùy chỉnh đặc biệt cho một nhiệm vụ nhất định. Tận dụng các mô hình tốt nhất khác nhau cho các nhiệm vụ khác nhau tại thời điểm đó, thêm ELMo đã cải thiện trạng thái của nghệ thuật trong sáu nhiệm vụ xử lý ngôn ngữ tự nhiên: phân tích tình cảm, suy luận ngôn ngữ tự nhiên, ghi nhãn vai trò ngữ nghĩa, giải quyết coreference, nhận dạng thực thể được đặt tên và trả lời câu hỏi. 

## Từ nhiệm vụ cụ thể đến nhiệm vụ Agnostic

Mặc dù ELMo đã cải thiện đáng kể các giải pháp cho một tập hợp các nhiệm vụ xử lý ngôn ngữ tự nhiên đa dạng, mỗi giải pháp vẫn dựa trên một kiến trúc *cụ thể tác vụ*. Tuy nhiên, thực tế là không tầm thường để tạo ra một kiến trúc cụ thể cho mọi nhiệm vụ xử lý ngôn ngữ tự nhiên. Mô hình GPT (Generative Pre-Training) đại diện cho một nỗ lực trong việc thiết kế mô hình *nhiệm vụ chung cho các biểu diễn nhạy cảm với ngữ cảnh :cite:`Radford.Narasimhan.Salimans.ea.2018`. Được xây dựng trên một bộ giải mã biến áp, GPT pretrain một mô hình ngôn ngữ sẽ được sử dụng để đại diện cho chuỗi văn bản. Khi áp dụng GPT cho một tác vụ hạ lưu, đầu ra của mô hình ngôn ngữ sẽ được đưa vào một lớp đầu ra tuyến tính bổ sung để dự đoán nhãn của tác vụ. Ngược lại sắc nét với ELMo đóng băng các thông số của mô hình được đào tạo trước, GPT tinh chỉnh * tất cả* các thông số trong bộ giải mã biến áp được đào tạo trước trong quá trình học tập được giám sát về nhiệm vụ hạ lưu. GPT được đánh giá trên mười hai nhiệm vụ suy luận ngôn ngữ tự nhiên, trả lời câu hỏi, tương tự câu, và phân loại, và cải thiện trạng thái của nghệ thuật trong chín trong số đó với những thay đổi tối thiểu đối với kiến trúc mô hình. 

Tuy nhiên, do tính chất tự hồi quy của các mô hình ngôn ngữ, GPT chỉ nhìn về phía trước (từ trái sang phải). Trong bối cảnh “tôi đã đến ngân hàng để gửi tiền mặt” và “tôi đã đi đến ngân hàng để ngồi xuống”, vì “ngân hàng” nhạy cảm với bối cảnh bên trái của nó, GPT sẽ trả lại đại diện tương tự cho “ngân hàng”, mặc dù nó có ý nghĩa khác nhau. 

## BERT: Kết hợp tốt nhất của cả hai thế giới

Như chúng ta đã thấy, ELMo mã hóa ngữ cảnh hai chiều nhưng sử dụng các kiến trúc cụ thể tác vụ; trong khi GPT là nhiệm vụ bất khả tri nhưng mã hóa ngữ cảnh từ trái sang phải. Kết hợp tốt nhất của cả hai thế giới, BERT (Bidirectional Encoder Representations from Transformers) mã hóa bối cảnh hai chiều và yêu cầu thay đổi kiến trúc tối thiểu cho một loạt các nhiệm vụ xử lý ngôn ngữ tự nhiên :cite:`Devlin.Chang.Lee.ea.2018`. Sử dụng bộ mã hóa biến áp được đào tạo trước, BERT có thể đại diện cho bất kỳ mã thông báo nào dựa trên bối cảnh hai chiều của nó. Trong quá trình học tập giám sát các nhiệm vụ hạ lưu, BERT tương tự như GPT về hai khía cạnh. Đầu tiên, các đại diện BERT sẽ được đưa vào một lớp đầu ra bổ sung, với những thay đổi tối thiểu đối với kiến trúc mô hình tùy thuộc vào tính chất của các nhiệm vụ, chẳng hạn như dự đoán cho mọi token so với dự đoán cho toàn bộ chuỗi. Thứ hai, tất cả các thông số của bộ mã hóa biến áp được đào tạo trước đều được tinh chỉnh, trong khi lớp đầu ra bổ sung sẽ được đào tạo từ đầu. :numref:`fig_elmo-gpt-bert` mô tả sự khác biệt giữa ELMo, GPT và BERT. 

![A comparison of ELMo, GPT, and BERT.](../img/elmo-gpt-bert.svg)
:label:`fig_elmo-gpt-bert`

BERT cải thiện hơn nữa trạng thái của nghệ thuật trên mười một nhiệm vụ xử lý ngôn ngữ tự nhiên theo các loại rộng của (i) phân loại văn bản đơn (ví dụ, phân tích tình cảm), (ii) phân loại cặp văn bản (ví dụ, suy luận ngôn ngữ tự nhiên), (iii) trả lời câu hỏi, (iv) gắn thẻ văn bản (ví dụ, nhận dạng thực thể được đặt tên) . Tất cả được đề xuất vào năm 2018, từ ELMo nhạy cảm với ngữ cảnh đến GPT và BERT bất khả tri nhiệm vụ, đơn giản về mặt khái niệm nhưng mạnh mẽ về mặt thực nghiệm về các biểu diễn sâu sắc cho các ngôn ngữ tự nhiên đã cách mạng hóa các giải pháp cho các nhiệm vụ xử lý ngôn ngữ tự nhiên khác nhau. 

Trong phần còn lại của chương này, chúng tôi sẽ đi sâu vào pretraining của BERT. Khi các ứng dụng xử lý ngôn ngữ tự nhiên được giải thích trong :numref:`chap_nlp_app`, chúng tôi sẽ minh họa tinh chỉnh BERT cho các ứng dụng hạ nguồn.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

## Đại diện đầu vào
:label:`subsec_bert_input_rep`

Trong xử lý ngôn ngữ tự nhiên, một số nhiệm vụ (ví dụ, phân tích tình cảm) lấy văn bản đơn làm đầu vào, trong khi trong một số nhiệm vụ khác (ví dụ, suy luận ngôn ngữ tự nhiên), đầu vào là một cặp chuỗi văn bản. Chuỗi đầu vào BERT rõ ràng đại diện cho cả hai cặp văn bản và văn bản đơn lẻ. Trước đây, chuỗi đầu vào BERT là sự nối của mã thông báo phân loại đặc biệt “<cls>”, mã thông báo của một chuỗi văn bản, và mã thông báo tách đặc biệt “<sep>”. Sau này, chuỗi đầu vào BERT là sự nối của “<cls>”, mã thông báo của chuỗi văn bản đầu tiên, “<sep>”, mã thông báo của chuỗi văn bản thứ hai, và “<sep>”. Chúng tôi sẽ liên tục phân biệt thuật ngữ “chuỗi đầu vào BERT” với các loại “chuỗi” khác. Ví dụ: một trình tự đầu vào *BERT* có thể bao gồm một chuỗi văn bản* hoặc hai chuỗi văn bản*. 

Để phân biệt các cặp văn bản, các nhúng phân đoạn đã học được $\mathbf{e}_A$ và $\mathbf{e}_B$ được thêm vào các bản nhúng mã thông báo của chuỗi thứ nhất và chuỗi thứ hai, tương ứng. Đối với các đầu vào văn bản đơn, chỉ có $\mathbf{e}_A$ được sử dụng. 

`get_tokens_and_segments` sau lấy một câu hoặc hai câu làm đầu vào, sau đó trả về token của chuỗi đầu vào BERT và ID phân đoạn tương ứng của chúng.

```{.python .input}
#@tab all
#@save
def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of the BERT input sequence and their segment IDs."""
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 and 1 are marking segment A and B, respectively
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
```

BERT chọn bộ mã hóa biến áp làm kiến trúc hai chiều của nó. Phổ biến trong bộ mã hóa biến áp, nhúng vị trí được thêm vào ở mọi vị trí của chuỗi đầu vào BERT. Tuy nhiên, khác với bộ mã hóa biến áp ban đầu, BERT sử dụng * learnable* embeddings vị trí. Tóm lại, :numref:`fig_bert-input` cho thấy các nhúng của chuỗi đầu vào BERT là tổng của các nhúng mã thông báo, nhúng phân đoạn và nhúng vị trí. 

! [Các embeddings của chuỗi đầu vào BERT là tổng của các mã thông báo embeddings, segment embeddings, and positional embeddings.](.. /img/bert-input.svg) :label:`fig_bert-input` 

Lớp `BERTEncoder` sau tương tự như lớp `TransformerEncoder` như được triển khai vào năm :numref:`sec_transformer`. Khác với `TransformerEncoder`, `BERTEncoder` sử dụng nhúng phân đoạn và nhúng vị trí có thể học được.

```{.python .input}
#@save
class BERTEncoder(nn.Block):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = self.params.get('pos_embedding',
                                             shape=(1, max_len, num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data(ctx=X.ctx)[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

```{.python .input}
#@tab pytorch
#@save
class BERTEncoder(nn.Module):
    """BERT encoder."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # In BERT, positional embeddings are learnable, thus we create a
        # parameter of positional embeddings that are long enough
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, segments, valid_lens):
        # Shape of `X` remains unchanged in the following code snippet:
        # (batch size, max sequence length, `num_hiddens`)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

Giả sử rằng kích thước từ vựng là 10000. Để chứng minh suy luận chuyển tiếp của `BERTEncoder`, chúng ta hãy tạo một thể hiện của nó và khởi tạo các tham số của nó.

```{.python .input}
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
num_layers, dropout = 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
```

```{.python .input}
#@tab pytorch
vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 768, 1024, 4
norm_shape, ffn_num_input, num_layers, dropout = [768], 768, 2, 0.2
encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                      ffn_num_hiddens, num_heads, num_layers, dropout)
```

Chúng tôi định nghĩa `tokens` là 2 chuỗi đầu vào BERT có độ dài 8, trong đó mỗi mã thông báo là một chỉ số của từ vựng. Suy luận chuyển tiếp của `BERTEncoder` với đầu vào `tokens` trả về kết quả được mã hóa trong đó mỗi mã thông báo được biểu diễn bởi một vectơ có độ dài được xác định trước bởi siêu tham số `num_hiddens`. Siêu tham số này thường được gọi là * hidden size* (số đơn vị ẩn) của bộ mã hóa biến áp.

```{.python .input}
tokens = np.random.randint(0, vocab_size, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

```{.python .input}
#@tab pytorch
tokens = torch.randint(0, vocab_size, (2, 8))
segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

## Nhiệm vụ Pretraining
:label:`subsec_bert_pretraining_tasks`

Suy luận chuyển tiếp của `BERTEncoder` cho phép đại diện BERT của mỗi mã thông báo của văn bản đầu vào và các mã thông báo đặc biệt được chèn “<cls>” và “<seq>”. Tiếp theo, chúng tôi sẽ sử dụng các đại diện này để tính toán hàm mất cho BERT pretraining. Việc đào tạo trước bao gồm hai nhiệm vụ sau: mô hình hóa ngôn ngữ đeo mặt nạ và dự đoán câu tiếp theo. 

### Mô hình hóa ngôn ngữ đeo mặt nạ
:label:`subsec_mlm`

Như minh họa trong :numref:`sec_language_model`, một mô hình ngôn ngữ dự đoán một mã thông báo sử dụng ngữ cảnh bên trái của nó. Để mã hóa ngữ cảnh hai chiều để đại diện cho mỗi mã thông báo, BERT ngẫu nhiên che dấu mã thông báo và sử dụng mã thông báo từ bối cảnh hai chiều để dự đoán các token đeo mặt nạ theo cách tự giám sát. Nhiệm vụ này được gọi là mô hình ngôn ngữ * được đeo mặt*. 

Trong nhiệm vụ đào tạo trước này, 15% mã thông báo sẽ được chọn ngẫu nhiên làm mã thông báo đeo mặt nạ để dự đoán. Để dự đoán một token đeo mặt nạ mà không gian lận bằng cách sử dụng nhãn, một cách tiếp cận đơn giản là luôn thay thế nó bằng một <mask>mã thông báo “” đặc biệt trong chuỗi đầu vào BERT. Tuy nhiên, mã thông báo đặc biệt nhân tạo “<mask>” sẽ không bao giờ xuất hiện trong tinh chỉnh. Để tránh sự không phù hợp giữa đào tạo trước và tinh chỉnh, nếu một mã thông báo được che dấu để dự đoán (ví dụ: “great” được chọn để che dấu và dự đoán trong “bộ phim này là tuyệt vời”), trong đầu vào nó sẽ được thay thế bằng: 

* một <mask>mã thông báo “” đặc biệt cho 80% thời gian (ví dụ: “bộ phim này thật tuyệt” trở thành “bộ phim này là <mask>“);
* một mã thông báo ngẫu nhiên cho 10% thời gian (ví dụ: “bộ phim này là tuyệt vời” trở thành “bộ phim này là đồ uống”);
* mã thông báo nhãn không thay đổi trong 10% thời gian (ví dụ: “bộ phim này thật tuyệt” trở thành “bộ phim này thật tuyệt”).

Lưu ý rằng trong 10% thời gian 15%, một mã thông báo ngẫu nhiên được chèn vào. Tiếng ồn thỉnh thoảng này khuyến khích BERT ít thiên vị hơn đối với mã thông báo được đeo mặt nạ (đặc biệt là khi mã thông báo nhãn vẫn không thay đổi) trong mã hóa ngữ cảnh hai chiều của nó. 

Chúng tôi thực hiện lớp `MaskLM` sau đây để dự đoán các token đeo mặt nạ trong nhiệm vụ mô hình ngôn ngữ đeo mặt nạ của BERT pretraining. Dự đoán sử dụng MLP một lớp ẩn (`self.mlp`). Trong suy luận chuyển tiếp, phải mất hai đầu vào: kết quả được mã hóa của `BERTEncoder` và các vị trí mã thông báo để dự đoán. Đầu ra là kết quả dự đoán tại các vị trí này.

```{.python .input}
#@save
class MaskLM(nn.Block):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `np.array([0, 0, 0, 1, 1, 1])`
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class MaskLM(nn.Module):
    """The masked language model task of BERT."""
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # Suppose that `batch_size` = 2, `num_pred_positions` = 3, then
        # `batch_idx` is `torch.tensor([0, 0, 0, 1, 1, 1])`
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

Để chứng minh suy luận về phía trước của `MaskLM`, chúng tôi tạo ra phiên bản `mlm` của nó và khởi tạo nó. Nhớ lại rằng `encoded_X` từ suy luận về phía trước của `BERTEncoder` đại diện cho 2 chuỗi đầu vào BERT. Chúng tôi định nghĩa `mlm_positions` là 3 chỉ số để dự đoán trong một trong hai chuỗi đầu vào BERT là `encoded_X`. Suy luận về phía trước của `mlm` trả về kết quả dự đoán `mlm_Y_hat` tại tất cả các vị trí đeo mặt nạ `mlm_positions` của `encoded_X`. Đối với mỗi dự đoán, kích thước của kết quả bằng với kích thước từ vựng.

```{.python .input}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input}
#@tab pytorch
mlm = MaskLM(vocab_size, num_hiddens)
mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

Với nhãn chân lý mặt đất `mlm_Y` của các mã thông báo dự đoán `mlm_Y_hat` dưới mặt nạ, chúng ta có thể tính toán sự mất mát chéo entropy của nhiệm vụ mô hình ngôn ngữ đeo mặt nạ trong pretraining BERT.

```{.python .input}
mlm_Y = np.array([[7, 8, 9], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

```{.python .input}
#@tab pytorch
mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
loss = nn.CrossEntropyLoss(reduction='none')
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

### Dự đoán câu tiếp theo
:label:`subsec_nsp`

Mặc dù mô hình hóa ngôn ngữ đeo mặt nạ có thể mã hóa ngữ cảnh hai chiều để biểu diễn các từ, nó không mô hình hóa rõ ràng mối quan hệ logic giữa các cặp văn bản. Để giúp hiểu mối quan hệ giữa hai chuỗi văn bản, BERT xem xét một nhiệm vụ phân loại nhị phân, * dự đoán câu tiếp theo*, trong pretraining của nó. Khi tạo ra các cặp câu cho pretraining, trong một nửa thời gian chúng thực sự là những câu liên tiếp với nhãn “True”; trong khi trong nửa còn lại của thời gian câu thứ hai được lấy mẫu ngẫu nhiên từ corpus với nhãn “False”. 

Lớp `NextSentencePred` sau sử dụng MLP một lớp ẩn để dự đoán câu thứ hai có phải là câu tiếp theo của câu thứ nhất trong chuỗi đầu vào BERT hay không. Do sự tự chú ý trong bộ mã hóa biến áp, đại diện BERT của mã thông báo đặc biệt “<cls>” mã hóa cả hai câu từ đầu vào. Do đó, lớp đầu ra (`self.output`) của phân loại MLP lấy `X` làm đầu vào, trong đó `X` là đầu ra của lớp ẩn MLP có đầu vào là mã <cls>thông báo “” được mã hóa.

```{.python .input}
#@save
class NextSentencePred(nn.Block):
    """The next sentence prediction task of BERT."""
    def __init__(self, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Dense(2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

```{.python .input}
#@tab pytorch
#@save
class NextSentencePred(nn.Module):
    """The next sentence prediction task of BERT."""
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        # `X` shape: (batch size, `num_hiddens`)
        return self.output(X)
```

Chúng ta có thể thấy rằng suy luận chuyển tiếp của một phiên bản `NextSentencePred` trả về dự đoán nhị phân cho mỗi chuỗi đầu vào BERT.

```{.python .input}
nsp = NextSentencePred()
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input}
#@tab pytorch
# PyTorch by default won't flatten the tensor as seen in mxnet where, if
# flatten=True, all but the first axis of input data are collapsed together
encoded_X = torch.flatten(encoded_X, start_dim=1)
# input_shape for NSP: (batch size, `num_hiddens`)
nsp = NextSentencePred(encoded_X.shape[-1])
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

Sự mất mát cross-entropy của 2 phân loại nhị phân cũng có thể được tính toán.

```{.python .input}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

```{.python .input}
#@tab pytorch
nsp_y = torch.tensor([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

Đáng chú ý là tất cả các nhãn trong cả hai nhiệm vụ đào tạo trước nói trên đều có thể thu được một cách tầm thường từ cơ thể đào tạo trước mà không cần nỗ lực ghi nhãn thủ công. BERT ban đầu đã được đào tạo trước về việc nối BookCorpus :cite:`Zhu.Kiros.Zemel.ea.2015` và Wikipedia tiếng Anh. Hai tập đoàn văn bản này rất lớn: họ có 800 triệu từ và 2,5 tỷ từ, tương ứng. 

## Đặt tất cả mọi thứ lại với nhau

Khi đào tạo trước BERT, hàm mất cuối cùng là sự kết hợp tuyến tính của cả hàm mất cho mô hình ngôn ngữ đeo mặt nạ và dự đoán câu tiếp theo. Bây giờ chúng ta có thể xác định lớp `BERTModel` bằng cách khởi tạo ba lớp `BERTEncoder`, `MaskLM` và `NextSentencePred`. Suy luận chuyển tiếp trả về các đại diện BERT được mã hóa `encoded_X`, dự đoán về mô hình hóa ngôn ngữ đeo mặt nạ `mlm_Y_hat`, và dự đoán câu tiếp theo `nsp_Y_hat`.

```{.python .input}
#@save
class BERTModel(nn.Block):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, max_len=1000):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout, max_len)
        self.hidden = nn.Dense(num_hiddens, activation='tanh')
        self.mlm = MaskLM(vocab_size, num_hiddens)
        self.nsp = NextSentencePred()

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

```{.python .input}
#@tab pytorch
#@save
class BERTModel(nn.Module):
    """The BERT model."""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=1000, key_size=768, query_size=768, value_size=768,
                 hid_in_features=768, mlm_in_features=768,
                 nsp_in_features=768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_features)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        # The hidden layer of the MLP classifier for next sentence prediction.
        # 0 is the index of the '<cls>' token
        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

## Tóm tắt

* Các mô hình nhúng từ như word2vec và Glove độc lập với ngữ cảnh. Họ gán cùng một vector được đào tạo trước cho cùng một từ bất kể ngữ cảnh của từ (nếu có). Thật khó để họ xử lý tốt ngữ nghĩa polysemy hoặc phức tạp trong các ngôn ngữ tự nhiên.
* Đối với các biểu diễn từ nhạy cảm với ngữ cảnh như ELMo và GPT, biểu diễn các từ phụ thuộc vào ngữ cảnh của chúng.
* ELMo mã hóa ngữ cảnh hai chiều nhưng sử dụng kiến trúc cụ thể tác vụ (tuy nhiên, thực tế là không tầm thường để chế tạo một kiến trúc cụ thể cho mọi nhiệm vụ xử lý ngôn ngữ tự nhiên); trong khi GPT là nhiệm vụ bất khả tri nhưng mã hóa ngữ cảnh từ trái sang phải.
* BERT kết hợp tốt nhất của cả hai thế giới: nó mã hóa bối cảnh hai chiều và đòi hỏi những thay đổi kiến trúc tối thiểu cho một loạt các nhiệm vụ xử lý ngôn ngữ tự nhiên.
* Các embeddings của chuỗi đầu vào BERT là tổng của các embeddings token, phân đoạn embeddings, và embeddings vị trí.
* Pretraining BERT bao gồm hai nhiệm vụ: mô hình hóa ngôn ngữ đeo mặt nạ và dự đoán câu tiếp theo. Cái trước có thể mã hóa ngữ cảnh hai chiều để đại diện cho các từ, trong khi sau này mô hình rõ ràng mối quan hệ logic giữa các cặp văn bản.

## Bài tập

1. Tại sao BERT lại thành công?
1. Tất cả những thứ khác đều bình đẳng, một mô hình ngôn ngữ đeo mặt nạ sẽ yêu cầu nhiều hơn hoặc ít hơn các bước đào tạo trước để hội tụ hơn một mô hình ngôn ngữ từ trái sang phải? Tại sao?
1. Trong việc triển khai ban đầu của BERT, mạng chuyển tiếp nguồn cấp dữ liệu định vị trong `BERTEncoder` (thông qua `d2l.EncoderBlock`) và lớp kết nối hoàn toàn trong `MaskLM` cả hai đều sử dụng đơn vị tuyến tính lỗi Gaussian (GELU) :cite:`Hendrycks.Gimpel.2016` làm chức năng kích hoạt. Nghiên cứu về sự khác biệt giữa GELU và ReLU.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/388)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1490)
:end_tab:

# Từ tương tự và tương tự
:label:`sec_synonyms`

Trong :numref:`sec_word2vec_pretraining`, chúng tôi đã đào tạo một mô hình word2vec trên một tập dữ liệu nhỏ và áp dụng nó để tìm các từ tương tự về mặt ngữ nghĩa cho một từ đầu vào. Trong thực tế, các vectơ từ được đào tạo trước trên thể lớn có thể được áp dụng cho các nhiệm vụ xử lý ngôn ngữ tự nhiên hạ nguồn, sẽ được đề cập sau này vào năm :numref:`chap_nlp_app`. Để chứng minh ngữ nghĩa của vectơ từ được đào tạo trước từ thể lớn một cách đơn giản, chúng ta hãy áp dụng chúng trong các nhiệm vụ tương tự và tương tự từ.

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
from torch import nn
import os
```

## Đang tải Pretrained Word Vectơ

Dưới đây liệt kê các bản nhúng Glove được đào tạo trước có kích thước 50, 100 và 300, có thể tải xuống từ [GloVe website](https://nlp.stanford.edu/projects/glove/). Các bản nhúng fastText được đào tạo trước có sẵn bằng nhiều ngôn ngữ. Ở đây chúng tôi xem xét một phiên bản tiếng Anh (300-dimensional “wiki.en”) có thể được tải xuống từ [fastText website](https://fasttext.cc/).

```{.python .input}
#@tab all
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

Để tải các Glove được đào tạo trước này và các embeddings fastText, chúng tôi xác định lớp `TokenEmbedding` sau.

```{.python .input}
#@tab all
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, d2l.tensor(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[d2l.tensor(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Dưới đây chúng tôi tải các nhúng Glove 50 chiều (được đào tạo trước trên một tập con Wikipedia). Khi tạo phiên bản `TokenEmbedding`, tệp nhúng được chỉ định phải được tải xuống nếu chưa.

```{.python .input}
#@tab all
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Xuất kích thước từ vựng. Từ vựng chứa 400000 từ (mã thông báo) và một mã thông báo không xác định đặc biệt.

```{.python .input}
#@tab all
len(glove_6b50d)
```

Chúng ta có thể lấy chỉ mục của một từ trong từ vựng, và ngược lại.

```{.python .input}
#@tab all
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Áp dụng vectơ Word Pretrained

Sử dụng các vectơ Glove được tải, chúng tôi sẽ chứng minh ngữ nghĩa của chúng bằng cách áp dụng chúng trong các tác vụ tương tự và tương tự từ sau đây. 

### Từ tương tự

Tương tự như :numref:`subsec_apply-word-embed`, để tìm các từ tương tự về mặt ngữ nghĩa cho một từ đầu vào dựa trên sự tương đồng cosin giữa các vectơ từ, chúng tôi thực hiện hàm `knn` ($k$-láng giềng gần nhất) sau đây.

```{.python .input}
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

```{.python .input}
#@tab pytorch
def knn(W, x, k):
    # Add 1e-9 for numerical stability
    cos = torch.mv(W, x.reshape(-1,)) / (
        torch.sqrt(torch.sum(W * W, axis=1) + 1e-9) *
        torch.sqrt((x * x).sum()))
    _, topk = torch.topk(cos, k=k)
    return topk, [cos[int(i)] for i in topk]
```

Sau đó, chúng ta tìm kiếm các từ tương tự bằng cách sử dụng vectơ từ được đào tạo trước từ trường hợp `TokenEmbedding` `embed`.

```{.python .input}
#@tab all
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Exclude the input word
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

Từ vựng của các vectơ từ được đào tạo trước trong `glove_6b50d` chứa 400000 từ và một mã thông báo không xác định đặc biệt. Không bao gồm từ đầu vào và mã thông báo không xác định, trong số từ vựng này cho phép chúng ta tìm thấy ba từ tương tự về mặt ngữ nghĩa nhất với từ “chip”.

```{.python .input}
#@tab all
get_similar_tokens('chip', 3, glove_6b50d)
```

Dưới đây xuất ra các từ tương tự như “em bé” và “đẹp”.

```{.python .input}
#@tab all
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
#@tab all
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Từ tương tự

Bên cạnh việc tìm các từ tương tự, chúng ta cũng có thể áp dụng vectơ từ cho các tác vụ tương tự từ. Ví dụ, “man”: “woman”:: “con trai”: “con gái” là hình thức của một từ tương tự: “người đàn ông” là “người phụ nữ” là “con trai” là “con gái”. Cụ thể, nhiệm vụ hoàn thành từ tương tự có thể được định nghĩa là: đối với một từ tương tự $a : b :: c : d$, cho ba từ đầu tiên $a$, $b$ và $c$, tìm $d$. Biểu thị vector của từ $w$ bởi $\text{vec}(w)$. Để hoàn thành sự tương tự, chúng ta sẽ tìm thấy từ có vector tương tự nhất với kết quả của $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```{.python .input}
#@tab all
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Hãy để chúng tôi xác minh sự tương tự “nam-nữ” bằng cách sử dụng các vectơ từ được tải.

```{.python .input}
#@tab all
get_analogy('man', 'woman', 'son', glove_6b50d)
```

Dưới đây hoàn thành một cách tương tự “thủ đô-quốc gia”: “beijing”: “china”:: “tokyo”: “japan”. Điều này thể hiện ngữ nghĩa trong vectơ từ được đào tạo trước.

```{.python .input}
#@tab all
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

Đối với tính từ tương tự “tính từ siêu” như “bad”: “worst”:: “big”: “big”: “biggest”, chúng ta có thể thấy rằng các vectơ từ được đào tạo trước có thể nắm bắt thông tin cú pháp.

```{.python .input}
#@tab all
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

Để thể hiện khái niệm bị bắt về thì quá khứ trong vectơ từ được đào tạo trước, chúng ta có thể kiểm tra cú pháp bằng cách sử dụng tương tự “thì quá khứ hiện tại”: “do”: “did”:: “go”: “went”.

```{.python .input}
#@tab all
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Tóm tắt

* Trong thực tế, các vectơ từ được đào tạo trước trên thể lớn có thể được áp dụng cho các nhiệm vụ xử lý ngôn ngữ tự nhiên hạ nguồn.
* Vectơ từ được đào tạo trước có thể được áp dụng cho các tác vụ tương tự và tương tự từ.

## Bài tập

1. Kiểm tra kết quả fastText bằng cách sử dụng `TokenEmbedding('wiki.en')`.
1. Khi từ vựng cực kỳ lớn, làm thế nào chúng ta có thể tìm thấy các từ tương tự hoặc hoàn thành một từ tương tự nhanh hơn?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/387)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1336)
:end_tab:

<!--
# Finding Synonyms and Analogies
-->

# Tìm kiếm từ Đồng nghĩa và Loại suy
:label:`sec_synonyms`


<!--
In :numref:`sec_word2vec_pretraining` we trained a word2vec word embedding model on a small-scale dataset and searched for synonyms using the cosine similarity of word vectors.
In practice, word vectors pretrained on a large-scale corpus can often be applied to downstream natural language processing tasks.
This section will demonstrate how to use these pretrained word vectors to find synonyms and analogies.
We will continue to apply pretrained word vectors in subsequent sections.
-->

Trong :numref:`sec_word2vec_pretraining` ta đã huấn luyện mô hình embedding từ word2vec trên tập dữ liệu cỡ nhỏ và tìm kiếm các từ đồng nghĩa sử dụng độ tương tự cô-sin giữa các vector từ.
Trong thực tế, các vector từ được tiền huấn luyện trên kho ngữ liệu cỡ lớn thường được áp dụng cho các bài toán xử lý ngôn ngữ tự nhiên cụ thể.
Phần này sẽ trình bày cách sử dụng các vector từ đã tiền huấn luyện để tìm các từ đồng nghĩa và các loại suy (*analogy*).
Ta sẽ tiếp tục áp dụng các vector từ được tiền huấn luyện trong các phần sau.


```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import os

npx.set_np()
```


<!--
## Using Pretrained Word Vectors
-->

## Sử dụng các Vector Từ đã được Tiền Huấn luyện


<!--
Below lists pretrained GloVe embeddings of dimensions 50, 100, and 300, which can be downloaded from the [GloVe website](https://nlp.stanford.edu/projects/glove/).
The pretrained fastText embeddings are available in multiple languages.
Here we consider one English version (300-dimensional "wiki.en") that can be downloaded from the [fastText website](https://fasttext.cc/).
-->

Dưới đây là các embedding GloVe đã được tiền huấn luyện với kích thước chiều là 50, 100, và 300, có thể được tải từ [trang web GloVe](https://nlp.stanford.edu/projects/glove/).
Các embedding cho fastText được tiền huấn luyện trên nhiều ngôn ngữ.
Ở đây, ta quan tâm tới phiên bản cho tiếng Anh ("wiki.en" có chiều là 300) có thể được tải từ [trang web fastText](https://fasttext.cc/).


```{.python .input  n=35}
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


<!--
We define the following `TokenEmbedding` class to load the above pretrained Glove and fastText embeddings.
-->

Ta định nghĩa lớp `TokenEmbedding` để nạp các embedding GloVe và fastText ở trên.


```{.python .input}
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
        return idx_to_token, np.array(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[np.array(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```


<!--
Next, we use 50-dimensional GloVe embeddings pretrained on a subset of the Wikipedia.
The corresponding word embedding is automatically downloaded the first time we create a pretrained word embedding instance.
-->

Tiếp theo, ta sử dụng embedding GloVe có chiều là 50 được tiền huấn luyện trên tập con của Wikipedia.
Embedding tương ứng của từ sẽ được tự động tải về khi tạo một thực thể `TokenEmbedding` lần đầu.


```{.python .input  n=11}
glove_6b50d = TokenEmbedding('glove.6b.50d')
```


<!--
Output the dictionary size. The dictionary contains $400,000$ words and a special unknown token.
-->

Ta có thể in ra kích thước từ điển. Từ điển chứa $400,000$ từ và một token đặc biệt cho các từ không biết. 


```{.python .input}
len(glove_6b50d)
```


<!--
We can use a word to get its index in the dictionary, or we can get the word from its index.
-->

Ta có thể lấy chỉ số của một từ trong từ điển, hoặc ngược lại tra từ tương ứng với chỉ số cho trước.


```{.python .input  n=12}
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```


<!--
## Applying Pretrained Word Vectors
-->

## Áp dụng các Vector Từ đã được Tiền huấn luyện


<!--
Below, we demonstrate the application of pretrained word vectors, using GloVe as an example.
-->

Dưới đây, ta minh họa việc áp dụng các vector từ đã được tiền huấn luyện sử dụng Glove làm ví dụ.


<!--
### Finding Synonyms
-->

### Tìm các từ đồng nghĩa


<!--
Here, we re-implement the algorithm used to search for synonyms by cosine similarity introduced in :numref:`sec_word2vec`
-->

Tại đây, ta lập trình lại thuật toán tìm các từ đồng nghĩa bằng độ tương tự cô-sin giữa hai vector trong :numref:`sec_word2vec`.


<!--
In order to reuse the logic for seeking the $k$ nearest neighbors when seeking analogies,
we encapsulate this part of the logic separately in the `knn` ($k$-nearest neighbors) function.
-->

Để sử dụng lại logic tìm kiếm $k$ láng giềng gần nhất ($k$*-nearest neighbors*) khi tìm kiếm các từ loại suy,
ta đóng gói phần này một cách tách biệt trong hàm `knn` .


```{.python .input}
def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```


<!--
Then, we search for synonyms by pre-training the word vector instance `embed`.
-->

Kế tiếp, ta tìm kiếm các từ đồng nghĩa nhờ tiền huấn luyện thực thể vector từ `embed`.


```{.python .input}
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Remove input words
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```


<!--
The dictionary of pretrained word vector instance `glove_6b50d` already created contains 400,000 words and a special unknown token.
Excluding input words and unknown words, we search for the three words that are the most similar in meaning to "chip".
-->

Từ điển vector từ được tiền huấn luyện `glove_6b50d` đã tạo chứa 400,000 từ và một token các từ không biết.
Loại trừ những từ đầu vào và những từ không biết, ta tìm kiếm ba từ có nghĩa gần với từ "chip".


```{.python .input}
get_similar_tokens('chip', 3, glove_6b50d)
```


<!--
Next, we search for the synonyms of "baby" and "beautiful".
-->

Kế tiếp, ta tìm các từ gần nghĩa với "baby" và "beautiful".


```{.python .input}
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
get_similar_tokens('beautiful', 3, glove_6b50d)
```


<!--
### Finding Analogies
-->

### Tìm kiếm các Loại suy


<!--
In addition to seeking synonyms, we can also use the pretrained word vector to seek the analogies between words.
For example, “man”:“woman”::“son”:“daughter” is an example of analogy, “man” is to “woman” as “son” is to “daughter”.
The problem of seeking analogies can be defined as follows: for four words in the analogical relationship $a : b :: c : d$, 
given the first three words, $a$, $b$ and $c$, we want to find $d$.
Assume the word vector for the word $w$ is $\text{vec}(w)$.
To solve the analogy problem, we need to find the word vector that is most similar to the result vector of $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.
-->

Bên cạnh việc tìm kiếm các từ đồng nghĩa, ta cũng có thể sử dụng các vector từ đã tiền huấn luyện để tìm kiếm các loại suy giữa các từ.
Ví dụ, “man”:“woman”::“son”:“daughter” là một loại suy, "man (nam)" với "woman (nữ)" giống như "son (con trai)" với "daughter (con gái)".
Bài toán tìm kiếm loại suy có thể được định nghĩa như sau: với bốn từ trong quan hệ loại suy $a : b :: c : d$, 
cho trước ba từ $a$, $b$ và $c$, ta muốn tìm từ $d$.
Giả sử, vector từ cho từ $w$ là $\text{vec}(w)$.
Để giải quyết bài toán loại suy, ta cần tìm vector từ gần nhất với vector là kết quả của $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.


```{.python .input}
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```


<!--
Verify the "male-female" analogy.
-->

Kiểm tra quan hệ loại suy "nam giới - nữ giới".


```{.python .input  n=18}
get_analogy('man', 'woman', 'son', glove_6b50d)
```


<!--
“Capital-country” analogy: "beijing" is to "china" as "tokyo" is to what? The answer should be "japan".
-->

Loại suy "thủ đô-quốc gia”: từ "beijing" với từ "china" tương tự như từ "tokyo" với từ nào? Đáp án là "japan".


```{.python .input  n=19}
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```


<!--
"Adjective-superlative adjective" analogy: "bad" is to "worst" as "big" is to what? The answer should be "biggest".
-->

Loại suy "tính từ - tính từ so sánh nhất": từ "bad" với từ "worst" tương tự như từ "big" với từ nào? Đáp án là "biggest".


```{.python .input  n=20}
get_analogy('bad', 'worst', 'big', glove_6b50d)
```


<!--
"Present tense verb-past tense verb" analogy: "do" is to "did" as "go" is to what? The answer should be "went".
-->

Loại suy "động từ thì hiện tại - động từ thì quá khứ": từ "do" với từ "did" tương tự như từ "go" với từ nào? Đáp án là "went".


```{.python .input  n=21}
get_analogy('do', 'did', 'go', glove_6b50d)
```


## Tóm tắt

<!--
* Word vectors pre-trained on a large-scale corpus can often be applied to downstream natural language processing tasks.
* We can use pre-trained word vectors to seek synonyms and analogies.
-->

* Các vector từ được tiền huấn luyện trên kho ngữ liệu cỡ lớn thường được áp dụng cho các tác vụ xử lý ngôn ngữ tự nhiên.
* Ta có thể sử dụng các vector từ được tiền huấn luyện để tìm kiếm các từ đồng nghĩa và các loại suy.

## Bài tập

<!--
1. Test the fastText results using `TokenEmbedding('wiki.en')`.
2. If the dictionary is extremely large, how can we accelerate finding synonyms and analogies?
-->

1. Hãy kiểm tra kết quả với fastText bằng cách sử dụng `TokenEmbedding('wiki.en')`.
2. Nếu từ điển quá lớn, ta có thể tăng tốc tìm kiếm các từ đồng nghĩa và các loại suy bằng cách nào?


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/387)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Mai Hoàng Long
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Nguyễn Văn Cường

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 01/07/2020)*

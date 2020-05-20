<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE BẮT ĐẦU =================================== -->

<!--
# Text Preprocessing
-->

# Tiền Xử lý Dữ liệu Văn bản

:label:`sec_text_preprocessing`

<!--
Text is an important example of sequence data.
An article can be simply viewed as a sequence of words, or a sequence of characters.
Given text data is a major data format besides images we are using in this book, this section will dedicate to explain the common preprocessing steps for text data.
Such preprocessing often consists of four steps:
-->

Dữ liệu văn bản là một ví dụ điển hình của dữ liệu chuỗi.
Một bài báo có thể coi là một chuỗi các từ, hoặc một chuỗi các ký tự.
Với dữ liệu văn bản là một định dạng dữ liệu quan trọng bên cạnh những dữ liệu hình ảnh được chúng ta sử dụng trong cuốn sách này, phần này được dành để giải thích các bước tiền xử lý phổ biến cho dữ liệu văn bản.
Các bước tiền xử lý thường bao gồm bốn bước sau:

<!--
1. Load text as strings into memory.
2. Split strings into tokens, where a token could be a word or a character.
3. Build a vocabulary for these tokens to map them into numerical indices.
4. Map all the tokens in data into indices for ease of feeding into models.
-->

1. Nạp dữ liệu văn bản ở dạng chuỗi ký tự vào bộ nhớ.
2. Chia chuỗi thành các token trong đó một token có thể là một từ hoặc một ký tự.
3. Xây dựng một bộ từ vựng cho các token để ánh xạ chúng thành các chỉ số (*index*).
4. Ánh xạ tất cả các token trong dữ liệu văn bản thành các chỉ số để dễ dàng đưa vào các mô hình.



<!--
## Reading the Dataset
-->

## Đọc Dataset


<!--
To get started we load text from H. G. Wells' [Time Machine](http://www.gutenberg.org/ebooks/35).
This is a fairly small corpus of just over $30,000$ words, but for the purpose of what we want to illustrate this is just fine.
More realistic document collections contain many billions of words.
The following function reads the dataset into a list of sentences, each sentence is a string.
Here we ignore punctuation and capitalization.
-->

Để bắt đầu chúng ta nạp dữ liệu văn bản từ cuốn sách [Cỗ máy Thời gian] (http://www.gutenberg.org/ebooks/35) của tác giả H. G. Wells' (*Time Machine*).
Đây là một kho ngữ liệu khá nhỏ chỉ hơn $30,000$ từ, nhưng nó đủ tốt cho mục đích minh họa.
Nhiều bộ dữ liệu trên thực tế chứa hàng tỷ từ.
Hàm sau đây đọc dữ liệu thành một danh sách các câu, mỗi câu là một chuỗi.
Chúng ta bỏ qua dấu chấm câu và viết hoa.


```{.python .input}
import collections
import d2l
import re

# Saved in the d2l package for later use
d2l.DATA_HUB['time_machine'] = (d2l.DATA_URL + 'timemachine.txt',
                                '090b5e7e70c295757f55df93cb0a180b9691891a')

# Saved in the d2l package for later use
def read_time_machine():
    """Load the time machine book into a list of sentences."""
    with open(d2l.download('time_machine'), 'r') as f:
        lines = f.readlines()
    return [re.sub('[^A-Za-z]+', ' ', line.strip().lower())
            for line in lines]

lines = read_time_machine()
'# sentences %d' % len(lines)
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Tokenization
-->

## *dịch tiêu đề phía trên*

<!--
For each sentence, we split it into a list of tokens.
A token is a data point the model will train and predict.
The following function supports splitting a sentence into words or characters, and returns a list of split strings.
-->

*dịch đoạn phía trên*

```{.python .input}
# Saved in the d2l package for later use
def tokenize(lines, token='word'):
    """Split sentences into word or char tokens."""
    if token == 'word':
        return [line.split(' ') for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type '+token)

tokens = tokenize(lines)
tokens[0:2]
```

<!--
## Vocabulary
-->

## *dịch tiêu đề phía trên*

<!--
The string type of the token is inconvenient to be used by models, which take numerical inputs.
Now let us build a dictionary, often called *vocabulary* as well, to map string tokens into numerical indices starting from 0.
To do so, we first count the unique tokens in all documents, called *corpus*, and then assign a numerical index to each unique token according to its frequency.
Rarely appeared tokens are often removed to reduce the complexity.
A token does not exist in corpus or has been removed is mapped into a special unknown (“&lt;unk&gt;”) token.
We optionally add a list of reserved tokens, such as “&lt;pad&gt;” a token for padding, “&lt;bos&gt;” to present the beginning for a sentence, and “&lt;eos&gt;” for the ending of a sentence.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
# Saved in the d2l package for later use
class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self.token_freqs = sorted(counter.items(), key=lambda x: x[0])
        self.token_freqs.sort(key=lambda x: x[1], reverse=True)
        self.unk, uniq_tokens = 0, ['<unk>'] + reserved_tokens
        uniq_tokens += [token for token, freq in self.token_freqs
                        if freq >= min_freq and token not in uniq_tokens]
        self.idx_to_token, self.token_to_idx = [], dict()
        for token in uniq_tokens:
            self.idx_to_token.append(token)
            self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

# Saved in the d2l package for later use
def count_corpus(sentences):
    # Flatten a list of token lists into a list of tokens
    tokens = [tk for line in sentences for tk in line]
    return collections.Counter(tokens)
```

<!--
We construct a vocabulary with the time machine dataset as the corpus, and then print the map between a few tokens and their indices.
-->

*dịch đoạn phía trên*

```{.python .input  n=23}
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])
```

<!--
After that, we can convert each sentence into a list of numerical indices.
To illustrate in detail, we print two sentences with their corresponding indices.
-->

*dịch đoạn phía trên*

```{.python .input  n=25}
for i in range(8, 10):
    print('words:', tokens[i])
    print('indices:', vocab[tokens[i]])
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Putting All Things Together
-->

## *dịch tiêu đề phía trên*

<!--
Using the above functions, we package everything into the `load_corpus_time_machine` function, 
which returns `corpus`, a list of token indices, and `vocab`, the vocabulary of the time machine corpus.
The modification we did here is that `corpus` is a single list, not a list of token lists, since we do not keep the sequence information in the following models.
Besides, we use character tokens to simplify the training in later sections.
-->

*dịch đoạn phía trên*


```{.python .input}
# Saved in the d2l package for later use
def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[tk] for line in tokens for tk in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)
```

<!--
## Summary
-->

## Tóm tắt

<!--
* We preprocessed the documents by tokenizing them into words or characters and then mapping into indices.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
Tokenization is a key preprocessing step.
It varies for different languages.
Try to find another 3 commonly used methods to tokenize sentences.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE KẾT THÚC =================================== -->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2363)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Văn Quang

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE BẮT ĐẦU =================================== -->

<!--
# Text Preprocessing
-->

# Tiền Xử lý Dữ liệu Văn bản
:label:`sec_text_preprocessing`

<!--
Text is an important example of sequence data.
An article can be simply viewed aspels a sequence of words, or a sequence of characters.
Given text data is a major data format besides images we are using in this book, this section will dedicate to explain the common preprocessing steps for text data.
Such preprocessing often consists of four steps:
-->

Dữ liệu văn bản là một ví dụ điển hình của dữ liệu chuỗi. 
Một bài báo có thể coi là một chuỗi các từ, hoặc một chuỗi các ký tự. 
Dữ liệu văn bản là một dạng dữ liệu quan trọng bên cạnh dữ liệu hình ảnh được sử dụng trong cuốn sách này, phần này sẽ được dành để giải thích các bước tiền xử lý thường gặp cho loại dữ liệu này.
Quá trình tiền xử lý thường bao gồm bốn bước sau: 

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

## Đọc Bộ dữ liệu


<!--
To get started we load text from H. G. Wells' [Time Machine](http://www.gutenberg.org/ebooks/35).
This is a fairly small corpus of just over $30,000$ words, but for the purpose of what we want to illustrate this is just fine.
More realistic document collections contain many billions of words.
The following function reads the dataset into a list of sentences, each sentence is a string.
Here we ignore punctuation and capitalization.
-->

Để bắt đầu chúng ta nạp dữ liệu văn bản từ cuốn sách [Cỗ máy Thời gian (*Time Machine*)](http://www.gutenberg.org/ebooks/35) của tác giả H. G. Wells. 
Đây là một kho ngữ liệu khá nhỏ chỉ hơn $30.000$ từ, nhưng nó đủ tốt cho mục đích minh họa. 
Nhiều bộ dữ liệu trên thực tế chứa hàng tỷ từ. 
Hàm sau đây đọc dữ liệu thành một danh sách các câu, mỗi câu là một chuỗi. 
Chúng ta bỏ qua dấu câu và chữ viết hoa. 


```{.python .input}
import collections
from d2l import mxnet as d2l
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

## Token hoá


<!--
For each sentence, we split it into a list of tokens.
A token is a data point the model will train and predict.
The following function supports splitting a sentence into words or characters, and returns a list of split strings.
-->

Với mỗi câu, chúng ta chia nó thành một danh sách các token. 
Một token là một điểm dữ liệu mà mô hình sẽ huấn luyện và đưa ra dự đoán từ nó. 
Hàm dưới đây làm nhiệm vụ tách một câu thành các từ hoặc các ký tự, và trả về một danh sách các chuỗi đã được phân tách. 


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

## Bộ Từ vựng


<!--
The string type of the token is inconvenient to be used by models, which take numerical inputs.
Now let us build a dictionary, often called *vocabulary* as well, to map string tokens into numerical indices starting from 0.
To do so, we first count the unique tokens in all documents, called *corpus*, and then assign a numerical index to each unique token according to its frequency.
Rarely appeared tokens are often removed to reduce the complexity.
A token does not exist in corpus or has been removed is mapped into a special unknown (“&lt;unk&gt;”) token.
We optionally add a list of reserved tokens, such as “&lt;pad&gt;” a token for padding, “&lt;bos&gt;” to present the beginning for a sentence, and “&lt;eos&gt;” for the ending of a sentence.
-->

Token kiểu chuỗi không phải là kiểu dữ liệu tiện lợi được sử dụng bởi các mô hình, thay vào đó chúng thường nhận dữ liệu đầu vào dưới dạng số. 
Bây giờ, chúng ta sẽ xây dựng một bộ từ điển, thường được gọi là *bộ từ vựng* (*vocabulary*), để ánh xạ chuỗi token thành các chỉ số bắt đầu từ 0. 
Để làm điều này, đầu tiên chúng ta lấy các token xuất hiện (*không lặp lại*) trong toàn bộ tài liệu, thường được gọi là kho ngữ liệu (*corpus*), và sau đó gán một giá trị số (*chỉ số*) cho mỗi token dựa trên tần suất xuất hiện của chúng. 
Các token có tần suất xuất hiện rất ít thường được loại bỏ để giảm độ phức tạp. 
Một token không xuất hiện trong kho ngữ liệu hay đã bị loại bỏ thường được ánh xạ vào một token vô danh đặc biệt (“&lt;unk&gt;”). 
Chúng ta có thể tùy chọn thêm vào các token dự trữ, ví dụ token “&lt;pad&gt;” được sử dụng để đệm từ, token “&lt;bos&gt;” để biểu thị vị trí bắt đầu của câu, và token “&lt;eos&gt;” để biểu thị vị trí kết thúc của câu. 


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

Chúng ta xây dựng một bộ từ vựng với tập dữ liệu cỗ máy thời gian nói trên thành một kho ngữ liệu, và in ra phép ánh xạ giữa một vài token với các chỉ số của chúng.


```{.python .input  n=23}
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[0:10])
```

<!--
After that, we can convert each sentence into a list of numerical indices.
To illustrate in detail, we print two sentences with their corresponding indices.
-->

Sau đó, chúng ta có thể chuyển đổi từng câu vào một danh sách các chỉ số. 
Để minh họa một cách chi tiết, chúng ta in hai câu với các chỉ số tương ứng của chúng. 


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

## Kết hợp Tất cả lại


<!--
Using the above functions, we package everything into the `load_corpus_time_machine` function, 
which returns `corpus`, a list of token indices, and `vocab`, the vocabulary of the time machine corpus.
The modification we did here is that `corpus` is a single list, not a list of token lists, since we do not keep the sequence information in the following models.
Besides, we use character tokens to simplify the training in later sections.
-->

Chúng ta đóng gói tất cả các hàm trên thành hàm `load_corpus_time_machine`, trả về `corpus`, một danh sách các chỉ số của token, và bộ từ vựng `vocab` của kho ngữ liệu cỗ máy thời gian. 
Chúng ta đã sửa đổi một vài thứ ở đây là: `corpus` là một danh sách đơn nhất, không phải một danh sách các danh sách token, vì chúng ta không lưu các thông tin chuỗi trong các mô hình bên dưới. 
Bên cạnh đó, chúng ta sẽ sử dụng các token ký tự để đơn giản hóa việc huấn luyện mô hình trong các phần sau.



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

Chúng ta đã tiền xử lý các tài liệu văn bản bằng cách token hóa chúng thành các từ hoặc ký tự, và sau đó ánh xạ chúng thành các chỉ số tương ứng.



<!--
## Exercises
-->

## Bài tập

<!--
Tokenization is a key preprocessing step.
It varies for different languages.
Try to find another 3 commonly used methods to tokenize sentences.
-->

Token hóa là một bước tiền xử lý quan trọng. 
Mỗi ngôn ngữ có đều có các cách làm khác nhau. 
Hãy thử tìm thêm 3 phương pháp thường dùng để token hóa các câu. 


<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE KẾT THÚC =================================== -->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2363)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Lê Quang Nhật
* Phạm Hồng Vinh
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
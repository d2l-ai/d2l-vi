<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Machine Translation and the Dataset
-->

# Dịch Máy và Tập dữ liệu
:label:`sec_machine_translation`

<!--
So far we see how to use recurrent neural networks for language models, in which we predict the next token given all previous tokens in an article.
Now let us have a look at a different application, machine translation, whose predict output is no longer a single token, but a list of tokens.
-->

Cho đến nay ta đã thấy cách sử dụng các mạng nơ-ron truy hồi cho các mô hình ngôn ngữ, trong đó ta dự đoán token tiếp theo khi đã biết tất cả các token trước đó trong bài viết.
Bây giờ chúng ta sẽ xem xét một ứng dụng khác, với đầu ra dự đoán không còn là một token duy nhất mà là một chuỗi các token.

<!--
Machine translation (MT) refers to the automatic translation of a segment of text from one language to another.
Solving this problem with neural networks is often called neural machine translation (NMT).
Compared to language models (:numref:`sec_language_model`), in which the corpus only contains a single language, 
machine translation dataset has at least two languages, the source language and the target language.
In addition, each sentence in the source language is mapped to the according translation in the target language.
Therefore, the data preprocessing for machine translation data is different to the one for language models.
This section is dedicated to demonstrate how to pre-process such a dataset and then load into a set of minibatches.
-->

Dịch máy (_Machine translation_ - MT) đề cập đến việc dịch tự động một đoạn văn bản từ ngôn ngữ này sang ngôn ngữ khác.
Giải quyết bài toán này với các mạng nơ-ron thường được gọi là dịch máy nơ-ron (_neural machine translation_ - NMT).
So với các mô hình ngôn ngữ (:numref:`sec_language_model`), trong đó kho ngữ liệu chỉ chứa một ngôn ngữ duy nhất, bộ dữ liệu dịch máy có ít nhất hai ngôn ngữ, ngôn ngữ nguồn và ngôn ngữ đích.
Ngoài ra, mỗi câu trong ngôn ngữ nguồn được ánh xạ tới bản dịch tương ứng trong ngôn ngữ đích.
Do đó, cách tiền xử lý dữ liệu dịch máy sẽ khác so với dữ liệu của mô hình ngôn ngữ.
Phần này được dành riêng để trình bày cách tiền xử lý và nạp một tập dữ liệu như vậy vào các minibatch.


```{.python .input  n=1}
import d2l
from mxnet import np, npx, gluon
import os
npx.set_np()
```

<!--
## Reading and Preprocessing the Dataset
-->

## Đọc và Tiền Xử lý Dữ liệu

<!--
We first download a dataset that contains a set of English sentences with the corresponding French translations.
As can be seen that each line contains an English sentence with its French translation, which are separated by a `TAB`.
-->

Trước tiên ta tải xuống bộ dữ liệu có chứa một tập các câu tiếng Anh cùng với các bản dịch tiếng Pháp tương ứng.
Ta có thể thấy mỗi dòng chứa một câu tiếng Anh cùng với bản dịch tiếng Pháp tương ứng, được phân chia bằng một ký tự `TAB`.


```{.python .input  n=2}
# Saved in the d2l package for later use
d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')

# Saved in the d2l package for later use
def read_data_nmt():
    data_dir = d2l.download_extract('fra-eng')
    with open(os.path.join(data_dir, 'fra.txt'), 'r') as f:
        return f.read()

raw_text = read_data_nmt()
print(raw_text[0:106])
```

<!--
We perform several preprocessing steps on the raw text data, including ignoring cases, replacing UTF-8 non-breaking space with space, and adding space between words and punctuation marks.
-->

Ta sẽ thực hiện một số bước tiền xử lý trên dữ liệu văn bản thô, bao gồm chuyển đổi tất cả ký tự sang chữ thường, thay thế các ký tự khoảng trắng không ngắt (*non-breaking space*) UTF-8 bằng dấu cách, thêm dấu cách vào giữa các từ và các dấu câu.

```{.python .input  n=3}
# Saved in the d2l package for later use
def preprocess_nmt(text):
    def no_space(char, prev_char):
        return char in set(',.!') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

text = preprocess_nmt(raw_text)
print(text[0:95])
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Tokenization
-->

## *dịch tiêu đề phía trên*

<!--
Different to using character tokens in :numref:`sec_language_model`, here a token is either a word or a punctuation mark.
The following function tokenizes the text data to return `source` and `target`.
Each one is a list of token list, with `source[i]` is the $i^\mathrm{th}$ sentence in the source language and `target[i]` is the $i^\mathrm{th}$ sentence in the target language.
To make the latter training faster, we sample the first `num_examples` sentences pairs.
-->

*dịch đoạn phía trên*


```{.python .input  n=4}
# Saved in the d2l package for later use
def tokenize_nmt(text, num_examples=None):
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

source, target = tokenize_nmt(text)
source[0:3], target[0:3]
```

<!--
We visualize the histogram of the number of tokens per sentence in the following figure.
As can be seen, a sentence in average contains 5 tokens, and most of the sentences have less than 10 tokens.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
d2l.set_figsize((3.5, 2.5))
d2l.plt.hist([[len(l) for l in source], [len(l) for l in target]],
             label=['source', 'target'])
d2l.plt.legend(loc='upper right');
```

<!--
## Vocabulary
-->

## *dịch tiêu đề phía trên*

<!--
Since the tokens in the source language could be different to the ones in the target language, we need to build a vocabulary for each of them.
Since we are using words instead of characters  as tokens, it makes the vocabulary size significantly large.
Here we map every token that appears less than 3 times into the &lt;unk&gt; token :numref:`sec_text_preprocessing`.
In addition, we need other special tokens such as padding and sentence beginnings.
-->

*dịch đoạn phía trên*

```{.python .input  n=6}
src_vocab = d2l.Vocab(source, min_freq=3,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

<!--
## Loading the Dataset
-->

## *dịch tiêu đề phía trên*

<!--
In language models, each example is a `num_steps` length sequence from the corpus, which may be a segment of a sentence, or span over multiple sentences.
In machine translation, an example should contain a pair of source sentence and target sentence.
These sentences might have different lengths, while we need same length examples to form a minibatch.
-->

*dịch đoạn phía trên*

<!--
One way to solve this problem is that if a sentence is longer than `num_steps`, we trim its length, otherwise pad with a special &lt;pad&gt; token to meet the length.
Therefore we could transform any sentence to a fixed length.
-->

*dịch đoạn phía trên*


```{.python .input  n=7}
# Saved in the d2l package for later use
def truncate_pad(line, num_steps, padding_token):
    if len(line) > num_steps:
        return line[:num_steps]  # Trim
    return line + [padding_token] * (num_steps - len(line))  # Pad

truncate_pad(src_vocab[source[0]], 10, src_vocab['<pad>'])
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->


<!--
Now we can convert a list of sentences into an `(num_example, num_steps)` index array.
We also record the length of each sentence without the padding tokens, called *valid length*, which might be used by some models.
In addition, we add the special “&lt;bos&gt;” and “&lt;eos&gt;” tokens to the target sentences so that our model will know the signals for starting and ending predicting.
-->

*dịch đoạn phía trên*


```{.python .input  n=8}
# Saved in the d2l package for later use
def build_array(lines, vocab, num_steps, is_source):
    lines = [vocab[l] for l in lines]
    if not is_source:
        lines = [[vocab['<bos>']] + l + [vocab['<eos>']] for l in lines]
    array = np.array([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = (array != vocab['<pad>']).sum(axis=1)
    return array, valid_len
```


<!--
Then we can construct minibatches based on these arrays.
-->

*dịch đoạn phía trên*

<!--
## Putting All Things Together
-->

## *dịch tiêu đề phía trên*

<!--
Finally, we define the function `load_data_nmt` to return the data iterator with the vocabularies for source language and target language.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
# Saved in the d2l package for later use
def load_data_nmt(batch_size, num_steps, num_examples=1000):
    text = preprocess_nmt(read_data_nmt())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = d2l.Vocab(source, min_freq=3,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = d2l.Vocab(target, min_freq=3,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array(
        source, src_vocab, num_steps, True)
    tgt_array, tgt_valid_len = build_array(
        target, tgt_vocab, num_steps, False)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = d2l.load_array(data_arrays, batch_size)
    return src_vocab, tgt_vocab, data_iter
```

<!--
Let us read the first batch.
-->

*dịch đoạn phía trên*


```{.python .input  n=10}
src_vocab, tgt_vocab, train_iter = load_data_nmt(batch_size=2, num_steps=8)
for X, X_vlen, Y, Y_vlen in train_iter:
    print('X:', X.astype('int32'))
    print('Valid lengths for X:', X_vlen)
    print('Y:', Y.astype('int32'))
    print('Valid lengths for Y:', Y_vlen)
    break
```

<!--
## Summary
-->

## Tóm tắt

<!--
* Machine translation (MT) refers to the automatic translation of a segment of text from one language to another.
* We read, preprocess, and tokenize the datasets from both source language and target language.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
Find a machine translation dataset online and process it.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC =================================== -->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2396)
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
* Nguyễn Duy Du
* Nguyễn Văn Quang

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Machine Translation and the Dataset
-->

# *dịch tiêu đề phía trên*
:label:`sec_machine_translation`

<!--
So far we see how to use recurrent neural networks for language models, in which we predict the next token given all previous tokens in an article.
Now let us have a look at a different application, machine translation, whose predict output is no longer a single token, but a list of tokens.
-->

*dịch đoạn phía trên*

<!--
Machine translation (MT) refers to the automatic translation of a segment of text from one language to another.
Solving this problem with neural networks is often called neural machine translation (NMT).
Compared to language models (:numref:`sec_language_model`), in which the corpus only contains a single language, 
machine translation dataset has at least two languages, the source language and the target language.
In addition, each sentence in the source language is mapped to the according translation in the target language.
Therefore, the data preprocessing for machine translation data is different to the one for language models.
This section is dedicated to demonstrate how to pre-process such a dataset and then load into a set of minibatches.
-->

*dịch đoạn phía trên*


```{.python .input  n=1}
import d2l
from mxnet import np, npx, gluon
import os
npx.set_np()
```

<!--
## Reading and Preprocessing the Dataset
-->

## *dịch tiêu đề phía trên*

<!--
We first download a dataset that contains a set of English sentences with the corresponding French translations.
As can be seen that each line contains an English sentence with its French translation, which are separated by a `TAB`.
-->

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*

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

## Token hóa

<!--
Different to using character tokens in :numref:`sec_language_model`, here a token is either a word or a punctuation mark.
The following function tokenizes the text data to return `source` and `target`.
Each one is a list of token list, with `source[i]` is the $i^\mathrm{th}$ sentence in the source language and `target[i]` is the $i^\mathrm{th}$ sentence in the target language.
To make the latter training faster, we sample the first `num_examples` sentences pairs.
-->

Khác với việc sử dụng token trong :numref:`sec_language_model`, ở đây token là một từ hoặc dấu câu.
Hàm sau đây sẽ token hóa dữ liệu văn bản để trả về `source` và `target`.
Mỗi đầu ra là một danh sách các token, với `source [i]` là câu thứ $i$ trong ngôn ngữ nguồn và `target [i]` là câu thứ $i$ câu trong ngôn ngữ đích.
Để việc huấn luyện sau này nhanh hơn, chúng ta sẽ lấy mẫu `num_examples` cặp câu đầu tiên.


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

Dưới đây là biểu đồ tần suất của số lượng token cho mỗi câu.
Có thể thấy, trung bình một câu chứa 5 token và hầu hết các câu có ít hơn 10 token.

```{.python .input  n=5}
d2l.set_figsize((3.5, 2.5))
d2l.plt.hist([[len(l) for l in source], [len(l) for l in target]],
             label=['source', 'target'])
d2l.plt.legend(loc='upper right');
```

<!--
## Vocabulary
-->

## Từ vựng

<!--
Since the tokens in the source language could be different to the ones in the target language, we need to build a vocabulary for each of them.
Since we are using words instead of characters  as tokens, it makes the vocabulary size significantly large.
Here we map every token that appears less than 3 times into the &lt;unk&gt; token :numref:`sec_text_preprocessing`.
In addition, we need other special tokens such as padding and sentence beginnings.
-->

Vì các token trong ngôn ngữ nguồn có thể khác với các token trong ngôn ngữ đích, ta cần xây dựng một bộ từ vựng cho mỗi ngôn ngữ.
Do ta đang sử dụng các từ thay vì các ký tự để làm token, kích thước bộ từ vựng sẽ lớn hơn đáng kể.
Ở đây ta sẽ ánh xạ mọi token xuất hiện ít hơn 3 lần vào token &lt;unk&gt; :numref:`sec_text_preprocessing`.
Ngoài ra, ta cần các token đặc biệt khác như phần đệm hay phần bắt đầu câu.

```{.python .input  n=6}
src_vocab = d2l.Vocab(source, min_freq=3,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
len(src_vocab)
```

<!--
## Loading the Dataset
-->

## Đọc Dữ liệu

<!--
In language models, each example is a `num_steps` length sequence from the corpus, which may be a segment of a sentence, or span over multiple sentences.
In machine translation, an example should contain a pair of source sentence and target sentence.
These sentences might have different lengths, while we need same length examples to form a minibatch.
-->

Trong các mô hình ngôn ngữ, mỗi mẫu là một chuỗi có độ dài `num_steps` từ kho ngữ liệu, có thể là một phân đoạn của một câu hoặc trải dài trên nhiều câu.
Trong dịch máy, một mẫu bao gồm một cặp câu nguồn và câu đích.
Những câu này có thể có độ dài khác nhau, trong khi đó ta cần các mẫu có cùng độ dài để tạo thành một minibatch.

<!--
One way to solve this problem is that if a sentence is longer than `num_steps`, we trim its length, otherwise pad with a special &lt;pad&gt; token to meet the length.
Therefore we could transform any sentence to a fixed length.
-->

Một cách để giải quyết vấn đề này là nếu một câu dài hơn `num_steps`, ta sẽ cắt bớt độ dài của nó, ngược lại nếu một câu ngắn hơn `num_steps`, thì ta sẽ đệm với một token đặc biệt &lt;pad&gt; để đáp ứng độ dài.
Do vậy, với cách trên, chúng ta có thể chuyển bất cứ câu nào về cùng một độ dài cố định.

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
*

<!-- Phần 2 -->
* Nguyễn Duy Du
* Nguyễn Văn Quang

<!-- Phần 3 -->
*

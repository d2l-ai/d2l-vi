<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Subword Embedding
-->

# Nhúng từ tố
:label:`sec_fasttext`


<!--
English words usually have internal structures and formation methods.
For example, we can deduce the relationship between "dog", "dogs", and "dogcatcher" by their spelling.
All these words have the same root, "dog", but they use different suffixes to change the meaning of the word.
Moreover, this association can be extended to other words.
For example, the relationship between "dog" and "dogs" is just like the relationship between "cat" and "cats".
The relationship between "boy" and "boyfriend" is just like the relationship between "girl" and "girlfriend".
This characteristic is not unique to English.
In French and Spanish, a lot of verbs can have more than 40 different forms depending on the context.
In Finnish, a noun may have more than 15 forms.
In fact, morphology, which is an important branch of linguistics, studies the internal structure and formation of words.
-->

Các từ tiếng Anh thường có cấu những trúc nội tại và phương thức cấu thành.
Chẳng hạn, ta có thể suy ra mối quan hệ giữa các từ "dog", "dogs" và "dogcatcher" thông qua cách viết của chúng.
Tất cả các từ đó có cùng từ gốc là "dog" nhưng có hậu tố khác nhau làm thay đổi nghĩa của từ.
Hơn nữa, sự liên kết này có thể được mở rộng ra đối với các từ khác.
Chẳng hạn, mối quan hệ giữa từ "dog" và "dogs" đơn giản giống như mối quan hệ giữa từ "cat" và "cats".
Mối quan hệ giữa từ "boy" và "boyfriend" đơn giản giống mối quan hệ giữa từ "girl" và "girlfriend".
Đặc tính này không phải là duy nhất trong tiếng Anh.
Trong tiếng Pháp và Tây Ban Nha, rất nhiều động từ có thể có hơn 40 dạng khác nhau tùy thuộc vào ngữ cảnh.
Trong tiếng Phần Lan, một danh từ có thể có hơn 15 dạng.
Thật vậy, hình thái học (*morphology*) là một nhánh quan trọng của ngôn ngữ học chuyên nghiên cứu về cấu trúc và hình thái của các từ. 


<!--
## fastText
-->

## fastText


<!--
In word2vec, we did not directly use morphology information.
In both the skip-gram model and continuous bag-of-words model, we use different vectors to represent words with different forms.
For example, "dog" and "dogs" are represented by two different vectors, while the relationship between these two vectors is not directly represented in the model.
In view of this, fastText :cite:`Bojanowski.Grave.Joulin.ea.2017` proposes the method of subword embedding, 
thereby attempting to introduce morphological information in the skip-gram model in word2vec.
-->

Trong word2vec, ta không trực tiếp sử dụng thông tin hình thái học.
Trong cả mô hình skip-gram và bag-of-word liên tục, ta sử dụng các vector khác nhau để biểu diễn các từ ở các dạng khác nhau.
Chẳng hạn, "dog" và "dogs" được biểu diễn bởi hai vector khác nhau, trong khi mối quan hệ giữa hai vector đó không biểu thị trực tiếp trong mô hình. 
Từ quan điểm này, fastText :cite:`Bojanowski.Grave.Joulin.ea.2017` đề xuất phương thức embedding từ con,
thông qua việc thực hiện đưa thông tin hình thái học vào trong mô hình skip-gram trong word2vec.


<!--
In fastText, each central word is represented as a collection of subwords.
Below we use the word "where" as an example to understand how subwords are formed.
First, we add the special characters “&lt;” and “&gt;” at the beginning and end of the word to distinguish the subwords used as prefixes and suffixes.
Then, we treat the word as a sequence of characters to extract the $n$-grams.
For example, when $n=3$, we can get all subwords with a length of $3$:
-->

Trong fastText, mỗi từ trung tâm được biểu diễn như một tập hợp của các từ con.
Dưới đây ta sử dụng từ "where" làm ví dụ để hiểu làm thế nào các từ tố được tạo thành.
Trước hết, ta thêm một số ký tự đặc biệt “&lt;” và “&gt;” vào phần bắt đầu và kết thúc của từ để phân biệt các từ con được dùng làm tiền tố và hậu tố.
Rồi ta sẽ xem từ này như một chuỗi các ký tự để trích xuất $n$-grams.
Chẳng hạn, khi $n=3$, ta có thể nhận tất cả từ tố với chiều dài là $3$:

$$\textrm{"<wh"}, \ \textrm{"whe"}, \ \textrm{"her"}, \ \textrm{"ere"}, \ \textrm{"re>"},$$


<!--
and the special subword $\textrm{"<where>"}$.
-->

và từ tố đặc biệt  $\textrm{"<where>"}$.


<!--
In fastText, for a word $w$, we record the union of all its subwords with length of $3$ to $6$ and special subwords as $\mathcal{G}_w$.
Thus, the dictionary is the union of the collection of subwords of all words.
Assume the vector of the subword $g$ in the dictionary is $\mathbf{z}_g$.
Then, the central word vector $\mathbf{u}_w$ for the word $w$ in the skip-gram model can be expressed as
-->

Trong fastText, với một từ $w$, ta ghi tập hợp của tất cả các từ con của nó với chiều dài từ $3$ đến $6$ và các từ con đặc biệt là $\mathcal{G}_w$.
Do đó, từ điển này là tập hợp các từ con của tất cả các từ.
Giả sử vector của từ con $g$ trong từ điển này là $\mathbf{z}_g$.
Thì vector từ trung tâm $\mathbf{u}_w$ cho từ $w$ trong mô hình skip-gram có thể biểu diễn là

$$\mathbf{u}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$


<!--
The rest of the fastText process is consistent with the skip-gram model, so it is not repeated here.
As we can see, compared with the skip-gram model, the dictionary in fastText is larger, resulting in more model parameters.
Also, the vector of one word requires the summation of all subword vectors, which results in higher computation complexity.
However, we can obtain better vectors for more uncommon complex words, even words not existing in the dictionary, by looking at other words with similar structures.
-->

Phần còn lại tiến trình xử lý trong fastText đồng nhất với mô hình skip-gram, nên ta không mô tả lại ở đây.
Như chúng ta có thể thấy, so sánh với mô hình skip-gram, từ điển trong fastText lớn hơn dẫn tới số tham số của mô hình nhiều hơn.
Hơn nữa, vector của một từ đòi hỏi tính tổng của tất cả vector từ con dẫn tới kết quả làm độ phức tạp tính toán cao hơn.
Tuy nhiên, ta có thể thu được các vector tốt hơn cho nhiều từ phức hợp không thông dụng hơn, thậm chí các từ không hiện diện trong từ điển này, nhờ tham chiếu các từ khác có cấu trúc tương tự.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Byte Pair Encoding
-->

## *dịch đoạn phía trên*
:label:`subsec_Byte_Pair_Encoding`


<!--
In fastText, all the extracted subwords have to be of the specified lengths, such as $3$ to $6$, thus the vocabulary size cannot be predefined.
To allow for variable-length subwords in a fixed-size vocabulary, we can apply a compression algorithm
called *byte pair encoding* (BPE) to extract subwords :cite:`Sennrich.Haddow.Birch.2015`.
-->

*dịch đoạn phía trên*


<!--
Byte pair encoding performs a statistical analysis of the training dataset to discover common symbols within a word, such as consecutive characters of arbitrary length.
Starting from symbols of length $1$, byte pair encoding iteratively merges the most frequent pair of consecutive symbols to produce new longer symbols.
Note that for efficiency, pairs crossing word boundaries are not considered.
In the end, we can use such symbols as subwords to segment words.
Byte pair encoding and its variants has been used for input representations in popular natural language processing pretraining models 
such as GPT-2 :cite:`Radford.Wu.Child.ea.2019` and RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`.
In the following, we will illustrate how byte pair encoding works.
-->

*dịch đoạn phía trên*


<!--
First, we initialize the vocabulary of symbols as all the English lowercase characters, a special end-of-word symbol `'_'`, and a special unknown symbol `'[UNK]'`.
-->

*dịch đoạn phía trên*


```{.python .input}
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```


<!--
Since we do not consider symbol pairs that cross boundaries of words,
we only need a dictionary `raw_token_freqs` that maps words to their frequencies (number of occurrences) in a dataset.
Note that the special symbol `'_'` is appended to each word so that we can easily recover a word sequence (e.g., "a taller man")
from a sequence of output symbols ( e.g., "a_ tall er_ man").
Since we start the merging process from a vocabulary of only single characters and special symbols,
space is inserted between every pair of consecutive characters within each word (keys of the dictionary `token_freqs`).
In other words, space is the delimiter between symbols within a word.
-->

*dịch đoạn phía trên*


```{.python .input}
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```


<!--
We define the following `get_max_freq_pair` function that 
returns the most frequent pair of consecutive symbols within a word,
where words come from keys of the input dictionary `token_freqs`.
-->

*dịch đoạn phía trên*


```{.python .input}
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```


<!--
As a greedy approach based on frequency of consecutive symbols,
byte pair encoding will use the following `merge_symbols` function to merge the most frequent pair of consecutive symbols to produce new symbols.
-->

*dịch đoạn phía trên*


```{.python .input}
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
Now we iteratively perform the byte pair encoding algorithm over the keys of the dictionary `token_freqs`.
In the first iteration, the most frequent pair of consecutive symbols are `'t'` and `'a'`, thus byte pair encoding merges them to produce a new symbol `'ta'`.
In the second iteration, byte pair encoding continues to merge `'ta'` and `'l'` to result in another new symbol `'tal'`.
-->

*dịch đoạn phía trên*


```{.python .input}
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```


<!--
After 10 iterations of byte pair encoding, we can see that list `symbols` now contains 10 more symbols that are iteratively merged from other symbols.
-->

*dịch đoạn phía trên*


```{.python .input}
print(symbols)
```


<!--
For the same dataset specified in the keys of the dictionary `raw_token_freqs`, 
each word in the dataset is now segmented by subwords "fast_", "fast", "er_", "tall_", and "tall"
as a result of the byte pair encoding algorithm.
For instance, words "faster_" and "taller_" are segmented as "fast er_" and "tall er_", respectively.
-->

*dịch đoạn phía trên*


```{.python .input}
print(list(token_freqs.keys()))
```


<!--
Note that the result of byte pair encoding depends on the dataset being used.
We can also use the subwords learned from one dataset to segment words of another dataset.
As a greedy approach, the following `segment_BPE` function tries to break words into the longest possible subwords from the input argument `symbols`.
-->

*dịch đoạn phía trên*


```{.python .input}
def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # Segment token with the longest possible subwords from symbols
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs
```


<!--
In the following, we use the subwords in list `symbols`, which is learned from the aforementioned dataset,
to segment `tokens` that represent another dataset.
-->

*dịch đoạn phía trên*


```{.python .input}
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

## Tóm tắt

<!--
* FastText proposes a subword embedding method. Based on the skip-gram model in word2vec, it represents the central word vector as the sum of the subword vectors of the word.
* Subword embedding utilizes the principles of morphology, which usually improves the quality of representations of uncommon words.
* Byte pair encoding performs a statistical analysis of the training dataset to discover common symbols within a word.
As a greedy approach, byte pair encoding iteratively merges the most frequent pair of consecutive symbols.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. When there are too many subwords (for example, 6 words in English result in about $3\times 10^8$ combinations), what problems arise?
Can you think of any methods to solve them? Hint: Refer to the end of section 3.2 of the fastText paper[1].
2. How can you design a subword embedding model based on the continuous bag-of-words model?
3. To get a vocabulary of size $m$, how many merging operations are needed when the initial symbol vocabulary size is $n$?
4. How can we extend the idea of byte pair encoding to extract phrases?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/386)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.
Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Mai Hoàng Long

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

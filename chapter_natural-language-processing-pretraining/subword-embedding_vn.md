<!--
# Subword Embedding
-->

# Embedding từ con
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

Các từ tiếng Anh thường có những cấu những trúc nội tại và phương thức cấu thành.
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
Trong cả mô hình skip-gram và túi từ (*bag-of-word*) liên tục, ta sử dụng các vector khác nhau để biểu diễn các từ ở các dạng khác nhau.
Chẳng hạn, "dog" và "dogs" được biểu diễn bởi hai vector khác nhau, trong khi mối quan hệ giữa hai vector đó không biểu thị trực tiếp trong mô hình. 
Từ quan điểm này, fastText :cite:`Bojanowski.Grave.Joulin.ea.2017` đề xuất phương thức embedding từ con (*subword embedding*),
thông qua việc thực hiện đưa thông tin hình thái học vào trong mô hình skip-gram trong word2vec.


<!--
In fastText, each central word is represented as a collection of subwords.
Below we use the word "where" as an example to understand how subwords are formed.
First, we add the special characters “&lt;” and “&gt;” at the beginning and end of the word to distinguish the subwords used as prefixes and suffixes.
Then, we treat the word as a sequence of characters to extract the $n$-grams.
For example, when $n=3$, we can get all subwords with a length of $3$:
-->

Trong fastText, mỗi từ trung tâm được biểu diễn như một tập hợp của các từ con.
Dưới đây ta sử dụng từ "where" làm ví dụ để hiểu cách các từ tố được tạo thành.
Trước hết, ta thêm một số ký tự đặc biệt “&lt;” và “&gt;” vào phần bắt đầu và kết thúc của từ để phân biệt các từ con được dùng làm tiền tố và hậu tố.
Rồi ta sẽ xem từ này như một chuỗi các ký tự để trích xuất $n$-grams.
Chẳng hạn, khi $n=3$, ta có thể nhận tất cả từ tố với chiều dài là $3$:

$$\textrm{"<wh"}, \ \textrm{"whe"}, \ \textrm{"her"}, \ \textrm{"ere"}, \ \textrm{"re>"},$$


<!--
and the special subword $\textrm{"<where>"}$.
-->

và từ con đặc biệt $\textrm{"<where>"}$.


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

Phần còn lại của tiến trình xử lý trong fastText đồng nhất với mô hình skip-gram, vì vậy ta không mô tả lại ở đây.
Như chúng ta có thể thấy, so sánh với mô hình skip-gram, từ điển của fastText lớn hơn dẫn tới nhiều tham số mô hình hơn.
Hơn nữa, vector của một từ đòi hỏi tính tổng của tất cả vector từ con dẫn tới độ phức tạp tính toán cao hơn.
Tuy nhiên, ta có thể thu được các vector tốt hơn cho nhiều từ phức hợp ít thông dụng, 
thậm chí cho cả các từ không hiện diện trong từ điển này nhờ tham chiếu tới các từ khác có cấu trúc tương tự.


<!--
## Byte Pair Encoding
-->

## Mã hoá cặp byte
:label:`subsec_Byte_Pair_Encoding`


<!--
In fastText, all the extracted subwords have to be of the specified lengths, such as $3$ to $6$, thus the vocabulary size cannot be predefined.
To allow for variable-length subwords in a fixed-size vocabulary, we can apply a compression algorithm
called *byte pair encoding* (BPE) to extract subwords :cite:`Sennrich.Haddow.Birch.2015`.
-->

Trong fastText, tất cả các từ con được trích xuất phải nằm trong khoảng độ dài cho trước, 
ví dụ như từ $3$ đến $6$, do đó kích thước bộ từ vựng không thể được xác định trước.
Để cho phép các từ con có độ dài biến thiên trong bộ từ vựng có kích thước cố định, 
chúng ta có thể áp dụng thuật toán nén gọi là *mã hoá cặp byte* (*Byte Pair Encoding* -BPE) để trích xuất các từ con :cite:`Sennrich.Haddow.Birch.2015`. 


<!--
Byte pair encoding performs a statistical analysis of the training dataset to discover common symbols within a word, such as consecutive characters of arbitrary length.
Starting from symbols of length $1$, byte pair encoding iteratively merges the most frequent pair of consecutive symbols to produce new longer symbols.
Note that for efficiency, pairs crossing word boundaries are not considered.
In the end, we can use such symbols as subwords to segment words.
Byte pair encoding and its variants has been used for input representations in popular natural language processing pretraining models 
such as GPT-2 :cite:`Radford.Wu.Child.ea.2019` and RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`.
In the following, we will illustrate how byte pair encoding works.
-->

Mã hóa cặp byte thực hiện phân tích thống kê tập dữ liệu huấn luyện để tìm các ký hiệu chung trong một từ, chẳng hạn như các ký tự liên tiếp có độ dài tùy ý. 
Bắt đầu từ các ký hiệu có độ dài bằng $1$, mã hóa cặp byte lặp đi lặp lại việc gộp các cặp ký hiệu liên tiếp thường gặp nhất để tạo ra các ký hiệu mới dài hơn. 
Lưu ý rằng để tăng hiệu năng, các cặp vượt qua ranh giới từ sẽ không được xét.
Cuối cùng, chúng ta có thể sử dụng các ký hiệu đó như từ con để phân đoạn các từ.
Mã hóa cặp byte và các biến thể của nó đã được sử dụng để biểu diễn đầu vào trong các mô hình tiền huấn luyện cho 
xử lý ngôn ngữ tự nhiên phổ biến như GPT-2 :cite:`Radford.Wu.Child.ea.2019` và RoBERTa :cite:`Liu.Ott.Goyal.ea.2019`. 
Tiếp theo, chúng tôi sẽ minh hoạ cách hoạt động của mã hoá cặp byte.


<!--
First, we initialize the vocabulary of symbols as all the English lowercase characters, a special end-of-word symbol `'_'`, and a special unknown symbol `'[UNK]'`.
-->

Đầu tiên, ta khởi tạo bộ từ vựng của các ký hiệu dưới dạng tất cả các ký tự viết thường trong tiếng Anh 
và hai ký hiệu đặc biệt: ký hiệu kết thúc của từ `'_'` , và ký hiệu không xác định `'[UNK]'`. 


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

Vì không xét các cặp ký hiệu vượt qua ranh giới của các từ,
chúng ta chỉ cần một từ điển `raw_token_freqs` ánh xạ các từ tới tần suất của chúng (số lần xuất hiện) trong một tập dữ liệu.
Lưu ý rằng ký hiệu đặc biệt `'_'` được thêm vào mỗi từ để có thể dễ dàng khôi phục chuỗi từ (ví dụ: "a taller man")
từ chuỗi ký hiệu đầu ra (ví dụ: "a_ tall er_ man").
Vì chúng ta bắt đầu quá trình gộp một từ vựng chỉ gồm các ký tự đơn và các ký hiệu đặc biệt, 
khoảng trắng được chèn giữa mọi cặp ký tự liên tiếp trong mỗi từ (các khóa của từ điển `token_freqs`). 
Nói cách khác, khoảng trắng là ký tự phân cách (*delimiter*) giữa các ký hiệu trong một từ. 


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
Chúng ta định nghĩa hàm `get_max_freq_pair` trả về cặp ký hiệu liên tiếp thường gặp nhất trong một từ, với từ là các khóa của từ điển đầu vào `token_freqs`. 


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

Là một thuật toán tham lam dựa trên tần suất của các ký hiệu liên tiếp nhau, mã hoá cặp byte sẽ dùng hàm `merge_symbols` để gộp cặp ký hiệu thường gặp nhất để tạo ra những ký hiệu mới. 


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


<!--
Now we iteratively perform the byte pair encoding algorithm over the keys of the dictionary `token_freqs`.
In the first iteration, the most frequent pair of consecutive symbols are `'t'` and `'a'`, thus byte pair encoding merges them to produce a new symbol `'ta'`.
In the second iteration, byte pair encoding continues to merge `'ta'` and `'l'` to result in another new symbol `'tal'`.
-->

Bây giờ ta thực hiện vòng lặp giải thuật biểu diễn cặp byte với các khóa của từ điển `token_freqs`. 
Ở vòng lặp đầu tiên, cặp biểu tượng liền kề có tần suất cao nhất là `'t'` và `'a'`, do đó biểu diễn cặp byte ghép chúng lại để tạo ra một biểu tượng mới là `'ta'`. 
Ở vòng lặp thứ hai, biểu diễn cặp byte tiếp tục ghép 2 biểu tượng `'ta'` và `'l'` tạo ra một biểu tượng mới khác là `'tal'`. 


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

Sau 10 vòng lặp biểu diễn cặp byte, ta có thể thấy là danh sách `symbols` lúc này chứa hơn 10 biểu tượng đã được lần lượt ghép từ các biểu tượng khác. 


```{.python .input}
print(symbols)
```


<!--
For the same dataset specified in the keys of the dictionary `raw_token_freqs`, 
each word in the dataset is now segmented by subwords "fast_", "fast", "er_", "tall_", and "tall"
as a result of the byte pair encoding algorithm.
For instance, words "faster_" and "taller_" are segmented as "fast er_" and "tall er_", respectively.
-->

Với cùng tập dữ liệu đặc tả trong các khóa của từ điển `raw_token_freqs`, mỗi từ trong tập dữ liệu này 
bây giờ được phân đoạn bởi các từ con là "fast_", "fast", "er_", "tall_", và "tall" theo giải thuật biểu diễn cặp byte.
Chẳng hạn, từ "faster_" và từ "taller_" được phân đoạn lần lượt là "fast er_" và "tall er_".


```{.python .input}
print(list(token_freqs.keys()))
```


<!--
Note that the result of byte pair encoding depends on the dataset being used.
We can also use the subwords learned from one dataset to segment words of another dataset.
As a greedy approach, the following `segment_BPE` function tries to break words into the longest possible subwords from the input argument `symbols`.
-->

Chú ý là kết quả của biểu diễn cặp byte tùy thuộc vào tập dữ liệu đang được sử dụng.
Ta cũng có thể dùng các từ con đã học từ một tập dữ liệu để phân đoạn các từ của một tập dữ liệu khác.
Với cách tiếp cận tham lam, hàm `segment_BPE` sau đây cố gắng tách các từ thành các từ con dài nhất có thể từ đối số đầu vào `symbols`.


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

Trong phần tiếp theo, ta sử dụng các từ con trong danh sách `symbols` đã được học từ tập dữ liệu ở trên để phân đoạn các `tokens` biểu diễn tập dữ liệu khác.


```{.python .input}
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```


## Tóm tắt

<!--
* FastText proposes a subword embedding method. Based on the skip-gram model in word2vec, it represents the central word vector as the sum of the subword vectors of the word.
* Subword embedding utilizes the principles of morphology, which usually improves the quality of representations of uncommon words.
* Byte pair encoding performs a statistical analysis of the training dataset to discover common symbols within a word.
As a greedy approach, byte pair encoding iteratively merges the most frequent pair of consecutive symbols.
-->

* FastText đề xuất phương pháp embedding cho từ con. Dựa trên mô hình skip-gram trong word2vec, phương pháp này biểu diễn vector từ trung tâm thành tổng các vector từ con của từ đó.
* Embedding cho từ con sử dụng nguyên tắc trong hình thái học, thường giúp cải thiện chất lượng biểu diễn của các từ ít gặp.
* Mã hoá cặp byte thực hiện phân tích thống kê trên tập dữ liệu huấn luyện để phát hiện các ký hiệu chung trong một từ.
Là một giải thuật tham lam, mã hoá cặp byte lần lượt gộp các cặp ký hiệu liên tiếp thường gặp nhất lại với nhau.


## Bài tập

<!--
1. When there are too many subwords (for example, 6 words in English result in about $3\times 10^8$ combinations), what problems arise?
Can you think of any methods to solve them? Hint: Refer to the end of section 3.2 of the fastText paper[1].
2. How can you design a subword embedding model based on the continuous bag-of-words model?
3. To get a vocabulary of size $m$, how many merging operations are needed when the initial symbol vocabulary size is $n$?
4. How can we extend the idea of byte pair encoding to extract phrases?
-->

1. Khi có quá nhiều từ con (ví dụ, 6 từ trong tiếng Anh có thể tạo ra $3\times 10^8$ các tổ hợp khác nhau), vấn đề gì sẽ xảy ra?
Bạn có thể giải quyết vấn đề trên không? Gợi ý: Tham khảo đoạn cuối phần 3.2 của bài báo fastText [1].
2. Làm sao để thiết kế một mô hình embedding cho từ con dựa trên mô hình túi từ liên tục CBOW ?
3. Để thu được bộ từ vựng có kích thước $m$, bao nhiêu phép gộp cần được thực hiện khi bộ từ vựng ký hiệu ban đầu có kích thước là $n$? 
4. Ta có thể mở rộng ý tưởng của thuật toán mã hoá cặp byte để trích xuất các cụm từ bằng cách nào?


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/386)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Nguyễn Mai Hoàng Long
* Phạm Đăng Khoa
* Nguyễn Văn Cường

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 30/06/2020)*

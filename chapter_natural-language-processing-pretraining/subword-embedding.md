# Subword Nhúng
:label:`sec_fasttext`

Trong tiếng Anh, các từ như “help”, “helped”, và “help” là những dạng uốn cong của cùng một từ “help”. Mối quan hệ giữa “chó” và “chó” cũng giống như mối quan hệ giữa “mèo” và “mèo”, và mối quan hệ giữa “cậu bé” và “bạn trai” cũng giống như giữa “cô gái” và “bạn gái”. Trong các ngôn ngữ khác như tiếng Pháp và tiếng Tây Ban Nha, nhiều động từ có trên 40 dạng uốn, trong khi trong tiếng Phần Lan, một danh từ có thể có tới 15 trường hợp. Trong ngôn ngữ học, hình thái học nghiên cứu hình thành từ và các mối quan hệ từ. Tuy nhiên, cấu trúc nội bộ của các từ không được khám phá trong word2vec cũng như trong Glove. 

## Mô hình fastText

Nhớ lại cách các từ được biểu diễn trong word2vec. Trong cả mô hình skip-gram và mô hình túi-of-words liên tục, các dạng uốn khác nhau của cùng một từ được biểu diễn trực tiếp bởi các vectơ khác nhau mà không có tham số chia sẻ. Để sử dụng thông tin hình thái học, mô hình *fastText* đề xuất cách tiếp cận nhúng từ con*, trong đó một từ con là ký tự $n$-gram :cite:`Bojanowski.Grave.Joulin.ea.2017`. Thay vì học biểu diễn vectơ cấp từ, fastText có thể được coi là biểu đồ bỏ qua cấp từ phụ, trong đó mỗi từ *center* được biểu diễn bằng tổng các vectơ từ con của nó. 

Hãy để chúng tôi minh họa làm thế nào để có được các từ con cho mỗi từ trung tâm trong fastText bằng cách sử dụng từ “ở đâu”. Đầu tiên, thêm các ký tự đặc biệt “<” and “>” ở đầu và cuối của từ để phân biệt tiền tố và hậu tố với các từ con khác. Sau đó, trích xuất ký tự $n$-gram từ từ. Ví dụ, khi $n=3$, ta có được tất cả các từ con có độ dài 3: “<wh”, “whe”, “her”, “ere”, “re>”, và từ con đặc biệt "<where>”. 

Trong fastText, đối với bất kỳ từ nào $w$, biểu thị bằng $\mathcal{G}_w$ sự kết hợp của tất cả các từ con của nó có độ dài giữa 3 và 6 và từ con đặc biệt của nó. Từ vựng là sự kết hợp của các từ con của tất cả các từ. Để $\mathbf{z}_g$ là vectơ của subword $g$ trong từ điển, vector $\mathbf{v}_w$ cho từ $w$ như một từ trung tâm trong mô hình skip-gram là tổng của vectơ từ con của nó: 

$$\mathbf{v}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

Phần còn lại của fastText giống như mô hình skip-gram. So với mô hình skip-gram, từ vựng trong fastText lớn hơn, dẫn đến nhiều tham số mô hình hơn. Bên cạnh đó, để tính toán biểu diễn của một từ, tất cả các vectơ từ con của nó phải được tóm tắt, dẫn đến độ phức tạp tính toán cao hơn. Tuy nhiên, nhờ các tham số được chia sẻ từ các từ con giữa các từ có cấu trúc tương tự, các từ hiếm và thậm chí các từ ngoài từ vựng có thể có được biểu diễn vector tốt hơn trong fastText. 

## Mã hóa cặp byte
:label:`subsec_Byte_Pair_Encoding`

Trong fastText, tất cả các từ con được trích xuất phải có độ dài được chỉ định, chẳng hạn như $3$ đến $6$, do đó kích thước từ vựng không thể được xác định trước. Để cho phép các từ con có độ dài biến đổi trong từ vựng có kích thước cố định, chúng ta có thể áp dụng một thuật toán nén gọi là mã hóa cặp byte * (BPE) để trích xuất các từ con :cite:`Sennrich.Haddow.Birch.2015`. 

Mã hóa cặp byte thực hiện phân tích thống kê của tập dữ liệu đào tạo để khám phá các ký hiệu phổ biến trong một từ, chẳng hạn như các ký tự liên tiếp có độ dài tùy ý. Bắt đầu từ các ký hiệu có độ dài 1, mã hóa cặp byte kết hợp lặp đi lặp lại cặp ký hiệu liên tiếp thường xuyên nhất để tạo ra các ký hiệu dài hơn mới. Lưu ý rằng đối với hiệu quả, các cặp vượt qua ranh giới từ không được xem xét. Cuối cùng, chúng ta có thể sử dụng các ký hiệu như các từ con để phân đoạn các từ. Mã hóa cặp byte và các biến thể của nó đã được sử dụng để biểu diễn đầu vào trong các mô hình pretraining xử lý ngôn ngữ tự nhiên phổ biến như GPT-2 :cite:`Radford.Wu.Child.ea.2019` và roberTa :cite:`Liu.Ott.Goyal.ea.2019`. Sau đây, chúng tôi sẽ minh họa cách mã hóa cặp byte hoạt động. 

Đầu tiên, chúng ta khởi tạo từ vựng của các ký hiệu như tất cả các ký tự chữ thường tiếng Anh, một ký hiệu cuối từ đặc biệt `'_'` và một ký hiệu không xác định đặc biệt `'[UNK]'`.

```{.python .input}
#@tab all
import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
           '_', '[UNK]']
```

Vì chúng ta không xem xét các cặp biểu tượng vượt qua ranh giới của các từ, chúng ta chỉ cần một từ điển `raw_token_freqs` ánh xạ các từ đến tần số của chúng (số lần xuất hiện) trong một tập dữ liệu. Lưu ý rằng ký hiệu đặc biệt `'_'` được gắn vào mỗi từ để chúng ta có thể dễ dàng khôi phục một chuỗi từ (ví dụ, “một người đàn ông cao hơn”) từ một chuỗi các ký hiệu đầu ra (ví dụ: “a_ tall er_ man”). Vì chúng ta bắt đầu quá trình hợp nhất từ một từ vựng chỉ gồm các ký tự duy nhất và các ký hiệu đặc biệt, khoảng trắng được chèn giữa mỗi cặp ký tự liên tiếp trong mỗi từ (các phím của từ điển `token_freqs`). Nói cách khác, không gian là dấu phân cách giữa các ký hiệu trong một từ.

```{.python .input}
#@tab all
raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}
token_freqs = {}
for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]
token_freqs
```

Chúng tôi xác định hàm `get_max_freq_pair` sau trả về cặp ký hiệu liên tiếp thường xuyên nhất trong một từ, trong đó các từ đến từ khóa của từ điển đầu vào `token_freqs`.

```{.python .input}
#@tab all
def get_max_freq_pair(token_freqs):
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # Key of `pairs` is a tuple of two consecutive symbols
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of `pairs` with the max value
```

Như một cách tiếp cận tham lam dựa trên tần số của các ký hiệu liên tiếp, mã hóa cặp byte sẽ sử dụng hàm `merge_symbols` sau để hợp nhất cặp ký hiệu liên tiếp thường xuyên nhất để tạo ra các ký hiệu mới.

```{.python .input}
#@tab all
def merge_symbols(max_freq_pair, token_freqs, symbols):
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),
                                  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs
```

Bây giờ chúng ta lặp đi lặp lại thực hiện thuật toán mã hóa cặp byte trên các phím của từ điển `token_freqs`. Trong lần lặp đầu tiên, cặp ký hiệu liên tiếp thường xuyên nhất là `'t'` và `'a'`, do đó mã hóa cặp byte hợp nhất chúng để tạo ra một ký hiệu mới `'ta'`. Trong lần lặp thứ hai, mã hóa cặp byte tiếp tục hợp nhất `'ta'` và `'l'` để dẫn đến một ký hiệu mới khác `'tal'`.

```{.python .input}
#@tab all
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'merge #{i + 1}:', max_freq_pair)
```

Sau 10 lần lặp lại mã hóa cặp byte, chúng ta có thể thấy rằng danh sách `symbols` bây giờ chứa thêm 10 ký hiệu được sáp nhập lặp đi lặp lại từ các ký hiệu khác.

```{.python .input}
#@tab all
print(symbols)
```

Đối với cùng một tập dữ liệu được chỉ định trong các khóa của từ điển `raw_token_freqs`, mỗi từ trong tập dữ liệu bây giờ được phân đoạn bằng các từ con “fast_”, “fast”, “er_”, “tall_”, và “tall” do kết quả của thuật toán mã hóa cặp byte. Ví dụ, các từ “faster_” và “taller_” được phân đoạn là “fast er_” và “tall er_”, tương ứng.

```{.python .input}
#@tab all
print(list(token_freqs.keys()))
```

Lưu ý rằng kết quả của mã hóa cặp byte phụ thuộc vào tập dữ liệu đang được sử dụng. Chúng ta cũng có thể sử dụng các từ con học được từ một tập dữ liệu để phân đoạn các từ của tập dữ liệu khác. Như một cách tiếp cận tham lam, hàm `segment_BPE` sau cố gắng phá vỡ các từ thành các từ con dài nhất có thể từ đối số đầu vào `symbols`.

```{.python .input}
#@tab all
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

Sau đây, chúng ta sử dụng các từ con trong danh sách `symbols`, được học từ tập dữ liệu nói trên, đến phân đoạn `tokens` đại diện cho một tập dữ liệu khác.

```{.python .input}
#@tab all
tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))
```

## Tóm tắt

* Mô hình fastText đề xuất một cách tiếp cận nhúng từ con. Dựa trên mô hình skip-gram trong word2vec, nó đại diện cho một từ trung tâm làm tổng của vectơ từ con của nó.
* Mã hóa cặp byte thực hiện phân tích thống kê của tập dữ liệu đào tạo để khám phá các ký hiệu phổ biến trong một từ. Như một cách tiếp cận tham lam, mã hóa cặp byte lặp đi lặp lại hợp nhất cặp ký hiệu liên tiếp thường xuyên nhất.
* Nhúng từ phụ có thể cải thiện chất lượng biểu diễn của các từ hiếm và các từ ngoài từ điển.

## Bài tập

1. Ví dụ, có khoảng $3\times 10^8$ $6$-gram có thể bằng tiếng Anh. Vấn đề khi có quá nhiều subwords là gì? Làm thế nào để giải quyết vấn đề? Hint: refer to the end of Section 3.2 of the fastText paper :cite:`Bojanowski.Grave.Joulin.ea.2017`.
1. Làm thế nào để thiết kế một mô hình nhúng từ con dựa trên mô hình túi-of-từ liên tục?
1. Để có được từ vựng có kích thước $m$, cần có bao nhiêu thao tác hợp nhất khi kích thước từ vựng biểu tượng ban đầu là $n$?
1. Làm thế nào để mở rộng ý tưởng mã hóa cặp byte để trích xuất cụm từ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/386)
:end_tab:

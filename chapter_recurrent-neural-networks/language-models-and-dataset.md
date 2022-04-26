# Mô hình ngôn ngữ và bộ dữ liệu
:label:`sec_language_model`

Trong :numref:`sec_text_preprocessing`, chúng ta thấy cách ánh xạ dữ liệu văn bản thành mã thông báo, trong đó các mã thông báo này có thể được xem như một chuỗi các quan sát rời rạc, chẳng hạn như từ hoặc ký tự. Giả sử rằng các thẻ trong một chuỗi văn bản chiều dài $T$ lần lượt $x_1, x_2, \ldots, x_T$. Sau đó, trong chuỗi văn bản, $x_t$ ($1 \leq t \leq T$) có thể được coi là quan sát hoặc nhãn tại bước thời gian $t$. Với một chuỗi văn bản như vậy, mục tiêu của mô hình ngôn ngữ* là ước tính xác suất chung của chuỗi 

$$P(x_1, x_2, \ldots, x_T).$$

Mô hình ngôn ngữ cực kỳ hữu ích. Ví dụ, một mô hình ngôn ngữ lý tưởng sẽ có thể tự tạo văn bản tự nhiên, chỉ cần vẽ một mã thông báo tại một thời điểm $x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1)$. Không giống như con khỉ sử dụng một máy đánh chữ, tất cả các văn bản nổi lên từ một mô hình như vậy sẽ truyền như ngôn ngữ tự nhiên, ví dụ, văn bản tiếng Anh. Hơn nữa, nó sẽ là đủ để tạo ra một hộp thoại có ý nghĩa, chỉ bằng cách điều chỉnh văn bản trên các đoạn hộp thoại trước đó. Rõ ràng chúng ta vẫn còn rất xa so với việc thiết kế một hệ thống như vậy, vì nó sẽ cần phải * hiểu văn bản chứ không phải chỉ tạo ra nội dung hợp lý ngữ pháp. 

Tuy nhiên, các mô hình ngôn ngữ có dịch vụ tuyệt vời ngay cả ở dạng hạn chế của chúng. Ví dụ, các cụm từ “để nhận ra lời nói” và “để phá hỏng một bãi biển đẹp” nghe rất giống nhau. Điều này có thể gây ra sự mơ hồ trong nhận dạng giọng nói, dễ dàng giải quyết thông qua một mô hình ngôn ngữ từ chối bản dịch thứ hai là kỳ lạ. Tương tự như vậy, trong một thuật toán tóm tắt tài liệu, đáng để biết rằng “chó cắn người đàn ông” thường xuyên hơn nhiều so với “người đàn ông cắn chó”, hoặc “Tôi muốn ăn bà” là một tuyên bố khá đáng lo ngại, trong khi “Tôi muốn ăn, bà” lành tính hơn nhiều. 

## Học một mô hình ngôn ngữ

Câu hỏi rõ ràng là làm thế nào chúng ta nên mô hình hóa một tài liệu, hoặc thậm chí là một chuỗi các mã thông báo. Giả sử rằng chúng ta mã hóa dữ liệu văn bản ở cấp độ từ. Chúng tôi có thể truy tìm phân tích mà chúng tôi áp dụng cho các mô hình trình tự trong :numref:`sec_sequence`. Hãy để chúng tôi bắt đầu bằng cách áp dụng các quy tắc xác suất cơ bản: 

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t  \mid  x_1, \ldots, x_{t-1}).$$

Ví dụ, xác suất của một chuỗi văn bản chứa bốn từ sẽ được đưa ra là: 

$$P(\text{deep}, \text{learning}, \text{is}, \text{fun}) =  P(\text{deep}) P(\text{learning}  \mid  \text{deep}) P(\text{is}  \mid  \text{deep}, \text{learning}) P(\text{fun}  \mid  \text{deep}, \text{learning}, \text{is}).$$

Để tính toán mô hình ngôn ngữ, chúng ta cần tính xác suất của các từ và xác suất có điều kiện của một từ được đưa ra vài từ trước đó. Xác suất như vậy về cơ bản là các tham số mô hình ngôn ngữ. 

Ở đây, chúng tôi giả định rằng tập dữ liệu đào tạo là một cơ thể văn bản lớn, chẳng hạn như tất cả các mục Wikipedia, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg) và tất cả văn bản được đăng trên Web. Xác suất của các từ có thể được tính từ tần số từ tương đối của một từ nhất định trong tập dữ liệu đào tạo. Ví dụ, ước tính $\hat{P}(\text{deep})$ có thể được tính là xác suất của bất kỳ câu nào bắt đầu bằng từ “sâu”. Một cách tiếp cận ít chính xác hơn một chút sẽ là đếm tất cả các lần xuất hiện của từ “sâu” và chia nó cho tổng số từ trong corpus. Điều này hoạt động khá tốt, đặc biệt là đối với các từ thường xuyên. Moving Di chuyển on, we could attempt cố gắng to estimate ước tính 

$$\hat{P}(\text{learning} \mid \text{deep}) = \frac{n(\text{deep, learning})}{n(\text{deep})},$$

trong đó $n(x)$ và $n(x, x')$ là số lần xuất hiện của singletons và các cặp từ liên tiếp, tương ứng. Thật không may, ước tính xác suất của một cặp từ có phần khó khăn hơn, vì sự xuất hiện của “học sâu” ít thường xuyên hơn rất nhiều. Đặc biệt, đối với một số kết hợp từ bất thường, có thể khó tìm đủ lần xuất hiện để có được ước tính chính xác. Mọi thứ thay đổi cho tồi tệ hơn cho các kết hợp ba từ và hơn thế nữa. Sẽ có nhiều kết hợp ba từ hợp lý mà chúng ta có thể sẽ không thấy trong tập dữ liệu của mình. Trừ khi chúng tôi cung cấp một số giải pháp để gán các kết hợp từ như vậy không đếm, chúng tôi sẽ không thể sử dụng chúng trong một mô hình ngôn ngữ. Nếu tập dữ liệu nhỏ hoặc nếu các từ rất hiếm, chúng ta có thể không tìm thấy ngay cả một trong số chúng. 

Một chiến lược phổ biến là thực hiện một số hình thức của * Laplace smooth*. Giải pháp là thêm một hằng số nhỏ cho tất cả các số đếm. Biểu thị bằng $n$ tổng số từ trong bộ đào tạo và $m$ số từ duy nhất. Giải pháp này giúp với singletons, ví dụ, thông qua 

$$\begin{aligned}
	\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1}, \\
	\hat{P}(x' \mid x) & = \frac{n(x, x') + \epsilon_2 \hat{P}(x')}{n(x) + \epsilon_2}, \\
	\hat{P}(x'' \mid x,x') & = \frac{n(x, x',x'') + \epsilon_3 \hat{P}(x'')}{n(x, x') + \epsilon_3}.
\end{aligned}$$

Ở đây $\epsilon_1,\epsilon_2$, và $\epsilon_3$ là siêu tham số. Lấy $\epsilon_1$ làm ví dụ: khi $\epsilon_1 = 0$, không áp dụng làm mịn; khi $\epsilon_1$ tiếp cận vô cực dương, $\hat{P}(x)$ tiếp cận xác suất thống nhất $1/m$. Trên đây là một biến thể khá nguyên thủy của những kỹ thuật khác có thể thực hiện :cite:`Wood.Gasthaus.Archambeau.ea.2011`. 

Thật không may, các mô hình như thế này trở nên khó sử dụng khá nhanh vì những lý do sau. Đầu tiên, chúng ta cần lưu trữ tất cả các số lượng. Thứ hai, điều này hoàn toàn bỏ qua ý nghĩa của các từ. Ví dụ, “mèo” và “mèo” sẽ xảy ra trong các bối cảnh liên quan. Khá khó để điều chỉnh các mô hình như vậy thành các bối cảnh bổ sung, trong khi, các mô hình ngôn ngữ dựa trên học tập sâu rất phù hợp để tính đến điều này. Cuối cùng, chuỗi từ dài gần như chắc chắn là mới lạ, do đó một mô hình chỉ đơn giản là đếm tần suất của các chuỗi từ đã thấy trước đó bị ràng buộc để thực hiện kém ở đó. 

## Mô hình Markov và $n$-gram

Trước khi thảo luận về các giải pháp liên quan đến học sâu, chúng ta cần thêm một số thuật ngữ và khái niệm. Nhớ lại cuộc thảo luận của chúng tôi về các mô hình Markov trong :numref:`sec_sequence`. Hãy để chúng tôi áp dụng điều này cho mô hình ngôn ngữ. Một phân phối trên chuỗi thỏa mãn thuộc tính Markov của thứ tự đầu tiên nếu $P(x_{t+1} \mid x_t, \ldots, x_1) = P(x_{t+1} \mid x_t)$. Đơn hàng cao hơn tương ứng với các phụ thuộc dài hơn. Điều này dẫn đến một số xấp xỉ mà chúng ta có thể áp dụng cho mô hình một chuỗi: 

$$
\begin{aligned}
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2) P(x_3) P(x_4),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_2) P(x_4  \mid  x_3),\\
P(x_1, x_2, x_3, x_4) &=  P(x_1) P(x_2  \mid  x_1) P(x_3  \mid  x_1, x_2) P(x_4  \mid  x_2, x_3).
\end{aligned}
$$

Các công thức xác suất liên quan đến một, hai và ba biến thường được gọi là mô hình *unigram*, *bigram* và * trigram*, tương ứng. Sau đây, chúng ta sẽ học cách thiết kế các mô hình tốt hơn. 

## Natural Language Thống kê

Hãy để chúng tôi xem làm thế nào điều này hoạt động trên dữ liệu thực. Chúng tôi xây dựng một từ vựng dựa trên tập dữ liệu cỗ máy thời gian như được giới thiệu trong :numref:`sec_text_preprocessing` và in top 10 từ thường xuyên nhất.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

```{.python .input}
#@tab all
tokens = d2l.tokenize(d2l.read_time_machine())
# Since each text line is not necessarily a sentence or a paragraph, we
# concatenate all text lines 
corpus = [token for line in tokens for token in line]
vocab = d2l.Vocab(corpus)
vocab.token_freqs[:10]
```

Như chúng ta có thể thấy, (** những từ phổ biến nhất là**) thực sự khá nhàm chán khi nhìn vào. Chúng thường được gọi là (*** stop words***) và do đó lọc ra. Tuy nhiên, chúng vẫn mang ý nghĩa và chúng tôi vẫn sẽ sử dụng chúng. Bên cạnh đó, khá rõ ràng rằng tần số từ phân rã khá nhanh. Từ $10^{\mathrm{th}}$ thường xuyên nhất là ít hơn $1/5$ phổ biến như từ phổ biến nhất. Để có được một ý tưởng tốt hơn, chúng tôi [** vẽ hình của tần số từ **].

```{.python .input}
#@tab all
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```

Chúng tôi đang ở trên một cái gì đó khá cơ bản ở đây: tần số từ phân rã nhanh chóng theo một cách rõ ràng. Sau khi xử lý một vài từ đầu tiên là ngoại lệ, tất cả các từ còn lại gần như theo một đường thẳng trên một âm mưu log-log. Điều này có nghĩa là các từ thỏa mãn định luật *Zipf*, trong đó tuyên bố rằng tần số $n_i$ của từ $i^\mathrm{th}$ thường xuyên nhất là: 

$$n_i \propto \frac{1}{i^\alpha},$$
:eqlabel:`eq_zipf_law`

which mà is equivalent tương đương to 

$$\log n_i = -\alpha \log i + c,$$

trong đó $\alpha$ là số mũ đặc trưng cho sự phân bố và $c$ là một hằng số. Điều này sẽ cho chúng ta tạm dừng nếu chúng ta muốn mô hình hóa các từ bằng cách đếm số liệu thống kê và làm mịn. Rốt cuộc, chúng ta sẽ đánh giá quá cao đáng kể tần số của đuôi, còn được gọi là những từ không thường xuyên. Nhưng [** những gì về các kết hợp từ khác, chẳng hạn như bigrams, trigrams**], và hơn thế nữa? Chúng ta hãy xem liệu tần số bigram có hoạt động theo cách tương tự như tần số unigram hay không.

```{.python .input}
#@tab all
bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
bigram_vocab = d2l.Vocab(bigram_tokens)
bigram_vocab.token_freqs[:10]
```

Một điều đáng chú ý ở đây. Trong số mười cặp từ thường xuyên nhất, chín cặp được cấu tạo từ cả hai từ dừng và chỉ có một cặp có liên quan đến cuốn sách thực tế—"thời gian”. Hơn nữa, chúng ta hãy xem liệu tần số trigram có hoạt động theo cùng một cách hay không.

```{.python .input}
#@tab all
trigram_tokens = [triple for triple in zip(
    corpus[:-2], corpus[1:-1], corpus[2:])]
trigram_vocab = d2l.Vocab(trigram_tokens)
trigram_vocab.token_freqs[:10]
```

Cuối cùng, chúng ta hãy [** hình dung tần số token**] trong số ba mô hình này: unigrams, bigram và trigram.

```{.python .input}
#@tab all
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token: x',
         ylabel='frequency: n(x)', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

Con số này khá thú vị vì một số lý do. Đầu tiên, ngoài các từ unigram, chuỗi các từ cũng dường như tuân theo định luật Zipf, mặc dù với số mũ nhỏ hơn $\alpha$ trong :eqref:`eq_zipf_law`, tùy thuộc vào độ dài trình tự. Thứ hai, số lượng $n$-gram khác biệt không phải là lớn. Điều này cho chúng ta hy vọng rằng có khá nhiều cấu trúc trong ngôn ngữ. Thứ ba, nhiều $n$-gram xảy ra rất hiếm khi, khiến Laplace làm mịn khá không phù hợp với mô hình ngôn ngữ. Thay vào đó, chúng tôi sẽ sử dụng các mô hình dựa trên học tập sâu. 

## Đọc dữ liệu trình tự dài

Vì dữ liệu trình tự là theo bản chất tuần tự của chúng, chúng ta cần giải quyết vấn đề xử lý nó. Chúng tôi đã làm như vậy một cách khá đặc biệt trong :numref:`sec_sequence`. Khi chuỗi quá lâu để được xử lý bởi các mô hình cùng một lúc, chúng tôi có thể muốn chia các chuỗi như vậy để đọc. Bây giờ chúng ta hãy mô tả các chiến lược chung. Trước khi giới thiệu mô hình, chúng ta hãy giả định rằng chúng ta sẽ sử dụng mạng thần kinh để đào tạo một mô hình ngôn ngữ, trong đó mạng xử lý một loạt các chuỗi nhỏ với độ dài được xác định trước, giả sử các bước thời gian $n$, tại một thời điểm. Bây giờ câu hỏi là làm thế nào để [** đọc minibatches của các tính năng và nhãn một cách ngẫu nhiên.**] 

Để bắt đầu, vì một chuỗi văn bản có thể dài tùy ý, chẳng hạn như toàn bộ cuốn sách * Máy thời gian*, chúng ta có thể phân vùng một chuỗi dài như vậy thành các dãy con với cùng một số bước thời gian. Khi đào tạo mạng thần kinh của chúng tôi, một minibatch của các dãy con như vậy sẽ được đưa vào mô hình. Giả sử rằng mạng xử lý một dãy con $n$ bước thời gian tại một thời điểm. :numref:`fig_timemachine_5gram` cho thấy tất cả các cách khác nhau để có được dãy con từ một chuỗi văn bản gốc, trong đó $n=5$ và một mã thông báo tại mỗi bước thời gian tương ứng với một ký tự. Lưu ý rằng chúng ta có khá tự do vì chúng ta có thể chọn một phần bù tùy ý cho biết vị trí ban đầu. 

![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

Do đó, chúng ta nên chọn cái nào từ :numref:`fig_timemachine_5gram`? Trong thực tế, tất cả chúng đều tốt như nhau. Tuy nhiên, nếu chúng ta chỉ chọn một bù đắp, có phạm vi bảo hiểm hạn chế của tất cả các dãy con có thể để đào tạo mạng của chúng tôi. Do đó, chúng ta có thể bắt đầu với một offset ngẫu nhiên để phân vùng một chuỗi để có được cả *coverage* và *randomness*. Sau đây, chúng tôi mô tả cách thực hiện điều này cho cả hai
*lấy mẫu ngẫu nhiên* và * phân vùng tuần tự* chiến lược.

### Lấy mẫu ngẫu nhiên

(**Trong lấy mẫu ngẫu nhiên, mỗi ví dụ là một dãy con được chụp tùy tiện trên chuỗi dài ban đầu**) Các dãy con từ hai minibatches ngẫu nhiên liền kề trong quá trình lặp lại không nhất thiết liền kề trên chuỗi gốc. Đối với mô hình hóa ngôn ngữ, mục tiêu là dự đoán token tiếp theo dựa trên những mã thông báo mà chúng ta đã thấy cho đến nay, do đó các nhãn là chuỗi ban đầu, được dịch chuyển bởi một mã thông báo. 

Mã sau ngẫu nhiên tạo ra một minibatch từ dữ liệu mỗi lần. Ở đây, đối số `batch_size` chỉ định số ví dụ dãy con trong mỗi minibatch và `num_steps` là số bước thời gian được xác định trước trong mỗi dãy con.

```{.python .input}
#@tab all
def seq_data_iter_random(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using random sampling."""
    # Start with a random offset (inclusive of `num_steps - 1`) to partition a
    # sequence
    corpus = corpus[random.randint(0, num_steps - 1):]
    # Subtract 1 since we need to account for labels
    num_subseqs = (len(corpus) - 1) // num_steps
    # The starting indices for subsequences of length `num_steps`
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # In random sampling, the subsequences from two adjacent random
    # minibatches during iteration are not necessarily adjacent on the
    # original sequence
    random.shuffle(initial_indices)

    def data(pos):
        # Return a sequence of length `num_steps` starting from `pos`
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Here, `initial_indices` contains randomized starting indices for
        # subsequences
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield d2l.tensor(X), d2l.tensor(Y)
```

Hãy để chúng tôi [** tự tạo một chuỗi từ 0 đến 34.**] Chúng tôi giả định rằng kích thước lô và số bước thời gian là 2 và 5, tương ứng. Điều này có nghĩa là chúng ta có thể tạo ra $\lfloor (35 - 1) / 5 \rfloor= 6$ các cặp con nhãn tính năng. Với kích thước minibatch là 2, chúng tôi chỉ nhận được 3 minibatches.

```{.python .input}
#@tab all
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

### Phân vùng tuần tự

Ngoài việc lấy mẫu ngẫu nhiên của chuỗi gốc, [** chúng tôi cũng có thể đảm bảo rằng các dãy con từ hai minibatches liền kề trong quá trình lặp lại liền kề trên chuỗi gốc. **] Chiến lược này bảo tồn thứ tự phân chia dãy khi lặp qua minibatches, do đó được gọi là tuần tự phân vùng.

```{.python .input}
#@tab mxnet, pytorch
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

```{.python .input}
#@tab tensorflow
def seq_data_iter_sequential(corpus, batch_size, num_steps):  #@save
    """Generate a minibatch of subsequences using sequential partitioning."""
    # Start with a random offset to partition a sequence
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = d2l.tensor(corpus[offset: offset + num_tokens])
    Ys = d2l.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs = d2l.reshape(Xs, (batch_size, -1))
    Ys = d2l.reshape(Ys, (batch_size, -1))
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        yield X, Y
```

Sử dụng cùng một cài đặt, chúng ta hãy [** tính năng in `X` và nhãn `Y` cho mỗi minibatch**] của dãy con được đọc bởi phân vùng tuần tự. Lưu ý rằng các dãy con từ hai minibatches liền kề trong khi lặp lại thực sự liền kề trên dãy ban đầu.

```{.python .input}
#@tab all
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
```

Bây giờ chúng ta bọc hai hàm lấy mẫu ở trên vào một lớp để chúng ta có thể sử dụng nó như một bộ lặp dữ liệu sau này.

```{.python .input}
#@tab all
class SeqDataLoader:  #@save
    """An iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_sequential
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```

[**Cuối cùng, chúng ta định nghĩa một hàm `load_data_time_machine` trả về cả bộ lặp dữ liệu và từ vựng **], vì vậy chúng ta có thể sử dụng nó tương tự như các hàm khác với tiền tố `load_data`, chẳng hạn như `d2l.load_data_fashion_mnist` được định nghĩa trong :numref:`sec_fashion_mnist`.

```{.python .input}
#@tab all
def load_data_time_machine(batch_size, num_steps,  #@save
                           use_random_iter=False, max_tokens=10000):
    """Return the iterator and the vocabulary of the time machine dataset."""
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

## Tóm tắt

* Mô hình ngôn ngữ là chìa khóa để xử lý ngôn ngữ tự nhiên.
* $n$-gram cung cấp một mô hình thuận tiện để xử lý các chuỗi dài bằng cách cắt ngắn sự phụ thuộc.
* Chuỗi dài bị vấn đề mà chúng xảy ra rất hiếm khi hoặc không bao giờ.
* Định luật Zipf chi phối từ phân phối cho không chỉ unigrams mà còn là $n$-gram khác.
* Có rất nhiều cấu trúc nhưng không đủ tần số để đối phó với các kết hợp từ không thường xuyên một cách hiệu quả thông qua Laplace làm mịn.
* Các lựa chọn chính để đọc các chuỗi dài là lấy mẫu ngẫu nhiên và phân vùng tuần tự. Cái sau có thể đảm bảo rằng các dãy con từ hai minibatches liền kề trong quá trình lặp lại liền kề trên chuỗi gốc.

## Bài tập

1. Giả sử có $100,000$ từ trong tập dữ liệu đào tạo. Bốn gram cần lưu trữ bao nhiêu tần số từ và tần số liền kề nhiều từ?
1. Làm thế nào bạn sẽ mô hình một cuộc đối thoại?
1. Ước tính số mũ của định luật Zipf cho unigrams, bigram, và trigram.
1. Bạn có thể nghĩ đến những phương pháp nào khác để đọc dữ liệu trình tự dài?
1. Hãy xem xét bù ngẫu nhiên mà chúng ta sử dụng để đọc các chuỗi dài.
    1. Tại sao nó là một ý tưởng tốt để có một bù đắp ngẫu nhiên?
    1. Liệu nó có thực sự dẫn đến một phân phối hoàn toàn thống nhất trên các chuỗi trên tài liệu?
    1. Bạn sẽ phải làm gì để làm cho mọi thứ trở nên thống nhất hơn?
1. Nếu chúng ta muốn một ví dụ trình tự là một câu hoàn chỉnh, điều này giới thiệu loại vấn đề nào trong việc lấy mẫu minibatch? Làm thế nào chúng ta có thể khắc phục sự cố?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/117)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/118)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1049)
:end_tab:

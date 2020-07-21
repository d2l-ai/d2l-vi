<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Language Models and the Dataset
-->

# Mô hình Ngôn ngữ và Tập dữ liệu
:label:`sec_language_model`


<!--
In :numref:`sec_text_preprocessing`, we see how to map text data into tokens, and these tokens can be viewed as a time series of discrete observations.
Assuming the tokens in a text of length $T$ are in turn $x_1, x_2, \ldots, x_T$, 
then, in the discrete time series, $x_t$($1 \leq t \leq T$) can be considered as the output or label of timestep $t$.
Given such a sequence, the goal of a language model is to estimate the probability
-->

:numref:`sec_text_preprocessing` đã trình bày cách ánh xạ dữ liệu văn bản sang token, những token này có thể được xem như một chuỗi thời gian của các quan sát rời rạc.
Giả sử văn bản độ dài $T$ có dãy token là $x_1, x_2, \ldots, x_T$, thì $x_t$($1 \leq t \leq T$) có thể coi là đầu ra (hoặc nhãn) tại bước thời gian $t$.
Khi đã có chuỗi thời gian trên, mục tiêu của mô hình ngôn ngữ là ước tính xác suất của 

$$p(x_1, x_2, \ldots, x_T).$$

<!--
Language models are incredibly useful.
For instance, an ideal language model would be able to generate natural text just on its own, simply by drawing one word at a time $w_t \sim p(w_t \mid w_{t-1}, \ldots, w_1)$.
Quite unlike the monkey using a typewriter, all text emerging from such a model would pass as natural language, e.g., English text.
Furthermore, it would be sufficient for generating a meaningful dialog, simply by conditioning the text on previous dialog fragments.
Clearly we are still very far from designing such a system, since it would need to *understand* the text rather than just generate grammatically sensible content.
-->

Mô hình ngôn ngữ vô cùng hữu dụng. 
Chẳng hạn, một mô hình lý tưởng có thể tự tạo ra văn bản tự nhiên, chỉ bằng cách chọn một từ $w_t$ tại thời điểm $t$ với $w_t \sim p(w_t \mid w_{t-1}, \ldots, w_1)$.
Khác hoàn toàn với việc chỉ gõ phím ngẫu nhiên như trong định lý con khỉ vô hạn (*infinite monkey theorem*), văn bản được sinh ra từ mô hình này giống ngôn ngữ tự nhiên, giống tiếng Anh chẳng hạn.
Hơn nữa, mô hình đủ khả năng tạo ra một đoạn hội thoại có ý nghĩa mà chỉ cần dựa vào đoạn hội thoại trước đó.
Trên thực tế, còn rất xa để thiết kế được hệ thống như vậy, vì mô hình sẽ cần *hiểu* văn bản hơn là chỉ tạo ra nội dung đúng ngữ pháp.

<!--
Nonetheless language models are of great service even in their limited form.
For instance, the phrases "to recognize speech" and "to wreck a nice beach" sound very similar.
This can cause ambiguity in speech recognition, ambiguity that is easily resolved through a language model which rejects the second translation as outlandish.
Likewise, in a document summarization algorithm it is worth while knowing that "dog bites man" is much more frequent than "man bites dog", 
or that "I want to eat grandma" is a rather disturbing statement, whereas "I want to eat, grandma" is much more benign.
-->

Tuy nhiên, mô hình ngôn ngữ vẫn rất hữu dụng ngay cả khi còn hạn chế.
Chẳng hạn, cụm từ “nhận dạng giọng nói” và “nhân gian rộng lối” có phát âm khá giống nhau.
Điều này có thể gây ra sự mơ hồ trong việc nhận dạng giọng nói, nhưng có thể dễ dàng được giải quyết với một mô hình ngôn ngữ. Mô hình sẽ loại bỏ ngay phương án thứ hai do mang ý nghĩa kì lạ.
Tương tự, một thuật toán tóm tắt tài liệu nên phân biệt được rằng câu “chó cắn người" xuất hiện thường xuyên hơn nhiều so với “người cắn chó”, hay như “Cháu muốn ăn bà ngoại" nghe khá kinh dị trong khi “Cháu muốn ăn, bà ngoại" lại là bình thường.


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Estimating a Language Model
-->

## Ước tính một Mô hình Ngôn ngữ

<!--
The obvious question is how we should model a document, or even a sequence of words.
We can take recourse to the analysis we applied to sequence models in the previous section.
Let us start by applying basic probability rules:
-->

Làm thế nào để mô hình hóa một tài liệu hay thậm chí là một chuỗi các từ?
Ta có thể sử dụng cách phân tích đã dùng trong mô hình chuỗi ở phần trước.
Bắt đầu bằng việc áp dụng quy tắc xác suất cơ bản sau:

$$p(w_1, w_2, \ldots, w_T) = p(w_1) \prod_{t=2}^T p(w_t  \mid  w_1, \ldots, w_{t-1}).$$

<!--
For example, the probability of a text sequence containing four tokens consisting of words and punctuation would be given as:
-->

Ví dụ, xác suất của chuỗi văn bản chứa bốn token bao gồm các từ và dấu chấm câu được tính như sau:

<!--
$$p(\mathrm{Statistics}, \mathrm{is}, \mathrm{fun}, \mathrm{.}) =  p(\mathrm{Statistics}) p(\mathrm{is}  \mid  \mathrm{Statistics}) p(\mathrm{fun}  \mid  \mathrm{Statistics}, \mathrm{is}) p(\mathrm{.}  \mid  \mathrm{Statistics}, \mathrm{is}, \mathrm{fun}).$$
-->

$$p(\mathrm{Statistics}, \mathrm{is}, \mathrm{fun}, \mathrm{.}) =  p(\mathrm{Statistics}) p(\mathrm{is}  \mid  \mathrm{Statistics}) p(\mathrm{fun}  \mid  \mathrm{Statistics}, \mathrm{is}) p(\mathrm{.}  \mid  \mathrm{Statistics}, \mathrm{is}, \mathrm{fun}).$$


<!--
In order to compute the language model, we need to calculate the probability of words and the conditional probability of a word given the previous few words, i.e., language model parameters.
Here, we assume that the training dataset is a large text corpus, such as all Wikipedia entries, [Project Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg), or all text posted online on the web.
The probability of words can be calculated from the relative word frequency of a given word in the training dataset.
-->

Để tính toán mô hình ngôn ngữ, ta cần tính xác suất các từ và xác suất có điều kiện của một từ khi đã có vài từ trước đó. 
Đây chính là các tham số của mô hình ngôn ngữ.
Ở đây chúng ta giả định rằng, tập dữ liệu huấn luyện là một kho ngữ liệu lớn, chẳng hạn như là tất cả các mục trong Wikipedia của [Dự án Gutenberg](https://en.wikipedia.org/wiki/Project_Gutenberg), hoặc tất cả văn bản được đăng trên mạng.
Xác suất riêng lẻ của từng từ có thể tính bằng tần suất của từ đó trong tập dữ liệu huấn luyện.

<!--
For example, $p(\mathrm{Statistics})$ can be calculated as the probability of any sentence starting with the word "statistics".
A slightly less accurate approach would be to count all occurrences of the word "statistics" and divide it by the total number of words in the corpus.
This works fairly well, particularly for frequent words.
Moving on, we could attempt to estimate
-->

Ví dụ, $p(\mathrm{Statistics})$ có thể được tính là xác suất của bất kỳ câu nào bắt đầu bằng “statistics”.
Một cách thiếu chính xác hơn là đếm tất cả số lần xuất hiện của ”statistics” và chia số lần đó cho tổng số từ trong kho ngữ liệu văn bản.
Cách làm này khá hiệu quả, đặc biệt là với các từ xuất hiện thường xuyên. 
Tiếp theo, ta tính 

$$\hat{p}(\mathrm{is} \mid \mathrm{Statistics}) = \frac{n(\mathrm{Statistics, is})}{n(\mathrm{Statistics})}.$$

<!--
Here $n(w)$ and $n(w, w')$ are the number of occurrences of singletons and pairs of words respectively.
Unfortunately, estimating the probability of a word pair is somewhat more difficult, since the occurrences of "Statistics is" are a lot less frequent.
In particular, for some unusual word combinations it may be tricky to find enough occurrences to get accurate estimates.
Things take a turn for the worse for 3-word combinations and beyond.
There will be many plausible 3-word combinations that we likely will not see in our dataset.
Unless we provide some solution to give such word combinations nonzero weight, we will not be able to use these as a language model.
If the dataset is small or if the words are very rare, we might not find even a single one of them.
-->

Ở đây $n(w)$ và $n(w, w')$ lần lượt là số lần xuất hiện của các từ đơn và cặp từ ghép.
Đáng tiếc là việc ước tính xác suất của một cặp từ thường khó khăn hơn, bởi vì sự xuất hiện của cặp từ “Statistics is” hiếm khi xảy ra hơn.
Đặc biệt, với các cụm từ ít đi cùng nhau, rất khó tìm đủ số lần xuất hiện để ước tính chính xác. 
Mọi thứ thậm chí sẽ khó hơn đối với các cụm ba từ trở lên.
Sẽ có nhiều cụm ba từ hợp lý mà hầu như không hề xuất hiện trong tập dữ liệu.
Trừ khi có giải pháp để đánh trọng số khác không cho các tổ hợp từ đó, nếu không sẽ không thể sử dụng chúng trong một mô hình ngôn ngữ.
Nếu kích thước tập dữ liệu nhỏ hoặc nếu các từ rất hiếm, chúng ta thậm chí có thể không tìm thấy nổi một lần xuất hiện của các tổ hợp từ đó.


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
A common strategy is to perform some form of Laplace smoothing.
We already encountered this in our discussion of naive Bayes in :numref:`sec_naive_bayes` where the solution was to add a small constant to all counts.
This helps with singletons, e.g., via
-->

Một kỹ thuật phổ biến là làm mượt Laplace (*Laplace smoothing*).
Chúng ta đã biết kỹ thuật này khi thảo luận về Naive Bayes trong :numref:`sec_naive_bayes`, với giải pháp là cộng thêm một hằng số nhỏ vào tất cả các số đếm như sau

$$\begin{aligned}
\hat{p}(w) & = \frac{n(w) + \epsilon_1/m}{n + \epsilon_1}, \\
\hat{p}(w' \mid w) & = \frac{n(w, w') + \epsilon_2 \hat{p}(w')}{n(w) + \epsilon_2}, \\
\hat{p}(w'' \mid w',w) & = \frac{n(w, w',w'') + \epsilon_3 \hat{p}(w',w'')}{n(w, w') + \epsilon_3}.
\end{aligned}$$


<!--
Here the coefficients $\epsilon_i > 0$ determine how much we use the estimate for a shorter sequence as a fill-in for longer ones.
Moreover, $m$ is the total number of words we encounter.
The above is a rather primitive variant of what is Kneser-Ney smoothing and Bayesian nonparametrics can accomplish.
See e.g., :cite:`Wood.Gasthaus.Archambeau.ea.2011` for more detail of how to accomplish this.
Unfortunately, models like this get unwieldy rather quickly for the following reasons. First, we need to store all counts.
Second, this entirely ignores the meaning of the words.
For instance, *"cat"* and *"feline"* should occur in related contexts.
It is quite difficult to adjust such models to additional contexts, whereas, deep learning based language models are well suited to take this into account.
Last, long word sequences are almost certain to be novel, hence a model that simply counts the frequency of previously seen word sequences is bound to perform poorly there.
-->

Ở đây các hệ số $\epsilon_i > 0$ xác định mức độ ảnh hưởng của chuỗi ngắn hơn khi ước tính chuỗi dài hơn,
$m$ là tổng số từ trong tập văn bản.
Công thức trên là một biến thể khá nguyên thủy của kỹ thuật làm mượt Kneser-Ney và Bayesian phi tham số.
Xem :cite:`Wood.Gasthaus.Archambeau.ea.2011` để biết thêm chi tiết.
Thật không may, các mô hình như vậy là bất khả thi vì những lý do sau.
Đầu tiên, chúng ta cần lưu trữ tất cả các số đếm. 
Thứ hai, các mô hình hoàn toàn bỏ qua ý nghĩa của các từ.
Chẳng hạn, danh từ *“mèo”(“cat")* và tính từ *“thuộc về mèo”(“feline”)* nên xuất hiện trong các ngữ cảnh có liên quan đến nhau.
Rất khó để thêm các ngữ cảnh bổ trợ vào các mô hình đó, trong khi các mô hình ngôn ngữ dựa trên học sâu hoàn toàn có thể làm được.
Cuối cùng, các chuỗi từ dài gần như hoàn toàn mới lạ, do đó một mô hình chỉ đơn giản đếm tần số của các chuỗi từ đã thấy trước đó sẽ hoạt động rất kém.


<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Markov Models and $n$-grams
-->

## Mô hình Markov và $n$-grams 

<!--
Before we discuss solutions involving deep learning, we need some more terminology and concepts.
Recall our discussion of Markov Models in the previous section.
Let us apply this to language modeling.
A distribution over sequences satisfies the Markov property of first order if $p(w_{t+1} \mid w_t, \ldots, w_1) = p(w_{t+1} \mid w_t)$.
Higher orders correspond to longer dependencies.
This leads to a number of approximations that we could apply to model a sequence:
-->

Trước khi thảo luận các giải pháp sử dụng học sâu, chúng ta sẽ giải thích một số thuật ngữ và khái niệm.
Hãy nhớ lại mô hình Markov đề cập ở phần trước,
và áp dụng để mô hình hóa ngôn ngữ. 
Một phân phối trên các chuỗi thỏa mãn điều kiện Markov bậc nhất nếu $p(w_{t+1} \mid w_t, \ldots, w_1) = p(w_{t+1} \mid w_t)$.
Những bậc cao hơn tương ứng với những chuỗi phụ thuộc dài hơn. 
Do đó chúng ta có thể áp dụng các phép xấp xỉ để mô hình hóa một chuỗi:

$$
\begin{aligned}
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2) p(w_3) p(w_4),\\
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2  \mid  w_1) p(w_3  \mid  w_2) p(w_4  \mid  w_3),\\
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2  \mid  w_1) p(w_3  \mid  w_1, w_2) p(w_4  \mid  w_2, w_3).
\end{aligned}
$$

<!--
The probability formulae that involve one, two, and three variables are typically referred to as unigram, bigram, and trigram models respectively.
In the following, we will learn how to design better models.
-->

Các công thức xác suất liên quan đến một, hai và ba biến được gọi là các mô hình unigram, bigram và trigram.
Sau đây, chúng ta sẽ tìm hiểu cách thiết kế các mô hình tốt hơn.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Natural Language Statistics
-->

## Thống kê Ngôn ngữ Tự nhiên

<!--
Let us see how this works on real data.
We construct a vocabulary based on the time machine data similar to :numref:`sec_text_preprocessing` and print the top $10$ most frequent words.
-->

Hãy cùng xem mô hình hoạt động thế nào trên dữ liệu thực tế.
Chúng ta sẽ xây dựng bộ từ vựng dựa trên tập dữ liệu "cỗ máy thời gian" tương tự như ở :numref:`sec_text_preprocessing` và in ra $10$ từ có tần suất xuất hiện cao nhất.


```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()

tokens = d2l.tokenize(d2l.read_time_machine())
vocab = d2l.Vocab(tokens)
print(vocab.token_freqs[:10])
```


<!--
As we can see, the most popular words are actually quite boring to look at.
They are often referred to as [stop words](https://en.wikipedia.org/wiki/Stop_words) and thus filtered out.
That said, they still carry meaning and we will use them nonetheless.
However, one thing that is quite clear is that the word frequency decays rather rapidly.
The $10^{\mathrm{th}}$ most frequent word is less than $1/5$ as common as the most popular one.
To get a better idea we plot the graph of the word frequency.
-->

Có thể thấy những từ xuất hiện nhiều nhất không có gì đáng chú ý.
Các từ này được gọi là [từ dừng (*stop words*)](https://en.wikipedia.org/wiki/Stop_words) và vì thế chúng thường được lọc ra.
Dù vậy, những từ này vẫn có nghĩa và ta vẫn sẽ sử dụng chúng.
Tuy nhiên, rõ ràng là tần số của từ suy giảm khá nhanh.
Từ phổ biến thứ $10$ xuất hiện ít hơn, chỉ bằng $1/5$ lần so với từ phổ biến nhất.
Để hiểu rõ hơn, chúng ta sẽ vẽ đồ thị tần số của từ.


```{.python .input  n=2}
freqs = [freq for token, freq in vocab.token_freqs]
d2l.plot(freqs, xlabel='token: x', ylabel='frequency: n(x)',
         xscale='log', yscale='log')
```


<!--
We are on to something quite fundamental here: the word frequency decays rapidly in a well defined way.
After dealing with the first four words as exceptions ('the', 'i', 'and', 'of'), all remaining words follow a straight line on a log-log plot.
This means that words satisfy [Zipf's law](https://en.wikipedia.org/wiki/Zipf%27s_law) which states that the item frequency is given by
-->

Chúng ta đang tiến gần tới một đặc điểm cơ bản: tần số của từ suy giảm nhanh chóng theo một cách được xác định rõ.
Ngoại trừ bốn từ đầu tiên ('the', 'i', 'and', 'of'), tất cả các từ còn lại đi theo một đường thẳng trên biểu đồ thang log.
Theo đó các từ tuân theo định luật [Zipf](https://en.wikipedia.org/wiki/Zipf's_law), tức là tần suất xuất hiện của từ được xác định bởi


$$n(x) \propto (x + c)^{-\alpha} \text{ và~do~đó }
\log n(x) = -\alpha \log (x+c) + \mathrm{const.}$$


<!--
This should already give us pause if we want to model words by count statistics and smoothing.
After all, we will significantly overestimate the frequency of the tail, also known as the infrequent words.
But what about the other word combinations (such as bigrams, trigrams, and beyond)?
Let us see whether the bigram frequency behaves in the same manner as the unigram frequency.
-->

Điều này khiến chúng ta cần suy nghĩ kĩ khi mô hình hóa các từ bằng cách đếm và kỹ thuật làm mượt.
Rốt cuộc, chúng ta sẽ ước tính quá cao những từ có tần suất xuất hiện thấp.
Vậy còn các tổ hợp từ khác như 2-gram, 3-gram và nhiều hơn thì sao?
Hãy xem liệu tần số của bigram có tương tự như unigram hay không.


```{.python .input  n=3}
bigram_tokens = [[pair for pair in zip(
    line[:-1], line[1:])] for line in tokens]
bigram_vocab = d2l.Vocab(bigram_tokens)
print(bigram_vocab.token_freqs[:10])
```


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
Two things are notable.
Out of the 10 most frequent word pairs, 9 are composed of stop words and only one is relevant to the actual book---"the time".
Furthermore, let us see whether the trigram frequency behaves in the same manner.
-->

Có một điều đáng chú ý ở đây.
9 trong số 10 cặp từ thường xuyên xuất hiện là các từ dừng và chỉ có một là liên quan đến cuốn sách --- cặp từ "the time".
Hãy xem tần số của trigram có tương tự hay không.


```{.python .input  n=4}
trigram_tokens = [[triple for triple in zip(line[:-2], line[1:-1], line[2:])]
                  for line in tokens]
trigram_vocab = d2l.Vocab(trigram_tokens)
print(trigram_vocab.token_freqs[:10])
```

<!--
Last, let us visualize the token frequency among these three gram models: unigrams, bigrams, and trigrams.
-->

Cuối cùng, hãy quan sát biểu đồ tần số token của các mô hình: unigram, bigram, và trigram.


```{.python .input  n=5}
bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
d2l.plot([freqs, bigram_freqs, trigram_freqs], xlabel='token',
         ylabel='frequency', xscale='log', yscale='log',
         legend=['unigram', 'bigram', 'trigram'])
```

<!--
The graph is quite exciting for a number of reasons.
First, beyond unigram words, also sequences of words appear to be following Zipf's law, albeit with a lower exponent, depending on  sequence length.
Second, the number of distinct n-grams is not that large.
This gives us hope that there is quite a lot of structure in language.
Third, many n-grams occur very rarely, which makes Laplace smoothing rather unsuitable for language modeling. Instead, we will use deep learning based models.
-->

Có vài điều khá thú vị ở biểu đồ này.
Thứ nhất, ngoài unigram, các cụm từ cũng tuân theo định luật Zipf, với số mũ thấp hơn tùy vào chiều dài cụm từ.
Thứ hai, số lượng các n-gram độc nhất là không nhiều.
Điều này có thể liên quan đến số lượng lớn các cấu trúc trong ngôn ngữ.
Thứ ba, rất nhiều n-gram hiếm khi xuất hiện, khiến phép làm mượt Laplace không thích hợp để xây dựng mô hình ngôn ngữ. Thay vào đó, chúng ta sẽ sử dụng các mô hình học sâu.


<!--
## Training Data Preparation
-->

## Chuẩn bị Dữ liệu Huấn luyện


<!--
Before introducing the model, let us assume we will use a neural network to train a language model.
Now the question is how to read minibatches of examples and labels at random.
Since sequence data is by its very nature sequential, we need to address the issue of processing it.
We did so in a rather ad-hoc manner when we introduced in :numref:`sec_sequence`.
Let us formalize this a bit.
-->

Giả sử cần sử dụng mạng nơ-ron để huấn luyện mô hình ngôn ngữ.
Với tính chất tuần tự của dữ liệu chuỗi, làm thế nào để đọc ngẫu nhiên các mini-batch gồm các mẫu và nhãn?
Ví dụ đơn giản trong :numref:`sec_sequence` đã giới thiệu một cách thực hiện.
Hãy tổng quát hóa cách làm này một chút.

<!--
In :numref:`fig_timemachine_5gram`, we visualized several possible ways to obtain 5-grams in a sentence, here a token is a character.
Note that we have quite some freedom since we could pick an arbitrary offset.
-->

:numref:`fig_timemachine_5gram`, biểu diễn các cách để chia một câu thành các 5-gram, ở đây mỗi token là một ký tự.
Ta có thể chọn tùy ý độ dời ở vị trí bắt đầu.


<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
![Different offsets lead to different subsequences when splitting up text.](../img/timemachine-5gram.svg)
-->

![Các độ dời khác nhau dẫn đến các chuỗi con khác nhau khi phân tách văn bản.](../img/timemachine-5gram.svg)
:label:`fig_timemachine_5gram`

<!--
In fact, any one of these offsets is fine.
Hence, which one should we pick? In fact, all of them are equally good.
But if we pick all offsets we end up with rather redundant data due to overlap, particularly if the sequences are long.
Picking just a random set of initial positions is no good either since it does not guarantee uniform coverage of the array.
For instance, if we pick $n​$ elements at random out of a set of $n​$ with random replacement, the probability for a particular element not being picked is $(1-1/n)^n \to e^{-1}​$.
This means that we cannot expect uniform coverage this way.
Even randomly permuting a set of all offsets does not offer good guarantees.
Instead we can use a simple trick to get both *coverage* and *randomness*: use a random offset, after which one uses the terms sequentially.
We describe how to accomplish this for both random sampling and sequential partitioning strategies below.
-->

Chúng ta nên chọn giá trị độ dời nào? Trong thực tế, tất cả các giá trị đó đều tốt như nhau.
Nhưng nếu chọn tất cả các giá trị độ dời, dữ liệu sẽ khá dư thừa do trùng lặp lẫn nhau, đặc biệt trong trường hợp các chuỗi rất dài.
Việc chỉ chọn một tập ngẫu nhiên các vị trí đầu cũng không tốt vì không đảm bảo sẽ bao quát đồng đều cả mảng.
Ví dụ, nếu lấy ngẫu nhiên có hoàn lại $n$ phần tử từ một tập có $n$ phần tử, xác suất một phần tử cụ thể không được chọn là $(1-1/n)^n \to e^{-1}​$.
Nghĩa là ta không thể kỳ vọng vào sự bao quát đồng đều, ngay cả khi hoán vị ngẫu nhiên một tập giá trị độ dời.
Thay vào đó, có thể sử dụng một cách đơn giản để có được cả tính *bao quát* và tính *ngẫu nhiên*, đó là: chọn một độ dời ngẫu nhiên, sau đó sử dụng tuần tự các giá trị tiếp theo.
Điều này được mô tả trong phép lấy mẫu ngẫu nhiên và phép phân tách tuần tự dưới đây.


<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Random Sampling
-->

### Lấy Mẫu Ngẫu nhiên


<!--
The following code randomly generates a minibatch from the data each time.
Here, the batch size `batch_size` indicates the number of examples in each minibatch and `num_steps` is the length of the sequence (or timesteps if we have a time series) included in each example.
In random sampling, each example is a sequence arbitrarily captured on the original sequence.
The positions of two adjacent random minibatches on the original sequence are not necessarily adjacent.
The target is to predict the next character based on what we have seen so far, hence the labels are the original sequence, shifted by one character.
-->

Đoạn mã sau tạo ngẫu nhiên một minibatch dữ liệu.
Ở đây, kích thước batch `batch_size` biểu thị số mẫu trong mỗi minibatch, `num_steps` biểu thị chiều dài mỗi mẫu (là số bước thời gian trong trường hợp chuỗi thời gian).
Trong phép lấy mẫu ngẫu nhiên, mỗi mẫu là một chuỗi tùy ý được lấy ra từ chuỗi gốc.
Hai minibatch ngẫu nhiên liên tiếp không nhất thiết phải liền kề nhau trong chuỗi góc.
Mục tiêu của ta là dự đoán phần tử tiếp theo dựa trên các phần tử đã thấy cho đến hiện tại, do đó nhãn của một mẫu chính là mẫu đó dịch chuyển sang phải một phần tử.


```{.python .input  n=1}
# Saved in the d2l package for later use
def seq_data_iter_random(corpus, batch_size, num_steps):
    # Offset the iterator over the data for uniform starts
    corpus = corpus[random.randint(0, num_steps):]
    # Subtract 1 extra since we need to account for label
    num_examples = ((len(corpus) - 1) // num_steps)
    example_indices = list(range(0, num_examples * num_steps, num_steps))
    random.shuffle(example_indices)

    def data(pos):
        # This returns a sequence of the length num_steps starting from pos
        return corpus[pos: pos + num_steps]

    # Discard half empty batches
    num_batches = num_examples // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # Batch_size indicates the random examples read each time
        batch_indices = example_indices[i:(i+batch_size)]
        X = [data(j) for j in batch_indices]
        Y = [data(j + 1) for j in batch_indices]
        yield np.array(X), np.array(Y)
```


<!--
Let us generate an artificial sequence from 0 to 30.
We assume that the batch size and numbers of timesteps are 2 and 6 respectively.
This means that depending on the offset we can generate between 4 and 5 $(x, y)$ pairs.
With a minibatch size of 2, we only get 2 minibatches.
-->

Hãy tạo ra một chuỗi từ 0 đến 29, rồi sinh các minibatch từ chuỗi đó với kích thước batch là 2 và số bước thời gian là 6.
Nghĩa là tùy vào độ dời, ta có thể sinh tối đa 4 hoặc 5 cặp $(x, y)$.
Với kích thước batch bằng 2, ta thu được 2 minibatch.


```{.python .input  n=6}
my_seq = list(range(30))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y)
```

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
### Sequential Partitioning
-->

### Phân tách Tuần tự

<!--
In addition to random sampling of the original sequence, we can also make the positions of two adjacent random minibatches adjacent in the original sequence.
-->

Ngoài phép lấy mẫu ngẫu nhiên từ chuỗi gốc, chúng ta cũng có thể làm hai minibatch ngẫu nhiên liên tiếp có vị trí liền kề nhau trong chuỗi gốc.

```{.python .input  n=7}
# Saved in the d2l package for later use
def seq_data_iter_consecutive(corpus, batch_size, num_steps):
    # Offset for the iterator over the data for uniform starts
    offset = random.randint(0, num_steps)
    # Slice out data - ignore num_steps and just wrap around
    num_indices = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = np.array(corpus[offset:offset+num_indices])
    Ys = np.array(corpus[offset+1:offset+1+num_indices])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i:(i+num_steps)]
        Y = Ys[:, i:(i+num_steps)]
        yield X, Y
```

<!--
Using the same settings, print input `X` and label `Y` for each minibatch of examples read by random sampling.
The positions of two adjacent minibatches on the original sequence are adjacent.
-->

Sử dụng các đối số như ở trên, ta sẽ in đầu vào `X` và nhãn `Y` cho mỗi minibatch sau khi phân tách tuần tự.
Hai minibatch liên tiếp sẽ có vị trí trên chuỗi ban đầu liền kề nhau.

```{.python .input  n=8}
for X, Y in seq_data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y)
```

<!--
Now we wrap the above two sampling functions to a class so that we can use it as a Gluon data iterator later.
-->

Hãy gộp hai hàm lấy mẫu theo hai cách trên vào một lớp để duyệt dữ liệu trong Gluon ở các phần sau.


```{.python .input}
# Saved in the d2l package for later use
class SeqDataLoader:
    """A iterator to load sequence data."""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = d2l.seq_data_iter_random
        else:
            self.data_iter_fn = d2l.seq_data_iter_consecutive
        self.corpus, self.vocab = d2l.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)
```


<!--
Last, we define a function `load_data_time_machine` that returns both the data iterator and the vocabulary, so we can use it similarly as other functions with `load_data` prefix.
-->

Cuối cùng, ta sẽ viết hàm `load_data_time_machine` trả về cả iterator dữ liệu và bộ từ vựng để sử dụng như các hàm `load_data` khác.

```{.python .input}
# Saved in the d2l package for later use
def load_data_time_machine(batch_size, num_steps, use_random_iter=False,
                           max_tokens=10000):
    data_iter = SeqDataLoader(
        batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab
```

<!--
## Summary
-->

## Tóm tắt

<!--
* Language models are an important technology for natural language processing.
* $n$-grams provide a convenient model for dealing with long sequences by truncating the dependence.
* Long sequences suffer from the problem that they occur very rarely or never.
* Zipf's law governs the word distribution for not only unigrams but also the other $n$-grams.
* There is a lot of structure but not enough frequency to deal with infrequent word combinations efficiently via Laplace smoothing.
* The main choices for sequence partitioning are picking between consecutive and random sequences.
* Given the overall document length, it is usually acceptable to be slightly wasteful with the documents and discard half-empty minibatches.
-->

* Mô hình ngôn ngữ là một kĩ thuật quan trọng trong xử lý ngôn ngữ tự nhiên.
* $n$-gram là một mô hình khá tốt để xử lý các chuỗi dài bằng cách cắt giảm số phụ thuộc.
* Vấn đề của các chuỗi dài là chúng rất hiếm hoặc thậm chí không bao giờ xuất hiện.
* Định luật Zipf không chỉ mô tả phân phối từ 1-gram mà còn cả các $n$-gram khác.
* Có nhiều cấu trúc trong ngôn ngữ nhưng tần suất xuất hiện lại không đủ cao, để xử lý các tổ hợp từ hiếm ta sử dụng làm mượt Laplace.
* Hai giải pháp chủ yếu cho bài toán phân tách chuỗi là lấy mẫu ngẫu nhiên và phân tách tuần tự.
* Nếu tài liệu đủ dài, việc lãng phí một chút và loại bỏ các minibatch rỗng một nửa là điều chấp nhận được.

<!--
## Exercises
-->

## Bài tập

<!--
1. Suppose there are $100.000$ words in the training dataset. How much word frequency and multi-word adjacent frequency does a four-gram need to store?
2. Review the smoothed probability estimates. Why are they not accurate? Hint: we are dealing with a contiguous sequence rather than singletons.
3. How would you model a dialogue?
4. Estimate the exponent of Zipf's law for unigrams, bigrams, and trigrams.
5. What other minibatch data sampling methods can you think of?
6. Why is it a good idea to have a random offset?
    * Does it really lead to a perfectly uniform distribution over the sequences on the document?
    * What would you have to do to make things even more uniform?
7. If we want a sequence example to be a complete sentence, what kinds of problems does this introduce in minibatch sampling? Why would we want to do this anyway?
-->

1. Giả sử có $100.000$ từ trong tập dữ liệu huấn luyện. Mô hình 4-gram cần phải lưu trữ bao nhiêu tần số của từ đơn và cụm từ liền kề?
2. Hãy xem lại các ước lượng xác suất đã qua làm mượt. Tại sao chúng không chính xác? Gợi ý: chúng ta đang xử lý một chuỗi liền kề chứ không phải riêng lẻ.
3. Bạn sẽ mô hình hóa một cuộc đối thoại như thế nào?
4. Hãy ước tính luỹ thừa của định luật Zipf cho 1-gram, 2-gram, và 3-gram.
5. Hãy thử tìm các cách lấy mẫu minibatch khác.
6. Tại sao việc lấy giá trị độ dời ngẫu nhiên lại là một ý tưởng hay?
    * Liệu việc đó có làm các chuỗi dữ liệu văn bản tuân theo phân phối đều một cách hoàn hảo không?
    * Phải làm gì để có phân phối đều hơn?
7. Những vấn đề gì sẽ nảy sinh khi lấy mẫu minibatch từ một câu hoàn chỉnh? Có lợi ích gì khi lấy mẫu một câu hoàn chỉnh?

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2361)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
* Nguyễn Lê Quang Nhật
* Đinh Đắc
* Nguyễn Văn Quang
* Phạm Hồng Vinh
* Nguyễn Cảnh Thướng
* Phạm Minh Đức

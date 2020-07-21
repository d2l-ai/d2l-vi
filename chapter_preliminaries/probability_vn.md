<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Probability
-->

# Xác suất
:label:`sec_prob`

<!--
In some form or another, machine learning is all about making predictions.
We might want to predict the *probability* of a patient suffering a heart attack in the next year, given their clinical history. In anomaly detection, we might want to assess how *likely* a set of readings from an airplane's jet engine would be, were it operating normally. In reinforcement learning, we want an agent to act intelligently in an environment. This means we need to think about the probability of getting a high reward under each of the available action. And when we build recommender systems we also need to think about probability. For example, say *hypothetically* that we worked for a large online bookseller. We might want to estimate the probability that a particular user would buy a particular book. For this we need to use the language of probability.
Entire courses, majors, theses, careers, and even departments, are devoted to probability. So naturally, our goal in this section is not to teach the whole subject. Instead we hope to get you off the ground, to teach you just enough that you can start building your first deep learning models, and to give you enough of a flavor for the subject that you can begin to explore it on your own if you wish.
-->

Theo cách này hay cách khác, học máy đơn thuần là đưa ra các dự đoán.
Chúng ta có thể muốn dự đoán *xác suất* của một bệnh nhân có thể bị đau tim vào năm sau, khi đã biết tiền sử lâm sàng của họ.
Trong tác vụ phát hiện điều bất thường, chúng ta có thể muốn đánh giá *khả năng* các thông số động cơ máy bay ở mức nào, liệu có ở mức hoạt động bình thường không.
Trong học tăng cường, chúng ta muốn có một tác nhân hoạt động thông minh trong một môi trường.
Nghĩa là chúng ta cần tính tới xác suất đạt điểm thưởng cao nhất cho từng hành động có thể thực hiện.
Và khi xây dựng một hệ thống gợi ý chúng ta cũng cần quan tâm tới xác suất.
Ví dụ, *giả thiết* rằng chúng ta làm việc cho một hãng bán sách trực tuyến lớn.
Chúng ta có thể muốn ước lượng xác suất một khách hàng cụ thể muốn mua một cuốn sách cụ thể nào đó.
Để làm được điều này, chúng ta cần dùng tới ngôn ngữ xác suất.
Có những khóa học, chuyên ngành, luận văn, sự nghiệp, và cả các ban ngành đều dành toàn bộ cho xác suất.
Vì thế đương nhiên mục tiêu của chúng tôi trong chương này không phải để dạy toàn bộ môn xác suất.
Thay vào đó, chúng tôi hy vọng đưa tới cho bạn đọc các kiến thức nền tảng, đủ để bạn đọc có thể bắt đầu xây dựng mô hình học sâu đầu tiên của chính mình, và truyền cảm hứng cho bạn thêm yêu thích xác suất để có thể bắt đầu tự khám phá nếu muốn.

<!--
We have already invoked probabilities in previous sections without articulating what precisely they are or giving a concrete example. Let's get more serious now by considering the first case: distinguishing cats and dogs based on photographs. This might sound simple but it is actually a formidable challenge. To start with, the difficulty of the problem may depend on the resolution of the image.
-->

Chúng tôi đã nhắc tới xác suất trong các chương trước mà không nói rõ chính xác nó là gì hay là đưa ra một ví dụ cụ thể nào.
Giờ hãy cùng bắt đầu nghiêm túc hơn bằng cách xem xét trường hợp đầu tiên: phân biệt mèo và chó dựa trên các bức ảnh.
Điều này tưởng chừng đơn giản nhưng thực ra là một thách thức.
Để bắt đầu, độ phức tạp của bài toán này có thể phụ thuộc vào độ phân giải của ảnh.

<!--
![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat_dog_pixels.png)
-->

![Ảnh ở các độ phân giải khác nhau ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ điểm ảnh).](../img/cat_dog_pixels.png)
:width:`300px`
:label:`fig_cat_dog`

<!--
As shown in :numref:`fig_cat_dog`,
while it is easy for humans to recognize cats and dogs at the resolution of $160 \times 160$ pixels,
it becomes challenging at $40 \times 40$ pixels and next to impossible at $10 \times 10$ pixels. In
other words, our ability to tell cats and dogs apart at a large distance (and thus low resolution) might approach uninformed guessing. Probability gives us a
formal way of reasoning about our level of certainty.
If we are completely sure
that the image depicts a cat, we say that the *probability* that the corresponding label $y$ is "cat", denoted $P(y=$ "cat"$)$ equals $1$.
If we had no evidence to suggest that $y =$ "cat" or that $y =$ "dog", then we might say that the two possibilities were equally
*likely* expressing this as $P(y=$ "cat"$) = P(y=$ "dog"$) = 0.5$. If we were reasonably
confident, but not sure that the image depicted a cat, we might assign a
probability $0.5  < P(y=$ "cat"$) < 1$.
-->

Như thể hiện trong :numref:`fig_cat_dog`, con người phân biệt mèo và chó dễ dàng ở độ phân giải $160 \times 160$ điểm ảnh, có chút thử thách hơn ở $40 \times 40$ điểm ảnh, và gần như không thể ở $10 \times 10$ điểm ảnh.
Nói cách khác, khả năng phân biệt mèo và chó của chúng ta ở khoảng cách càng xa (đồng nghĩa với độ phân giải thấp) càng giống đoán mò.
Xác suất trang bị cho ta một cách suy luận hình thức về mức độ chắc chắn.
Nếu chúng ta hoàn toàn chắc chắn rằng bức ảnh mô tả một con mèo, ta có thể nói rằng *xác suất* nhãn tương ứng $y$ là "mèo", ký hiệu là $P(y=$ "mèo"$)$ equals $1$.
Nếu chúng ta không có manh mối nào để đoán rằng $y =$ "mèo" hoặc là $y =$ "chó", thì ta có thể nói rằng hai xác suất này có *khả năng* bằng nhau, biễu diễn bởi $P(y=$ "mèo"$) = P(y=$ "chó"$) = 0.5$.
Nếu ta khá tự tin, nhưng không thực sự chắc chắn bức ảnh mô tả một con mèo, ta có thể gán cho nó một xác suất $0.5  < P(y=$ "mèo"$) < 1$.

<!--
Now consider the second case: given some weather monitoring data, we want to predict the probability that it will rain in Taipei tomorrow. If it is summertime, the rain might come with probability $0.5$.
-->

Giờ hãy xem xét trường hợp thứ hai: cho dữ liệu theo dõi khí tượng, chúng ta muốn dự đoán xác suất ngày mai trời sẽ mưa ở Đài Bắc.
Nếu vào mùa hè, xác suất trời mưa có thể là $0.5$.

<!--
In both cases, we have some value of interest. And in both cases we are uncertain about the outcome.
But there is a key difference between the two cases. In this first case, the image is in fact either a dog or a cat, and we just do not know which. In the second case, the outcome may actually be a random event, if you believe in such things (and most physicists do). So probability is a flexible language for reasoning about our level of certainty, and it can be applied effectively in a broad set of contexts.
-->

Trong cả hai trường hợp, chúng ta đều quan tâm tới một đại lượng nào đó và cùng không chắc chắn về giá trị đầu ra.
Nhưng có một khác biệt quan trọng giữa hai trường hợp.
Trong trường hợp đầu tiên, bức ảnh chỉ có thể là chó hoặc mèo, và chúng ta chỉ không biết là loài nào.
Trong trường hợp thứ hai, đầu ra thực sự có thể là một sự kiện ngẫu nhiên, nếu bạn tin vào những thứ như vậy (và hầu hết các nhà vật lý tin vậy).
Như vậy xác suất là một ngôn ngữ linh hoạt để suy đoán về mức độ chắc chắn của chúng ta, và nó có thể được áp dụng hiệu quả trong vô vàn ngữ cảnh khác nhau.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Basic Probability Theory
-->

## Lý thuyết Xác suất cơ bản

<!--
Say that we cast a die and want to know what the chance is of seeing a $1$ rather than another digit. If the die is fair, all the $6$ outcomes $\{1, \ldots, 6\}$ are equally likely to occur, and thus we would see a $1$ in one out of six cases. Formally we state that $1$ occurs with probability $\frac{1}{6}$.
-->

Giả sử, ta tung xúc xắc và muốn biết cơ hội để thấy mặt số $1$ so với các mặt khác là bao nhiêu?
Nếu chiếc xúc xắc có chất liệu đồng nhất, thì cả $6$ mặt $\{1, \ldots, 6\}$ đều có khả năng xuất hiện như nhau, nên ta sẽ thấy mặt $1$ xuất hiện một lần trong mỗi sáu lần tung xúc xắc như trên.
Ta có thể nói rằng mặt $1$ xuất hiện với xác suất là $\frac{1}{6}$.

<!--
For a real die that we receive from a factory, we might not know those proportions and we would need to check whether it is tainted. The only way to investigate the die is by casting it many times and recording the outcomes. For each cast of the die, we will observe a value in $\{1, \ldots, 6\}$. Given these outcomes, we want to investigate the probability of observing each outcome.
-->

Với một chiếc xúc xắc thật, ta có thể không biết được tỷ lệ này và cần kiểm tra liệu xúc xắc có bị hư hỏng gì không.
Cách duy nhất để kiểm tra là tung thật nhiều lần rồi ghi lại kết quả.
Mỗi lần tung, ta quan sát thấy một số trong $\{1, \ldots, 6\}$ xuất hiện.
Với kết quả này, ta muốn kiểm chứng xác suất xuất hiện của từng mặt số.

<!--
One natural approach for each value is to take the
individual count for that value and to divide it by the total number of tosses.
This gives us an *estimate* of the probability of a given *event*. The *law of
large numbers* tell us that as the number of tosses grows this estimate will draw closer and closer to the true underlying probability. Before going into the details of what is going here, let's try it out.
-->

Cách tính trực quan nhất là lấy số lần xuất hiện của mỗi mặt số chia cho tổng số lần tung.
Cách này cho ta một *ước lượng* của xác suất ứng với một *sự kiện* cho trước.
*Luật số lớn* cho ta biết rằng số lần tung xúc xắc càng tăng thì ước lượng này càng gần hơn với xác xuất thực.
Trước khi giải thích chi tiết hơn, hãy cùng lập trình thí nghiệm này.

<!--
To start, let's import the necessary packages.
-->

Bắt đầu, ta nhập các gói lệnh cần thiết.

```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

<!--
Next, we will want to be able to cast the die. In statistics we call this process
of drawing examples from probability distributions *sampling*.
The distribution
that assigns probabilities to a number of discrete choices is called the
*multinomial distribution*. We will give a more formal definition of
*distribution* later, but at a high level, think of it as just an assignment of
probabilities to events. In MXNet, we can sample from the multinomial
distribution via the aptly named `np.random.multinomial` function.
The function
can be called in many ways, but we will focus on the simplest.
To draw a single sample, we simply pass in a vector of probabilities.
The output of the `np.random.multinomial` function is another vector of the same length:
its value at index $i$ is the number of times the sampling outcome corresponds to $i$.
-->

Tiếp theo, ta sẽ cần tung xúc xắc.
Trong thống kê, ta gọi quá trình thu các mẫu từ phân phối xác suất là quá trình *lấy mẫu*.
Phân phối mà gán các xác suất cho các lựa chọn rời rạc (*discrete choices*) được gọi là *phân phối đa thức* (*multinomial distribution*).
Sau này, ta sẽ đưa ra định nghĩa chính quy *phân phối* là gì; nhưng để hình dung, hãy xem nó như phép gán xác suất xảy ra cho các sự kiện.
Trong MXNet, ta có thể lấy mẫu từ phân phối đa thức với hàm `np.random.multinomial`.
Có nhiều cách sử dụng hàm này, nhưng ta tập trung vào cách dùng đơn giản nhất.
Muốn lấy một mẫu đơn, ta chỉ cần đưa vào hàm này một vector chứa các xác suất.
Hàm `np.random.multinomial` sẽ cho kết quả là một vector có chiều dài tương tự: trong vector này, giá trị tại chỉ số $i$ là số lần kết quả $i$ xuất hiện.

```{.python .input  n=2}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
If you run the sampler a bunch of times, you will find that you get out random
values each time. As with estimating the fairness of a die, we often want to
generate many samples from the same distribution. It would be unbearably slow to
do this with a Python `for` loop, so `random.multinomial` supports drawing
multiple samples at once, returning an array of independent samples in any shape
we might desire.
-->

Nếu chạy hàm lấy mẫu vài lần, bạn sẽ thấy rằng mỗi lần các giá trị trả về đều là ngẫu nhiên.
Giống với việc đánh giá một con xúc xắc có đều hay không, chúng ta thường muốn tạo nhiều mẫu từ cùng một phân phối.
Tạo dữ liệu như trên với vòng lặp `for` trong Python là rất chậm, vì vậy hàm `random.multinomial` hỗ trợ sinh nhiều mẫu trong một lần gọi, trả về một mảng chứa các mẫu độc lập với kích thước bất kỳ.

```{.python .input  n=3}
np.random.multinomial(10, fair_probs)
```

<!--
We can also conduct, say $3$, groups of experiments, where each group draws $10$ samples, all at once.
-->

Chúng ta cũng có thể giả sử làm $3$ thí nghiệm, trong đó mỗi thí nghiệm cùng lúc lấy ra $10$ mẫu.

```{.python .input  n=4}
counts = np.random.multinomial(10, fair_probs, size=3)
counts
```

<!--
Now that we know how to sample rolls of a die, we can simulate 1000 rolls. We
can then go through and count, after each of the 1000 rolls, how many times each
number was rolled.
Specifically, we calculate the relative frequency as the estimate of the true probability.
-->

Giờ chúng ta đã biết cách lấy mẫu các lần tung của một con xúc xắc, ta có thể giả lập 1000 lần tung.
Sau đó, chúng ta có thể đếm xem mỗi mặt xuất hiện bao nhiêu lần.
Cụ thể, chúng ta tính toán tần suất tương đối như là một ước lượng của xác suất thực.

```{.python .input  n=5}
# Store the results as 32-bit floats for division
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000  # Reletive frequency as the estimate
```

<!--
Because we generated the data from a fair die, we know that each outcome has true probability $\frac{1}{6}$, roughly $0.167$, so the above output estimates look good.
-->

Do dữ liệu được sinh bởi một con xúc xắc đều, ta biết mỗi đầu ra đều có xác suất thực bằng $\frac{1}{6}$, cỡ $0.167$, do đó kết quả ước lượng bên trên trông khá ổn.

<!--
We can also visualize how these probabilities converge over time towards the true probability.
Let's conduct $500$ groups of experiments where each group draws $10$ samples.
-->

Chúng ta cũng có thể minh họa những xác suất này hội tụ tới xác suất thực như thế nào.
Hãy cũng làm $500$ thí nghiệm trong đó mỗi thí nghiệm lấy ra $10$ mẫu.

```{.python .input  n=6}
counts = np.random.multinomial(10, fair_probs, size=500)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].asnumpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

<!--
Each solid curve corresponds to one of the six values of the die and gives our estimated probability that the die turns up that value as assessed after each group of experiments.
The dashed black line gives the true underlying probability.
As we get more data by conducting more experiments,
the $6$ solid curves converge towards the true probability.
-->

Mỗi đường cong liền tương ứng với một trong sáu giá trị của xúc xắc và chỉ ra xác suất ước lượng của sự kiện xúc xắc ra mặt tương ứng sau mỗi thí nghiệm.
Đường đứt đoạn màu đen tương ứng với xác suất thực.
Khi ta lấy thêm dữ liệu bằng cách thực hiện thêm các thí nghiệm, thì $6$ đường cong liền sẽ hội tụ tiến tới xác suất thực.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Axioms of Probability Theory
-->

### Các Tiên đề của Lý thuyết Xác suất

<!--
When dealing with the rolls of a die,
we call the set $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ the *sample space* or *outcome space*, where each element is an *outcome*.
An *event* is a set of outcomes from a given sample space.
For instance, "seeing a $5$" ($\{5\}$) and "seeing an odd number" ($\{1, 3, 5\}$) are both valid events of rolling a die.
Note that if the outcome of a random experiment is in event $\mathcal{A}$,
then event $\mathcal{A}$ has occurred.
That is to say, if $3$ dots faced up after rolling a die, since $3 \in \{1, 3, 5\}$,
we can say that the event "seeing an odd number" has occurred.
-->

Khi thực hiện tung một con xúc xắc, chúng ta gọi tập hợp $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ là *không gian mẫu* hoặc *không gian kết quả*, trong đó mỗi phần tử là một *kết quả*.
Một *sự kiện* là một tập hợp các kết quả của không gian mẫu.
Ví dụ, "tung được một số $5$" ($\{5\}$) và "tung được một số lẻ" ($\{1, 3, 5\}$) đều là những sự kiện hợp lệ khi tung một con xúc xắc. 
Chú ý rằng nếu kết quả của một phép tung ngẫu nhiên nằm trong sự kiện $\mathcal{A}$, sự kiện $\mathcal{A}$ đã xảy ra.
Như vậy, nếu mặt $3$ chấm ngửa lên sau khi xúc xắc được tung, chúng ta nói sự kiện "tung được một số lẻ" đã xảy ra bởi vì $3 \in \{1, 3, 5\}$.

<!--
Formally, *probability* can be thought of a function that maps a set to a real value.
The probability of an event $\mathcal{A}$ in the given sample space $\mathcal{S}$,
denoted as $P(\mathcal{A})$, satisfies the following properties:
-->

Một cách chính thống hơn, *xác suất* có thể được xem là một hàm số ánh xạ một tập hợp các sự kiện tới một số thực. 
Xác suất của sự kiện $\mathcal{A}$ trong không gian mẫu $\mathcal{S}$, được kí hiệu là $P(\mathcal{A})$, phải thoả mãn những tính chất sau:

<!--
* For any event $\mathcal{A}$, its probability is never negative, i.e., $P(\mathcal{A}) \geq 0$;
* Probability of the entire sample space is $1$, i.e., $P(\mathcal{S}) = 1$;
* For any countable sequence of events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ that are *mutually exclusive* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$), the probability that any happens is equal to the sum of their individual probabilities, i.e., $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.
-->

* Với mọi sự kiện $\mathcal{A}$, xác suất của nó là không âm, tức là: $P(\mathcal{A}) \geq 0$;
* Xác suất của toàn không gian mẫu luôn bằng $1$, tức: $P(\mathcal{S}) = 1$;
* Đối với mọi dãy sự kiện có thể đếm được $\mathcal{A}_1, \mathcal{A}_2, \ldots$ *xung khắc lẫn nhau* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ với mọi $i \neq j$), xác suất có ít nhất một sự kiện xảy ra sẽ là tổng của những giá trị xác suất riêng lẻ, hay: $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

<!--
These are also the axioms of probability theory, proposed by Kolmogorov in 1933.
Thanks to this axiom system, we can avoid any philosophical dispute on randomness;
instead, we can reason rigorously with a mathematical language.
For instance, by letting event $\mathcal{A}_1$ be the entire sample space and $\mathcal{A}_i = \emptyset$ for all $i > 1$, we can prove that $P(\emptyset) = 0$, i.e., the probability of an impossible event is $0$.
-->
Đây cũng là những tiên đề của lý thuyết xác suất, được đề xuất bởi Kolmogorov năm 1933.
Nhờ vào hệ thống tiên đề này, ta có thể tránh được những tranh luận chủ quan về sự ngẫu nhiên; và ta có thể có được những suy luận chặt chẽ sử dụng ngôn ngữ toán học.
Ví dụ, cho sự kiện $\mathcal{A}_1$ là toàn bộ không gian mẫu và $\mathcal{A}_i = \emptyset$ với mọi $i > 1$, chúng ta có thể chứng minh rằng $P(\emptyset) = 0$, nghĩa là xác suất của sự kiện không thể xảy ra bằng $0$.


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### Random Variables
-->

### Biến ngẫu nhiên

<!--
In our random experiment of casting a die, we introduced the notion of a *random variable*. A random variable can be pretty much any quantity and is not deterministic. It could take one value among a set of possibilities in a random experiment.
Consider a random variable $X$ whose value is in the sample space $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ of rolling a die. We can denote the event "seeing a $5$" as $\{X = 5\}$ or $X = 5$, and its probability as $P(\{X = 5\})$ or $P(X = 5)$.
By $P(X = a)$, we make a distinction between the random variable $X$ and the values (e.g., $a$) that $X$ can take.
However, such pedantry results in a cumbersome notation.
For a compact notation,
on one hand, we can just denote $P(X)$ as the *distribution* over the random variable $X$:
the distribution tells us the probability that $X$ takes any value.
On the other hand,
we can simply write $P(a)$ to denote the probability that a random variable takes the value $a$.
Since an event in probability theory is a set of outcomes from the sample space,
we can specify a range of values for a random variable to take.
For example, $P(1 \leq X \leq 3)$ denotes the probability of the event $\{1 \leq X \leq 3\}$,
which means $\{X = 1, 2, \text{or}, 3\}$. Equivalently, $P(1 \leq X \leq 3)$ represents the probability that the random variable $X$ can take a value from $\{1, 2, 3\}$.
-->

Trong thí nghiệm tung xúc xắc ngẫu nhiên, chúng ta đã giới thiệu khái niệm của một *biến ngẫu nhiên*.
Một biến ngẫu nhiên có thể dùng để biểu diễn cho hầu như bất kỳ đại lượng nào và giá trị của nó không cố định.
Nó có thể nhận một giá trị trong tập các giá trị khả dĩ từ một thí nghiệm ngẫu nhiên.
Hãy xét một biến ngẫu nhiên $X$ có thể nhận một trong những giá trị từ tập không gian mẫu $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ của thí nghiêm tung xúc xắc.
Chúng ta có thể biểu diễn sự kiện "trông thấy mặt $5$" là $\{X = 5\}$ hoặc $X = 5$, và xác suất của nó là $P(\{X = 5\})$ hoặc $P(X = 5)$.
Khi viết $P(X = a)$, chúng ta đã phân biệt giữa biến ngẫu nhiên $X$ và các giá trị (ví dụ như $a$) mà $X$ có thể nhận.
Tuy nhiên, ký hiệu như vậy khá là rườm rà.
Để đơn giản hóa ký hiệu, một mặt, chúng ta có thể chỉ cần dùng $P(X)$ để biểu diễn *phân phối* của biến ngẫu nhiên $X$: phân phối này cho chúng ta biết xác xuất mà $X$ có thể nhận cho bất kỳ giá trị nào.
Mặt khác, chúng ta có thể đơn thuần viết $P(a)$ để biểu diễn xác suất mà một biến ngẫu nhiên nhận giá trị $a$.
Bởi vì một sự kiện trong lý thuyết xác suất là một tập các kết quả từ không gian mẫu, chúng ta có thể xác định rõ một khoảng các giá trị mà một biến ngẫu nhiên có thể nhận.
Ví dụ, $P(1 \leq X \leq 3)$ diễn tả xác suất của sự kiện $\{1 \leq X \leq 3\}$, nghĩa là $\{X = 1, 2, \text{hoặc}, 3\}$.
Tương tự, $P(1 \leq X \leq 3)$ biểu diễn xác suất mà biến ngẫu nhiên $X$ có thể nhận giá trị trong tập $\{1, 2, 3\}$.

<!--
Note that there is a subtle difference between *discrete* random variables, like the sides of a die, and *continuous* ones, like the weight and the height of a person. There is little point in asking whether two people have exactly the same height. If we take precise enough measurements you will find that no two people on the planet have the exact same height. In fact, if we take a fine enough measurement, you will not have the same height when you wake up and when you go to sleep. So there is no purpose in asking about the probability
that someone is $1.80139278291028719210196740527486202$ meters tall. Given the world population of humans the probability is virtually $0$. It makes more sense in this case to ask whether someone's height falls into a given interval, say between $1.79$ and $1.81$ meters. In these cases we quantify the likelihood that we see a value as a *density*. The height of exactly $1.80$ meters has no probability, but nonzero density. In the interval between any two different heights we have nonzero probability.
In the rest of this section, we consider probability in discrete space.
For probability over continuous random variables, you may refer to :numref:`sec_random_variables`.
-->

Lưu ý rằng có một sự khác biệt tinh tế giữa các biến ngẫu nhiên *rời rạc*, ví dụ như các mặt của xúc xắc, và các biến ngẫu nhiên *liên tục*, ví dụ như cân nặng và chiều cao của một người.
Việc hỏi rằng hai người có cùng chính xác chiều cao hay không khá là vô nghĩa.
Nếu ta đo với đủ độ chính xác, ta sẽ thấy rằng không có hai người nào trên hành tinh này mà có cùng chính xác chiều cao cả.
Thật vậy, nếu đo đủ chính xác, chiều cao của bạn lúc mới thức dậy và khi đi ngủ sẽ khác nhau.
Cho nên không có lý do gì để tìm xác suất một người nào đó cao $1.80139278291028719210196740527486202$ mét cả.
Trong toàn bộ dân số trên thế giới, xác suất này gần như bằng $0$.
Sẽ có lý hơn nếu ta hỏi chiều cao của một người nào đó có rơi vào một khoảng cho trước hay không, ví dụ như giữa $1.79$ và $1.81$ mét.
Trong các trường hợp này, ta có thể định lượng khả năng mà ta thấy một giá trị nào đó theo một *mật độ xác suất*.
Xác suất để có chiều cao chính xác $1.80$ mét không tồn tại, nhưng mật độ của sự kiện này khác không.
Trong bất kỳ khoảng nào giữa hai chiều cao khác nhau ta đều có xác suất khác không.
Trong phần còn lại của mục này, ta sẽ xem xét xác suất trong không gian rời rạc.
Về xác suất của biến ngẫu nhiên liên tục, bạn có thể xem ở :numref:`sec_random_variables`.

<!-- Kết thúc revise phần 3 -->
<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## Dealing with Multiple Random Variables
-->

## Làm việc với Nhiều Biến Ngẫu nhiên

<!--
Very often, we will want to consider more than one random variable at a time.
For instance, we may want to model the relationship between diseases and symptoms. Given a disease and a symptom, say "flu" and "cough", either may or may not occur in a patient with some probability. While we hope that the probability of both would be close to zero, we may want to estimate these probabilities and their relationships to each other so that we may apply our inferences to effect better medical care.
-->

Chúng ta sẽ thường xuyên phải làm việc với nhiều hơn một biến ngẫu nhiên cùng lúc.
Ví dụ, chúng ta có thể muốn mô hình hóa mối quan hệ giữa các loại bệnh và các triệu chứng bệnh.
Cho một loại bệnh và một triệu chứng bệnh, giả sử "cảm cúm" và "ho", chúng có thể xuất hiện hoặc không trên một bệnh nhân với xác suất nào đó.
Mặc dù chúng ta hy vọng xác suất cả hai xảy ra gần bằng không, ta có thể vẫn muốn ước lượng các xác suất này và mối quan hệ giữa chúng để có thể thực hiện các biện pháp chăm sóc y tế tốt hơn.

<!--
As a more complicated example, images contain millions of pixels, thus millions of random variables. And in many cases images will come with a
label, identifying objects in the image. We can also think of the label as a
random variable. We can even think of all the metadata as random variables
such as location, time, aperture, focal length, ISO, focus distance, and camera type.
All of these are random variables that occur jointly. When we deal with multiple random variables, there are several quantities of interest.
-->

Xét một ví dụ phức tạp hơn: các bức ảnh chứa hàng triệu điểm ảnh, tương ứng với hàng triệu biến ngẫu nhiên.
Và trong nhiều trường hợp các bức ảnh sẽ được gán một nhãn chứa tên các vật xuất hiện trong ảnh.
Chúng ta cũng có thể xem nhãn này như một biến ngẫu nhiên.
Thậm chí, ta còn có thể xem tất cả các siêu dữ liệu như địa điểm, thời gian, khẩu độ, tiêu cự, ISO, khoảng lấy nét và loại máy ảnh, là các biến ngẫu nhiên.
Tất cả các những biến ngẫu nhiên này xảy ra đồng thời.
Khi làm việc với nhiều biến ngẫu nhiên, có một số đại lượng đáng được quan tâm.

<!--
### Joint Probability
-->

### Xác suất Đồng thời

<!--
The first is called the *joint probability* $P(A = a, B=b)$. Given any values $a$ and $b$, the joint probability lets us answer, what is the probability that $A=a$ and $B=b$ simultaneously?
Note that for any values $a$ and $b$, $P(A=a, B=b) \leq P(A=a)$.
This has to be the case, since for $A=a$ and $B=b$ to happen, $A=a$ has to happen *and* $B=b$ also has to happen (and vice versa). Thus, $A=a$ and $B=b$ cannot be more likely than $A=a$ or $B=b$ individually.
-->
Đầu tiên là *xác suất đồng thời* $P(A = a, B=b)$.
Cho hai biến $a$ và $b$ bất kỳ, xác suất đồng thời cho ta biết xác suất để cả $A=a$ và $B=b$ đều xảy ra.
Ta có thể thấy rằng với mọi giá trị $a$ và $b$, $P(A=a, B=b) \leq P(A=a)$.
Bởi để $A=a$ và $B=b$ xảy ra thì $A=a$ phải xảy ra *và* $B=b$ cũng phải xảy ra (và ngược lại).
Do đó, khả năng $A=a$ và $B=b$ xảy ra đồng thời không thể lớn hơn khả năng $A=a$ hoặc $B=b$ xảy ra một cách độc lập được.

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
### Conditional Probability
-->

### Xác suất có điều kiện

<!--
This brings us to an interesting ratio: $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$. We call this ratio a *conditional probability*
and denote it by $P(B=b \mid A=a)$: it is the probability of $B=b$, provided that
$A=a$ has occurred.
-->

Điều này giúp ta thu được một tỉ lệ thú vị: $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$.
Chúng ta gọi tỉ lệ này là *xác suất có điều kiện* và ký hiệu là $P(B=b \mid A=a)$: xác suất để $B=b$, với điều kiện $A=a$ đã xảy ra.

<!--
### Bayes' theorem
-->

### Định lý Bayes

<!--
Using the definition of conditional probabilities, we can derive one of the most useful and celebrated equations in statistics: *Bayes' theorem*.
It goes as follows.
By construction, we have the *multiplication rule* that $P(A, B) = P(B \mid A) P(A)$. By symmetry, this also holds for $P(A, B) = P(A \mid B) P(B)$. Assume that $P(B) > 0$. Solving for one of the conditional variables we get
-->

Sử dụng định nghĩa của xác suất có điều kiện, chúng ta có thể thu được một trong những phương trình nổi tiếng và hữu dụng nhất trong thống kê: *định lý Bayes*.
Cụ thể như sau:
Theo định nghĩa chúng ta có *quy tắc nhân* $P(A, B) = P(B \mid A) P(A)$.
Tương tự, ta cũng có $P(A, B) = P(A \mid B) P(B)$.
Giả sử $P(B) > 0$.
Kết hợp các điều kiện trên ta có:

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

<!--
Note that here we use the more compact notation where $P(A, B)$ is a *joint distribution* and $P(A \mid B)$ is a *conditional distribution*. Such distributions can be evaluated for particular values $A = a, B=b$.
-->

Lưu ý rằng ở đây chúng ta sử dụng ký hiệu ngắn gọn hơn, với $P(A, B)$ là *xác suất đồng thời* và $P(A \mid B)$ là *xác suất có điều kiện*.
Các phân phối này có thể được tính tại các giá trị cụ thể $A = a, B=b$.

<!--
### Marginalization
-->

### Phép biên hóa

<!--
Bayes' theorem is very useful if we want to infer one thing from the other, say cause and effect, but we only know the properties in the reverse direction, as we will see later in this section. One important operation that we need, to make this work, is *marginalization*.
It is the operation of determining $P(B)$ from $P(A, B)$. We can see that the probability of $B$ amounts to accounting for all possible choices of $A$ and aggregating the joint probabilities over all of them:
-->

Định lý Bayes rất hữu ích nếu chúng ta muốn suy luận một điều gì đó từ một điều khác, như là nguyên nhân và kết quả, nhưng ta chỉ biết các đặc tính theo chiều ngược lại, như ta sẽ thấy trong phần sau của chương này.
Chúng ta cần làm một thao tác quan trọng để đạt được điều này, đó là *phép biên hóa*.
Có thể hiểu là việc xác định $P(B)$ từ $P(A, B)$.
Chúng ta có thể tính được xác suất của B bằng tổng xác suất kết hợp của A và B tại mọi giá trị có thể của A:

$$P(B) = \sum_{A} P(A, B),$$

<!--
which is also known as the *sum rule*. The probability or distribution as a result of marginalization is called a *marginal probability* or a *marginal distribution*.
-->

Công thức này cũng được biết đến với tên gọi *quy tắc tổng*.
Xác suất hay phân phối thu được từ thao tác biên hóa được gọi là *xác suất biên* hoặc *phân phối biên*.

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC =================================== -->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
### Independence
-->

### Tính độc lập

<!--
Another useful property to check for is *dependence* vs. *independence*.
Two random variables $A$ and $B$ are independent
means that the occurrence of one event of $A$
does not reveal any information about the occurrence of an event of $B$.
In this case $P(B \mid A) = P(B)$. Statisticians typically express this as $A \perp  B$. From Bayes' theorem, it follows immediately that also $P(A \mid B) = P(A)$.
In all the other cases we call $A$ and $B$ dependent. For instance, two successive rolls of a die are independent. In contrast, the position of a light switch and the brightness in the room are not (they are not perfectly deterministic, though, since we could always have a broken light bulb, power failure, or a broken switch).
-->
Một tính chất hữu ích khác cần kiểm tra là *tính phụ thuộc* và *tính độc lập*.
Hai biến ngẫu nhiên $A$ và $B$ độc lập nghĩa là việc một sự kiện của $A$ xảy ra không tiết lộ bất kỳ thông tin nào về việc xảy ra một sự kiện của $B$.
Trong trường hợp này $P(B \mid A) = P(B)$.
Các nhà thống kê thường biểu diễn điều này bằng ký hiệu $A \perp B$. 
Từ định lý Bayes, ta có $P(A \mid B) = P(A)$.
Trong tất cả các trường hợp khác, chúng ta gọi $A$ và $B$ là hai biến phụ thuộc. 
Ví dụ, hai lần đổ liên tiếp của một con xúc xắc là độc lập. 
Ngược lại, vị trí của công tắc đèn và độ sáng trong phòng là không độc lập (tuy nhiên chúng không hoàn toàn xác định, vì bóng đèn luôn có thể bị hỏng, mất điện hoặc công tắc bị hỏng).

<!--
Since $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ is equivalent to $P(A, B) = P(A)P(B)$, two random variables are independent if and only if their joint distribution is the product of their individual distributions.
Likewise, two random variables $A$ and $B$ are *conditionally independent* given another random variable $C$
if and only if $P(A, B \mid C) = P(A \mid C)P(B \mid C)$. This is expressed as $A \perp B \mid C$.
-->

Vì $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ tương đương với $P(A, B) = P(A)P(B)$, hai biến ngẫu nhiên là độc lập khi và chỉ khi phân phối đồng thời của chúng là tích các phân phối riêng lẻ của chúng.
Tương tự, cho một biến ngẫu nhiên $C$ khác, hai biến ngẫu nhiên $A$ và $B$ là *độc lập có điều kiện* khi và chỉ khi $P(A, B \mid C) = P(A \mid C)P(B \mid C)$.
Điều này được ký hiệu là $A \perp B \mid C$.

<!--
### Application
-->

### Ứng dụng
:label:`subsec_probability_hiv_app`

<!--
Let's put our skills to the test. Assume that a doctor administers an AIDS test to a patient. This test is fairly accurate and it fails only with $1\%$ probability if the patient is healthy but reporting him as diseased. Moreover,
it never fails to detect HIV if the patient actually has it. We use $D_1$ to indicate the diagnosis ($1$ if positive and $0$ if negative) and $H$ to denote the HIV status ($1$ if positive and $0$ if negative).
:numref:`conditional_prob_D1` lists such conditional probability.
-->

Hãy thử nghiệm các kiến thức chúng ta vừa học.
Giả sử rằng một bác sĩ phụ trách xét nghiệm AIDS cho một bệnh nhân. 
Việc xét nghiệm này khá chính xác và nó chỉ thất bại với xác suất $1\%$, khi nó cho kết quả dương tính dù bệnh nhân khỏe mạnh. 
Hơn nữa, nó không bao giờ thất bại trong việc phát hiện HIV nếu bệnh nhân thực sự bị nhiễm bệnh. 
Ta sử dụng $D_1$ để biểu diễn kết quả chẩn đoán ($1$ nếu dương tính và $0$ nếu âm tính) và $H$ để biểu thị tình trạng nhiễm HIV ($1$ nếu dương tính và $0$ nếu âm tính).
:numref:`conditional_prob_D1` liệt kê xác suất có điều kiện đó.

<!--
:Conditional probability of $P(D_1 \mid H)$.

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`
-->

:Xác suất có điều kiện của $P(D_1 \mid H)$.

| Xác suất có điều kiện | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

<!--
Note that the column sums are all $1$ (but the row sums are not), since the conditional probability needs to sum up to $1$, just like the probability. Let's work out the probability of the patient having AIDS if the test comes back positive, i.e., $P(H = 1 \mid D_1 = 1)$. Obviously this is going to depend on how common the disease is, since it affects the number of false alarms. Assume that the population is quite healthy, e.g., $P(H=1) = 0.0015$. To apply Bayes' Theorem, we need to apply marginalization and the multiplication rule to determine
-->

Lưu ý rằng tổng của từng cột đều bằng $1$ (nhưng tổng từng hàng thì không), vì xác suất có điều kiện cần có tổng bằng $1$, giống như xác suất.
Hãy cùng tìm xác suất bệnh nhân bị AIDS nếu xét nghiệm trả về kết quả dương tính, tức $P(H = 1 \mid D_1 = 1)$.
Rõ ràng điều này sẽ phụ thuộc vào mức độ phổ biến của bệnh, bởi vì nó ảnh hưởng đến số lượng dương tính giả.
Giả sử rằng dân số khá khỏe mạnh, ví dụ: $P(H=1) = 0.0015$.
Để áp dụng Định lý Bayes, chúng ta cần áp dụng phép biên hóa và quy tắc nhân để xác định

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!--
Thus, we get
-->

Do đó, ta có

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$


<!--
In other words, there is only a 13.06% chance that the patient actually has AIDS, despite using a very accurate test. As we can see, probability can be quite counterintuitive.
-->

Nói cách khác, chỉ có 13,06% khả năng bệnh nhân thực sự mắc bệnh AIDS, dù ta dùng một bài kiểm tra rất chính xác. 
Như ta có thể thấy, xác suất có thể trở nên khá phản trực giác.
<!--
What should a patient do upon receiving such terrifying news? Likely, the patient
would ask the physician to administer another test to get clarity. The second
test has different characteristics and it is not as good as the first one, as shown in :numref:`conditional_prob_D2`.
-->

Một bệnh nhân phải làm gì nếu nhận được tin dữ như vậy? 
Nhiều khả năng họ sẽ yêu cầu bác sĩ thực hiện một xét nghiệm khác để làm rõ sự việc.
Bài kiểm tra thứ hai có những đặc điểm khác và không tốt bằng bài thứ nhất, như ta có thể thấy trong :numref:`conditional_prob_D2`.


<!--
:Conditional probability of $P(D_2 \mid H)$.
-->
<!--
| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

<!--
Unfortunately, the second test comes back positive, too. Let's work out the requisite probabilities to invoke Bayes' Theorem by assuming the conditional independence:
-->

:Xác suất có điều kiện của $P(D_2 \mid H)$.

| Xác xuất có điều kiện | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

Không may thay, bài kiểm tra thứ hai cũng có kết quả dương tính. 
Hãy cùng tính các xác suất cần thiết để sử dụng định lý Bayes bằng cách giả định tính độc lập có điều kiện:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 0) \\
=& P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)  \\
=& 0.0003,
\end{aligned}
$$

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1 \mid H = 1) \\
=& P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)  \\
=& 0.98.
\end{aligned}
$$

<!--
Now we can apply marginalization and the multiplication rule:
-->

Bây giờ chúng ta có thể áp dụng phép biên hóa và quy tắc nhân xác suất:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

<!--
In the end, the probability of the patient having AIDS given both positive tests is
-->

Cuối cùng xác suất bệnh nhân mắc bệnh AIDS qua hai lần dương tính là

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

<!--
That is, the second test allowed us to gain much higher confidence that not all is well. Despite the second test being considerably less accurate than the first one, it still significantly improved our estimate.
-->
Cụ thể hơn, thử nghiệm thứ hai mang lại độ tin cậy cao hơn rằng không phải mọi chuyện đều ổn. 
Mặc dù bài kiểm tra thứ hai kém chính xác hơn bài đầu, nó vẫn cải thiện đáng kể dự đoán.

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 10 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 6 - BẮT ĐẦU ===================================-->

<!--
## Expectation and Variance
-->

## Kỳ vọng và Phương sai

<!--
To summarize key characteristics of probability distributions,
we need some measures.
The *expectation* (or average) of the random variable $X$ is denoted as
-->
Để tóm tắt những đặc tính then chốt của các phân phối xác suất, chúng ta cần một vài phép đo.
*Kỳ vọng* (hay trung bình) của một biến ngẫu nhiên $X$, được ký hiệu là

$$E[X] = \sum_{x} x P(X = x).$$

<!--
When the input of a function $f(x)$ is a random variable drawn from the distribution $P$ with different values $x$,
the expectation of $f(x)$ is computed as
-->

Khi giá trị đầu vào của phương trình $f(x)$ là một biến ngẫu nhiên nhiên cho trước theo phân phối $P$ với các giá trị $x$ khác nhau, kỳ vọng của $f(x)$ sẽ được tính theo phương trình:

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$


<!--
In many cases we want to measure by how much the random variable $X$ deviates from its expectation. This can be quantified by the variance
-->

Trong nhiều trường hợp, chúng ta muốn đo độ lệch của biến ngẫu nhiên $X$ so với kỳ vọng của nó.
Đại lượng này có thể được đo bằng phương sai

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$


<!--
Its square root is called the *standard deviation*.
The variance of a function of a random variable measures
by how much the function deviates from the expectation of the function,
as different values $x$ of the random variable are sampled from its distribution:
-->

Nếu lấy căn bậc hai của kết quả ta sẽ được độ lệch chuẩn.
Phương sai của một hàm của một biến ngẫu nhiên đo độ lệch của hàm số đó từ kỳ vọng của nó khi các giá trị $x$ khác nhau được lấy mẫu từ phân phối của biến ngẫu nhiên đó:

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$


<!--
## Summary
-->

## Tóm tắt

<!--
* We can use MXNet to sample from probability distributions.
* We can analyze multiple random variables using joint distribution, conditional distribution, Bayes' theorem, marginalization, and independence assumptions.
* Expectation and variance offer useful measures to summarize key characteristics of probability distributions.
-->

* Chúng ta có thể sử dụng MXNet để lấy mẫu từ phân phối xác suất.
* Các biến ngẫu nhiên có thể được phân tích bằng các phương pháp như phân phối đồng thời (*joint distribution*), phân phối có điều kiện (*conditional distribution*), định lý Bayes, phép biên hóa (*marginalization*) và giả định độc lập (*independence assumptions*).
* Kỳ vọng và phương sai là các phép đo hữu ích để tóm tắt các đặc điểm chính của phân phối xác suất.


<!--
## Exercises
-->

## Bài tập

<!--
1. We conducted $m=500$ groups of experiments where each group draws $n=10$ samples. Vary $m$ and $n$. Observe and analyze the experimental results.
1. Given two events with probability $P(\mathcal{A})$ and $P(\mathcal{B})$, compute upper and lower bounds on $P(\mathcal{A} \cup \mathcal{B})$ and $P(\mathcal{A} \cap \mathcal{B})$. (Hint: display the situation using a [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram).)
1. Assume that we have a sequence of random variables, say $A$, $B$, and $C$, where $B$ only depends on $A$, and $C$ only depends on $B$, can you simplify the joint probability $P(A, B, C)$? (Hint: this is a [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain).)
1. In :numref:`subsec_probability_hiv_app`, the first test is more accurate. Why not just run the first test a second time?
-->

1. Tiến hành $m=500$ nhóm thí nghiệm với mỗi nhóm lấy ra $n=10$ mẫu.
Thay đổi $m$ và $n$. Quan sát và phân tích kết quả của thí nghiệm.
2. Cho hai sự kiện với xác suất $P(\mathcal{A})$ và $P(\mathcal{B})$, tính giới hạn trên và dưới của $P(\mathcal{A} \cup \mathcal{B})$ và $P(\mathcal{A} \cap \mathcal{B})$.
(Gợi ý: sử dụng [biểu đồ Venn](https://en.wikipedia.org/wiki/Venn_diagram).)
3. Giả sử chúng ta có các biến ngẫu nhiên $A$, $B$ và $C$, với $B$ chỉ phụ thuộc $A$, và $C$ chỉ phụ thuộc vào $B$.
Làm thế nào để đơn giản hóa xác suất đồng thời của $P(A, B, C)$?
(Gợi ý: đây là một [Chuỗi Markov](https://en.wikipedia.org/wiki/Markov_chain).)
4. Trong :numref:`subsec_probability_hiv_app`, bài xét nghiệm đầu tiên có độ chính xác cao hơn.
Vậy tại sao chúng ta không sử dụng bài xét nghiệm đầu tiên cho lần thử tiếp theo?

<!-- ===================== Kết thúc dịch Phần 10 ===================== -->

<!-- ========================================= REVISE PHẦN 6 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2319)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2319)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

* Đoàn Võ Duy Thanh
* Nguyễn Văn Tâm
* Vũ Hữu Tiệp
* Nguyễn Cảnh Thướng
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Mai Sơn Hải
* Trần Kiến An
* Tạ H. Duy Nguyên
* Phạm Minh Đức
* Trần Thị Hồng Hạnh
* Lê Thành Vinh
* Nguyễn Minh Thư

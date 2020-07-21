<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Sequence Models
-->

# Mô hình chuỗi
:label:`sec_sequence`

<!--
Imagine that you are watching movies on Netflix.
As a good Netflix user, you decide to rate each of the movies religiously.
After all, a good movie is a good movie, and you want to watch more of them, right?
As it turns out, things are not quite so simple.
People's opinions on movies can change quite significantly over time.
In fact, psychologists even have names for some of the effects:
-->


Hãy tưởng tượng rằng bạn đang xem phim trên Netflix.
Là một người dùng Netflix tốt, bạn quyết định đánh giá từng bộ phim một cách cẩn thận.
Xét cho cùng, bạn muốn xem thêm nhiều bộ phim hay phải không?
Nhưng hóa ra, mọi thứ không hề đơn giản như vậy.
Đánh giá của mỗi người về một bộ phim có thể thay đổi đáng kể theo thời gian.
Trên thực tế, các nhà tâm lý học thậm chí còn đặt tên cho một số hiệu ứng:

<!--
* There is [anchoring](https://en.wikipedia.org/wiki/Anchoring), based on someone else's opinion. 
For instance after the Oscar awards, ratings for the corresponding movie go up, even though it is still the same movie. 
This effect persists for a few months until the award is forgotten. 
:cite:`Wu.Ahmed.Beutel.ea.2017` showed that the effect lifts rating by over half a point.
* There is the [Hedonic adaptation](https://en.wikipedia.org/wiki/Hedonic_treadmill), where humans quickly adapt to accept an improved (or a bad) situation as the new normal. 
For instance, after watching many good movies, 
the expectations that the next movie is equally good or better are high, hence even an average movie might be considered a bad movie after many great ones.
* There is seasonality. Very few viewers like to watch a Santa Claus movie in August.
* In some cases movies become unpopular due to the misbehaviors of directors or actors in the production.
* Some movies become cult movies, because they were almost comically bad. *Plan 9 from Outer Space* and *Troll 2* achieved a high degree of notoriety for this reason.
-->

* [Hiệu ứng mỏ neo](https://en.wikipedia.org/wiki/Anchoring): dựa trên ý kiến của người khác.
Ví dụ, xếp hạng của một bộ phim sẽ tăng lên sau khi nó thắng giải Oscar, mặc dù đoàn làm phim này không có bất kỳ tác động nào về mặt quảng bá đến bộ phim.
Hiệu ứng này kéo dài trong vòng một vài tháng cho đến khi giải thưởng bị lãng quên.
:cite:`Wu.Ahmed.Beutel.ea.2017` chỉ ra rằng hiệu ứng này tăng chỉ số xếp hạng thêm hơn nửa điểm.
* [Hiệu ứng vòng xoáy khoái lạc](https://en.wikipedia.org/wiki/Hedonic_treadmill): con người nhanh chóng thích nghi để chấp nhận một tình huống tốt hơn (hoặc xấu đi) như một điều bình thường mới.
Chẳng hạn, sau khi xem nhiều bộ phim hay, sự kỳ vọng rằng bộ phim tiếp theo sẽ hay tương đương hoặc thậm chí phải hay hơn trở nên khá cao, do đó ngay cả một bộ phim trung bình cũng có thể bị coi là một bộ phim tồi.
* Tính thời vụ: rất ít khán giả thích xem một bộ phim về ông già Noel vào tháng 8.
* Trong một số trường hợp, các bộ phim trở nên không được ưa chuộng do những hành động sai trái của các đạo diễn hoặc diễn viên tham gia vào quá trình sản xuất phim.
* Một số phim trở thành "phim cult" vì chúng gần như tệ đến mức phát cười. *Plan 9 from Outer Space* và *Troll 2* là hai ví dụ nổi tiếng.

<!--
In short, ratings are anything but stationary.
Using temporal dynamics helped :cite:`Koren.2009` to recommend movies more accurately.
But it is not just about movies.
-->

Tóm lại, thứ bậc xếp hạng không hề cố định.
Sử dụng các động lực dựa trên thời gian đã giúp :cite:`Koren.2009` đề xuất phim chính xác hơn.
Tuy nhiên, vấn đề không chỉ là về phim ảnh.

<!--
* Many users have highly particular behavior when it comes to the time when they open apps. 
For instance, social media apps are much more popular after school with students. 
Stock market trading apps are more commonly used when the markets are open.
* It is much harder to predict tomorrow's stock prices than to fill in the blanks for a stock price we missed yesterday, even though both are just a matter of estimating one number. 
After all, hindsight is so much easier than foresight. 
In statistics the former is called *extrapolation* whereas the latter is called *interpolation*.
* Music, speech, text, movies, steps, etc. are all sequential in nature. 
If we were to permute them they would make little sense. 
The headline *dog bites man* is much less surprising than *man bites dog*, even though the words are identical.
* Earthquakes are strongly correlated, i.e., after a massive earthquake there are very likely several smaller aftershocks, much more so than without the strong quake. 
In fact, earthquakes are spatiotemporally correlated, i.e., the aftershocks typically occur within a short time span and in close proximity.
* Humans interact with each other in a sequential nature, as can be seen in Twitter fights, dance patterns and debates.
-->

* Nhiều người dùng có thói quen rất đặc biệt liên quan tới thời gian mở ứng dụng.
Chẳng hạn, học sinh sử dụng các ứng dụng mạng xã hội nhiều hơn hẳn sau giờ học.
Các ứng dụng giao dịch chứng khoán được sử dụng nhiều khi thị trường mở cửa.
* Việc dự đoán giá cổ phiếu ngày mai khó hơn nhiều so với việc dự đoán giá cổ phiếu bị bỏ lỡ ngày hôm qua, mặc dù cả hai đều là bài toán ước tính một con số.
Rốt cuộc, nhìn lại quá khứ dễ hơn nhiều so với dự đoán tương lai.
Trong thống kê, bài toán đầu tiên được gọi là *ngoại suy* và bài toán sau được gọi là *nội suy*.
* Âm nhạc, giọng nói, văn bản, phim ảnh, bước đi, v.v ... đều có tính chất tuần tự.
Nếu chúng ta hoán vị chúng, chúng sẽ không còn nhiều ý nghĩa.
Dòng tiêu đề *chó cắn người* ít gây ngạc nhiên hơn nhiều so với *người cắn chó*, mặc dù các từ giống hệt nhau.
* Các trận động đất có mối tương quan mạnh mẽ, tức sau một trận động đất lớn, rất có thể sẽ có một số dư chấn nhỏ hơn và xác suất xảy ra dư chấn cao hơn nhiều so với trường hợp trận động đất lớn không xảy ra trước đó.
Trên thực tế, các trận động đất có mối tương quan về mặt không-thời gian, tức các dư chấn thường xảy ra trong một khoảng thời gian ngắn và ở gần nhau.
* Con người tương tác với nhau một cách tuần tự, điều này có thể được thấy trong các cuộc tranh cãi trên Twitter, các điệu nhảy và các cuộc tranh luận.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Statistical Tools
-->

## Các công cụ thống kê

<!--
In short, we need statistical tools and new deep neural networks architectures to deal with sequence data.
To keep things simple, we use the stock price illustrated in :numref:`fig_ftse100` as an example.
-->


Tóm lại, ta cần các công cụ thống kê và các kiến trúc mạng nơ-ron sâu mới để xử lý dữ liệu chuỗi.
Để đơn giản hóa mọi việc, ta sẽ sử dụng giá cổ phiếu được minh họa trong :numref:`fig_ftse100` để làm ví dụ.

<!--
![FTSE 100 index over 30 years](../img/ftse100.png)
-->


![Giá cổ phiếu FTSE 100 trong vòng 30 năm](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`

<!--
Let us denote the prices by $x_t \geq 0$, i.e., at time $t \in \mathbb{N}$ we observe price $x_t$.
For a trader to do well in the stock market on day $t$ he should want to predict $x_t$ via
-->


Ta sẽ gọi giá cổ phiếu là $x_t \geq 0$, tức tại thời điểm $t \in \mathbb{N}$ ta thấy giá cổ phiếu bằng $x_t$.
Để có thể kiếm lời trên thị trường chứng khoán vào ngày $t$, một nhà giao dịch sẽ muốn dự đoán $x_t$ thông qua

$$x_t \sim p(x_t \mid x_{t-1}, \ldots, x_1).$$

<!--
### Autoregressive Models
-->

### Mô hình Tự hồi quy

<!--
In order to achieve this, our trader could use a regressor such as the one we trained in :numref:`sec_linear_gluon`.
There is just a major problem: the number of inputs, $x_{t-1}, \ldots, x_1$ varies, depending on $t$.
That is, the number increases with the amount of data that we encounter, and we will need an approximation to make this computationally tractable.
Much of what follows in this chapter will revolve around how to estimate $p(x_t \mid x_{t-1}, \ldots, x_1)$ efficiently.
In a nutshell it boils down to two strategies:
-->

Để dự đoán giá cổ phiếu, các nhà giao dịch có thể sử dụng một mô hình hồi quy, chẳng hạn như mô hình mà ta đã huấn luyện trong :numref:`sec_linear_gluon`.
Chỉ có một vấn đề lớn ở đây, đó là số lượng đầu vào, $x_{t-1}, \ldots, x_1$ thay đổi tùy thuộc vào $t$.
Cụ thể, số lượng đầu vào sẽ tăng cùng với lượng dữ liệu thu được và ta sẽ cần một phép tính xấp xỉ để làm cho giải pháp này khả thi về mặt tính toán.
Phần lớn nội dung tiếp theo trong chương này sẽ xoay quanh việc làm thế nào để ước lượng $p(x_t \mid x_{t-1}, \ldots, x_1)$ một cách hiệu quả.
Nói ngắn gọn, ta có hai chiến lược:

<!--
1. Assume that the potentially rather long sequence $x_{t-1}, \ldots, x_1$ is not really necessary. 
In this case we might content ourselves with some timespan $\tau$ and only use $x_{t-1}, \ldots, x_{t-\tau}$ observations. 
The immediate benefit is that now the number of arguments is always the same, at least for $t > \tau$. 
This allows us to train a deep network as indicated above. 
Such models will be called *autoregressive* models, as they quite literally perform regression on themselves.
2. Another strategy, shown in :numref:`fig_sequence-model`, is to try and keep some summary $h_t$ of the past observations, at the same time update $h_t$ in addition to the prediction $\hat{x}_t$. 
This leads to models that estimate $x_t$ with $\hat{x}_t = p(x_t \mid x_{t-1}, h_{t})$ and moreover updates of the form  $h_t = g(h_{t-1}, x_{t-1})$. 
Since $h_t$ is never observed, these models are also called *latent autoregressive models*. 
LSTMs and GRUs are examples of this.
-->

1. Giả sử rằng việc sử dụng một chuỗi có thể rất dài $x_{t-1}, \ldots, x_1$ là không thực sự cần thiết.
Trong trường hợp này, ta có thể hài lòng với một khoảng thời gian $\tau$ và chỉ sử dụng các quan sát $x_{t-1}, \ldots, x_{t-\tau}$.
Lợi ích trước mắt là bây giờ số lượng đối số luôn bằng nhau, ít nhất là với $t > \tau$.
Điều này sẽ cho phép ta huấn luyện một mạng sâu như được đề cập ở bên trên.
Các mô hình như vậy được gọi là các mô hình *tự hồi quy* (_autoregressive_), vì chúng tự thực hiện hồi quy trên chính mình.
2. Một chiến lược khác, được minh họa trong :numref:`fig_sequence-model`, là giữ một giá trị $h_t$ để tóm tắt các quan sát trong quá khứ, đồng thời cập nhật $h_t$ bên cạnh việc dự đoán $\hat{x}_t$.
Kết quả là mô hình sẽ ước tính $x_t$ với $\hat{x}_t = p(x_t \mid x_{t-1}, h_{t})$ và cập nhật $h_t = g(h_{t-1}, x_{t-1})$.
Do $h_t$ không bao giờ được quan sát nên các mô hình này còn được gọi là các *mô hình tự hồi quy tiềm ẩn* (_latent autoregressive model_).
LSTM và GRU là hai ví dụ cho kiểu mô hình này.

<!--
![A latent autoregressive model. ](../img/sequence-model.svg)
-->

![Một mô hình tự hồi quy tiềm ẩn. ](../img/sequence-model.svg)
:label:`fig_sequence-model`

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
Both cases raise the obvious question of how to generate training data.
One typically uses historical observations to predict the next observation given the ones up to right now.
Obviously we do not expect time to stand still.
However, a common assumption is that while the specific values of $x_t$ might change, at least the dynamics of the time series itself will not.
This is reasonable, since novel dynamics are just that, novel and thus not predictable using data that we have so far.
Statisticians call dynamics that do not change *stationary*.
Regardless of what we do, we will thus get an estimate of the entire time series via
-->

Cả hai trường hợp đều đặt ra câu hỏi về cách tạo ra dữ liệu huấn luyện.
Người ta thường sử dụng các quan sát từ quá khứ cho đến hiện tại để dự đoán các quan sát xảy ra trong tương lai.
Rõ ràng chúng ta không thể trông đợi thời gian sẽ đứng yên.
Tuy nhiên, một giả định phổ biến là: tuy các giá trị cụ thể của $x_t$ có thể thay đổi, ít ra động lực của chuỗi thời gian sẽ không đổi.
Điều này khá hợp lý, vì nếu động lực thay đổi thì ta sẽ không thể dự đoán được nó bằng cách sử dụng dữ liệu mà ta đang có.
Các nhà thống kê gọi các động lực không thay đổi này là *cố định* (*stationary*).
Dù có làm gì đi chăng nữa, chúng ta vẫn sẽ tìm được ước lượng của toàn bộ chuỗi thời gian thông qua


$$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_{t-1}, \ldots, x_1).$$

<!--
Note that the above considerations still hold if we deal with discrete objects, such as words, rather than numbers.
The only difference is that in such a situation we need to use a classifier rather than a regressor to estimate $p(x_t \mid  x_{t-1}, \ldots, x_1)$.
-->

Lưu ý rằng các xem xét trên vẫn đúng trong trường hợp chúng ta làm việc với các đối tượng rời rạc, chẳng hạn như từ ngữ thay vì số.
Sự khác biệt duy nhất trong trường hợp này là chúng ta cần sử dụng một bộ phân loại thay vì một bộ hồi quy để ước lượng $p(x_t \mid  x_{t-1}, \ldots, x_1)$.

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Markov Model
-->

### Mô hình Markov

<!--
Recall the approximation that in an autoregressive model we use only $(x_{t-1}, \ldots, x_{t-\tau})$ instead of $(x_{t-1}, \ldots, x_1)$ to estimate $x_t$.
Whenever this approximation is accurate we say that the sequence satisfies a *Markov condition*.
In particular, if $\tau = 1$, we have a *first order* Markov model and $p(x)$ is given by
-->

Nhắc lại phép xấp xỉ trong một mô hình tự hồi quy, chúng ta chỉ sử dụng $(x_{t-1}, \ldots, x_{t-\tau})$ thay vì $(x_{t-1}, \ldots, x_1)$ để ước lượng $x_t$. 
Bất cứ khi nào phép xấp xỉ này là chính xác, chúng ta nói rằng chuỗi thỏa mãn *điều kiện Markov*. 
Cụ thể, nếu $\tau = 1$, chúng ta có mô hình Markov *bậc một* và $p(x)$ như sau 

$$p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_{t-1}).$$

<!--
Such models are particularly nice whenever $x_t$ assumes only a discrete value, since in this case dynamic programming can be used to compute values along the chain exactly.
For instance, we can compute $p(x_{t+1} \mid x_{t-1})$ efficiently using the fact that we only need to take into account a very short history of past observations:
-->

Các mô hình như trên rất hữu dụng bất cứ khi nào $x_t$ chỉ là các giá trị rời rạc, vì trong trường hợp này, quy hoạch động có thể được sử dụng để tính toán chính xác các giá trị theo chuỗi.
Ví dụ, chúng ta có thể tính toán $p(x_{t+1} \mid x_{t-1})$ một cách hiệu quả bằng cách chỉ sử dụng các quan sát trong một khoảng thời gian ngắn tại quá khứ:

$$p(x_{t+1} \mid x_{t-1}) = \sum_{x_t} p(x_{t+1} \mid x_t) p(x_t \mid x_{t-1}).$$


<!--
Going into details of dynamic programming is beyond the scope of this section, but we will introduce it in :numref:`sec_bi_rnn`.
Control and reinforcement learning algorithms use such tools extensively.
-->

Chi tiết về quy hoạch động nằm ngoài phạm vi của phần này, nhưng chúng tôi sẽ giới thiệu nó trong :numref:`sec_bi_rnn`. 
Các công cụ trên được sử dụng rất phổ biến trong các thuật toán điều khiển và học tăng cường. 

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### Causality
-->

### Quan hệ Nhân quả

<!--
In principle, there is nothing wrong with unfolding $p(x_1, \ldots, x_T)$ in reverse order.
After all, by conditioning we can always write it via
-->

Về nguyên tắc, không có gì sai khi trải (*unfolding*) $p(x_1, \ldots, x_T)$ theo thứ tự ngược lại. 
Bằng cách đặt điều kiện như vậy, chúng ta luôn có thể viết chúng như sau 

$$p(x_1, \ldots, x_T) = \prod_{t=T}^1 p(x_t \mid x_{t+1}, \ldots, x_T).$$

<!--
In fact, if we have a Markov model, we can obtain a reverse conditional probability distribution, too.
In many cases, however, there exists a natural direction for the data, namely going forward in time.
It is clear that future events cannot influence the past.
Hence, if we change $x_t$, we may be able to influence what happens for $x_{t+1}$ going forward but not the converse.
That is, if we change $x_t$, the distribution over past events will not change.
Consequently, it ought to be easier to explain $p(x_{t+1} \mid x_t)$ rather than $p(x_t \mid x_{t+1})$.
For instance, :cite:`Hoyer.Janzing.Mooij.ea.2009` show that in some cases we can find $x_{t+1} = f(x_t) + \epsilon$ for some additive noise, whereas the converse is not true.
This is great news, since it is typically the forward direction that we are interested in estimating.
For more on this topic see e.g., the book by :cite:`Peters.Janzing.Scholkopf.2017`.
We are barely scratching the surface of it.
-->

Trên thực tế, nếu có một mô hình Markov, chúng ta cũng có thể thu được một phân phối xác suất có điều kiện ngược. 
Tuy nhiên trong nhiều trường hợp vẫn tồn tại một trật tự tự nhiên cho dữ liệu, cụ thể đó là chiều thuận theo thời gian. 
Rõ ràng là các sự kiện trong tương lai không thể ảnh hưởng đến quá khứ. 
Do đó, nếu thay đổi $x_t$ thì ta có thể ảnh hưởng đến những gì xảy ra tại $x_{t+1}$ trong tương lai, nhưng lại không thể ảnh hưởng tới quá khứ theo chiều ngược lại. 
Nếu chúng ta thay đổi $x_t$, phân phối trên các sự kiện trong quá khứ sẽ không thay đổi. 
Do đó, việc giải thích $p(x_{t+1} \mid x_t)$ sẽ đơn giản hơn là $p(x_t \mid x_{t+1})$. 
Ví dụ: :cite:`Hoyer.Janzing.Mooij.ea.2009` chỉ ra rằng trong một số trường hợp chúng ta có thể tìm $x_{t+1} = f(x_t) + \epsilon$ khi có thêm nhiễu, trong khi điều ngược lại thì không đúng. 
Đây là một tin tuyệt vời vì chúng ta thường quan tâm tới việc ước lượng theo chiều thuận hơn. 
Để tìm hiểu thêm về chủ đề này, có thể tìm đọc cuốn sách :cite:`Peters.Janzing.Scholkopf.2017`.
Chúng ta sẽ chỉ tìm hiểu sơ qua trong phần này. 

<!--
## A Toy Example
-->

## Một ví dụ đơn giản

<!--
After so much theory, let us try this out in practice.
Let us begin by generating some data.
To keep things simple we generate our time series by using a sine function with some additive noise.
-->

Sau khi đề cập nhiều về lý thuyết, bây giờ chúng ta hãy thử lập trình minh họa. 
Đầu tiên, hãy khởi tạo một vài dữ liệu như sau. 
Để đơn giản, chúng ta tạo chuỗi thời gian bằng cách sử dụng hàm sin cộng thêm một chút nhiễu. 


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()

T = 1000  # Generate a total of 1000 points
time = np.arange(0, T)
x = np.sin(0.01 * time) + 0.2 * np.random.normal(size=T)
d2l.plot(time, [x])
```

<!--
Next we need to turn this time series into features and labels that the network can train on.
Based on the embedding dimension $\tau$ we map the data into pairs $y_t = x_t$ and $\mathbf{z}_t = (x_{t-1}, \ldots, x_{t-\tau})$.
The astute reader might have noticed that this gives us $\tau$ fewer data points, since we do not have sufficient history for the first $\tau$ of them.
A simple fix, in particular if the time series is long is to discard those few terms.
Alternatively we could pad the time series with zeros.
The code below is essentially identical to the training code in previous sections.
We kept the architecture fairly simple.
A few layers of a fully connected network, ReLU activation and $\ell_2$ loss.
Since much of the modeling is identical to the previous sections when we built regression estimators in Gluon, we will not delve into much detail.
-->

Tiếp theo, chúng ta cần biến chuỗi thời gian này thành các đặc trưng và nhãn có thể được sử dụng để huấn luyện mạng. 
Dựa trên kích thước embedding $\tau$, chúng ta ánh xạ dữ liệu thành các cặp $y_t = x_t$ và $\mathbf{z}_t = (x_{t-1}, \ldots, x_{t-\tau})$. 
Để ý kĩ, có thể thấy rằng ta sẽ mất $\tau$ điểm dữ liệu đầu tiên, vì chúng ta không có đủ $\tau$ điểm dữ liệu trong quá khứ để làm đặc trưng cho chúng.
Một cách đơn giản để khắc phục điều này, đặc biệt là khi chuỗi thời gian rất dài, là loại bỏ đi số ít các phần tử đó. 
Một cách khác là đệm giá trị 0 vào chuỗi thời gian. 
Mã nguồn dưới đây về cơ bản là giống hệt với mã nguồn huấn luyện trong các phần trước. 
Chúng tôi cố gắng giữ cho kiến trúc đơn giản với vài tầng kết nối đầy đủ, hàm kích hoạt ReLU và hàm mất mát $\ell_2$. 
Do việc mô hình hóa phần lớn là giống với khi ta xây dựng các bộ ước lượng hồi quy viết bằng Gluon trong các phần trước, nên chúng ta sẽ không đi sâu vào chi tiết trong phần này. 


```{.python .input}
tau = 4
features = np.zeros((T-tau, tau))
for i in range(tau):
    features[:, i] = x[i: T-tau+i]
labels = x[tau:]

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)
test_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                           batch_size, is_train=False)

# Vanilla MLP architecture
def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(10, activation='relu'),
            nn.Dense(1))
    net.initialize(init.Xavier())
    return net

# Least mean squares loss
loss = gluon.loss.L2Loss()
```

<!--
Now we are ready to train.
-->

Bây giờ chúng ta đã sẵn sàng để huấn luyện. 

```{.python .input}
def train_net(net, train_iter, loss, epochs, lr):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(1, epochs + 1):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        print('epoch %d, loss: %f' % (
            epoch, d2l.evaluate_loss(net, train_iter, loss)))

net = get_net()
train_net(net, train_iter, loss, 10, 0.01)
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Predictions
-->

## Dự đoán của Mô hình

<!--
Since both training and test loss are small, we would expect our model to work well.
Let us see what this means in practice.
The first thing to check is how well the model is able to predict what happens in the next timestep.
-->

Vì cả hai giá trị mất mát trên tập huấn luyện và kiểm tra đều nhỏ, chúng ta kỳ vọng mô hình trên sẽ hoạt động tốt. 
Hãy cùng xác nhận điều này trên thực tế. 
Điều đầu tiên cần kiểm tra là mô hình có thể dự đoán những gì sẽ xảy ra trong bước thời gian kế tiếp tốt như thế nào. 

```{.python .input}
estimates = net(features)
d2l.plot([time, time[tau:]], [x, estimates],
         legend=['data', 'estimate'])
```

<!--
This looks nice, just as we expected it.
Even beyond 600 observations the estimates still look rather trustworthy.
There is just one little problem to this: if we observe data only until timestep 600, we cannot hope to receive the ground truth for all future predictions.
Instead, we need to work our way forward one step at a time:
-->

Kết quả khá tốt, đúng như những gì chúng ta mong đợi. 
Thậm chí sau hơn 600 mẫu quan sát, phép ước lượng vẫn trông khá tin cậy. 
Chỉ có một chút vấn đề: nếu chúng ta quan sát dữ liệu tới bước thời gian thứ 600, chúng ta không thể hy vọng sẽ nhận được nhãn gốc cho tất cả các dự đoán tương lai. 
Thay vào đó, chúng ta cần tiến lên từng bước một: 


$$\begin{aligned}
x_{601} & = f(x_{600}, \ldots, x_{597}), \\
x_{602} & = f(x_{601}, \ldots, x_{598}), \\
x_{603} & = f(x_{602}, \ldots, x_{599}).
\end{aligned}$$


<!--
In other words, we will have to use our own predictions to make future predictions.
Let us see how well this goes.
-->

Nói cách khác, chúng ta sẽ phải sử dụng những dự đoán của mình để đưa ra dự đoán trong tương lai. 
Hãy cùng xem cách này có ổn không.


```{.python .input}
predictions = np.zeros(T)
predictions[:n_train] = x[:n_train]
for i in range(n_train, T):
    predictions[i] = net(
        predictions[(i-tau):i].reshape(1, -1)).reshape(1)
d2l.plot([time, time[tau:], time[n_train:]],
         [x, estimates, predictions[n_train:]],
         legend=['data', 'estimate', 'multistep'], figsize=(4.5, 2.5))
```

<!--
As the above example shows, this is a spectacular failure.
The estimates decay to a constant pretty quickly after a few prediction steps.
Why did the algorithm work so poorly?
This is ultimately due to the fact that the errors build up.
Let us say that after step 1 we have some error $\epsilon_1 = \bar\epsilon$.
Now the *input* for step 2 is perturbed by $\epsilon_1$, hence we suffer some error in the order of $\epsilon_2 = \bar\epsilon + L \epsilon_1$, and so on.
The error can diverge rather rapidly from the true observations.
This is a common phenomenon.
For instance, weather forecasts for the next 24 hours tend to be pretty accurate but beyond that the accuracy declines rapidly.
We will discuss methods for improving this throughout this chapter and beyond.
-->

Ví dụ trên cho thấy, cách này đã thất bại thảm hại. 
Các giá trị ước lượng rất nhanh chóng suy giảm thành một hằng số chỉ sau một vài bước. 
Tại sao thuật toán trên hoạt động tệ đến thế? 
Suy cho cùng, lý do là trên thực tế các sai số dự đoán bị chồng chất qua các bước thời gian.
Cụ thể, sau bước thời gian 1 chúng ta có nhận được sai số $\epsilon_1 = \bar\epsilon$.
Tiếp theo, *đầu vào* cho bước thời gian 2 bị nhiễu loạn bởi $\epsilon_1$, do đó chúng ta nhận được sai số dự đoán $\epsilon_2 = \bar\epsilon + L \epsilon_1$. Tương tự như thế cho các bước thời gian tiếp theo. 
Sai số có thể phân kỳ khá nhanh khỏi các quan sát đúng. 
Đây là một hiện tượng phổ biến. 
Ví dụ, dự báo thời tiết trong 24 giờ tới có độ chính xác khá cao nhưng nó giảm đi nhanh chóng với những dự báo xa hơn quãng thời gian đó. 
Chúng ta sẽ thảo luận về các phương pháp để cải thiện vấn đề trên trong chương này và những chương tiếp theo. 

<!--
Let us verify this observation by computing the $k$-step predictions on the entire sequence.
-->

Chúng ta hãy kiểm chứng quan sát trên bằng cách tính toán dự đoán $k$ bước thời gian trên toàn bộ chuỗi.

```{.python .input}
k = 33  # Look up to k - tau steps ahead

features = np.zeros((k, T-k))
for i in range(tau):  # Copy the first tau features from x
    features[i] = x[i:T-k+i]

for i in range(tau, k):  # Predict the (i-tau)-th step
    features[i] = net(features[(i-tau):i].T).T

steps = (4, 8, 16, 32)
d2l.plot([time[i:T-k+i] for i in steps], [features[i] for i in steps],
         legend=['step %d' % i for i in steps], figsize=(4.5, 2.5))
```

<!--
This clearly illustrates how the quality of the estimates changes as we try to predict further into the future.
While the 8-step predictions are still pretty good, anything beyond that is pretty useless.
-->

Điều này minh họa rõ ràng chất lượng của các ước lượng thay đổi như thế nào khi chúng ta cố gắng dự đoán xa hơn trong tương lai. 
Mặc dù những dự đoán có độ dài là 8 bước vẫn còn khá tốt, bất cứ kết quả dự đoán nào vượt ra ngoài khoảng đó thì khá là vô dụng.

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
## Summary
-->

## Tóm tắt

<!--
* Sequence models require specialized statistical tools for estimation.
Two popular choices are autoregressive models and latent-variable autoregressive models.
* As we predict further in time, the errors accumulate and the quality of the estimates degrades, often dramatically.
* There is quite a difference in difficulty between interpolation and extrapolation. 
Consequently, if you have a time series, always respect the temporal order of the data when training, i.e., never train on future data.
* For causal models (e.g., time going forward), estimating the forward direction is typically a lot easier than the reverse direction.
-->

* Các mô hình chuỗi thường yêu cầu các công cụ thống kê chuyên biệt để ước lượng.
Hai lựa chọn phổ biến đó là các mô hình tự hồi quy và mô hình tự hồi quy biến tiềm ẩn. 
* Sai số bị tích lũy và chất lượng của phép ước lượng suy giảm đáng kể khi mô hình dự đoán các bước thời gian xa hơn. 
* Khó khăn trong phép nội suy và ngoại suy khá khác biệt.
Do đó, nếu bạn có một kiểu dữ liệu chuỗi thời gian, hãy luôn để ý trình tự thời gian của dữ liệu khi huấn luyện, hay nói cách khác, không bao giờ huấn luyện trên dữ liệu thuộc về bước thời gian trong tương lai. 
* Đối với các mô hình nhân quả (ví dụ, ở đó thời gian đi về phía trước), ước lượng theo chiều xuôi thường dễ dàng hơn rất nhiều so với chiều ngược lại. 



<!--
## Exercises
-->

## Bài tập

<!--
1. Improve the above model.
    * Incorporate more than the past 4 observations? How many do you really need?
    * How many would you need if there was no noise? Hint: you can write $\sin$ and $\cos$ as a differential equation.
    * Can you incorporate older features while keeping the total number of features constant? Does this improve accuracy? Why?
    * Change the neural network architecture and see what happens.
2. An investor wants to find a good security to buy. She looks at past returns to decide which one is likely to do well. What could possibly go wrong with this strategy?
3. Does causality also apply to text? To which extent?
4. Give an example for when a latent autoregressive model might be needed to capture the dynamic of the data.
-->

1. Hãy cải thiện mô hình nói trên bằng cách
    * Kết hợp nhiều hơn 4 mẫu quan sát trong quá khứ? Bao nhiêu mẫu quan sát là thực sự cần thiết? 
    * Bạn sẽ cần bao nhiêu mẫu nếu dữ liệu không có nhiễu? Gợi ý: bạn có thể viết $\sin$ và $\cos$ dưới dạng phương trình vi phân. 
    * Có thể kết hợp các đặc trưng cũ hơn trong khi đảm bảo tổng số đặc trưng là không đổi không? Điều này có cải thiện độ chính xác không? Tại sao? 
    * Thay đổi cấu trúc mạng nơ-ron và quan sát tác động của nó. 
2. Nếu một nhà đầu tư muốn tìm một mã chứng khoán tốt để mua. Cô ta sẽ nhìn vào lợi nhuận trong quá khứ để quyết định mã nào có khả năng sinh lời. Điều gì có thể khiến chiến lược này trở thành sai lầm? 
3. Liệu có thể áp dụng quan hệ nhân quả cho dữ liệu văn bản được không? Nếu có thì ở mức độ nào? 
4. Hãy cho một ví dụ khi mô hình tự hồi quy tiềm ẩn có thể cần được dùng để nắm bắt động lực của dữ liệu.


<!-- ===================== Kết thúc dịch Phần 6 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2860)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Duy Du
* Nguyễn Cảnh Thướng
* Phạm Minh Đức
* Nguyễn Lê Quang Nhật
* Nguyễn Văn Quang
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh

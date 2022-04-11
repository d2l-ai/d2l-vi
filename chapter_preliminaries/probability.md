# Xác suất
:label:`sec_prob`

Trong một số hình thức này hay hình thức khác, machine learning là tất cả về việc đưa ra dự đoán. Chúng tôi có thể muốn dự đoán * xác suất* của một bệnh nhân bị đau tim trong năm tới, với tiền sử lâm sàng của họ. Trong phát hiện bất thường, chúng ta có thể muốn đánh giá mức độ* tương tự* một tập hợp các bài đọc từ động cơ phản lực của máy bay phản lực sẽ như thế nào, nó hoạt động bình thường. Trong học tập củng cố, chúng tôi muốn một đại lý hành động thông minh trong một môi trường. Điều này có nghĩa là chúng ta cần suy nghĩ về xác suất nhận được phần thưởng cao theo từng hành động có sẵn. Và khi chúng ta xây dựng hệ thống giới thiệu, chúng ta cũng cần phải suy nghĩ về xác suất. Ví dụ: giả thuyết *rằng chúng tôi đã làm việc cho một người bán sách trực tuyến lớn. Chúng tôi có thể muốn ước tính xác suất mà một người dùng cụ thể sẽ mua một cuốn sách cụ thể. Đối với điều này, chúng ta cần sử dụng ngôn ngữ xác suất. Toàn bộ các khóa học, chuyên ngành, luận án, nghề nghiệp và thậm chí cả các bộ phận, được dành cho xác suất. Vì vậy, một cách tự nhiên, mục tiêu của chúng tôi trong phần này là không dạy toàn bộ chủ đề. Thay vào đó, chúng tôi hy vọng sẽ đưa bạn ra khỏi mặt đất, để dạy bạn vừa đủ để bạn có thể bắt đầu xây dựng các mô hình học sâu đầu tiên của mình và cung cấp cho bạn đủ hương vị cho chủ đề mà bạn có thể bắt đầu tự khám phá nó nếu bạn muốn. 

Chúng tôi đã gọi xác suất trong các phần trước mà không cần khớp nối chính xác chúng là gì hoặc đưa ra một ví dụ cụ thể. Chúng ta hãy nghiêm túc hơn bây giờ bằng cách xem xét trường hợp đầu tiên: phân biệt mèo và chó dựa trên các bức ảnh. Điều này nghe có vẻ đơn giản nhưng nó thực sự là một thách thức ghê gớm. Để bắt đầu, khó khăn của vấn đề có thể phụ thuộc vào độ phân giải của hình ảnh. 

![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

Như thể hiện trong :numref:`fig_cat_dog`, trong khi con người dễ dàng nhận ra mèo và chó ở độ phân giải $160 \times 160$ pixel, nó trở nên khó khăn ở $40 \times 40$ pixel và bên cạnh không thể ở $10 \times 10$ pixel. Nói cách khác, khả năng của chúng ta để nói với mèo và chó xa nhau ở một khoảng cách lớn (và do đó độ phân giải thấp) có thể tiếp cận đoán không hiểu biết. Xác suất cho chúng ta một cách lý luận chính thức về mức độ chắc chắn của chúng ta. Nếu chúng ta hoàn toàn chắc chắn rằng hình ảnh mô tả một con mèo, chúng ta nói rằng * xác suất* rằng nhãn tương ứng $y$ là “mèo”, ký hiệu $P(y=$ “mèo"$)$ bằng $1$. Nếu chúng tôi không có bằng chứng để gợi ý rằng $y =$ “mèo” hoặc $y =$ “chó” đó, thì chúng ta có thể nói rằng hai khả năng là như nhau
*likely* thể hiện điều này như $P(y=$ “mèo"$) = P(y=$ “chó"$) = 0.5$. Nếu chúng ta hợp lý
tự tin, nhưng không chắc chắn rằng hình ảnh mô tả một con mèo, chúng ta có thể gán một xác suất $0.5  < P(y=$ “mèo"$) < 1$. 

Bây giờ hãy xem xét trường hợp thứ hai: đưa ra một số dữ liệu theo dõi thời tiết, chúng tôi muốn dự đoán xác suất nó sẽ mưa ở Đài Bắc vào ngày mai. Nếu đó là mùa hè, mưa có thể đi kèm với xác suất 0,5. 

Trong cả hai trường hợp, chúng tôi có một số giá trị quan tâm. Và trong cả hai trường hợp, chúng tôi không chắc chắn về kết quả. Nhưng có một sự khác biệt chính giữa hai trường hợp. Trong trường hợp đầu tiên này, hình ảnh trên thực tế là chó hoặc một con mèo, và chúng tôi chỉ không biết cái nào. Trong trường hợp thứ hai, kết quả thực sự có thể là một sự kiện ngẫu nhiên, nếu bạn tin vào những điều như vậy (và hầu hết các nhà vật lý làm). Vì vậy, xác suất là một ngôn ngữ linh hoạt để lý luận về mức độ chắc chắn của chúng ta, và nó có thể được áp dụng hiệu quả trong một tập hợp rộng lớn các ngữ cảnh. 

## Lý thuyết xác suất cơ bản

Nói rằng chúng tôi đã chết và muốn biết cơ hội nhìn thấy 1 chứ không phải là một chữ số khác. Nếu chết là công bằng, tất cả sáu kết quả $\{1, \ldots, 6\}$ đều có khả năng xảy ra như nhau, và do đó chúng ta sẽ thấy một $1$ trong một trong sáu trường hợp. Chính thức chúng tôi tuyên bố rằng $1$ xảy ra với xác suất $\frac{1}{6}$. 

Đối với một cái chết thực sự mà chúng tôi nhận được từ một nhà máy, chúng ta có thể không biết những tỷ lệ đó và chúng tôi sẽ cần phải kiểm tra xem nó có bị nhiễm bẩn hay không. Cách duy nhất để điều tra khuôn là đúc nó nhiều lần và ghi lại kết quả. Đối với mỗi diễn viên của khuôn, chúng ta sẽ quan sát một giá trị trong $\{1, \ldots, 6\}$. Với những kết quả này, chúng tôi muốn điều tra xác suất quan sát từng kết quả. 

Một cách tiếp cận tự nhiên cho mỗi giá trị là lấy số lượng cá nhân cho giá trị đó và chia nó cho tổng số lần tung. Điều này cho chúng ta một * ước tính* về xác suất của một *sự kiện* nhất định. Quy luật * số lượng lớn* cho chúng ta biết rằng khi số lượng ném tăng lên ước tính này sẽ tiến gần hơn và gần hơn với xác suất cơ bản thực sự. Trước khi đi sâu vào chi tiết về những gì đang xảy ra ở đây, chúng ta hãy thử nó. 

Để bắt đầu, chúng ta hãy nhập các gói cần thiết.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions import multinomial
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
```

Tiếp theo, chúng ta sẽ muốn có thể đúc chết. Trong thống kê, chúng tôi gọi quá trình vẽ ví dụ này từ phân phối xác suất * mẫu*. Phân phối gán xác suất cho một số lựa chọn rời rạc được gọi là
*phân phối đa ngôn ngữ*. Chúng tôi sẽ đưa ra một định nghĩa chính thức hơn về
*phân phối* sau đó, nhưng ở mức cao, hãy nghĩ về nó như chỉ là một nhiệm vụ
probabilities xác suất to events các sự kiện. 

Để vẽ một mẫu duy nhất, chúng tôi chỉ cần vượt qua một vector xác suất. Đầu ra là một vectơ khác có cùng độ dài: giá trị của nó tại chỉ số $i$ là số lần kết quả lấy mẫu tương ứng với $i$.

```{.python .input}
fair_probs = [1.0 / 6] * 6
np.random.multinomial(1, fair_probs)
```

```{.python .input}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
multinomial.Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfp.distributions.Multinomial(1, fair_probs).sample()
```

Nếu bạn chạy sampler một loạt các lần, bạn sẽ thấy rằng bạn nhận ra các giá trị ngẫu nhiên mỗi lần. Như với việc ước tính tính công bằng của khuôn, chúng ta thường muốn tạo ra nhiều mẫu từ cùng một phân phối. Nó sẽ là không thể chịu nổi chậm để làm điều này với một vòng lặp Python `for`, vì vậy chức năng chúng tôi đang sử dụng hỗ trợ vẽ nhiều mẫu cùng một lúc, trả về một loạt các mẫu độc lập trong bất kỳ hình dạng nào chúng ta có thể mong muốn.

```{.python .input}
np.random.multinomial(10, fair_probs)
```

```{.python .input}
#@tab pytorch
multinomial.Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfp.distributions.Multinomial(10, fair_probs).sample()
```

Bây giờ chúng ta đã biết làm thế nào để lấy mẫu cuộn của một khuôn, chúng ta có thể mô phỏng 1000 cuộn. Sau đó chúng ta có thể đi qua và đếm, sau mỗi 1000 cuộn, bao nhiêu lần mỗi số được cuộn. Cụ thể, chúng tôi tính toán tần số tương đối như ước tính xác suất thực sự.

```{.python .input}
counts = np.random.multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input}
#@tab pytorch
# Store the results as 32-bit floats for division
counts = multinomial.Multinomial(1000, fair_probs).sample()
counts / 1000  # Relative frequency as the estimate
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(1000, fair_probs).sample()
counts / 1000
```

Bởi vì chúng tôi tạo ra dữ liệu từ một chết công bằng, chúng tôi biết rằng mỗi kết quả có xác suất thực sự $\frac{1}{6}$, khoảng $0.167$, vì vậy các ước tính đầu ra trên trông tốt. 

Chúng ta cũng có thể hình dung cách các xác suất này hội tụ theo thời gian hướng tới xác suất thực sự. Hãy để chúng tôi tiến hành 500 nhóm thí nghiệm trong đó mỗi nhóm vẽ 10 mẫu.

```{.python .input}
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

```{.python .input}
#@tab pytorch
counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
#@tab tensorflow
counts = tfp.distributions.Multinomial(10, fair_probs).sample(500)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Mỗi đường cong rắn tương ứng với một trong sáu giá trị của khuôn và cho xác suất ước tính của chúng ta rằng khuôn biến giá trị đó như được đánh giá sau mỗi nhóm thí nghiệm. Đường đen đứt nét cho xác suất cơ bản thực sự. Khi chúng ta nhận được nhiều dữ liệu hơn bằng cách tiến hành nhiều thí nghiệm hơn, các đường cong rắn $6$ hội tụ về xác suất thực sự. 

### Tiên đề của lý thuyết xác suất

Khi xử lý các cuộn của một chết, chúng tôi gọi bộ $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ là * không gian mẫu* hoặc *không gian kết thúc*, trong đó mỗi phần tử là một *kết thúc*. Một *event* là một tập hợp các kết quả từ một không gian mẫu nhất định. Ví dụ: “nhìn thấy một $5$" ($\{5\}$) và “nhìn thấy một số lẻ” ($\{1, 3, 5\}$) đều là những sự kiện hợp lệ của việc lăn chết. Lưu ý rằng nếu kết quả của một thí nghiệm ngẫu nhiên là trong sự kiện $\mathcal{A}$, thì sự kiện $\mathcal{A}$ đã xảy ra. Điều đó có nghĩa là, nếu $3$ chấm đối mặt sau khi lăn chết, kể từ $3 \in \{1, 3, 5\}$, chúng ta có thể nói rằng sự kiện “nhìn thấy một số lẻ” đã xảy ra. 

Về mặt chính thức, *xác suất* có thể được coi là một hàm ánh xạ một tập hợp thành giá trị thực. Xác suất của một sự kiện $\mathcal{A}$ trong không gian mẫu đã cho $\mathcal{S}$, ký hiệu là $P(\mathcal{A})$, thỏa mãn các thuộc tính sau: 

* Đối với bất kỳ sự kiện nào $\mathcal{A}$, xác suất của nó không bao giờ tiêu cực, tức là, $P(\mathcal{A}) \geq 0$;
* Xác suất của toàn bộ không gian mẫu là $1$, tức là, $P(\mathcal{S}) = 1$;
* Đối với bất kỳ chuỗi các sự kiện có thể đếm được $\mathcal{A}_1, \mathcal{A}_2, \ldots$ mà là * loại trừ lẫn nhau* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ cho tất cả $i \neq j$), xác suất rằng bất kỳ xảy ra là bằng tổng xác suất cá nhân của họ, tức là, $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Đây cũng là các tiên đề của lý thuyết xác suất, được đề xuất bởi Kolmogorov năm 1933. Nhờ hệ thống tiên đề này, chúng ta có thể tránh được bất kỳ tranh chấp triết học nào về tính ngẫu nhiên; thay vào đó, chúng ta có thể lý luận nghiêm ngặt với một ngôn ngữ toán học. Ví dụ, bằng cách cho phép sự kiện $\mathcal{A}_1$ là toàn bộ không gian mẫu và $\mathcal{A}_i = \emptyset$ cho tất cả $i > 1$, chúng ta có thể chứng minh rằng $P(\emptyset) = 0$, tức là xác suất của một sự kiện bất khả thi là $0$. 

### Biến ngẫu nhiên

Trong thí nghiệm ngẫu nhiên của chúng tôi về đúc chết, chúng tôi đã giới thiệu khái niệm về một biến thể ngẫu nhiên*. Một biến ngẫu nhiên có thể khá nhiều bất kỳ số lượng và không phải là xác định. Nó có thể mất một giá trị trong một tập hợp các khả năng trong một thí nghiệm ngẫu nhiên. Hãy xem xét một biến ngẫu nhiên $X$ có giá trị nằm trong không gian mẫu $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ lăn một khuôn. Chúng ta có thể biểu thị sự kiện “nhìn thấy một $5$" là $\{X = 5\}$ hoặc $X = 5$, và xác suất của nó là $P(\{X = 5\})$ hoặc $P(X = 5)$. Đến $P(X = a)$, chúng ta phân biệt giữa biến ngẫu nhiên $X$ và các giá trị (ví dụ, $a$) mà $X$ có thể lấy. Tuy nhiên, pedantry như vậy dẫn đến một ký hiệu rườm rà. Đối với một ký hiệu nhỏ gọn, một mặt, chúng ta chỉ có thể biểu thị $P(X)$ là *distribution* so với biến ngẫu nhiên $X$: phân phối cho chúng ta biết xác suất $X$ có bất kỳ giá trị nào. Mặt khác, chúng ta chỉ cần viết $P(a)$ để biểu thị xác suất một biến ngẫu nhiên lấy giá trị $a$. Vì một sự kiện trong lý thuyết xác suất là một tập hợp các kết quả từ không gian mẫu, chúng ta có thể chỉ định một phạm vi các giá trị cho một biến ngẫu nhiên cần lấy. Ví dụ, $P(1 \leq X \leq 3)$ biểu thị xác suất của sự kiện $\{1 \leq X \leq 3\}$, có nghĩa là $\{X = 1, 2, \text{or}, 3\}$. Tương đương, $P(1 \leq X \leq 3)$ thể hiện xác suất biến ngẫu nhiên $X$ có thể lấy một giá trị từ $\{1, 2, 3\}$. 

Lưu ý rằng có một sự khác biệt tinh tế giữa các biến ngẫu nhiên *discrete*, như các cạnh của một cái chết, và *liên tục* các biến, như trọng lượng và chiều cao của một người. Có rất ít điểm trong việc hỏi liệu hai người có chính xác cùng chiều cao hay không. Nếu chúng ta thực hiện các phép đo đủ chính xác, bạn sẽ thấy rằng không có hai người nào trên hành tinh có cùng chiều cao chính xác. Trên thực tế, nếu chúng ta thực hiện một phép đo đủ tốt, bạn sẽ không có cùng chiều cao khi bạn thức dậy và khi bạn đi ngủ. Vì vậy, không có mục đích nào trong việc hỏi về xác suất mà ai đó là 1.80139278291028719210196740527486202 mét cao. Với dân số thế giới của con người xác suất hầu như là 0. Nó có ý nghĩa hơn trong trường hợp này để hỏi xem chiều cao của ai đó có rơi vào một khoảng thời gian nhất định hay không, nói từ 1,79 đến 1,81 mét. Trong những trường hợp này, chúng tôi định lượng khả năng chúng tôi thấy giá trị là * mật độ*. Chiều cao chính xác 1,80 mét không có xác suất, nhưng mật độ nonzero. Trong khoảng thời gian giữa bất kỳ hai độ cao khác nhau chúng ta có xác suất nonzero. Trong phần còn lại của phần này, chúng tôi xem xét xác suất trong không gian rời rạc. Để xác suất qua các biến ngẫu nhiên liên tục, bạn có thể tham khảo :numref:`sec_random_variables`. 

## Xử lý nhiều biến ngẫu nhiên

Rất thường xuyên, chúng ta sẽ muốn xem xét nhiều hơn một biến ngẫu nhiên tại một thời điểm. Ví dụ, chúng tôi có thể muốn mô hình hóa mối quan hệ giữa các bệnh và triệu chứng. Với một bệnh và một triệu chứng, nói “cúm” và “ho”, hoặc có thể xảy ra hoặc không thể xảy ra ở một bệnh nhân có một số xác suất. Mặc dù chúng tôi hy vọng rằng xác suất của cả hai sẽ gần bằng 0, chúng tôi có thể muốn ước tính những xác suất này và mối quan hệ của chúng với nhau để chúng tôi có thể áp dụng các suy luận của mình để có tác dụng chăm sóc y tế tốt hơn. 

Như một ví dụ phức tạp hơn, hình ảnh chứa hàng triệu pixel, do đó hàng triệu biến ngẫu nhiên. Và trong nhiều trường hợp, hình ảnh sẽ đi kèm với một nhãn, xác định các đối tượng trong hình ảnh. Chúng ta cũng có thể nghĩ nhãn như một biến ngẫu nhiên. Chúng ta thậm chí có thể nghĩ về tất cả các siêu dữ liệu như các biến ngẫu nhiên như vị trí, thời gian, khẩu độ, độ dài tiêu cự, ISO, khoảng cách lấy nét và loại máy ảnh. Tất cả những điều này là các biến ngẫu nhiên xảy ra cùng nhau. Khi chúng ta đối phó với nhiều biến ngẫu nhiên, có một số lượng quan tâm. 

### Xác suất chung

Đầu tiên được gọi là xác suất khớp *$P(A = a, B=b)$. Với bất kỳ giá trị $a$ và $b$, xác suất chung cho phép chúng ta trả lời, xác suất $A=a$ và $B=b$ đồng thời là bao nhiêu? Lưu ý rằng đối với bất kỳ giá trị $a$ và $b$, $P(A=a, B=b) \leq P(A=a)$. Điều này phải xảy ra, vì đối với $A=a$ và $B=b$ xảy ra, $A=a$ phải xảy ra*và* $B=b$ cũng phải xảy ra (và ngược lại). Do đó, $A=a$ và $B=b$ không thể có nhiều khả năng hơn $A=a$ hoặc $B=b$ riêng lẻ. 

### Xác suất có điều kiện

Điều này đưa chúng ta đến một tỷ lệ thú vị: $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$. Chúng tôi gọi tỷ lệ này là xác suất có điều kiện* và biểu thị nó bằng $P(B=b \mid A=a)$: đó là xác suất $B=b$, với điều kiện là $A=a$ đã xảy ra. 

### Định lý Bayes'

Sử dụng định nghĩa xác suất có điều kiện, chúng ta có thể lấy được một trong những phương trình hữu ích và nổi tiếng nhất trong thống kê: Định lý *Bayes'. Nó đi như sau. Bằng cách xây dựng, chúng tôi có quy tắc nhân ** rằng $P(A, B) = P(B \mid A) P(A)$. Theo đối xứng, điều này cũng giữ cho $P(A, B) = P(A \mid B) P(B)$. Giả sử rằng $P(B) > 0$. Giải quyết cho một trong các biến có điều kiện mà chúng ta nhận được 

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}.$$

Lưu ý rằng ở đây chúng tôi sử dụng ký hiệu nhỏ gọn hơn trong đó $P(A, B)$ là bản phân phối * chung * và $P(A \mid B)$ là bản phân phối có điều kiện*. Các phân phối như vậy có thể được đánh giá cho các giá trị cụ thể $A = a, B=b$. 

### Marginalization

Định lý Bayes rất hữu ích nếu chúng ta muốn suy ra một điều từ cái kia, nói nguyên nhân và hiệu quả, nhưng chúng ta chỉ biết các thuộc tính theo hướng ngược lại, như chúng ta sẽ thấy sau trong phần này. Một hoạt động quan trọng mà chúng ta cần, để thực hiện công việc này, là *marginalization*. Đó là hoạt động xác định $P(B)$ từ $P(A, B)$. Chúng ta có thể thấy rằng xác suất $B$ chiếm tất cả các lựa chọn có thể là $A$ và tổng hợp xác suất chung trên tất cả chúng: 

$$P(B) = \sum_{A} P(A, B),$$

còn được gọi là quy tắc tổng *.*. Xác suất hoặc phân phối là kết quả của việc biên giới được gọi là xác suất * biên độ* hoặc phân phối biên *. 

### Độc lập

Một thuộc tính hữu ích khác để kiểm tra là *dependence* so với *independence*. Hai biến ngẫu nhiên $A$ và $B$ độc lập có nghĩa là sự xuất hiện của một sự kiện $A$ không tiết lộ bất kỳ thông tin nào về sự xuất hiện của một sự kiện $B$. Trong trường hợp này $P(B \mid A) = P(B)$. Các nhà thống kê thường thể hiện điều này là $A \perp  B$. Từ định lý Bayes', nó đi theo ngay lập tức đó cũng là $P(A \mid B) = P(A)$. Trong tất cả các trường hợp khác, chúng tôi gọi $A$ và $B$ phụ thuộc. Ví dụ, hai cuộn liên tiếp của một khuôn là độc lập. Ngược lại, vị trí của công tắc ánh sáng và độ sáng trong phòng không phải là (mặc dù chúng không hoàn toàn xác định, vì chúng ta luôn có thể có bóng đèn bị hỏng, mất điện hoặc công tắc bị hỏng). 

Vì $P(A \mid B) = \frac{P(A, B)}{P(B)} = P(A)$ tương đương với $P(A, B) = P(A)P(B)$, hai biến ngẫu nhiên độc lập nếu và chỉ khi phân phối chung của chúng là tích của các phân phối riêng lẻ của chúng. Tương tự như vậy, hai biến ngẫu nhiên $A$ và $B$ là * độc lập có điều kiện* cho một biến ngẫu nhiên khác $C$ nếu và chỉ khi $P(A, B \mid C) = P(A \mid C)P(B \mid C)$. Điều này được thể hiện là $A \perp B \mid C$. 

### Ứng dụng
:label:`subsec_probability_hiv_app`

Hãy để chúng tôi đưa các kỹ năng của chúng tôi để kiểm tra. Giả sử rằng bác sĩ quản lý xét nghiệm HIV cho bệnh nhân. Xét nghiệm này khá chính xác và nó chỉ thất bại với xác suất 1% nếu bệnh nhân khỏe mạnh nhưng báo cáo anh ta là bệnh. Hơn nữa, nó không bao giờ không phát hiện ra HIV nếu bệnh nhân thực sự có nó. Chúng tôi sử dụng $D_1$ để chỉ ra chẩn đoán ($1$ nếu dương tính và $0$ nếu âm tính) và $H$ để biểu thị tình trạng HIV ($1$ nếu dương tính và $0$ nếu âm tính). :numref:`conditional_prob_D1` liệt kê các xác suất có điều kiện như vậy. 

: Xác suất điều kiện của $P(D_1 \mid H)$. 

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_1 = 1 \mid H)$|            1 |         0.01 |
|$P(D_1 = 0 \mid H)$|            0 |         0.99 |
:label:`conditional_prob_D1`

Lưu ý rằng các tổng cột là tất cả 1 (nhưng tổng hàng không), vì xác suất có điều kiện cần phải tổng hợp lên đến 1, giống như xác suất. Chúng ta hãy tìm ra xác suất bệnh nhân nhiễm HIV nếu xét nghiệm trở lại dương tính, tức là $P(H = 1 \mid D_1 = 1)$. Rõ ràng điều này sẽ phụ thuộc vào mức độ phổ biến của bệnh, vì nó ảnh hưởng đến số lượng báo động sai. Giả sử rằng dân số khá khỏe mạnh, ví dụ, $P(H=1) = 0.0015$. Để áp dụng định lý Bayes', chúng ta cần áp dụng marginalization và quy tắc nhân để xác định 

$$\begin{aligned}
&P(D_1 = 1) \\
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Vì vậy, chúng tôi nhận được 

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1)\\ =& \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} \\ =& 0.1306 \end{aligned}.$$

Nói cách khác, chỉ có 13,06% khả năng bệnh nhân thực sự bị nhiễm HIV, mặc dù sử dụng một xét nghiệm rất chính xác. Như chúng ta có thể thấy, xác suất có thể phản trực quan. 

Bệnh nhân nên làm gì khi nhận được những tin tức đáng sợ như vậy? Có khả năng, bệnh nhân sẽ yêu cầu bác sĩ quản lý một xét nghiệm khác để có được sự rõ ràng. Thử nghiệm thứ hai có các đặc điểm khác nhau và nó không tốt bằng thử nghiệm đầu tiên, như thể hiện trong :numref:`conditional_prob_D2`. 

: Xác suất điều kiện của $P(D_2 \mid H)$. 

| Conditional probability | $H=1$ | $H=0$ |
|---|---|---|
|$P(D_2 = 1 \mid H)$|            0.98 |         0.03 |
|$P(D_2 = 0 \mid H)$|            0.02 |         0.97 |
:label:`conditional_prob_D2`

Thật không may, thử nghiệm thứ hai trở lại tích cực, quá. Chúng ta hãy tìm ra các xác suất cần thiết để gọi định lý Bayes' bằng cách giả định độc lập có điều kiện: 

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

Bây giờ chúng ta có thể áp dụng marginalization và quy tắc nhân: 

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1) \\
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
=& 0.00176955.
\end{aligned}
$$

Cuối cùng, xác suất bệnh nhân nhiễm HIV được đưa ra cả hai xét nghiệm dương tính là 

$$\begin{aligned}
&P(H = 1 \mid D_1 = 1, D_2 = 1)\\
=& \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)} \\
=& 0.8307.
\end{aligned}
$$

Đó là, bài kiểm tra thứ hai cho phép chúng tôi đạt được sự tự tin cao hơn nhiều rằng không phải tất cả đều tốt. Mặc dù thử nghiệm thứ hai kém chính xác hơn đáng kể so với thử nghiệm đầu tiên, nhưng nó vẫn cải thiện đáng kể ước tính của chúng tôi. 

## Kỳ vọng và phương sai

Để tóm tắt các đặc điểm chính của phân phối xác suất, chúng ta cần một số biện pháp. *kỳ vị* (hoặc trung bình) của biến ngẫu nhiên $X$ được ký hiệu là 

$$E[X] = \sum_{x} x P(X = x).$$

Khi đầu vào của một hàm $f(x)$ là một biến ngẫu nhiên rút ra từ phân phối $P$ với các giá trị khác nhau $x$, kỳ vọng của $f(x)$ được tính là 

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x).$$

Trong nhiều trường hợp, chúng tôi muốn đo lường bằng bao nhiêu biến ngẫu nhiên $X$ lệch khỏi kỳ vọng của nó. Điều này có thể được định lượng bởi phương sai 

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] =
E[X^2] - E[X]^2.$$

Căn bậc hai của nó được gọi là độ lệch tiêu chuẩn*. Phương sai của một hàm số của một phép đo biến ngẫu nhiên bằng bao nhiêu hàm lệch so với kỳ vọng của hàm, vì các giá trị khác nhau $x$ của biến ngẫu nhiên được lấy mẫu từ phân phối của nó: 

$$\mathrm{Var}[f(x)] = E\left[\left(f(x) - E[f(x)]\right)^2\right].$$

## Tóm tắt

* Chúng tôi có thể lấy mẫu từ phân phối xác suất.
* Chúng ta có thể phân tích nhiều biến ngẫu nhiên bằng cách sử dụng phân phối chung, phân phối có điều kiện, định lý Bayes', marginalization, và giả định độc lập.
* Kỳ vọng và phương sai đưa ra các biện pháp hữu ích để tóm tắt các đặc điểm chính của phân phối xác suất.

## Bài tập

1. Chúng tôi đã tiến hành $m=500$ nhóm thí nghiệm trong đó mỗi nhóm rút ra $n=10$ mẫu. Vary $m$ và $n$. Quan sát và phân tích kết quả thử nghiệm.
1. Cho hai sự kiện với xác suất $P(\mathcal{A})$ và $P(\mathcal{B})$, tính toán giới hạn trên và dưới trên $P(\mathcal{A} \cup \mathcal{B})$ và $P(\mathcal{A} \cap \mathcal{B})$. (Gợi ý: hiển thị tình hình bằng cách sử dụng [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram).)
1. Giả sử rằng chúng ta có một chuỗi các biến ngẫu nhiên, nói $A$, $B$, và $C$, trong đó $B$ chỉ phụ thuộc vào $A$, và $C$ chỉ phụ thuộc vào $B$, bạn có thể đơn giản hóa xác suất chung $P(A, B, C)$? (Gợi ý: đây là một [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain).)
1. Năm :numref:`subsec_probability_hiv_app`, bài kiểm tra đầu tiên chính xác hơn. Tại sao không chạy thử nghiệm đầu tiên hai lần thay vì chạy cả thử nghiệm thứ nhất và thứ hai?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:

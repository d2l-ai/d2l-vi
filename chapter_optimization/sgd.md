# Stochastic Gradient Descent
:label:`sec_sgd`

Tuy nhiên, trong các chương trước, chúng tôi tiếp tục sử dụng gốc gradient ngẫu nhiên trong quy trình đào tạo của chúng tôi, tuy nhiên, mà không giải thích lý do tại sao nó hoạt động. Để làm sáng tỏ nó, chúng tôi chỉ mô tả các nguyên tắc cơ bản của gradient gốc trong :numref:`sec_gd`. Trong phần này, chúng tôi tiếp tục thảo luận
*xuống dốc ngẫu nhiên* chi tiết hơn.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## Stochastic Gradient Những Thông Tin Cập Nhập

Trong học sâu, chức năng khách quan thường là trung bình của các hàm mất cho mỗi ví dụ trong tập dữ liệu đào tạo. Với một tập dữ liệu đào tạo $n$ ví dụ, chúng tôi giả định rằng $f_i(\mathbf{x})$ là hàm mất liên quan đến ví dụ đào tạo của chỉ số $i$, trong đó $\mathbf{x}$ là vectơ tham số. Sau đó, chúng tôi đến chức năng mục tiêu 

$$f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\mathbf{x}).$$

Gradient của hàm khách quan tại $\mathbf{x}$ được tính là 

$$\nabla f(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}).$$

Nếu sử dụng gradient descent, chi phí tính toán cho mỗi lần lặp biến độc lập là $\mathcal{O}(n)$, phát triển tuyến tính với $n$. Do đó, khi tập dữ liệu đào tạo lớn hơn, chi phí của gradient descent cho mỗi lần lặp sẽ cao hơn. 

Stochastic gradient descent (SGD) giảm chi phí tính toán tại mỗi lần lặp lại. Tại mỗi lần lặp lại của dòng dốc ngẫu nhiên, chúng tôi lấy mẫu thống nhất một chỉ số $i\in\{1,\ldots, n\}$ cho các ví dụ dữ liệu một cách ngẫu nhiên và tính toán gradient $\nabla f_i(\mathbf{x})$ để cập nhật $\mathbf{x}$: 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f_i(\mathbf{x}),$$

trong đó $\eta$ là tỷ lệ học tập. Chúng ta có thể thấy rằng chi phí tính toán cho mỗi lần lặp giảm từ $\mathcal{O}(n)$ của gradient gốc xuống hằng số $\mathcal{O}(1)$. Hơn nữa, chúng tôi muốn nhấn mạnh rằng gradient ngẫu nhiên $\nabla f_i(\mathbf{x})$ là một ước tính không thiên vị của gradient đầy đủ $\nabla f(\mathbf{x})$ bởi vì 

$$\mathbb{E}_i \nabla f_i(\mathbf{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(\mathbf{x}).$$

Điều này có nghĩa là, trung bình, gradient stochastic là một ước tính tốt của gradient. 

Bây giờ, chúng ta sẽ so sánh nó với gradient descent bằng cách thêm nhiễu ngẫu nhiên với trung bình 0 và phương sai 1 với gradient để mô phỏng một gradient gốc ngẫu nhiên.

```{.python .input}
#@tab all
def f(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # Gradient of the objective function
    return 2 * x1, 4 * x2
```

```{.python .input}
#@tab mxnet, pytorch
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal(0.0, 1, (1,))
    g2 += d2l.normal(0.0, 1, (1,))
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab tensorflow
def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # Simulate noisy gradient
    g1 += d2l.normal([1], 0.0, 1)
    g2 += d2l.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)
```

```{.python .input}
#@tab all
def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # Constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Như chúng ta có thể thấy, quỹ đạo của các biến trong gốc gradient ngẫu nhiên ồn hơn nhiều so với quỹ đạo mà chúng ta quan sát thấy trong gradient gốc trong :numref:`sec_gd`. Điều này là do tính chất ngẫu nhiên của gradient. Đó là, ngay cả khi chúng tôi đến gần mức tối thiểu, chúng tôi vẫn phải chịu sự không chắc chắn được tiêm bởi gradient tức thời qua $\eta \nabla f_i(\mathbf{x})$. Ngay cả sau 50 bước chất lượng vẫn không tốt như vậy. Thậm chí tệ hơn, nó sẽ không cải thiện sau các bước bổ sung (chúng tôi khuyến khích bạn thử nghiệm với một số lượng lớn hơn các bước để xác nhận điều này). Điều này để lại cho chúng ta sự thay thế duy nhất: thay đổi tỷ lệ học tập $\eta$. Tuy nhiên, nếu chúng ta chọn điều này quá nhỏ, ban đầu chúng ta sẽ không đạt được bất kỳ tiến bộ có ý nghĩa nào. Mặt khác, nếu chúng ta chọn nó quá lớn, chúng ta sẽ không nhận được một giải pháp tốt, như đã thấy ở trên. Cách duy nhất để giải quyết các mục tiêu mâu thuẫn này là giảm tỷ lệ học tập * năng lượng* khi tối ưu hóa tiến triển. 

Đây cũng là lý do để thêm một hàm tốc độ học tập `lr` vào hàm bước `sgd`. Trong ví dụ trên bất kỳ chức năng nào để lập kế hoạch tỷ lệ học tập nằm không hoạt động khi chúng ta đặt hàm `lr` liên quan là hằng số. 

## Tốc độ học động

Thay thế $\eta$ bằng tốc độ học tập phụ thuộc vào thời gian $\eta(t)$ thêm vào sự phức tạp của việc kiểm soát sự hội tụ của một thuật toán tối ưu hóa. Đặc biệt, chúng ta cần tìm ra cách nhanh chóng $\eta$ sẽ phân rã. Nếu quá nhanh, chúng tôi sẽ ngừng tối ưu hóa sớm. Nếu chúng ta giảm nó quá chậm, chúng ta lãng phí quá nhiều thời gian để tối ưu hóa. Sau đây là một vài chiến lược cơ bản được sử dụng để điều chỉnh $\eta$ theo thời gian (chúng tôi sẽ thảo luận về các chiến lược nâng cao hơn sau): 

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \text{piecewise constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \text{exponential decay} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \text{polynomial decay}
\end{aligned}
$$

Trong kịch bản *piecewise constant* đầu tiên, chúng tôi giảm tỷ lệ học tập, ví dụ, bất cứ khi nào tiến bộ trong các quầy hàng tối ưu hóa. Đây là một chiến lược phổ biến để đào tạo các mạng sâu. Ngoài ra, chúng ta có thể giảm nó mạnh mẽ hơn nhiều bởi sự phân rã theo cấp số nhân *. Thật không may điều này thường dẫn đến dừng sớm trước khi thuật toán đã hội tụ. Một lựa chọn phổ biến là * phân rã đa thứ* với $\alpha = 0.5$. Trong trường hợp tối ưu hóa lồi, có một số bằng chứng cho thấy tỷ lệ này được cư xử tốt. 

Chúng ta hãy xem sự phân rã theo cấp số nhân trông như thế nào trong thực tế.

```{.python .input}
#@tab all
def exponential_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return math.exp(-0.1 * t)

t = 1
lr = exponential_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000, f_grad=f_grad))
```

Đúng như dự đoán, phương sai trong các thông số được giảm đáng kể. Tuy nhiên, điều này đến với chi phí không hội tụ với giải pháp tối ưu $\mathbf{x} = (0, 0)$. Ngay cả sau 1000 bước lặp lại chúng ta vẫn còn rất xa giải pháp tối ưu. Thật vậy, thuật toán không hội tụ ở tất cả. Mặt khác, nếu chúng ta sử dụng phân rã đa thức trong đó tốc độ học tập phân rã với căn bậc hai nghịch đảo của số bước, sự hội tụ sẽ tốt hơn chỉ sau 50 bước.

```{.python .input}
#@tab all
def polynomial_lr():
    # Global variable that is defined outside this function and updated inside
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = polynomial_lr
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50, f_grad=f_grad))
```

Tồn tại nhiều lựa chọn hơn cho cách thiết lập tỷ lệ học tập. Ví dụ, chúng ta có thể bắt đầu với một tốc độ nhỏ, sau đó nhanh chóng tăng lên và sau đó giảm nó một lần nữa, mặc dù chậm hơn. Chúng tôi thậm chí có thể thay thế giữa tỷ lệ học tập nhỏ hơn và lớn hơn. Có tồn tại một loạt các lịch trình như vậy. Bây giờ chúng ta hãy tập trung vào lịch trình học tập mà có thể phân tích lý thuyết toàn diện, tức là về tốc độ học tập trong một môi trường lồi. Đối với các bài toán phi lồi nói chung, rất khó để có được những đảm bảo hội tụ có ý nghĩa, vì nói chung giảm thiểu các bài toán phi lồi phi tuyến là NP cứng. Đối với một cuộc khảo sát xem ví dụ, xuất sắc [bài giảng ghi chú](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) của Tibshirani 2015. 

## Phân tích hội tụ cho mục tiêu lồi

Phân tích hội tụ sau đây của gốc gradient ngẫu nhiên cho các chức năng mục tiêu lồi là tùy chọn và chủ yếu phục vụ để truyền đạt nhiều trực giác hơn về vấn đề. Chúng tôi giới hạn bản thân mình với một trong những bằng chứng đơn giản nhất :cite:`Nesterov.Vial.2000`. Các kỹ thuật chứng minh tiên tiến hơn đáng kể tồn tại, ví dụ, bất cứ khi nào chức năng mục tiêu được cư xử đặc biệt tốt. 

Giả sử hàm khách quan $f(\boldsymbol{\xi}, \mathbf{x})$ là lồi trong $\mathbf{x}$ cho tất cả $\boldsymbol{\xi}$. Cụ thể hơn, chúng tôi xem xét bản cập nhật gradient gốc stochastic: 

$$\mathbf{x}_{t+1} = \mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}),$$

trong đó $f(\boldsymbol{\xi}_t, \mathbf{x})$ là chức năng khách quan đối với ví dụ đào tạo $\boldsymbol{\xi}_t$ rút ra từ một số phân phối ở bước $t$ và $\mathbf{x}$ là tham số mô hình. Biểu thị bởi 

$$R(\mathbf{x}) = E_{\boldsymbol{\xi}}[f(\boldsymbol{\xi}, \mathbf{x})]$$

rủi ro dự kiến và đến $R^*$ mức tối thiểu của nó đối với $\mathbf{x}$. Cuối cùng cho phép $\mathbf{x}^*$ là minimizer (chúng tôi giả định rằng nó tồn tại trong miền nơi $\mathbf{x}$ được xác định). Trong trường hợp này, chúng ta có thể theo dõi khoảng cách giữa tham số hiện tại $\mathbf{x}_t$ tại thời điểm $t$ và bộ giảm thiểu rủi ro $\mathbf{x}^*$ và xem liệu nó có cải thiện theo thời gian hay không: 

$$\begin{aligned}    &\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \\ =& \|\mathbf{x}_{t} - \eta_t \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x}) - \mathbf{x}^*\|^2 \\    =& \|\mathbf{x}_{t} - \mathbf{x}^*\|^2 + \eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 - 2 \eta_t    \left\langle \mathbf{x}_t - \mathbf{x}^*, \partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\right\rangle.   \end{aligned}$$
:eqlabel:`eq_sgd-xt+1-xstar`

Chúng tôi giả định rằng định mức $L_2$ của gradient ngẫu nhiên $\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})$ được giới hạn bởi một số hằng số $L$, do đó chúng tôi có 

$$\eta_t^2 \|\partial_\mathbf{x} f(\boldsymbol{\xi}_t, \mathbf{x})\|^2 \leq \eta_t^2 L^2.$$
:eqlabel:`eq_sgd-L`

Chúng tôi chủ yếu quan tâm đến khoảng cách giữa $\mathbf{x}_t$ và $\mathbf{x}^*$ thay đổi như thế nào *theo mong đợ*. Trên thực tế, đối với bất kỳ trình tự cụ thể nào của các bước, khoảng cách có thể tăng lên, tùy thuộc vào bất kỳ $\boldsymbol{\xi}_t$ nào chúng ta gặp phải. Do đó chúng ta cần phải ràng buộc các sản phẩm dot. Kể từ khi đối với bất kỳ chức năng lồi $f$ nó giữ rằng $f(\mathbf{y}) \geq f(\mathbf{x}) + \langle f'(\mathbf{x}), \mathbf{y} - \mathbf{x} \rangle$ cho tất cả $\mathbf{x}$ và $\mathbf{y}$, bởi lồi chúng tôi có 

$$f(\boldsymbol{\xi}_t, \mathbf{x}^*) \geq f(\boldsymbol{\xi}_t, \mathbf{x}_t) + \left\langle \mathbf{x}^* - \mathbf{x}_t, \partial_{\mathbf{x}} f(\boldsymbol{\xi}_t, \mathbf{x}_t) \right\rangle.$$
:eqlabel:`eq_sgd-f-xi-xstar`

Cắm cả hai bất đẳng thức :eqref:`eq_sgd-L` và :eqref:`eq_sgd-f-xi-xstar` vào :eqref:`eq_sgd-xt+1-xstar` chúng tôi có được một ràng buộc về khoảng cách giữa các tham số tại thời điểm $t+1$ như sau: 

$$\|\mathbf{x}_{t} - \mathbf{x}^*\|^2 - \|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2 \geq 2 \eta_t (f(\boldsymbol{\xi}_t, \mathbf{x}_t) - f(\boldsymbol{\xi}_t, \mathbf{x}^*)) - \eta_t^2 L^2.$$
:eqlabel:`eqref_sgd-xt-diff`

Điều này có nghĩa là chúng tôi đạt được tiến bộ miễn là sự khác biệt giữa tổn thất hiện tại và tổn thất tối ưu lớn hơn $\eta_t L^2/2$. Vì sự khác biệt này được ràng buộc để hội tụ với 0 nên sau đó tỷ lệ học tập $\eta_t$ cũng cần phải * biến mất *. 

Tiếp theo, chúng tôi có kỳ vọng hơn :eqref:`eqref_sgd-xt-diff`. This yields năng suất 

$$E\left[\|\mathbf{x}_{t} - \mathbf{x}^*\|^2\right] - E\left[\|\mathbf{x}_{t+1} - \mathbf{x}^*\|^2\right] \geq 2 \eta_t [E[R(\mathbf{x}_t)] - R^*] -  \eta_t^2 L^2.$$

Bước cuối cùng liên quan đến việc tổng hợp các bất bình đẳng cho $t \in \{1, \ldots, T\}$. Kể từ khi tổng kính thiên văn và bằng cách giảm thuật ngữ thấp hơn, chúng tôi thu được 

$$\|\mathbf{x}_1 - \mathbf{x}^*\|^2 \geq 2 \left (\sum_{t=1}^T   \eta_t \right) [E[R(\mathbf{x}_t)] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
:eqlabel:`eq_sgd-x1-xstar`

Lưu ý rằng chúng tôi đã khai thác rằng $\mathbf{x}_1$ được đưa ra và do đó kỳ vọng có thể được giảm xuống. Định nghĩa cuối 

$$\bar{\mathbf{x}} \stackrel{\mathrm{def}}{=} \frac{\sum_{t=1}^T \eta_t \mathbf{x}_t}{\sum_{t=1}^T \eta_t}.$$

Kể từ 

$$E\left(\frac{\sum_{t=1}^T \eta_t R(\mathbf{x}_t)}{\sum_{t=1}^T \eta_t}\right) = \frac{\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)]}{\sum_{t=1}^T \eta_t} = E[R(\mathbf{x}_t)],$$

bởi sự bất bình đẳng của Jensen (thiết lập $i=t$, $\alpha_i = \eta_t/\sum_{t=1}^T \eta_t$ trong :eqref:`eq_jensens-inequality`) và độ lồi của $R$ nó theo sau đó $E[R(\mathbf{x}_t)] \geq E[R(\bar{\mathbf{x}})]$, do đó 

$$\sum_{t=1}^T \eta_t E[R(\mathbf{x}_t)] \geq \sum_{t=1}^T \eta_t  E\left[R(\bar{\mathbf{x}})\right].$$

Cắm này vào bất bình đẳng :eqref:`eq_sgd-x1-xstar` mang lại sự ràng buộc 

$$
\left[E[\bar{\mathbf{x}}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t},
$$

trong đó $r^2 \stackrel{\mathrm{def}}{=} \|\mathbf{x}_1 - \mathbf{x}^*\|^2$ là một ràng buộc về khoảng cách giữa sự lựa chọn ban đầu của các tham số và kết quả cuối cùng. Nói tóm lại, tốc độ hội tụ phụ thuộc vào cách định mức của gradient ngẫu nhiên được giới hạn ($L$) và cách tối ưu giá trị tham số ban đầu là bao xa ($r$). Lưu ý rằng sự ràng buộc là về $\bar{\mathbf{x}}$ chứ không phải là $\mathbf{x}_T$. Đây là trường hợp vì $\bar{\mathbf{x}}$ là một phiên bản được làm mịn của đường dẫn tối ưu hóa. Bất cứ khi nào $r, L$, và $T$ được biết đến, chúng tôi có thể chọn tỷ lệ học tập $\eta = r/(L \sqrt{T})$. Điều này mang lại như trên ràng buộc $rL/\sqrt{T}$. Đó là, chúng tôi hội tụ với tỷ lệ $\mathcal{O}(1/\sqrt{T})$ đến giải pháp tối ưu. 

## Gradient Stochastic và mẫu hữu hạn

Cho đến nay chúng tôi đã chơi một chút nhanh và lỏng lẻo khi nói về gốc gradient ngẫu nhiên. Chúng tôi xác định rằng chúng tôi vẽ các trường hợp $x_i$, thường với nhãn $y_i$ từ một số phân phối $p(x, y)$ và chúng tôi sử dụng điều này để cập nhật các tham số mô hình theo một cách nào đó. Đặc biệt, đối với một kích thước mẫu hữu hạn, chúng tôi chỉ đơn giản lập luận rằng phân phối rời rạc $p(x, y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ cho một số chức năng $\delta_{x_i}$ và $\delta_{y_i}$ cho phép chúng tôi thực hiện chuyển đổi gradient ngẫu nhiên trên nó. 

Tuy nhiên, đây không thực sự là những gì chúng tôi đã làm. Trong các ví dụ đồ chơi trong phần hiện tại, chúng tôi chỉ cần thêm tiếng ồn vào một gradient không ngẫu nhiên khác, tức là, chúng tôi giả vờ có cặp $(x_i, y_i)$. Nó chỉ ra rằng điều này là hợp lý ở đây (xem các bài tập để thảo luận chi tiết). Rắc rối hơn là trong tất cả các cuộc thảo luận trước đó, chúng tôi rõ ràng đã không làm điều này. Thay vào đó, chúng tôi lặp lại tất cả các phiên bản * chính xác một lầu*. Để xem lý do tại sao điều này là thích hợp hơn, hãy xem xét cuộc trò chuyện, cụ thể là chúng tôi đang lấy mẫu $n$ quan sát từ phân phối rời rạc * với thay thế*. Xác suất chọn một phần tử $i$ ngẫu nhiên là $1/n$. Do đó để chọn nó * ít nhất* một lần là 

$$P(\mathrm{choose~} i) = 1 - P(\mathrm{omit~} i) = 1 - (1-1/n)^n \approx 1-e^{-1} \approx 0.63.$$

Một lý do tương tự cho thấy xác suất chọn một số mẫu (tức là ví dụ đào tạo) * chính xác một lầu* được đưa ra bởi 

$${n \choose 1} \frac{1}{n} \left(1-\frac{1}{n}\right)^{n-1} = \frac{n}{n-1} \left(1-\frac{1}{n}\right)^{n} \approx e^{-1} \approx 0.37.$$

Điều này dẫn đến sự gia tăng phương sai và giảm hiệu quả dữ liệu so với lấy mẫu * mà không thay thế*. Do đó, trong thực tế, chúng tôi thực hiện sau này (và đây là lựa chọn mặc định trong suốt cuốn sách này). Lưu ý cuối cùng rằng lặp đi lặp lại qua tập dữ liệu đào tạo đi qua nó theo thứ tự ngẫu nhiên *khác nhau*. 

## Tóm tắt

* Đối với các vấn đề lồi, chúng ta có thể chứng minh rằng đối với một sự lựa chọn rộng của tỷ lệ học tập stochastic gradient gốc sẽ hội tụ với giải pháp tối ưu.
* Đối với học sâu, điều này thường không phải là trường hợp. Tuy nhiên, việc phân tích các bài toán lồi cho chúng ta cái nhìn sâu sắc hữu ích về cách tiếp cận tối ưu hóa, cụ thể là giảm tốc độ học tập dần dần, mặc dù không quá nhanh.
* Vấn đề xảy ra khi tỷ lệ học tập quá nhỏ hoặc quá lớn. Trong thực tế, một tỷ lệ học tập phù hợp thường chỉ được tìm thấy sau nhiều thí nghiệm.
* Khi có nhiều ví dụ hơn trong tập dữ liệu đào tạo, chi phí nhiều hơn để tính toán mỗi lần lặp cho gradient descent, do đó, stochastic gradient descent được ưa thích trong những trường hợp này.
* Đảm bảo tối ưu cho dòng dốc ngẫu nhiên nói chung không có sẵn trong các trường hợp không lồi vì số lượng minima cục bộ yêu cầu kiểm tra cũng có thể là cấp số nhân.

## Bài tập

1. Thử nghiệm với các lịch trình tỷ lệ học tập khác nhau cho gốc gradient ngẫu nhiên và với các số lần lặp khác nhau. Đặc biệt, vẽ khoảng cách từ giải pháp tối ưu $(0, 0)$ như một hàm của số lần lặp lại.
1. Chứng minh rằng đối với hàm $f(x_1, x_2) = x_1^2 + 2 x_2^2$ thêm nhiễu bình thường vào gradient tương đương với việc giảm thiểu hàm mất $f(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ trong đó $\mathbf{x}$ được rút ra từ một phân phối bình thường.
1. So sánh sự hội tụ của gốc gradient ngẫu nhiên khi bạn lấy mẫu từ $\{(x_1, y_1), \ldots, (x_n, y_n)\}$ với sự thay thế và khi bạn lấy mẫu mà không cần thay thế.
1. Làm thế nào bạn sẽ thay đổi các stochastic gradient descent solver nếu một số gradient (hoặc đúng hơn là một số phối hợp liên quan đến nó) đã luôn lớn hơn tất cả các gradient khác?
1. Giả sử rằng $f(x) = x^2 (1 + \sin x)$. $f$ có bao nhiêu minima địa phương? Bạn có thể thay đổi $f$ theo cách để giảm thiểu nó, người ta cần đánh giá tất cả các minima địa phương không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/352)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/497)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1067)
:end_tab:

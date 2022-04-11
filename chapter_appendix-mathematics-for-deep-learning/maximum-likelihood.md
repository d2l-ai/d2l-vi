# Khả năng tối đa
:label:`sec_maximum_likelihood`

Một trong những cách suy nghĩ phổ biến nhất trong học máy là quan điểm khả năng tối đa. Đây là khái niệm rằng khi làm việc với một mô hình xác suất với các tham số không xác định, các tham số làm cho dữ liệu có xác suất cao nhất là những thông số có khả năng nhất. 

## Nguyên tắc khả năng tối đa

Điều này có một cách giải thích Bayesian có thể hữu ích để suy nghĩ về. Giả sử rằng chúng ta có một mô hình với các tham số $\boldsymbol{\theta}$ và một tập hợp các ví dụ dữ liệu $X$. Đối với sự cụ thể, chúng ta có thể tưởng tượng rằng $\boldsymbol{\theta}$ là một giá trị duy nhất đại diện cho xác suất một đồng xu xuất hiện đầu khi lật, và $X$ là một chuỗi các lật đồng xu độc lập. Chúng tôi sẽ xem xét ví dụ này trong chiều sâu sau. 

Nếu chúng ta muốn tìm giá trị có khả năng nhất cho các tham số của mô hình của chúng ta, điều đó có nghĩa là chúng ta muốn tìm 

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`

Theo quy tắc của Bayes, đây là điều tương tự như 

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

Biểu thức $P(X)$, một tham số xác suất bất khả tri của việc tạo dữ liệu, không phụ thuộc vào $\boldsymbol{\theta}$ ở tất cả, và do đó có thể được giảm mà không thay đổi sự lựa chọn tốt nhất của $\boldsymbol{\theta}$. Tương tự, bây giờ chúng ta có thể khẳng định rằng chúng tôi không có giả định trước về tập hợp các tham số nào tốt hơn bất kỳ tham số nào khác, vì vậy chúng tôi có thể tuyên bố rằng $P(\boldsymbol{\theta})$ cũng không phụ thuộc vào theta! Ví dụ, điều này có ý nghĩa trong ví dụ lật đồng xu của chúng tôi, nơi xác suất nó xuất hiện đầu có thể là bất kỳ giá trị nào trong $[0,1]$ mà không có bất kỳ niềm tin nào trước đó là công bằng hay không (thường được gọi là ưu tiên * không thông tin*). Do đó, chúng tôi thấy rằng việc áp dụng quy tắc Bayes' cho thấy sự lựa chọn tốt nhất của chúng tôi về $\boldsymbol{\theta}$ là ước tính khả năng tối đa cho $\boldsymbol{\theta}$: 

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

Như một vấn đề của thuật ngữ phổ biến, xác suất của dữ liệu được đưa ra các tham số ($P(X \mid \boldsymbol{\theta})$) được gọi là *likelihood*. 

### Một ví dụ cụ thể

Hãy để chúng tôi xem làm thế nào điều này hoạt động trong một ví dụ cụ thể. Giả sử rằng chúng ta có một tham số duy nhất $\theta$ đại diện cho xác suất rằng một đồng xu lật là đầu. Sau đó, xác suất nhận được đuôi là $1-\theta$, và vì vậy nếu dữ liệu quan sát của chúng tôi $X$ là một chuỗi với $n_H$ đầu và $n_T$ đuôi, chúng ta có thể sử dụng thực tế là xác suất độc lập nhân lên để thấy rằng  

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

Nếu chúng ta lật $13$ coin và nhận được chuỗi “HHHTHTTHHHHHT”, có $n_H = 9$ và $n_T = 4$, chúng ta thấy rằng đây là 

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

Một điều hay về ví dụ này sẽ là chúng ta biết câu trả lời đang diễn ra. Thật vậy, nếu chúng tôi nói bằng lời nói, “Tôi lật 13 đồng xu, và 9 người đứng đầu, đoán tốt nhất của chúng tôi cho xác suất đồng xu đến chúng ta đứng đầu là gì? , "mọi người sẽ đoán chính xác $9/13$. Phương pháp khả năng tối đa này sẽ cung cấp cho chúng ta là một cách để có được con số đó từ các hiệu trưởng đầu tiên theo cách sẽ khái quát hóa đến các tình huống phức tạp hơn rất nhiều. 

Đối với ví dụ của chúng tôi, cốt truyện của $P(X \mid \theta)$ như sau:

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

Điều này có giá trị tối đa của nó ở đâu đó gần $9/13 \approx 0.7\ldots$ dự kiến của chúng tôi. Để xem liệu nó có chính xác ở đó không, chúng ta có thể chuyển sang tính toán. Lưu ý rằng ở mức tối đa, gradient của hàm là phẳng. Do đó, chúng ta có thể tìm thấy ước tính khả năng tối đa :eqref:`eq_max_like` bằng cách tìm các giá trị của $\theta$ trong đó đạo hàm bằng 0 và tìm giá trị cho xác suất cao nhất. Chúng tôi tính toán: 

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

Điều này có ba giải pháp: $0$, $1$ và $9/13$. Hai đầu tiên rõ ràng là minima, không phải maxima như họ gán xác suất $0$ cho trình tự của chúng tôi. Giá trị cuối cùng không * không* gán xác suất bằng 0 cho trình tự của chúng tôi, và do đó phải là ước tính khả năng tối đa $\hat \theta = 9/13$. 

## Tối ưu hóa số và tiêu cực Log-Likelihood

Ví dụ trước là tốt đẹp, nhưng nếu chúng ta có hàng tỷ tham số và ví dụ dữ liệu thì sao? 

Đầu tiên, lưu ý rằng nếu chúng ta đưa ra giả định rằng tất cả các ví dụ dữ liệu đều độc lập, chúng ta không còn thực tế có thể xem xét khả năng vì nó là sản phẩm của nhiều xác suất. Thật vậy, mỗi xác suất là trong $[0,1]$, nói thường có giá trị khoảng $1/2$, và sản phẩm của $(1/2)^{1000000000}$ thấp hơn nhiều so với độ chính xác của máy. Chúng tôi không thể làm việc trực tiếp với điều đó.   

Tuy nhiên, hãy nhớ lại rằng logarit biến sản phẩm thành một khoản tiền, trong trường hợp đó  

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

Con số này phù hợp hoàn hảo ngay cả trong một độ chính xác duy nhất $32$-bit float. Vì vậy, chúng ta nên xem xét *log-likelihood*, đó là 

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

Kể từ khi chức năng $x \mapsto \log(x)$ đang tăng lên, tối đa hóa khả năng là điều tương tự như tối đa hóa khả năng log. Thật vậy trong :numref:`sec_naive_bayes`, chúng ta sẽ thấy lý do này được áp dụng khi làm việc với ví dụ cụ thể về phân loại Bayes ngây thơ. 

Chúng tôi thường làm việc với các chức năng mất mát, nơi chúng tôi muốn giảm thiểu tổn thất. Chúng tôi có thể biến khả năng tối đa thành việc giảm thiểu tổn thất bằng cách lấy $-\log(P(X \mid \boldsymbol{\theta}))$, đó là *âm log-likelihood*. 

Để minh họa điều này, hãy xem xét vấn đề lật đồng xu từ trước, và giả vờ rằng chúng ta không biết giải pháp hình thức đóng. We may Tháng Năm compute tính toán that 

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

Điều này có thể được viết thành mã, và tự do tối ưu hóa ngay cả đối với hàng tỷ flips đồng xu.

```{.python .input}
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

Sự tiện lợi về số không phải là lý do duy nhất tại sao mọi người thích sử dụng tiêu cực log-likelihoods. Có một số lý do khác tại sao nó là thích hợp hơn. 

Lý do thứ hai chúng tôi xem xét khả năng đăng nhập là việc áp dụng đơn giản hóa các quy tắc tính toán. Như đã thảo luận ở trên, do các giả định độc lập, hầu hết các xác suất chúng ta gặp phải trong học máy là sản phẩm của xác suất cá nhân. 

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

Điều này có nghĩa là nếu chúng ta trực tiếp áp dụng quy tắc sản phẩm để tính toán một đạo hàm, chúng ta nhận được 

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

Điều này đòi hỏi $n(n-1)$ phép nhân, cùng với $(n-1)$ bổ sung, vì vậy nó tỷ lệ thuận với thời gian bậc hai trong các đầu vào! Đủ thông minh trong các thuật ngữ nhóm sẽ làm giảm điều này đến thời gian tuyến tính, nhưng nó đòi hỏi một số suy nghĩ. Đối với bản ghi âm khả năng chúng tôi có thay vào đó 

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

mà sau đó cho 

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

Điều này chỉ đòi hỏi $n$ phân chia và $n-1$ tổng, và do đó là thời gian tuyến tính trong các đầu vào. 

Lý do thứ ba và cuối cùng để xem xét khả năng đăng nhập tiêu cực là mối quan hệ với lý thuyết thông tin, mà chúng ta sẽ thảo luận chi tiết trong :numref:`sec_information_theory`. Đây là một lý thuyết toán học nghiêm ngặt mà đưa ra một cách để đo mức độ thông tin hoặc ngẫu nhiên trong một biến ngẫu nhiên. Đối tượng quan trọng của nghiên cứu trong lĩnh vực đó là entropy  

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

which đo the randomnessngẫu nhiên of a sourcenguồn. Lưu ý rằng đây không có gì khác hơn là xác suất $-\log$ trung bình, và do đó nếu chúng ta lấy khả năng log âm của chúng tôi và chia cho số lượng các ví dụ dữ liệu, chúng tôi nhận được một người họ hàng của entropy được gọi là cross-entropy. Việc giải thích lý thuyết này một mình sẽ đủ hấp dẫn để thúc đẩy báo cáo khả năng nhật ký âm trung bình trên tập dữ liệu như một cách đo lường hiệu suất mô hình. 

## Khả năng tối đa cho các biến liên tục

Tất cả mọi thứ mà chúng ta đã làm cho đến nay giả định chúng ta đang làm việc với các biến ngẫu nhiên rời rạc, nhưng nếu chúng ta muốn làm việc với các biến liên tục thì sao? 

Tóm tắt ngắn gọn là không có gì ở tất cả các thay đổi, ngoại trừ chúng tôi thay thế tất cả các trường hợp của xác suất với mật độ xác suất. Nhắc lại rằng chúng tôi viết mật độ với chữ thường $p$, điều này có nghĩa là ví dụ chúng tôi nói 

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

Câu hỏi trở thành, “Tại sao điều này ổn?” Rốt cuộc, lý do chúng tôi giới thiệu mật độ là do xác suất nhận được kết quả cụ thể là bằng không, và do đó không phải là xác suất tạo dữ liệu của chúng tôi cho bất kỳ tập hợp các tham số không? 

Thật vậy, đây là trường hợp, và hiểu lý do tại sao chúng ta có thể chuyển sang mật độ là một bài tập trong việc truy tìm những gì xảy ra với epsilon. 

Trước tiên chúng ta hãy xác định lại mục tiêu của chúng tôi. Giả sử rằng đối với các biến ngẫu nhiên liên tục, chúng ta không còn muốn tính toán xác suất nhận được chính xác giá trị phù hợp, mà thay vào đó phù hợp với trong một số phạm vi $\epsilon$. Để đơn giản, chúng tôi giả định dữ liệu của chúng tôi được lặp đi lặp lại quan sát $x_1, \ldots, x_N$ của các biến ngẫu nhiên phân phối giống hệt nhau $X_1, \ldots, X_N$. Như chúng ta đã thấy trước đây, điều này có thể được viết là 

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

Do đó, nếu chúng ta lấy logarit âm của điều này, chúng ta có được 

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

Nếu chúng ta kiểm tra biểu thức này, nơi duy nhất mà $\epsilon$ xảy ra là trong hằng số phụ gia $-N\log(\epsilon)$. Điều này hoàn toàn không phụ thuộc vào các thông số $\boldsymbol{\theta}$, vì vậy sự lựa chọn tối ưu của $\boldsymbol{\theta}$ không phụ thuộc vào sự lựa chọn của chúng tôi là $\epsilon$! Nếu chúng tôi yêu cầu bốn chữ số hoặc bốn trăm, sự lựa chọn tốt nhất của $\boldsymbol{\theta}$ vẫn giữ nguyên, do đó chúng tôi có thể tự do thả epsilon để thấy rằng những gì chúng tôi muốn tối ưu hóa là 

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

Do đó, chúng ta thấy rằng quan điểm khả năng tối đa có thể hoạt động với các biến ngẫu nhiên liên tục dễ dàng như với các biến rời rạc bằng cách thay thế xác suất bằng mật độ xác suất. 

## Tóm tắt * Nguyên tắc khả năng tối đa cho chúng ta biết rằng mô hình phù hợp nhất cho một tập dữ liệu nhất định là mô hình tạo ra dữ liệu có xác suất cao nhất.* Thường mọi người làm việc với khả năng ghi âm thay vì nhiều lý do: ổn định số, chuyển đổi sản phẩm thành tổng (và kết quả đơn giản hóa tính toán gradient), và mối quan hệ lý thuyết với lý thuyết thông tin * Mặc dù đơn giản nhất để thúc đẩy trong cài đặt rời rạc, nó có thể được tự do khái quát hóa để cài đặt liên tục cũng như bằng cách tối đa hóa mật độ xác suất được gán cho các datapoints. 

## Bài tập 1. Giả sử rằng bạn biết rằng một biến ngẫu nhiên có mật độ $\frac{1}{\alpha}e^{-\alpha x}$ cho một số giá trị $\alpha$. Bạn có được một quan sát duy nhất từ biến ngẫu nhiên đó là số $3$. Ước tính khả năng tối đa cho $\alpha$ là gì? 2. Giả sử rằng bạn có một tập dữ liệu của các mẫu $\{x_i\}_{i=1}^N$ rút ra từ một Gaussian với trung bình không xác định, nhưng phương sai $1$. Ước tính khả năng tối đa cho nghĩa là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab:

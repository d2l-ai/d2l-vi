<!--
# Single Variable Calculus
-->

# Giải tích một biến
:label:`sec_single_variable_calculus`

<!--
In :numref:`sec_calculus`, we saw the basic elements of differential calculus. 
This section takes a deeper dive into the fundamentals of calculus and how we can understand and apply it in the context of machine learning.
-->

Trong :numref:`sec_calculus`, chúng ta đã thấy những thành phần cơ bản của giải tích vi phân.
Trong mục này chúng ta sẽ đi sâu vào kiến thức nền tảng của giải tích và cách áp dụng chúng trong trong ngữ cảnh học máy.

<!--
## Differential Calculus
-->

## Giải tích Vi phân

<!--
Differential calculus is fundamentally the study of how functions behave under small changes.  To see why this is so core to deep learning, let us consider an example.
-->

Giải tích vi phân là nhánh toán học nghiên cứu về hành vi của các hàm số dưới các biến đổi nhỏ.
Để thấy được tại sao đây lại là phần cốt lõi của học sâu, hãy cùng xem xét một ví dụ dưới đây.

<!--
Suppose that we have a deep neural network where the weights are, for convenience, concatenated into a single vector $\mathbf{w} = (w_1, \ldots, w_n)$.  
Given a training dataset, we consider the loss of our neural network on this dataset, which we will write as $\mathcal{L}(\mathbf{w})$.
-->

Giả sử chúng ta có một mạng nơ-ron sâu với các trọng số được biễu diễn bằng một vector duy nhất $\mathbf{w} = (w_1, \ldots, w_n)$.
Cho trước một tập huấn luyện, chúng ta sẽ tập trung vào giá trị mất mát $\mathcal{L}(\mathbf{w})$ của mạng nơ-ron trên tập huấn luyện đó.

<!--
This function is extraordinarily complex, encoding the performance of all possible models of the given architecture on this dataset, 
so it is nearly impossible to tell what set of weights $\mathbf{w}$ will minimize the loss. 
Thus, in practice, we often start by initializing our weights *randomly*, 
and then iteratively take small steps in the direction which makes the loss decrease as rapidly as possible.
-->

Đây là một hàm số cực kì phức tạp, biểu diễn chất lượng của tất cả các mô hình khả dĩ của một cấu trúc mạng cho trước trên tập dữ liệu này, 
nên gần như không thể chỉ ra được ngay một tập các trọng số $\mathbf{w}$ để cực tiểu hóa mất mát.
Do vậy trên thực tế, chúng ta thường bắt đầu bằng việc khởi tạo *ngẫu nhiên* các trọng số, và tiến từng bước nhỏ theo hướng mà sẽ giảm giá trị mất mát nhanh nhất có thể.

<!--
The question then becomes something that on the surface is no easier: how do we find the direction which makes the weights decrease as quickly as possible?
To dig into this, let us first examine the case with only a single weight: $L(\mathbf{w}) = L(x)$ for a single real value $x$.
-->

Vấn đề bây giờ thoạt nhìn cũng không dễ hơn bao nhiêu: làm thế nào để tìm được hướng đi sẽ giảm giá trị hàm mất mát nhanh nhất có thể?
Để trả lời câu hỏi này, trước hết ta hãy xét trường hợp chỉ có một trọng số: $L(\mathbf{w}) = L(x)$  với một số thực $x$ duy nhất.


<!--
Let us take $x$ and try to understand what happens when we change it by a small amount to $x + \epsilon$. 
If you wish to be concrete, think a number like $\epsilon = 0.0000001$.  
To help us visualize what happens, Let us graph an example function, $f(x) = \sin(x^x)$, over the $[0, 3]$.
-->

Hãy cùng tìm hiểu xem chuyện gì sẽ xảy ra khi ta lấy giá trị $x$ và thay đổi nó với một lượng rất nhỏ thành $x + \epsilon$.
Nếu bạn muốn một con số rõ ràng, hãy nghĩ về một số như $\epsilon = 0.0000001$.
Để minh họa chuyện gì sẽ diễn ra, hãy vẽ ví dụ đồ thị của hàm số $f(x) = \sin(x^x)$, trên khoảng $[0, 3]$.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()
# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch
# Plot a function in a normal range
x_big = torch.arange(0.01, 3.01, 0.01)
ys = torch.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow
# Plot a function in a normal range
x_big = tf.range(0.01, 3.01, 0.01)
ys = tf.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```


<!--
At this large scale, the function's behavior is not simple. 
However, if we reduce our range to something smaller like $[1.75,2.25]$, we see that the graph becomes much simpler.
-->

Trong một khoảng lớn thế này, cách hàm số biến đổi rất khó nắm bắt.
Tuy nhiên, nếu ta thu nhỏ khoảng xuống ví dụ như thành $[1.75,2.25]$, ta thấy đồ thị trở nên đơn giản hơn rất nhiều.


```{.python .input}
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_med = torch.arange(1.75, 2.25, 0.001)
ys = torch.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_med = tf.range(1.75, 2.25, 0.001)
ys = tf.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```


<!--
Taking this to an extreme, if we zoom into a tiny segment, the behavior becomes far simpler: it is just a straight line.
-->

Đỉnh điểm, nếu ta phóng gần vào một đoạn rất nhỏ, cách hàm số biến đổi trở nên đơn giản hơn rất nhiều: chỉ là một đường thẳng.


```{.python .input}
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
# Plot a the same function in a tiny range
x_small = torch.arange(2.0, 2.01, 0.0001)
ys = torch.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
# Plot a the same function in a tiny range
x_small = tf.range(2.0, 2.01, 0.0001)
ys = tf.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```


<!--
This is the key observation of single variable calculus: the behavior of familiar functions can be modeled by a line in a small enough range.  
This means that for most functions, it is reasonable to expect that as we shift the $x$ value of the function by a little bit, the output $f(x)$ will also be shifted by a little bit.  
The only question we need to answer is, "How large is the change in the output compared to the change in the input?  
Is it half as large?  Twice as large?"
-->

Đây là một trong những quan sát cốt lõi nhất trong giải tích: hành vi của các hàm số phổ biến có thể được mô hình hóa bằng một đường thẳng trên một khoảng đủ nhỏ.
Điều này nghĩa là với hầu hết các hàm số, chúng ta có thể trông đợi rằng khi dịch chuyển $x$ một khoảng nhỏ, $f(x)$ cũng sẽ dịch chuyển một khoảng nhỏ.
Câu hỏi duy nhất mà chúng ta cần trả lời là "Sự thay đổi của giá trị đầu ra lớn gấp bao nhiêu lần so với sự thay đổi của giá trị đầu vào? Bằng một nửa? Hay sẽ lớn gấp đôi?"

<!--
Thus, we can consider the ratio of the change in the output of a function for a small change in the input of the function.  We can write this formally as
-->

Ta cũng có thể xét nó như tỷ lệ giữa sự thay đổi của đầu ra so với sự thay đổi nhỏ trong đầu vào của một hàm số. Chúng ta có thể biễu diễn nó dưới dạng toán học là:


$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$


<!--
This is already enough to start to play around with in code.  
For instance, suppose that we know that $L(x) = x^{2} + 1701(x-4)^3$, then we can see how large this value is at the point $x = 4$ as follows.
-->

Những kiến thức trên đã đủ để chúng ta bắt đầu thực hành lập trình.
Ví dụ, giả sử $L(x) = x^{2} + 1701(x-4)^3$, ta có thể biết được độ lớn của giá trị này tại điểm $x = 4$ như sau:


```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3
# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```


<!--
Now, if we are observant, we will notice that the output of this number is suspiciously close to $8$.  
Indeed, if we decrease $\epsilon$, we will see value becomes progressively closer to $8$.  
Thus we may conclude, correctly, that the value we seek (the degree a change in the input changes the output) should be $8$ at the point $x=4$.  
The way that a mathematician encodes this fact is
-->

Nếu để ý kĩ, chúng ta sẽ nhận ra rằng kết quả của con số này xấp xỉ $8$.
Trong trường hợp ta giảm $\epsilon$ thì giá trị đầu ra ngày càng tiến gần đến $8$.
Vì vậy chúng ta có thể kết luận một cách chính xác, rằng mức độ thay đổi của đầu ra khi đầu vào thay đổi là $8$ tại điểm $x=4$.
Có thể viết dưới dạng toán học như sau:


$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$


<!--
As a bit of a historical digression: in the first few decades of neural network research, 
scientists used this algorithm (the *method of finite differences*) to evaluate how a loss function changed under small perturbation: 
just change the weights and see how the loss changed.  
This is computationally inefficient, requiring two evaluations of the loss function to see how a single change of one variable influenced the loss.  
If we tried to do this with even a paltry few thousand parameters, it would require several thousand evaluations of the network over the entire dataset!  
It was not solved until 1986 that the *backpropagation algorithm* introduced in :cite:`Rumelhart.Hinton.Williams.ea.1988` provided 
a way to calculate how *any* change of the weights together would change the loss in the same computation time as a single prediction of the network over the dataset.
-->

Một chút bàn luận ngoài lề về lịch sử: trong những thập kỷ đầu tiên của các nghiên cứu mạng nơ-ron,
các nhà khoa học đã sử dụng thuật toán này (*sai phân hữu hạn - finite differences*) để đánh giá một hàm mất mát dưới các nhiễu loạn nhỏ:
chỉ cần thay đổi trọng số và xem cách thức mà hàm mất mát thay đổi.
Đây là một cách tính toán không hiệu quả, đòi hỏi đến hai lần tính hàm mất mát để thấy được sự tác động của một thay đổi lên hàm mất mát đó.
Thậm chí nếu chúng ta sử dụng phương pháp này với vài nghìn tham số nhỏ, nó cũng sẽ đòi hỏi phải chạy mạng nơ-ron hàng nghìn lần trên toàn bộ dữ liệu.
Phải đến năm 1986 thì vấn đề này với được giải quyết khi *thuật toán lan truyền ngược* (*backpropagation algorithm*) được giới thiệu ở :cite:`Rumelhart.Hinton.Williams.ea.1988` 
đã đem đến một giải pháp để tính toán sức ảnh hưởng của những thay đổi *bất kỳ* từ các trọng số lên hàm mất mát
với thời gian tính toán chỉ bằng thời gian mô hình đưa ra dự đoán trên tập dữ liệu.

<!--
Back in our example, this value $8$ is different for different values of $x$, so it makes sense to define it as a function of $x$.  
More formally, this value dependent rate of change is referred to as the *derivative* which is written as
-->

Quay lại với ví dụ của chúng ta, giá trị $8$ này biến thiên với các trị khác nhau của $x$, vậy nên sẽ là hợp lý nếu chúng ta định nghĩa nó như là một hàm của $x$.
Một cách chính thống hơn, độ biến thiên của giá trị này được gọi là *đạo hàm* và được viết là:

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

<!--
Different texts will use different notations for the derivative. 
For instance, all of the below notations indicate the same thing:
-->

Các văn bản khác nhau sẽ sử dụng các ký hiệu khác nhau cho đạo hàm.
Chẳng hạn, tất cả các ký hiệu dưới đây đều diễn giải cùng một ý nghĩa:


$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

<!--
Most authors will pick a single notation and stick with it, however even that is not guaranteed.  
It is best to be familiar with all of these.  
We will use the notation $\frac{df}{dx}$ throughout this text, unless we want to take the derivative of a complex expression, in which case we will use $\frac{d}{dx}f$ to write expressions like
-->

Phần lớn các tác giả sẽ chọn một ký hiệu duy nhất để sử dụng xuyên suốt, tuy nhiên không phải lúc nào điều này cũng được đảm bảo.
Tốt hơn hết là chúng ta nên làm quen với tất cả các ký hiệu này.
Ký hiệu $\frac{df}{dx}$ sẽ được sử dụng trong toàn bộ cuốn sách này, trừ trường hợp chúng ta cần lấy đạo hàm của một biểu thức phức tạp, 
khi đó chúng ta sẽ sử dụng $\frac{d}{dx}f$ để biểu diễn những biểu thức như


$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$


<!--
Oftentimes, it is intuitively useful to unravel the definition of derivative :eqref:`eq_der_def` again to see how a function changes when we make a small change of $x$:
-->

Đôi khi, việc sử dụng định nghĩa của đạo hàm :eqref:`eq_der_def` để thấy một cách trực quan cách một hàm thay đổi khi $x$ thay đổi một khoảng nhỏ là rất hữu ích:


$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`


<!--
The last equation is worth explicitly calling out.  
It tells us that if you take any function and change the input by a small amount, the output would change by that small amount scaled by the derivative.
-->

Cần phải nói rõ hơn về phương trình cuối cùng.
Nó cho chúng ta biết rằng nếu ta chọn một hàm số bất kỳ và thay đổi đầu vào một lượng nhỏ, sự thay đổi của đầu ra sẽ bằng với lượng nhỏ đó nhân với đạo hàm.

<!--
In this way, we can understand the derivative as the scaling factor that tells us how large of change we get in the output from a change in the input.
-->

Bằng cách này, chúng ta có thể hiểu đạo hàm là hệ số tỷ lệ cho biết mức độ biến thiên của đầu ra khi đầu vào thay đổi.


<!--
## Rules of Calculus
-->

## Quy tắc Giải tích
:label:`sec_derivative_table`

<!--
We now turn to the task of understanding how to compute the derivative of an explicit function.  
A full formal treatment of calculus would derive everything from first principles.  
We will not indulge in this temptation here, but rather provide an understanding of the common rules encountered.
-->

Bây giờ chúng ta sẽ học cách để tính đạo hàm của một hàm cụ thể.
Dạy giải tích một cách chính quy sẽ phải chứng minh lại tất cả mọi thứ từ những định đề căn bản nhất.
Tuy nhiên chúng tôi sẽ không làm như vậy mà sẽ cung cấp các quy tắc tính đạo hàm phổ biến thường gặp.

<!--
### Common Derivatives
-->

### Các Đạo hàm phổ biến

<!--
As was seen in :numref:`sec_calculus`, when computing derivatives one can oftentimes use a series of rules to reduce the computation to a few core functions.  
We repeat them here for ease of reference.
-->

Như ở :numref:`sec_calculus`, khi tính đạo hàm ta có thể sử dụng một chuỗi các quy tắc để chia nhỏ tính toán thành các hàm cơ bản.
Chúng tôi sẽ nhắc lại chúng ở đây để bạn đọc dễ tham khảo.

<!--
* **Derivative of constants.** $\frac{d}{dx}c = 0$.
* **Derivative of linear functions.** $\frac{d}{dx}(ax) = a$.
* **Power rule.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Derivative of exponentials.** $\frac{d}{dx}e^x = e^x$.
* **Derivative of the logarithm.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.
-->

* **Đạo hàm hằng số:** $\frac{d}{dx}c = 0$.
* **Đạo hàm hàm tuyến tính:** $\frac{d}{dx}(ax) = a$.
* **Quy tắc lũy thừa:** $\frac{d}{dx}x^n = nx^{n-1}$. 
* **Đạo hàm hàm mũ cơ số tự nhiên:** $\frac{d}{dx}e^x = e^x$.
* **Đạo hàm hàm logarit cơ số tự nhiên:** $\frac{d}{dx}\log(x) = \frac{1}{x}$.


<!--
### Derivative Rules
-->

### Các Quy tắc tính Đạo hàm

<!--
If every derivative needed to be separately computed and stored in a table, differential calculus would be near impossible.  
It is a gift of mathematics that we can generalize the above derivatives and compute more complex derivatives like finding the derivative of $f(x) = \log\left(1+(x-1)^{10}\right)$.  As was mentioned in :numref:`sec_calculus`, the key to doing so is to codify what happens when we take functions and combine them in various ways, most importantly: sums, products, and compositions.
-->

Nếu mọi đạo hàm cần được tính một cách riêng biệt và lưu vào một bảng, giải tích vi phân sẽ gần như bất khả thi.
Toán học đã mang lại một món quà giúp tổng quát hóa các đạo hàm ở phần trên và giúp tính các đạo hàm phức tạp hơn như đạo hàm của $f(x) = \log\left(1+(x-1)^{10}\right)$. 
Như được đề cập trong :numref:`sec_calculus`, chìa khóa để thực hiện việc này là hệ thống hóa việc tính đạo hàm cho các hàm kết hợp theo nhiều cách: tổng, tích và hợp.

<!--
* **Sum rule.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **Product rule.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Chain rule.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.
-->

* **Quy tắc tổng.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **Quy tắc tích.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Quy tắc dây chuyền.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

<!--
Let us see how we may use :eqref:`eq_small_change` to understand these rules.  For the sum rule, consider following chain of reasoning:
-->

Cùng xem chúng ta có thể sử dụng :eqref:`eq_small_change` như thế nào để hiểu những quy tắc này. Với quy tắc tổng, xét chuỗi biến đổi sau đây:


$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$


<!--
By comparing this result with the fact that $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$, we see that $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ as desired.  
The intuition here is: when we change the input $x$, $g$ and $h$ jointly contribute to the change of the output by $\frac{dg}{dx}(x)$ and $\frac{dh}{dx}(x)$.
-->

Đồng nhất hệ số với $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$, ta có $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ như mong đợi.
Một cách trực quan, ta có thể giải thích như sau: khi thay đổi đầu vào $x$, $g$ và $h$ cùng đóng góp tới sự thay đổi của $\frac{dg}{dx}(x)$ và $\frac{dh}{dx}(x)$ ở đầu ra.


<!--
The product is more subtle, and will require a new observation about how to work with these expressions.  We will begin as before using :eqref:`eq_small_change`:
-->

Đối với quy tắc tích thì phức tạp hơn một chút và đòi hỏi một quan sát mới khi xử lý các biểu thức này.
Cũng giống như trước, ta bắt đầu bằng :eqref:`eq_small_change`:


$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$


<!--
This resembles the computation done above, and indeed we see our answer ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) sitting next to $\epsilon$, but there is the issue of that term of size $\epsilon^{2}$.  
We will refer to this as a *higher-order term*, since the power of $\epsilon^2$ is higher than the power of $\epsilon^1$.  
We will see in a later section that we will sometimes want to keep track of these, however for now observe that if $\epsilon = 0.0000001$, then $\epsilon^{2}= 0.0000000000001$, which is vastly smaller.  
As we send $\epsilon \rightarrow 0$, we may safely ignore the higher order terms.  
As a general convention in this appendix, we will use "$\approx$" to denote that the two terms are equal up to higher order terms.  
However, if we wish to be more formal we may examine the difference quotient
-->

Việc này giống với những tính toán trước đây, và dễ thấy kết quả của ta ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) 
là số hạng được nhân với $\epsilon$, nhưng vấn đề là ở số hạng nhân với giá trị $\epsilon^{2}$.
Chúng ta sẽ gọi số hạng này là *số hạng bậc cao*, bởi số mũ của $\epsilon^2$ cao hơn số mũ của $\epsilon^1$.
Về sau ta sẽ thấy rằng thi thoảng ta muốn giữ các số hạng này, tuy nhiên hiện tại có thể thấy rằng nếu $\epsilon = 0.0000001$, thì $\epsilon^{2}= 0.0000000000001$, là một số nhỏ hơn rất nhiều.
Khi đưa $\epsilon \rightarrow 0$, ta có thể bỏ qua các số hạng bậc cao.
Ta sẽ quy ước sử dụng "$\approx$" để ký hiệu rằng hai số hạng bằng nhau với sai số là các thành phần bậc cao.
Nếu muốn biểu diễn chính quy hơn, ta có thể xét phương trình


$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$


<!--
and see that as we send $\epsilon \rightarrow 0$, the right hand term goes to zero as well.
-->

và thấy rằng khi $\epsilon \rightarrow 0$, số hạng bên phải cũng tiến về không.

<!--
Finally, with the chain rule, we can again progress as before using :eqref:`eq_small_change` and see that
-->

Cuối cùng, với quy tắc dây chuyền, ta vẫn có thể tiếp tục khai triển sử dụng :eqref:`eq_small_change` và thấy rằng:

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x).
\end{aligned}
$$


<!--
where in the second line we view the function $g$ as having its input ($h(x)$) shifted by the tiny quantity $\epsilon \frac{dh}{dx}(x)$.
-->

Chú ý là ở dòng thứ hai trong chuỗi khai triển trên, chúng ta đã xem đối số $h(x)$ của hàm $g$ như là bị dịch đi bởi một lượng rất nhỏ $\epsilon \frac{dh}{dx}(x)$.


<!--
These rule provide us with a flexible set of tools to compute essentially any expression desired.  For instance,
-->

Các quy tắc này cung cấp cho chúng ta một tập hợp các công cụ linh hoạt để tính toán đạo hàm của hầu như bất kỳ biểu thức nào ta muốn.
Chẳng hạn như trong ví dụ sau:


$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$


<!--
Where each line has used the following rules:
-->

Mỗi dòng của ví dụ này đã sử dụng các quy tắc sau:

<!--
1. The chain rule and derivative of logarithm.
2. The sum rule.
3. The derivative of constants, chain rule, and power rule.
4. The sum rule, derivative of linear functions, derivative of constants.
-->

1. Quy tắc dây chuyền và công thức đạo hàm của hàm logarit.
2. Quy tắc đạo hàm của tổng.
3. Đạo hàm của hằng số, quy tắc dây chuyền, và quy tắc đạo hàm của lũy thừa.
4. Quy tắc đạo hàm của tổng, đạo hàm của hàm tuyến tính, đạo hàm của hằng số.


<!--
Two things should be clear after doing this example:
-->

Từ ví dụ trên, chúng ta có thể dễ dàng rút ra được hai điều:

<!--
1. Any function we can write down using sums, products, constants, powers, exponentials, and logarithms can have its derivate computed mechanically by following these rules.
2. Having a human follow these rules can be tedious and error prone!
-->

1. Chúng ta có thể lấy đạo hàm của bất kỳ hàm số nào mà có thể diễn tả được bằng tổng, tích, hằng số, lũy thừa, hàm mũ, và hàm logarit bằng cách sử dụng những quy tắc trên một cách máy móc.
2. Quá trình dùng những quy tắc này để tính đạo hàm bằng tay có thể sẽ rất tẻ nhạt và dễ mắc lỗi.

<!--
Thankfully, these two facts together hint towards a way forward: this is a perfect candidate for mechanization!
Indeed backpropagation, which we will revisit later in this section, is exactly that.
-->

Rất may là hai điều này gộp chung lại gợi ý cho chúng ta một hướng phát triển: đây chính là cơ hội lý tưởng để tự động hóa bằng máy tính! 
Thật vậy, kỹ thuật lan truyền ngược, mà chúng ta sẽ gặp lại sau ở mục này, là một cách hiện thực hóa ý tưởng này.


<!--
### Linear Approximation
-->

### Xấp xỉ Tuyến tính

<!--
When working with derivatives, it is often useful to geometrically interpret the approximation used above.  In particular, note that the equation
-->

Thông thường khi làm việc với đạo hàm, sẽ rất hữu ích nếu chúng ta có thể diễn tả sự xấp xỉ ở trên theo phương diện hình học.
Nói một cách cụ thể, phương trình này


$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$


<!--
approximates the value of $f$ by a line which passes through the point $(x, f(x))$ and has slope $\frac{df}{dx}(x)$.  
In this way we say that the derivative gives a linear approximation to the function $f$, as illustrated below:
-->

xấp xỉ giá trị của $f$ bằng một đường thẳng đi qua điểm $(x, f(x))$ và có độ dốc $\frac{df}{dx}(x)$. 
Với cách hiểu này, ta nói rằng đạo hàm cho ta một xấp xỉ tuyến tính của hàm số $f$, như minh họa dưới đây:


```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]
# Compute some linear approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))
d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]
# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)))
d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]
# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)))
d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```


<!--
### Higher Order Derivatives
-->

### Đạo hàm Cấp cao

<!--
Let us now do something that may on the surface seem strange.  
Take a function $f$ and compute the derivative $\frac{df}{dx}$.  
This gives us the rate of change of $f$ at any point.
-->

Bây giờ, hãy cùng làm một việc mà nhìn sơ qua thì có vẻ kỳ quặc.
Bắt đầu bằng việc lấy một hàm số $f$ và tính đạo hàm $\frac{df}{dx}$.
Nó sẽ cho chúng ta tốc độ thay đổi của $f$ tại bất cứ điểm nào.

<!--
However, the derivative, $\frac{df}{dx}$, can be viewed as a function itself, so nothing stops us from computing
 the derivative of $\frac{df}{dx}$ to get $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$.  
We will call this the second derivative of $f$.  
This function is the rate of change of the rate of change of $f$, or in other words, how the rate of change is changing. 
We may apply the derivative any number of times to obtain what is called the $n$-th derivative. 
To keep the notation clean, we will denote the $n$-th derivative as
-->

Tuy nhiên, vì bản thân đạo hàm $\frac{df}{dx}$ cũng là một hàm số, không có gì ngăn cản chúng ta tiếp tục tính đạo hàm của 
$\frac{df}{dx}$ để có $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$.
Chúng ta sẽ gọi đây là đạo hàm cấp hai của $f$.
Hàm số này là tốc độ thay đổi của tốc độ thay đổi của $f$, hay nói cách khác, nó thể hiện tốc độ thay đổi của $f$ đang thay đổi như thế nào.
Chúng ta có thể tiếp tục lấy đạo hàm như vậy thêm nhiều lần nữa để có được thứ gọi là đạo hàm cấp $n$.
Để ký hiệu được gọn gàng, chúng ta sẽ biểu thị đạo hàm cấp $n$ như sau:


$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


<!--
Let us try to understand *why* this is a useful notion.
Below, we visualize $f^{(2)}(x)$, $f^{(1)}(x)$, and $f(x)$.
-->

Hãy tìm hiểu xem *tại sao* đây lại là một khái niệm hữu ích.
Các hàm số $f^{(2)}(x)$, $f^{(1)}(x)$, và $f(x)$ được biểu diễn trong các đồ thị dưới đây.


<!--
First, consider the case that the second derivative $f^{(2)}(x)$ is a positive constant.  
This means that the slope of the first derivative is positive.  
As a result, the first derivative $f^{(1)}(x)$ may start out negative, becomes zero at a point, and then becomes positive in the end. 
This tells us the slope of our original function $f$ and therefore, the function $f$ itself decreases, flattens out, then increases.  
In other words, the function $f$ curves up, and has a single minimum as is shown in :numref:`fig_positive-second`.
-->

Đầu tiên, xét trường hợp đạo hàm bậc hai $f^{(2)}(x)$ là một hằng số dương.
Điều này nghĩa là độ dốc của đạo hàm bậc nhất là dương.
Hệ quả là, đạo hàm bậc nhất $f^{(1)}(x)$ có thể khởi đầu ở âm, bằng không tại một điểm nào đó, rồi cuối cùng tăng lên dương.
Điều này cho chúng ta biết độ dốc của hàm $f$ ban đầu và do đó, giá trị hàm $f$ sẽ giảm xuống đến điểm nào đó rồi tăng lên.
Nói cách khác, đồ thị hàm $f$ là đường cong đi lên, có một cực tiểu như trong :numref:`fig_positive-second`.


<!--
![If we assume the second derivative is a positive constant, then the fist derivative in increasing, which implies the function itself has a minimum.](../img/posSecDer.svg)
-->

![Nếu giả định rằng đạo hàm bậc hai là một hằng số dương, thì đạo hàm bậc nhất đồng biến, nghĩa là bản thân hàm đó có một cực tiểu.](../img/posSecDer.svg)
:label:`fig_positive-second`


<!--
Second, if the second derivative is a negative constant, that means that the first derivative is decreasing.  
This implies the first derivative may start out positive, becomes zero at a point, and then becomes negative. 
Hence, the function $f$ itself increases, flattens out, then decreases.  
In other words, the function $f$ curves down, and has a single maximum as is shown in :numref:`fig_negative-second`.
-->

Thứ hai là, nếu đạo hàm bậc hai là một hằng số âm, nghĩa là đạo hàm bậc nhất nghịch biến.
Vậy tức là đạo hàm bậc nhất có thể khời đầu là dương, bằng không ở điểm nào đó, rồi giảm xuống âm.
Do vậy, giá trị hàm $f$ tăng lên đến điểm nào đó rồi giảm xuống.
Nói cách khác, đồ thị hàm $f$ là đường cong đi xuống, có một cực đại như trong :numref:`fig_negative-second`.

<!--
![If we assume the second derivative is a negative constant, then the fist derivative in decreasing, which implies the function itself has a maximum.](../img/negSecDer.svg)
-->

![Nếu giả định đạo hàm bậc hai là một hằng số âm, thì đạo hàm bậc nhất nghịch biến, nghĩa là hàm số có một cực đại.](../img/negSecDer.svg)
:label:`fig_negative-second`


<!--
Third, if the second derivative is a always zero, then the first derivative will never change---it is constant!  
This means that $f$ increases (or decreases) at a fixed rate, and $f$ is itself a straight line  as is shown in :numref:`fig_zero-second`.
-->

Thứ ba là, nếu đạo hàm bậc hai luôn luôn bằng không, thì đạo hàm bậc nhất là hằng số! 
Nghĩa là hàm $f$ tăng (hoặc giảm) với tốc độ cố định, và đồ thị $f$ là một đường thẳng giống như trong :numref:`fig_zero-second`.

<!--
![If we assume the second derivative is zero, then the fist derivative is constant, which implies the function itself is a straight line.](../img/zeroSecDer.svg)
-->

![Nếu ta giả định đạo hàm bậc hai bằng không, thì đạo hàm bậc nhất là hằng số, nên đồ thị hàm này là một đường thẳng.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

<!--
To summarize, the second derivative can be interpreted as describing the way that the function $f$ curves.  
A positive second derivative leads to a upwards curve, while a negative second derivative means that $f$ curves downwards, and a zero second derivative means that $f$ does not curve at all.
-->

Tóm lại, đạo hàm bậc hai có thể được hiểu như một cách miêu tả đường cong của đồ thị hàm $f$.
Đạo hàm bậc hai dương thì đồ thị cong lên, đạo hàm bậc hai âm thì hàm $f$ cong xuống, và nếu bằng không thì $f$ là một đường thẳng.


<!--
Let us take this one step further. Consider the function $g(x) = ax^{2}+ bx + c$.  We can then compute that
-->

Hãy thử tiến xa hơn một bước.
Xét hàm $g(x) = ax^{2}+ bx + c$. 
Ta có thể tính được


$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$


<!--
If we have some original function $f(x)$ in mind, we may compute the first two derivatives and find the values for $a, b$, and $c$ that make them match this computation.  
Similarly to the previous section where we saw that the first derivative gave the best approximation with a straight line, 
this construction provides the best approximation by a quadratic.  Let us visualize this for $f(x) = \sin(x)$.
-->

Nếu đã có sẵn một hàm $f(x)$, ta có thể tính đạo hàm cấp một và cấp hai của nó để tìm các giá trị $a, b$, và $c$ thỏa mãn hệ phương trình này.
Cũng giống như ở mục trước ta đã thấy đạo hàm bậc một cho ra xấp xỉ tốt nhất bằng một đường thẳng, đạo hàm bậc hai cung cấp một xấp xỉ tốt nhất bằng một parabol.
Hãy minh họa với trường hợp $f(x) = \sin(x)$.


```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]
# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)
d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab pytorch
# Compute sin
xs = torch.arange(-torch.pi, torch.pi, 0.01)
plots = [torch.sin(xs)]
# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(torch.sin(torch.tensor(x0)) + (xs - x0) * 
                 torch.cos(torch.tensor(x0)) - (xs - x0)**2 *
                 torch.sin(torch.tensor(x0)) / 2)
d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

```{.python .input}
#@tab tensorflow
# Compute sin
xs = tf.range(-tf.pi, tf.pi, 0.01)
plots = [tf.sin(xs)]
# Compute some quadratic approximations. Use d(sin(x)) / dx = cos(x)
for x0 in [-1.5, 0.0, 2.0]:
    plots.append(tf.sin(tf.constant(x0)) + (xs - x0) * 
                 tf.cos(tf.constant(x0)) - (xs - x0)**2 *
                 tf.sin(tf.constant(x0)) / 2)
d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```


<!--
We will extend this idea to the idea of a *Taylor series* in the next section.
-->

Ta sẽ mở rộng ý tưởng này thành ý tưởng của *chuỗi Taylor* trong mục tiếp theo.


<!--
### Taylor Series
-->

### Chuỗi Taylor


<!--
The *Taylor series* provides a method to approximate the function $f(x)$ if we are given values for 
the first $n$ derivatives at a point $x_0$, i.e., $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$.
The idea will be to find a degree $n$ polynomial that matches all the given derivatives at $x_0$.
-->

*Chuỗi Taylor* cung cấp một phương pháp để xấp xỉ phương trình $f(x)$ nếu ta đã biết trước giá trị của $n$ cấp đạo hàm đầu tiên tại điểm $x_0$:
$\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$.
Ý tưởng là tìm một đa thức bậc $n$ có các đạo hàm tại $x_0$ khớp với các đạo hàm đã biết.

<!--
We saw the case of $n=2$ in the previous section and a little algebra shows this is
-->

Ta đã thấy với trường hợp $n=2$ ở chương trước và với một chút biến đổi đại số, ta có được


$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$



<!--
As we can see above, the denominator of $2$ is there to cancel out the $2$ we get when we take two derivatives of $x^2$, while the other terms are all zero.  
Same logic applies for the first derivative and the value itself.
-->

Như ta đã thấy ở trên, mẫu số $2$ là để rút gọn thừa số $2$ khi lấy đạo hàm bậc hai của $x^2$, các đạo hàm bậc cao hơn đều bằng không.
Cùng một cách lập luận cũng được áp dụng cho đạo hàm bậc một và phần giá trị $f(x_0)$.

<!--
If we push the logic further to $n=3$, we will conclude that
-->

Nếu ta mở rộng cách lập luận này cho trường hợp $n=3$, ta sẽ kết luận được


$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$


<!--
where the $6 = 3 \times 2 = 3!$ comes from the constant we get in front if we take three derivatives of $x^3$.
-->

với $6 = 3 \times 2 = 3!$ đến từ phần hằng số ta có được khi lấy đạo hàm bậc 3 của $x^3$.


<!--
Furthermore, we can get a degree $n$ polynomial by
-->

Hơn nữa, ta có thể lấy một đa thức bậc $n$ bằng cách


$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

<!--
where the notation
-->

với quy ước


$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$


<!--
Indeed, $P_n(x)$ can be viewed as the best $n$-th degree polynomial approximation to our function $f(x)$.
-->

Quả thật, $P_n(x)$ có thể được xem là đa thức bậc $n$ xấp xỉ tốt nhất của hàm $f(x)$.

<!--
While we are not going to dive all the way into the error of the above approximations, it is worth mentioning the the infinite limit. 
In this case, for well behaved functions (known as real analytic functions) like $\cos(x)$ or $e^{x}$, we can write out the infinite number of terms and approximate the exactly same function
-->

Dù ta sẽ không tìm hiểu kỹ sai số của xấp xỉ này, ta cũng nên nhắc tới giới hạn vô cùng.
Trong trường hợp này, các hàm khả vi vô hạn lần như $\cos(x)$ hoặc $e^{x}$ có thể được biểu diễn xấp xỉ bằng vô số các số hạng.


$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$


<!--
Take $f(x) = e^{x}$ as am example. Since $e^{x}$ is its own derivative, we know that $f^{(n)}(x) = e^{x}$. 
Therefore, $e^{x}$ can be reconstructed by taking the Taylor series at $x_0 = 0$, i.e.,
-->

Lấy hàm $f(x) = e^{x}$ làm ví dụ.
Vì $e^{x}$ là đạo hàm của chính nó, ta có $f^{(n)}(x) = e^{x}$.
Do đó, hàm $e^{x}$ có thể được tái tạo bằng cách tính chuỗi Taylor tại $x_0 = 0$:


$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$


<!--
Let us see how this works in code and observe how increasing the degree of the Taylor approximation brings us closer to the desired function $e^x$.
-->

Hãy cùng tìm hiểu cách lập trình và quan sát xem việc tăng bậc của xấp xỉ Taylor đưa ta đến gần hơn với hàm mong muốn $e^x$ như thế nào.


```{.python .input}
# Compute the exponential function
xs = np.arange(0, 3, 0.01)
ys = np.exp(xs)
# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120
d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab pytorch
# Compute the exponential function
xs = torch.arange(0, 3, 0.01)
ys = torch.exp(xs)
# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120
d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```

```{.python .input}
#@tab tensorflow
# Compute the exponential function
xs = tf.range(0, 3, 0.01)
ys = tf.exp(xs)
# Compute a few Taylor series approximations
P1 = 1 + xs
P2 = 1 + xs + xs**2 / 2
P5 = 1 + xs + xs**2 / 2 + xs**3 / 6 + xs**4 / 24 + xs**5 / 120
d2l.plot(xs, [ys, P1, P2, P5], 'x', 'f(x)', legend=[
    "Exponential", "Degree 1 Taylor Series", "Degree 2 Taylor Series",
    "Degree 5 Taylor Series"])
```


<!--
Taylor series have two primary applications:
-->

Chuỗi Taylor có hai ứng dụng chính: 

<!--
1. *Theoretical applications*: Often when we try to understand a too complex function, using Taylor series enables us to turn it into a polynomial that we can work with directly.
-->

1. *Ứng dụng lý thuyết*:
Khi muốn tìm hiểu một hàm số quá phức tạp, ta thường dùng chuỗi Taylor để biến nó thành một đa thức để có thể làm việc trực tiếp. 

<!--
2. *Numerical applications*: Some functions like $e^{x}$ or $\cos(x)$ are  difficult for machines to compute.  
They can store tables of values at a fixed precision (and this is often done), but it still leaves open questions like "What is the 1000-th digit of $\cos(1)$?"  
Taylor series are often helpful to answer such questions.
-->

2. *Ứng dụng số học*:
Việc tính toán một số hàm như $e^x$ hoặc $\cos(x)$ không đơn giản đối với máy tính.
Chúng có thể lưu trữ một bảng giá trị với độ chính xác nhất định (và thường thì chúng làm vậy),
nhưng việc đó vẫn không giải quyết được những câu hỏi như "Chữ số thứ 1000 của $\cos(1)$ là gì?".
Chuỗi Taylor thường có ích cho việc trả lời các câu hỏi như vậy.


## Tóm tắt

<!--
* Derivatives can be used to express how functions change when we change the input by a small amount.
* Elementary derivatives can be combined using derivative rules to create arbitrarily complex derivatives.
* Derivatives can be iterated to get second or higher order derivatives.  Each increase in order provides more fine grained information on the behavior of the function.
* Using information in the derivatives of a single data example, we can approximate well behaved functions by polynomials obtained from the Taylor series.
-->

* Đạo hàm có thể được sử dụng để biểu diễn mức độ thay đổi của hàm số khi đầu vào thay đổi một lượng nhỏ.
* Các phép lấy đạo hàm cơ bản có thể kết hợp với nhau theo các quy tắc đạo hàm để tính những đạo hàm phức tạp tùy ý.
* Đạo hàm có thể được tính nhiều lần để lấy đạo hàm cấp hai hoặc các cấp cao hơn. Mỗi lần tăng cấp đạo hàm cho ta thông tin chi tiết hơn về hành vi của hàm số.
* Bằng việc sử dụng thông tin từ đạo hàm của một điểm dữ liệu, ta có thể xấp xỉ các hàm khả vi vô hạn lần bằng các đa thức lấy từ chuỗi Taylor.


## Bài tập

<!--
1. What is the derivative of $x^3-4x+1$?
2. What is the derivative of $\log(\frac{1}{x})$?
3. True or False: If $f'(x) = 0$ then $f$ has a maximum or minimum at $x$?
4. Where is the minimum of $f(x) = x\log(x)$ for $x\ge0$ (where we assume that $f$ takes the limiting value of $0$ at $f(0)$)?
-->

1. Đạo hàm của $x^3-4x+1$ là gì?
2. Đạo hàm của $\log(\frac{1}{x})$ là gì? 
3. Đúng hay Sai: Nếu $f'(x) = 0$ thì $f$ có cực đại hoặc cực tiểu tại $x$?
4. Cực tiểu của $f(x) = x\log(x)$ với $x\ge0$ ở đâu (ở đây ta giả sử rằng $f$ có giới hạn bằng $0$ tại $f(0)$)?


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/412), [Pytorch](https://discuss.d2l.ai/t/1088), [Tensorflow](https://discuss.d2l.ai/t/1089)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Vũ Hữu Tiệp
* Nguyễn Lê Quang Nhật
* Đoàn Võ Duy Thanh
* Tạ H. Duy Nguyên
* Mai Sơn Hải
* Phạm Minh Đức
* Nguyễn Văn Tâm
* Nguyễn Văn Cường

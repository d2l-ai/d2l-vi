# Tính toán biến đơn
:label:`sec_single_variable_calculus`

Năm :numref:`sec_calculus`, chúng ta đã thấy các yếu tố cơ bản của phép tính vi phân. Phần này sẽ tìm hiểu sâu hơn về các nguyên tắc cơ bản của giải tích và cách chúng ta có thể hiểu và áp dụng nó trong bối cảnh học máy. 

## Vi phân Giải tích vi phân về cơ bản là nghiên cứu về cách các chức năng hoạt động dưới những thay đổi nhỏ. Để xem lý do tại sao điều này là cốt lõi để học sâu, chúng ta hãy xem xét một ví dụ. 

Giả sử rằng chúng ta có một mạng thần kinh sâu, nơi các trọng lượng, để thuận tiện, được nối thành một vector $\mathbf{w} = (w_1, \ldots, w_n)$ duy nhất. Với một tập dữ liệu đào tạo, chúng tôi xem xét việc mất mạng thần kinh của chúng tôi trên tập dữ liệu này, mà chúng tôi sẽ viết là $\mathcal{L}(\mathbf{w})$.   

Chức năng này cực kỳ phức tạp, mã hóa hiệu suất của tất cả các mô hình có thể có của kiến trúc đã cho trên tập dữ liệu này, vì vậy gần như không thể biết bộ trọng lượng $\mathbf{w}$ sẽ giảm thiểu tổn thất. Do đó, trong thực tế, chúng ta thường bắt đầu bằng cách khởi tạo trọng lượng* ngẫu nhiên*, và sau đó lặp đi lặp lại các bước nhỏ theo hướng làm cho tổn thất giảm càng nhanh càng tốt. 

Câu hỏi sau đó trở thành một cái gì đó mà trên bề mặt không dễ dàng hơn: làm thế nào để chúng ta tìm thấy hướng làm cho trọng lượng giảm càng nhanh càng tốt? Để đào sâu vào điều này, trước tiên chúng ta hãy kiểm tra trường hợp chỉ với một trọng lượng duy nhất: $L(\mathbf{w}) = L(x)$ cho một giá trị thực duy nhất $x$.  

Chúng ta hãy lấy $x$ và cố gắng hiểu những gì sẽ xảy ra khi chúng ta thay đổi nó bằng một lượng nhỏ thành $x + \epsilon$. Nếu bạn muốn cụ thể, hãy nghĩ một con số như $\epsilon = 0.0000001$. Để giúp chúng ta hình dung những gì xảy ra, chúng ta hãy biểu đồ một hàm ví dụ, $f(x) = \sin(x^x)$, trên $[0, 3]$.

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

Ở quy mô lớn này, hành vi của hàm không đơn giản. Tuy nhiên, nếu chúng ta giảm phạm vi của mình xuống một cái gì đó nhỏ hơn như $[1.75,2.25]$, chúng ta thấy rằng biểu đồ trở nên đơn giản hơn nhiều.

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

Đưa điều này đến cực đoan, nếu chúng ta phóng to thành một đoạn nhỏ, hành vi trở nên đơn giản hơn nhiều: nó chỉ là một đường thẳng.

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

Đây là quan sát chính của phép tính biến đơn lẻ: hành vi của các hàm quen thuộc có thể được mô hình hóa bởi một dòng trong một phạm vi đủ nhỏ. Điều này có nghĩa là đối với hầu hết các chức năng, thật hợp lý khi mong đợi rằng khi chúng ta thay đổi giá trị $x$ của hàm một chút, đầu ra $f(x)$ cũng sẽ được dịch chuyển một chút. Câu hỏi duy nhất chúng ta cần trả lời là, “Sự thay đổi trong đầu ra lớn bao nhiêu so với sự thay đổi trong đầu vào? Nó có lớn một nửa không? Gấp đôi lớn?” 

Do đó, chúng ta có thể xem xét tỷ lệ thay đổi trong đầu ra của một hàm cho một sự thay đổi nhỏ trong đầu vào của hàm. Chúng tôi có thể viết điều này chính thức như 

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

Điều này đã đủ để bắt đầu chơi xung quanh với mã. Ví dụ, giả sử rằng chúng ta biết rằng $L(x) = x^{2} + 1701(x-4)^3$, sau đó chúng ta có thể thấy giá trị này lớn như thế nào tại điểm $x = 4$ như sau.

```{.python .input}
#@tab all
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print(f'epsilon = {epsilon:.5f} -> {(L(4+epsilon) - L(4)) / epsilon:.5f}')
```

Bây giờ, nếu chúng ta quan sát, chúng ta sẽ nhận thấy rằng đầu ra của con số này là đáng ngờ gần với $8$. Thật vậy, nếu chúng ta giảm $\epsilon$, chúng ta sẽ thấy giá trị trở nên dần dần gần hơn với $8$. Vì vậy, chúng ta có thể kết luận, một cách chính xác, rằng giá trị chúng ta tìm kiếm (mức độ thay đổi trong đầu vào thay đổi đầu ra) nên là $8$ tại điểm $x=4$. Cách mà một nhà toán học mã hóa thực tế này là 

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

Như một chút của một cuộc di truyền lịch sử: trong vài thập kỷ đầu tiên của nghiên cứu mạng thần kinh, các nhà khoa học đã sử dụng thuật toán này (phương pháp * của sự khác biệt hữu hạn *) để đánh giá làm thế nào một chức năng mất thay đổi theo nhiễu loạn nhỏ: chỉ cần thay đổi trọng lượng và xem mất thay đổi như thế nào. Đây là tính toán không hiệu quả, đòi hỏi hai đánh giá của hàm mất để xem một thay đổi duy nhất của một biến ảnh hưởng đến tổn thất như thế nào. Nếu chúng tôi cố gắng làm điều này với thậm chí một vài nghìn tham số nhẹ, nó sẽ yêu cầu vài nghìn đánh giá của mạng trên toàn bộ dữ liệu! Nó không được giải quyết cho đến năm 1986 rằng thuật toán *backpropagation* được giới thiệu trong :cite:`Rumelhart.Hinton.Williams.ea.1988` cung cấp một cách để tính toán cách *any* thay đổi trọng số cùng nhau sẽ thay đổi tổn thất trong cùng một thời gian tính toán như một dự đoán duy nhất của mạng qua tập dữ liệu. 

Trở lại trong ví dụ của chúng tôi, giá trị này $8$ là khác nhau cho các giá trị khác nhau của $x$, vì vậy nó có ý nghĩa để xác định nó như một hàm của $x$. Chính thức hơn, tỷ lệ thay đổi phụ thuộc vào giá trị này được gọi là *phái sinh* được viết là 

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

Các văn bản khác nhau sẽ sử dụng các ký hiệu khác nhau cho đạo hàm. Ví dụ, tất cả các ký hiệu dưới đây chỉ ra điều tương tự: 

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

Hầu hết các tác giả sẽ chọn một ký hiệu duy nhất và gắn bó với nó, tuy nhiên ngay cả điều đó không được đảm bảo. Tốt nhất là làm quen với tất cả những điều này. Chúng ta sẽ sử dụng ký hiệu $\frac{df}{dx}$ trong suốt văn bản này, trừ khi chúng ta muốn lấy đạo hàm của một biểu thức phức tạp, trong trường hợp đó chúng ta sẽ sử dụng $\frac{d}{dx}f$ để viết các biểu thức như $$\ frac {d} {dx}\ left [x^4+\ cos\ left (\ frac {x^2+1} {2x-1}\ right)\ right] . $$ 

Thông thường, nó là trực giác hữu ích để làm sáng tỏ định nghĩa của đạo hàm :eqref:`eq_der_def` một lần nữa để xem một hàm thay đổi như thế nào khi chúng ta thực hiện một thay đổi nhỏ $x$: 

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

Phương trình cuối cùng đáng để gọi ra một cách rõ ràng. Nó cho chúng ta biết rằng nếu bạn lấy bất kỳ hàm nào và thay đổi đầu vào bằng một lượng nhỏ, đầu ra sẽ thay đổi theo số lượng nhỏ đó được thu nhỏ theo đạo hàm. 

Bằng cách này, chúng ta có thể hiểu đạo hàm là hệ số mở rộng cho chúng ta biết thay đổi lớn như thế nào chúng ta nhận được trong đầu ra từ một sự thay đổi trong đầu vào. 

## Quy tắc của Calculus
:label:`sec_derivative_table`

Bây giờ chúng ta chuyển sang nhiệm vụ hiểu làm thế nào để tính toán đạo hàm của một hàm rõ ràng. Một điều trị chính thức đầy đủ của giải tích sẽ lấy được tất cả mọi thứ từ các nguyên tắc đầu tiên. Chúng tôi sẽ không thưởng thức sự cám dỗ này ở đây, mà là cung cấp một sự hiểu biết về các quy tắc chung gặp phải. 

### Các dẫn xuất phổ biến Như đã thấy trong :numref:`sec_calculus`, khi các dẫn xuất tính toán người ta thường có thể sử dụng một loạt các quy tắc để giảm tính toán cho một vài chức năng cốt lõi. Chúng tôi lặp lại chúng ở đây để dễ tham khảo. 

* ** Derivative của constants.** $\frac{d}{dx}c = 0$.
* ** Derivative của các chức năng tuyến tính** $\frac{d}{dx}(ax) = a$.
* ** Quy tắc điện.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Derivative của số mũ. ** $\frac{d}{dx}e^x = e^x$.
* ** Derivative của logarit** $\frac{d}{dx}\log(x) = \frac{1}{x}$.

### Quy tắc phái sinh Nếu mọi đạo hàm cần được tính toán riêng và lưu trữ trong một bảng, phép tính vi phân sẽ gần như không thể. Đó là một món quà của toán học mà chúng ta có thể khái quát hóa các dẫn xuất trên và tính toán các dẫn xuất phức tạp hơn như tìm ra đạo hàm của $f(x) = \log\left(1+(x-1)^{10}\right)$. Như đã đề cập trong :numref:`sec_calculus`, chìa khóa để làm như vậy là mã hóa những gì xảy ra khi chúng ta thực hiện các chức năng và kết hợp chúng theo nhiều cách khác nhau, quan trọng nhất là: tổng, sản phẩm và thành phần. 

* **Sum rule.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* ** Quy tắc sản phẩm.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* ** Quy tắc dây chuyển.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.

Hãy để chúng tôi xem cách chúng tôi có thể sử dụng :eqref:`eq_small_change` để hiểu các quy tắc này. Đối với quy tắc tổng, hãy xem xét chuỗi lý luận sau: 

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon) + h(x+\epsilon) \\
& \approx g(x) + \epsilon \frac{dg}{dx}(x) + h(x) + \epsilon \frac{dh}{dx}(x) \\
& = g(x) + h(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right) \\
& = f(x) + \epsilon\left(\frac{dg}{dx}(x) + \frac{dh}{dx}(x)\right).
\end{aligned}
$$

Bằng cách so sánh kết quả này với thực tế là $f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x)$, chúng ta thấy rằng $\frac{df}{dx}(x) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$ như mong muốn. Trực giác ở đây là: khi chúng ta thay đổi đầu vào $x$, $g$ và $h$ cùng góp phần vào sự thay đổi đầu ra của $\frac{dg}{dx}(x)$ và $\frac{dh}{dx}(x)$. 

Sản phẩm tinh tế hơn và sẽ yêu cầu một quan sát mới về cách làm việc với các biểu thức này. Chúng tôi sẽ bắt đầu như trước khi sử dụng :eqref:`eq_small_change`: 

$$
\begin{aligned}
f(x+\epsilon) & = g(x+\epsilon)\cdot h(x+\epsilon) \\
& \approx \left(g(x) + \epsilon \frac{dg}{dx}(x)\right)\cdot\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& = g(x)\cdot h(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x) \\
& = f(x) + \epsilon\left(g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)\right) + \epsilon^2\frac{dg}{dx}(x)\frac{dh}{dx}(x). \\
\end{aligned}
$$

Điều này giống như tính toán được thực hiện ở trên, và thực sự chúng ta thấy câu trả lời của chúng tôi ($\frac{df}{dx}(x) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$) ngồi bên cạnh $\epsilon$, nhưng có vấn đề về thuật ngữ đó có kích thước $\epsilon^{2}$. Chúng tôi sẽ đề cập đến điều này là một điều khoản * bậc cao hơn*, vì sức mạnh của $\epsilon^2$ cao hơn sức mạnh của $\epsilon^1$. Chúng ta sẽ thấy trong một phần sau mà đôi khi chúng ta sẽ muốn theo dõi những điều này, tuy nhiên bây giờ quan sát thấy rằng nếu $\epsilon = 0.0000001$, thì $\epsilon^{2}= 0.0000000000001$, nhỏ hơn rất nhiều. Khi chúng tôi gửi $\epsilon \rightarrow 0$, chúng tôi có thể bỏ qua một cách an toàn các điều khoản đặt hàng cao hơn. Như một quy ước chung trong phụ lục này, chúng ta sẽ sử dụng “$\approx$" để biểu thị rằng hai thuật ngữ này bằng với các điều khoản bậc cao hơn. Tuy nhiên, nếu chúng ta muốn chính thức hơn, chúng ta có thể kiểm tra sự khác biệt thương 

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

và thấy rằng khi chúng tôi gửi $\epsilon \rightarrow 0$, thuật ngữ tay phải cũng đi đến không. 

Cuối cùng, với quy tắc chuỗi, chúng ta có thể tiến bộ một lần nữa như trước khi sử dụng :eqref:`eq_small_change` và thấy rằng 

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x),
\end{aligned}
$$

trong đó trong dòng thứ hai, chúng tôi xem hàm $g$ là có đầu vào của nó ($h(x)$) được dịch chuyển bởi số lượng nhỏ $\epsilon \frac{dh}{dx}(x)$. 

Quy tắc này cung cấp cho chúng ta một bộ công cụ linh hoạt để tính toán về cơ bản bất kỳ biểu thức nào mong muốn. For instance ví dụ, 

$$
\begin{aligned}
\frac{d}{dx}\left[\log\left(1+(x-1)^{10}\right)\right] & = \left(1+(x-1)^{10}\right)^{-1}\frac{d}{dx}\left[1+(x-1)^{10}\right]\\
& = \left(1+(x-1)^{10}\right)^{-1}\left(\frac{d}{dx}[1] + \frac{d}{dx}[(x-1)^{10}]\right) \\
& = \left(1+(x-1)^{10}\right)^{-1}\left(0 + 10(x-1)^9\frac{d}{dx}[x-1]\right) \\
& = 10\left(1+(x-1)^{10}\right)^{-1}(x-1)^9 \\
& = \frac{10(x-1)^9}{1+(x-1)^{10}}.
\end{aligned}
$$

Trong trường hợp mỗi dòng đã sử dụng các quy tắc sau: 

1. Quy tắc chuỗi và đạo hàm của logarit.
2. Quy tắc tổng.
3. Đạo hàm của hằng số, quy tắc chuỗi, và quy tắc quyền lực.
4. Quy tắc tổng, đạo hàm của các hàm tuyến tính, đạo hàm của hằng số.

Hai điều nên rõ ràng sau khi thực hiện ví dụ này: 

1. Bất kỳ chức năng nào chúng ta có thể viết ra bằng cách sử dụng tổng, sản phẩm, hằng số, quyền hạn, số mũ và logarit có thể có đạo hàm của nó được tính toán một cách cơ học bằng cách tuân theo các quy tắc này.
2. Có một con người tuân theo các quy tắc này có thể tẻ nhạt và dễ bị lỗi!

Rất may, hai sự kiện này cùng nhau gợi ý hướng tới một con đường phía trước: đây là một ứng cử viên hoàn hảo cho cơ giới hóa! Thật vậy, tuyên truyền ngược, mà chúng ta sẽ xem lại sau trong phần này, chính xác là điều đó. 

### Xấp xỉ tuyến tính Khi làm việc với các dẫn xuất, nó thường hữu ích để giải thích hình học xấp xỉ được sử dụng ở trên. Đặc biệt, lưu ý rằng phương trình  

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

xấp xỉ giá trị của $f$ bởi một đường đi qua điểm $(x, f(x))$ và có độ dốc $\frac{df}{dx}(x)$. Bằng cách này, chúng tôi nói rằng đạo hàm đưa ra một xấp xỉ tuyến tính cho hàm $f$, như minh họa dưới đây:

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

### Cao hơn Thiết bị trật tự

Bây giờ chúng ta hãy làm một cái gì đó có thể trên bề mặt có vẻ lạ. Lấy một hàm $f$ và tính toán đạo hàm $\frac{df}{dx}$. Điều này cho chúng ta tỷ lệ thay đổi $f$ tại bất kỳ điểm nào. 

Tuy nhiên, đạo hàm, $\frac{df}{dx}$, có thể được xem như một hàm chính nó, vì vậy không có gì ngăn cản chúng ta tính toán đạo hàm của $\frac{df}{dx}$ để có được $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$. Chúng ta sẽ gọi đây là đạo hàm thứ hai của $f$. Chức năng này là tốc độ thay đổi tỷ lệ thay đổi $f$, hay nói cách khác, tốc độ thay đổi đang thay đổi như thế nào. Chúng tôi có thể áp dụng đạo hàm bất kỳ số lần nào để có được cái được gọi là đạo hàm $n$-th. Để giữ cho ký hiệu sạch sẽ, chúng ta sẽ biểu thị đạo hàm $n$-th là  

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Hãy để chúng tôi cố gắng hiểu * tại sao * đây là một khái niệm hữu ích. Dưới đây, chúng tôi hình dung $f^{(2)}(x)$, $f^{(1)}(x)$ và $f(x)$.   

Đầu tiên, hãy xem xét trường hợp đạo hàm thứ hai $f^{(2)}(x)$ là một hằng số dương. Điều này có nghĩa là độ dốc của đạo hàm đầu tiên là dương. Kết quả là, đạo hàm đầu tiên $f^{(1)}(x)$ có thể bắt đầu âm, trở thành 0 tại một điểm, và sau đó trở thành dương cuối cùng. Điều này cho chúng ta biết độ dốc của chức năng ban đầu của chúng tôi $f$ và do đó, chức năng $f$ tự giảm, làm phẳng, sau đó tăng lên. Nói cách khác, hàm $f$ cong lên, và có một mức tối thiểu duy nhất như được hiển thị trong :numref:`fig_positive-second`. 

![If we assume the second derivative is a positive constant, then the fist derivative in increasing, which implies the function itself has a minimum.](../img/posSecDer.svg)
:label:`fig_positive-second`

Thứ hai, nếu đạo hàm thứ hai là hằng số âm, điều đó có nghĩa là đạo hàm thứ nhất đang giảm. Điều này ngụ ý đạo hàm đầu tiên có thể bắt đầu dương, trở thành 0 tại một điểm, và sau đó trở thành âm. Do đó, chức năng $f$ tự tăng lên, làm phẳng ra, sau đó giảm. Nói cách khác, hàm $f$ cong xuống, và có một tối đa duy nhất như được thể hiện trong :numref:`fig_negative-second`. 

![If we assume the second derivative is a negative constant, then the fist derivative in decreasing, which implies the function itself has a maximum.](../img/negSecDer.svg)
:label:`fig_negative-second`

Thứ ba, nếu đạo hàm thứ hai là một luôn bằng 0, thì đạo hàm đầu tiên sẽ không bao giờ thay đổi—nó là hằng số! Điều này có nghĩa là $f$ tăng (hoặc giảm) với tốc độ cố định và $f$ bản thân nó là một đường thẳng như thể hiện trong :numref:`fig_zero-second`. 

![If we assume the second derivative is zero, then the fist derivative is constant, which implies the function itself is a straight line.](../img/zeroSecDer.svg)
:label:`fig_zero-second`

Tóm lại, đạo hàm thứ hai có thể được hiểu là mô tả cách hàm $f$ đường cong. Một đạo hàm thứ hai dương dẫn đến một đường cong lên trên, trong khi đạo hàm thứ hai âm có nghĩa là $f$ đường cong xuống dưới, và đạo hàm thứ hai bằng 0 có nghĩa là $f$ không cong chút nào. 

Hãy để chúng tôi thực hiện điều này thêm một bước nữa. Hãy xem xét chức năng $g(x) = ax^{2}+ bx + c$. We can then computetính toán that 

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

Nếu chúng ta có một số hàm gốc $f(x)$ trong tâm trí, chúng ta có thể tính toán hai dẫn xuất đầu tiên và tìm các giá trị cho $a, b$, và $c$ mà làm cho chúng phù hợp với tính toán này. Tương tự như phần trước, nơi chúng ta thấy rằng đạo hàm đầu tiên đã đưa ra xấp xỉ tốt nhất với một đường thẳng, cấu trúc này cung cấp xấp xỉ tốt nhất bằng một bậc hai. Hãy để chúng tôi hình dung điều này cho $f(x) = \sin(x)$.

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

Chúng tôi sẽ mở rộng ý tưởng này cho ý tưởng về một loạt *Taylor * trong phần tiếp theo.  

### Dòng Taylor

Các dòng *Taylor * cung cấp một phương thức để xấp xỉ hàm $f(x)$ nếu chúng ta được đưa ra các giá trị cho các dẫn xuất $n$ đầu tiên tại một điểm $x_0$, tức là, $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$. Ý tưởng sẽ là tìm một đa thức độ $n$ phù hợp với tất cả các dẫn xuất đã cho tại $x_0$. 

Chúng tôi đã thấy trường hợp của $n=2$ trong phần trước và một đại số nhỏ cho thấy điều này là 

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

Như chúng ta có thể thấy ở trên, mẫu số của $2$ có để hủy bỏ $2$ chúng ta nhận được khi chúng ta lấy hai dẫn xuất là $x^2$, trong khi các thuật ngữ khác đều bằng 0. Logic tương tự áp dụng cho đạo hàm đầu tiên và chính giá trị. 

Nếu chúng ta đẩy logic hơn nữa lên $n=3$, chúng ta sẽ kết luận rằng 

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

nơi $6 = 3\ lần 2 = 3! $ comes from the constant we get in front if we take three derivatives of $x^3$. 

Hơn nữa, chúng ta có thể nhận được một mức độ $n$ đa thức bởi  

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

where the notation ký hiệu  

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

Thật vậy, $P_n(x)$ có thể được xem là xấp xỉ đa thức độ $n$-th tốt nhất với chức năng của chúng tôi $f(x)$. 

Mặc dù chúng ta sẽ không đi sâu vào lỗi của các xấp xỉ trên, nhưng điều đáng nói là giới hạn vô hạn. Trong trường hợp này, đối với các hàm được xử lý tốt (được gọi là hàm phân tích thực) như $\cos(x)$ hoặc $e^{x}$, chúng ta có thể viết ra số lượng thuật ngữ vô hạn và xấp xỉ cùng một hàm 

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

Lấy $f(x) = e^{x}$ làm ví dụ. Vì $e^{x}$ là đạo hàm riêng của nó, chúng ta biết rằng $f^{(n)}(x) = e^{x}$. Do đó, $e^{x}$ có thể được xây dựng lại bằng cách lấy loạt Taylor tại $x_0 = 0$, tức là, 

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

Chúng ta hãy xem cách điều này hoạt động trong mã và quan sát cách tăng mức độ xấp xỉ Taylor đưa chúng ta đến gần hơn với chức năng mong muốn $e^x$.

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

Taylor series có hai ứng dụng chính: 

1. * Các ứng dụng lý đầu*: Thường khi chúng ta cố gắng hiểu một hàm quá phức tạp, sử dụng Taylor series cho phép chúng ta biến nó thành một đa thức mà chúng ta có thể làm việc trực tiếp.

2. *Các ứng dụng số*: Một số chức năng như $e^{x}$ hoặc $\cos(x)$ rất khó để máy tính toán. Họ có thể lưu trữ các bảng giá trị ở độ chính xác cố định (và điều này thường được thực hiện), nhưng nó vẫn để lại các câu hỏi mở như “Chữ số 1000 của $\cos(1)$ là gì?” Taylor loạt thường hữu ích để trả lời những câu hỏi như vậy.  

## Tóm tắt

* Các dẫn xuất có thể được sử dụng để diễn tả các hàm thay đổi như thế nào khi chúng ta thay đổi đầu vào bằng một lượng nhỏ.
* Các dẫn xuất tiểu học có thể được kết hợp bằng cách sử dụng các quy tắc phái sinh để tạo ra các dẫn xuất phức tạp tùy ý.
* Các dẫn xuất có thể được lặp lại để có được các dẫn xuất bậc hai hoặc cao hơn. Mỗi sự gia tăng theo thứ tự cung cấp thông tin hạt mịn hơn về hành vi của chức năng.
* Sử dụng thông tin trong các dẫn xuất của một ví dụ dữ liệu duy nhất, chúng ta có thể xấp xỉ các hàm hoạt động tốt bằng các đa thức thu được từ chuỗi Taylor.

## Bài tập

1. Đạo hàm của $x^3-4x+1$ là gì?
2. Đạo hàm của $\log(\frac{1}{x})$ là gì?
3. Đúng hay Sai: Nếu $f'(x) = 0$ thì $f$ có tối đa hoặc tối thiểu là $x$?
4. Nơi tối thiểu là $f(x) = x\log(x)$ cho $x\ge0$ (trong đó chúng tôi giả định rằng $f$ có giá trị giới hạn là $0$ tại $f(0)$)?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/412)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1088)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1089)
:end_tab:

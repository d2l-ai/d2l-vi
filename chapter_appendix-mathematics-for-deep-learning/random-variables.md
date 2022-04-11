# Biến ngẫu nhiên
:label:`sec_random_variables`

Trong :numref:`sec_prob`, chúng ta đã thấy những điều cơ bản về cách làm việc với các biến ngẫu nhiên rời rạc, trong trường hợp của chúng ta đề cập đến những biến ngẫu nhiên đó lấy một tập hợp hữu hạn các giá trị có thể hoặc các số nguyên. Trong phần này, chúng ta phát triển lý thuyết về biến ngẫu nhiên *liên tục*, là các biến ngẫu nhiên có thể mang theo bất kỳ giá trị thực nào. 

## Biến ngẫu nhiên liên tục

Biến ngẫu nhiên liên tục là một chủ đề tinh tế hơn đáng kể so với các biến ngẫu nhiên rời rạc. Một sự tương tự công bằng để thực hiện là bước nhảy kỹ thuật có thể so sánh với bước nhảy giữa việc thêm danh sách các số và các chức năng tích hợp. Như vậy, chúng ta sẽ cần phải mất một thời gian để phát triển lý thuyết. 

### Từ rời rạc đến liên tục

Để hiểu những thách thức kỹ thuật bổ sung gặp phải khi làm việc với các biến ngẫu nhiên liên tục, chúng ta hãy thực hiện một thử nghiệm tư tưởng. Giả sử rằng chúng ta đang ném một phi tiêu vào bảng phi tiêu, và chúng tôi muốn biết xác suất rằng nó chạm chính xác $2 \text{cm}$ từ trung tâm của bảng. 

Để bắt đầu, chúng tôi tưởng tượng việc đo một chữ số chính xác duy nhất, nghĩa là với các thùng cho $0 \text{cm}$, $1 \text{cm}$, $2 \text{cm}$, v.v. Chúng tôi ném nói $100$ phi tiêu vào bảng phi tiêu, và nếu $20$ trong số họ rơi vào thùng cho $2\text{cm}$, chúng tôi kết luận rằng $20\ %$ of the darts we throw hit the board $2\ văn bản {cm} $ cách trung tâm. 

Tuy nhiên, khi chúng ta nhìn kỹ hơn, điều này không phù hợp với câu hỏi của chúng tôi! Chúng tôi muốn bình đẳng chính xác, trong khi những thùng chứa tất cả những gì rơi vào giữa $1.5\text{cm}$ và $2.5\text{cm}$. 

Không ngăn cản, chúng tôi tiếp tục xa hơn. Chúng tôi đo thậm chí chính xác hơn, nói $1.9\text{cm}$, $2.0\text{cm}$, $2.1\text{cm}$, và bây giờ thấy rằng có lẽ $3$ của phi tiêu $100$ đánh vào bảng trong xô $2.0\text{cm}$. Vì vậy, chúng tôi kết luận xác suất là $3\ %$. 

Tuy nhiên, điều này không giải quyết được bất cứ điều gì! Chúng tôi vừa đẩy vấn đề xuống một chữ số xa hơn. Hãy để chúng tôi trừu tượng một chút. Hãy tưởng tượng chúng ta biết xác suất $k$ chữ số đầu tiên khớp với $2.00000\ldots$ và chúng tôi muốn biết xác suất nó khớp với $k+1$ chữ số đầu tiên. Khá hợp lý khi giả định rằng chữ số ${k+1}^{\mathrm{th}}$ về cơ bản là một lựa chọn ngẫu nhiên từ bộ $\{0, 1, 2, \ldots, 9\}$. Ít nhất, chúng ta không thể hình thành một quá trình có ý nghĩa về thể chất, điều này sẽ buộc số lượng micromet tạo thành trung tâm thích kết thúc trong một $7$ so với $3$. 

Điều này có nghĩa là về bản chất mỗi chữ số bổ sung độ chính xác mà chúng ta yêu cầu sẽ giảm xác suất khớp theo hệ số $10$. Hoặc đặt một cách khác, chúng tôi sẽ mong đợi rằng 

$$
P(\text{distance is}\; 2.00\ldots, \;\text{to}\; k \;\text{digits} ) \approx p\cdot10^{-k}.
$$

Giá trị $p$ về cơ bản mã hóa những gì xảy ra với vài chữ số đầu tiên và $10^{-k}$ xử lý phần còn lại. 

Lưu ý rằng nếu chúng ta biết vị trí chính xác đến $k=4$ chữ số sau thập phân. Điều đó có nghĩa là chúng ta biết giá trị nằm trong khoảng thời gian nói $[(1.99995,2.00005]$ đó là một khoảng thời gian dài $2.00005-1.99995 = 10^{-4}$. Do đó, nếu chúng ta gọi độ dài của khoảng thời gian này $\epsilon$, chúng ta có thể nói 

$$
P(\text{distance is in an}\; \epsilon\text{-sized interval around}\; 2 ) \approx \epsilon \cdot p.
$$

Hãy để chúng tôi thực hiện một bước cuối cùng này hơn nữa. Chúng tôi đã suy nghĩ về điểm $2$ toàn bộ thời gian, nhưng không bao giờ nghĩ về những điểm khác. Không có gì khác nhau ở đó về cơ bản, nhưng đó là trường hợp giá trị $p$ có thể sẽ khác nhau. Ít nhất chúng tôi hy vọng rằng một người ném phi tiêu có nhiều khả năng đạt một điểm gần trung tâm, như $2\text{cm}$ hơn là $20\text{cm}$. Do đó, giá trị $p$ không cố định, mà phải phụ thuộc vào điểm $x$. Điều này cho chúng ta biết rằng chúng ta nên mong đợi 

$$P(\text{distance is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_deriv`

Thật vậy, :eqref:`eq_pdf_deriv` xác định chính xác hàm mật độ *xác suất*. Nó là một chức năng $p(x)$ mã hóa xác suất tương đối của đánh gần một điểm so với điểm khác. Hãy để chúng tôi hình dung những gì một chức năng như vậy có thể trông như thế nào.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot the probability density function for some random variable
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2)/np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2)/np.sqrt(2 * np.pi)

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2  # Define pi in torch

# Plot the probability density function for some random variable
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi)) + \
    0.8*torch.exp(-(x + 1)**2 / 2)/torch.sqrt(2 * torch.tensor(torch.pi))

d2l.plot(x, p, 'x', 'Density')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)).numpy() * 2  # Define pi in TensorFlow

# Plot the probability density function for some random variable
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi)) + \
    0.8*tf.exp(-(x + 1)**2 / 2)/tf.sqrt(2 * tf.constant(tf.pi))

d2l.plot(x, p, 'x', 'Density')
```

Các vị trí có giá trị hàm lớn cho biết các vùng mà chúng ta có nhiều khả năng tìm thấy giá trị ngẫu nhiên hơn. Các phần thấp là những khu vực mà chúng ta không có khả năng tìm thấy giá trị ngẫu nhiên. 

### Hàm mật độ xác suất

Bây giờ chúng ta hãy điều tra điều này thêm. Chúng ta đã thấy hàm mật độ xác suất là gì một cách trực giác cho một biến ngẫu nhiên $X$, cụ thể là hàm mật độ là một hàm $p(x)$ sao cho 

$$P(X \; \text{is in an}\; \epsilon \text{-sized interval around}\; x ) \approx \epsilon \cdot p(x).$$
:eqlabel:`eq_pdf_def`

Nhưng điều này ngụ ý những gì đối với các thuộc tính của $p(x)$? 

Đầu tiên, xác suất không bao giờ tiêu cực, do đó chúng ta nên mong đợi rằng $p(x) \ge 0$ là tốt. 

Thứ hai, chúng ta hãy tưởng tượng rằng chúng ta cắt $\mathbb{R}$ thành một số lượng vô hạn các lát rộng $\epsilon$, nói với lát $(\epsilon\cdot i, \epsilon \cdot (i+1)]$. Đối với mỗi trong số này, chúng tôi biết từ :eqref:`eq_pdf_def` xác suất là xấp xỉ 

$$
P(X \; \text{is in an}\; \epsilon\text{-sized interval around}\; x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$

vì vậy tóm tắt tất cả trong số họ nó nên được 

$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$

Đây không gì khác hơn là xấp xỉ của một tích phân được thảo luận trong :numref:`sec_integral_calculus`, do đó chúng ta có thể nói rằng 

$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$

Chúng ta biết rằng $P(X\in\mathbb{R}) = 1$, vì biến ngẫu nhiên phải đảm nhận số * some*, chúng ta có thể kết luận rằng đối với bất kỳ mật độ nào 

$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

Thật vậy, đào sâu vào điều này cho thấy rằng đối với bất kỳ $a$ và $b$, chúng ta thấy rằng 

$$
P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.
$$

Chúng ta có thể xấp xỉ điều này trong mã bằng cách sử dụng các phương thức xấp xỉ rời rạc giống như trước. Trong trường hợp này, chúng ta có thể xấp xỉ xác suất rơi vào vùng màu xanh.

```{.python .input}
# Approximate probability using numerical integration
epsilon = 0.01
x = np.arange(-5, 5, 0.01)
p = 0.2*np.exp(-(x - 3)**2 / 2) / np.sqrt(2 * np.pi) + \
    0.8*np.exp(-(x + 1)**2 / 2) / np.sqrt(2 * np.pi)

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {np.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab pytorch
# Approximate probability using numerical integration
epsilon = 0.01
x = torch.arange(-5, 5, 0.01)
p = 0.2*torch.exp(-(x - 3)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi)) +\
    0.8*torch.exp(-(x + 1)**2 / 2) / torch.sqrt(2 * torch.tensor(torch.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.tolist()[300:800], p.tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {torch.sum(epsilon*p[300:800])}'
```

```{.python .input}
#@tab tensorflow
# Approximate probability using numerical integration
epsilon = 0.01
x = tf.range(-5, 5, 0.01)
p = 0.2*tf.exp(-(x - 3)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi)) +\
    0.8*tf.exp(-(x + 1)**2 / 2) / tf.sqrt(2 * tf.constant(tf.pi))

d2l.set_figsize()
d2l.plt.plot(x, p, color='black')
d2l.plt.fill_between(x.numpy().tolist()[300:800], p.numpy().tolist()[300:800])
d2l.plt.show()

f'approximate Probability: {tf.reduce_sum(epsilon*p[300:800])}'
```

Nó chỉ ra rằng hai thuộc tính này mô tả chính xác không gian của các hàm mật độ xác suất có thể (hoặc *p.d.f.* cho chữ viết tắt thường gặp). Chúng là các chức năng không tiêu cực $p(x) \ge 0$ sao cho 

$$\int_{-\infty}^{\infty} p(x) \; dx = 1.$$
:eqlabel:`eq_pdf_int_one`

Chúng ta diễn giải hàm này bằng cách sử dụng tích hợp để có được xác suất biến ngẫu nhiên của chúng ta nằm trong một khoảng thời gian cụ thể: 

$$P(X\in(a, b]) = \int _ {a}^{b} p(x) \; dx.$$
:eqlabel:`eq_pdf_int_int`

Trong :numref:`sec_distributions`, chúng ta sẽ thấy một số bản phân phối phổ biến, nhưng chúng ta hãy tiếp tục làm việc trong bản tóm tắt. 

### Hàm phân phối tích lũy

Trong phần trước, chúng ta đã thấy khái niệm p.d.f Trong thực tế, đây là một phương pháp thường gặp để thảo luận về các biến ngẫu nhiên liên tục, nhưng nó có một cạm bẫy đáng kể: rằng các giá trị của p.d.f. không phải là xác suất, mà là một hàm mà chúng ta phải tích hợp để mang lại xác suất. Không có gì sai với mật độ lớn hơn $10$, miễn là nó không lớn hơn $10$ trong hơn một khoảng thời gian dài $1/10$. Điều này có thể phản trực quan, vì vậy mọi người thường nghĩ về hàm phân phối tích luỹ* hoặc c.d.f., mà * là* một xác suất. 

Đặc biệt, bằng cách sử dụng :eqref:`eq_pdf_int_int`, ta định nghĩa c.d.f. cho một biến ngẫu nhiên $X$ với mật độ $p(x)$ bởi 

$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$

Hãy để chúng tôi quan sát một vài tài sản. 

* $F(x) \rightarrow 0$ như $x\rightarrow -\infty$.
* $F(x) \rightarrow 1$ như $x\rightarrow \infty$.
* $F(x)$ không giảm ($y > x \implies F(y) \ge F(x)$).
* $F(x)$ là liên tục (không có nhảy) nếu $X$ là một biến ngẫu nhiên liên tục.

Với dấu đầu dòng thứ tư, lưu ý rằng điều này sẽ không đúng nếu $X$ rời rạc, giả sử lấy các giá trị $0$ và $1$ cả hai với xác suất $1/2$. Trong trường hợp đó 

$$
F(x) = \begin{cases}
0 & x < 0, \\
\frac{1}{2} & x < 1, \\
1 & x \ge 1.
\end{cases}
$$

Trong ví dụ này, chúng ta thấy một trong những lợi ích của việc làm việc với c.d.f., khả năng đối phó với các biến ngẫu nhiên liên tục hoặc rời rạc trong cùng một khuôn khổ, hoặc thực sự hỗn hợp của hai (lật một đồng xu: nếu đầu trả lại cuộn chết, nếu đuôi trả lại khoảng cách ném phi tiêu từ trung tâm của một phi tiêu bảng). 

### Phương tiện

Giả sử rằng chúng ta đang đối phó với một biến ngẫu nhiên $X$. Bản thân phân phối có thể khó giải thích. Nó thường hữu ích để có thể tóm tắt hành vi của một biến ngẫu nhiên một cách chính xác. Các số giúp chúng ta nắm bắt hành vi của một biến ngẫu nhiên được gọi là *thống kê tóm lượng*. Những cái thường gặp nhất là *mean*, *variance*, và * độ lệch chuẩn*. 

*mean* mã hóa giá trị trung bình của một biến ngẫu nhiên. Nếu chúng ta có một biến ngẫu nhiên rời rạc $X$, lấy các giá trị $x_i$ với xác suất $p_i$, thì trung bình được đưa ra bởi trung bình có trọng số: tổng các giá trị lần xác suất biến ngẫu nhiên có giá trị đó: 

$$\mu_X = E[X] = \sum_i x_i p_i.$$
:eqlabel:`eq_exp_def`

Cách chúng ta nên giải thích trung bình (mặc dù thận trọng) là nó cho chúng ta biết về cơ bản nơi biến ngẫu nhiên có xu hướng được đặt. 

Như một ví dụ tối giản mà chúng ta sẽ kiểm tra trong suốt phần này, chúng ta hãy lấy $X$ là biến ngẫu nhiên lấy giá trị $a-2$ với xác suất $p$, $a+2$ với xác suất $p$ và $a$ với xác suất $1-2p$. Chúng ta có thể tính toán bằng cách sử dụng :eqref:`eq_exp_def` rằng, cho bất kỳ lựa chọn nào có thể là $a$ và $p$, trung bình là 

$$
\mu_X = E[X] = \sum_i x_i p_i = (a-2)p + a(1-2p) + (a+2)p = a.
$$

Vì vậy, chúng ta thấy rằng trung bình là $a$. Điều này phù hợp với trực giác kể từ $a$ là vị trí xung quanh mà chúng tôi tập trung biến ngẫu nhiên của chúng tôi. 

Bởi vì chúng rất hữu ích, chúng ta hãy tóm tắt một vài thuộc tính. 

* Đối với bất kỳ biến ngẫu nhiên $X$ và số $a$ và $b$, chúng tôi có $\mu_{aX+b} = a\mu_X + b$.
* Nếu chúng ta có hai biến ngẫu nhiên $X$ và $Y$, chúng ta có $\mu_{X+Y} = \mu_X+\mu_Y$.

Phương tiện rất hữu ích để hiểu hành vi trung bình của một biến ngẫu nhiên, tuy nhiên trung bình là không đủ để thậm chí có một sự hiểu biết trực quan đầy đủ. Kiếm lợi nhuận $\$10\ pm\ $1$ mỗi lần bán rất khác so với việc tạo ra $\$10\ pm\ $15$ mỗi lần bán mặc dù có cùng giá trị trung bình. Cái thứ hai có mức độ dao động lớn hơn nhiều, và do đó đại diện cho rủi ro lớn hơn nhiều. Do đó, để hiểu được hành vi của một biến ngẫu nhiên, chúng ta sẽ cần tối thiểu thêm một thước đo nữa: một số thước đo về cách biến ngẫu nhiên dao động rộng rãi. 

### Phương sai

Điều này dẫn chúng ta xem xét *variance* của một biến ngẫu nhiên. Đây là một thước đo định lượng về cách xa một biến ngẫu nhiên lệch khỏi trung bình. Hãy xem xét biểu thức $X - \mu_X$. Đây là độ lệch của biến ngẫu nhiên so với trung bình của nó. Giá trị này có thể là dương hoặc âm, vì vậy chúng ta cần phải làm một cái gì đó để làm cho nó tích cực để chúng ta đo độ lớn của độ lệch. 

Một điều hợp lý để thử là nhìn vào $\left|X-\mu_X\right|$, và thực sự điều này dẫn đến một đại lượng hữu ích gọi là độ lệch tuyệt đối * trung bình, tuy nhiên do kết nối với các lĩnh vực toán học và thống kê khác, người ta thường sử dụng một giải pháp khác. 

Đặc biệt, họ nhìn vào $(X-\mu_X)^2.$ Nếu chúng ta nhìn vào kích thước điển hình của số lượng này bằng cách lấy trung bình, chúng tôi đến phương sai 

$$\sigma_X^2 = \mathrm{Var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2.$$
:eqlabel:`eq_var_def`

Sự bình đẳng cuối cùng trong :eqref:`eq_var_def` nắm giữ bằng cách mở rộng định nghĩa ở giữa và áp dụng các thuộc tính của kỳ vọng. 

Chúng ta hãy nhìn vào ví dụ của chúng tôi nơi $X$ là biến ngẫu nhiên mà lấy giá trị $a-2$ với xác suất $p$, $a+2$ với xác suất $p$ và $a$ với xác suất $1-2p$. Trong trường hợp này $\mu_X = a$, vì vậy tất cả chúng ta cần phải tính toán là $E\left[X^2\right]$. Điều này có thể dễ dàng được thực hiện: 

$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)^2p = a^2 + 8p.
$$

Do đó, chúng ta thấy rằng bởi :eqref:`eq_var_def` phương sai của chúng tôi là 

$$
\sigma_X^2 = \mathrm{Var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

Kết quả này một lần nữa có ý nghĩa. $p$ lớn nhất có thể là $1/2$ tương ứng với việc chọn $a-2$ hoặc $a+2$ với lật đồng xu. Phương sai của việc này là $4$ tương ứng với thực tế là cả $a-2$ và $a+2$ đều là $2$ đơn vị cách xa trung bình và $2^2 = 4$. Ở đầu kia của quang phổ, nếu $p=0$, biến ngẫu nhiên này luôn lấy giá trị $0$ và do đó nó không có phương sai nào cả. 

Chúng tôi sẽ liệt kê một vài thuộc tính của phương sai dưới đây: 

* Đối với bất kỳ biến ngẫu nhiên $X$, $\mathrm{Var}(X) \ge 0$, với $\mathrm{Var}(X) = 0$ nếu và chỉ khi $X$ là một hằng số.
* Đối với bất kỳ biến ngẫu nhiên $X$ và số $a$ và $b$, chúng tôi có $\mathrm{Var}(aX+b) = a^2\mathrm{Var}(X)$.
* Nếu chúng ta có hai biến ngẫu nhiên * độc lập* $X$ và $Y$, chúng ta có $\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$.

Khi giải thích các giá trị này, có thể có một chút nấc cục. Đặc biệt, chúng ta hãy thử tưởng tượng điều gì sẽ xảy ra nếu chúng ta theo dõi các đơn vị thông qua tính toán này. Giả sử rằng chúng tôi đang làm việc với xếp hạng sao được gán cho một sản phẩm trên trang web. Sau đó, $a$, $a-2$, và $a+2$ đều được đo bằng đơn vị sao. Tương tự, trung bình $\mu_X$ sau đó cũng được đo bằng sao (là trung bình có trọng số). Tuy nhiên, nếu chúng ta đi đến phương sai, chúng ta ngay lập tức gặp phải một vấn đề, đó là chúng ta muốn xem xét $(X-\mu_X)^2$, đó là đơn vị của * bình phương sao*. Điều này có nghĩa là bản thân phương sai không thể so sánh với các phép đo ban đầu. Để làm cho nó có thể hiểu được, chúng ta sẽ cần phải trở lại các đơn vị ban đầu của chúng tôi. 

### Độ lệch chuẩn

Thống kê tóm tắt này luôn có thể được suy ra từ phương sai bằng cách lấy căn bậc hai! Do đó chúng tôi xác định độ lệch tiêu chuẩn* là 

$$
\sigma_X = \sqrt{\mathrm{Var}(X)}.
$$

Trong ví dụ của chúng tôi, điều này có nghĩa là bây giờ chúng ta có độ lệch chuẩn là $\sigma_X = 2\sqrt{2p}$. Nếu chúng ta đang đối phó với các đơn vị sao cho ví dụ đánh giá của chúng tôi, $\sigma_X$ một lần nữa trong các đơn vị sao. 

Các thuộc tính chúng tôi có cho phương sai có thể được đặt lại cho độ lệch chuẩn. 

* Đối với bất kỳ biến ngẫu nhiên $X$, $\sigma_{X} \ge 0$.
* Đối với bất kỳ biến ngẫu nhiên $X$ và số $a$ và $b$, chúng tôi có $\sigma_{aX+b} = |a|\sigma_{X}$
* Nếu chúng ta có hai biến ngẫu nhiên * độc lập* $X$ và $Y$, chúng ta có $\sigma_{X+Y} = \sqrt{\sigma_{X}^2 + \sigma_{Y}^2}$.

Nó là tự nhiên tại thời điểm này để hỏi, “Nếu độ lệch chuẩn nằm trong các đơn vị của biến ngẫu nhiên ban đầu của chúng ta, nó có đại diện cho một cái gì đó chúng ta có thể vẽ liên quan đến biến ngẫu nhiên đó?” Câu trả lời là một có vang dội! Thật vậy giống như trung bình nói với chúng ta vị trí điển hình của biến ngẫu nhiên của chúng ta, độ lệch chuẩn cho phạm vi điển hình của biến thể của biến ngẫu nhiên đó. Chúng ta có thể làm cho điều này nghiêm ngặt với cái được gọi là bất bình đẳng của Chebyshev: 

$$P\left(X \not\in [\mu_X - \alpha\sigma_X, \mu_X + \alpha\sigma_X]\right) \le \frac{1}{\alpha^2}.$$
:eqlabel:`eq_chebyshev`

Hoặc để nêu nó bằng lời nói trong trường hợp $\alpha=10$, $99\ %$ of the samples from any random variable fall within $10$ độ lệch chuẩn của trung bình. Điều này đưa ra một giải thích ngay lập tức cho thống kê tóm tắt tiêu chuẩn của chúng tôi. 

Để xem cách tuyên bố này là khá tinh tế, chúng ta hãy xem xét ví dụ chạy của chúng tôi một lần nữa nơi $X$ là biến ngẫu nhiên mà lấy giá trị $a-2$ với xác suất $p$, $a+2$ với xác suất $p$ và $a$ với xác suất $1-2p$. Chúng tôi thấy rằng trung bình là $a$ và độ lệch chuẩn là $2\sqrt{2p}$. Điều này có nghĩa là, nếu chúng ta lấy sự bất bình đẳng của Chebyshev :eqref:`eq_chebyshev` với $\alpha = 2$, chúng ta thấy rằng biểu thức là 

$$
P\left(X \not\in [a - 4\sqrt{2p}, a + 4\sqrt{2p}]\right) \le \frac{1}{4}.
$$

Điều này có nghĩa là $75\ %$ của thời gian, biến ngẫu nhiên này sẽ nằm trong khoảng thời gian này cho bất kỳ giá trị nào là $p$. Bây giờ, nhận thấy rằng như $p\ rightarrow 0$, this interval also converges to the single point $a$.  But we know that our random variable takes the values $a-2, a$, and $a+2$ only so eventually we can be certain $a-2$ and $a+2$ sẽ rơi ra ngoài khoảng thời gian! Câu hỏi đặt ra là, tại những gì $p$ làm điều đó xảy ra. Vì vậy, chúng tôi muốn giải quyết: cho những gì $p$ làm $a+4\sqrt{2p} = a+2$, được giải quyết khi $p=1/8$, đó là * chính xác* $p$ đầu tiên nơi nó có thể xảy ra mà không vi phạm tuyên bố của chúng tôi rằng không quá $1/4$ mẫu từ phân phối sẽ nằm ngoài khoảng thời gian ($1/8$ ở bên trái và $1/8$ bên phải). 

Hãy để chúng tôi hình dung điều này. Chúng tôi sẽ hiển thị xác suất nhận được ba giá trị dưới dạng ba thanh dọc với chiều cao tỷ lệ thuận với xác suất. Khoảng thời gian sẽ được vẽ dưới dạng một đường ngang ở giữa. Cốt truyện đầu tiên cho thấy những gì xảy ra cho $p > 1/8$ trong đó khoảng thời gian an toàn chứa tất cả các điểm.

```{.python .input}
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * np.sqrt(2 * p),
                   a + 4 * np.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * np.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, 0.2)
```

```{.python .input}
#@tab pytorch
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * torch.sqrt(2 * p),
                   a + 4 * torch.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * torch.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, torch.tensor(0.2))
```

```{.python .input}
#@tab tensorflow
# Define a helper to plot these figures
def plot_chebyshev(a, p):
    d2l.set_figsize()
    d2l.plt.stem([a-2, a, a+2], [p, 1-2*p, p], use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')

    d2l.plt.hlines(0.5, a - 4 * tf.sqrt(2 * p),
                   a + 4 * tf.sqrt(2 * p), 'black', lw=4)
    d2l.plt.vlines(a - 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.vlines(a + 4 * tf.sqrt(2 * p), 0.53, 0.47, 'black', lw=1)
    d2l.plt.title(f'p = {p:.3f}')

    d2l.plt.show()

# Plot interval when p > 1/8
plot_chebyshev(0.0, tf.constant(0.2))
```

Thứ hai cho thấy ở $p = 1/8$, khoảng thời gian chính xác chạm vào hai điểm. Điều này cho thấy bất đẳng thức là * sharp*, vì không có khoảng thời gian nhỏ hơn có thể được thực hiện trong khi vẫn giữ sự bất bình đẳng đúng.

```{.python .input}
# Plot interval when p = 1/8
plot_chebyshev(0.0, 0.125)
```

```{.python .input}
#@tab pytorch
# Plot interval when p = 1/8
plot_chebyshev(0.0, torch.tensor(0.125))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p = 1/8
plot_chebyshev(0.0, tf.constant(0.125))
```

Thứ ba cho thấy rằng đối với $p < 1/8$ khoảng thời gian chỉ chứa trung tâm. Điều này không làm mất hiệu lực bất bình đẳng vì chúng ta chỉ cần đảm bảo rằng không quá $1/4$ xác suất nằm ngoài khoảng thời gian, có nghĩa là một lần $p < 1/8$, hai điểm ở $a-2$ và $a+2$ có thể bị loại bỏ.

```{.python .input}
# Plot interval when p < 1/8
plot_chebyshev(0.0, 0.05)
```

```{.python .input}
#@tab pytorch
# Plot interval when p < 1/8
plot_chebyshev(0.0, torch.tensor(0.05))
```

```{.python .input}
#@tab tensorflow
# Plot interval when p < 1/8
plot_chebyshev(0.0, tf.constant(0.05))
```

### Phương tiện và phương sai trong liên tục

Điều này đã được tất cả về các biến ngẫu nhiên rời rạc, nhưng trường hợp của các biến ngẫu nhiên liên tục là tương tự nhau. Để hiểu trực giác điều này hoạt động như thế nào, hãy tưởng tượng rằng chúng ta chia dòng số thực thành các khoảng thời gian $\epsilon$ được đưa ra bởi $(\epsilon i, \epsilon (i+1)]$. Khi chúng tôi làm điều này, biến ngẫu nhiên liên tục của chúng tôi đã được thực hiện rời rạc và chúng tôi có thể sử dụng :eqref:`eq_exp_def` nói rằng 

$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$

trong đó $p_X$ là mật độ $X$. Đây là một xấp xỉ với tích phân của $xp_X(x)$, vì vậy chúng ta có thể kết luận rằng 

$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

Tương tự, sử dụng :eqref:`eq_var_def` phương sai có thể được viết là 

$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

Tất cả mọi thứ đã nêu ở trên về trung bình, phương sai và độ lệch chuẩn vẫn được áp dụng trong trường hợp này. Ví dụ, nếu chúng ta xem xét biến ngẫu nhiên với mật độ 

$$
p(x) = \begin{cases}
1 & x \in [0,1], \\
0 & \text{otherwise}.
\end{cases}
$$

we can compute tính toán 

$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$

và 

$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

Như một cảnh báo, chúng ta hãy xem xét thêm một ví dụ, được gọi là bản phân phối *Cauchy*. Đây là bản phân phối với p.d.f. được đưa ra bởi 

$$
p(x) = \frac{1}{1+x^2}.
$$

```{.python .input}
# Plot the Cauchy distribution p.d.f.
x = np.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
# Plot the Cauchy distribution p.d.f.
x = torch.arange(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
# Plot the Cauchy distribution p.d.f.
x = tf.range(-5, 5, 0.01)
p = 1 / (1 + x**2)

d2l.plot(x, p, 'x', 'p.d.f.')
```

Hàm này trông vô tội, và thực sự tham khảo một bảng tích phân sẽ cho thấy nó có diện tích một bên dưới nó, và do đó nó định nghĩa một biến ngẫu nhiên liên tục. 

Để xem những gì đi lạc lối, chúng ta hãy cố gắng tính toán phương sai của điều này. Điều này sẽ liên quan đến việc sử dụng máy tính :eqref:`eq_var_def` 

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$

Chức năng ở bên trong trông như thế này:

```{.python .input}
# Plot the integrand needed to compute the variance
x = np.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab pytorch
# Plot the integrand needed to compute the variance
x = torch.arange(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

```{.python .input}
#@tab tensorflow
# Plot the integrand needed to compute the variance
x = tf.range(-20, 20, 0.01)
p = x**2 / (1 + x**2)

d2l.plot(x, p, 'x', 'integrand')
```

Chức năng này rõ ràng có diện tích vô hạn dưới nó vì về cơ bản nó là hằng số với một cú nhúng nhỏ gần 0, và thực sự chúng ta có thể chỉ ra rằng 

$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$

Điều này có nghĩa là nó không có phương sai hữu hạn được xác định rõ. 

Tuy nhiên, nhìn sâu hơn cho thấy một kết quả thậm chí còn đáng lo ngại hơn. Hãy để chúng tôi cố gắng tính toán trung bình bằng cách sử dụng :eqref:`eq_exp_def`. Sử dụng sự thay đổi của công thức biến, chúng ta thấy 

$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du.
$$

Tích phân bên trong là định nghĩa của logarit, vì vậy điều này là về bản chất $\log(\infty) = \infty$, do đó không có giá trị trung bình được xác định rõ cả! 

Các nhà khoa học máy học xác định mô hình của họ để chúng ta thường không cần phải đối phó với những vấn đề này và trong phần lớn các trường hợp sẽ đối phó với các biến ngẫu nhiên với các phương tiện và phương sai được xác định rõ ràng. Tuy nhiên, mọi biến ngẫu nhiên thường có * đuôi nặng* (đó là những biến ngẫu nhiên trong đó xác suất nhận được các giá trị lớn đủ lớn để làm cho những thứ như trung bình hoặc phương sai không xác định) rất hữu ích trong việc mô hình hóa các hệ thống vật lý, do đó đáng để biết rằng chúng tồn tại. 

### Chức năng mật độ khớp

Công việc trên tất cả giả định chúng ta đang làm việc với một biến ngẫu nhiên có giá trị thực duy nhất. Nhưng điều gì sẽ xảy ra nếu chúng ta đang đối phó với hai hoặc nhiều biến ngẫu nhiên có khả năng tương quan cao? Hoàn cảnh này là chuẩn mực trong học máy: tưởng tượng các biến ngẫu nhiên như $R_{i, j}$ mã hóa giá trị màu đỏ của pixel tại tọa độ $(i, j)$ trong một hình ảnh, hoặc $P_t$ là một biến ngẫu nhiên được đưa ra bởi giá cổ phiếu tại thời điểm $t$. Các pixel gần đó có xu hướng có màu tương tự và thời gian gần đó có xu hướng có giá tương tự. Chúng ta không thể coi chúng như các biến ngẫu nhiên riêng biệt và mong đợi tạo ra một mô hình thành công (chúng ta sẽ thấy trong :numref:`sec_naive_bayes` một mô hình hoạt động kém do giả định như vậy). Chúng ta cần phát triển ngôn ngữ toán học để xử lý các biến ngẫu nhiên liên tục tương quan này. 

Rất may, với nhiều tích phân trong :numref:`sec_integral_calculus`, chúng ta có thể phát triển một ngôn ngữ như vậy. Giả sử rằng chúng ta có, để đơn giản, hai biến ngẫu nhiên $X, Y$ có thể tương quan. Sau đó, tương tự như trường hợp của một biến duy nhất, chúng ta có thể đặt câu hỏi: 

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ).
$$

Lý luận tương tự như trường hợp biến duy nhất cho thấy điều này nên xấp xỉ 

$$
P(X \;\text{is in an}\; \epsilon \text{-sized interval around}\; x \; \text{and} \;Y \;\text{is in an}\; \epsilon \text{-sized interval around}\; y ) \approx \epsilon^{2}p(x, y),
$$

cho một số chức năng $p(x, y)$. Đây được gọi là mật độ khớp của $X$ và $Y$. Các thuộc tính tương tự đúng với điều này như chúng ta đã thấy trong trường hợp biến duy nhất. Cụ thể là: 

* $p(x, y) \ge 0$;
* $\int _ {\mathbb{R}^2} p(x, y) \;dx \;dy = 1$;
* $P((X, Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x, y) \;dx \;dy$.

Bằng cách này, chúng ta có thể đối phó với nhiều biến ngẫu nhiên có khả năng tương quan. Nếu chúng ta muốn làm việc với nhiều hơn hai biến ngẫu nhiên, chúng ta có thể mở rộng mật độ đa biến đến nhiều tọa độ như mong muốn bằng cách xem xét $p(\mathbf{x}) = p(x_1, \ldots, x_n)$. Các tính chất tương tự là không âm, và có tổng tích phân của một vẫn giữ. 

### Phân phối biên Khi xử lý nhiều biến, chúng ta thường muốn có thể bỏ qua các mối quan hệ và hỏi, “biến này được phân phối như thế nào?” Phân phối như vậy được gọi là phân phối cận biên *. 

Để được cụ thể, chúng ta hãy giả sử rằng chúng ta có hai biến ngẫu nhiên $X, Y$ với mật độ khớp được đưa ra bởi $p _ {X, Y}(x, y)$. Chúng ta sẽ sử dụng chỉ số dưới để chỉ ra các biến ngẫu nhiên mà mật độ dùng để làm gì. Câu hỏi tìm ra sự phân bố cận biên là dùng chức năng này, và sử dụng nó để tìm $p _ X(x)$. 

Như với hầu hết mọi thứ, tốt nhất là quay lại hình ảnh trực quan để tìm ra những gì nên đúng. Nhớ lại rằng mật độ là chức năng $p _ X$ để 

$$
P(X \in [x, x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$

Không có đề cập đến $Y$, nhưng nếu tất cả chúng ta được đưa ra là $p _{X, Y}$, chúng ta cần bao gồm $Y$ bằng cách nào đó. Trước tiên chúng ta có thể quan sát rằng điều này giống như 

$$
P(X \in [x, x+\epsilon] \text{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$

Mật độ của chúng tôi không trực tiếp cho chúng tôi biết về những gì xảy ra trong trường hợp này, chúng ta cần phải chia thành các khoảng nhỏ trong $y$ là tốt, vì vậy chúng tôi có thể viết điều này như 

$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \text{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X, Y}(x, \epsilon\cdot i).
\end{aligned}
$$

![By summing along the columns of our array of probabilities, we are able to obtain the marginal distribution for just the random variable represented along the $x$-axis.](../img/marginal.svg)
:label:`fig_marginal`

Điều này cho chúng ta biết thêm giá trị của mật độ dọc theo một loạt các ô vuông trong một đường như được thể hiện trong :numref:`fig_marginal`. Thật vậy, sau khi hủy bỏ một yếu tố epsilon từ cả hai phía, và nhận ra tổng ở bên phải là tích phân trên $y$, chúng ta có thể kết luận rằng 

$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X, Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
\end{aligned}
$$

Vì vậy chúng ta thấy 

$$
p _ X(x) = \int_{-\infty}^\infty p_{X, Y}(x, y) \; dy.
$$

Điều này cho chúng ta biết rằng để có được một phân phối biên, chúng tôi tích hợp trên các biến mà chúng tôi không quan tâm. Quá trình này thường được gọi là *tích hợp out* hoặc *marginalized out* các biến không cần thiết. 

### Hiệp phương sai

Khi xử lý nhiều biến ngẫu nhiên, có một thống kê tóm tắt bổ sung rất hữu ích để biết: * covariance*. Điều này đo mức độ mà hai biến ngẫu nhiên dao động với nhau. 

Giả sử rằng chúng ta có hai biến ngẫu nhiên $X$ và $Y$, để bắt đầu, chúng ta hãy giả sử chúng rời rạc, lấy giá trị $(x_i, y_j)$ với xác suất $p_{ij}$. Trong trường hợp này, phương sai được định nghĩa là 

$$\sigma_{XY} = \mathrm{Cov}(X, Y) = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}. = E[XY] - E[X]E[Y].$$
:eqlabel:`eq_cov_def`

Để suy nghĩ về điều này một cách trực giác: hãy xem xét cặp biến ngẫu nhiên sau đây. Giả sử rằng $X$ lấy các giá trị $1$ và $3$, và $Y$ lấy các giá trị $-1$ và $3$. Giả sử rằng chúng ta có xác suất sau 

$$
\begin{aligned}
P(X = 1 \; \text{and} \; Y = -1) & = \frac{p}{2}, \\
P(X = 1 \; \text{and} \; Y = 3) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = -1) & = \frac{1-p}{2}, \\
P(X = 3 \; \text{and} \; Y = 3) & = \frac{p}{2},
\end{aligned}
$$

trong đó $p$ là một tham số trong $[0,1]$ chúng tôi nhận được để chọn. Lưu ý rằng nếu $p=1$ thì cả hai đều luôn là giá trị tối thiểu hoặc tối đa của chúng cùng một lúc và nếu $p=0$, chúng được đảm bảo lấy giá trị lật của chúng đồng thời (một giá trị lớn khi giá kia nhỏ và ngược lại). Nếu $p=1/2$, thì bốn khả năng đều có khả năng như nhau và không nên liên quan. Let us computetính toán the covariance đồng phương sai. Đầu tiên, lưu ý $\mu_X = 2$ và $\mu_Y = 1$, vì vậy chúng tôi có thể tính toán bằng :eqref:`eq_cov_def`: 

$$
\begin{aligned}
\mathrm{Cov}(X, Y) & = \sum_{i, j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2.
\end{aligned}
$$

Khi $p=1$ (trường hợp cả hai đều dương tính tối đa hoặc tiêu cực cùng một lúc) có sự đồng phương sai là $2$. Khi $p=0$ (trường hợp chúng bị lật) thì đồng phương sai là $-2$. Cuối cùng, khi $p=1/2$ (trường hợp chúng không liên quan), phương sai là $0$. Do đó chúng ta thấy rằng sự đồng phương sai đo lường cách thức hai biến ngẫu nhiên này có liên quan. 

Một lưu ý nhanh về sự đồng phương sai là nó chỉ đo các mối quan hệ tuyến tính này. Các mối quan hệ phức tạp hơn như $X = Y^2$ trong đó $Y$ được chọn ngẫu nhiên từ $\{-2, -1, 0, 1, 2\}$ với xác suất bằng nhau có thể bị bỏ qua. Thật vậy một tính toán nhanh chóng cho thấy các biến ngẫu nhiên này có sự đồng phương sai 0, mặc dù một là một hàm xác định của cái kia. 

Đối với các biến ngẫu nhiên liên tục, nhiều câu chuyện tương tự giữ. Tại thời điểm này, chúng tôi khá thoải mái khi thực hiện quá trình chuyển đổi giữa rời rạc và liên tục, vì vậy chúng tôi sẽ cung cấp tương tự liên tục của :eqref:`eq_cov_def` mà không có bất kỳ nguồn gốc nào. 

$$
\sigma_{XY} = \int_{\mathbb{R}^2} (x-\mu_X)(y-\mu_Y)p(x, y) \;dx \;dy.
$$

Để trực quan hóa, chúng ta hãy xem một tập hợp các biến ngẫu nhiên với phương sai có thể điều chỉnh.

```{.python .input}
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = covs[i]*X + np.random.normal(0, 1, (500))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = covs[i]*X + torch.randn(500)

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable covariance
covs = [-0.9, 0.0, 1.2]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = covs[i]*X + tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i+1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cov = {covs[i]}')
d2l.plt.show()
```

Hãy để chúng tôi xem một số tính chất của đồng phương sai: 

* Đối với bất kỳ biến ngẫu nhiên $X$, $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$.
* Đối với bất kỳ biến ngẫu nhiên $X, Y$ và số $a$ và $b$, $\mathrm{Cov}(aX+b, Y) = \mathrm{Cov}(X, aY+b) = a\mathrm{Cov}(X, Y)$.
* Nếu $X$ và $Y$ độc lập thì $\mathrm{Cov}(X, Y) = 0$.

Ngoài ra, chúng ta có thể sử dụng phương sai để mở rộng mối quan hệ mà chúng ta đã thấy trước đây. Nhớ lại đó là $X$ và $Y$ là hai biến ngẫu nhiên độc lập sau đó 

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y).
$$

Với kiến thức về đồng phương sai, chúng ta có thể mở rộng mối quan hệ này. Indeedthật vậy, some algebrađại số can showchỉ that in generalchung, 

$$
\mathrm{Var}(X+Y) = \mathrm{Var}(X) + \mathrm{Var}(Y) + 2\mathrm{Cov}(X, Y).
$$

Điều này cho phép chúng ta khái quát hóa quy tắc tổng hợp phương sai cho các biến ngẫu nhiên tương quan. 

### Tương quan

Như chúng ta đã làm trong trường hợp phương tiện và phương sai, bây giờ chúng ta hãy xem xét các đơn vị. Nếu $X$ được đo bằng một đơn vị (nói inch) và $Y$ được đo bằng một đơn vị khác (ví dụ đô la), phương sai được đo bằng tích của hai đơn vị này $\text{inches} \times \text{dollars}$. Các đơn vị này có thể khó diễn giải. Những gì chúng ta thường muốn trong trường hợp này là một phép đo ít đơn vị về sự liên quan. Thật vậy, thường thì chúng ta không quan tâm đến mối tương quan định lượng chính xác, mà là hỏi xem mối tương quan có theo cùng một hướng hay không, và mối quan hệ mạnh mẽ như thế nào. 

Để xem điều gì có ý nghĩa, chúng ta hãy thực hiện một thí nghiệm suy nghĩ. Giả sử rằng chúng ta chuyển đổi các biến ngẫu nhiên của chúng tôi trong inch và đô la để được tính bằng inch và xu. Trong trường hợp này biến ngẫu nhiên $Y$ được nhân với $100$. Nếu chúng ta làm việc thông qua định nghĩa, điều này có nghĩa là $\mathrm{Cov}(X, Y)$ sẽ được nhân với $100$. Như vậy chúng ta thấy rằng trong trường hợp này, sự thay đổi của các đơn vị thay đổi phương sai theo hệ số $100$. Do đó, để tìm ra biện pháp tương quan bất biến đơn vị của chúng ta, chúng ta sẽ cần phải chia cho một cái gì đó khác cũng được thu nhỏ bởi $100$. Thật vậy, chúng tôi có một ứng cử viên rõ ràng, độ lệch chuẩn! Thật vậy nếu chúng ta định nghĩa hệ số tương quan *là 

$$\rho(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\sigma_{X}\sigma_{Y}},$$
:eqlabel:`eq_cor_def`

we see that this is a unit-lessđơn vị lessít hơn value giá trị. Một toán học nhỏ có thể chỉ ra rằng con số này nằm trong khoảng $-1$ và $1$ với $1$ có nghĩa là tương quan tích cực tối đa, trong khi $-1$ có nghĩa là tương quan tiêu cực tối đa. 

Quay trở lại ví dụ riêng biệt rõ ràng của chúng tôi ở trên, chúng ta có thể thấy rằng $\sigma_X = 1$ và $\sigma_Y = 2$, vì vậy chúng ta có thể tính toán mối tương quan giữa hai biến ngẫu nhiên sử dụng :eqref:`eq_cor_def` để thấy rằng 

$$
\rho(X, Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$

Điều này hiện dao động trong khoảng $-1$ và $1$ với hành vi mong đợi là $1$ có nghĩa là tương quan nhất, và $-1$ có nghĩa là tương quan tối thiểu. 

Như một ví dụ khác, coi $X$ là bất kỳ biến ngẫu nhiên nào, và $Y=aX+b$ như bất kỳ hàm xác định tuyến tính nào của $X$. Then, one can computetính toán that 

$$\sigma_{Y} = \sigma_{aX+b} = |a|\sigma_{X},$$

$$\mathrm{Cov}(X, Y) = \mathrm{Cov}(X, aX+b) = a\mathrm{Cov}(X, X) = a\mathrm{Var}(X),$$

và do đó bởi :eqref:`eq_cor_def` 

$$
\rho(X, Y) = \frac{a\mathrm{Var}(X)}{|a|\sigma_{X}^2} = \frac{a}{|a|} = \mathrm{sign}(a).
$$

Do đó, chúng ta thấy rằng mối tương quan là $+1$ cho bất kỳ $a > 0$ nào và $-1$ cho bất kỳ $a < 0$ nào minh họa rằng tương quan đo mức độ và định hướng của hai biến ngẫu nhiên có liên quan, chứ không phải quy mô mà biến thể mất. 

Chúng ta hãy một lần nữa vẽ một tập hợp các biến ngẫu nhiên với tương quan điều chỉnh.

```{.python .input}
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = np.random.normal(0, 1, 500)
    Y = cors[i] * X + np.sqrt(1 - cors[i]**2) * np.random.normal(0, 1, 500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.asnumpy(), Y.asnumpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = torch.randn(500)
    Y = cors[i] * X + torch.sqrt(torch.tensor(1) -
                                 cors[i]**2) * torch.randn(500)

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
# Plot a few random variables adjustable correlations
cors = [-0.9, 0.0, 1.0]
d2l.plt.figure(figsize=(12, 3))
for i in range(3):
    X = tf.random.normal((500, ))
    Y = cors[i] * X + tf.sqrt(tf.constant(1.) -
                                 cors[i]**2) * tf.random.normal((500, ))

    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.scatter(X.numpy(), Y.numpy())
    d2l.plt.xlabel('X')
    d2l.plt.ylabel('Y')
    d2l.plt.title(f'cor = {cors[i]}')
d2l.plt.show()
```

Hãy để chúng tôi liệt kê một vài thuộc tính của mối tương quan dưới đây. 

* Đối với bất kỳ biến ngẫu nhiên $X$, $\rho(X, X) = 1$.
* Đối với bất kỳ biến ngẫu nhiên $X, Y$ và số $a$ và $b$, $\rho(aX+b, Y) = \rho(X, aY+b) = \rho(X, Y)$.
* Nếu $X$ và $Y$ độc lập với phương sai không thì $\rho(X, Y) = 0$.

Như một lưu ý cuối cùng, bạn có thể cảm thấy như một số công thức này quen thuộc. Thật vậy, nếu chúng ta mở rộng mọi thứ ra giả định rằng $\mu_X = \mu_Y = 0$, chúng ta thấy rằng đây là 

$$
\rho(X, Y) = \frac{\sum_{i, j} x_iy_ip_{ij}}{\sqrt{\sum_{i, j}x_i^2 p_{ij}}\sqrt{\sum_{i, j}y_j^2 p_{ij}}}.
$$

Điều này trông giống như một tổng của một sản phẩm của các thuật ngữ chia cho căn bậc hai của các khoản tiền. Đây chính xác là công thức cho cosin của góc giữa hai vectơ $\mathbf{v}, \mathbf{w}$ với tọa độ khác nhau có trọng số bởi $p_{ij}$: 

$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

Thật vậy, nếu chúng ta nghĩ về các định mức là liên quan đến độ lệch chuẩn, và tương quan như là cosin của các góc, phần lớn trực giác chúng ta có từ hình học có thể được áp dụng để suy nghĩ về các biến ngẫu nhiên. 

## Tóm lượng* Biến ngẫu nhiên liên tục là các biến ngẫu nhiên có thể thực hiện một liên tục của các giá trị. Chúng có một số khó khăn kỹ thuật khiến chúng trở nên khó khăn hơn khi làm việc so với các biến ngẫu nhiên rời rạc * Hàm mật độ xác suất cho phép chúng ta làm việc với các biến ngẫu nhiên liên tục bằng cách đưa ra một hàm trong đó khu vực dưới đường cong trên một khoảng thời gian cho xác suất tìm thấy một điểm mẫu trong khoảng thời gian đó* Hàm phân phối tích lũy là xác suất quan sát biến ngẫu nhiên nhỏ hơn một ngưỡng nhất định. Nó có thể cung cấp một quan điểm thay thế hữu ích thống nhất các biến rời rạc và liên tục.* trung bình là giá trị trung bình của một biến ngẫu nhiên. * phương sai là bình phương dự kiến của sự khác biệt giữa biến ngẫu nhiên và trung bình của nó.* Độ lệch chuẩn là căn bậc hai của phương sai. Nó có thể được coi là đo phạm vi các giá trị mà biến ngẫu nhiên có thể mất. * Bất đẳng thức của Chebyshev cho phép chúng ta làm cho trực giác này nghiêm ngặt bằng cách đưa ra một khoảng rõ ràng chứa biến ngẫu nhiên hầu hết thời gian* Mật độ chung cho phép chúng ta làm việc với các biến ngẫu nhiên tương quan. Chúng ta có thể lề mật độ chung bằng cách tích hợp các biến ngẫu nhiên không mong muốn để có được sự phân bố của biến ngẫu nhiên mong muốn* hệ số đồng phương sai và hệ số tương quan cung cấp một cách để đo bất kỳ mối quan hệ tuyến tính nào giữa hai biến ngẫu nhiên tương quan. 

## Bài tập 1. Giả sử rằng chúng ta có biến ngẫu nhiên với mật độ được đưa ra bởi $p(x) = \frac{1}{x^2}$ cho $x \ge 1$ và $p(x) = 0$ nếu không. $P(X > 2)$ là cái gì? 2. Phân bố Laplace là một biến ngẫu nhiên có mật độ được cho bởi $p(x = \frac{1}{2}e^{-|x|}$. Ý nghĩa và độ lệch chuẩn của chức năng này là gì? Như một gợi ý, $\int_0^\infty xe^{-x} \; dx = 1$ và $\int_0^\infty x^2e^{-x} \; dx = 2$. Tôi đi bộ đến bạn trên đường phố và nói “Tôi có một biến ngẫu nhiên với trung bình $1$, độ lệch chuẩn $2$, và tôi quan sát $25\ %$ of my samples taking a value larger than $9$.” Bạn có tin tôi không? Tại sao hoặc tại sao không? 4. Giả sử rằng bạn có hai biến ngẫu nhiên $X, Y$, với mật độ khớp được đưa ra bởi $p_{XY}(x, y) = 4xy$ cho $x, y \in [0,1]$ và $p_{XY}(x, y) = 0$ nếu không. Sự đồng phương sai của $X$ và $Y$ là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/415)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1094)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1095)
:end_tab:

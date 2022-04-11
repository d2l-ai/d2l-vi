# Độ lồi
:label:`sec_convexity`

Lồi đóng một vai trò quan trọng trong việc thiết kế các thuật toán tối ưu hóa. Điều này phần lớn là do thực tế là nó dễ dàng hơn nhiều để phân tích và kiểm tra các thuật toán trong một bối cảnh như vậy. Nói cách khác, nếu thuật toán thực hiện kém ngay cả trong cài đặt lồi, thông thường chúng ta không nên hy vọng sẽ thấy kết quả tuyệt vời khác. Hơn nữa, mặc dù các vấn đề tối ưu hóa trong học sâu nói chung là không lồi, chúng thường thể hiện một số tính chất của các vấn đề lồi gần minima cục bộ. Điều này có thể dẫn đến các biến thể tối ưu hóa mới thú vị như :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
from mpl_toolkits import mplot3d
import tensorflow as tf
```

## Định nghĩa

Trước khi phân tích lồi, chúng ta cần xác định * bộ lồi * và * hàm lồi *. Chúng dẫn đến các công cụ toán học thường được áp dụng cho học máy. 

### Bộ lồi

Bộ là cơ sở của sự lồi lên. Nói một cách đơn giản, một bộ $\mathcal{X}$ trong một không gian vector là * lồi * nếu đối với bất kỳ $a, b \in \mathcal{X}$ đoạn đường nối $a$ và $b$ cũng là trong $\mathcal{X}$. Trong thuật ngữ toán học, điều này có nghĩa là đối với tất cả $\lambda \in [0, 1]$ chúng tôi có 

$$\lambda  a + (1-\lambda)  b \in \mathcal{X} \text{ whenever } a, b \in \mathcal{X}.$$

Điều này nghe có vẻ hơi trừu tượng. Hãy xem xét :numref:`fig_pacman`. Tập đầu tiên không lồi vì tồn tại các đoạn đường không chứa trong đó. Hai bộ còn lại không gặp vấn đề như vậy. 

![The first set is nonconvex and the other two are convex.](../img/pacman.svg)
:label:`fig_pacman`

Định nghĩa của riêng họ không đặc biệt hữu ích trừ khi bạn có thể làm điều gì đó với họ. Trong trường hợp này, chúng ta có thể xem xét các nút giao như thể hiện trong :numref:`fig_convex_intersect`. Giả sử rằng $\mathcal{X}$ và $\mathcal{Y}$ là các bộ lồi. Sau đó $\mathcal{X} \cap \mathcal{Y}$ cũng lồi. Để xem điều này, hãy xem xét bất kỳ $a, b \in \mathcal{X} \cap \mathcal{Y}$. Vì $\mathcal{X}$ và $\mathcal{Y}$ là lồi, các đoạn đường nối $a$ và $b$ được chứa trong cả $\mathcal{X}$ và $\mathcal{Y}$. Cho rằng, chúng cũng cần phải được chứa trong $\mathcal{X} \cap \mathcal{Y}$, do đó chứng minh định lý của chúng ta. 

![The intersection between two convex sets is convex.](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

Chúng ta có thể củng cố kết quả này với ít nỗ lực: cho bộ lồi $\mathcal{X}_i$, giao lộ của chúng $\cap_{i} \mathcal{X}_i$ là lồi. Để thấy rằng cuộc trò chuyện là không đúng sự thật, hãy xem xét hai bộ tách rời $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Bây giờ chọn $a \in \mathcal{X}$ và $b \in \mathcal{Y}$. Đoạn đường trong :numref:`fig_nonconvex` kết nối $a$ và $b$ cần chứa một số phần không phải trong $\mathcal{X}$ cũng không phải trong $\mathcal{Y}$, vì chúng tôi giả định rằng $\mathcal{X} \cap \mathcal{Y} = \emptyset$. Do đó đoạn đường cũng không nằm trong $\mathcal{X} \cup \mathcal{Y}$, do đó chứng minh rằng trong các công đoàn chung của bộ lồi không cần phải lồi. 

![The union of two convex sets need not be convex.](../img/nonconvex.svg)
:label:`fig_nonconvex`

Thông thường các vấn đề trong học sâu được xác định trên các bộ lồi. Ví dụ, $\mathbb{R}^d$, tập hợp các vectơ $d$ chiều của số thực, là một tập lồi (sau tất cả, đường giữa hai điểm bất kỳ trong $\mathbb{R}^d$ vẫn còn trong $\mathbb{R}^d$). Trong một số trường hợp, chúng tôi làm việc với các biến có độ dài giới hạn, chẳng hạn như các quả bóng bán kính $r$ theo định nghĩa bởi $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\| \leq r\}$. 

### Hàm lồi

Bây giờ chúng ta có bộ lồi, chúng ta có thể giới thiệu các chức năng lồi * $f$. Cho một bộ lồi $\mathcal{X}$, một chức năng $f: \mathcal{X} \to \mathbb{R}$ là * lồi * nếu cho tất cả $x, x' \in \mathcal{X}$ và cho tất cả $\lambda \in [0, 1]$ chúng tôi có 

$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$

Để minh họa điều này, chúng ta hãy vẽ một vài chức năng và kiểm tra cái nào đáp ứng yêu cầu. Dưới đây chúng tôi xác định một vài hàm, cả lồi và không lồi.

```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2  # Convex
g = lambda x: d2l.cos(np.pi * x)  # Nonconvex
h = lambda x: d2l.exp(0.5 * x)  # Convex

x, segment = d2l.arange(-2, 2, 0.01), d2l.tensor([-1.5, 1])
d2l.use_svg_display()
_, axes = d2l.plt.subplots(1, 3, figsize=(9, 3))
for ax, func in zip(axes, [f, g, h]):
    d2l.plot([x, segment], [func(x), func(segment)], axes=ax)
```

Đúng như dự đoán, hàm cosin là *nonconvex*, trong khi parabol và hàm mũ là. Lưu ý rằng yêu cầu $\mathcal{X}$ là một bộ lồi là cần thiết để điều kiện có ý nghĩa. Nếu không, kết quả của $f(\lambda x + (1-\lambda) x')$ có thể không được xác định rõ. 

### Sự bất bình đẳng của Jensen

Với hàm lồi $f$, một trong những công cụ toán học hữu ích nhất là sự bất bình đẳng của *Jensen*. Nó lên tới một khái quát hóa định nghĩa về độ lồi: 

$$\sum_i \alpha_i f(x_i)  \geq f\left(\sum_i \alpha_i x_i\right)    \text{ and }    E_X[f(X)]  \geq f\left(E_X[X]\right),$$
:eqlabel:`eq_jensens-inequality`

trong đó $\alpha_i$ là số thực không âm sao cho $\sum_i \alpha_i = 1$ và $X$ là một biến ngẫu nhiên. Nói cách khác, kỳ vọng của một hàm lồi không kém hàm lồi của một kỳ vọng, trong đó hàm sau thường là một biểu thức đơn giản hơn. Để chứng minh sự bất bình đẳng đầu tiên, chúng tôi liên tục áp dụng định nghĩa lồi cho một thuật ngữ trong tổng tại một thời điểm. 

Một trong những ứng dụng phổ biến của bất bình đẳng của Jensen là ràng buộc một biểu thức phức tạp hơn bằng một biểu thức đơn giản hơn. Ví dụ, ứng dụng của nó có thể liên quan đến khả năng đăng nhập của các biến ngẫu nhiên được quan sát một phần. Đó là, chúng tôi sử dụng 

$$E_{Y \sim P(Y)}[-\log P(X \mid Y)] \geq -\log P(X),$$

kể từ $\int P(Y) P(X \mid Y) dY = P(X)$. Điều này có thể được sử dụng trong các phương pháp biến thể. Ở đây $Y$ thường là biến ngẫu nhiên không quan sát được, $P(Y)$ là đoán tốt nhất về cách nó có thể được phân phối và $P(X)$ là phân phối với $Y$ tích hợp ra. Ví dụ, trong phân nhóm $Y$ có thể là nhãn cụm và $P(X \mid Y)$ là mô hình tạo khi áp dụng nhãn cụm. 

## Thuộc tính

Hàm lồi có nhiều tính chất hữu ích. Chúng tôi mô tả một vài cái được sử dụng phổ biến dưới đây. 

### Minima địa phương là Minima toàn cầu

Đầu tiên và quan trọng nhất, minima cục bộ của các hàm lồi cũng là minima toàn cầu. Chúng ta có thể chứng minh điều đó bằng mâu thuẫn như sau. 

Hãy xem xét một hàm lồi $f$ được xác định trên một bộ lồi $\mathcal{X}$. Giả sử rằng $x^{\ast} \in \mathcal{X}$ là mức tối thiểu cục bộ: tồn tại một giá trị dương nhỏ $p$ sao cho $x \in \mathcal{X}$ thỏa mãn $0 < |x - x^{\ast}| \leq p$ chúng ta có $f(x^{\ast}) < f(x)$. 

Giả sử rằng mức tối thiểu địa phương $x^{\ast}$ không phải là minum um toàn cầu của $f$: có tồn tại $x' \in \mathcal{X}$ mà $f(x') < f(x^{\ast})$. Cũng tồn tại $\lambda \in [0, 1)$ như $\lambda = 1 - \frac{p}{|x^{\ast} - x'|}$ sao cho $0 < |\lambda x^{\ast} + (1-\lambda) x' - x^{\ast}| \leq p$.  

Tuy nhiên, theo định nghĩa của các hàm lồi, ta có 

$$\begin{aligned}
    f(\lambda x^{\ast} + (1-\lambda) x') &\leq \lambda f(x^{\ast}) + (1-\lambda) f(x') \\
    &< \lambda f(x^{\ast}) + (1-\lambda) f(x^{\ast}) \\
    &= f(x^{\ast}),
\end{aligned}$$

mâu thuẫn với tuyên bố của chúng tôi rằng $x^{\ast}$ là mức tối thiểu địa phương. Do đó, không tồn tại $x' \in \mathcal{X}$ mà $f(x') < f(x^{\ast})$. Mức tối thiểu địa phương $x^{\ast}$ cũng là mức tối thiểu toàn cầu. 

Ví dụ, hàm lồi $f(x) = (x-1)^2$ có mức tối thiểu cục bộ là $x=1$, đây cũng là mức tối thiểu toàn cầu.

```{.python .input}
#@tab all
f = lambda x: (x - 1) ** 2
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```

Thực tế là minima cục bộ cho các chức năng lồi cũng là minima toàn cầu rất thuận tiện. Điều đó có nghĩa là nếu chúng ta giảm thiểu các chức năng, chúng ta không thể “gặp khó khăn”. Tuy nhiên, lưu ý rằng điều này không có nghĩa là không thể có nhiều hơn một mức tối thiểu toàn cầu hoặc thậm chí có thể tồn tại. Ví dụ, hàm $f(x) = \mathrm{max}(|x|-1, 0)$ đạt được giá trị tối thiểu của nó trong khoảng $[-1, 1]$. Ngược lại, chức năng $f(x) = \exp(x)$ không đạt được giá trị tối thiểu trên $\mathbb{R}$: đối với $x \to -\infty$ nó asymptotes đến $0$, nhưng không có $x$ mà $f(x) = 0$. 

### Các bộ hàm lồi dưới đây là lồi

Chúng tôi có thể thuận tiện xác định các bộ lồi thông qua * dưới bộ* của hàm lồi. Cụ thể, đưa ra một hàm lồi $f$ được xác định trên một bộ lồi $\mathcal{X}$, bất kỳ thiết lập dưới đây 

$$\mathcal{S}_b := \{x | x \in \mathcal{X} \text{ and } f(x) \leq b\}$$

là lồi.  

Hãy để chúng tôi chứng minh điều này một cách nhanh chóng. Nhớ lại rằng đối với bất kỳ $x, x' \in \mathcal{S}_b$ chúng ta cần phải cho thấy rằng $\lambda x + (1-\lambda) x' \in \mathcal{S}_b$ miễn là $\lambda \in [0, 1]$. Kể từ $f(x) \leq b$ và $f(x') \leq b$, theo định nghĩa về độ lồi chúng ta có  

$$f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b.$$

### Lồi và dẫn xuất thứ hai

Bất cứ khi nào đạo hàm thứ hai của một hàm $f: \mathbb{R}^n \rightarrow \mathbb{R}$ tồn tại thì rất dễ dàng để kiểm tra xem $f$ có lồi hay không. Tất cả những gì chúng ta cần làm là kiểm tra xem Hessian của $f$ là semidefinite dương: $\nabla^2f \succeq 0$, tức là, biểu thị ma trận Hessian $\nabla^2f$ bởi $\mathbf{H}$, $\mathbf{x}^\top \mathbf{H} \mathbf{x} \geq 0$ cho tất cả $\mathbf{x} \in \mathbb{R}^n$. Ví dụ, hàm $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2$ là lồi từ $\nabla^2 f = \mathbf{1}$, tức là, Hessian của nó là một ma trận nhận dạng. 

Về mặt chính thức, một hàm một chiều có thể phân biệt hai lần $f: \mathbb{R} \rightarrow \mathbb{R}$ là lồi nếu và chỉ khi đạo hàm thứ hai của nó $f'' \geq 0$. Đối với bất kỳ chức năng đa chiều hai lần khác biệt $f: \mathbb{R}^{n} \rightarrow \mathbb{R}$, nó là lồi nếu và chỉ khi Hessian $\nabla^2f \succeq 0$ của nó. 

Đầu tiên, chúng ta cần chứng minh trường hợp một chiều. Để thấy rằng lồi của $f$ ngụ ý $f'' \geq 0$ chúng tôi sử dụng thực tế là 

$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$

Vì đạo hàm thứ hai được đưa ra bởi giới hạn trên sự khác biệt hữu hạn nó sau đó 

$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$

Để thấy rằng $f'' \geq 0$ ngụ ý rằng $f$ là lồi, chúng tôi sử dụng thực tế là $f'' \geq 0$ ngụ ý rằng $f'$ là một chức năng không giảm đơn điệu. Hãy để $a < x < b$ được ba điểm trong $\mathbb{R}$, nơi $x = (1-\lambda)a + \lambda b$ và $\lambda \in (0, 1)$. Theo định lý giá trị trung bình, có tồn tại $\alpha \in [a, x]$ và $\beta \in [x, b]$ sao cho 

$$f'(\alpha) = \frac{f(x) - f(a)}{x-a} \text{ and } f'(\beta) = \frac{f(b) - f(x)}{b-x}.$$

Bởi monotonicity $f'(\beta) \geq f'(\alpha)$, do đó 

$$\frac{x-a}{b-a}f(b) + \frac{b-x}{b-a}f(a) \geq f(x).$$

Kể từ khi $x = (1-\lambda)a + \lambda b$, chúng tôi có 

$$\lambda f(b) + (1-\lambda)f(a) \geq f((1-\lambda)a + \lambda b),$$

thy như vậy proving chứng minh lồi. 

Thứ hai, chúng ta cần một đề tài trước khi chứng minh trường hợp đa chiều: $f: \mathbb{R}^n \rightarrow \mathbb{R}$ là lồi nếu và chỉ khi cho tất cả $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 

$$g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y}) \text{ where } z \in [0,1]$$ 

là lồi. 

Để chứng minh rằng độ lồi của $f$ ngụ ý rằng $g$ là lồi, chúng ta có thể chỉ ra rằng đối với tất cả $a, b, \lambda \in [0, 1]$ (do đó $0 \leq \lambda a + (1-\lambda) b \leq 1$) 

$$\begin{aligned} &g(\lambda a + (1-\lambda) b)\\
=&f\left(\left(\lambda a + (1-\lambda) b\right)\mathbf{x} + \left(1-\lambda a - (1-\lambda) b\right)\mathbf{y} \right)\\
=&f\left(\lambda \left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) \left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \right)\\
\leq& \lambda f\left(a \mathbf{x} + (1-a)  \mathbf{y}\right)  + (1-\lambda) f\left(b \mathbf{x} + (1-b)  \mathbf{y}\right) \\
=& \lambda g(a) + (1-\lambda) g(b).
\end{aligned}$$

Để chứng minh cuộc trò chuyện, chúng ta có thể chỉ ra rằng cho tất cả $\lambda \in [0, 1]$  

$$\begin{aligned} &f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y})\\
=&g(\lambda \cdot 1 + (1-\lambda) \cdot 0)\\
\leq& \lambda g(1)  + (1-\lambda) g(0) \\
=& \lambda f(\mathbf{x}) + (1-\lambda) g(\mathbf{y}).
\end{aligned}$$

Cuối cùng, sử dụng lemma ở trên và kết quả của trường hợp một chiều, trường hợp đa chiều có thể được chứng minh như sau. Một chức năng đa chiều $f: \mathbb{R}^n \rightarrow \mathbb{R}$ là lồi nếu và chỉ khi cho tất cả $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ $g(z) \stackrel{\mathrm{def}}{=} f(z \mathbf{x} + (1-z)  \mathbf{y})$, trong đó $z \in [0,1]$, là lồi. Theo trường hợp một chiều, điều này giữ nếu và chỉ khi $g'' = (\mathbf{x} - \mathbf{y})^\top \mathbf{H}(\mathbf{x} - \mathbf{y}) \geq 0$ ($\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2f$) cho tất cả $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$, tương đương với $\mathbf{H} \succeq 0$ trên mỗi định nghĩa của ma trận bán xác định dương. 

## Ràng buộc

Một trong những tính chất tốt đẹp của tối ưu hóa lồi là nó cho phép chúng ta xử lý các ràng buộc một cách hiệu quả. Đó là, nó cho phép chúng tôi giải quyết các vấn đề tối ưu hóa hạn chế* của biểu mẫu: 

$$\begin{aligned} \mathop{\mathrm{minimize~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{ subject to } & c_i(\mathbf{x}) \leq 0 \text{ for all } i \in \{1, \ldots, n\},
\end{aligned}$$

trong đó $f$ là mục tiêu và các chức năng $c_i$ là các hàm hạn chế. Để xem điều này xem xét những gì trường hợp $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. Trong trường hợp này, các thông số $\mathbf{x}$ được hạn chế vào quả bóng đơn vị. Nếu một ràng buộc thứ hai là $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$, thì điều này tương ứng với tất cả $\mathbf{x}$ nằm trên một nửa không gian. Thỏa mãn cả hai ràng buộc đồng thời số tiền để lựa chọn một lát của một quả bóng. 

### Lagrangian

Nói chung, việc giải quyết một vấn đề tối ưu hóa bị hạn chế là khó khăn. Một cách để giải quyết nó bắt nguồn từ vật lý với một trực giác khá đơn giản. Hãy tưởng tượng một quả bóng bên trong một hộp. Quả bóng sẽ lăn đến nơi thấp nhất và lực hấp dẫn sẽ được cân bằng với các lực mà các cạnh của hộp có thể áp đặt lên quả bóng. Nói tóm lại, gradient của hàm khách quan (tức là trọng lực) sẽ được bù đắp bởi gradient của hàm ràng buộc (quả bóng cần phải ở lại bên trong hộp nhờ đức hạnh của các bức tường “đẩy lùi”). Lưu ý rằng một số hạn chế có thể không hoạt động: các bức tường không được chạm vào bởi quả bóng sẽ không thể tác dụng bất kỳ lực nào lên quả bóng. 

Bỏ qua nguồn gốc của * Lagrangian* $L$, lý luận trên có thể được thể hiện thông qua bài toán tối ưu hóa điểm yên sau: 

$$L(\mathbf{x}, \alpha_1, \ldots, \alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_i c_i(\mathbf{x}) \text{ where } \alpha_i \geq 0.$$

Ở đây các biến $\alpha_i$ ($i=1,\ldots,n$) là cái gọi là hệ số nhân * Lagrange* đảm bảo rằng các ràng buộc được thực thi đúng cách. Chúng được chọn vừa đủ lớn để đảm bảo rằng $c_i(\mathbf{x}) \leq 0$ cho tất cả $i$. Ví dụ, đối với bất kỳ $\mathbf{x}$ nơi $c_i(\mathbf{x}) < 0$ một cách tự nhiên, chúng tôi sẽ chọn $\alpha_i = 0$. Hơn nữa, đây là một vấn đề tối ưu hóa điểm yên ngựa mà người ta muốn * tối đa hóa* $L$ đối với tất cả $\alpha_i$ và đồng thời * giảm xỉnh* nó đối với $\mathbf{x}$. Có một cơ thể văn học phong phú giải thích làm thế nào để đến chức năng $L(\mathbf{x}, \alpha_1, \ldots, \alpha_n)$. Đối với mục đích của chúng tôi, nó là đủ để biết rằng điểm yên của $L$ là nơi mà vấn đề tối ưu hóa hạn chế ban đầu được giải quyết một cách tối ưu. 

### Hình phạt

Một cách để đáp ứng các vấn đề tối ưu hóa hạn chế ít nhất * gần đúng * là thích ứng với Lagrangian $L$. Thay vì thỏa mãn $c_i(\mathbf{x}) \leq 0$, chúng tôi chỉ cần thêm $\alpha_i c_i(\mathbf{x})$ vào hàm khách quan $f(x)$. Điều này đảm bảo rằng các ràng buộc sẽ không bị vi phạm quá nặng. 

Trong thực tế, chúng tôi đã sử dụng thủ thuật này tất cả cùng. Xem xét phân rã cân nặng năm :numref:`sec_weight_decay`. Trong đó, chúng tôi thêm $\frac{\lambda}{2} \|\mathbf{w}\|^2$ vào chức năng khách quan để đảm bảo rằng $\mathbf{w}$ không phát triển quá lớn. Từ quan điểm tối ưu hóa bị hạn chế, chúng ta có thể thấy rằng điều này sẽ đảm bảo rằng $\|\mathbf{w}\|^2 - r^2 \leq 0$ cho một số bán kính $r$. Điều chỉnh giá trị của $\lambda$ cho phép chúng tôi thay đổi kích thước của $\mathbf{w}$. 

Nói chung, thêm hình phạt là một cách tốt để đảm bảo sự hài lòng hạn chế gần đúng. Trong thực tế, điều này hóa ra mạnh mẽ hơn nhiều so với sự hài lòng chính xác. Hơn nữa, đối với các vấn đề không lồi, nhiều thuộc tính làm cho cách tiếp cận chính xác trở nên hấp dẫn trong trường hợp lồi (ví dụ, tối ưu) không còn giữ được nữa. 

### Dự báo

Một chiến lược thay thế để thỏa mãn các ràng buộc là dự đoán. Một lần nữa, chúng tôi đã gặp chúng trước đây, ví dụ, khi xử lý cắt gradient trong :numref:`sec_rnn_scratch`. Ở đó chúng tôi đảm bảo rằng một gradient có chiều dài giới hạn bởi $\theta$ qua 

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, \theta/\|\mathbf{g}\|).$$

Điều này hóa ra là một * dự áo* của $\mathbf{g}$ lên quả bóng bán kính $\theta$. Nói chung hơn, một phép chiếu trên một bộ lồi $\mathcal{X}$ được định nghĩa là 

$$\mathrm{Proj}_\mathcal{X}(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in \mathcal{X}} \|\mathbf{x} - \mathbf{x}'\|,$$

đó là điểm gần nhất trong $\mathcal{X}$ đến $\mathbf{x}$.  

![Convex Projections.](../img/projections.svg)
:label:`fig_projections`

Định nghĩa toán học của các dự báo có vẻ hơi trừu tượng. :numref:`fig_projections` giải thích nó một chút rõ ràng hơn. Trong đó chúng ta có hai bộ lồi, một vòng tròn và một viên kim cương. Các điểm bên trong cả hai bộ (màu vàng) vẫn không thay đổi trong quá trình chiếu. Các điểm bên ngoài cả hai bộ (màu đen) được chiếu vào các điểm bên trong các bộ (màu đỏ) là tủ quần áo đến các điểm ban đầu (màu đen). Trong khi đối với $L_2$ quả bóng này để lại hướng không thay đổi, điều này không cần phải là trường hợp nói chung, như có thể thấy trong trường hợp của kim cương. 

Một trong những công dụng cho các phép chiếu lồi là tính toán các vectơ trọng lượng thưa thớt. Trong trường hợp này, chúng tôi dự án vectơ trọng lượng lên một quả bóng $L_1$, là một phiên bản tổng quát của vỏ kim cương trong :numref:`fig_projections`. 

## Tóm tắt

Trong bối cảnh học sâu, mục đích chính của các hàm lồi là thúc đẩy các thuật toán tối ưu hóa và giúp chúng ta hiểu chi tiết chúng. Trong phần sau đây, chúng ta sẽ thấy độ dốc và gốc gradient ngẫu nhiên có thể được bắt nguồn cho phù hợp như thế nào. 

* Giao điểm của bộ lồi lồi là lồi. Đoàn thể thì không.
* Kỳ vọng của một hàm lồi không kém hàm lồi của một kỳ vọng (bất bình đẳng của Jensen).
* Một hàm phân biệt hai lần là lồi nếu và chỉ khi Hessian của nó (một ma trận của các dẫn xuất thứ hai) là semidefinite dương.
* Ràng buộc lồi có thể được thêm vào thông qua Lagrangian. Trong thực tế, chúng ta có thể chỉ cần thêm chúng với một hình phạt cho chức năng khách quan.
* Các hình chiếu ánh xạ đến các điểm trong bộ lồi gần nhất với các điểm ban đầu.

## Bài tập

1. Giả sử rằng chúng ta muốn xác minh độ lồi của một tập hợp bằng cách vẽ tất cả các đường giữa các điểm trong tập hợp và kiểm tra xem các dòng có chứa hay không.
    1. Chứng minh rằng nó là đủ để chỉ kiểm tra các điểm trên ranh giới.
    1. Chứng minh rằng nó là đủ để chỉ kiểm tra các đỉnh của tập hợp.
1. Biểu thị bằng $\mathcal{B}_p[r] \stackrel{\mathrm{def}}{=} \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ bóng bán kính $r$ bằng cách sử dụng $p$-chuẩn. Chứng minh rằng $\mathcal{B}_p[r]$ là lồi cho tất cả $p \geq 1$.
1. Cho hàm lồi $f$ và $g$, cho thấy $\mathrm{max}(f, g)$ cũng lồi. Chứng minh rằng $\mathrm{min}(f, g)$ không lồi.
1. Chứng minh rằng việc bình thường hóa chức năng softmax là lồi. Cụ thể hơn chứng minh sự lồi của $f(x) = \log \sum_i \exp(x_i)$.
1. Chứng minh rằng không gian con tuyến tính, tức là, $\mathcal{X} = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$, là các bộ lồi.
1. Chứng minh rằng trong trường hợp không gian con tuyến tính với $\mathbf{b} = \mathbf{0}$ phép chiếu $\mathrm{Proj}_\mathcal{X}$ có thể được viết là $\mathbf{M} \mathbf{x}$ cho một số ma trận $\mathbf{M}$.
1. Cho thấy rằng đối với các hàm lồi hai lần khác biệt $f$ chúng ta có thể viết $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ cho một số $\xi \in [0, \epsilon]$.
1. Đưa ra một vector $\mathbf{w} \in \mathbb{R}^d$ với $\|\mathbf{w}\|_1 > 1$ tính toán phép chiếu trên quả bóng đơn vị $L_1$.
    1. Như một bước trung gian viết ra mục tiêu bị phạt $\|\mathbf{w} - \mathbf{w}'\|^2 + \lambda \|\mathbf{w}'\|_1$ và tính toán các giải pháp cho một $\lambda > 0$ nhất định.
    1. Bạn có thể tìm thấy giá trị “đúng” của $\lambda$ mà không có nhiều thử và sai không?
1. Cho một bộ lồi $\mathcal{X}$ và hai vectơ $\mathbf{x}$ và $\mathbf{y}$, chứng minh rằng các dự báo không bao giờ tăng khoảng cách, tức là $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_\mathcal{X}(\mathbf{x}) - \mathrm{Proj}_\mathcal{X}(\mathbf{y})\|$.

[Discussions](https://discuss.d2l.ai/t/350)

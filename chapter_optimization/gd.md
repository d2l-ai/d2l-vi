# Gradient Descent
:label:`sec_gd`

Trong phần này, chúng ta sẽ giới thiệu các khái niệm cơ bản bên dưới * gradient descent*. Mặc dù nó hiếm khi được sử dụng trực tiếp trong học sâu, một sự hiểu biết về gradient descent là chìa khóa để hiểu các thuật toán gốc gradient ngẫu nhiên. Ví dụ, vấn đề tối ưu hóa có thể phân kỳ do tốc độ học tập quá lớn. Hiện tượng này đã có thể được nhìn thấy trong gradient gốc. Tương tự như vậy, preconditioning là một kỹ thuật phổ biến trong chuyển đổi gradient và mang đến các thuật toán tiên tiến hơn. Hãy để chúng tôi bắt đầu với một trường hợp đặc biệt đơn giản. 

## Một chiều Gradient Descent

Gradient gốc trong một chiều là một ví dụ tuyệt vời để giải thích lý do tại sao thuật toán gốc gradient có thể làm giảm giá trị của hàm khách quan. Xem xét một số chức năng có giá trị thực liên tục khác biệt $f: \mathbb{R} \rightarrow \mathbb{R}$. Sử dụng bản mở rộng Taylor, chúng tôi có được 

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

Đó là, trong xấp xỉ thứ tự thứ nhất $f(x+\epsilon)$ được đưa ra bởi giá trị hàm $f(x)$ và đạo hàm đầu tiên $f'(x)$ tại $x$. Nó không phải là không hợp lý để giả định rằng đối với nhỏ $\epsilon$ di chuyển theo hướng của gradient âm sẽ giảm $f$. Để giữ cho mọi thứ đơn giản, chúng tôi chọn một kích thước bước cố định $\eta > 0$ và chọn $\epsilon = -\eta f'(x)$. Cắm điều này vào bản mở rộng Taylor ở trên chúng tôi nhận được 

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$
:eqlabel:`gd-taylor-2`

Nếu đạo hàm $f'(x) \neq 0$ không biến mất chúng ta tiến bộ kể từ $\eta f'^2(x)>0$. Hơn nữa, chúng tôi luôn có thể chọn $\eta$ đủ nhỏ để các điều khoản bậc cao hơn trở nên không liên quan. Do đó chúng tôi đến 

$$f(x - \eta f'(x)) \lessapprox f(x).$$

Điều này có nghĩa rằng, nếu chúng ta sử dụng 

$$x \leftarrow x - \eta f'(x)$$

để lặp lại $x$, giá trị của hàm $f(x)$ có thể giảm. Do đó, trong gradient descent đầu tiên chúng ta chọn một giá trị ban đầu $x$ và một hằng số $\eta > 0$ và sau đó sử dụng chúng để liên tục lặp lại $x$ cho đến khi đạt đến điều kiện dừng, ví dụ, khi độ lớn của gradient $|f'(x)|$ là đủ nhỏ hoặc số lần lặp đã đạt đến một số nhất định giá trị. 

Để đơn giản, chúng tôi chọn hàm mục tiêu $f(x)=x^2$ để minh họa cách thực hiện gradient descent. Mặc dù chúng ta biết rằng $x=0$ là giải pháp để giảm thiểu $f(x)$, chúng tôi vẫn sử dụng chức năng đơn giản này để quan sát cách $x$ thay đổi.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

Tiếp theo, chúng tôi sử dụng $x=10$ làm giá trị ban đầu và giả sử $\eta=0.2$. Sử dụng gradient descent để lặp lại $x$ trong 10 lần chúng ta có thể thấy rằng, cuối cùng, giá trị của $x$ tiếp cận giải pháp tối ưu.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

Tiến trình tối ưu hóa hơn $x$ có thể được vẽ như sau.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Tỷ lệ học tập
:label:`subsec_gd-learningrate`

Tỷ lệ học tập $\eta$ có thể được thiết lập bởi nhà thiết kế thuật toán. Nếu chúng ta sử dụng tốc độ học tập quá nhỏ, nó sẽ khiến $x$ cập nhật rất chậm, đòi hỏi nhiều lần lặp lại hơn để có được giải pháp tốt hơn. Để hiển thị những gì xảy ra trong trường hợp như vậy, hãy xem xét tiến trình trong cùng một vấn đề tối ưu hóa cho $\eta = 0.05$. Như chúng ta có thể thấy, ngay cả sau 10 bước, chúng ta vẫn còn rất xa giải pháp tối ưu.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Ngược lại, nếu chúng ta sử dụng tỷ lệ học tập quá cao, $\left|\eta f'(x)\right|$ có thể quá lớn đối với công thức mở rộng Taylor bậc nhất. Đó là, thuật ngữ $\mathcal{O}(\eta^2 f'^2(x))$ trong :eqref:`gd-taylor-2` có thể trở nên quan trọng. Trong trường hợp này, chúng tôi không thể đảm bảo rằng việc lặp lại $x$ sẽ có thể hạ giá trị $f(x)$. Ví dụ, khi chúng ta đặt tỷ lệ học tập thành $\eta=1.1$, $x$ vượt qua giải pháp tối ưu $x=0$ và dần dần phân kỳ.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Minima địa phương

Để minh họa những gì xảy ra cho các chức năng không lồi xem xét trường hợp của $f(x) = x \cdot \cos(cx)$ cho một số hằng số $c$. Chức năng này có vô hạn nhiều minima cục bộ. Tùy thuộc vào sự lựa chọn của chúng tôi về tỷ lệ học tập và tùy thuộc vào mức độ điều kiện của vấn đề, chúng tôi có thể kết thúc với một trong nhiều giải pháp. Ví dụ dưới đây minh họa mức độ học tập cao (không thực tế) sẽ dẫn đến mức tối thiểu địa phương kém.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Đa biến Gradient Descent

Bây giờ chúng ta có một trực giác tốt hơn về trường hợp thống nhất, chúng ta hãy xem xét tình huống mà $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$. Đó là, hàm khách quan $f: \mathbb{R}^d \to \mathbb{R}$ ánh xạ vectơ thành vô hướng. Tương ứng gradient của nó là đa biến, quá. Nó là một vectơ gồm $d$ dẫn xuất từng phần: 

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$

Mỗi phần tử phái sinh một phần $\partial f(\mathbf{x})/\partial x_i$ trong gradient cho biết tốc độ thay đổi của $f$ tại $\mathbf{x}$ đối với đầu vào $x_i$. Như trước đây trong trường hợp thống nhất, chúng ta có thể sử dụng xấp xỉ Taylor tương ứng cho các chức năng đa biến để có được một số ý tưởng về những gì chúng ta nên làm. Đặc biệt, chúng tôi có điều đó 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`

Nói cách khác, lên đến các thuật ngữ bậc hai trong $\boldsymbol{\epsilon}$ hướng xuống dốc nhất được đưa ra bởi gradient âm $-\nabla f(\mathbf{x})$. Chọn một tỷ lệ học tập phù hợp $\eta > 0$ mang lại thuật toán gốc gradient nguyên mẫu: 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$

Để xem cách thuật toán hoạt động trong thực tế chúng ta hãy xây dựng một hàm khách quan $f(\mathbf{x})=x_1^2+2x_2^2$ với một vector hai chiều $\mathbf{x} = [x_1, x_2]^\top$ như đầu vào và vô hướng như đầu ra. Gradient được đưa ra bởi $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$. Chúng ta sẽ quan sát quỹ đạo của $\mathbf{x}$ bằng cách giảm gradient từ vị trí ban đầu $[-5, -2]$.  

Để bắt đầu, chúng ta cần thêm hai chức năng trợ giúp. Đầu tiên sử dụng chức năng cập nhật và áp dụng nó 20 lần cho giá trị ban đầu. Người trợ giúp thứ hai hình dung quỹ đạo của $\mathbf{x}$.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used later
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Tiếp theo, chúng ta quan sát quỹ đạo của biến tối ưu hóa $\mathbf{x}$ cho tỷ lệ học tập $\eta = 0.1$. Chúng ta có thể thấy rằng sau 20 bước, giá trị của $\mathbf{x}$ đạt đến mức tối thiểu của nó ở mức $[0, 0]$. Tiến bộ khá cư xử tốt mặc dù khá chậm.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Phương pháp thích ứng

Như chúng ta có thể thấy trong :numref:`subsec_gd-learningrate`, nhận được tỷ lệ học tập $\eta$ “vừa phải” là khó khăn. Nếu chúng ta chọn nó quá nhỏ, chúng ta sẽ tiến bộ rất ít. Nếu chúng ta chọn nó quá lớn, giải pháp dao động và trong trường hợp xấu nhất nó thậm chí có thể phân kỳ. Điều gì sẽ xảy ra nếu chúng ta có thể xác định $\eta$ tự động hoặc thoát khỏi việc phải chọn một tỷ lệ học tập ở tất cả? Các phương thức thứ hai không chỉ nhìn vào giá trị và gradient của hàm mục tiêu mà còn ở độ cong * của nócó thể giúp ích trong trường hợp này. Mặc dù các phương pháp này không thể áp dụng trực tiếp vào deep learning do chi phí tính toán, chúng cung cấp trực giác hữu ích về cách thiết kế các thuật toán tối ưu hóa nâng cao bắt chước nhiều thuộc tính mong muốn của các thuật toán được nêu dưới đây. 

### Phương pháp Newton

Xem xét việc mở rộng Taylor của một số chức năng $f: \mathbb{R}^d \rightarrow \mathbb{R}$ không cần phải dừng lại sau nhiệm kỳ đầu tiên. Trong thực tế, chúng ta có thể viết nó như 

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`

Để tránh ký hiệu rườm rà, chúng tôi định nghĩa $\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$ là Hessian của $f$, là ma trận $d \times d$. Đối với $d$ nhỏ và các vấn đề đơn giản $\mathbf{H}$ rất dễ tính toán. Đối với các mạng thần kinh sâu, mặt khác, $\mathbf{H}$ có thể rất lớn, do chi phí lưu trữ $\mathcal{O}(d^2)$ mục. Hơn nữa nó có thể quá tốn kém để tính toán thông qua backpropagation. Bây giờ chúng ta hãy bỏ qua những cân nhắc như vậy và nhìn vào thuật toán nào chúng ta sẽ nhận được. 

Rốt cuộc, tối thiểu $f$ thỏa mãn $\nabla f = 0$. Tuân theo các quy tắc giải tích trong :numref:`subsec_calculus-grad`, bằng cách dùng các dẫn xuất của :eqref:`gd-hot-taylor` liên quan đến $\boldsymbol{\epsilon}$ và bỏ qua các điều khoản bậc cao hơn, chúng tôi đến 

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

Đó là, chúng ta cần đảo ngược Hessian $\mathbf{H}$ như một phần của vấn đề tối ưu hóa. 

Như một ví dụ đơn giản, đối với $f(x) = \frac{1}{2} x^2$, chúng tôi có $\nabla f(x) = x$ và $\mathbf{H} = 1$. Do đó đối với bất kỳ $x$ chúng tôi có được $\epsilon = -x$. Nói cách khác, bước * single* là đủ để hội tụ hoàn hảo mà không cần bất kỳ điều chỉnh nào! Than ôi, chúng tôi đã có một chút may mắn ở đây: bản mở rộng Taylor là chính xác kể từ $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$.  

Hãy để chúng tôi xem những gì xảy ra trong các vấn đề khác. Với hàm cosin hyperbol lồi $f(x) = \cosh(cx)$ cho một số hằng số $c$, chúng ta có thể thấy rằng mức tối thiểu toàn cầu tại $x=0$ đạt được sau một vài lần lặp lại.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Bây giờ chúng ta hãy xem xét một hàm *nonconvex*, chẳng hạn như $f(x) = x \cos(c x)$ cho một số hằng số $c$. Rốt cuộc, lưu ý rằng trong phương pháp của Newton, chúng ta sẽ chia cho Hessian. Điều này có nghĩa là nếu đạo hàm thứ hai là *tiêu cực* chúng ta có thể đi theo hướng * tăng* giá trị của $f$. Đó là một lỗ hổng gây tử vong của thuật toán. Hãy để chúng tôi xem những gì xảy ra trong thực tế.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

Điều này đã đi sai một cách ngoạn mục. Làm thế nào chúng ta có thể sửa nó? Một cách sẽ là “sửa chữa” Hessian bằng cách lấy giá trị tuyệt đối của nó thay thế. Một chiến lược khác là mang lại tốc độ học tập. Điều này dường như đánh bại mục đích, nhưng không hoàn toàn. Có thông tin thứ hai cho phép chúng ta thận trọng bất cứ khi nào độ cong lớn và thực hiện các bước lâu hơn bất cứ khi nào chức năng khách quan phẳng hơn. Hãy để chúng tôi xem cách điều này hoạt động với tốc độ học tập nhỏ hơn một chút, nói $\eta = 0.5$. Như chúng ta có thể thấy, chúng ta có một thuật toán khá hiệu quả.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Phân tích hội tụ

Chúng tôi chỉ phân tích tốc độ hội tụ của phương pháp Newton đối với một số hàm khách quan lồi và ba lần khác biệt $f$, trong đó đạo hàm thứ hai là nonzero, tức là $f'' > 0$. Bằng chứng đa biến là một phần mở rộng đơn giản của đối số một chiều bên dưới và bỏ qua vì nó không giúp chúng ta nhiều về trực giác. 

Biểu thị bằng $x^{(k)}$ giá trị của $x$ tại lần lặp $k^\mathrm{th}$ và để $e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$ là khoảng cách từ tối ưu tại lần lặp $k^\mathrm{th}$. Bằng cách mở rộng Taylor, chúng tôi có rằng điều kiện $f'(x^*) = 0$ có thể được viết là 

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$

which mà holds giữ for some $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$. Chia việc mở rộng trên cho sản lượng $f''(x^{(k)})$ 

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$

Nhớ lại rằng chúng tôi có bản cập nhật $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$. Cắm vào phương trình cập nhật này và lấy giá trị tuyệt đối của cả hai bên, chúng ta có 

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$

Do đó, bất cứ khi nào chúng tôi đang ở trong một khu vực có giới hạn $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$, chúng tôi có một lỗi giảm bốn lần  

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$

Ngoài ra, các nhà nghiên cứu tối ưu hóa gọi sự hội tụ * tuyến tính* này, trong khi một điều kiện như $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ sẽ được gọi là tốc độ hội tụ * không đổi*. Lưu ý rằng phân tích này đi kèm với một số cảnh báo. Đầu tiên, chúng tôi không thực sự có nhiều sự đảm bảo khi chúng tôi sẽ đạt được khu vực hội tụ nhanh chóng. Thay vào đó, chúng ta chỉ biết rằng một khi chúng ta đạt được nó, sự hội tụ sẽ rất nhanh chóng. Thứ hai, phân tích này đòi hỏi $f$ được cư xử tốt đến các dẫn xuất bậc cao hơn. Nó đi xuống để đảm bảo rằng $f$ không có bất kỳ thuộc tính “đáng ngạc nhiên” nào về cách nó có thể thay đổi giá trị của nó. 

### Điều hòa trước

Khá không ngạc nhiên khi tính toán và lưu trữ Hessian đầy đủ là rất tốn kém. Do đó, nó là mong muốn để tìm lựa chọn thay thế. Một cách để cải thiện vấn đề là * điều kiện tiên chuẩn*. Nó tránh tính toán toàn bộ Hessian nhưng chỉ tính toán các mục nhập *diagonal*. Điều này dẫn đến cập nhật các thuật toán của biểu mẫu 

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$

Mặc dù điều này không hoàn toàn tốt bằng phương pháp Newton đầy đủ, nhưng nó vẫn tốt hơn nhiều so với việc không sử dụng nó. Để xem lý do tại sao đây có thể là một ý tưởng tốt, hãy xem xét một tình huống mà một biến biểu thị chiều cao tính bằng milimét và cái kia biểu thị chiều cao tính bằng km. Giả sử rằng đối với cả hai quy mô tự nhiên đều tính bằng mét, chúng ta có một sự không phù hợp khủng khiếp trong các tham số hóa. May mắn thay, sử dụng preconditioning loại bỏ điều này. Điều hòa trước hiệu quả với số lượng gốc gradient để lựa chọn một tốc độ học tập khác nhau cho mỗi biến (tọa độ của vector $\mathbf{x}$). Như chúng ta sẽ thấy sau này, điều hòa trước thúc đẩy một số sự đổi mới trong các thuật toán tối ưu hóa dòng dốc ngẫu nhiên.  

### Gradient Descent với tìm kiếm dòng

Một trong những vấn đề chính trong gradient gốc là chúng ta có thể vượt qua mục tiêu hoặc không đủ tiến bộ. Một sửa chữa đơn giản cho vấn đề là sử dụng tìm kiếm dòng kết hợp với gradient gốc. Đó là, chúng tôi sử dụng hướng được đưa ra bởi $\nabla f(\mathbf{x})$ và sau đó thực hiện tìm kiếm nhị phân như tỷ lệ học tập $\eta$ giảm thiểu $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$. 

Thuật toán này hội tụ nhanh chóng (để phân tích và chứng minh xem ví dụ, :cite:`Boyd.Vandenberghe.2004`). Tuy nhiên, với mục đích học sâu, điều này không hoàn toàn khả thi, vì mỗi bước của tìm kiếm dòng sẽ yêu cầu chúng ta đánh giá chức năng khách quan trên toàn bộ tập dữ liệu. Đây là cách quá tốn kém để thực hiện. 

## Tóm tắt

* Tỷ lệ học vấn đề. Quá lớn và chúng tôi phân kỳ, quá nhỏ và chúng tôi không đạt được tiến bộ.
* Gradient gốc có thể bị mắc kẹt trong minima địa phương.
* Ở kích thước cao, việc điều chỉnh tốc độ học tập rất phức tạp.
* Điều hòa trước có thể giúp điều chỉnh quy mô.
* Phương pháp của Newton nhanh hơn rất nhiều khi nó đã bắt đầu hoạt động bình thường trong các bài toán lồi.
* Cẩn thận với việc sử dụng phương pháp Newton mà không có bất kỳ điều chỉnh nào đối với các bài toán không lồi.

## Bài tập

1. Thử nghiệm với các tốc độ học tập khác nhau và chức năng khách quan để chuyển đổi độ dốc.
1. Thực hiện tìm kiếm dòng để giảm thiểu một hàm lồi trong khoảng $[a, b]$.
    1. Bạn có cần các dẫn xuất cho tìm kiếm nhị phân, tức là, để quyết định chọn $[a, (a+b)/2]$ hoặc $[(a+b)/2, b]$.
    1. Tốc độ hội tụ cho thuật toán nhanh như thế nào?
    1. Thực hiện thuật toán và áp dụng nó để giảm thiểu $\log (\exp(x) + \exp(-2x -3))$.
1. Thiết kế một chức năng khách quan được xác định trên $\mathbb{R}^2$ trong đó chuyển đổi độ dốc cực kỳ chậm. Gợi ý: quy mô tọa độ khác nhau khác nhau.
1. Thực hiện phiên bản nhẹ của phương pháp Newton bằng cách sử dụng điều hòa trước:
    1. Sử dụng đường chéo Hessian như preconditioner.
    1. Sử dụng các giá trị tuyệt đối của giá trị đó chứ không phải là giá trị thực tế (có thể ký).
    1. Áp dụng điều này cho vấn đề trên.
1. Áp dụng thuật toán trên cho một số hàm khách quan (lồi hay không). Điều gì xảy ra nếu bạn xoay tọa độ $45$ độ?

[Discussions](https://discuss.d2l.ai/t/351)

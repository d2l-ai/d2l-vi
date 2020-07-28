<!--
# Gradient Descent
-->

# Hạ Gradient
:label:`sec_gd`

<!--
In this section we are going to introduce the basic concepts underlying gradient descent.
This is brief by necessity.
See e.g., :cite:`Boyd.Vandenberghe.2004` for an in-depth introduction to convex optimization.
Although the latter is rarely used directly in deep learning, an understanding of gradient descent is key to understanding stochastic gradient descent algorithms.
For instance, the optimization problem might diverge due to an overly large learning rate.
This phenomenon can already be seen in gradient descent.
Likewise, preconditioning is a common technique in gradient descent and carries over to more advanced algorithms.
Let us start with a simple special case.
-->

Trong phần này chúng tôi sẽ giới thiệu các khái niệm cơ bản trong thuật toán hạ gradient.
Nội dung cần thiết sẽ được trình bày ngắn gọn.
Độc giả có thể tham khảo :cite:`Boyd.Vandenberghe.2004` để có góc nhìn sâu về bài toán tối ưu lồi.
Mặc dù tối ưu lồi hiếm khi được áp dụng trực tiếp trong học sâu, kiến thức về thuật toán hạ gradient là chìa khóa để hiểu rõ hơn về thuật toán hạ gradient ngẫu nhiên.
Ví dụ, bài toán tối ưu có thể phân kỳ do tốc độ học quá lớn.
Hiện tượng này có thể quan sát được trong thuật toán hạ gradient.
Tương tự, tiền điều kiện (*preconditioning*) là một kỹ thuật phổ biến trong thuật toán hạ gradient và nó cũng được áp dụng trong các thuật toán tân tiến hơn.
Hãy bắt đầu với một trường hợp đặc biệt và đơn giản.


<!--
## Gradient Descent in One Dimension
-->

## Hạ Gradient trong Một Chiều 


<!--
Gradient descent in one dimension is an excellent example to explain why the gradient descent algorithm may reduce the value of the objective function.
Consider some continuously differentiable real-valued function $f: \mathbb{R} \rightarrow \mathbb{R}$.
Using a Taylor expansion (:numref:`sec_single_variable_calculus`) we obtain that 
-->

Hạ gradient trong một chiều là ví dụ tuyệt vời để giải thích tại sao thuật toán hạ gradient có thể giảm giá trị hàm mục tiêu.
Hãy xem xét một hàm số thực khả vi liên tục $f: \mathbb{R} \rightarrow \mathbb{R}$.
Áp dụng khai triển Taylor (:numref:`sec_single_variable_calculus`), ta có

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$
:eqlabel:`gd-taylor`

<!--
That is, in first approximation $f(x+\epsilon)$ is given by the function value $f(x)$ and the first derivative $f'(x)$ at $x$.
It is not unreasonable to assume that for small $\epsilon$ moving in the direction of the negative gradient will decrease $f$.
To keep things simple we pick a fixed step size $\eta > 0$ and choose $\epsilon = -\eta f'(x)$.
Plugging this into the Taylor expansion above we get
-->

Trong đó xấp xỉ bậc nhất $f(x+\epsilon)$ được tính bằng giá trị hàm $f(x)$ và đạo hàm bậc nhất $f'(x)$ tại $x$. 
Có lý khi giả sử rằng di chuyển theo hướng ngược chiều gradient với $\epsilon$ nhỏ sẽ làm suy giảm giá trị $f$. 
Để đơn giản hóa vấn đề, ta cố định sải bước cập nhật (tốc độ học) $\eta > 0$ và chọn $\epsilon = -\eta f'(x)$. 
Thay biểu thức này vào khai triển Taylor ở trên, ta thu được 

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$


<!--
If the derivative $f'(x) \neq 0$ does not vanish we make progress since $\eta f'^2(x)>0$.
Moreover, we can always choose $\eta$ small enough for the higher order terms to become irrelevant.
Hence we arrive at
-->

Nếu đạo hàm $f'(x) \neq 0$ không tiêu biến, quá trình tối ưu sẽ có tiến triển do $\eta f'^2(x)>0$.  
Hơn nữa, chúng ta luôn có thể chọn $\eta$ đủ nhỏ để loại bỏ các hạng tử bậc cao hơn trong phép cập nhật. 
Do đó, ta có  

$$f(x - \eta f'(x)) \lessapprox f(x).$$


<!--
This means that, if we use
-->

Điều này có nghĩa là, nếu chúng ta áp dụng 

$$x \leftarrow x - \eta f'(x)$$


<!--
to iterate $x$, the value of function $f(x)$ might decline.
Therefore, in gradient descent we first choose an initial value $x$ and a constant $\eta > 0$ and then use them to continuously iterate $x$ until the stop condition is reached,
for example, when the magnitude of the gradient $|f'(x)|$ is small enough or the number of iterations has reached a certain value.
-->

để cập nhật $x$, giá trị của hàm $f(x)$ có thể giảm. 
Do đó, trong thuật toán hạ gradient, đầu tiên chúng ta chọn giá trị khởi tạo cho $x$ và hằng số $\eta > 0$, từ đó cập nhật giá trị $x$ liên tục cho tới khi thỏa mãn điều kiện dừng, ví dụ như khi độ lớn của gradient $|f'(x)|$ đủ nhỏ hoặc số lần cập nhật đạt một ngưỡng nhất định. 


<!--
For simplicity we choose the objective function $f(x)=x^2$ to illustrate how to implement gradient descent.
Although we know that $x=0$ is the solution to minimize $f(x)$, we still use this simple function to observe how $x$ changes.
As always, we begin by importing all required modules.
-->

Để đơn giản hóa vấn đề, chúng ta chọn hàm mục tiêu $f(x)=x^2$ để minh họa cách lập trình thuật toán hạ gradient. 
Ta sử dụng ví dụ đơn giản này để quan sát cách mà $x$ thay đổi, dù đã biết rằng $x=0$ là nghiệm để cực tiểu hóa $f(x)$. 
Như mọi khi, chúng ta bắt đầu bằng cách nhập tất cả các mô-đun cần thiết. 

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

<!--
```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```
-->

```{.python .input}
#@tab all
f = lambda x: x**2  # Objective function
gradf = lambda x: 2 * x  # Its derivative
```


<!--
Next, we use $x=10$ as the initial value and assume $\eta=0.2$.
Using gradient descent to iterate $x$ for 10 times we can see that, eventually, the value of $x$ approaches the optimal solution.
-->

Tiếp theo, chúng ta sử dụng $x=10$ làm giá trị khởi tạo và chọn $\eta=0.2$. 
Áp dụng thuật toán hạ gradient để cập nhật $x$ trong 10 vòng lặp, chúng ta có thể thấy cuối cùng giá trị của $x$ cũng tiệm cận nghiệm tối ưu. 


```{.python .input}
#@tab all
def gd(eta):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * gradf(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results
res = gd(0.2)
```


<!--
The progress of optimizing over $x$ can be plotted as follows.
-->

Đồ thị quá trình tối ưu hóa theo $x$ được vẽ như sau.


```{.python .input}
#@tab all
def show_trace(res):
    n = max(abs(min(res)), abs(max(res)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, res], [[f(x) for x in f_line], [f(x) for x in res]],
             'x', 'f(x)', fmts=['-', '-o'])

show_trace(res)
```


<!--
### Learning Rate
-->

### Tốc độ học

:label:`section_gd-learningrate`

<!--
The learning rate $\eta$ can be set by the algorithm designer.
If we use a learning rate that is too small, it will cause $x$ to update very slowly, requiring more iterations to get a better solution.
To show what happens in such a case, consider the progress in the same optimization problem for $\eta = 0.05$.
As we can see, even after 10 steps we are still very far from the optimal solution.
-->

Tốc độ học $\eta$ có thể được thiết lập khi thiết kế thuật toán. 
Nếu ta sử dụng tốc độ học quá nhỏ thì $x$ sẽ được cập nhật rất chậm, đòi hỏi số bước cập nhật nhiều hơn để thu được nghiệm tốt hơn. 
Để minh họa, hãy xem xét quá trình học trong cùng bài toán tối ưu ở phía trên với $\eta = 0.05$. 
Như chúng ta có thể thấy, ngay cả sau 10 bước cập nhật, chúng ta vẫn còn ở rất xa nghiệm tối ưu. 


```{.python .input}
#@tab all
show_trace(gd(0.05))
```


<!--
Conversely, if we use an excessively high learning rate, $\left|\eta f'(x)\right|$ might be too large for the first-order Taylor expansion formula.
That is, the term $\mathcal{O}(\eta^2 f'^2(x))$ in :eqref:`gd-taylor` might become significant.
In this case, we cannot guarantee that the iteration of $x$ will be able to lower the value of $f(x)$.
For example, when we set the learning rate to $\eta=1.1$, $x$ overshoots the optimal solution $x=0$ and gradually diverges.
-->

Ngược lại, nếu ta sử dụng tốc độ học quá cao, giá trị $\left|\eta f'(x)\right|$ có thể rất lớn trong khai triển Taylor bậc nhất. 
Cụ thể, hạng tử $\mathcal{O}(\eta^2 f'^2(x))$ trong :eqref: `gd-taylor` sẽ có thể có giá trị lớn. 
Trong trường hợp này, ta không thể đảm bảo rằng việc cập nhật $x$ sẽ có thể làm suy giảm giá trị của $f(x)$. 
Ví dụ, khi chúng ta thiết lập tốc độ học $\eta=1.1$, $x$ sẽ lệch rất xa so với nghiệm tối ưu $x=0$ và dần dần phân kỳ. 



```{.python .input}
#@tab all
show_trace(gd(1.1))
```


<!--
### Local Minima
-->

### Cực Tiểu


<!--
To illustrate what happens for nonconvex functions consider the case of $f(x) = x \cdot \cos c x$.
This function has infinitely many local minima.
Depending on our choice of learning rate and depending on how well conditioned the problem is, we may end up with one of many solutions.
The example below illustrates how an (unrealistically) high learning rate will lead to a poor local minimum.
-->

Để minh họa quá trình học các hàm không lồi, ta xem xét trường hợp $f(x) = x \cdot \cos c x$.
Hàm này có vô số cực tiểu.
Tùy thuộc vào tốc độ học được chọn và điều kiện của bài toán, chúng ta có thể thu được một trong số rất nhiều nghiệm.
Ví dụ dưới đây minh họa việc thiết lập tốc độ học quá cao (không thực tế) sẽ dẫn đến điểm cực tiểu không tốt.


```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)
f = lambda x: x * d2l.cos(c * x)
gradf = lambda x: d2l.cos(c * x) - c * x * d2l.sin(c * x)
show_trace(gd(2))
```


<!--
## Multivariate Gradient Descent
-->

## Hạ Gradient Đa biến


<!--
Now that we have a better intuition of the univariate case, let us consider the situation where $\mathbf{x} \in \mathbb{R}^d$.
That is, the objective function $f: \mathbb{R}^d \to \mathbb{R}$ maps vectors into scalars. Correspondingly its gradient is multivariate, too.
It is a vector consisting of $d$ partial derivatives:
-->

Bây giờ chúng ta đã có trực quan tốt hơn về trường hợp đơn biến, ta hãy xem xét trường hợp trong đó $\mathbf{x} \in \mathbb{R}^d$.
Cụ thể, hàm mục tiêu $f: \mathbb{R}^d \to \mathbb{R}$ ánh xạ các vector tới các giá trị vô hướng.
Gradient tương ứng cũng là đa biến, là một vector gồm $d$ đạo hàm riêng:


$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$


<!--
Each partial derivative element $\partial f(\mathbf{x})/\partial x_i$ in the gradient indicates the rate of change of $f$ at $\mathbf{x}$ with respect to the input $x_i$.
As before in the univariate case we can use the corresponding Taylor approximation for multivariate functions to get some idea of what we should do.
In particular, we have that 
-->

Mỗi đạo hàm riêng $\partial f(\mathbf{x})/\partial x_i$ trong gradient biểu diễn tốc độ thay đổi theo $x_i$ của $f$ tại $\mathbf{x}$. 
Như trong trường hợp đơn biến giới thiệu ở phần trước, ta sử dụng khai triển Taylor tương ứng cho các hàm đa biến. 
Cụ thể, ta có 


$$f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\mathbf{\epsilon}\|^2).$$
:eqlabel:`gd-multi-taylor`


<!--
In other words, up to second order terms in $\mathbf{\epsilon}$ the direction of steepest descent is given by the negative gradient $-\nabla f(\mathbf{x})$.
Choosing a suitable learning rate $\eta > 0$ yields the prototypical gradient descent algorithm:
-->

Nói cách khác, chiều giảm mạnh nhất được cho bởi gradient âm $-\nabla f(\mathbf{x})$, các hạng tử từ bậc hai trở lên trong $\mathbf{\epsilon}$ có thể bỏ qua. 
Chọn một tốc độ học phù hợp $\eta > 0$, ta được thuật toán hạ gradient nguyên bản dưới đây: 


$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$


<!--
To see how the algorithm behaves in practice let us construct an objective function $f(\mathbf{x})=x_1^2+2x_2^2$ with a two-dimensional vector $\mathbf{x} = [x_1, x_2]^\top$ as input and a scalar as output.
The gradient is given by $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$.
We will observe the trajectory of $\mathbf{x}$ by gradient descent from the initial position $[-5, -2]$.
We need two more helper functions.
The first uses an update function and applies it $20$ times to the initial value.
The second helper visualizes the trajectory of $\mathbf{x}$.
-->

Để xem thuật toán hoạt động như thế nào trong thực tế, ta hãy xây dựng một hàm mục tiêu
$f(\mathbf{x})=x_1^2+2x_2^2$ với đầu vào là vector hai chiều $\mathbf{x} = [x_1, x_2]^\top$ và đầu ra là một số vô hướng.
Gradient được cho bởi $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$.
Ta sẽ quan sát đường đi của $\mathbf{x}$ được sinh bởi thuật toán hạ gradient bắt đầu từ vị trí $[-5, -2]$.
Chúng ta cần thêm hai hàm hỗ trợ.
Hàm đầu tiên là hàm cập nhật và được sử dụng $20$ lần cho giá trị khởi tạo ban đầu.
Hàm thứ hai là hàm vẽ biểu đồ đường đi của $\mathbf{x}$.


```{.python .input}
#@tab all
def train_2d(trainer, steps=20):  #@save
    """Optimize a 2-dim objective function with a customized trainer."""
    # s1 and s2 are internal state variables and will
    # be used later in the chapter
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
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


<!--
Next, we observe the trajectory of the optimization variable $\mathbf{x}$ for learning rate $\eta = 0.1$.
We can see that after 20 steps the value of $\mathbf{x}$ approaches its minimum at $[0, 0]$.
Progress is fairly well-behaved albeit rather slow.
-->

Tiếp theo, chúng ta sẽ quan sát quỹ đạo của biến tối ưu hóa $\mathbf{x}$ với tốc độ học $\eta = 0.1$.
Chúng ta có thể thấy rằng sau 20 bước, giá trị $\mathbf{x}$ đã đạt cực tiểu tại $[0, 0]$.
Quá trình khá tốt mặc dù hơi chậm.


```{.python .input}
#@tab all
f = lambda x1, x2: x1 ** 2 + 2 * x2 ** 2  # Objective
gradf = lambda x1, x2: (2 * x1, 4 * x2)  # Gradient

def gd(x1, x2, s1, s2):
    (g1, g2) = gradf(x1, x2)  # Compute gradient
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)  # Update variables

eta = 0.1
show_trace_2d(f, train_2d(gd))
```


<!--
## Adaptive Methods
-->

## Những Phương pháp Thích nghi

<!--
As we could see in :numref:`section_gd-learningrate`, getting the learning rate $\eta$ "just right" is tricky.
If we pick it too small, we make no progress.
If we pick it too large, the solution oscillates and in the worst case it might even diverge.
What if we could determine $\eta$ automatically or get rid of having to select a step size at all?
Second order methods that look not only at the value and gradient of the objective but also at its *curvature* can help in this case.
While these methods cannot be applied to deep learning directly due to the computational cost, 
they provide useful intuition into how to design advanced optimization algorithms that mimic many of the desirable properties of the algorithms outlined below.
-->

Như chúng ta có thể thấy ở :numref:`section_gd-learningrate`, chọn tốc độ học $\eta$ "vừa đủ" rất khó. 
Nếu chọn giá trị quá nhỏ, ta sẽ không có tiến triển.
Nếu chọn giá trị quá lớn, nghiệm sẽ dao động và trong trường hợp tệ nhất, thậm chí sẽ phân kỳ.
Sẽ ra sao nếu chúng ta có thể chọn $\eta$ một cách tự động, hoặc giả như loại bỏ được việc chọn kích thước bước?
Các phương pháp bậc hai không chỉ dựa vào giá trị và gradient của hàm mục tiêu mà còn dựa vào "độ cong" của hàm, từ đó có thể điều chỉnh tốc độ học.
Dù những phương pháp này không thể áp dụng vào học sâu một cách trực tiếp do chi phí tính toán lớn, chúng đem đến những gợi ý hữu ích để thiết kế các thuật toán tối ưu cao cấp hơn, mang nhiều tính chất mong muốn dựa trên các thuật toán dưới đây.

<!--
### Newton's Method
-->

### Phương pháp Newton

<!--
Reviewing the Taylor expansion of $f$ there is no need to stop after the first term.
In fact, we can write it as 
-->

Trong khai triển Taylor của $f$, ta không cần phải dừng ngay sau số hạng đầu tiên.
Trên thực tế, ta có thể viết lại như sau 


$$f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \mathbf{\epsilon}^\top \nabla \nabla^\top f(\mathbf{x}) \mathbf{\epsilon} + \mathcal{O}(\|\mathbf{\epsilon}\|^3).$$
:eqlabel:`gd-hot-taylor`


<!--
To avoid cumbersome notation we define $H_f := \nabla \nabla^\top f(\mathbf{x})$ to be the *Hessian* of $f$.
This is a $d \times d$ matrix. For small $d$ and simple problems $H_f$ is easy to compute.
For deep networks, on the other hand, $H_f$ may be prohibitively large, due to the cost of storing $\mathcal{O}(d^2)$ entries.
Furthermore it may be too expensive to compute via backprop as we would need to apply backprop to the backpropagation call graph.
For now let us ignore such considerations and look at what algorithm we'd get. 
-->

Để tránh việc kí hiệu quá nhiều, ta định nghĩa $H_f := \nabla \nabla^\top f(\mathbf{x})$ là *ma trận Hessian* của $f$. 
Đây là ma trận kích thước $d \times d$. Với $d$ nhỏ và trong các bài toán đơn giản, ta sẽ dễ tính được $H_f$. 
Nhưng với các mạng sâu, kích thước của $H_f$ có thể cực lớn, do chi phí lưu trữ bậc hai $\mathcal{O}(d^2)$. 
Hơn nữa việc tính toán lan truyền ngược có thể đòi hỏi rất nhiều chi phí tính toán.
Tạm thời hãy bỏ qua những lưu ý đó và nhìn vào thuật toán mà ta có được.


<!--
After all, the minimum of $f$ satisfies $\nabla f(\mathbf{x}) = 0$.
Taking derivatives of :eqref:`gd-hot-taylor` with regard to $\mathbf{\epsilon}$ and ignoring higher order terms we arrive at 
-->

Suy cho cùng, cực tiểu của $f$ sẽ thỏa $\nabla f(\mathbf{x}) = 0$. 
Lấy các đạo hàm của :eqref:`gd-hot-taylor` theo $\mathbf{\epsilon}$ và bỏ qua các số hạng bậc cao ta thu được 


$$\nabla f(\mathbf{x}) + H_f \mathbf{\epsilon} = 0 \text{ và~do~đó } 
\mathbf{\epsilon} = -H_f^{-1} \nabla f(\mathbf{x}).$$


<!--
That is, we need to invert the Hessian $H_f$ as part of the optimization problem.
-->

Nghĩa là, ta cần phải nghịch đảo ma trận Hessian $H_f$ như một phần của bài toán tối ưu hóa.

<!--
For $f(x) = \frac{1}{2} x^2$ we have $\nabla f(x) = x$ and $H_f = 1$.
Hence for any $x$ we obtain $\epsilon = -x$.
In other words, a single step is sufficient to converge perfectly without the need for any adjustment!
Alas, we got a bit lucky here since the Taylor expansion was exact.
Let us see what happens in other problems.
-->

Với $f(x) = \frac{1}{2} x^2$ ta có $\nabla f(x) = x$ và $H_f = 1$.
Do đó với $x$ bất kỳ, ta đều thu được $\epsilon = -x$.
Nói cách khác, một bước đơn lẻ là đã đủ để hội tụ một cách hoàn hảo mà không cần bất kỳ tinh chỉnh nào!
Chúng ta khá may mắn ở đây vì khai triển Taylor không cần xấp xỉ.
Hãy xem thử điều gì sẽ xảy ra với các bài toán khác.


```{.python .input}
#@tab all
c = d2l.tensor(0.5)
f = lambda x: d2l.cosh(c * x)  # Objective
gradf = lambda x: c * d2l.sinh(c * x)  # Derivative
hessf = lambda x: c**2 * d2l.cosh(c * x)  # Hessian

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * gradf(x) / hessf(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results
show_trace(newton())
```


<!--
Now let us see what happens when we have a *nonconvex* function, such as $f(x) = x \cos(c x)$.
After all, note that in Newton's method we end up dividing by the Hessian.
This means that if the second derivative is *negative* we would walk into the direction of *increasing* $f$.
That is a fatal flaw of the algorithm.
Let us see what happens in practice.
-->

Giờ hãy xem điều gì xảy ra với một hàm *không lồi*, ví dụ như $f(x) = x \cos(c x)$.
Sau tất cả, hãy lưu ý rằng trong phương pháp Newton, chúng ta cuối cùng sẽ phải chia cho ma trận Hessian.
Điều này nghĩa là nếu đạo hàm bậc hai là *âm* thì chúng ta phải đi theo hướng *tăng* $f$.
Đó là khiếm khuyết chết người của thuật toán này.
Hãy xem điều gì sẽ xảy ra trong thực tế.


```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)
f = lambda x: x * d2l.cos(c * x)
gradf = lambda x: d2l.cos(c * x) - c * x * d2l.sin(c * x)
hessf = lambda x: - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton())
```


<!--
This went spectacularly wrong.
How can we fix it? One way would be to "fix" the Hessian by taking its absolute value instead.
Another strategy is to bring back the learning rate.
This seems to defeat the purpose, but not quite.
Having second order information allows us to be cautious whenever the curvature is large and to take longer steps whenever the objective is flat.
Let us see how this works with a slightly smaller learning rate, say $\eta = 0.5$. As we can see, we have quite an efficient algorithm.
-->

Kết quả trả về là cực kỳ sai.
Có một cách khắc phục là "sửa" ma trận Hessian bằng cách lấy giá trị tuyệt đối của nó. 
Một chiến lược khác là đưa tốc độ học trở lại. 
Điều này có vẻ sẽ phá hỏng mục tiêu ban đầu nhưng không hẳn. 
Có được thông tin bậc hai sẽ cho phép chúng ta thận trọng bất cứ khi nào độ cong trở nên lớn và cho phép thực hiện các bước dài hơn mỗi khi hàm mục tiêu phẳng. 
Hãy xem nó hoạt động như thế nào với một tốc độ học khá nhỏ, $\eta = 0.5$ chẳng hạn. Như ta có thể thấy, chúng ta có một thuật toán khá hiệu quả.


```{.python .input}
#@tab all
show_trace(newton(0.5))
```


<!--
### Convergence Analysis
-->

### Phân tích Hội tụ

<!--
We only analyze the convergence rate for convex and three times differentiable $f$, where at its minimum $x^*$ the second derivative is nonzero, i.e., where $f''(x^*) > 0$.
The multivariate proof is a straightforward extension of the argument below and omitted since it doesn't help us much in terms of intuition. 
-->

Chúng ta sẽ chỉ phân tích tốc độ hội tụ đối với hàm $f$ lồi và khả vi ba lần, đây là hàm số có đạo hàm bậc hai tại cực tiểu $x^*$ khác không ($f''(x^*) > 0$). 

<!--
Denote by $x_k$ the value of $x$ at the $k$-th iteration and let $e_k := x_k - x^*$ be the distance from optimality.
By Taylor series expansion we have that the condition $f'(x^*) = 0$ can be written as
-->

Đặt $x_k$ là giá trị của $x$ tại vòng lặp thứ $k$ và $e_k := x_k - x^*$ là khoảng cách đến điểm tối ưu.
Theo khai triển Taylor, điều kiện $f'(x^*) = 0$ được viết lại thành 


$$0 = f'(x_k - e_k) = f'(x_k) - e_k f''(x_k) + \frac{1}{2} e_k^2 f'''(\xi_k).$$


<!--
This holds for some $\xi_k \in [x_k - e_k, x_k]$. Recall that we have the update $x_{k+1} = x_k - f'(x_k) / f''(x_k)$.
Dividing the above expansion by $f''(x_k)$ yields
-->

Điều này đúng với một vài $\xi_k \in [x_k - e_k, x_k]$. Hãy nhớ rằng chúng ta có công thức cập nhật $x_{k+1} = x_k - f'(x_k) / f''(x_k)$. 
Chia khai triển Taylor ở trên cho $f''(x_k)$, ta thu được 


$$e_k - f'(x_k) / f''(x_k) = \frac{1}{2} e_k^2 f'''(\xi_k) / f''(x_k).$$ 


<!--
Plugging in the update equations leads to the following bound $e_{k+1} \leq e_k^2 f'''(\xi_k) / f'(x_k)$.
Consequently, whenever we are in a region of bounded $f'''(\xi_k) / f''(x_k) \leq c$, we have a quadratically decreasing error $e_{k+1} \leq c e_k^2$. 
-->

Thay vào phương trình cập nhật sẽ dẫn đến ràng buộc $e_{k+1} \leq e_k^2 f'''(\xi_k) / f'(x_k)$.
Do đó, khi nằm trong miền ràng buộc $f'''(\xi_k) / f''(x_k) \leq c$, ta sẽ có sai số giảm theo bình phương $e_{k+1} \leq c e_k^2$.

<!--
As an aside, optimization researchers call this *linear* convergence, whereas a condition such as $e_{k+1} \leq \alpha e_k$ would be called a *constant* rate of convergence. 
Note that this analysis comes with a number of caveats: We do not really have much of a guarantee when we will reach the region of rapid convergence.
Instead, we only know that once we reach it, convergence will be very quick.
Second, this requires that $f$ is well-behaved up to higher order derivatives.
It comes down to ensuring that $f$ does not have any "surprising" properties in terms of how it might change its values. 
-->

Bên cạnh đó, các nhà nghiên cứu tối ưu hóa gọi đây là hội tụ *tuyến tính*, còn điều kiện $e_{k+1} \leq \alpha e_k$ được gọi là tốc độ hội tụ *không đổi*. 
Lưu ý rằng phân tích này đi kèm với một số lưu ý: Chúng ta không thực sự biết rằng khi nào mình sẽ tiến tới được vùng hội tụ nhanh. 
Thay vào đó, ta chỉ biết rằng một khi đến được đó, việc hội tụ sẽ xảy ra rất nhanh chóng. 
Thêm nữa, điều này yêu cầu $f$ được xử lý tốt ở các đạo hàm bậc cao. 
Nó đảm bảo không có bất cứ một tính chất "bất ngờ" nào của $f$ có thể dẫn đến sự thay đổi giá trị của nó. 


<!--
### Preconditioning
-->

### Tiền Điều kiện

<!--
Quite unsurprisingly computing and storing the full Hessian is very expensive.
It is thus desirable to find alternatives.
One way to improve matters is by avoiding to compute the Hessian in its entirety but only compute the *diagonal* entries.
While this is not quite as good as the full Newton method, it is still much better than not using it.
Moreover, estimates for the main diagonal elements are what drives some of the innovation in stochastic gradient descent optimization algorithms.
This leads to update algorithms of the form
-->

Không có gì ngạc nhiên khi việc tính toán và lưu trữ toàn bộ ma trận Hessian là rất tốn kém.
Do đó ta cần tìm kiếm một phương pháp thay thế.
Một cách để cải thiện vấn đề này là tránh tính toán toàn bộ ma trận Hessian, chỉ tính toán các giá trị thuộc *đường chéo*.
Mặc dù cách trên không tốt bằng phương pháp Newton hoàn chỉnh nhưng vẫn tốt hơn nhiều so với không sử dụng nó.
Hơn nữa, ước lượng các giá trị đường chéo chính là thứ thúc đẩy sự đổi mới trong các thuật toán tối ưu hóa hạ gradient ngẫu nhiên.
Thuật toán cập nhật sẽ có dạng


$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(H_f)^{-1} \nabla \mathbf{x}.$$


<!--
To see why this might be a good idea consider a situation where one variable denotes height in millimeters and the other one denotes height in kilometers.
Assuming that for both the natural scale is in meters we have a terrible mismatch in parameterizations.
Using preconditioning removes this.
Effectively preconditioning with gradient descent amounts to selecting a different learning rate for each coordinate. 
-->

Để thấy tại sao điều này có thể là một ý tưởng tốt, ta ví dụ có hai biến số biểu thị chiều cao, một biến với đơn vị mm, biến còn lại với đơn vị km.
Với cả hai đơn vị đo, khi quy đổi ra mét, chúng ta đều có sự sai lệch lớn trong việc tham số hóa.
Sử dụng tiền điều kiện sẽ loại bỏ vấn đề này.
Tiền điều kiện một cách hiệu quả cùng hạ gradient giúp chọn ra các tốc độ học khác nhau cho từng trục tọa độ.

<!--
### Gradient Descent with Line Search
-->

### Hạ gradient cùng Tìm kiếm Đường thẳng

<!--
One of the key problems in gradient descent was that we might overshoot the goal or make insufficient progress.
A simple fix for the problem is to use line search in conjunction with gradient descent.
That is, we use the direction given by $\nabla f(\mathbf{x})$ and then perform binary search as to which step length $\eta$ minimizes $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$. 
-->

Một trong những vấn đề chính của hạ gradient là chúng ta có thể vượt quá khỏi mục tiêu hoặc không đạt đủ sự tiến bộ. 
Có một cách khắc phục đơn giản cho vấn đề này là sử dụng tìm kiếm đường thẳng (*line search*) kết hợp với hạ gradient.  
Chúng ta sử dụng hướng được cho bởi $\nabla f(\mathbf{x})$ và sau đó dùng tìm kiếm nhị phân để tìm ra độ dài bước $\eta$ có thể cực tiểu hóa $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$. 

<!--
This algorithm converges rapidly (for an analysis and proof see e.g., :cite:`Boyd.Vandenberghe.2004`).
However, for the purpose of deep learning this is not quite so feasible, since each step of the line search would require us to evaluate the objective function on the entire dataset.
This is way too costly to accomplish. 
-->

Thuật toán này sẽ hội tụ nhanh chóng (xem phân tích và chứng minh ở :cite:`Boyd.Vandenberghe.2004`). 
Tuy nhiên, đối với mục đích của học sâu thì nó không thực sự khả thi, lý do là mỗi bước của tìm kiếm đường thẳng sẽ yêu cầu chúng ta ước lượng hàm mục tiêu trên toàn bộ tập dữ liệu. 
Điều này quá tốn kém để có thể thực hiện. 


<!--
## Summary
-->

## Tổng kết

<!--
* Learning rates matter. Too large and we diverge, too small and we do not make progress.
* Gradient descent can get stuck in local minima.
* In high dimensions adjusting learning the learning rate is complicated.
* Preconditioning can help with scale adjustment.
* Newton's method is a lot faster *once* it has started working properly in convex problems.
* Beware of using Newton's method without any adjustments for nonconvex problems. 
-->

* Tốc độ học rất quan trọng. Quá lớn sẽ khiến việc tối ưu hóa phân kỳ, quá nhỏ sẽ không thu được sự tiến bộ nào.
* Hạ gradient có thể bị kẹt tại cực tiểu cục bộ.
* Trong bài toán nhiều chiều, tinh chỉnh việc học tốc độ học sẽ phức tạp.
* Tiền điều kiện có thể giúp trong việc tinh chỉnh thang đo.
* Phương pháp Newton nhanh hơn rất nhiều *một khi* hoạt động trên bài toán lồi phù hợp.
* Hãy cẩn trọng trong việc dùng phương pháp Newton cho các bài toán không lồi mà không tinh chỉnh.

<!--
## Exercises
-->

## Bài tập

<!--
1. Experiment with different learning rates and objective functions for gradient descent.
2. Implement line search to minimize a convex function in the interval $[a, b]$.
    * Do you need derivatives for binary search, i.e., to decide whether to pick $[a, (a+b)/2]$ or $[(a+b)/2, b]$. 
    * How rapid is the rate of convergence for the algorithm?
    * Implement the algorithm and apply it to minimizing $\log (\exp(x) + \exp(-2*x -3))$.
3. Design an objective function defined on $\mathbb{R}^2$ where gradient descent is exceedingly slow. Hint - scale different coordinates differently.
4. Implement the lightweight version of Newton's method using preconditioning:
    * Use diagonal Hessian as preconditioner.
    * Use the absolute values of that rather than the actual (possibly signed) values. 
    * Apply this to the problem above.
5. Apply the algorithm above to a number of objective functions (convex or not). What happens if you rotate coordinates by $45$ degrees?
-->

1. Hãy thử các tốc độ học, hàm mục tiêu khác nhau cho hạ gradient.
2. Khởi tạo tìm kiếm đường thẳng để cực tiểu hóa hàm lồi trong khoảng $[a, b]$.
    * Bạn có cần đạo hàm để tìm kiếm nhị phân không, ví dụ, để quyết định xem sẽ chọn $[a, (a+b)/2]$ hay $[(a+b)/2, b]$?
    * Tốc độ hội tụ của thuật toán nhanh chậm thế nào?
    * Hãy khởi tạo thuật toán và áp dụng nó để cực tiểu hóa $\log (\exp(x) + \exp(-2*x -3))$.
3. Thiết kế một hàm mục tiêu thuộc $\mathbb{R}^2$ mà việc hạ gradient rất chậm. Gợi ý: sử dụng trục tọa độ có thang đo khác nhau.
4. Khởi tạo một phiên bản nhỏ gọn của phương pháp Newton sử dụng tiền điều kiện:
    * Dùng ma trận đường chéo Hessian làm tiền điều kiện.
    * Sử dụng các giá trị tuyệt đối của nó thay vì các giá trị có dấu.
    * Áp dụng điều này cho bài toán phía trên.
5. Áp dụng thuật toán phía trên cho các hàm mục tiêu (lồi lẫn không lồi). Điều gì sẽ xảy ra nếu xoay các trục tọa độ một góc $45$ độ?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/351)
* [Tiếng Anh - Pytorch](https://discuss.d2l.ai/t/491)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Lê Quang Nhật
* Nguyễn Văn Quang
* Nguyễn Văn Cường
* Phạm Hồng Vinh
* Phạm Minh Đức
* Nguyễn Thanh Hòa
* Võ Tấn Phát

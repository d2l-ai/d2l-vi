<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Convexity
-->

# Tính lồi
:label:`sec_convexity`

<!--
Convexity plays a vital role in the design of optimization algorithms.
This is largely due to the fact that it is much easier to analyze and test algorithms in this context.
In other words, if the algorithm performs poorly even in the convex setting we should not hope to see great results otherwise.
Furthermore, even though the optimization problems in deep learning are generally nonconvex, they often exhibit some properties of convex ones near local minima.
This can lead to exciting new optimization variants such as :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`.
-->

Tính lồi đóng vai trò then chốt trong việc thiết kế các thuật toán tối ưu.
Điều này phần lớn là do tính lồi giúp việc phân tích và kiểm tra thuật toán trở nên dễ dàng hơn.
Nói cách khác, nếu thuật toán hoạt động kém ngay cả khi có tính lồi thì ta không nên kì vọng rằng sẽ thu được kết quả tốt trong trường hợp khác.
Hơn nữa, mặc dù các bài toán tối ưu hóa trong học sâu đa phần là không lồi, chúng lại thường thể hiện một số tính chất lồi gần các cực tiểu.
Điều này dẫn đến các biến thể tối ưu hóa thú vị mới như :cite:`Izmailov.Podoprikhin.Garipov.ea.2018`.

<!--
## Basics
-->

## Kiến thức Cơ bản

<!--
Let us begin with the basics.
-->

Chúng ta hãy bắt đầu với các kiến thức cơ bản trước.

<!--
### Sets
-->

### Tập hợp

<!--
Sets are the basis of convexity.
Simply put, a set $X$ in a vector space is convex if for any $a, b \in X$ the line segment connecting $a$ and $b$ is also in $X$.
In mathematical terms this means that for all $\lambda \in [0, 1]$ we have
-->

Tập hợp là nền tảng của tính lồi.
Nói một cách đơn giản, một tập hợp $X$ trong không gian vector là lồi nếu với bất kì $a, b \in X$, đoạn thẳng nối $a$ và $b$ cũng thuộc $X$.
Theo thuật ngữ toán học, điều này có nghĩa là với mọi $\lambda \in [0, 1]$, ta có


$$\lambda \cdot a + (1-\lambda) \cdot b \in X \text{với mọi} a, b \in X.$$


<!--
This sounds a bit abstract.
Consider the picture :numref:`fig_pacman`.
The first set is not convex since there are line segments that are not contained in it.
The other two sets suffer no such problem.
-->

Điều này nghe có vẻ hơi trừu tượng.
Hãy xem qua bức ảnh :numref:`fig_pacman`.
Tập hợp đầu tiên là không lồi do tồn tại các đoạn thẳng không nằm trong tập hợp.
Hai tập hợp còn lại thì không gặp vấn đề như vậy.

<!--
![Three shapes, the left one is nonconvex, the others are convex](../img/pacman.svg)
-->

![Ba hình dạng, hình bên trái là không lồi, hai hình còn lại là lồi](../img/pacman.svg)
:label:`fig_pacman`

<!--
Definitions on their own are not particularly useful unless you can do something with them.
In this case we can look at unions and intersections as shown in :numref:`fig_convex_intersect`.
Assume that $X$ and $Y$ are convex sets.
Then $X \cap Y$ is also convex.
To see this, consider any $a, b \in X \cap Y$. Since $X$ and $Y$ are convex, the line segments connecting $a$ and $b$ are contained in both $X$ and $Y$.
Given that, they also need to be contained in $X \cap Y$, thus proving our first theorem.
-->

Chỉ một mình định nghĩa thôi thì sẽ không có tác dụng gì trừ khi bạn có thể làm gì đó với chúng.
Trong trường hợp này, ta có thể nhìn vào phép hợp và phép giao trong :numref:`fig_convex_intersect`.
Giả sử $X$ và $Y$ là các tập hợp lồi, khi đó $X \cap Y$ cũng sẽ lồi.
Để thấy được điều này, hãy xét bất kì $a, b \in X \cap Y$. Vì $X$ và $Y$ lồi, khi đó đoạn thẳng nối $a$ và $b$ sẽ nằm trong cả $X$ và $Y$.
Do đó, chúng cũng cần phải thuộc $X \cap Y$, từ đó chứng minh được định lý đầu tiên của chúng ta.

<!--
![The intersection between two convex sets is convex](../img/convex-intersect.svg)
-->

![Giao của hai tập lồi là một tập lồi](../img/convex-intersect.svg)
:label:`fig_convex_intersect`

<!--
We can strengthen this result with little effort: given convex sets $X_i$, their intersection $\cap_{i} X_i$ is convex.
To see that the converse is not true, consider two disjoint sets $X \cap Y = \emptyset$.
Now pick $a \in X$ and $b \in Y$.
The line segment in :numref:`fig_nonconvex` connecting $a$ and $b$ needs to contain some part that is neither in $X$ nor $Y$, since we assumed that $X \cap Y = \emptyset$.
Hence the line segment is not in $X \cup Y$ either, thus proving that in general unions of convex sets need not be convex.
-->

Ta sẽ củng cố kết quả này thêm một chút với mệnh đề: giao của các tập lồi $X_i$ là một tập lồi $\cap_{i} X_i$.
Để thấy rằng điều ngược lại là không đúng, hãy xem xét hai tập hợp không giao nhau $X \cap Y = \emptyset$.
Giờ ta chọn ra $a \in X$ và $b \in Y$.
Đoạn thẳng nối $a$ và $b$ trong :numref:`fig_nonconvex` chứa một vài phần không thuộc cả $X$ và $Y$, vì chúng ta đã giả định rằng $X \cap Y = \emptyset$.
Do đó đoạn thẳng này cũng không nằm trong $X \cup Y$, từ đó chứng minh rằng hợp của các tập lồi nói chung không nhất thiết phải là tập lồi.

<!--
![The union of two convex sets need not be convex](../img/nonconvex.svg)
-->

![Hợp của hai tập lồi không nhất thiết phải là tập lồi](../img/nonconvex.svg)
:label:`fig_nonconvex`

<!--
Typically the problems in deep learning are defined on convex domains.
For instance $\mathbb{R}^d$ is a convex set (after all, the line between any two points in $\mathbb{R}^d$ remains in $\mathbb{R}^d$).
In some cases we work with variables of bounded length, such as balls of radius $r$ as defined by $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_2 \leq r\}$.
-->

Thông thường, các bài toán trong học sâu đều được định nghĩa trong các miền lồi.
Ví dụ $\mathbb{R}^d$ là tập lồi (xét cho cùng, đoạn thẳng nối hai điểm bất kỳ thuộc $\mathbb{R}^d$ vẫn thuộc $\mathbb{R}^d$).
Trong một vài trường hợp, chúng ta sẽ làm việc với các biến có biên, ví dụ như khối cầu có bán kính $r$ được định nghĩa bằng $\{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ và } \|\mathbf{x}\|_2 \leq r\}$.


<!--
### Functions
-->

### Hàm số

<!--
Now that we have convex sets we can introduce convex functions $f$.
Given a convex set $X$ a function defined on it $f: X \to \mathbb{R}$ is convex if for all $x, x' \in X$ and for all $\lambda \in [0, 1]$ we have
-->


Giờ ta đã biết về tập hợp lồi, ta sẽ làm việc tiếp với các hàm số lồi $f$.
Cho một tập hợp lồi $X$, một hàm số được định nghĩa trên tập đó $f: X \to \mathbb{R}$ là hàm lồi nếu với mọi $x, x' \in X$ và mọi $\lambda \in [0, 1]$, ta có


$$\lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$


<!--
To illustrate this let us plot a few functions and check which ones satisfy the requirement.
We need to import a few  libraries.
-->

Để minh họa cho điều này, chúng ta sẽ vẽ đồ thị của một vài hàm số và kiểm tra xem hàm số nào thỏa mãn điều kiện trên.
Ta sẽ cần phải nhập một vài gói thư viện.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()
```

<!--
```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
from mpl_toolkits import mplot3d
import torch
```
-->

<!--
Let us define a few functions, both convex and nonconvex.
-->

Hãy định nghĩa một vài hàm số, cả lồi lẫn không lồi.


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


<!--
As expected, the cosine function is nonconvex, whereas the parabola and the exponential function are.
Note that the requirement that $X$ is a convex set is necessary for the condition to make sense.
Otherwise the outcome of $f(\lambda x + (1-\lambda) x')$ might not be well defined.
Convex functions have a number of desirable properties.
-->

Như dự đoán, hàm cô-sin là hàm không lồi, trong khi hàm parabol và hàm số mũ là hàm lồi.
Lưu ý rằng để điều kiện trên có ý nghĩa thì $X$ cần phải là tập hợp lồi.
Nếu không, kết quả của $f(\lambda x + (1-\lambda) x')$ sẽ không được định nghĩa rõ.
Các hàm lồi có một số tính chất mong muốn sau.

<!--
### Jensen's Inequality
-->

### Bất đẳng thức Jensen
<!--
One of the most useful tools is Jensen's inequality.
It amounts to a generalization of the definition of convexity:
-->

Một trong những công cụ hữu dụng nhất là bất đẳng thức Jensen.
Nó là sự tổng quát hóa của định nghĩa về tính lồi:


$$\begin{aligned}
    \sum_i \alpha_i f(x_i) & \geq f\left(\sum_i \alpha_i x_i\right)
    \text{ và }
    E_x[f(x)] & \geq f\left(E_x[x]\right),
\end{aligned}$$


<!--
where $\alpha_i$ are nonnegative real numbers such that $\sum_i \alpha_i = 1$.
In other words, the expectation of a convex function is larger than the convex function of an expectation.
To prove the first inequality we repeatedly apply the definition of convexity to one term in the sum at a time.
The expectation can be proven by taking the limit over finite segments.
-->


với $\alpha_i$ là các số thực không âm sao cho $\sum_i \alpha_i = 1$.
Nói cách khác, kỳ vọng của hàm lồi lớn hơn hàm lồi của kỳ vọng.
Để chứng minh bất đẳng thức đầu tiên này, chúng ta áp dụng định nghĩa của tính lồi cho từng số hạng của tổng.
Kỳ vọng có thể được chứng minh bằng cách tính giới hạn trên các đoạn hữu hạn.

<!--
One of the common applications of Jensen's inequality is with regard to the log-likelihood of partially observed random variables.
That is, we use
-->

Một trong các ứng dụng phổ biến của bất đẳng thức Jensen liên quan đến log hợp lý của các biến ngẫu nhiên quan sát được một phần.
Ta có


$$E_{y \sim P(y)}[-\log P(x \mid y)] \geq -\log P(x).$$


<!--
This follows since $\int P(y) P(x \mid y) dy = P(x)$.
This is used in variational methods.
Here $y$ is typically the unobserved random variable, $P(y)$ is the best guess of how it might be distributed and $P(x)$ is the distribution with $y$ integrated out.
For instance, in clustering $y$ might be the cluster labels and $P(x \mid y)$ is the generative model when applying cluster labels.
-->

Điều này xảy ra vì $\int P(y) P(x \mid y) dy = P(x)$.
Nó được sử dụng trong những phương pháp biến phân.
$y$ ở đây thường là một biến ngẫu nhiên không quan sát được, $P(y)$ là dự đoán tốt nhất về phân phối của nó và $P(x)$ là phân phối đã được lấy tích phân theo $y$.
Ví dụ như trong bài toán phân cụm, $y$ có thể là nhãn cụm và $P(x \mid y)$ là mô hình sinh khi áp dụng các nhãn cụm.

<!--
## Properties
-->

## Tính chất

<!--
Convex functions have a few useful properties.
We describe them as follows.
-->

Các hàm lồi có một vài tính chất hữu ích dưới đây.

<!--
### No Local Minima
-->

### Không có Cực tiểu Cục bộ

<!--
In particular, convex functions do not have local minima.
Let us assume the contrary and prove it wrong. If $x \in X$ is a local minimum there exists some neighborhood of $x$ for which $f(x)$ is the smallest value.
Since $x$ is only a local minimum there has to be another $x' \in X$ for which $f(x') < f(x)$.
However, by convexity the function values on the entire *line* $\lambda x + (1-\lambda) x'$ have to be less than $f(x')$ since for $\lambda \in [0, 1)$
-->

Cụ thể, các hàm lồi không có cực tiểu cục bộ.
Hãy giả định điều ngược lại là đúng và chứng minh nó sai. Nếu $x \in X$ là cực tiểu cục bộ thì sẽ tồn tại một vùng lân cận nào đó của $x$ mà $f(x)$ là giá trị nhỏ nhất.
Vì $x$ chỉ là cực tiểu cục bộ nên phải tồn tại một $x' \in X$ nào khác mà $f(x') < f(x)$.
Tuy nhiên, theo tính lồi, các giá trị hàm số trên toàn bộ *đường thẳng* $\lambda x + (1-\lambda) x'$ phải nhỏ hơn $f(x')$ với $\lambda \in [0, 1)$ 


$$f(x) > \lambda f(x) + (1-\lambda) f(x') \geq f(\lambda x + (1-\lambda) x').$$


<!--
This contradicts the assumption that $f(x)$ is a local minimum.
For instance, the function $f(x) = (x+1) (x-1)^2$ has a local minimum for $x=1$.
However, it is not a global minimum.
-->

Điều này mâu thuẫn với giả định rằng $f(x)$ là cực tiểu cục bộ.
Ví dụ, hàm $f(x) = (x+1) (x-1)^2$ có cực tiểu cục bộ tại $x=1$.
Tuy nhiên nó lại không phải là cực tiểu toàn cục.


```{.python .input}
#@tab all
f = lambda x: (x-1)**2 * (x+1)
d2l.set_figsize()
d2l.plot([x, segment], [f(x), f(segment)], 'x', 'f(x)')
```


<!--
The fact that convex functions have no local minima is very convenient.
It means that if we minimize functions we cannot "get stuck".
Note, though, that this does not mean that there cannot be more than one global minimum or that there might even exist one.
For instance, the function $f(x) = \mathrm{max}(|x|-1, 0)$ attains its minimum value over the interval $[-1, 1]$.
Conversely, the function $f(x) = \exp(x)$ does not attain a minimum value on $\mathbb{R}$.
For $x \to -\infty$ it asymptotes to $0$, however there is no $x$ for which $f(x) = 0$.
-->

Tính chất "các hàm lồi không có cực tiểu cục bộ" rất tiện lợi.
Điều này có nghĩa là ta sẽ không bao giờ "mắc kẹt" khi cực tiểu hóa các hàm số.
Dù vậy, hãy lưu ý rằng điều này không có nghĩa là hàm số không thể có nhiều hơn một cực tiểu toàn cục, hoặc liệu hàm số có tồn tại cực tiểu toàn cục hay không.
Ví dụ, hàm $f(x) = \mathrm{max}(|x|-1, 0)$ đạt giá trị nhỏ nhất trên khoảng $[-1, 1]$.
Ngược lại, hàm $f(x) = \exp(x)$ không có giá trị nhỏ nhất trên $\mathbb{R}$.
Với $x \to -\infty$ nó sẽ tiệm cận tới $0$, tuy nhiên không tồn tại giá trị $x$ mà tại đó $f(x) = 0$.

<!--
### Convex Functions and Sets
-->

### Hàm số và Tập hợp Lồi

<!--
Convex functions define convex sets as *below-sets*.
They are defined as
-->

Các hàm số lồi định nghĩa các tập hợp lồi là các *tập-dưới* (*below-sets*) như sau: 

$$S_b := \{x | x \in X \text{ and } f(x) \leq b\}.$$


<!--
Such sets are convex.
Let us prove this quickly.
Remember that for any $x, x' \in S_b$ we need to show that $\lambda x + (1-\lambda) x' \in S_b$ as long as $\lambda \in [0, 1]$.
But this follows directly from the definition of convexity since $f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b$.
-->

Ta hãy chứng minh nó một cách vắn tắt.
Hãy nhớ rằng với mọi $x, x' \in S_b$, ta cần chứng minh $\lambda x + (1-\lambda) x' \in S_b$ với mọi $\lambda \in [0, 1]$.
Nhưng điều này lại trực tiếp tuân theo định nghĩa về tính lồi vì $f(\lambda x + (1-\lambda) x') \leq \lambda f(x) + (1-\lambda) f(x') \leq b$.

<!--
Have a look at the function $f(x, y) = 0.5 x^2 + \cos(2 \pi y)$ below.
It is clearly nonconvex.
The level sets are correspondingly nonconvex.
In fact, they are typically composed of disjoint sets.
-->

Hãy nhìn vào đồ thị hàm $f(x, y) = 0.5 x^2 + \cos(2 \pi y)$ bên dưới.
Nó rõ ràng là không lồi.
Các tập mức tương ứng cũng không lồi.
Thực tế, chúng thường được cấu thành từ các tập hợp rời rạc.



```{.python .input}
#@tab all
x, y = d2l.meshgrid(d2l.linspace(-1, 1, 101), d2l.linspace(-1, 1, 101))
z = x**2 + 0.5 * d2l.cos(2 * np.pi * y)

# Plot the 3D surface
d2l.set_figsize((6, 4))
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.contour(x, y, z, offset=-1)
ax.set_zlim(-1, 1.5)

# Adjust labels
for func in [d2l.plt.xticks, d2l.plt.yticks, ax.set_zticks]:
    func([-1, 0, 1])
```


<!--
### Derivatives and Convexity
-->

### Đạo hàm và tính Lồi

<!--
Whenever the second derivative of a function exists it is very easy to check for convexity.
All we need to do is check whether $\partial_x^2 f(x) \succeq 0$, i.e., whether all of its eigenvalues are nonnegative.
For instance, the function $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2_2$ is convex since $\partial_{\mathbf{x}}^2 f = \mathbf{1}$, i.e., its derivative is the identity matrix.
-->


Bất cứ khi nào đạo hàm bậc hai của một hàm số tồn tại, việc kiểm tra tính lồi của hàm số là rất đơn giản.
Tất cả những gì cần làm là kiểm tra liệu $\partial_x^2 f(x) \succeq 0$, tức là liệu toàn bộ trị riêng của nó đều không âm hay không.
Chẳng hạn, hàm $f(\mathbf{x}) = \frac{1}{2} \|\mathbf{x}\|^2_2$ là lồi vì $\partial_{\mathbf{x}}^2 f = \mathbf{1}$, tức là đạo hàm của nó là ma trận đơn vị.


<!--
The first thing to realize is that we only need to prove this property for one-dimensional functions.
After all, in general we can always define some function $g(z) = f(\mathbf{x} + z \cdot \mathbf{v})$.
This function has the first and second derivatives $g' = (\partial_{\mathbf{x}} f)^\top \mathbf{v}$ and $g'' = \mathbf{v}^\top (\partial^2_{\mathbf{x}} f) \mathbf{v}$ respectively.
In particular, $g'' \geq 0$ for all $\mathbf{v}$ whenever the Hessian of $f$ is positive semidefinite, i.e., whenever all of its eigenvalues are greater equal than zero.
Hence back to the scalar case.
-->


Có thể nhận ra rằng chúng ta chỉ cần chứng minh tính chất này cho các hàm số một chiều.
Xét cho cùng, ta luôn có thể định nghĩa một hàm số $g(z) = f(\mathbf{x} + z \cdot \mathbf{v})$.
Hàm số này có đạo hàm bậc một và bậc hai lần lượt là $g' = (\partial_{\mathbf{x}} f)^\top \mathbf{v}$ và $g'' = \mathbf{v}^\top (\partial^2_{\mathbf{x}} f) \mathbf{v}$.
Cụ thể, $g'' \geq 0$ với mọi $\mathbf{v}$ mỗi khi ma trận Hessian của $f$ là nửa xác định dương, tức là tất cả các trị riêng của ma trận đều lớn hơn hoặc bằng không.
Do đó quay về lại trường hợp vô hướng.


<!--
To see that $f''(x) \geq 0$ for convex functions we use the fact that
-->

Để thấy tại sao $f''(x) \geq 0$ đối với các hàm lồi, ta dùng lập luận 


$$\frac{1}{2} f(x + \epsilon) + \frac{1}{2} f(x - \epsilon) \geq f\left(\frac{x + \epsilon}{2} + \frac{x - \epsilon}{2}\right) = f(x).$$


<!--
Since the second derivative is given by the limit over finite differences it follows that
-->

Vì đạo hàm bậc hai được đưa ra bởi giới hạn trên sai phân hữu hạn, nó dẫn tới


$$f''(x) = \lim_{\epsilon \to 0} \frac{f(x+\epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0.$$


<!--
To see that the converse is true we use the fact that $f'' \geq 0$ implies that $f'$ is a monotonically increasing function.
Let $a < x < b$ be three points in $\mathbb{R}$.
We use the mean value theorem to express
-->

Để chứng minh điều ngược lại, ta dùng lập luận rằng $f'' \geq 0$ ngụ ý rằng $f'$ là một hàm tăng đơn điệu. 
Cho $a < x < b$ là ba điểm thuộc $\mathbb{R}$.
Chúng ta sử dụng định lý giá trị trung bình để biểu diễn 


$$\begin{aligned}
f(x) - f(a) & = (x-a) f'(\alpha) \text{ với } \alpha \in [a, x] \text{ và } \\  
f(b) - f(x) & = (b-x) f'(\beta) \text{ với } \beta \in [x, b]. 
\end{aligned}$$ 


<!--
By monotonicity $f'(\beta) \geq f'(\alpha)$, hence
-->

Từ tính chất đơn điệu $f'(\beta) \geq f'(\alpha)$, ta có 

$$\begin{aligned}
    f(b) - f(a) & = f(b) - f(x) + f(x) - f(a) \\
    & = (b-x) f'(\beta) + (x-a) f'(\alpha) \\
    & \geq (b-a) f'(\alpha).
\end{aligned}$$



<!--
By geometry it follows that $f(x)$ is below the line connecting $f(a)$ and $f(b)$, thus proving convexity.
We omit a more formal derivation in favor of a graph below.
-->

Theo hình học, nó dẫn đến $f(x)$ nằm dưới đường thẳng nối $f(a)$ và $f(b)$, do đó chứng minh được tính lồi.
Ta sẽ bỏ qua việc chứng minh một cách chính quy và thay bằng đồ thị bên dưới.


```{.python .input}
#@tab all
f = lambda x: 0.5 * x**2
x = d2l.arange(-2, 2, 0.01)
axb, ab = d2l.tensor([-1.5, -0.5, 1]), d2l.tensor([-1.5, 1])
d2l.set_figsize()
d2l.plot([x, axb, ab], [f(x) for x in [x, axb, ab]], 'x', 'f(x)')
d2l.annotate('a', (-1.5, f(-1.5)), (-1.5, 1.5))
d2l.annotate('b', (1, f(1)), (1, 1.5))
d2l.annotate('x', (-0.5, f(-0.5)), (-1.5, f(-0.5)))
```

<!--
## Constraints
-->

## Ràng buộc

<!--
One of the nice properties of convex optimization is that it allows us to handle constraints efficiently.
That is, it allows us to solve problems of the form:
-->

Một trong những tính chất hữu ích của tối ưu hóa lồi là nó cho phép chúng ta xử lý các ràng buộc một cách hiệu quả.
Nó cho phép ta giải quyết các bài toán dưới dạng:


$$\begin{aligned} \mathop{\mathrm{~cực~tiểu~hóa~}}_{\mathbf{x}} & f(\mathbf{x}) \\
    \text{~theo~} & c_i(\mathbf{x}) \leq 0 \text{~với~mọi~} i \in \{1, \ldots, N\}.
\end{aligned}$$

<!--
Here $f$ is the objective and the functions $c_i$ are constraint functions.
To see what this does consider the case where $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$.
In this case the parameters $\mathbf{x}$ are constrained to the unit ball.
If a second constraint is $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$, then this corresponds to all $\mathbf{x}$ lying on a halfspace.
Satisfying both constraints simultaneously amounts to selecting a slice of a ball as the constraint set.
-->

$f$ ở đây là mục tiêu và các hàm $c_i$ là các hàm số ràng buộc. 
Hãy xem nó xử lý thế nào trong trường hợp $c_1(\mathbf{x}) = \|\mathbf{x}\|_2 - 1$. 
Ở trường hợp này, các tham số $\mathbf{x}$ bị ràng buộc vào khối cầu đơn vị. 
Nếu ràng buộc thứ hai là $c_2(\mathbf{x}) = \mathbf{v}^\top \mathbf{x} + b$ thì điều này ứng với mọi $\mathbf{x}$ nằm trên nửa khoảng. 
Đáp ứng đồng thời hai ràng buộc này nghĩa là chọn ra một lát cắt của khối cầu làm tập hợp ràng buộc. 

<!--
### Lagrange Function
-->

### Hàm số Lagrange

<!--
In general, solving a constrained optimization problem is difficult.
One way of addressing it stems from physics with a rather simple intuition.
Imagine a ball inside a box.
The ball will roll to the place that is lowest and the forces of gravity will be balanced out with the forces that the sides of the box can impose on the ball.
In short, the gradient of the objective function (i.e., gravity) will be offset by the gradient of the constraint function (need to remain inside the box by virtue of the walls "pushing back")
Note that any constraint that is not active (i.e., the ball does not touch the wall) will not be able to exert any force on the ball.
-->

Nhìn chung, giải quyết một bài toán tối ưu hóa bị ràng buộc là tương đối khó khăn. 
Có một cách giải quyết bắt nguồn từ vật lý dựa trên một trực giác khá đơn giản. 
Hãy tưởng tượng có một quả banh bên trong một chiếc hộp. 
Quả banh sẽ lăn đến nơi thấp nhất và trọng lực sẽ cân bằng với lực nâng của các cạnh hộp tác động lên quả banh. 
Tóm lại, gradient của hàm mục tiêu (ở đây là trọng lực) sẽ được bù lại bởi gradient của hàm ràng buộc (cần phải nằm trong chiếc hộp, bị các bức tưởng "đẩy lại"). 
Lưu ý rằng bất kỳ ràng buộc nào không kích hoạt (quả banh không đụng đến bức tường) thì sẽ không có bất kỳ một lực tác động nào lên quả banh.

<!--
Skipping over the derivation of the Lagrange function $L$ (see e.g., the book by Boyd and Vandenberghe for details :cite:`Boyd.Vandenberghe.2004`) 
the above reasoning can be expressed via the following saddlepoint optimization problem:
-->

Ta hãy bỏ qua phần diễn giải chứng minh của hàm số Lagrange $L$ (Xem sách của Boyd và Vandenberghe về vấn đề này :cite:`Boyd.Vandenberghe.2004`). 
Lý luận bên trên có thể được mô tả thông qua bài toán tối ưu hóa điểm yên ngựa: 


$$L(\mathbf{x},\alpha) = f(\mathbf{x}) + \sum_i \alpha_i c_i(\mathbf{x}) \text{ với } \alpha_i \geq 0.$$
<!-- dịch where -->


<!--
Here the variables $\alpha_i$ are the so-called *Lagrange Multipliers* that ensure that a constraint is properly enforced.
They are chosen just large enough to ensure that $c_i(\mathbf{x}) \leq 0$ for all $i$.
For instance, for any $\mathbf{x}$ for which $c_i(\mathbf{x}) < 0$ naturally, we'd end up picking $\alpha_i = 0$.
Moreover, this is a *saddlepoint* optimization problem where one wants to *maximize* $L$ with respect to $\alpha$ and simultaneously *minimize* it with respect to $\mathbf{x}$.
There is a rich body of literature explaining how to arrive at the function $L(\mathbf{x}, \alpha)$.
For our purposes it is sufficient to know that the saddlepoint of $L$ is where the original constrained optimization problem is solved optimally.
-->

Các biến $\alpha_i$ ở đây được gọi là *nhân tử Lagrange* (*Lagrange Multipliers*), chúng đảm bảo rằng các ràng buộc sẽ được tuân thủ đàng hoàng.
Chúng được chọn vừa đủ lớn để đảm bảo rằng $c_i(\mathbf{x}) \leq 0$ với mọi $i$.
Ví dụ, với mọi $\mathbf{x}$ mà $c_i(\mathbf{x}) < 0$ một cách tự nhiên, chúng ta rốt cuộc sẽ chọn $\alpha_i = 0$.
Hơn nữa, đây là bài toán tối ưu hóa *điểm yên ngựa*, nơi ta muốn *cực đại hóa* $L$ theo $\alpha$ và đồng thời *cực tiểu hóa* nó theo $\mathbf{x}$.
Có rất nhiều tài liệu giải thích về cách đưa đến hàm $L(\mathbf{x}, \alpha)$.
Đối với mục đích của chúng ta, sẽ là đủ khi biết rằng điểm yên ngựa của $L$ là nơi bài toán tối ưu hóa bị ràng buộc ban đầu được giải quyết một cách tối ưu.

<!--
### Penalties
-->

### Lượng phạt

<!--
One way of satisfying constrained optimization problems at least approximately is to adapt the Lagrange function $L$.
Rather than satisfying $c_i(\mathbf{x}) \leq 0$ we simply add $\alpha_i c_i(\mathbf{x})$ to the objective function $f(x)$.
This ensures that the constraints will not be violated too badly.
-->

Có một cách để thỏa mãn, ít nhất là theo xấp xỉ, các bài toán tối ưu hóa bị ràng buộc là phỏng theo hàm Lagrange $L$. 
Thay vì thỏa mãn $c_i(\mathbf{x}) \leq 0$, chúng ta chỉ cần thêm $\alpha_i c_i(\mathbf{x})$ vào hàm mục tiêu $f(x)$. 
Điều này sẽ đảm bảo rằng các ràng buộc không bị vi phạm quá mức. 

<!--
In fact, we have been using this trick all along.
Consider weight decay in :numref:`sec_weight_decay`.
In it we add $\frac{\lambda}{2} \|\mathbf{w}\|^2$ to the objective function to ensure that $\mathbf{w}$ does not grow too large.
Using the constrained optimization point of view we can see that this will ensure that $\|\mathbf{w}\|^2 - r^2 \leq 0$ for some radius $r$.
Adjusting the value of $\lambda$ allows us to vary the size of $\mathbf{w}$.
-->

Thực tế, chúng ta đã dùng thủ thuật này khá thường xuyên.
Hãy xét đến suy giảm trọng số trong :numref:`sec_weight_decay`.
Ở đó chúng ta thêm $\frac{\lambda}{2} \|\mathbf{w}\|^2$ vào hàm mục tiêu để đảm bảo rằng giá trị $\mathbf{w}$ không trở nên quá lớn.
Dưới góc nhìn tối ưu hóa có ràng buộc, ta có thể thấy nó sẽ đảm bảo $\|\mathbf{w}\|^2 - r^2 \leq 0$ với giá trị bán kính $r$ nào đó.
Điều chỉnh giá trị của $\lambda$ cho phép chúng ta thay đổi độ lớn của $\mathbf{w}$.

<!--
In general, adding penalties is a good way of ensuring approximate constraint satisfaction.
In practice this turns out to be much more robust than exact satisfaction.
Furthermore, for nonconvex problems many of the properties that make the exact approach so appealing in the convex case (e.g., optimality) no longer hold.
-->

Nhìn chung, thêm các lượng phạt là một cách tốt để đảm bảo việc thỏa mãn ràng buộc xấp xỉ.
Trong thực tế, hóa ra phương pháp này ổn định hơn rất nhiều so với trường hợp thỏa mãn chuẩn xác.
Hơn nữa, với các bài toán không lồi, những tính chất khiến phương án tiếp cận chuẩn xác trở nên rất thu hút trong trường hợp lồi (ví dụ như tính tối ưu) không còn đảm bảo nữa.

<!--
### Projections
-->

### Các phép chiếu

<!--
An alternative strategy for satisfying constraints are projections.
Again, we encountered them before, e.g., when dealing with gradient clipping in :numref:`sec_rnn_scratch`.
There we ensured that a gradient has length bounded by $c$ via
-->

Một chiến lược khác để thỏa mãn các ràng buộc là các phép chiếu.
Chúng ta cũng đã gặp chúng trước đây, ví dụ như khi bàn về phương pháp gọt gradient ở :numref:`sec_rnn_scratch`.
Ở phần đó chúng ta đã đảm bảo rằng gradient có độ dài ràng buộc bởi $c$ thông qua 


$$\mathbf{g} \leftarrow \mathbf{g} \cdot \mathrm{min}(1, c/\|\mathbf{g}\|).$$


<!--
This turns out to be a *projection* of $g$ onto the ball of radius $c$. More generally, a projection on a (convex) set $X$ is defined as
-->

Hóa ra đây là một *phép chiếu* của $g$ lên khối cầu có bán kính $c$. Tổng quát hơn, một phép chiếu lên một tập (lồi) $X$ được định nghĩa là 


$$\mathrm{Proj}_X(\mathbf{x}) = \mathop{\mathrm{argmin}}_{\mathbf{x}' \in X} \|\mathbf{x} - \mathbf{x}'\|_2.$$


<!--
It is thus the closest point in $X$ to $\mathbf{x}$.
This sounds a bit abstract.
:numref:`fig_projections` explains it somewhat more clearly.
In it we have two convex sets, a circle and a diamond.
Points inside the set (yellow) remain unchanged.
Points outside the set (black) are mapped to the closest point inside the set (red).
While for $\ell_2$ balls this leaves the direction unchanged, this need not be the case in general, as can be seen in the case of the diamond.
-->

Do đó đây là điểm gần nhất trong $X$ tới $\mathbf{x}$.
Điều này nghe có vẻ hơi trừu tượng.
:numref:`fig_projections` sẽ giải thích nó một cách rõ ràng hơn.
Ở đó ta có hai tập lồi, một hình tròn và một hình thoi.
Các điểm nằm bên trong tập (màu vàng) giữ nguyên không đổi.
Các điểm nằm bên ngoài tập (màu đen) được ánh xạ tới điểm gần nhất bên trong tập (màu đỏ).
Trong khi với các khối cầu $\ell_2$ hướng của phép chiếu được giữ nguyên không đổi, điều này có thể không đúng trong trường hợp tổng quát, như có thể thấy trong trường hợp của hình thoi.

<!--
![Convex Projections](../img/projections.svg)
-->

![Các phép chiếu lồi](../img/projections.svg)
:label:`fig_projections`

<!--
One of the uses for convex projections is to compute sparse weight vectors.
In this case we project $\mathbf{w}$ onto an $\ell_1$ ball (the latter is a generalized version of the diamond in the picture above).
-->

Một trong những ứng dụng của các phép chiếu lồi là để tính toán các vector trọng số thưa.
Trong trường hợp này chúng ta chiếu $\mathbf{w}$ lên khối cầu $\ell_1$ (phiên bản tổng quát của hình thoi ở hình minh họa phía trên).


<!--
## Summary
-->

## Tóm tắt

<!--
In the context of deep learning the main purpose of convex functions is to motivate optimization algorithms and help us understand them in detail.
In the following we will see how gradient descent and stochastic gradient descent can be derived accordingly.
-->

Trong bối cảnh học sâu, mục đích chính của các hàm lồi là để thúc đẩy sự phát triển các thuật toán tối ưu hóa và giúp ta hiểu chúng một cách chi tiết.
Phần tiếp theo chúng ta sẽ thấy cách mà hạ gradient và hạ gradient ngẫu nhiên có thể được suy ra từ đó.

<!--
* Intersections of convex sets are convex. Unions are not.
* The expectation of a convex function is larger than the convex function of an expectation (Jensen's inequality).
* A twice-differentiable function is convex if and only if its second derivative has only nonnegative eigenvalues throughout.
* Convex constraints can be added via the Lagrange function. In practice simply add them with a penalty to the objective function.
* Projections map to points in the (convex) set closest to the original point.
-->

* Giao của các tập lồi là tập lồi. Hợp của các tập lồi không bắt buộc phải là tập lồi.
* Kỳ vọng của hàm lồi lớn hơn hàm lồi của kỳ vọng (Bất đẳng thức Jensen).
* Hàm khả vi hai lần là hàm lồi khi và chỉ khi đạo hàm bậc hai của nó chỉ có các trị riêng không âm ở mọi nơi.
* Các ràng buộc lồi có thể được thêm vào hàm Lagrange. Trong thực tế, ta chỉ việc thêm chúng cùng với một mức phạt vào hàm mục tiêu.
* Các phép chiếu ánh xạ đến các điểm trong tập (lồi) nằm gần nhất với điểm gốc.

<!--
## Exercises
-->

## Bài tập

<!--
1. Assume that we want to verify convexity of a set by drawing all lines between points within the set and checking whether the lines are contained.
    * Prove that it is sufficient to check only the points on the boundary.
    * Prove that it is sufficient to check only the vertices of the set.
2. Denote by $B_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ and } \|\mathbf{x}\|_p \leq r\}$ the ball of radius $r$ using the $p$-norm. Prove that $B_p[r]$ is convex for all $p \geq 1$.
3. Given convex functions $f$ and $g$ show that $\mathrm{max}(f, g)$ is convex, too. Prove that $\mathrm{min}(f, g)$ is not convex.
4. Prove that the normalization of the softmax function is convex. More specifically prove the convexity of $f(x) = \log \sum_i \exp(x_i)$.
5. Prove that linear subspaces are convex sets, i.e., $X = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$.
6. Prove that in the case of linear subspaces with $\mathbf{b} = 0$ the projection $\mathrm{Proj}_X$ can be written as $\mathbf{M} \mathbf{x}$ for some matrix $\mathbf{M}$.
7. Show that for convex twice differentiable functions $f$ we can write $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ for some $\xi \in [0, \epsilon]$.
8. Given a vector $\mathbf{w} \in \mathbb{R}^d$ with $\|\mathbf{w}\|_1 > 1$ compute the projection on the $\ell_1$ unit ball.
    * As intermediate step write out the penalized objective $\|\mathbf{w} - \mathbf{w}'\|_2^2 + \lambda \|\mathbf{w}'\|_1$ and compute the solution for a given $\lambda > 0$.
    * Can you find the 'right' value of $\lambda$ without a lot of trial and error?
9. Given a convex set $X$ and two vectors $\mathbf{x}$ and $\mathbf{y}$ prove that projections never increase distances, i.e., $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_X(\mathbf{x}) - \mathrm{Proj}_X(\mathbf{y})\|$.
-->

1. Giả sử chúng ta muốn xác minh tính lồi của tập hợp bằng cách vẽ mọi đoạn thẳng giữa các điểm bên trong tập hợp và kiểm tra liệu các đoạn thẳng có nằm trong tập hợp đó hay không.
    * Hãy chứng mình rằng ta chỉ cần kiểm tra các điểm ở biên là đủ.
    * Hãy chứng minh rằng ta chỉ cần kiểm tra các đỉnh của tập hợp là đủ.
2. Ký hiệu khối cầu có bán kính $r$ sử dụng chuẩn $p$ là $B_p[r] := \{\mathbf{x} | \mathbf{x} \in \mathbb{R}^d \text{ và } \|\mathbf{x}\|_p \leq r\}$. Hãy chứng minh rằng $B_p[r]$ là lồi với mọi $p \geq 1$. 
3. Cho các hàm lồi $f$ và $g$ sao cho $\mathrm{max}(f, g)$ cũng là hàm lồi. Hãy chứng minh rằng $\mathrm{min}(f, g)$ không lồi.
4. Hãy chứng minh rằng hàm softmax được chuẩn hóa là hàm lồi. Cụ thể hơn, chứng minh tính lồi của $f(x) = \log \sum_i \exp(x_i)$.
5. Hãy chứng minh rằng các không gian con tuyến tính là các tập lồi. Ví dụ, $X = \{\mathbf{x} | \mathbf{W} \mathbf{x} = \mathbf{b}\}$.
6. Hãy chứng minh rằng trong trường hợp của các không gian con tuyến tính với $\mathbf{b} = 0$, phép chiếu $\mathrm{Proj}_X$ có thể được viết dưới dạng $\mathbf{M} \mathbf{x}$ với một ma trận $\mathbf{M}$ nào đó.
7. Hãy chỉ ra rằng với các hàm số khả vi hai lần $f$, ta có thể viết $f(x + \epsilon) = f(x) + \epsilon f'(x) + \frac{1}{2} \epsilon^2 f''(x + \xi)$ với một giá trị $\xi \in [0, \epsilon]$ nào đó.
8. Cho vector $\mathbf{w} \in \mathbb{R}^d$ với $\|\mathbf{w}\|_1 > 1$, hãy tính phép chiếu lên khối cầu đơn vị $\ell_1$.
    * Như một bước trung gian, hãy viết ra mục tiêu có lượng phạt $\|\mathbf{w} - \mathbf{w}'\|_2^2 + \lambda \|\mathbf{w}'\|_1$ và tính ra đáp án với $\lambda > 0$.
    * Bạn có thể tìm ra giá trị 'chính xác' của $\lambda$ mà không phải đoán mò quá nhiều lần không?
9. Cho tập lồi $X$ và hai vector $\mathbf{x}$, $\mathbf{y}$, hãy chứng minh rằng các phép chiếu không bao giờ làm tăng khoảng cách, ví dụ, $\|\mathbf{x} - \mathbf{y}\| \geq \|\mathrm{Proj}_X(\mathbf{x}) - \mathrm{Proj}_X(\mathbf{y})\|$.


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/350)
* [Tiếng Anh - Pytorch](https://discuss.d2l.ai/t/488)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
* Nguyễn Văn Quang
* Nguyễn Lê Quang Nhật
* Phạm Minh Đức
* Võ Tấn Phát

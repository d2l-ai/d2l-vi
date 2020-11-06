<!--
# Multivariable Calculus
-->

# Giải tích Nhiều biến
:label:`sec_multivariable_calculus`


<!--
Now that we have a fairly strong understanding of derivatives of a function of a single variable, 
let us return to our original question where we were considering a loss function of potentially billions of weights.
-->

Bây giờ chúng ta đã có hiểu biết vững chắc về đạo hàm của một hàm đơn biến,
hãy cùng trở lại câu hỏi ban đầu về hàm mất mát của (nhiều khả năng là) hàng tỷ trọng số.


<!--
## Higher-Dimensional Differentiation
-->

## Đạo hàm trong Không gian Nhiều chiều


<!--
What :numref:`sec_single_variable_calculus` tells us is that if we change a single one of these billions of weights leaving every other one fixed, we know what will happen!
This is nothing more than a function of a single variable, so we can write
-->

Nhớ lại :numref:`sec_single_variable_calculus`, ta đã bàn luận về điều gì sẽ xảy ra nếu chỉ thay đổi một trong số hàng tỷ các trọng số và giữ nguyên những trọng số còn lại.
Điều này hoàn toàn không có gì khác với một hàm đơn biến, nên ta có thể viết

$$L(w_1+\epsilon_1, w_2, \ldots, w_N) \approx L(w_1, w_2, \ldots, w_N) + \epsilon_1 \frac{d}{dw_1} L(w_1, w_2, \ldots, w_N).$$
:eqlabel:`eq_part_der`


<!--
We will call the derivative in one variable while fixing the other the *partial derivative*, 
and we will use the notation $\frac{\partial}{\partial w_1}$ for the derivative in :eqref:`eq_part_der`.
-->

Chúng ta sẽ gọi đạo hàm của một biến trong khi không thay đổi những biến còn lại là *đạo hàm riêng* (*partial derivative*), 
và ký hiệu đạo hàm này là $\frac{\partial}{\partial w_1}$ trong phương trình :eqref:`eq_part_der`. 

<!--
Now, let us take this and change $w_2$ a little bit to $w_2 + \epsilon_2$:
-->

Bây giờ, tiếp tục thay đổi $w_2$ một khoảng nhỏ thành $w_2 + \epsilon_2$:


$$
\begin{aligned}
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N) & \approx L(w_1, w_2+\epsilon_2, \ldots, w_N) + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2+\epsilon_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1\epsilon_2\frac{\partial}{\partial w_2}\frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N) \\
& \approx L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_2\frac{\partial}{\partial w_2} L(w_1, w_2, \ldots, w_N) \\
& \quad + \epsilon_1 \frac{\partial}{\partial w_1} L(w_1, w_2, \ldots, w_N).
\end{aligned}
$$


<!--
We have again used the idea that $\epsilon_1\epsilon_2$ is a higher order term that we can discard in 
the same way we could discard $\epsilon^{2}$ in the previous section, along with what we saw in :eqref:`eq_part_der`.
By continuing in this manner, we may write that
-->

Một lần nữa, ta lại sử dụng ý tưởng đã thấy ở :eqref:`eq_part_der` rằng $\epsilon_1\epsilon_2$ là một số hạng bậc cao 
và có thể được loại bỏ tương tự như cách mà ta có thể loại bỏ $\epsilon^{2}$ trong mục trước.
Cứ tiếp tục theo cách này, ta có


$$
L(w_1+\epsilon_1, w_2+\epsilon_2, \ldots, w_N+\epsilon_N) \approx L(w_1, w_2, \ldots, w_N) + \sum_i \epsilon_i \frac{\partial}{\partial w_i} L(w_1, w_2, \ldots, w_N).
$$


<!--
This may look like a mess, but we can make this more familiar by noting that the sum on the right looks exactly like a dot product, so if we let
-->

Thoạt nhìn đây có vẻ là một mớ hỗn độn, tuy nhiên chú ý rằng phép tổng bên phải chính là biểu diễn của phép tích vô hướng và ta có thể khiến chúng trở nên quen thuộc hơn. Với


$$
\boldsymbol{\epsilon} = [\epsilon_1, \ldots, \epsilon_N]^\top \; \text{và} \;
\nabla_{\mathbf{x}} L = \left[\frac{\partial L}{\partial x_1}, \ldots, \frac{\partial L}{\partial x_N}\right]^\top,
$$


<!--
then
-->

ta có


$$L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).$$
:eqlabel:`eq_nabla_use`


<!--
We will call the vector $\nabla_{\mathbf{w}} L$ the *gradient* of $L$.
-->

Ta gọi vector $\nabla_{\mathbf{w}} L$ là *gradient* của $L$.


<!--
Equation :eqref:`eq_nabla_use` is worth pondering for a moment.
It has exactly the format that we encountered in one dimension, just we have converted everything to vectors and dot products.
It allows us to tell approximately how the function $L$ will change given any perturbation to the input. 
As we will see in the next section, this will provide us with an important tool in understanding geometrically how we can learn using information contained in the gradient.
-->

Phương trình :eqref:`eq_nabla_use` đáng để ta suy ngẫm.
Nó có dạng đúng y như những gì ta đã thấy trong trường hợp một chiều, chỉ khác là tất cả đã được biến đổi về dạng vector và tích vô hướng.
Điều này cho chúng ta biết một cách xấp xỉ hàm $L$ sẽ thay đổi như thế nào với một nhiễu loạn bất kỳ ở đầu vào.
Như ta sẽ thấy trong mục tiếp theo, đây sẽ là một công cụ quan trọng giúp chúng ta hiểu được cách học từ thông tin chứa trong gradient dưới góc nhìn hình học.


<!--
But first, let us see this approximation at work with an example.
Suppose that we are working with the function
-->

Nhưng trước tiên, hãy cùng kiểm tra phép xấp xỉ này với một ví dụ.
Giả sử ta đang làm việc với hàm


$$
f(x, y) = \log(e^x + e^y) \text{ với gradient } \nabla f (x, y) = \left[\frac{e^x}{e^x+e^y}, \frac{e^y}{e^x+e^y}\right].
$$


<!--
If we look at a point like $(0, \log(2))$, we see that
-->

Xét một điểm $(0, \log(2))$, ta có


$$
f(x, y) = \log(3) \text{ với gradient } \nabla f (x, y) = \left[\frac{1}{3}, \frac{2}{3}\right].
$$


<!--
Thus, if we want to approximate $f$ at $(\epsilon_1, \log(2) + \epsilon_2)$,
we see that we should have the specific instance of :eqref:`eq_nabla_use`:
-->

Vì thế, nếu muốn tính xấp xỉ $f$ tại $(\epsilon_1, \log(2) + \epsilon_2)$, ta có một ví dụ cụ thể của :eqref:`eq_nabla_use`:


$$
f(\epsilon_1, \log(2) + \epsilon_2) \approx \log(3) + \frac{1}{3}\epsilon_1 + \frac{2}{3}\epsilon_2.
$$


<!--
We can test this in code to see how good the approximation is.
-->

Ta có thể kiểm tra với đoạn mã bên dưới để xem phép xấp xỉ chính xác tới mức nào.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import autograd, np, npx
npx.set_np()

def f(x, y):
    return np.log(np.exp(x) + np.exp(y))
def grad_f(x, y):
    return np.array([np.exp(x) / (np.exp(x) + np.exp(y)),
                     np.exp(y) / (np.exp(x) + np.exp(y))])

epsilon = np.array([0.01, -0.03])
grad_approx = f(0, np.log(2)) + epsilon.dot(grad_f(0, np.log(2)))
true_value = f(0 + epsilon[0], np.log(2) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch
import numpy as np

def f(x, y):
    return torch.log(torch.exp(x) + torch.exp(y))
def grad_f(x, y):
    return torch.tensor([torch.exp(x) / (torch.exp(x) + torch.exp(y)),
                     torch.exp(y) / (torch.exp(x) + torch.exp(y))])

epsilon = torch.tensor([0.01, -0.03])
grad_approx = f(torch.tensor([0.]), torch.log(
    torch.tensor([2.]))) + epsilon.dot(
    grad_f(torch.tensor([0.]), torch.log(torch.tensor(2.))))
true_value = f(torch.tensor([0.]) + epsilon[0], torch.log(
    torch.tensor([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf
import numpy as np

def f(x, y):
    return tf.math.log(tf.exp(x) + tf.exp(y))
def grad_f(x, y):
    return tf.constant([(tf.exp(x) / (tf.exp(x) + tf.exp(y))).numpy(),
                        (tf.exp(y) / (tf.exp(x) + tf.exp(y))).numpy()])

epsilon = tf.constant([0.01, -0.03])
grad_approx = f(tf.constant([0.]), tf.math.log(
    tf.constant([2.]))) + tf.tensordot(
    epsilon, grad_f(tf.constant([0.]), tf.math.log(tf.constant(2.))), axes=1)
true_value = f(tf.constant([0.]) + epsilon[0], tf.math.log(
    tf.constant([2.])) + epsilon[1])
f'approximation: {grad_approx}, true Value: {true_value}'
```


<!--
## Geometry of Gradients and Gradient Descent
-->

## Ý nghĩa Hình học của Gradient và Thuật toán Hạ Gradient 


<!--
Consider the again :eqref:`eq_nabla_use`:
-->

Nhìn lại :eqref:`eq_nabla_use`:


$$
L(\mathbf{w} + \boldsymbol{\epsilon}) \approx L(\mathbf{w}) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}).
$$


<!--
Let us suppose that I want to use this to help minimize our loss $L$.
Let us understand geometrically the algorithm of gradient descent first described in :numref:`sec_autograd`.
What we will do is the following:
-->

Giả sử ta muốn sử dụng thông tin gradient để cực tiểu hóa mất mát $L$. 
Hãy cùng tìm hiểu cách hoạt động về mặt hình học của thuật toán hạ gradient được mô tả lần đầu ở :numref:`sec_autograd`. 
Các bước của thuật toán được miêu tả dưới đây:


<!--
1. Start with a random choice for the initial parameters $\mathbf{w}$.
2. Find the direction $\mathbf{v}$ that makes $L$ decrease the most rapidly at $\mathbf{w}$.
3. Take a small step in that direction: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$.
4. Repeat.
-->

1. Bắt đầu với giá trị ban đầu ngẫu nhiên của tham số $\mathbf{w}$. 
2. Tìm một hướng $\mathbf{v}$ tại $\mathbf{w}$ sao cho $L$ giảm một cách nhanh nhất. 
3. Tiến một bước nhỏ về hướng đó: $\mathbf{w} \rightarrow \mathbf{w} + \epsilon\mathbf{v}$. 
4. Lặp lại. 

<!--
The only thing we do not know exactly how to do is to compute the vector $\mathbf{v}$ in the second step.
We will call such a direction the *direction of steepest descent*.
Using the geometric understanding of dot products from :numref:`sec_geometry-linear-algebraic-ops`, we see that we can rewrite :eqref:`eq_nabla_use` as
-->

Thứ duy nhất mà chúng ta không biết chính xác cách làm là cách tính toán vector $\mathbf{v}$ tại bước thứ hai.
Ta gọi $\mathbf{v}$ là *hướng hạ dốc nhất* (*direction of steepest descent*).
Sử dụng những hiểu biết về mặt hình học của phép tích vô hướng từ :numref:`sec_geometry-linear-algebraic-ops`, ta có thể viết lại :eqref:`eq_nabla_use` như sau


$$
L(\mathbf{w} + \mathbf{v}) \approx L(\mathbf{w}) + \mathbf{v}\cdot \nabla_{\mathbf{w}} L(\mathbf{w}) = L(\mathbf{w}) + \|\nabla_{\mathbf{w}} L(\mathbf{w})\|\cos(\theta).
$$


<!--
Note that we have taken our direction to have length one for convenience, and used $\theta$ for the angle between $\mathbf{v}$ and $\nabla_{\mathbf{w}} L(\mathbf{w})$.
If we want to find the direction that decreases $L$ as rapidly as possible, we want to make this expression as negative as possible.
The only way the direction we pick enters into this equation is through $\cos(\theta)$, and thus we wish to make this cosine as negative as possible.
Now, recalling the shape of cosine, we can make this as negative as possible by making $\cos(\theta) = -1$ or equivalently 
making the angle between the gradient and our chosen direction to be $\pi$ radians, or equivalently $180$ degrees.
The only way to achieve this is to head in the exact opposite direction:
pick $\mathbf{v}$ to point in the exact opposite direction to $\nabla_{\mathbf{w}} L(\mathbf{w})$!
-->

Để thuận tiện, ta giả định hướng của chúng ta có độ dài bằng một và sử dụng $\theta$ để biểu diễn góc giữa $\mathbf{v}$ và $\nabla_{\mathbf{w}} L(\mathbf{w})$. 
Nếu muốn $L$ giảm càng nhanh, ta sẽ muốn giá trị của biểu thức trên càng âm càng tốt. 
Cách duy nhất để chọn hướng đi trong phương trình này là thông qua $\cos(\theta)$, vì thế ta sẽ muốn giá trị này âm nhất có thể. 
Nhắc lại kiến thức của hàm cô-sin, giá trị âm nhất của hàm này là $\cos(\theta) = -1$, là khi góc giữa vector gradient và hướng cần chọn là $\pi$ radian hay $180$ độ. 
Cách duy nhất để đạt được điều này là di chuyển theo hướng hoàn toàn ngược lại:
chọn $\mathbf{v}$ theo hướng hoàn toàn ngược chiều với $\nabla_{\mathbf{w}} L(\mathbf{w})$! 


<!--
This brings us to one of the most important mathematical concepts in machine learning: 
the direction of steepest decent points in the direction of $-\nabla_{\mathbf{w}}L(\mathbf{w})$.
Thus our informal algorithm can be rewritten as follows.
-->

Điều này dẫn ta đến với một trong những thuật toán quan trọng nhất của học máy:
hướng hạ dốc nhất cùng hướng với $-\nabla_{\mathbf{w}}L(\mathbf{w})$.
Vậy nên thuật toán của ta sẽ được viết lại như sau.


<!--
1. Start with a random choice for the initial parameters $\mathbf{w}$.
2. Compute $\nabla_{\mathbf{w}} L(\mathbf{w})$.
3. Take a small step in the opposite of that direction: $\mathbf{w} \rightarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. Repeat.
-->

1. Bắt đầu với một lựa chọn ngẫu nhiên cho giá trị ban đầu của các tham số $\mathbf{w}$.
2. Tính toán $\nabla_{\mathbf{w}} L(\mathbf{w})$.
3. Tiến một bước nhỏ về hướng ngược lại của nó: $\mathbf{w} \rightarrow \mathbf{w} - \epsilon\nabla_{\mathbf{w}} L(\mathbf{w})$.
4. Lặp lại.


<!--
This basic algorithm has been modified and adapted many ways by many researchers, but the core concept remains the same in all of them.
Use the gradient to find the direction that decreases the loss as rapidly as possible, and update the parameters to take a step in that direction.
-->

Thuật toán cơ bản này dù đã được chỉnh sửa và kết hợp theo nhiều cách bởi các nhà nghiên cứu, nhưng khái niệm cốt lõi vẫn là như nhau.
Sử dụng gradient để tìm hướng giảm mất mát nhanh nhất có thể và cập nhật các tham số để dịch chuyển về hướng đó.


<!--
## A Note on Mathematical Optimization
-->

## Một vài chú ý về Tối ưu hóa


<!--
Throughout this book, we focus squarely on numerical optimization techniques for the practical reason that all functions 
we encounter in the deep learning setting are too complex to minimize explicitly.
-->

Xuyên suốt cuốn sách, ta chỉ tập trung vào những kỹ thuật tối ưu hóa số học vì một nguyên nhân thực tế là: 
mọi hàm ta gặp phải trong học sâu quá phức tạp để có thể tối ưu hóa một cách tường minh.


<!--
However, it is a useful exercise to consider what the geometric understanding we obtained above tells us about optimizing functions directly.
-->

Tuy nhiên, sẽ rất hữu ích nếu hiểu được những kiến thức hình học ta có được ở trên nói gì về tối ưu hóa các hàm một cách trực tiếp.


<!--
Suppose that we wish to find the value of $\mathbf{x}_0$ which minimizes some function $L(\mathbf{x})$.
Let us suppose that moreover someone gives us a value and tells us that it is the value that minimizes $L$.
Is there anything we can check to see if their answer is even plausible?
-->

Giả sử ta muốn tìm giá trị của $\mathbf{x}_0$ giúp cực tiểu hóa một hàm $L(\mathbf{x})$ nào đó.
Và có một người nào đó đưa ta một giá trị và cho rằng đây là giá trị giúp cực tiểu hóa $L$.
Bằng cách nào ta có thể kiểm chứng rằng đáp án của họ là hợp lý?


<!--
Again consider :eqref:`eq_nabla_use`:
-->

Xét lại :eqref:`eq_nabla_use`:


$$
L(\mathbf{x}_0 + \boldsymbol{\epsilon}) \approx L(\mathbf{x}_0) + \boldsymbol{\epsilon}\cdot \nabla_{\mathbf{x}} L(\mathbf{x}_0).
$$


<!--
If the gradient is not zero, we know that we can take a step in the direction $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ to find a value of $L$ that is smaller.
Thus, if we truly are at a minimum, this cannot be the case!
We can conclude that if $\mathbf{x}_0$ is a minimum, then $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$.
We call points with $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ *critical points*.
-->

Nếu giá trị gradient khác không, ta biết rằng ta có thể bước một bước về hướng $-\epsilon \nabla_{\mathbf{x}} L(\mathbf{x}_0)$ để tìm một giá trị $L$ nhỏ hơn.
Do đó, nếu ta thực sự ở điểm cực tiểu, sẽ không thể có trường hợp đó!
Ta có thể kết luận rằng nếu $\mathbf{x}_0$ là một cực tiểu, thì $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$.
Ta gọi những điểm mà tại đó $\nabla_{\mathbf{x}} L(\mathbf{x}_0) = 0$ là *các điểm tới hạn* (*critical points*).


<!--
This is nice, because in some rare settings, we *can* explicitly find all the points where the gradient is zero, and find the one with the smallest value.  
-->

Điều này rất hữu ích, bởi vì trong một vài thiết lập hiếm gặp, ta *có thể* tìm được các điểm có gradient bằng không một cách tường minh, và từ đó tìm được điểm có giá trị nhỏ nhất.


<!--
For a concrete example, consider the function
-->

Với một ví dụ cụ thể, xét hàm


$$
f(x) = 3x^4 - 4x^3 -12x^2.
$$


<!--
This function has derivative
-->

Hàm này có đạo hàm


$$
\frac{df}{dx} = 12x^3 - 12x^2 -24x = 12x(x-2)(x+1).
$$


<!--
The only possible location of minima are at $x = -1, 0, 2$, where the function takes the values $-5,0, -32$ respectively, 
and thus we can conclude that we minimize our function when $x = 2$. A quick plot confirms this.
-->

Các điểm cực trị duy nhất khả dĩ là tại $x = -1, 0, 2$, khi hàm lấy giá trị lần lượt là $-5,0, -32$, 
và do đó ta có thể kết luận rằng ta cực tiểu hóa hàm khi $x = 2$. Ta có thể kiểm chứng nhanh bằng đồ thị. 


```{.python .input}
x = np.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 3, 0.01)
f = (3 * x**4) - (4 * x**3) - (12 * x**2)

d2l.plot(x, f, 'x', 'f(x)')
```


<!--
This highlights an important fact to know when working either theoretically or numerically: 
the only possible points where we can minimize (or maximize) a function will have gradient equal to zero, 
however, not every point with gradient zero is the true *global* minimum (or maximum).
-->

Điều này nhấn mạnh một thực tế quan trọng cần biết kể cả khi làm việc dưới dạng lý thuyết hay số học: 
các điểm khả dĩ duy nhất mà tại đó hàm là cực tiểu (hoặc cực đại) sẽ có đạo hàm tại đó bằng không, 
tuy nhiên, không phải tất cả các điểm có đạo hàm bằng không sẽ là cực tiểu (hay cực đại) *toàn cục*. 


<!--
## Multivariate Chain Rule
-->

## Quy tắc Dây chuyền cho Hàm đa biến


<!--
Let us suppose that we have a function of four variables ($w, x, y$, and $z$) which we can make by composing many terms:
-->

Giả sử là ta có một hàm bốn biến ($w, x, y$, and $z$) được tạo ra bằng cách kết hợp các hàm con:


$$\begin{aligned}f(u, v) & = (u+v)^{2} \\u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.\end{aligned}$$
:eqlabel:`eq_multi_func_def`


<!--
Such chains of equations are common when working with neural networks, so trying to understand how to compute gradients of such functions is key.
We can start to see visual hints of this connection in :numref:`fig_chain-1` if we take a look at what variables directly relate to one another.
-->

Các chuỗi phương trình như trên xuất hiện thường xuyên khi ta làm việc với các mạng nơ-ron, do đó cố gắng hiểu xem làm thế nào để tính gradient của các hàm này là thiết yếu.
:numref:`fig_chain-1` biểu diễn trực quan mỗi liên hệ trực tiếp giữa biến này với biến khác.


<!--
![The function relations above where nodes represent values and edges show functional dependence.](../img/chain-net1.svg)
-->

![Các quan hệ của hàm ở trên với các nút biểu diễn giá trị và mũi tên cho biết sự phụ thuộc hàm.](../img/chain-net1.svg)
:label:`fig_chain-1`


<!--
Nothing stops us from just composing everything from :eqref:`eq_multi_func_def` and writing out that
-->

Ta có thể kết hợp các phương trình trong :eqref:`eq_multi_func_def` để có 


$$
f(w, x, y, z) = \left(\left((w+x+y+z)^2+(w+x-y-z)^2\right)^2+\left((w+x+y+z)^2-(w+x-y-z)^2\right)^2\right)^2.
$$


<!--
We may then take the derivative by just using single variable derivatives, 
but if we did that we would quickly find ourself swamped with terms, many of which are repeats!
Indeed, one can see that, for instance:
-->

Tiếp theo ta có thể lấy đạo hàm bằng cách chỉ sử dụng các đạo hàm đơn biến,
nhưng nếu làm vậy ta sẽ nhanh chóng bị ngợp trong các số hạng, mà đa phần là bị lặp lại!
Thật vậy, ta có thể thấy ở ví dụ dưới đây:


$$
\begin{aligned}
\frac{\partial f}{\partial w} & = 2 \left(2 \left(2 (w + x + y + z) - 2 (w + x - y - z)\right) \left((w + x + y + z)^{2}- (w + x - y - z)^{2}\right) + \right.\\
& \left. \quad 2 \left(2 (w + x - y - z) + 2 (w + x + y + z)\right) \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)\right) \times \\
& \quad \left(\left((w + x + y + z)^{2}- (w + x - y - z)^2\right)^{2}+ \left((w + x - y - z)^{2}+ (w + x + y + z)^{2}\right)^{2}\right).
\end{aligned}
$$


<!--
If we then also wanted to compute $\frac{\partial f}{\partial x}$, we would end up with a similar equation again with many repeated terms, 
and many *shared* repeated terms between the two derivatives.
This represents a massive quantity of wasted work, and if we needed to compute derivatives this way, 
the whole deep learning revolution would have stalled out before it began!
-->

Kế đến nếu ta cũng muốn tính $\frac{\partial f}{\partial x}$, ta sẽ lại kết thúc với một phương trình tương tự với nhiều thành phần bị lặp lại,
và nhiều thành phần lặp lại *chung* giữa hai đạo hàm.
Điều này thể hiện một khối lượng lớn công việc bị lãng phí, và nếu ta tính các đạo hàm theo cách này,
toàn bộ cuộc cách mạng học sâu sẽ chấm dứt trước khi nó bắt đầu!


<!--
Let us break up the problem.
We will start by trying to understand how $f$ changes when we change $a$, essentially assuming that $w, x, y$, and $z$ all do not exist.
We will reason as we did back when we worked with the gradient for the first time. Let us take $a$ and add a small amount $\epsilon$ to it.
-->

Ta hãy chia nhỏ vấn đề này.
Ta sẽ bắt đầu bằng cách thử hiểu $f$ thay đổi thế nào khi $a$ thay đổi, giả định cần thiết là tất cả $w, x, y$, và $z$ không tồn tại.
Ta sẽ lập luận giống như lần đầu tiên ta làm việc với gradient. Hãy lấy $a$ và cộng một lượng nhỏ $\epsilon$ vào nó.


$$
\begin{aligned}
& f(u(a+\epsilon, b), v(a+\epsilon, b)) \\
\approx & f\left(u(a, b) + \epsilon\frac{\partial u}{\partial a}(a, b), v(a, b) + \epsilon\frac{\partial v}{\partial a}(a, b)\right) \\
\approx & f(u(a, b), v(a, b)) + \epsilon\left[\frac{\partial f}{\partial u}(u(a, b), v(a, b))\frac{\partial u}{\partial a}(a, b) + \frac{\partial f}{\partial v}(u(a, b), v(a, b))\frac{\partial v}{\partial a}(a, b)\right].
\end{aligned}
$$


<!--
The first line follows from the definition of partial derivative, and the second follows from the definition of gradient.
It is notationally burdensome to track exactly where we evaluate every derivative, 
as in the expression $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$, so we often abbreviate this to the much more memorable
-->

Dòng đầu tiên theo sau từ định nghĩa đạo hàm từng phần, và dòng thứ hai theo sau từ định nghĩa gradient.
Thật khó khăn để lần theo các biến khi tính đạo hàm,
như trong biểu thức $\frac{\partial f}{\partial u}(u(a, b), v(a, b))$, cho nên ta thường rút gọn nó để dễ nhớ hơn



$$
\frac{\partial f}{\partial a} = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}.
$$


<!--
It is useful to think about the meaning of the process.
We are trying to understand how a function of the form $f(u(a, b), v(a, b))$ changes its value with a change in $a$.
There are two pathways this can occur: there is the pathway where $a \rightarrow u \rightarrow f$ and where $a \rightarrow v \rightarrow f$.
We can compute both of these contributions via the chain rule: $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ 
and $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$ respectively, and added up.
-->

Sẽ rất hữu ích khi ta suy nghĩ về ý nghĩa của biến đổi này.
Ta đang cố gắng hiểu làm thế nào một hàm có dạng $f(u(a, b), v(a, b))$ thay đổi giá trị của nó khi $a$ thay đổi.
Có hai hướng có thể xảy ra: $a \rightarrow u \rightarrow f$ và $a \rightarrow v \rightarrow f$.
Ta có thể lần lượt tính toán đóng góp của cả hai hướng này thông qua quy tắc dây chuyền: $\frac{\partial w}{\partial u} \cdot \frac{\partial u}{\partial x}$ 
và $\frac{\partial w}{\partial v} \cdot \frac{\partial v}{\partial x}$, rồi cộng gộp lại.


<!--
Imagine we have a different network of functions where the functions on the right depend on those that are connected to on the left as is shown in :numref:`fig_chain-2`.
-->

các hàm được kết nối ở bên trái như trong :numref:`fig_chain-2`. 


<!--
![Another more subtle example of the chain rule.](../img/chain-net2.svg)
-->

![Một ví dụ khác về quy tắc dây chuyền.](../img/chain-net2.svg)
:label:`fig_chain-2`


<!--
To compute something like $\frac{\partial f}{\partial y}$, we need to sum over all (in this case $3$) paths from $y$ to $f$ giving
-->

Để tính toán $\frac{\partial f}{\partial y}$, chúng ta cần tính tổng toàn bộ đường đi từ $y$ đến $f$ (trường hợp này có 3 đường đi): 


$$
\frac{\partial f}{\partial y} = \frac{\partial f}{\partial a} \frac{\partial a}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial u} \frac{\partial u}{\partial y} + \frac{\partial f}{\partial b} \frac{\partial b}{\partial v} \frac{\partial v}{\partial y}.
$$


<!--
Understanding the chain rule in this way will pay great dividends when trying to understand how gradients flow through networks, 
and why various architectural choices like those in LSTMs (:numref:`sec_lstm`) or residual layers (:numref:`sec_resnet`) can help shape the learning process by controlling gradient flow.
-->

Hiểu quy tắc dây chuyền theo cách này giúp chúng ta thấy được dòng chảy của gradient xuyên suốt mạng 
và vì sao một số lựa chọn kiến trúc như trong LSTM (:numref:`sec_lstm`) hoặc các tầng phần dư (:numref:`sec_resnet`) 
có thể định hình quá trình học bằng cách kiểm soát dòng chảy gradient.


<!--
## The Backpropagation Algorithm
-->

## Thuật toán Lan truyền ngược (*Backpropagation*)


<!--
Let us return to the example of :eqref:`eq_multi_func_def` the previous section where
-->

Hãy xem lại ví dụ :eqref:`eq_multi_func_def` ở phần trước: 


$$
\begin{aligned}
f(u, v) & = (u+v)^{2} \\
u(a, b) & = (a+b)^{2}, \qquad v(a, b) = (a-b)^{2}, \\
a(w, x, y, z) & = (w+x+y+z)^{2}, \qquad b(w, x, y, z) = (w+x-y-z)^2.
\end{aligned}
$$


<!--
If we want to compute say $\frac{\partial f}{\partial w}$ we may apply the multi-variate chain rule to see:
-->

Nếu muốn tính $\frac{\partial f}{\partial w}$ chẳng hạn, ta có thể áp dụng quy tắc dây chuyền đa biến để thấy: 


$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial w} + \frac{\partial f}{\partial v}\frac{\partial v}{\partial w}, \\
\frac{\partial u}{\partial w} & = \frac{\partial u}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial u}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial v}{\partial w} & = \frac{\partial v}{\partial a}\frac{\partial a}{\partial w}+\frac{\partial v}{\partial b}\frac{\partial b}{\partial w}.
\end{aligned}
$$


<!--
Let us try using this decomposition to compute $\frac{\partial f}{\partial w}$.
Notice that all we need here are the various single step partials:
-->

Chúng ta hãy thử sử dụng cách phân tách này để tính $\frac{\partial f}{\partial w}$. 
Tất cả những gì chúng ta cần ở đây là các đạo hàm riêng: 

$$
\begin{aligned}
\frac{\partial f}{\partial u} = 2(u+v), & \quad\frac{\partial f}{\partial v} = 2(u+v), \\
\frac{\partial u}{\partial a} = 2(a+b), & \quad\frac{\partial u}{\partial b} = 2(a+b), \\
\frac{\partial v}{\partial a} = 2(a-b), & \quad\frac{\partial v}{\partial b} = -2(a-b), \\
\frac{\partial a}{\partial w} = 2(w+x+y+z), & \quad\frac{\partial b}{\partial w} = 2(w+x-y-z).
\end{aligned}
$$


<!--
If we write this out into code this becomes a fairly manageable expression.
-->

Khi lập trình, các tính toán này trở thành một biểu thức khá dễ quản lý. 


```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'    f at {w}, {x}, {y}, {z} is {f}')

# Compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)

# Compute the final result from inputs to outputs
du_dw, dv_dw = du_da*da_dw + du_db*db_dw, dv_da*da_dw + dv_db*db_dw
df_dw = df_du*du_dw + df_dv*dv_dw
print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
```


<!--
However, note that this still does not make it easy to compute something like $\frac{\partial f}{\partial x}$.
The reason for that is the *way* we chose to apply the chain rule.
If we look at what we did above, we always kept $\partial w$ in the denominator when we could.
In this way, we chose to apply the chain rule seeing how $w$ changed every other variable.
If that is what we wanted, this would be a good idea.
However, think back to our motivation from deep learning: we want to see how every parameter changes the *loss*.
In essence, we want to apply the chain rule keeping $\partial f$ in the numerator whenever we can!
-->

Tuy nhiên, cần lưu ý rằng điều này không làm cho các phép tính chẳng hạn như $\frac{\partial f}{\partial x}$ trở nên đơn giản. 
Lý do nằm ở *cách* chúng ta chọn để áp dụng quy tắc dây chuyền.
Nếu nhìn vào những gì chúng ta đã làm ở trên, chúng ta luôn giữ $\partial w$ ở mẫu khi có thể. 
Với cách này, chúng ta áp dụng quy tắc dây chuyền để xem $w$ thay đổi các biến khác như thế nào. 
Nếu đó là những gì chúng ta muốn thì cách này quả là một ý tưởng hay.
Tuy nhiên, nghĩ lại về mục tiêu của học sâu: chúng ta muốn thấy từng tham số thay đổi giá trị *mất mát* như thế nào. 
Về cốt lõi, chúng ta luôn muốn áp dụng quy tắc dây chuyền và giữ $\partial f$ ở tử số bất cứ khi nào có thể! 


<!--
To be more explicit, note that we can write
-->

Cụ thể hơn, chúng ta có thể viết như sau: 


$$
\begin{aligned}
\frac{\partial f}{\partial w} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial w} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial w}, \\
\frac{\partial f}{\partial a} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial a}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial a}, \\
\frac{\partial f}{\partial b} & = \frac{\partial f}{\partial u}\frac{\partial u}{\partial b}+\frac{\partial f}{\partial v}\frac{\partial v}{\partial b}.
\end{aligned}
$$


<!--
Note that this application of the chain rule has us explicitly compute 
$\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \text{and} \; \frac{\partial f}{\partial w}$. 
Nothing stops us from also including the equations:
-->

Lưu ý rằng cách áp dụng quy tắc dây chuyền này buộc chúng ta phải tính rõ 
$\frac{\partial f}{\partial u}, \frac{\partial f}{\partial v}, \frac{\partial f}{\partial a}, \frac{\partial f}{\partial b}, \; \text{và} \; \frac{\partial f}{\partial w}$. 
Chúng ta cũng có thể thêm vào các phương trình:


$$
\begin{aligned}
\frac{\partial f}{\partial x} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial x} + \frac{\partial f}{\partial b}\frac{\partial b}{\partial x}, \\
\frac{\partial f}{\partial y} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial y}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial y}, \\
\frac{\partial f}{\partial z} & = \frac{\partial f}{\partial a}\frac{\partial a}{\partial z}+\frac{\partial f}{\partial b}\frac{\partial b}{\partial z}.
\end{aligned}
$$


<!--
and then keeping track of how $f$ changes when we change *any* node in the entire network.  Let us implement it.
-->

và tiếp đó theo dõi $f$ biến đổi như thế nào khi chúng ta thay đổi *bất kỳ* nút nào trong toàn bộ mạng. Hãy cùng lập trình nó.


```{.python .input}
#@tab all
# Compute the value of the function from inputs to outputs
w, x, y, z = -1, 0, -2, 1
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2
print(f'f at {w}, {x}, {y}, {z} is {f}')

# Compute the derivative using the decomposition above
# First compute the single step partials
df_du, df_dv = 2*(u + v), 2*(u + v)
du_da, du_db, dv_da, dv_db = 2*(a + b), 2*(a + b), 2*(a - b), -2*(a - b)
da_dw, db_dw = 2*(w + x + y + z), 2*(w + x - y - z)
da_dx, db_dx = 2*(w + x + y + z), 2*(w + x - y - z)
da_dy, db_dy = 2*(w + x + y + z), -2*(w + x - y - z)
da_dz, db_dz = 2*(w + x + y + z), -2*(w + x - y - z)

# Now compute how f changes when we change any value from output to input
df_da, df_db = df_du*du_da + df_dv*dv_da, df_du*du_db + df_dv*dv_db
df_dw, df_dx = df_da*da_dw + df_db*db_dw, df_da*da_dx + df_db*db_dx
df_dy, df_dz = df_da*da_dy + df_db*db_dy, df_da*da_dz + df_db*db_dz

print(f'df/dw at {w}, {x}, {y}, {z} is {df_dw}')
print(f'df/dx at {w}, {x}, {y}, {z} is {df_dx}')
print(f'df/dy at {w}, {x}, {y}, {z} is {df_dy}')
print(f'df/dz at {w}, {x}, {y}, {z} is {df_dz}')
```


<!--
The fact that we compute derivatives from $f$ back towards the inputs rather than from the inputs forward to the outputs 
(as we did in the first code snippet above) is what gives this algorithm its name: *backpropagation*.
Note that there are two steps:
-->

Việc tính đạo hàm từ $f$ trở ngược về đầu vào thay vì từ đầu vào đến đầu ra (như chúng ta đã thực hiện ở đoạn mã đầu tiên ở trên) 
là lý do cho cái tên *lan truyền ngược* (*backpropagation*) của thuật toán. 
Có hai bước: 


<!--
1. Compute the value of the function, and the single step partials from front to back. While not done above, this can be combined into a single *forward pass*.
2. Compute the gradient of $f$ from back to front.  We call this the *backwards pass*.
-->

1. Tính giá trị của hàm và đạo hàm riêng theo từng bước đơn lẻ từ đầu đến cuối. 
Mặc dù không được thực hiện ở trên, hai việc này có thể được kết hợp vào một *lượt truyền xuôi* duy nhất. 
2. Tính toán đạo hàm của $f$ từ cuối về đầu. Chúng ta gọi đó là *lượt truyền ngược*. 


<!--
This is precisely what every deep learning algorithm implements to allow the computation of the gradient of the loss with respect to every weight in the network at one pass.
It is an astonishing fact that we have such a decomposition.
-->

Đây chính xác là những gì mỗi thuật toán học sâu thực thi để tính gradient của giá trị mất mát theo từng trọng số của mạng trong mỗi lượt lan truyền.
Thật thú vị vì chúng ta có một sự phân tách như trên.


<!--
To see how to encapsulated this, let us take a quick look at this example.
-->

Để tóm gọn phần này, hãy xem nhanh ví dụ sau.


```{.python .input}
# Initialize as ndarrays, then attach gradients
w, x, y, z = np.array(-1), np.array(0), np.array(-2), np.array(1)

w.attach_grad()
x.attach_grad()
y.attach_grad()
z.attach_grad()

# Do the computation like usual, tracking gradients
with autograd.record():
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w}, {x}, {y}, {z} is {w.grad}')
print(f'df/dx at {w}, {x}, {y}, {z} is {x.grad}')
print(f'df/dy at {w}, {x}, {y}, {z} is {y.grad}')
print(f'df/dz at {w}, {x}, {y}, {z} is {z.grad}')
```

```{.python .input}
#@tab pytorch
# Initialize as ndarrays, then attach gradients
w = torch.tensor([-1.], requires_grad=True)
x = torch.tensor([0.], requires_grad=True) 
y = torch.tensor([-2.], requires_grad=True)
z = torch.tensor([1.], requires_grad=True)
# Do the computation like usual, tracking gradients
a, b = (w + x + y + z)**2, (w + x - y - z)**2
u, v = (a + b)**2, (a - b)**2
f = (u + v)**2

# Execute backward pass
f.backward()

print(f'df/dw at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {w.grad.data.item()}')
print(f'df/dx at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {x.grad.data.item()}')
print(f'df/dy at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {y.grad.data.item()}')
print(f'df/dz at {w.data.item()}, {x.data.item()}, {y.data.item()}, '
      f'{z.data.item()} is {z.grad.data.item()}')
```

```{.python .input}
#@tab tensorflow
# Initialize as ndarrays, then attach gradients
w = tf.Variable(tf.constant([-1.]))
x = tf.Variable(tf.constant([0.]))
y = tf.Variable(tf.constant([-2.]))
z = tf.Variable(tf.constant([1.]))
# Do the computation like usual, tracking gradients
with tf.GradientTape(persistent=True) as t:
    a, b = (w + x + y + z)**2, (w + x - y - z)**2
    u, v = (a + b)**2, (a - b)**2
    f = (u + v)**2

# Execute backward pass
w_grad = t.gradient(f, w).numpy()
x_grad = t.gradient(f, x).numpy()
y_grad = t.gradient(f, y).numpy()
z_grad = t.gradient(f, z).numpy()

print(f'df/dw at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {w_grad}')
print(f'df/dx at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {x_grad}')
print(f'df/dy at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {y_grad}')
print(f'df/dz at {w.numpy()}, {x.numpy()}, {y.numpy()}, '
      f'{z.numpy()} is {z_grad}')
```


<!--
All of what we did above can be done automatically by calling `f.backwards()`.
-->

Tất cả những gì chúng ta làm ở trên có thể được thực hiện tự động bằng cách gọi hàm `f.backwards()`. 


<!--
## Hessians
-->

## Hessian


<!--
As with single variable calculus, it is useful to consider higher-order derivatives in order to 
get a handle on how we can obtain a better approximation to a function than using the gradient alone.
-->

Như với giải tích đơn biến, việc xem xét đạo hàm bậc cao hơn cũng hữu ích để xấp xỉ tốt hơn một hàm so với việc chỉ sử dụng gradient.


<!--
There is one immediate problem one encounters when working with higher order derivatives of functions of several variables, and that is there are a large number of them.
If we have a function $f(x_1, \ldots, x_n)$ of $n$ variables, then we can take $n^{2}$ many second derivatives, namely for any choice of $i$ and $j$:
-->

Một vấn đề trước mắt khi làm việc với đạo hàm bậc cao hơn của hàm đa biến đó là cần phải tính toán một số lượng lớn đạo hàm.
Nếu chúng ta có một hàm $f(x_1, \ldots, x_n)$ với $n$ biến, chúng ta có thể cần $n^{2}$ đạo hàm bậc 2, chẳng hạn để lựa chọn $i$ và $j$: 


$$
\frac{d^2f}{dx_idx_j} = \frac{d}{dx_i}\left(\frac{d}{dx_j}f\right).
$$


<!--
This is traditionally assembled into a matrix called the *Hessian*:
-->

Biểu thức này được hợp thành một ma trận gọi là *Hessian*: 


$$\mathbf{H}_f = \begin{bmatrix} \frac{d^2f}{dx_1dx_1} & \cdots & \frac{d^2f}{dx_1dx_n} \\ \vdots & \ddots & \vdots \\ \frac{d^2f}{dx_ndx_1} & \cdots & \frac{d^2f}{dx_ndx_n} \\ \end{bmatrix}.$$
:eqlabel:`eq_hess_def`


<!--
Not every entry of this matrix is independent.
Indeed, we can show that as long as both *mixed partials* (partial derivatives with respect to more than one variable) exist 
and are continuous, we can say that for any $i$, and $j$,
-->

Không phải mọi hạng tử của ma trận này đều độc lập.
Thật vậy, chúng ta có thể chứng minh rằng miễn là cả hai *đạo hàm riêng hỗn hợp - mixed partials* 
(đạo hàm riêng theo nhiều hơn một biến số) có tồn tại và liên tục, thì hàm số luôn tồn tại và liên tục với mọi $i$ và $j$,


$$
\frac{d^2f}{dx_idx_j} = \frac{d^2f}{dx_jdx_i}.
$$


<!--
This follows by considering first perturbing a function in the direction of $x_i$,
and then perturbing it in $x_j$ and then comparing the result of that with what happens if we perturb first $x_j$ and then $x_i$, 
with the knowledge that both of these orders lead to the same final change in the output of $f$.
-->

Điều này suy ra được bằng việc xem xét khi ta thay đổi hàm lần lượt theo $x_i$ rồi $x_j$, và ngược lại thay đổi $x_j$ rồi $x_i$, và so sánh hai kết quả này, 
biết rằng cả hai thứ tự này ảnh hưởng đến đầu ra của $f$ như nhau.


<!--
As with single variables, we can use these derivatives to get a far better idea of how the function behaves near a point.
In particular, we can use it to find the best fitting quadratic near a point $\mathbf{x}_0$, as we saw in a single variable.
-->

Như với các hàm đơn biến, chúng ta có thể sử dụng những đạo hàm này để hiểu rõ hơn về hành vi của hàm số lân cận một điểm. 
Cụ thể, chúng ta có thể sử dụng nó để tìm hàm bậc hai phù hợp nhất lân cận $\mathbf{x}_0$ tương tự như trong giải tích đơn biến. 


<!--
Let us see an example. Suppose that $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$.
This is the general form for a quadratic in two variables.
If we look at the value of the function, its gradient, and its Hessian :eqref:`eq_hess_def`, all at the point zero:
-->

Hãy tham khảo một ví dụ. Giả sử rằng $f(x_1, x_2) = a + b_1x_1 + b_2x_2 + c_{11}x_1^{2} + c_{12}x_1x_2 + c_{22}x_2^{2}$. 
Đây là một dạng tổng quát của hàm bậc hai 2 biến.
Nếu chúng ta nhìn vào giá trị của hàm, gradient và Hessian của nó :eqref:`eq_hess_def`, tất cả tại điểm 0: 


$$
\begin{aligned}
f(0,0) & = a, \\
\nabla f (0,0) & = \begin{bmatrix}b_1 \\ b_2\end{bmatrix}, \\
\mathbf{H} f (0,0) & = \begin{bmatrix}2 c_{11} & c_{12} \\ c_{12} & 2c_{22}\end{bmatrix},
\end{aligned}
$$


<!--
we can get our original polynomial back by saying
-->

Chúng ta có thể thu lại được đa thức ban đầu bằng cách đặt:


$$
f(\mathbf{x}) = f(0) + \nabla f (0) \cdot \mathbf{x} + \frac{1}{2}\mathbf{x}^\top \mathbf{H} f (0) \mathbf{x}.
$$


<!--
In general, if we computed this expansion any point $\mathbf{x}_0$, we see that
-->

Nhìn chung, nếu chúng ta tính toán khai triển này tại mọi điểm $\mathbf{x}_0$, ta có:


$$
f(\mathbf{x}) = f(\mathbf{x}_0) + \nabla f (\mathbf{x}_0) \cdot (\mathbf{x}-\mathbf{x}_0) + \frac{1}{2}(\mathbf{x}-\mathbf{x}_0)^\top \mathbf{H} f (\mathbf{x}_0) (\mathbf{x}-\mathbf{x}_0).
$$


<!--
This works for any dimensional input, and provides the best approximating quadratic to any function at a point.
To give an example, let us plot the function 
-->

Cách này hoạt động cho bất cứ đầu vào thứ nguyên nào và cung cấp gần đúng nhất hàm bậc hai cho một hàm bất kỳ tại một điểm.
Lấy biểu đồ của hàm sau làm ví dụ.


$$
f(x, y) = xe^{-x^2-y^2}.
$$


<!--
One can compute that the gradient and Hessian are
-->

Có thể tính toán gradient và Hessian như sau:


$$
\nabla f(x, y) = e^{-x^2-y^2}\begin{pmatrix}1-2x^2 \\ -2xy\end{pmatrix} \; \text{and} \; \mathbf{H}f(x, y) = e^{-x^2-y^2}\begin{pmatrix} 4x^3 - 6x & 4x^2y - 2y \\ 4x^2y-2y &4xy^2-2x\end{pmatrix}.
$$


<!--
And thus, with a little algebra, see that the approximating quadratic at $[-1,0]^\top$ is
-->

Kết hợp một chút đại số, ta thấy rằng hàm bậc hai xấp xỉ tại $[-1,0]^\top$ là:


$$
f(x, y) \approx e^{-1}\left(-1 - (x+1) +(x+1)^2+y^2\right).
$$

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101),
                   np.linspace(-2, 2, 101), indexing='ij')
z = x*np.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = np.exp(-1)*(-1 - (x + 1) + (x + 1)**2 + y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x, y, w, **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101),
                   torch.linspace(-2, 2, 101))

z = x*torch.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = torch.exp(torch.tensor([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101),
                   tf.linspace(-2., 2., 101))

z = x*tf.exp(- x**2 - y**2)

# Compute approximating quadratic with gradient and Hessian at (1, 0)
w = tf.exp(tf.constant([-1.]))*(-1 - (x + 1) + 2 * (x + 1)**2 + 2 * y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.numpy(), y.numpy(), z.numpy(),
                  **{'rstride': 10, 'cstride': 10})
ax.plot_wireframe(x.numpy(), y.numpy(), w.numpy(),
                  **{'rstride': 10, 'cstride': 10}, color='purple')
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-1, 1)
ax.dist = 12
```


<!--
This forms the basis for Newton's Algorithm discussed in :numref:`sec_gd`, 
where we perform numerical optimization iteratively finding the best fitting quadratic, 
and then exactly minimizing that quadratic.
-->

Điều này tạo cơ sở cho Thuật toán Newton được thảo luận ở :numref:`sec_gd`,
trong đó chúng ta lặp đi lặp lại việc tối ưu hoá để tìm ra hàm bậc hai phù hợp nhất và sau đó cực tiểu hoá hàm bậc hai đó.


<!--
## A Little Matrix Calculus
-->

## Giải tích Ma trận


<!--
Derivatives of functions involving matrices turn out to be particularly nice.
This section can become notationally heavy, so may be skipped in a first reading, 
but it is useful to know how derivatives of functions involving common matrix operations are often much cleaner than one might initially anticipate, 
particularly given how central matrix operations are to deep learning applications.
-->

Đạo hàm của các hàm có liên quan đến ma trận hoá ra rất đẹp.
Phần này sẽ nặng về mặt ký hiệu, vì vậy độc giả có thể bỏ qua trong lần đọc đầu tiên.
Tuy nhiên sẽ rất hữu ích khi biết rằng đạo hàm của các hàm liên quan đến các phép toán ma trận thường gọn gàng hơn nhiều so với suy nghĩ ban đầu của chúng ta, 
đặc biệt là bởi sự quan trọng của các phép tính ma trận trong các ứng dụng học sâu.


<!--
Let us begin with an example.  Suppose that we have some fixed column vector $\boldsymbol{\beta}$, 
and we want to take the product function $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$, 
and understand how the dot product changes when we change $\mathbf{x}$.
-->

Hãy xem một ví dụ. Giả sử chúng ta có một vài vector cột cố định $\boldsymbol{\beta}$, 
và chúng ta muốn lấy hàm tích $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$, 
và hiểu cách tích vô hướng thay đổi khi chúng ta thay đổi $\mathbf{x}$.


<!--
A bit of notation that will be useful when working with matrix derivatives in ML is called the *denominator layout matrix derivative* 
where we assemble our partial derivatives into the shape of whatever vector, matrix, or tensor is in the denominator of the differential.
In this case, we will write
-->

Ký hiệu có tên *ma trận đạo hàm sắp xếp theo mẫu số - denominator layout matrix derivative* sẽ hữu ích khi làm việc với ma trận đạo hàm trong học máy,
trong đó chúng ta tập hợp các đạo hàm riêng theo mẫu số của vi phân, biểu diễn thành các dạng vector, ma trận hoặc tensor. 
Trong trường hợp này, chúng ta viết:


$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix},
$$


<!--
where we matched the shape of the column vector $\mathbf{x}$. 
-->

mà ở đây nó khớp với hình dạng của vector cột $\mathbf{x}$. 


<!--
If we write out our function into components this is
-->

Triển khai hàm của chúng ta thành các thành tố 


$$
f(\mathbf{x}) = \sum_{i = 1}^{n} \beta_ix_i = \beta_1x_1 + \cdots + \beta_nx_n.
$$


<!--
If we now take the partial derivative with respect to say $\beta_1$, note that everything is zero but the first term, 
which is just $x_1$ multiplied by $\beta_1$, so the we obtain that
-->

Nếu bây giờ ta tính đạo hàm riêng theo $\beta_1$ chẳng hạn, để ý rằng tất cả các phần tử bằng không ngoại trừ số hạng đầu tiên
là $x_1$ nhân với $\beta_1$. Vì thế, ta có


$$
\frac{df}{dx_1} = \beta_1,
$$


<!--
or more generally that 
-->

hoặc tổng quát hơn đó là 


$$
\frac{df}{dx_i} = \beta_i.
$$


<!--
We can now reassemble this into a matrix to see
-->

Bây giờ ta có thể gộp chúng lại thành một ma trận như sau


$$
\frac{df}{d\mathbf{x}} = \begin{bmatrix}
\frac{df}{dx_1} \\
\vdots \\
\frac{df}{dx_n}
\end{bmatrix} = \begin{bmatrix}
\beta_1 \\
\vdots \\
\beta_n
\end{bmatrix} = \boldsymbol{\beta}.
$$


<!--
This illustrates a few factors about matrix calculus that we will often counter throughout this section:
-->

Biểu thức trên minh họa một vài yếu tố về giải tích ma trận mà ta sẽ gặp trong suốt phần này:


<!--
* First, The computations will get rather involved.
* Second, The final results are much cleaner than the intermediate process, and will always look similar to the single variable case.
In this case, note that $\frac{d}{dx}(bx) = b$ and $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$ are both similar. 
* Third, transposes can often appear seemingly from nowhere.
The core reason for this is the convention that we match the shape of the denominator, thus when we multiply matrices, 
we will need to take transposes to match back to the shape of the original term.
-->

* Đầu tiên, các tính toán sẽ trở nên khá phức tạp.
* Thứ hai, kết quả cuối cùng sẽ gọn gàng hơn quá trình tính toán trung gian, và sẽ luôn có bề ngoài giống với trường hợp đơn biến.
Trong trường hợp này, hãy lưu ý rằng $\frac{d}{dx}(bx) = b$ và $\frac{d}{d\mathbf{x}} (\boldsymbol{\beta}^\top\mathbf{x}) = \boldsymbol{\beta}$ là như nhau.
* Thứ ba, các chuyển vị có thể xuất hiện mà thoạt nhìn không biết chính xác từ đâu ra.
Lý do chủ yếu là do ta quy ước đạo hàm sẽ có cùng kích thước với mẫu số, do đó khi nhân ma trận,
ta cần lấy chuyển vị tương ứng để khớp với kích thước ban đầu.


<!--
To keep building intuition, let us try a computation that is a little harder.
Suppose that we have a column vector $\mathbf{x}$, and a square matrix $A$ and we want to compute
-->

Ta hãy thử một phép tính khó hơn làm ví dụ minh họa trực quan.
Giả sử ta có một vector cột $\mathbf{x}$ và một ma trận vuông $A$, và ta ta muốn tính biểu thức sau:


$$\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}).$$
:eqlabel:`eq_mat_goal_1`


<!--
To drive towards easier to manipulate notation, let us consider this problem using Einstein notation.
In this case we can write the function as
-->

Để thuận tiện cho việc ký hiệu, ta hãy viết lại bài toán bằng ký hiệu Einstein.


$$
\mathbf{x}^\top A \mathbf{x} = x_ia_{ij}x_j.
$$


<!--
To compute our derivative, we need to understand for every $k$, what the value of
-->

Để tính đạo hàm, ta cần tính các giá trị sau với từng giá trị của biến $k$:


$$
\frac{d}{dx_k}(\mathbf{x}^\top A \mathbf{x}) = \frac{d}{dx_k}x_ia_{ij}x_j.
$$


<!--
By the product rule, this is
-->

Theo quy tắc nhân, ta có 


$$
\frac{d}{dx_k}x_ia_{ij}x_j = \frac{dx_i}{dx_k}a_{ij}x_j + x_ia_{ij}\frac{dx_j}{dx_k}.
$$


<!--
For a term like $\frac{dx_i}{dx_k}$, it is not hard to see that this is one when $i=k$ and zero otherwise.
This means that every term where $i$ and $k$ are different vanish from this sum, so the only terms that remain in that first sum are the ones where $i=k$.
The same reasoning holds for the second term where we need $j=k$. This gives
-->

Với số hạng như $\frac{dx_i}{dx_k}$, không khó để thấy rằng đạo hàm trên có giá trị bằng 1 khi $i=k$, ngược lại nó sẽ bằng 0.
Điều này có nghĩa là mọi số hạng với $i$ và $k$ khác nhau sẽ biến mất khỏi tổng trên, vì thế các số hạng duy nhất còn lại trong tổng đầu tiên đó là những số hạng với $i=k$.
Lập luận tương tự cũng áp dụng cho số hạng thứ hai khi ta cần $j=k$. Từ đó, ta có


$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{kj}x_j + x_ia_{ik}.
$$


<!--
Now, the names of the indices in Einstein notation are arbitrary---the fact that $i$ and $j$ are different is immaterial to this computation at this point, 
so we can re-index so that they both use $i$ to see that
-->

Hiện tại, tên của các chỉ số trong ký hiệu Einstein là tùy ý - việc $i$ và $j$ khác nhau không quan trọng cho tính toán tại thời điểm này,
vì thế ta có thể gán lại chỉ số sao cho cả hai đều chứa $i$


$$
\frac{d}{dx_k}x_ia_{ij}x_j = a_{ki}x_i + x_ia_{ik} = (a_{ki} + a_{ik})x_i.
$$


<!--
Now, here is where we start to need some practice to go further.
Let us try and identify this outcome in terms of matrix operations.
$a_{ki} + a_{ik}$ is the $k, i$-th component of $\mathbf{A} + \mathbf{A}^\top$. This gives
-->

Bây giờ, ta cần luyện tập một chút để có thể đi sâu hơn.
Ta hãy thử xác định kết quả trên theo các phép toán ma trận.
$a_{ki} + a_{ik}$ là phần tử thứ $k, i$ của $\mathbf{A} + \mathbf{A}^\top$. Từ đó, ta có


$$
\frac{d}{dx_k}x_ia_{ij}x_j = [\mathbf{A} + \mathbf{A}^\top]_{ki}x_i.
$$


<!--
Similarly, this term is now the product of the matrix $\mathbf{A} + \mathbf{A}^\top$ by the vector $\mathbf{x}$, so we see that
-->

Tương tự, hạng tử này là tích của ma trận $\mathbf{A} + \mathbf{A}^\top$ với vector $\mathbf{x}$, nên ta có 


$$
\left[\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x})\right]_k = \frac{d}{dx_k}x_ia_{ij}x_j = [(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}]_k.
$$


<!--
Thus, we see that the $k$-th entry of the desired derivative from :eqref:`eq_mat_goal_1` is just the $k$-th entry of the vector on the right, 
and thus the two are the same. Thus yields
-->

Ta thấy phần tử thứ $k$ của đạo hàm mong muốn từ :eqref:`eq_mat_goal_1` đơn giản là phần tử thứ $k$ của vector bên vế phải,
và do đó hai phần tử này là như nhau. Điều này dẫn đến 


$$
\frac{d}{d\mathbf{x}}(\mathbf{x}^\top A \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}.
$$


<!--
This required significantly more work than our last one, but the final result is small.
More than that, consider the following computation for traditional single variable derivatives:
-->

Biểu thức trên cần nhiều biến đổi để suy ra được hơn ở phần trước, nhưng kết quả cuối cùng vẫn sẽ gọn gàng.
Hơn thế nữa, hãy xem xét tính toán dưới đây cho đạo hàm đơn biến thông thường:

$$
\frac{d}{dx}(xax) = \frac{dx}{dx}ax + xa\frac{dx}{dx} = (a+a)x.
$$


<!--
Equivalently $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$.
Again, we get a result that looks rather like the single variable result but with a transpose tossed in.
-->

Tương tự, $\frac{d}{dx}(ax^2) = 2ax = (a+a)x$.
Một lần nữa, ta lại thu được kết quả nhìn giống với trường hợp đơn biến nhưng với một phép chuyển vị.

<!--
At this point, the pattern should be looking rather suspicious, so let us try to figure out why.
When we take matrix derivatives like this, let us first assume that the expression we get will be another matrix expression: 
an expression we can write it in terms of products and sums of matrices and their transposes.
If such an expression exists, it will need to be true for all matrices.
In particular, it will need to be true of $1 \times 1$ matrices, in which case the matrix product is just the product of the numbers, 
the matrix sum is just the sum, and the transpose does nothing at all!
In other words, whatever expression we get *must* match the single variable expression.
This means that, with some practice, one can often guess matrix derivatives just by knowing what the associated single variable expression must look like!
-->

Tại thời điểm này, cách tính trên có vẻ khá đáng ngờ, vì vậy ta hãy thử tìm hiểu lý do tại sao.
Khi ta lấy đạo hàm ma trận như trên, đầu tiên ta giả sử biểu thức ta nhận được sẽ là một biểu thức ma trận khác:
một biểu thức mà ta có thể viết nó dưới dạng tích và tổng của các ma trận và chuyển vị của chúng.
Nếu một biểu thức như vậy tồn tại, nó sẽ phải đúng cho tất cả các ma trận.
Do đó, nó sẽ đúng với ma trận $1 \times 1$, trong đó tích ma trận chỉ là tích của các số,
tổng ma trận chỉ là tổng, và phép chuyển vị không có tác dụng gì!
Nói cách khác, bất kỳ biểu thức nào chúng ta nhận được *phải* phù hợp với biểu thức đơn biến.
Điều này có nghĩa là khi ta biết đạo hàm đơn biến tương ứng, với một chút luyện tập ta có thể đoán được các đạo hàm ma trận!


<!--
Let us try this out.
Suppose that $\mathbf{X}$ is a $n \times m$ matrix, 
$\mathbf{U}$ is an $n \times r$ and $\mathbf{V}$ is an $r \times m$.
Let us try to compute
-->

Cùng kiểm nghiệm điều này.
Giả sử $\mathbf{X}$ là ma trận $n \times m$, 
$\mathbf{U}$ là ma trận $n \times r$ và $\mathbf{V}$ là ma trận $r \times m$.
Ta sẽ tính



$$\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2} = \;?$$
:eqlabel:`eq_mat_goal_2`


<!--
This computation is important in an area called matrix factorization.
For us, however, it is just a derivative to compute.
Let us try to imaging what this would be for $1\times1$ matrices.
In that case, we get the expression
-->

Phép tính này khá quan trọng trong phân rã ma trận.
Tuy nhiên, ở đây nó chỉ đơn giản là một đạo hàm mà ta cần tính.
Hãy thử tưởng tượng xem nó sẽ như thế nào đối với ma trận $1\times1$.
Trong trường hợp này, ta có biểu thức sau


$$ 
\frac{d}{dv} (x-uv)^{2}= -2(x-uv)u,
$$


<!--
where, the derivative is rather standard.
If we try to convert this back into a matrix expression we get
-->

Có thể thấy, đây là một đạo hàm khá phổ thông.
Nếu ta thử chuyển đổi nó thành một biểu thức ma trận, ta có 


$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2(\mathbf{X} - \mathbf{U}\mathbf{V})\mathbf{U}.
$$


<!--
However, if we look at this it does not quite work. Recall that $\mathbf{X}$ is $n \times m$, as is $\mathbf{U}\mathbf{V}$, 
so the matrix $2(\mathbf{X} - \mathbf{U}\mathbf{V})$ is $n \times m$.
On the other hand $\mathbf{U}$ is $n \times r$, 
and we cannot multiply a $n \times m$ and a $n \times r$ matrix since the dimensions do not match! 
-->

Tuy nhiên, nếu ta nhìn kỹ, điều này không hoàn toàn đúng.
Hãy nhớ lại $\mathbf{X}$ có kích thước $n \times m$, giống $\mathbf{U}\mathbf{V}$,
nên ma trận $2(\mathbf{X} - \mathbf{U}\mathbf{V})$ có kích thước $n \times m$. 
Mặt khác $\mathbf{U}$ có kích thước $n \times r$, 
và ta không thể nhân một ma trận $n \times m$ với một ma trận $n \times r$ vì số chiều của chúng không khớp nhau!  


<!--
We want to get $\frac{d}{d\mathbf{V}}$, which is the same shape of $\mathbf{V}$, which is $r \times m$.
So somehow we need to take a $n \times m$ matrix and a $n \times r$ matrix, multiply them together (perhaps with some transposes) to get a $r \times m$.
We can do this by multiplying $U^\top$ by $(\mathbf{X} - \mathbf{U}\mathbf{V})$.
Thus, we can guess the solution to :eqref:`eq_mat_goal_2` is
-->

Ta muốn nhận $\frac{d}{d\mathbf{V}}$, cùng kích thước với $\mathbf{V}$ là $r \times m$.
Vì vậy ta bằng cách nào đó cần phải nhân một ma trận $n \times m$ với một ma trận $n \times r$ (có thể phải chuyển vị) để có ma trận $r \times m$. 
Ta có thể làm điều này bằng cách nhân $U^\top$ với $(\mathbf{X} - \mathbf{U}\mathbf{V})$. 
Vì vậy, ta có thể đoán nghiệm cho :eqref:`eq_mat_goal_2` là 


$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$

<!--
To show that this works, we would be remiss to not provide a detailed computation.
If we already believe that this rule-of-thumb works, feel free to skip past this derivation. To compute 
-->

Để chứng minh rằng điều này là đúng, ta cần một tính toán chi tiết.
Nếu bạn tin rằng quy tắc trực quan ở trên là đúng, bạn có thể bỏ qua phần trình bày này. Để tính toán


$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^2,
$$


<!--
we must find for every $a$, and $b$
-->

với mỗi $a$ và $b$, ta phải tính.


$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \frac{d}{dv_{ab}} \sum_{i, j}\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)^2.
$$


<!--
Recalling that all entries of $\mathbf{X}$ and $\mathbf{U}$ are constants as far as $\frac{d}{dv_{ab}}$ is concerned, 
we may push the derivative inside the sum, and apply the chain rule to the square to get
-->

Hãy nhớ lại rằng tất cả các phần tử của $\mathbf{X}$ và $\mathbf{U}$ là hằng số khi tính $\frac{d}{dv_{ab}}$, 
chúng ta có thể đẩy đạo hàm bên trong tổng, và áp dụng quy tắc dây chuyền sau đó bình phương lên để có 


$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= \sum_{i, j}2\left(x_{ij} - \sum_k u_{ik}v_{kj}\right)\left(-\sum_k u_{ik}\frac{dv_{kj}}{dv_{ab}} \right).
$$


<!--
As in the previous derivation, we may note that $\frac{dv_{kj}}{dv_{ab}}$ is only non-zero if the $k=a$ and $j=b$.
If either of those conditions do not hold, the term in the sum is zero, and we may freely discard it. We see that
-->

Tương tự phần diễn giải trước, ta có thể để ý rằng $\frac{dv_{kj}}{dv_{ab}}$ chỉ khác không nếu $k=a$ và $j=b$.
Nếu một trong hai điều kiện đó không thỏa, số hạng trong tổng bằng không, ta có thể tự do loại bỏ nó. Ta thấy rằng


$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}\left(x_{ib} - \sum_k u_{ik}v_{kb}\right)u_{ia}.
$$


<!--
An important subtlety here is that the requirement that $k=a$ does not occur inside the inner sum since 
that $k$ is a dummy variable which we are summing over inside the inner term.
For a notationally cleaner example, consider why
-->

Một điểm tinh tế quan trọng ở đây là yêu cầu về $k=a$ không xảy ra bên trong tổng phía trong bởi
$k$ chỉ là một biến tùy ý để tính tổng các số hạng trong tổng phía trong. 
Một ví dụ dễ hiểu hơn: 


$$
\frac{d}{dx_1} \left(\sum_i x_i \right)^{2}= 2\left(\sum_i x_i \right).
$$


<!--
From this point, we may start identifying components of the sum. First, 
-->

Từ đây, ta có thể bắt đầu xác định các thành phần của tổng. Đầu tiên, 


$$
\sum_k u_{ik}v_{kb} = [\mathbf{U}\mathbf{V}]_{ib}.
$$


<!--
So the entire expression in the inside of the sum is
-->

Cho nên toàn bộ biểu thức bên trong tổng là 


$$
x_{ib} - \sum_k u_{ik}v_{kb} = [\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$


<!--
This means we may now write our derivative as
-->

Điều này nghĩa là giờ đây đạo hàm của ta có thể viết dưới dạng


$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}u_{ia}.
$$


<!--
We want this to look like the $a, b$ element of a matrix so we can use the technique as in the previous example to arrive at a matrix expression, 
which means that we need to exchange the order of the indices on $u_{ia}$.
If we notice that $u_{ia} = [\mathbf{U}^\top]_{ai}$, we can then write
-->

Chúng ta có thể muốn nó trông giống như phần tử $a, b$ của một ma trận để có thể sử dụng các kỹ thuật trong các ví dụ trước đó nhằm đạt được một biểu thức ma trận,
nghĩa là ta cần phải hoán đổi thứ tự của các chỉ số trên $u_{ia}$.
Nếu để ý $u_{ia} = [\mathbf{U}^\top]_{ai}$, ta có thể viết


$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\sum_{i} [\mathbf{U}^\top]_{ai}[\mathbf{X}-\mathbf{U}\mathbf{V}]_{ib}.
$$


<!--
This is a matrix product, and thus we can conclude that
-->

Đây là tích một ma trận, vì thế ta có thể kết luận  


$$
\frac{d}{dv_{ab}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2[\mathbf{U}^\top(\mathbf{X}-\mathbf{U}\mathbf{V})]_{ab}.
$$


<!--
and thus we may write the solution to :eqref:`eq_mat_goal_2`
-->

và vì vậy ta có lời giải cho :eqref:`eq_mat_goal_2` 


$$
\frac{d}{d\mathbf{V}} \|\mathbf{X} - \mathbf{U}\mathbf{V}\|_2^{2}= -2\mathbf{U}^\top(\mathbf{X} - \mathbf{U}\mathbf{V}).
$$


<!--
This matches the solution we guessed above!
-->

Lời giải này trùng với biểu thức mà ta đoán ở phía trên! 


<!--
It is reasonable to ask at this point, "Why can I not just write down matrix versions of all the calculus rules I have learned?
It is clear this is still mechanical. Why do we not just get it over with!"
And indeed there are such rules and :cite:`Petersen.Pedersen.ea.2008` provides an excellent summary.
However, due to the plethora of ways matrix operations can be combined compared to single values, there are many more matrix derivative rules than single variable ones.
It is often the case that it is best to work with the indices, or leave it up to automatic differentiation when appropriate.
-->

Lúc này cũng dễ hiểu nếu ta tự hỏi "Tại sao không viết tất cả các quy tắc giải tích đã từng học thành dạng ma trận?
Điều này rõ ràng là công việc máy móc. Tại sao ta không đơn giản là làm hết một lần cho xong?" 
Và thực sự có những quy tắc như thế, :cite:`Petersen.Pedersen.ea.2008` cho ta một bản tóm tắt tuyệt vời.
Tuy nhiên, vì số cách kết hợp các phép toán ma trận nhiều hơn hẳn so với các giá trị một biến, nên có nhiều quy tắc đạo hàm ma trận hơn các quy tắc dành cho hàm cho một biến.
Thông thường, tốt nhất là làm việc với các chỉ số, hoặc dùng vi phân tự động khi thích hợp.


## Tóm tắt

<!--
* In higher dimensions, we can define gradients which serve the same purpose as derivatives in one dimension.
These allow us to see how a multi-variable function changes when we make an arbitrary small change to the inputs.
* The backpropagation algorithm can be seen to be a method of organizing the multi-variable chain rule to allow for the efficient computation of many partial derivatives.
* Matrix calculus allows us to write the derivatives of matrix expressions in concise ways.
-->

* Với không gian nhiều chiều, chúng ta có thể định nghĩa gradient cùng mục đích như các đạo hàm một chiều.
Điều này cho phép ta thấy cách một hàm đa biến thay đổi như thế nào khi có bất kỳ thay đổi nhỏ xảy ra ở đầu vào.
* Thuật toán lan truyền ngược có thể được xem như một phương pháp trong việc tổ chức quy tắc dây chuyền đa biến cho phép tính toán hiệu quả các đạo hàm riêng.
* Giải tích ma trận cho phép chúng ta viết các đạo hàm của biểu thức ma trận một cách gọn gàng hơn.


## Bài tập

<!--
1. Given a column vector $\boldsymbol{\beta}$, compute the derivatives of both $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ 
and $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$. Why do you get the same answer?
2. Let $\mathbf{v}$ be an $n$ dimension vector. What is $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$?
3. Let $L(x, y) = \log(e^x + e^y)$.  Compute the gradient.  What is the sum of the components of the gradient?
4. Let $f(x, y) = x^2y + xy^2$. Show that the only critical point is $(0,0)$. By considering $f(x, x)$, determine if $(0,0)$ is a maximum, minimum, or neither.
5. Suppose that we are minimizing a function $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$.
How can we geometrically interpret the condition of $\nabla f = 0$ in terms of $g$ and $h$?
-->

1. Cho một vector cột $\boldsymbol{\beta}$, tính các đạo hàm của cả hai ma trận $f(\mathbf{x}) = \boldsymbol{\beta}^\top\mathbf{x}$ 
và ma trận $g(\mathbf{x}) = \mathbf{x}^\top\boldsymbol{\beta}$. Hãy cho biết tại sao bạn lại ra cùng đáp án? 
2. Cho $\mathbf{v}$ là một vector $n$ chiều. Vậy $\frac{\partial}{\partial\mathbf{v}}\|\mathbf{v}\|_2$? là gì? 
3. Cho $L(x, y) = \log(e^x + e^y)$. Tính toán gradient. Tổng của các thành phần của gradient là gì? 
4. Cho $f(x, y) = x^2y + xy^2$. Chứng minh rằng điểm tới hạn duy nhất là $(0,0)$. Bằng việc xem xét $f(x, x)$, hãy xác định xem $(0,0)$ là cực đại, cực tiểu, hay không phải cả hai. 
5. Giả sử ta đang tối thiểu hàm $f(\mathbf{x}) = g(\mathbf{x}) + h(\mathbf{x})$.
Làm cách nào ta có thể diễn giải bằng hình học điều kiện $\nabla f = 0$ thông qua $g$ và $h$? 


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/413), [Pytorch](https://discuss.d2l.ai/t/1090), [Tensorflow](https://discuss.d2l.ai/t/1091)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Nguyễn Lê Quang Nhật
* Nguyễn Văn Quang
* Nguyễn Thanh Hòa
* Nguyễn Văn Cường
* Trần Yến Thy
* Nguyễn Mai Hoàng Long

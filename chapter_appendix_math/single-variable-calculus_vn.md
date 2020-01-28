<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Single Variable Calculus
-->

# *dịch tiêu đề phía trên*
:label:`sec_single_variable_calculus`

<!--
In :numref:`sec_calculus`, we saw the basic elements of differential calculus. 
This section takes a deeper dive into the fundamentals of calculus and how we can understand and apply it in the context of machine learning.
-->

*dịch đoạn phía trên*

<!--
## Differential Calculus
-->

## *dịch tiêu đề phía trên*

<!--
Differential calculus is fundamentally the study of how functions behave under small changes.  To see why this is so core to deep learning, let's consider an example.
-->

*dịch đoạn phía trên*

<!--
Suppose that we have a deep neural network where the weights are, for convenience, concatenated into a single vector $\mathbf{w} = (w_1, \ldots, w_n)$.  
Given a training dataset, we consider the loss of our neural network on this dataset, which we will write as $\mathcal{L}(\mathbf{w})$.
-->

*dịch đoạn phía trên*

<!--
This function is extraordinarily complex, encoding the performance of all possible models of the given architecture on this dataset, so it is nearly impossible to tell what set of weights $\mathbf{w}$ will minimize the loss. 
Thus, in practice, we often start by initializing our weights *randomly*, and then iteratively take small steps in the direction which makes the loss decrease as rapidly as possible.
-->

*dịch đoạn phía trên*

<!--
The question then becomes something that on the surface is no easier: how do we find the direction which makes the weights decrease as quickly as possible?  To dig into this, let's first examine the case with only a single weight: $L(\mathbf{w}) = L(x)$ for a single real value $x$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần  1 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 2 ==================== -->

<!--
Let's take $x$ and try to understand what happens when we change it by a small amount to $x + \epsilon$. 
If you wish to be concrete, think a number like $\epsilon = 0.0000001$.  
To help us visualize what happens, let's graph an example function, $f(x) = \sin(x^x)$, over the $[0, 3]$.
-->

*dịch đoạn phía trên*

```{.python .input}
%matplotlib inline
import d2l
from IPython import display
from mxnet import np, npx
npx.set_np()

# Plot a function in a normal range
x_big = np.arange(0.01, 3.01, 0.01)
ys = np.sin(x_big**x_big)
d2l.plot(x_big, ys, 'x', 'f(x)')
```

<!--
At this large scale, the function's behavior is not simple. 
However, if we reduce our range to something smaller like $[1.75,2.25]$, we see that the graph becomes much simpler.
-->

*dịch đoạn phía trên*

```{.python .input}
# Plot a the same function in a tiny range
x_med = np.arange(1.75, 2.25, 0.001)
ys = np.sin(x_med**x_med)
d2l.plot(x_med, ys, 'x', 'f(x)')
```

<!--
Taking this to an extreme, if we zoom into a tiny segment, the behavior becomes far simpler: it is just a straight line.
-->

*dịch đoạn phía trên*

```{.python .input}
# Plot a the same function in a tiny range
x_small = np.arange(2.0, 2.01, 0.0001)
ys = np.sin(x_small**x_small)
d2l.plot(x_small, ys, 'x', 'f(x)')
```

<!--
This is the key observation of single variable calculus: the behavior of familiar functions can be modeled by a line in a small enough range.  
This means that for most functions, it is reasonable to expect that as we shift the $x$ value of the function by a little bit, the output $f(x)$ will also be shifted by a little bit.  
The only question we need to answer is, "How large is the change in the output compared to the change in the input?  
Is it half as large?  Twice as large?"
-->

*dịch đoạn phía trên*

<!--
Thus, we can consider the ratio of the change in the output of a function for a small change in the input of the function.  We can write this formally as
-->

*dịch đoạn phía trên*

$$
\frac{L(x+\epsilon) - L(x)}{(x+\epsilon) - x} = \frac{L(x+\epsilon) - L(x)}{\epsilon}.
$$

<!-- ===================== Kết thúc dịch Phần 2 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 3 ==================== -->

<!--
This is already enough to start to play around with in code.  
For instance, suppose that we know that $L(x) = x^{2} + 1701(x-4)^3$, then we can see how large this value is at the point $x = 4$ as follows.
-->

*dịch đoạn phía trên*

```{.python .input}
# Define our function
def L(x):
    return x**2 + 1701*(x-4)**3

# Print the difference divided by epsilon for several epsilon
for epsilon in [0.1, 0.001, 0.0001, 0.00001]:
    print("epsilon = {:.5f} -> {:.5f}".format(
        epsilon, (L(4+epsilon) - L(4)) / epsilon))
```

<!--
Now, if we are observant, we will notice that the output of this number is suspiciously close to $8$.  
Indeed, if we decrease $\epsilon$, we will see value becomes progressively closer to $8$.  
Thus we may conclude, correctly, that the value we seek (the degree a change in the input changes the output) should be $8$ at the point $x=4$.  
The way that a mathematician encodes this fact is
-->

*dịch đoạn phía trên*

$$
\lim_{\epsilon \rightarrow 0}\frac{L(4+\epsilon) - L(4)}{\epsilon} = 8.
$$

<!--
As a bit of a historical digression: in the first few decades of neural network research, scientists used this algorithm (the *method of finite differences*) to evaluate how a loss function changed under small perturbation: just change the weights and see how the loss changed.  
This is computationally inefficient, requiring two evaluations of the loss function to see how a single change of one variable influenced the loss.  
If we tried to do this with even a paltry few thousand parameters, it would require several thousand evaluations of the network over the entire dataset!  
It was not solved until 1986 that the *backpropagation algorithm* introduced in :cite:`Rumelhart.Hinton.Williams.ea.1988` provided a way to calculate how *any* change of the weights together would change the loss in the same computation time as a single prediction of the network over the dataset.
-->

*dịch đoạn phía trên*

<!--
Back in our example, this value $8$ is different for different values of $x$, so it makes sense to define it as a function of $x$.  
More formally, this value dependent rate of change is referred to as the *derivative* which is written as
-->

*dịch đoạn phía trên*

$$\frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon}.$$
:eqlabel:`eq_der_def`

<!--
Different texts will use different notations for the derivative. 
For instance, all of the below notations indicate the same thing:
-->

*dịch đoạn phía trên*

$$
\frac{df}{dx} = \frac{d}{dx}f = f' = \nabla_xf = D_xf = f_x.
$$

<!--
Most authors will pick a single notation and stick with it, however even that is not guaranteed.  
It is best to be familiar with all of these.  
We will use the notation $\frac{df}{dx}$ throughout this text, unless we want to take the derivative of a complex expression, in which case we will use $\frac{d}{dx}f$ to write expressions like

$$
\frac{d}{dx}\left[x^4+\cos\left(\frac{x^2+1}{2x-1}\right)\right].
$$

Often times, it is intuitively useful to unravel the definition of derivative :eqref:`eq_der_def` again to see how a function changes when we make a small change of $x$:
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 4 ==================== -->

$$\begin{aligned} \frac{df}{dx}(x) = \lim_{\epsilon \rightarrow 0}\frac{f(x+\epsilon) - f(x)}{\epsilon} & \implies \frac{df}{dx}(x) \approx \frac{f(x+\epsilon) - f(x)}{\epsilon} \\ & \implies \epsilon \frac{df}{dx}(x) \approx f(x+\epsilon) - f(x) \\ & \implies f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x). \end{aligned}$$
:eqlabel:`eq_small_change`

<!--
The last equation is worth explicitly calling out.  
It tells us that if you take any function and change the input by a small amount, the output would change by that small amount scaled by the derivative.
-->

*dịch đoạn phía trên*

<!--
In this way, we can understand the derivative as the scaling factor that tells us how large of change we get in the output from a change in the input.
-->

*dịch đoạn phía trên*

<!--
## Rules of Calculus
-->

## *dịch tiêu đề phía trên*
:label:`sec_derivative_table`

<!--
We now turn to the task of understanding how to compute the derivative of an explicit function.  
A full formal treatment of calculus would derive everything from first principles.  
We will not indulge in this temptation here, but rather provide an understanding of the common rules encountered.
-->

*dịch đoạn phía trên*

<!--
### Common Derivatives
-->

### *dịch tiêu đề phía trên*

<!--
As was seen in :numref:`sec_calculus`, when computing derivatives one can often times use a series of rules to reduce the computation to a few core functions.  
We repeat them here for ease of reference.
-->

*dịch đoạn phía trên*

<!--
* **Derivative of constants.** $\frac{d}{dx}c = 0$.
* **Derivative of linear functions.** $\frac{d}{dx}(ax) = a$.
* **Power rule.** $\frac{d}{dx}x^n = nx^{n-1}$.
* **Derivative of exponentials.** $\frac{d}{dx}e^x = e^x$.
* **Derivative of the logarithm.** $\frac{d}{dx}\log(x) = \frac{1}{x}$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 5 ==================== -->

<!--
### Derivative Rules
-->

### *dịch tiêu đề phía trên*

<!--
If every derivative needed to be separately computed and stored in a table, differential calculus would be near impossible.  
It is a gift of mathematics that we can generalize the above derivatives and compute more complex derivatives like finding the derivative of $f(x) = \log\left(1+(x-1)^{10}\right)$.  As was mentioned in :numref:`sec_calculus`, the key to doing so is to codify what happens when we take functions and combine them in various ways, most importantly: sums, products, and compositions.
-->

*dịch đoạn phía trên*

<!--
* **Sum rule.** $\frac{d}{dx}\left(g(x) + h(x)\right) = \frac{dg}{dx}(x) + \frac{dh}{dx}(x)$.
* **Product rule.** $\frac{d}{dx}\left(g(x)\cdot h(x)\right) = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x)$.
* **Chain rule.** $\frac{d}{dx}g(h(x)) = \frac{dg}{dh}(h(x))\cdot \frac{dh}{dx}(x)$.
-->

*dịch đoạn phía trên*

<!--
Let's see how we may use :eqref:`eq_small_change` to understand these rules.  For the sum rule, consider following chain of reasoning:
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*


<!--
The product is more subtle, and will require a new observation about how to work with these expressions.  We will begin as before using :eqref:`eq_small_change`:
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

$$
\frac{f(x+\epsilon) - f(x)}{\epsilon} = g(x)\frac{dh}{dx}(x) + \frac{dg}{dx}(x)h(x) + \epsilon \frac{dg}{dx}(x)\frac{dh}{dx}(x),
$$

<!--
and see that as we send $\epsilon \rightarrow 0$, the right hand term goes to zero as well.
-->

*dịch đoạn phía trên*

<!--
Finally, with the chain rule, we can again progress as before using :eqref:`eq_small_change` and see that
-->

Cuối cùng, theo quy tắc dây chuyền, chúng ta có thể tiếp tục khai triển như lúc trước, sử dụng :eqref:`eq_small_change` và thấy rằng:

$$
\begin{aligned}
f(x+\epsilon) & = g(h(x+\epsilon)) \\
& \approx g\left(h(x) + \epsilon \frac{dh}{dx}(x)\right) \\
& \approx g(h(x)) + \epsilon \frac{dh}{dx}(x) \frac{dg}{dh}(h(x))\\
& = f(x) + \epsilon \frac{dg}{dh}(h(x))\frac{dh}{dx}(x).
\end{aligned}
$$

<!-- ===================== Kết thúc dịch Phần 5 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 6 ==================== -->

<!--
where in the second line we view the function $g$ as having its input ($h(x)$) shifted by the tiny quantity $\epsilon \frac{dh}{dx}(x)$.
-->

Chú ý là ở dòng thứ hai trong chuỗi khai triển trên, chúng ta đã xem đối số $h(x)$ của hàm $g$ như là bị dịch đi bởi một lượng rất nhỏ $\epsilon \frac{dh}{dx}(x)$.

<!--
These rule provide us with a flexible set of tools to compute essentially any expression desired.  For instance,
-->

Các luật này cung cấp cho chúng ta một tập hợp các công cụ linh hoạt để tính toán đạo hàm của hầu như là bất kỳ diễn tả toán học nào mà bạn muốn.
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

1. Luật đạo hàm của hàm hợp và công thức đạo hàm của hàm logarithm.
2. Quy tắc đạo hàm của tổng.
3. Đạo hàm của hằng số, quy tắc dây chuyền, và quy tắc đạo hàm của lũy thừa.
4. Luật đạo hàm của tổng, đạo hàm của hàm tuyến tính, đạo hàm của hằng số.

<!--
Two things should be clear after doing this example:
-->

Từ ví dụ trên, chúng ta có thể dễ dàng rút ra được hai điều:

<!--
1. Any function we can write down using sums, products, constants, powers, exponentials, and logarithms can have its derivate computed mechanically by following these rules.
2. Having a human follow these rules can be tedious and error prone!
-->

1. Chúng ta có thể lấy đạo hàm của bất kỳ hàm số nào mà có thể diễn tả được bằng tổng, tích, hằng số, lũy thừa, exponentials, và logarithms, bằng cách sử dụng những quy luật trên một cách máy móc.
2. Quá trình dùng những quy luật này để tính đạo hàm bằng tay có thể sẽ rất tẻ nhạt và dễ bị mắc lỗi.

<!--
Thankfully, these two facts together hint towards a way forward: this is a perfect candidate for mechanization!  Indeed backpropagation, which we will revisit later in this section, is exactly that.
-->

Rất may là hai điều này gộp chung lại chỉ dấu cho chúng ta một cách để tiếp tục: đây chính là cơ hội lý tưởng để máy tính có thể tự động hóa! Thật vậy, kỹ thuật lan truyền ngược mà chúng ta sẽ gặp lại sau đây chính xác là hiện thực hóa ý tưởng này.

<!-- ===================== Kết thúc dịch Phần 6 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 7 ==================== -->

<!--
### Linear Approximation
-->

### *dịch tiêu đề phía trên*

<!--
When working with derivatives, it is often useful to geometrically interpret the approximation used above.  In particular, note that the equation
-->

*dịch đoạn phía trên*

$$
f(x+\epsilon) \approx f(x) + \epsilon \frac{df}{dx}(x),
$$

<!--
approximates the value of $f$ by a line which passes through the point $(x, f(x))$ and has slope $\frac{df}{dx}(x)$.  
In this way we say that the derivative gives a linear approximation to the function $f$, as illustrated below:
-->

*dịch đoạn phía trên*

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some linear approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0))

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

<!--
### Higher Order Derivatives
-->

### *dịch tiêu đề phía trên*

<!--
Let's now do something that may on the surface seem strange.  
Take a function $f$ and compute the derivative $\frac{df}{dx}$.  
This gives us the rate of change of $f$ at any point.
-->

*dịch đoạn phía trên*

<!--
However, the derivative, $\frac{df}{dx}$, can be viewed as a function itself, so nothing stops us from computing the derivative of $\frac{df}{dx}$ to get $\frac{d^2f}{dx^2} = \frac{df}{dx}\left(\frac{df}{dx}\right)$.  
We will call this the second derivative of $f$.  
This function is the rate of change of the rate of change of $f$, or in other words, how the rate of change is changing. 
We may apply the derivative any number of times to obtain what is called the $n$-th derivative. 
To keep the notation clean, we will denote the $n$-th derivative as
-->

*dịch đoạn phía trên*

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

<!--
Let's try to understand *why* this is a useful notion.  
Below, we visualize $f^{(2)}(x)$, $f^{(1)}(x)$, and $f(x)$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 8 ==================== -->

<!--
First, consider the case that the second derivative $f^{(2)}(x)$ is a positive constant.  
This means that the slope of the first derivative is positive.  
As a result, the first derivative $f^{(1)}(x)$ may start out negative, becomes zero at a point, and then becomes positive in the end. 
This tells us the slope of our original function $f$ and therefore, the function $f$ itself decreases, flattens out, then increases.  
In other words, the function $f$ curves up, and has a single minimum as is shown in :numref:`fig_positive-second`.
-->

*dịch đoạn phía trên*

<!--
![If we assume the second derivative is a positive constant, then the fist derivative in increasing, which implies the function itself has a minimum.](../img/posSecDer.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/posSecDer.svg)
:label:`fig_positive-second`


<!--
Second, if the second derivative is a negative constant, that means that the first derivative is decreasing.  
This implies the first derivative may start out positive, becomes zero at a point, and then becomes negative. 
Hence, the function $f$ itself increases, flattens out, then decreases.  
In other words, the function $f$ curves down, and has a single maximum as is shown in :numref:`fig_negative-second`.
-->

*dịch đoạn phía trên*

<!--
![If we assume the second derivative is a negative constant, then the fist derivative in decreasing, which implies the function itself has a maximum.](../img/negSecDer.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/negSecDer.svg)
:label:`fig_negative-second`


<!--
Third, if the second derivative is a always zero, then the first derivative will never change---it is constant!  
This means that $f$ increases (or decreases) at a fixed rate, and $f$ is itself a straight line  as is shown in :numref:`fig_zero-second`.
-->

*dịch đoạn phía trên*

<!--
![If we assume the second derivative is zero, then the fist derivative is constant, which implies the function itself is a straight line.](../img/zeroSecDer.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/zeroSecDer.svg)
:label:`fig_zero-second`

<!--
To summarize, the second derivative can be interpreted as describing the way that the function $f$ curves.  
A positive second derivative leads to a upwards curve, while a negative second derivative means that $f$ curves downwards, and a zero second derivative means that $f$ does not curve at all.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 8 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 9 ==================== -->

<!--
Let's take this one step further. Consider the function $g(x) = ax^{2}+ bx + c$.  We can then compute that
-->

*dịch đoạn phía trên*

$$
\begin{aligned}
\frac{dg}{dx}(x) & = 2ax + b \\
\frac{d^2g}{dx^2}(x) & = 2a.
\end{aligned}
$$

<!--
If we have some original function $f(x)$ in mind, we may compute the first two derivatives and find the values for $a, b$, and $c$ that make them match this computation.  
Similarly to the previous section where we saw that the first derivative gave the best approximation with a straight line, this construction provides the best approximation by a quadratic.  Let's visualize this for $f(x) = \sin(x)$.
-->

*dịch đoạn phía trên*

```{.python .input}
# Compute sin
xs = np.arange(-np.pi, np.pi, 0.01)
plots = [np.sin(xs)]

# Compute some quadratic approximations. Use d(sin(x))/dx = cos(x)
for x0 in [-1.5, 0, 2]:
    plots.append(np.sin(x0) + (xs - x0) * np.cos(x0) -
                              (xs - x0)**2 * np.sin(x0) / 2)

d2l.plot(xs, plots, 'x', 'f(x)', ylim=[-1.5, 1.5])
```

<!--
We will extend this idea to the idea of a *Taylor series* in the next section.
-->

*dịch đoạn phía trên*

<!--
### Taylor Series
-->

### *dịch tiêu đề phía trên*


<!--
The *Taylor series* provides a method to approximate the function $f(x)$ if we are given values for the first $n$ derivatives at a point $x_0$, i.e., $\left\{ f(x_0), f^{(1)}(x_0), f^{(2)}(x_0), \ldots, f^{(n)}(x_0) \right\}$. The idea will be to find a degree $n$ polynomial that matches all the given derivatives at $x_0$.
-->

*dịch đoạn phía trên*

<!--
We saw the case of $n=2$ in the previous section and a little algebra shows this is
-->

*dịch đoạn phía trên*

$$
f(x) \approx \frac{1}{2}\frac{d^2f}{dx^2}(x_0)(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

<!-- ===================== Kết thúc dịch Phần 9 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 10 ==================== -->

<!--
As we can see above, the denominator of $2$ is there to cancel out the $2$ we get when we take two derivatives of $x^2$, while the other terms are all zero.  
Same logic applies for the first derivative and the value itself.
-->

*dịch đoạn phía trên*

<!--
If we push the logic further to $n=3$, we will conclude that
-->

*dịch đoạn phía trên*

$$
f(x) \approx \frac{\frac{d^3f}{dx^3}(x_0)}{6}(x-x_0)^3 + \frac{\frac{d^2f}{dx^2}(x_0)}{2}(x-x_0)^{2}+ \frac{df}{dx}(x_0)(x-x_0) + f(x_0).
$$

<!--
where the $6 = 3 \times 2 = 3!$ comes from the constant we get in front if we take three derivatives of $x^3$.
-->

*dịch đoạn phía trên*


<!--
Furthermore, we can get a degree $n$ polynomial by
-->

*dịch đoạn phía trên*

$$
P_n(x) = \sum_{i = 0}^{n} \frac{f^{(i)}(x_0)}{i!}(x-x_0)^{i}.
$$

<!--
where the notation
-->

*dịch đoạn phía trên*

$$
f^{(n)}(x) = \frac{d^{n}f}{dx^{n}} = \left(\frac{d}{dx}\right)^{n} f.
$$

<!-- ===================== Kết thúc dịch Phần 10 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 11 ==================== -->

<!--
Indeed, $P_n(x)$ can be viewed as the best $n$-th degree polynomial approximation to our function $f(x)$.
-->

*dịch đoạn phía trên*

<!--
While we are not going to dive all the way into the error of the above approximations, it is worth mentioning the the infinite limit. 
In this case, for well behaved functions (known as real analytic functions) like $\cos(x)$ or $e^{x}$, we can write out the infinite number of terms and approximate the exactly same function
-->

*dịch đoạn phía trên*

$$
f(x) = \sum_{n = 0}^\infty \frac{f^{(n)}(x_0)}{n!}(x-x_0)^{n}.
$$

<!--
Take $f(x) = e^{x}$ as am example. Since $e^{x}$ is its own derivative, we know that $f^{(n)}(x) = e^{x}$. 
Therefore, $e^{x}$ can be reconstructed by taking the Taylor series at $x_0 = 0$, i.e.,
-->

*dịch đoạn phía trên*

$$
e^{x} = \sum_{n = 0}^\infty \frac{x^{n}}{n!} = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots.
$$

<!--
Let's see how this works in code and observe how increasing the degree of the Taylor approximation brings us closer to the desired function $e^x$.
-->

*dịch đoạn phía trên*

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

<!--
Taylor series have two primary applications:
-->

*dịch đoạn phía trên*

<!--
1. *Theoretical applications*: Often when we try to understand a too complex function, using Taylor series enables we turn it into a polynomial that we can work with directly.
-->

*dịch đoạn phía trên*

<!--
2. *Numerical applications*: Some functions like $e^{x}$ or $\cos(x)$ are  difficult for machines to compute.  
They can store tables of values at a fixed precision (and this is often done), but it still leaves open questions like "What is the 1000-th digit of $\cos(1)$?"  
Taylor series are often helpful to answer such questions.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 11 ==================== -->
<!-- ===================== Bắt đầu dịch Phần 12 ==================== -->

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* Derivatives can be used to express how functions change when we change the input by a small amount.
* Elementary derivatives can be combined using derivative rules to create arbitrarily complex derivatives.
* Derivatives can be iterated to get second or higher order derivatives.  Each increase in order provides more fine grained information on the behavior of the function.
* Using information in the derivatives of a single data point, we can approximate well behaved functions by polynomials obtained from the Taylor series.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. What is the derivative of $x^3-4x+1$?
2. What is the derivative of $\log(\frac{1}{x})$?
3. True or False: If $f'(x) = 0$ then $f$ has a maximum or minimum at $x$?
4. Where is the minimum of $f(x) = x\log(x)$ for $x\ge0$ (where we assume that $f$ takes the limiting value of $0$ at $f(0)$)?
-->

*dịch đoạn phía trên*


<!--
## [Discussions](https://discuss.mxnet.io/t/5149)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/5149)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

<!--
![](../img/qr_single-variable-calculus.svg)
-->



<!-- ===================== Kết thúc dịch Phần 12 ==================== -->

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

<!-- Phần 1 -->
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

<!-- Phần 6 -->
* Duy-Nguyen Ta

<!-- Phần 7 -->
*

<!-- Phần 8 -->
*

<!-- Phần 9 -->
*

<!-- Phần 10 -->
*

<!-- Phần 11 -->
*

<!-- Phần 12 -->
*

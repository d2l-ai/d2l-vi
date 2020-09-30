<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Maximum Likelihood
-->

# Hợp lý cực đại
:label:`sec_maximum_likelihood`


<!--
One of the most commonly encountered way of thinking in machine learning is the maximum likelihood point of view.
This is the concept that when working with a probabilistic model with unknown parameters, 
the parameters which make the data have the highest probability are the most likely ones.
-->

Một trong những cách tư duy thường gặp nhất trong học máy là quan điểm về hợp lý cực đại.
Đây là khái niệm mà khi làm việc với một mô hình xác suất với các tham số chưa biết,
các tham số làm cho dữ liệu có xác suất cao nhất là những tham số có nhiều khả năng nhất.


<!--
## The Maximum Likelihood Principle
-->

## Định đề hợp lý cực đại


<!--
This has a Bayesian interpretation which can be helpful to think about.
Suppose that we have a model with parameters $\boldsymbol{\theta}$ and a collection of data examples $X$.
For concreteness, we can imagine that $\boldsymbol{\theta}$ is a single value representing the probability that a coin comes up heads when flipped, 
and $X$ is a sequence of independent coin flips.
We will look at this example in depth later.
-->

Điều này có cách giải thích theo kiểu Bayes có thể hữu ích để suy nghĩ.
Giả sử rằng ta có một mô hình với các tham số $\boldsymbol{\theta}$ và một tập hợp các mẫu dữ liệu $X$.
Để rõ ràng, ta có thể tưởng tượng rằng $\boldsymbol{\theta}$ là một giá trị duy nhất đại diện cho xác suất một đồng xu xuất hiện khi lật,
và $X$ là một chuỗi các lần tung đồng xu độc lập.
Chúng ta sẽ xem xét ví dụ này sâu hơn ở phần sau.


<!--
If we want to find the most likely value for the parameters of our model, that means we want to find
-->

Nếu ta muốn tìm giá trị có khả năng nhất cho các tham số của mô hình của mình, điều đó có nghĩa là ta muốn tìm


$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$
:eqlabel:`eq_max_like`


<!--
By Bayes' rule, this is the same thing as
-->

Theo quy tắc Bayes, điều này giống với


$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$


<!--
The expression $P(X)$, a parameter agnostic probability of generating the data, does not depend on $\boldsymbol{\theta}$ at all, 
and so can be dropped without changing the best choice of $\boldsymbol{\theta}$.
Similarly, we may now posit that we have no prior assumption on which set of parameters are better than any others, 
so we may declare that $P(\boldsymbol{\theta})$ does not depend on theta either!
This, for instance, makes sense in our coin flipping example where the probability it comes up heads could be 
any value in $[0,1]$ without any prior belief it is fair or not (often referred to as an *uninformative prior*).
Thus we see that our application of Bayes' rule shows that our best choice of $\boldsymbol{\theta}$ is the maximum likelihood estimate for $\boldsymbol{\theta}$:
-->

Biểu thức $P(X)$, một tham số xác suất không phân biệt để tạo dữ liệu, hoàn toàn không phụ thuộc vào $\boldsymbol{\theta}$,
và do đó, bạn có thể bỏ mà không thay đổi lựa chọn tốt nhất của $\boldsymbol{\theta}$.
Tương tự, bây giờ ta có thể cho rằng chúng ta không có giả định trước về bộ tham số nào tốt hơn bất kỳ bộ tham số nào khác,
vì vậy ta có thể phát biểu rằng $P(\boldsymbol{\theta})$ cũng không phụ thuộc vào theta!
Ví dụ, điều này có ý nghĩa trong ví dụ lật đồng xu của chúng ta, trong đó xác suất để ra mặt ngửa có thể là
bất kỳ giá trị nào trong khoảng $[0,1]$ mà không có bất kỳ sự tin tưởng nào trước đó là giá trị công bằng hay không (thường được gọi là *không thông tin trước đó*).
Do đó, chúng tôi thấy rằng việc áp dụng quy tắc Bayes cho thấy rằng lựa chọn tốt nhất của chúng tôi về $\boldsymbol{\theta}$ là ước tính khả năng xảy ra tối đa cho $\boldsymbol{\theta}$:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$


<!--
As a matter of common terminology, the probability of the data given the parameters ($P(X \mid \boldsymbol{\theta})$) is referred to as the *likelihood*.
-->

Theo thuật ngữ thông thường, xác suất của dữ liệu cho trước các tham số ($P(X \mid \boldsymbol{\theta})$) được gọi là *hợp lý*.


<!--
### A Concrete Example
-->

### Một ví dụ cụ thể


<!--
Let us see how this works in a concrete example.
Suppose that we have a single parameter $\theta$ representing the probability that a coin flip is heads.
Then the probability of getting a tails is $1-\theta$, and so if our observed data $X$ is a sequence with $n_H$ heads and $n_T$ tails, 
we can use the fact that independent probabilities multiply to see that 
-->

Hãy để chúng tôi xem cách này hoạt động như thế nào trong một ví dụ cụ thể.
Giả sử rằng chúng ta có một tham số duy nhất $\theta$ biểu diễn cho xác suất mà một lần lật đồng xu là mặt ngửa.
Khi đó, xác suất nhận được mặt sấp là $1-\theta$, và vì vậy nếu dữ liệu quan sát $X$ của chúng ta  là một chuỗi có $n_H$ mặt ngửa và $n_T$ mặt sấp,
chúng ta có thể sử dụng thực tế là các xác suất độc lập nhân lên để thấy rằng


$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$


<!--
If we flip $13$ coins and get the sequence "HHHTHTTHHHHHT", which has $n_H = 9$ and $n_T = 4$, we see that this is
-->

Nếu chúng ta lật $ 13 $ đồng xu và nhận được chuỗi "HHHTHTTHHHHHT", có $n_H = 9$ và $n_T = 4$, chúng ta thấy rằng đây là


$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$


<!--
One nice thing about this example will be that we know the answer going in.
Indeed, if we said verbally, "I flipped 13 coins, and 9 came up heads, what is our best guess for the probability that the coin comes us heads?, " 
everyone would correctly guess $9/13$.
What this maximum likelihood method will give us is a way to get that number from first principals in a way that will generalize to vastly more complex situations.
-->

Một điều thú vị về ví dụ này là chúng ta biết câu trả lời.
Thật vậy, nếu chúng ta phát biểu bằng lời, "Tôi đã lật 13 đồng xu và 9 đồng xu ra mặt ngửa, dự đoán tốt nhất của ta về xác suất đồng xu đến với chúng ta là bao nhiêu?"
mọi người sẽ đoán đúng $9/13$.
Điều mà phương pháp khả năng xảy ra tối đa này sẽ cung cấp cho chúng ta là một cách để lấy con số đó từ các nguyên tắc đầu tiên theo cách tổng quát hóa cho các tình huống phức tạp hơn rất nhiều.


<!--
For our example, the plot of $P(X \mid \theta)$ is as follows:
-->

Để cho ví dụ của ta, cụ thể của $P(X \mid \theta)$ như sau:


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


<!--
This has its maximum value somewhere near our expected $9/13 \approx 0.7\ldots$.
To see if it is exactly there, we can turn to calculus.
Notice that at the maximum, the function is flat.
Thus, we could find the maximum likelihood estimate :eqref:`eq_max_like` by finding the values of $\theta$ where the derivative is zero, 
and finding the one that gives the highest probability. We compute:
-->

Giá trị này có giá trị tối đa ở đâu đó gần $9/13 \approx 0.7\ldots$.
Để xem nó có chính xác ở đó không, chúng ta có thể chuyển sang giải tích.
Chú ý rằng ở cực đại, hàm là đồng phẳng.
Do đó, ta có thể tìm ước tính khả năng xảy ra tối đa :eqref:`eq_max_like` bằng cách tìm các giá trị của $\theta$ trong đó đạo hàm bằng 0,
và tìm giá trị trả về xác suất cao nhất. Ta tính toán:


$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$


<!--
This has three solutions: $0$, $1$ and $9/13$.
The first two are clearly minima, not maxima as they assign probability $0$ to our sequence.
The final value does *not* assign zero probability to our sequence, and thus must be the maximum likelihood estimate $\hat \theta = 9/13$.
-->

Hàm này có ba nghiệm: $0$, $1$ và $9/13$.
Hai giá trị đầu tiên rõ ràng là cực tiểu, không phải cực đại vì chúng gán xác suất $0$ cho chuỗi của chúng ta.
Giá trị cuối cùng *không* gán xác suất bằng 0 cho chuỗi và do đó nó phải là ước tính hợp lý cực đại $\hat \theta = 9/13$.


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Numerical Optimization and the Negative Log-Likelihood
-->

### *dịch tiêu đề trên*


<!--
The previous example is nice, but what if we have billions of parameters and data examples.
-->

*dịch đoạn phía trên*


<!--
First notice that, if we make the assumption that all the data examples are independent, 
we can no longer practically consider the likelihood itself as it is a product of many probabilities.
Indeed, each probability is in $[0,1]$, say typically of value about $1/2$, and the product of $(1/2)^{1000000000}$ is far below machine precision.
We cannot work with that directly.  
-->

*dịch đoạn phía trên*


<!--
However, recall that the logarithm turns products to sums, in which case 
-->

*dịch đoạn phía trên*


$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$


<!--
This number fits perfectly within even a single precision $32$-bit float.
Thus, we should consider the *log-likelihood*, which is
-->

*dịch đoạn phía trên*


$$
\log(P(X \mid \boldsymbol{\theta})).
$$


<!--
Since the function $x \mapsto \log(x)$ is increasing, maximizing the likelihood is the same thing as maximizing the log-likelihood.
Indeed in :numref:`sec_naive_bayes` we will see this reasoning applied when working with the specific example of the naive Bayes classifier.
-->

*dịch đoạn phía trên*


<!--
We often work with loss functions, where we wish to minimize the loss.
We may turn maximum likelihood into the minimization of a loss by taking $-\log(P(X \mid \boldsymbol{\theta}))$, which is the *negative log-likelihood*.
-->

*dịch đoạn phía trên*


<!--
To illustrate this, consider the coin flipping problem from before, and pretend that we do not know the closed form solution. We may compute that
-->

*dịch đoạn phía trên*


$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$


<!--
This can be written into code, and freely optimized even for billions of coin flips.
-->

*dịch đoạn phía trên*


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


<!--
Numerical convenience is only one reason people like to use negative log-likelihoods.
Indeed, there are a several reasons that it can be preferable.
-->

*dịch đoạn phía trên*


<!--
The second reason we consider the log-likelihood is the simplified application of calculus rules.
As discussed above, due to independence assumptions, most probabilities we encounter in machine learning are products of individual probabilities.
-->

*dịch đoạn phía trên*


$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$


<!--
This means that if we directly apply the product rule to compute a derivative we get
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$


<!--
This requires $n(n-1)$ multiplications, along with $(n-1)$ additions, so it is total of quadratic time in the inputs!
Sufficient cleverness in grouping terms will reduce this to linear time, but it requires some thought.
For the negative log-likelihood we have instead
-->

*dịch đoạn phía trên*


$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$


<!--
which then gives
-->

*dịch đoạn phía trên*


$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
This requires only $n$ divides and $n-1$ sums, and thus is linear time in the inputs.
-->

*dịch đoạn phía trên*


<!--
The third and final reason to consider the negative log-likelihood is the relationship to information theory, 
which we will discuss in detail in :numref:`sec_information_theory`.
This is a rigorous mathematical theory which gives a way to measure the degree of information or randomness in a random variable.
The key object of study in that field is the entropy which is 
-->

*dịch đoạn phía trên*


$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$


<!--
which measures the randomness of a source. Notice that this is nothing more than the average $-\log$ probability, 
and thus if we take our negative log-likelihood and divide by the number of data examples, we get a relative of entropy known as cross-entropy.
This theoretical interpretation alone would be sufficiently compelling to motivate reporting the average negative log-likelihood over the dataset as a way of measuring model performance.
-->

*dịch đoạn phía trên*


<!--
## Maximum Likelihood for Continuous Variables
-->

## *dịch tiêu đề trên*


<!--
Everything that we have done so far assumes we are working with discrete random variables, but what if we want to work with continuous ones?
-->

*dịch đoạn phía trên*


<!--
The short summary is that nothing at all changes, except we replace all the instances of the probability with the probability density.
Recalling that we write densities with lower case $p$, this means that for example we now say
-->

*dịch đoạn phía trên*


$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$


<!--
The question becomes, "Why is this OK?"
After all, the reason we introduced densities was because probabilities of getting specific outcomes themselves was zero, 
and thus is not the probability of generating our data for any set of parameters zero?
-->

*dịch đoạn phía trên*


<!--
Indeed, this is the case, and understanding why we can shift to densities is an exercise in tracing what happens to the epsilons.
-->

*dịch đoạn phía trên*


<!--
Let us first re-define our goal.
Suppose that for continuous random variables we no longer want to compute the probability of getting exactly the right value, 
but instead matching to within some range $\epsilon$.
For simplicity, we assume our data is repeated observations $x_1, \ldots, x_N$ of identically distributed random variables $X_1, \ldots, X_N$.
As we have seen previously, this can be written as
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$


<!--
Thus, if we take negative logarithms of this we obtain
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$


<!--
If we examine this expression, the only place that the $\epsilon$ occurs is in the additive constant $-N\log(\epsilon)$.
This does not depend on the parameters $\boldsymbol{\theta}$ at all, so the optimal choice of $\boldsymbol{\theta}$ does not depend on our choice of $\epsilon$!
If we demand four digits or four-hundred, the best choice of $\boldsymbol{\theta}$ remains the same, thus we may freely drop the epsilon to see that what we want to optimize is
-->

*dịch đoạn phía trên*


$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$


<!--
Thus, we see that the maximum likelihood point of view can operate with continuous random variables 
as easily as with discrete ones by replacing the probabilities with probability densities.
-->

*dịch đoạn phía trên*


## Tóm tắt

<!--
* The maximum likelihood principle tells us that the best fit model for a given dataset is the one that generates the data with the highest probability.
* Often people work with the negative log-likelihood instead for a variety of reasons: numerical stability, 
conversion of products to sums (and the resulting simplification of gradient computations), and theoretical ties to information theory.
* While simplest to motivate in the discrete setting, it may be freely generalized to the continuous setting as well by maximizing the probability density assigned to the datapoints.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. Suppose that you know that a random variable has density $\frac{1}{\alpha}e^{-\alpha x}$ for some value $\alpha$.
You obtain a single observation from the random variable which is the number $3$.  What is the maximum likelihood estimate for $\alpha$?
2. Suppose that you have a dataset of samples $\{x_i\}_{i=1}^N$ drawn from a Gaussian with unknown mean, but variance $1$.
What is the maximum likelihood estimate for the mean?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC ===================================-->


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/416)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Trần Yến Thy

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 


*Lần cập nhật gần nhất: 11/09/2020. (Cập nhật lần cuối từ nội dung gốc: 05/08/2020)*

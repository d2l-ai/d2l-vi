<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Maximum Likelihood
-->

# Hợp lý Cực đại
:label:`sec_maximum_likelihood`


<!--
One of the most commonly encountered way of thinking in machine learning is the maximum likelihood point of view.
This is the concept that when working with a probabilistic model with unknown parameters, 
the parameters which make the data have the highest probability are the most likely ones.
-->

Một trong những cách tư duy thường gặp nhất trong học máy là quan điểm về hợp lý cực đại.
Đây là khái niệm mà khi làm việc với một mô hình xác suất với các tham số chưa biết,
các tham số làm cho dữ liệu có xác suất cao nhất là những tham số hợp lý nhất.


<!--
## The Maximum Likelihood Principle
-->

## Nguyên lý Hợp lý Cực đại


<!--
This has a Bayesian interpretation which can be helpful to think about.
Suppose that we have a model with parameters $\boldsymbol{\theta}$ and a collection of data examples $X$.
For concreteness, we can imagine that $\boldsymbol{\theta}$ is a single value representing the probability that a coin comes up heads when flipped, 
and $X$ is a sequence of independent coin flips.
We will look at this example in depth later.
-->

Nguyên lý này có cách hiểu theo hướng Bayes khá hữu ích.
Giả sử rằng ta có một mô hình với các tham số $\boldsymbol{\theta}$ và một tập hợp các mẫu dữ liệu $X$.
Cụ thể hơn, ta có thể tưởng tượng rằng $\boldsymbol{\theta}$ là một giá trị duy nhất đại diện cho xác suất một đồng xu ngửa khi tung,
và $X$ là một chuỗi các lần tung đồng xu độc lập.
Chúng ta sẽ xem xét ví dụ này sâu hơn ở phần sau.


<!--
If we want to find the most likely value for the parameters of our model, that means we want to find
-->

Nếu ta muốn tìm giá trị hợp lý nhất cho các tham số của mô hình, điều đó có nghĩa là ta muốn tìm


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

Biểu thức $P(X)$ là xác suất tạo dữ liệu hoàn toàn không phụ thuộc vào tham số, tức $\boldsymbol{\theta}$,
do đó ta có thể bỏ nó mà lựa chọn tốt nhất của $\boldsymbol{\theta}$ vẫn không thay đổi.
Tương tự, bây giờ ta có thể cho rằng chúng ta không có giả định trước về bộ tham số nào tốt hơn bất kỳ bộ tham số nào khác,
vì vậy ta có thể phát biểu rằng $P(\boldsymbol{\theta})$ cũng không phụ thuộc vào theta!
Điều này có vẻ hợp lý trong ví dụ lật đồng xu, trong đó xác suất để ra mặt ngửa có thể là
bất kỳ giá trị nào trong khoảng $[0,1]$ mà không có bất kỳ sự tin tưởng nào trước đó rằng đồng xu có công bằng hay không (thường được gọi là *tiên nghiệm không chứa thông tin*).
Do đó, ta thấy rằng việc áp dụng quy tắc Bayes sẽ chỉ ra lựa chọn tốt nhất cho $\boldsymbol{\theta}$ là ước lượng hợp lý cực đại cho $\boldsymbol{\theta}$:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$


<!--
As a matter of common terminology, the probability of the data given the parameters ($P(X \mid \boldsymbol{\theta})$) is referred to as the *likelihood*.
-->

Theo thuật ngữ thông thường, xác suất của dữ liệu biết các tham số ($P(X \mid \boldsymbol{\theta})$) được gọi là *độ hợp lý*.


<!--
### A Concrete Example
-->

### Một ví dụ Cụ thể


<!--
Let us see how this works in a concrete example.
Suppose that we have a single parameter $\theta$ representing the probability that a coin flip is heads.
Then the probability of getting a tails is $1-\theta$, and so if our observed data $X$ is a sequence with $n_H$ heads and $n_T$ tails, 
we can use the fact that independent probabilities multiply to see that 
-->

Hãy cùng xem phương pháp này hoạt động như thế nào trong một ví dụ cụ thể.
Giả sử rằng chúng ta có một tham số duy nhất $\theta$ biểu diễn cho xác suất tung đồng xu một lần được mặt ngửa.
Khi đó, xác suất nhận được mặt sấp là $1-\theta$, và vì vậy nếu dữ liệu quan sát $X$ của chúng ta là một chuỗi có $n_H$ mặt ngửa và $n_T$ mặt sấp,
chúng ta có thể tận dụng việc các xác suất độc lập nhân được với nhau để thấy


$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$


<!--
If we flip $13$ coins and get the sequence "HHHTHTTHHHHHT", which has $n_H = 9$ and $n_T = 4$, we see that this is
-->

Nếu chúng ta tung $ 13 $ đồng xu và nhận được chuỗi "HHHTHTTHHHHHT", có $n_H = 9$ và $n_T = 4$, chúng ta thấy rằng đây là


$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$


<!--
One nice thing about this example will be that we know the answer going in.
Indeed, if we said verbally, "I flipped 13 coins, and 9 came up heads, what is our best guess for the probability that the coin comes us heads?, " 
everyone would correctly guess $9/13$.
What this maximum likelihood method will give us is a way to get that number from first principals in a way that will generalize to vastly more complex situations.
-->

Một điều thú vị về ví dụ này là chúng ta biết trước câu trả lời.
Thật vậy, nếu chúng ta phát biểu bằng lời, "Tôi đã tung 13 đồng xu và 9 đồng xu ra mặt ngửa, dự đoán tốt nhất cho xác suất tung đồng xu được mặt ngửa là bao nhiêu?"
mọi người sẽ đều đoán đúng $9/13$.
Điều mà phương pháp khả năng hợp lý cực đại cung cấp cho chúng ta là một cách để thu được con số đó từ các nguyên tắc cơ bản sao cho có thể khái quát được cho các tình huống phức tạp hơn rất nhiều.


<!--
For our example, the plot of $P(X \mid \theta)$ is as follows:
-->

Với ví dụ của ta, đồ thị của $P(X \mid \theta)$ có dạng như sau:


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

Xác suất này có giá trị tối đa ở đâu đó gần $9/13 \approx 0.7\ldots$ như đã dự đoán.
Để kiếm tra xem nó có nằm chính xác ở đó không, chúng ta có thể tận dụng giải tích.
Chú ý rằng ở điểm cực đại, hàm số sẽ nằm ngang.
Do đó, ta có thể tìm ước lượng hợp lý cực đại :eqref:`eq_max_like` bằng cách tìm các giá trị của $\theta$ có đạo hàm bằng 0,
rồi xem giá trị nào trả về xác suất cao nhất. Ta tính toán:


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

Phương trình này có ba nghiệm: $0$, $1$ và $9/13$.
Hai giá trị đầu tiên rõ ràng là cực tiểu, không phải cực đại vì chúng gán xác suất bằng $0$ cho chuỗi kết quả tung đồng xu.
Giá trị cuối cùng *không* gán xác suất bằng 0 cho chuỗi và do đó nó phải là ước lượng hợp lý cực đại $\hat \theta = 9/13$.


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Numerical Optimization and the Negative Log-Likelihood
-->

### Tối ưu hoá Số học và hàm Log hợp lí Âm


<!--
The previous example is nice, but what if we have billions of parameters and data examples.
-->

Ví dụ trước khá ổn, nhưng điều gì sẽ xảy ra nếu chúng ta có hàng tỷ tham số và mẫu dữ liệu.

<!--
First notice that, if we make the assumption that all the data examples are independent, 
we can no longer practically consider the likelihood itself as it is a product of many probabilities.
Indeed, each probability is in $[0,1]$, say typically of value about $1/2$, and the product of $(1/2)^{1000000000}$ is far below machine precision.
We cannot work with that directly.  
-->

Trước tiên, hãy lưu ý rằng, nếu ta giả định rằng tất cả các mẫu dữ liệu là độc lập, 
thì chúng ta gần như không thể xem xét độ hợp lý vì nó là tích của nhiều xác suất.
Thật vậy, mỗi xác suất nằm trong đoạn $[0,1]$, ví dụ như $1/2$ và tích của $(1/2)^{1000000000}$ thấp hơn nhiều so với độ chính xác của máy. 
Ta không thể làm việc trực tiếp với biểu thức này.

<!--
However, recall that the logarithm turns products to sums, in which case 
-->

Tuy nhiên, hãy nhớ lại rằng hàm log biến đổi tích thành tổng, trong trường hợp đó thì

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$


<!--
This number fits perfectly within even a single precision $32$-bit float.
Thus, we should consider the *log-likelihood*, which is
-->

Con số này hoàn toàn nằm vừa trong một số thực $32$-bit với độ chính xác đơn.
Vì vậy, chúng ta nên xem xét *log hợp lý* (*log-likelihood*), chính là

$$
\log(P(X \mid \boldsymbol{\theta})).
$$


<!--
Since the function $x \mapsto \log(x)$ is increasing, maximizing the likelihood is the same thing as maximizing the log-likelihood.
Indeed in :numref:`sec_naive_bayes` we will see this reasoning applied when working with the specific example of the naive Bayes classifier.
-->

Vì hàm $x \mapsto \log(x)$ đồng biến, việc cực đại hóa độ hợp lý đồng nghĩa với việc cực đại hóa log hợp lý.
Thật vậy trong :numref:`sec_naive_bayes`, chúng ta sẽ thấy lập luận này được áp dụng khi làm việc với ví dụ cụ thể về bộ phân loại Naive Bayes.

<!--
We often work with loss functions, where we wish to minimize the loss.
We may turn maximum likelihood into the minimization of a loss by taking $-\log(P(X \mid \boldsymbol{\theta}))$, which is the *negative log-likelihood*.
-->

Ta thường làm việc với các hàm mất mát, thứ mà ta muốn cực tiểu hóa.
Ta có thể biến đổi hợp lý cực đại thành việc cực tiểu hóa mất mát bằng cách lấy $-\log(P(X \mid \boldsymbol{\theta}))$, tức *hàm đối log hợp lý (negative log-likelihood)*.


<!--
To illustrate this, consider the coin flipping problem from before, and pretend that we do not know the closed form solution. We may compute that
-->

Để minh họa điều này, hãy xem xét bài toán tung đồng xu trước đó và giả vờ rằng ta không biết nghiệm dạng đóng. Ta có thể tính ra

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$


<!--
This can be written into code, and freely optimized even for billions of coin flips.
-->

Đẳng thức này có thể được lập trình và được tối ưu hóa thoải mái ngay cả với hàng tỷ lần tung đồng xu.

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

Sự thuận tiện số học chỉ là một lý do khiến mọi người thích dùng hàm đối log hợp lý.
Thật vậy, còn có một số lý do khác cho việc nó được ưa chuộng.

<!--
The second reason we consider the log-likelihood is the simplified application of calculus rules.
As discussed above, due to independence assumptions, most probabilities we encounter in machine learning are products of individual probabilities.
-->

Lý do thứ hai mà ta xem xét đến hàm log hợp lý là việc áp dụng các quy tắc giải tích trở nên đơn giản hơn.
Như đã thảo luận ở trên, do các giả định về tính độc lập, hầu hết các xác suất mà chúng ta gặp phải trong học máy là tích của các xác suất riêng lẻ.

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$


<!--
This means that if we directly apply the product rule to compute a derivative we get
-->
Đẳng thức này có nghĩa là nếu chúng ta áp dựng trực tiếp quy tắc nhân để tính đạo hàm thì ta sẽ có được

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

Biểu thức này đòi hỏi $n(n-1)$ phép nhân, kèm với $(n-1)$ phép cộng, vì vậy thời gian chạy tỉ lệ bình phương so với số lượng đầu vào!
Nếu ta khôn khéo trong việc nhóm các phần tử thì độ phức tạp thời gian sẽ giảm xuống tuyến tính, nhưng việc này yêu cầu ta phải suy nghĩ một chút.
Đối với hàm đối log hợp lý, chúng ta có

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$


<!--
which then gives
-->

sau đó nhận được kết quả là

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
This requires only $n$ divides and $n-1$ sums, and thus is linear time in the inputs.
-->

Đẳng thức này chỉ yêu cầu $n$ phép chia và $n-1$ phép cộng, và do đó thời gian chạy tỉ lệ tuyến tính so với số đầu vào.

<!--
The third and final reason to consider the negative log-likelihood is the relationship to information theory, 
which we will discuss in detail in :numref:`sec_information_theory`.
This is a rigorous mathematical theory which gives a way to measure the degree of information or randomness in a random variable.
The key object of study in that field is the entropy which is 
-->

Lý do thứ ba và cũng là cuối cùng để xem xét hàm đối log hợp lý đó là mối tương quan với lý thuyết thông tin,
mà chúng ta sẽ thảo luận chi tiết tại phần :numref:`sec_information_theory`.
Đây là một lý thuyết toán học chặt chẽ đưa ra cách đo lường mức độ thông tin hoặc tính ngẫu nhiên trong một biến ngẫu nhiên.
Đối tượng nghiên cứu chính trong lĩnh vực đó là entropy


$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$


<!--
which measures the randomness of a source. Notice that this is nothing more than the average $-\log$ probability, 
and thus if we take our negative log-likelihood and divide by the number of data examples, we get a relative of entropy known as cross-entropy.
This theoretical interpretation alone would be sufficiently compelling to motivate reporting the average negative log-likelihood over the dataset as a way of measuring model performance.
-->

công thức trên đo lường tính ngẫu nhiên của một nguồn. Cần lưu ý rằng đây chỉ là giá trị $-\log$ xác suất trung bình,
và do đó, nếu chúng ta lấy hàm đối log hợp lý và chia cho số lượng mẫu dữ liệu, chúng ta sẽ nhận được một đại lượng liên quan khác được gọi là entropy chéo.
Chỉ cần diễn giải lý thuyết này thôi cũng đủ thuyết phục để thúc đẩy việc sử dụng giá trị đối log hợp lý trung bình trên một tập dữ liệu như một cách đo lường chất lượng của mô hình.


<!--
## Maximum Likelihood for Continuous Variables
-->

## Hợp lý Cực đại cho Biến Liên tục


<!--
Everything that we have done so far assumes we are working with discrete random variables, but what if we want to work with continuous ones?
-->

Tất cả những gì chúng ta đã làm vừa rồi đều giả định rằng ta đang làm việc vói biến ngẫu nhiên rời rạc, nhưng nếu chúng ta muốn làm việc với các biến liên tục thì sao?

<!--
The short summary is that nothing at all changes, except we replace all the instances of the probability with the probability density.
Recalling that we write densities with lower case $p$, this means that for example we now say
-->

Nói ngắn gọn thì không có điều gì thay đổi cả, trừ việc ta thay thế tất cả xác suất bằng mật độ xác suất.
Hãy nhớ lại rằng chúng ta ký hiệu mật độ bằng chữ thường $p$, nghĩa là bây giờ ta sẽ có

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$


<!--
The question becomes, "Why is this OK?"
After all, the reason we introduced densities was because probabilities of getting specific outcomes themselves was zero,
and thus is not the probability of generating our data for any set of parameters zero?
-->

Câu hỏi lúc này trở thành, "Tại sao điều này lại ổn?"
Rốt cuộc, lý do chúng ta đưa ra khái niệm mật độ là vì chính bản thân xác suất nhận được một kết quả cụ thể là bằng không,
và do đó chẳng phải xác suất sinh dữ liệu đối với tập hợp tham số bất kỳ sẽ bằng không sao?

<!--
Indeed, this is the case, and understanding why we can shift to densities is an exercise in tracing what happens to the epsilons.
-->

Quả thật điều này là đúng, và việc hiểu tại sao chúng ta có thể chuyển sang mật độ là một bài tập trong việc truy ra những gì xảy ra đối với các epsilon.

<!--
Let us first re-define our goal.
Suppose that for continuous random variables we no longer want to compute the probability of getting exactly the right value, 
but instead matching to within some range $\epsilon$.
For simplicity, we assume our data is repeated observations $x_1, \ldots, x_N$ of identically distributed random variables $X_1, \ldots, X_N$.
As we have seen previously, this can be written as
-->

Đầu tiên hãy xác định lại mục tiêu của mình.
Giả sử rằng đối với các biến ngẫu nhiên liên tục, chúng ta không còn muốn tính xác suất để nhận được một giá trị chính xác,
nhưng thay vào đó ta sẽ tính xác suất trong một phạm vi $\epsilon$ nào đó.
Để đơn giản, ta giả định rằng dữ liệu là các mẫu quan sát lặp lại $x_1, \ldots, x_N$ của các biến ngẫu nhiên được phân phối giống nhau $X_1, \ldots, X_N$.
Như chúng ta đã thấy trước đây, giả định này có thể được biểu diễn như sau

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$


<!--
Thus, if we take negative logarithms of this we obtain
-->

Do đó, nếu ta lấy logarit âm cho kết quả này thì ta sẽ nhận được

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

Nếu chúng ta xem xét biểu thức này, vị trí duy nhất mà $\epsilon$ xuất hiện là tại hằng số cộng $-N\log(\epsilon)$.
Phần tử này hoàn toàn không phụ thuộc vào các tham số $\boldsymbol{\theta}$, vì vậy lựa chọn tối ưu của $\boldsymbol{\theta}$ không phụ thuộc vào sự lựa chọn $\epsilon$!
Dù ta có yêu cầu bốn hoặc bốn trăm chữ số, lựa chọn tốt nhất của $\boldsymbol{\theta}$ vẫn không thay đổi, do đó ta hoàn toàn có thể loại bỏ epsilon để thấy được thứ mà ta muốn tối ưu hóa là


$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$


<!--
Thus, we see that the maximum likelihood point of view can operate with continuous random variables 
as easily as with discrete ones by replacing the probabilities with probability densities.
-->

Do đó, chúng ta thấy rằng quan điểm hợp lý cực đại có thể áp dụng được với các biến ngẫu nhiên liên tục dễ dàng như với các biến rời rạc bằng cách thay thế các xác suất bằng mật độ xác suất.

## Tóm tắt

<!--
* The maximum likelihood principle tells us that the best fit model for a given dataset is the one that generates the data with the highest probability.
* Often people work with the negative log-likelihood instead for a variety of reasons: numerical stability, 
conversion of products to sums (and the resulting simplification of gradient computations), and theoretical ties to information theory.
* While simplest to motivate in the discrete setting, it may be freely generalized to the continuous setting as well by maximizing the probability density assigned to the datapoints.
-->

* Nguyên lý hợp lý cực đại cho ta biết rằng mô hình phù hợp nhất cho một tập dữ liệu nhất định là mô hình tạo ra dữ liệu với xác suất cao nhất.
* Thường thì mọi người hay làm việc với hàm đối log hợp lý hơn vì nhiều lý do: tính ổn định số học, khả năng biến đổi tích thành tổng (dẫn tới việc đơn giản hóa các phép tính gradient) và mối liên hệ về mặt lý thuyết tới lý thuyết thông tin.
* Dù việc áp dụng phương pháp này trong tình huống rời rạc là đơn giản nhất, nó hoàn toàn có thể được tổng quát hóa cho tình huống liên tục bằng cách cực đại hóa mật độ xác suất được gán cho các điểm dữ liệu.


## Bài tập

<!--
1. Suppose that you know that a random variable has density $\frac{1}{\alpha}e^{-\alpha x}$ for some value $\alpha$.
You obtain a single observation from the random variable which is the number $3$.  What is the maximum likelihood estimate for $\alpha$?
2. Suppose that you have a dataset of samples $\{x_i\}_{i=1}^N$ drawn from a Gaussian with unknown mean, but variance $1$.
What is the maximum likelihood estimate for the mean?
-->

1. Giả sử bạn biết rằng một biến ngẫu nhiên có mật độ bằng  $\frac{1}{\alpha}e^{-\alpha x}$ đối với một giá trị $\alpha$ nào đó.
Bạn có được một quan sát duy nhất từ biến ngẫu nhiên là số $3$. Giá trị ước lượng hợp lý cực đại cho $\alpha$ là bao nhiêu?
2. Giả sử rằng bạn có tập dữ liệu với các mẫu $\{x_i\}_{i=1}^N$ được lấy từ một phân phối Gauss với giá trị trung bình chưa biết, nhưng phương sai bằng $1$.
Giá trị ước lượng hợp lý cực đại của trung bình là bao nhiêu?

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
* Phạm Minh Đức

<!-- Phần 2 -->
* Phạm Đăng Khoa
* Phạm Minh Đức
* Phạm Hồng Vinh

<!-- Phần 3 -->
* Phạm Đăng Khoa


*Lần cập nhật gần nhất: 11/09/2020. (Cập nhật lần cuối từ nội dung gốc: 05/08/2020)*

<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Distributions
-->

# Phân phối
:label:`sec_distributions`


<!--
Now that we have learned how to work with probability in both the discrete and the continuous setting, let us get to know some of the common distributions encountered.
Depending on the area of machine learning, we may need to be familiar with vastly more of these, or for some areas of deep learning potentially none at all.
This is, however, a good basic list to be familiar with.
Let us first import some common libraries.
-->

Lúc này ta đã học cách làm việc với xác suất trong tình huống rời rạc và liên tục, hãy làm quen với một số phân phối thường gặp.
Tùy thuộc vào lĩnh vực học máy, ta có thể cần phải làm quen với một lượng các phân phối lớn hơn nhiều, hoặc đối với một số lĩnh vực học sâu thì có khả năng là hoàn toàn không gặp.
Tuy nhiên, đây là một danh sách tốt các phân phối cơ bản để làm quen.
Đầu tiên chúng ta hãy nhập một số thư viện phổ biến.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```


<!--
## Bernoulli
-->

## Phân phối Bernoulli


<!--
This is the simplest random variable usually encountered.
This random variable encodes a coin flip which comes up $1$ with probability $p$ and $0$ with probability $1-p$.
If we have a random variable $X$ with this distribution, we will write
-->

Đây là biến ngẫu nhiên đơn giản nhất thường gặp.
Biến ngẫu nhiên này biểu diễn giá trị mặt ngửa $1$ khi tung một đồng xu với xác suất $p$ và mặt sấp $0$ với xác suất $1-p$.
Nếu ta có một biến ngẫu nhiên $X$ với phân phối này, ta sẽ viết

$$
X \sim \mathrm{Bernoulli}(p).
$$


<!--
The cumulative distribution function is
-->

Hàm phân phối tích lũy là


$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`


<!--
The probability mass function is plotted below.
-->

Hàm khối xác suất có đồ thị như dưới đây.


```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```


<!--
Now, let us plot the cumulative distribution function :eqref:`eq_bernoulli_cdf`.
-->

Bây giờ, hãy vẽ đồ thị cho hàm phân phối tích lũy :eqref:`eq_bernoulli_cdf`.


```{.python .input}
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```


<!--
If $X \sim \mathrm{Bernoulli}(p)$, then:
-->

Nếu  $X \sim \mathrm{Bernoulli}(p)$, thì:


* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.


<!--
We can sample an array of arbitrary shape from a Bernoulli random variable as follows.
-->

Ta có thể lấy mẫu một mảng có kích thước tùy ý từ một biến ngẫu nhiên Bernoulli như sau.


```{.python .input}
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```


<!--
## Discrete Uniform
-->

## Phân phối Đồng nhất Rời rạc


<!--
The next commonly encountered random variable is a discrete uniform.
For our discussion here, we will assume that it is supported on the integers $\{1, 2, \ldots, n\}$, however any other set of values can be freely chosen.
The meaning of the word *uniform* in this context is that every possible value is equally likely.
The probability for each value $i \in \{1, 2, 3, \ldots, n\}$ is $p_i = \frac{1}{n}$.
We will denote a random variable $X$ with this distribution as
-->

Biến ngẫu nhiên thường gặp tiếp theo là biến đồng nhất rời rạc.
Để thảo luận ở đây, ta sẽ giả định là biến này được phân bổ trên các số nguyên $\{1, 2, \ldots, n\}$, tuy nhiên, có thể tự do chọn bất kỳ tập giá trị nào khác.
Ý nghĩa của từ *đồng nhất* trong ngữ cảnh này có nghĩa là mọi giá trị đều có thể xảy ra với khả năng như nhau.
Xác suất cho mỗi giá trị $i \in \{1, 2, 3, \ldots, n\}$ là $p_i = \frac{1}{n}$.
Chúng ta sẽ ký hiệu một biến ngẫu nhiên $X$ với phân phối này là


$$
X \sim U(n).
$$


<!--
The cumulative distribution function is 
-->

Hàm phân phối tích lũy của nó là 


$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \text{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`


<!--
Let us first plot the probability mass function.
-->

Trước hết ta hãy vẽ đồ thị cho hàm khối xác suất.


```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```


<!--
Now, let us plot the cumulative distribution function :eqref:`eq_discrete_uniform_cdf`.
-->

Bây giờ hãy vẽ đồ thị cho hàm phân phối tích luỹ :eqref:`eq_discrete_uniform_cdf`.


```{.python .input}
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```


<!--
If $X \sim U(n)$, then:
-->

Nếu  $X \sim U(n)$, thì

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.


<!--
We can sample an array of arbitrary shape from a discrete uniform random variable as follows.
-->

Ta có thể lấy mẫu một mảng có kích thước tùy ý từ một biến ngẫu nhiên đồng nhất rời rạc như sau.


```{.python .input}
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Continuous Uniform
-->

## *dịch tiêu đề trên*


<!--
Next, let us discuss the continuous uniform distribution.
The idea behind this random variable is that if we increase the $n$ in the discrete uniform distribution, 
and then scale it to fit within the interval $[a, b]$, we will approach a continuous random variable that just picks an arbitrary value in $[a, b]$ all with equal probability.
We will denote this distribution as
-->

*dịch đoạn phía trên*


$$
X \sim U(a, b).
$$


<!--
The probability density function is
-->

*dịch đoạn phía trên*


$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$
:eqlabel:`eq_cont_uniform_pdf`


<!--
The cumulative distribution function is
-->

*dịch đoạn phía trên*


$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`


<!--
Let us first plot the probability density function :eqref:`eq_cont_uniform_pdf`.
-->

*dịch đoạn phía trên*


```{.python .input}
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```


<!--
Now, let us plot the cumulative distribution function :eqref:`eq_cont_uniform_cdf`.
-->

*dịch đoạn phía trên*


```{.python .input}
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```


<!--
If $X \sim U(a, b)$, then:
-->

*dịch đoạn phía trên*


* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.


<!--
We can sample an array of arbitrary shape from a uniform random variable as follows.
Note that it by default samples from a $U(0,1)$, so if we want a different range we need to scale it.
-->

*dịch đoạn phía trên*


```{.python .input}
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```


<!--
## Binomial
-->

## *dịch tiêu đề trên*


<!--
Let us make things a little more complex and examine the *binomial* random variable.
This random variable originates from performing a sequence of $n$ independent experiments, 
each of which has probability $p$ of succeeding, and asking how many successes we expect to see.
-->

*dịch đoạn phía trên*


<!--
Let us express this mathematically.
Each experiment is an independent random variable $X_i$ where we will use $1$ to encode success, and $0$ to encode failure.
Since each is an independent coin flip which is successful with probability $p$, we can say that $X_i \sim \mathrm{Bernoulli}(p)$.
Then, the binomial random variable is
-->

*dịch đoạn phía trên*


$$
X = \sum_{i=1}^n X_i.
$$


<!--
In this case, we will write
-->

*dịch đoạn phía trên*


$$
X \sim \mathrm{Binomial}(n, p).
$$


<!--
To get the cumulative distribution function, we need to notice that getting exactly $k$ successes can occur 
in $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ ways each of which has a probability of $p^k(1-p)^{n-k}$ of occurring.
Thus the cumulative distribution function is
-->

*dịch đoạn phía trên*


$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \text{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`


<!--
Let us first plot the probability mass function.
-->

*dịch đoạn phía trên*


```{.python .input}
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = torch.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```


<!--
Now, let us plot the cumulative distribution function :eqref:`eq_binomial_cdf`.
-->

*dịch đoạn phía trên*


```{.python .input}
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```


<!--
While this result is not simple, the means and variances are.
If $X \sim \mathrm{Binomial}(n, p)$, then:
-->

*dịch đoạn phía trên*


* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.


<!--
This can be sampled as follows.
-->

*dịch đoạn phía trên*


```{.python .input}
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Poisson
-->

## *dịch tiêu đề trên*


<!--
Let us now perform a thought experiment.
We are standing at a bus stop and we want to know how many buses will arrive in the next minute.
Let us start by considering $X^{(1)} \sim \mathrm{Bernoulli}(p)$ which is simply the probability that a bus arrives in the one minute window.
For bus stops far from an urban center, this might be a pretty good approximation.
We may never see more than one bus in a minute.
-->

*dịch đoạn phía trên*


<!--
However, if we are in a busy area, it is possible or even likely that two buses will arrive.
We can model this by splitting our random variable into two parts for the first 30 seconds, or the second 30 seconds.
In this case we can write
-->

*dịch đoạn phía trên*


$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$


<!--
where $X^{(2)}$ is the total sum, and $X^{(2)}_i \sim \mathrm{Bernoulli}(p/2)$.
The total distribution is then $X^{(2)} \sim \mathrm{Binomial}(2, p/2)$.
-->

*dịch đoạn phía trên*


<!--
Why stop here?  Let us continue to split that minute into $n$ parts.
By the same reasoning as above, we see that
-->

*dịch đoạn phía trên*


$$X^{(n)} \sim \mathrm{Binomial}(n, p/n).$$
:eqlabel:`eq_eq_poisson_approx`


<!--
Consider these random variables.
By the previous section, we know that :eqref:`eq_eq_poisson_approx` has mean $\mu_{X^{(n)}} = n(p/n) = p$, and variance $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$.
If we take $n \rightarrow \infty$, we can see that these numbers stabilize to $\mu_{X^{(\infty)}} = p$, and variance $\sigma_{X^{(\infty)}}^2 = p$.
This indicates that there *could be* some random variable we can define in this infinite subdivision limit.
-->

*dịch đoạn phía trên*


<!--
This should not come as too much of a surprise, since in the real world we can just count the number of bus arrivals,
however it is nice to see that our mathematical model is well defined.
This discussion can be made formal as the *law of rare events*.
-->

*dịch đoạn phía trên*


<!--
Following through this reasoning carefully, we can arrive at the following model.
We will say that $X \sim \mathrm{Poisson}(\lambda)$ if it is a random variable which takes the values $\{0,1,2, \ldots\}$ with probability
-->

*dịch đoạn phía trên*


$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$
:eqlabel:`eq_poisson_mass`


<!--
The value $\lambda > 0$ is known as the *rate* (or the *shape* parameter), and denotes the average number of arrivals we expect in one unit of time.
-->

*dịch đoạn phía trên*


<!--
We may sum this probability mass function to get the cumulative distribution function.
-->

*dịch đoạn phía trên*


$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \text{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`


<!--
Let us first plot the probability mass function :eqref:`eq_poisson_mass`.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

```{.python .input}
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```


```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```


<!--
Now, let us plot the cumulative distribution function :eqref:`eq_poisson_cdf`.
-->

*dịch đoạn phía trên*


```{.python .input}
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```


<!--
As we saw above, the means and variances are particularly concise.
If $X \sim \mathrm{Poisson}(\lambda)$, then:
-->

*dịch đoạn phía trên*


* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.


<!--
This can be sampled as follows.
-->

*dịch đoạn phía trên*


```{.python .input}
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Gaussian
-->

## *dịch tiêu đề trên*


<!--
Now Let us try a different, but related experiment.
Let us say we again are performing $n$ independent $\mathrm{Bernoulli}(p)$ measurements $X_i$.
The distribution of the sum of these is $X^{(n)} \sim \mathrm{Binomial}(n, p)$.
Rather than taking a limit as $n$ increases and $p$ decreases, Let us fix $p$, and then send $n \rightarrow \infty$.
In this case $\mu_{X^{(n)}} = np \rightarrow \infty$ and $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$, 
so there is no reason to think this limit should be well defined.
-->

*dịch đoạn phía trên*


<!--
However, not all hope is lost!
Let us just make the mean and variance be well behaved by defining
-->

*dịch đoạn phía trên*


$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$


<!--
This can be seen to have mean zero and variance one, and so it is plausible to believe that it will converge to some limiting distribution.
If we plot what these distributions look like, we will become even more convinced that it will work.
-->

*dịch đoạn phía trên*


```{.python .input}
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```


```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```


<!--
One thing to note: compared to the Poisson case, we are now dividing by the standard deviation which means that we are squeezing the possible outcomes into smaller and smaller areas.
This is an indication that our limit will no longer be discrete, but rather a continuous.
-->

*dịch đoạn phía trên*


<!--
A derivation of what occurs is beyond the scope of this document, but the *central limit theorem* states that as $n \rightarrow \infty$, 
this will yield the Gaussian Distribution (or sometimes normal distribution).
More explicitly, for any $a, b$:
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$


<!--
where we say a random variable is normally distributed with given mean $\mu$ and variance $\sigma^2$, written $X \sim \mathcal{N}(\mu, \sigma^2)$ if $X$ has density
-->

*dịch đoạn phía trên*


$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$
:eqlabel:`eq_gaussian_pdf`


<!--
Let us first plot the probability density function :eqref:`eq_gaussian_pdf`.
-->

*dịch đoạn phía trên*


```{.python .input}
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```


<!--
Now, let us plot the cumulative distribution function.
It is beyond the scope of this appendix, but the Gaussian c.d.f. does not have a closed-form formula in terms of more elementary functions.
We will use `erf` which provides a way to compute this integral numerically.
-->

*dịch đoạn phía trên*


```{.python .input}
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(torch.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```


<!--
Keen-eyed readers will recognize some of these terms.
Indeed, we encountered this integral in :numref:`sec_integral_calculus`.
Indeed we need exactly that computation to see that this $p_X(x)$ has total area one and is thus a valid density.
-->

*dịch đoạn phía trên*


<!--
Our choice of working with coin flips made computations shorter, but nothing about that choice was fundamental.
Indeed, if we take any collection of independent identically distributed random variables $X_i$, and form
-->

*dịch đoạn phía trên*


$$
X^{(N)} = \sum_{i=1}^N X_i.
$$


<!--
Then
-->

*dịch đoạn phía trên*


$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$


<!--
will be approximately Gaussian.
There are additional requirements needed to make it work, most commonly $E[X^4] < \infty$, but the philosophy is clear.
-->

*dịch đoạn phía trên*


<!--
The central limit theorem is the reason that the Gaussian is fundamental to probability, statistics, and machine learning.
Whenever we can say that something we measured is a sum of many small independent contributions, we can assume that the thing being measured will be close to Gaussian.  
-->

*dịch đoạn phía trên*


<!--
There are many more fascinating properties of Gaussians, and we would like to discuss one more here.
The Gaussian is what is known as a *maximum entropy distribution*.
We will get into entropy more deeply in :numref:`sec_information_theory`, however all we need to know at this point is that it is a measure of randomness.
In a rigorous mathematical sense, we can think of the Gaussian as the *most* random choice of random variable with fixed mean and variance.
Thus, if we know that our random variable has some mean and variance, the Gaussian is in a sense the most conservative choice of distribution we can make.
-->

*dịch đoạn phía trên*


<!--
To close the section, Let us recall that if $X \sim \mathcal{N}(\mu, \sigma^2)$, then:
-->

*dịch đoạn phía trên*


* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.


<!--
We can sample from the Gaussian (or standard normal) distribution as shown below.
-->

*dịch đoạn phía trên*


```{.python .input}
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
## Exponential Family
-->

## Họ hàm Mũ
:label:`subsec_exponential_family`


<!--
One shared property for all the distributions listed above is that they all belong to which is known as the *exponential family*.
The exponential family is a set of distributions whose density can be expressed in the following form:
-->

Một tính chất chung của tất cả các phân phối liệt kê ở trên là chúng đều thuộc họ được gọi là *họ hàm mũ (exponential family)*.
Họ hàm mũ là tập các phân phối có mật độ có thể được biểu diễn dưới dạng sau:


$$p(\mathbf{x} | \mathbf{\eta}) = h(\mathbf{x}) \cdot \mathrm{exp} \big{(} \eta^{\top} \cdot T\mathbf(x) - A(\mathbf{\eta}) \big{)}$$
:eqlabel:`eq_exp_pdf`


<!--
As this definition can be a little subtle, let us examine it closely.  
-->

Do định nghĩa này có thể hơi khó hiểu, hãy cùng xem xét kĩ lưỡng hơn.


<!--
First, $h(\mathbf{x})$ is known as the *underlying measure* or the *base measure*.
This can be viewed as an original choice of measure we are modifying with our exponential weight.  
-->

Đầu tiên, $h(\mathbf{x})$ được gọi là *phép đo cơ bản (underlying measure)* hay *phép đo cơ sở (base measure)*.
Đây có thể được coi như lựa chọn ban đầu cho phép đo mà ta đang điều chỉnh với trọng số mũ.


<!--
Second, we have the vector $\mathbf{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in \mathbb{R}^l$ called the *natural parameters* or *canonical parameters*.
These define how the base measure will be modified.
The natural parameters enter into the new measure by taking the dot product of these parameters against some function 
$T(\cdot)$ of $\mathbf{x}= (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ and exponentiated.
$T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ is called the *sufficient statistics* for $\eta$.
This name is used since the information represented by $T(\mathbf{x})$ is sufficient to calculate the 
probability density and no other information from the sample $\mathbf{x}$'s are required.
-->

Thứ hai, ta có vector $\mathbf{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in \mathbb{R}^l$ được gọi là *tham số tự nhiên (natural parameters)* hay *tham số chính tắc (canonical parameters)*.
Các vector này xác định phép đo cơ sở sẽ được điều chỉnh thế nào.
Các tham số tự nhiên tiến hành phép đo mới bằng cách tính tích vô hướng của các tham số này với hàm
$T(\cdot)$ nào đó của $\mathbf{x}= (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ và lấy luỹ thừa.
$T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ được gọi là *thống kê đủ (sufficient statistics)* của $\eta$.
Cái tên này được sử dụng do thông tin biểu diễn bởi $T(\mathbf{x})$ đủ để tính
mật độ xác suất và không cần thêm bất cứ thông tin nào khác từ mẫu của $\mathbf{x}$.


<!--
Third, we have $A(\mathbf{\eta})$, which is referred to as the *cumulant function*,
which ensures that the above distribution :eqref:`eq_exp_pdf` integrates to one, i.e.,
-->

Thứ ba, ta có $A(\mathbf{\eta})$, được gọi là *hàm tích luỹ (cumulant function)*,
hàm này đảm bảo phân phối trên :eqref:`eq_exp_pdf` có tích phân bằng 1, ví dụ như


$$  A(\mathbf{\eta}) = \log \left[\int h(\mathbf{x}) \cdot \mathrm{exp} 
\big{(}\eta^{\top} \cdot T\mathbf(x) \big{)} dx \right].$$


<!--
To be concrete, let us consider the Gaussian.
Assuming that $\mathbf{x}$ is an univariate variable, we saw that it had a density of
-->

Để ngắn gọn, ta xét phân phối Gauss.
Giả sử rằng $\mathbf{x}$ là biến đơn thuộc tính (*univariate variable*), ta thấy rằng nó có mật độ bằng

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \mathrm{exp} 
\Big{\{} \frac{-(x-\mu)^2}{2 \sigma^2} \Big{\}} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \mathrm{exp} \Big{\{} \frac{\mu}{\sigma^2}x 
- \frac{1}{2 \sigma^2} x^2 - \big{(} \frac{1}{2 \sigma^2} \mu^2 
+ \log(\sigma) \big{)} \Big{\}} .
\end{aligned}
$$


<!--
This matches the definition of the exponential family with:
-->

*dịch đoạn phía trên*


<!--
* *underlying measure*: $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *natural parameters*: $\eta = \begin{bmatrix} \eta_1 \\ \eta_2 \end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\ \frac{1}{2 \sigma^2}  \end{bmatrix}$,
* *sufficient statistics*: $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, and
* *cumulant function*: $A(\eta) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma) = \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.
-->

*dịch đoạn phía trên*


<!--
It is worth noting that the exact choice of each of above terms is somewhat arbitrary.
Indeed, the important feature is that the distribution can be expressed in this form, not the exact form itself.
-->

*dịch đoạn phía trên*


<!--
As we allude to in :numref:`subsec_softmax_and_derivatives`, a widely used technique is to assume that the final output $\mathbf{y}$ follows an exponential family distribution.
The exponential family is a common and powerful family of distributions encountered frequently in machine learning.
-->

*dịch đoạn phía trên*


## Tóm tắt

<!--
* Bernoulli random variables can be used to model events with a yes/no outcome.
* Discrete uniform distributions model selects from a finite set of possibilities.
* Continuous uniform distributions select from an interval.
* Binomial distributions model a series of Bernoulli random variables, and count the number of successes.
* Poisson random variables model the arrival of rare events.
* Gaussian random variables model the result of adding a large number of independent random variables together.
* All the above distributions belong to exponential family.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. What is the standard deviation of a random variable that is the difference $X-Y$ of two independent binomial random variables $X, Y \sim \mathrm{Binomial}(16, 1/2)$.
2. If we take a Poisson random variable $X \sim \mathrm{Poisson}(\lambda)$ and consider $(X - \lambda)/\sqrt{\lambda}$ as $\lambda \rightarrow \infty$, 
we can show that this becomes approximately Gaussian. Why does this make sense?
3. What is the probability mass function for a sum of two discrete uniform random variables on $n$ elements?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/417)
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
* Nguyễn Mai Hoàng Long
* Lê Khắc Hồng Phúc
* Phạm Minh Đức

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* 

<!-- Phần 6 -->
* Đỗ Trường Giang

<!-- Phần 7 -->
* 

*Lần cập nhật gần nhất: 10/09/2020. (Cập nhật lần cuối từ nội dung gốc: 27/07/2020)*

<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Statistics
-->

# *dịch tiêu đề phía trên*
:label:`sec_statistics`

<!--
Undoubtedly, to be a top deep learning practitioner, the ability to train the state-of-the-art and high accurate models is crucial.  
However, it is often unclear when improvements are significant, or only the result of random fluctuations in the training process.  
To be able to discuss uncertainty in estimated values, we must learn some statistics.
-->

*dịch đoạn phía trên*


<!--
The earliest reference of *statistics* can be traced back to an Arab scholar Al-Kindi in the $9^{\mathrm{th}}$-century, who gave a detailed description of how to use statistics and frequency analysis to decipher encrypted messages. 
After 800 years, the modern statistics arose from Germany in 1700s, when the researchers focused on the demographic and economic data collection and analysis. 
Today, statistics is the science subject that concerns the collection, processing, analysis, interpretation and visualization of data. 
What is more, the core theory of statistics has been widely used in the research within academia, industry, and government.
-->

*dịch đoạn phía trên*


<!--
More specifically, statistics can be divided to *descriptive statistics* and *statistical inference*. 
The former focus on summarizing and illustrating the features of a collection of observed data, which is referred to as a *sample*. 
The sample is drawn from a *population*, denotes the total set of similar individuals, items, or events of our experiment interests. 
Contrary to descriptive statistics, *statistical inference* further deduces the characteristics of a population from the given *samples*, based on the assumptions that the sample distribution can replicate the population distribution at some degree.
-->

*dịch đoạn phía trên*


<!--
You may wonder: “What is the essential difference between machine learning and statistics?” Fundamentally speaking, statistics focuses on the inference problem. 
This type of problems includes modeling the relationship between the variables, such as causal inference, and testing the statistically significance of model parameters, such as A/B testing. 
In contrast, machine learning emphasizes on making accurate predictions, without explicitly programming and understanding each parameter's functionality.
-->

*dịch đoạn phía trên*


<!--
In this section, we will introduce three types of statistics inference methods: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals. 
These methods can help us infer the characteristics of a given population, i.e., the true parameter $\theta$. 
For brevity, we assume that the true parameter $\theta$ of a given population is a scalar value. 
It is straightforward to extend to the case where $\theta$ is a vector or a tensor, thus we omit it in our discussion.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 1 ================================-->

<!-- =================== Bắt đầu dịch Phần 2 ================================-->

<!--
## Evaluating and Comparing Estimators
-->

## Đánh giá và So sánh các Bộ ước lượng

<!--
In statistics, an *estimator* is a function of given samples used to estimate the true parameter $\theta$. 
We will write $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ for the estimate of $\theta$ after observing the samples {$x_1, x_2, \ldots, x_n$}.
-->

Trong thống kê, một *bộ ước lượng* là một hàm sử dụng những mẫu có sẵn để ước lượng giá trị thực của tham số $\theta$.
Ta gọi $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ là ước lượng của $\theta$ sau khi quan sát các mẫu {$x_1, x_2, \ldots, x_n$}.

<!--
We've seen simple examples of estimators before in section :numref:`sec_maximum_likelihood`.  
If you have a number of samples from a Bernoulli random variable, then the maximum likelihood estimate for the probability the random variable is one can be obtained by counting the number of ones observed and dividing by the total number of samples.  
Similarly, an exercise asked you to show that the maximum likelihood estimate of the mean of a Gaussian given a number of samples is given by the average value of all the samples.  
These estimators will almost never give the true value of the parameter, but ideally for a large number of samples the estimate will be close.
-->

Ta đã thấy nhiều ví dụ đơn giản của bộ ước lượng trong phần :numref:`sec_maximum_likelihood`.
Nếu bạn có một số mẫu ngẫu nhiên từ phân phối Bernoulli, thì ước lượng hợp lý cực đại (*maximum likelihood estimate*) cho xác xuất của biến ngẫu nhiên có thể có được bằng cách đếm số lần biến cố một xuất hiện và chia cho tổng số mẫu.
Tương tự, một bài tập yêu cầu bạn chứng minh rằng ước lượng hợp lý cực đại của kỳ vọng của một phân phối Gauss với một số lượng mẫu cho trước là giá trị trung bình của tập mẫu.
Các bộ ước lượng này dường như sẽ không bao giờ cho ra giá trị chính xác của tham số, nhưng với trường hợp số lượng mẫu lớn, ước lượng có được sẽ gần với giá trị thực.

<!--
As an example, we show below the true density of a Gaussian random variable with mean zero and variance one, along with a collection samples from that Gaussian.  
We constructed the $y$ coordinate so every point is visible and the relationship to the original density is clearer.
-->

Như một ví dụ, bên dưới là mật độ của phân phối Gauss với kỳ vọng là không và phương sai là một, cùng với một tập các mẫu lấy ra từ phân phối đó.
Tọa độ $y$ được xây dựng sao cho tất các điểm đều có thể nhìn thấy được và mối quan hệ giữa mật độ mẫu và mật độ gốc của phân phối có thể được nhìn thấy rõ hơn. 

```{.python .input}
import d2l
from mxnet import np, npx
import random
npx.set_np()

# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))

ys = [np.sum(np.exp(-(xs[0:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]

# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)

# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title("Sample Mean: {:.2f}".format(float(np.mean(xs))))
d2l.plt.show()
```

<!--
There can be many ways to compute an estimator of a parameter $\hat{\theta}_n$.  
In this section, we introduce three common methods to evaluate and compare estimators: the mean squared error, the standard deviation, and statistical bias.
-->

Có thể có nhiều cách để tính toán một bộ ước lượng cho một tham số $\hat{\theta}_n$.
Trong phần này, ta sẽ điểm qua ba phương thức phổ biến để đánh giá và so sánh các bộ ước lượng: trung bình bình phương sai số, độ lệch chuẩn và độ chệch thống kê.

<!-- =================== Kết thúc dịch Phần 2 ================================-->

<!-- =================== Bắt đầu dịch Phần 3 ================================-->

<!--
### Mean Squared Error
-->

### *dịch tiêu đề phía trên*

<!--
Perhaps the simplest metric used to evaluate estimators is the *mean squared error (MSE)* (or *$l_2$ loss*) of an estimator can be defined as
-->

*dịch đoạn phía trên*

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

<!--
This allows us to quantify the average squared deviation from the true value.  
MSE is always non-negative. If you have read :numref:`sec_linear_regression`, you will recognize it as the most commonly used regression loss function. 
As a measure to evaluate an estimator, the closer its value to zero, the closer the estimator is close to the true parameter $\theta$.
-->

*dịch đoạn phía trên*


<!--
### Statistical Bias
-->

### *dịch tiêu đề phía trên*

<!--
The MSE provides a natural metric, but we can easily imagine multiple different phenomena that might make it large.  
Two that we will see are fundamentally important are the fluctuation in the estimator due to randomness in the dataset, and systematic error in the estimator due to the estimation procedure.
-->

*dịch đoạn phía trên*


<!--
First, let's measure the systematic error. 
For an estimator $\hat{\theta}_n$, the mathematical illustration of *statistical bias* can be defined as
-->

*dịch đoạn phía trên*

$$\mathrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

<!--
Note that when $\mathrm{bias}(\hat{\theta}_n) = 0$, the expectation of the estimator $\hat{\theta}_n$ is equal to the true value of parameter.  
In this case, we say $\hat{\theta}_n$ is an unbiased estimator.  
In general, an unbiased estimator is better than a biased estimator since its expectation is the same as the true parameter.
-->

*dịch đoạn phía trên*


<!--
It is worth being aware, however, that biased estimators are frequently used in practice.  
There are cases where unbiased estimators do not exist without further assumptions, or are intractable to compute.  
This may seem like a significant flaw in an estimator, however the majority of estimators encountered in practice are at least asymptotically unbiased in the sense that the bias tends to zero as the number of available samples tends to infinity: $\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 3 ================================-->

<!-- =================== Bắt đầu dịch Phần 4 ================================-->

<!--
### Variance and Standard Deviation
-->

### Phương Sai và Độ Lệch Chuẩn

<!--
Second, let's measure the randomness in the estimator.  
Recall from :numref:`sec_random_variables`, the *standard deviation* (or *standard error*) is defined as the squared root of the variance.  
We may measure the degree of fluctuation of an estimator by measuring the standard deviation or variance of that estimator.
-->

Tiếp theo, cùng tính độ ngẫu nhiên trong bộ ước lượng.
Nhắc lại từ :numref:`sec_random_variables`, *độ lệch chuẩn* (còn được gọi là *sai số chuẩn*) được định nghĩa là căn bậc hai của phương sai.
Chúng ta có thể đo được độ dao động của bộ ước lượng bằng cách tính độ lệch chuẩn hoặc phương sai của bộ ước lượng đó.

$$\sigma_{\hat{\theta}_n} = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

<!--
It is important to compare :eqref:`eq_var_est` to :eqref:`eq_mse_est`.  
In this equation we do not compare to the true population value $\theta$, but instead to $E(\hat{\theta}_n)$, the expected sample mean.  
Thus we are not measuring how far the estimator tends to be from the true value, but instead we measuring the fluctuation of the estimator itself.
-->

So sánh :eqref:`eq_var_est` và :eqref:`eq_mse_est` là một việc quan trọng.
Trong công thức này, thay vì so sánh với giá trị tổng thể thực $\theta$, chúng ta sử dụng $E(\hat{\theta}_n)$, giá trị trung bình mẫu kỳ vọng.
Do đó chúng ta không đo độ lệch của bộ ước lượng so với giá trị thực mà là độ dao động của chính nó (bộ ước lượng).


<!--
### The Bias-Variance Trade-off
-->

### Sự đánh đổi Độ Chệch-Phương Sai

<!--
It is intuitively clear that these two components contribute to the mean squared error.  
What is somewhat shocking is that we can show that this is actually a *decomposition* of the mean squared error into two contributions.  
That is to say that we can write the mean squared error as the sum of the variance and the square or the bias.
-->

Cả hai yếu tố trên rõ ràng đều ảnh hưởng đến trung bình bình phương sai số.
Một điều ngạc nhiên là chúng ta có thể chứng minh trung bình bình phương sai số có thể phân tách thành hai thành phần đó. 
Điều này có nghĩa là chúng ta có thể viết trung bình bình phương sai số bằng tổng của phương sai và bình phương độ chệch.

$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - E(\hat{\theta}_n) + E(\hat{\theta}_n) - \theta)^2] \\
 &= E[(\hat{\theta}_n - E(\hat{\theta}_n))^2] + E[(E(\hat{\theta}_n) - \theta)^2] \\
 &= \mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2.\\
\end{aligned}
$$

<!--
We refer the above formula as *bias-variance trade-off*. 
The mean squared error can be divided into precisely two sources of error: the error from high bias and the error from high variance. 
On the one hand, the bias error is commonly seen in a simple model (such as a linear regression model), which cannot extract high dimensional relations between the features and the outputs. 
If a model suffers from high bias error, we often say it is *underfitting* or lack of *generalization* as introduced in (:numref:`sec_model_selection`). 
On the flip side, the other error source---high variance usually results from a too complex model, which overfits the training data. 
As a result, an *overfitting* model is sensitive to small fluctuations in the data. 
If a model suffers from high variance, we often say it is *overfitting* and lack of *flexibility* as introduced in (:numref:`sec_model_selection`).
-->

Chúng tôi gọi công thức trên là *sự đánh đổi độ chệch-phương sai*.
Giá trị trung bình bình phương sai số có thể được phân tách chính xác thành hai nguồn sai số khác nhau: sai số từ độ chệch cao và sai số từ phương sai cao.
Sai số độ chệch thường xuất hiện ở mô hình đơn giản (ví dụ mô hình hồi quy tuyến tính), khi nó không thể chiết xuất những quan hệ đa chiều giữa các đặc trưng và đầu ra.
Nếu một mô hình có độ chệch cao, chúng ta thường nói rằng nó *dưới khớp* (*underfitting*) hoặc là thiếu sự *tổng quát hóa* như đã giới thiệu ở (:numref:`sec_model_selection`).
Ngược lại, một mô hình *quá khớp* (*overfitting*) lại rất nhạy cảm với những dao động nhỏ trong dữ liệu.
Nếu một mô hình có phương sai cao, chúng ta thường nói rằng nó *quá khớp* và thiếu sự *uyển chuyển* như đã giới thiệu ở (:numref:`sec_model_selection`).

<!-- =================== Kết thúc dịch Phần 4 ================================-->

<!-- =================== Bắt đầu dịch Phần 5 ================================-->

<!--
### Evaluating Estimators in Code
-->

### *dịch tiêu đề phía trên*

<!--
Since the standard deviation of an estimator has been implementing in MXNet by simply calling `a.std()` for a `ndarray` "a", we will skip it but implement the statistical bias and the mean squared error in MXNet.
-->

*dịch đoạn phía trên*

```{.python .input}
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)

# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

<!--
To illustrate the equation of the bias-variance trade-off, let's simulate of normal distribution $\mathcal{N}(\theta, \sigma^2)$ with $10,000$ samples. 
Here, we use a $\theta = 1$ and $\sigma = 4$. 
As the estimator is a function of the given samples, here we use the mean of the samples as an estimator for true $\theta$ in this normal distribution $\mathcal{N}(\theta, \sigma^2)$ .
-->

*dịch đoạn phía trên*

```{.python .input}
theta_true = 1
sigma = 4
sample_length = 10000
samples = np.random.normal(theta_true, sigma, sample_length)
theta_est = np.mean(samples)
theta_est
```

<!--
Let's validate the trade-off equation by calculating the summation of the squared bias and the variance of our estimator. First, calculate the MSE of our estimator.
-->

*dịch đoạn phía trên*

```{.python .input}
mse(samples, theta_true)
```

<!--
Next, we calculate $\mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2$ as below. As you can see, the two values agree to numerical precision.
-->

*dịch đoạn phía trên*

```{.python .input}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

<!-- =================== Kết thúc dịch Phần 5 ================================-->

<!-- =================== Bắt đầu dịch Phần 6 ================================-->

<!--
## Conducting Hypothesis Tests
-->

## *dịch tiêu đề phía trên*


<!--
The most commonly encountered topic in statistical inference is hypothesis testing. 
While hypothesis testing was popularized in the early 20th century, the first use can be traced back to John Arbuthnot in the 1700s. 
John tracked 80-year birth records in London and concluded that more men were born than women each year. 
Following that, the modern significance testing is the intelligence heritage by Karl Pearson who invented $p$-value and Pearson's chi-squared test), William Gosset who is the father of Student's t-distribution, and Ronald Fisher who initialed the null hypothesis and the significance test.
-->

*dịch đoạn phía trên*

<!--
A *hypothesis test* is a way of evaluating some evidence against the default statement about a population. 
We refer the default statement as the *null hypothesis* $H_0$, which we try to reject using the observed data. 
Here, we use $H_0$ as a starting point for the statistical significance testing. 
The *alternative hypothesis* $H_A$ (or $H_1$) is a statement that is contrary to the null hypothesis. 
A null hypothesis is often stated in a declarative form which posits a relationship between variables. 
It should reflect the brief as explicit as possible, and be testable by statistics theory.
-->

*dịch đoạn phía trên*

<!--
Imagine you are a chemist. After spending thousands of hours in the lab, you develop a new medicine which can dramatically improve one's ability to understand math. 
To show its magic power, you need to test it. 
Naturally, you may need some volunteers to take the medicine and see whether it can help them learn math better. How do you get started?
-->

*dịch đoạn phía trên*

<!--
First, you will need carefully random selected two groups of volunteers, so that there is no difference between their math understanding ability measured by some metrics. 
The two groups are commonly referred to as the test group and the control group. 
The *test group* (or *treatment group*) is a group of individuals who will experience the medicine, while the *control group* represents the group of users who are set aside as a benchmark, i.e., identical environment setups except taking this medicine. 
In this way, the influence of all the variables are minimized, except the impact of the independent variable in the treatment.
-->

*dịch đoạn phía trên*

<!--
Second, after a period of taking the medicine, you will need to measure the two groups' math understanding by the same metrics, such as letting the volunteers do the same tests after learning a new math formula.
Then, you can collect their performance and compare the results.  
In this case, our null hypothesis will be that there is no difference between the two groups, and our alternate will be that there is.
-->

*dịch đoạn phía trên*

<!--
This is still not fully formal.  
There are many details you have to think of carefully. 
For example, what is the suitable metrics to test their math understanding ability? 
How many volunteers for your test so you can be confident to claim the effectiveness of your medicine? 
How long should you run the test? How do you decided if there is a difference between the two groups?  
Do you care about the average performance only, or do you also the range of variation of the scores. And so on.
-->

*dịch đoạn phía trên*

<!--
In this way, hypothesis testing provides framework for experimental design and reasoning about certainty in observed results.  
If we can now show that the null hypothesis is very unlikely to be true, we may reject it with confidence.
-->

*dịch đoạn phía trên*

<!--
To complete the story of how to work with hypothesis testing, we need to now introduce some additional terminology and make some of our concepts above formal.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 6 ================================-->

<!-- =================== Bắt đầu dịch Phần 7 ================================-->

<!--
### Statistical Significance
-->

### Ý nghĩa thống kê

<!--
The *statistical significance* measures the probability of erroneously reject the null hypothesis, $H_0$, when it should not be rejected, i.e.,
-->

*Ý nghĩa thống kê* đo xác suất lỗi khi loại bỏ giả thuyết gốc, $H_0$, trong khi đúng ra không nên loại bỏ nó. 

$$ \text{ý nghĩa thống kê }= 1 - \alpha = P(\text{loại bỏ} H_0 \mid H_0 \text{là đúng} ).$$

<!--
It is also referred to as the *type I error* or *false positive*. 
The $\alpha$, is called as the *significance level* and its commonly used value is $5\%$, i.e., $1-\alpha = 95\%$. 
The level of statistical significance level can be explained as the level of risk that we are willing to take, when we reject a true null hypothesis.
-->

Đây còn được gọi là *lỗi loại I* hay *dương tính giả*.
$\alpha$ ở đây là *mức ý nghĩa* và thường được chọn ở giá trị $5\%$, tức là $1-\alpha = 95\%$.
Mức ý nghĩa thống kê còn có thể hiểu như mức độ rủi ro mà chúng ta chấp nhận khi loại bỏ nhầm một giả thuyết gốc chính xác.

<!--
:numref:`fig_statistical_significance` shows the the observations' values and probability of a given normal distribution in a two-sample hypothesis test. 
If the observation data point is located outsides the $95\%$ threshold, it will be a very unlikely observation under the null hypothesis assumption. 
Hence, there might be something wrong with the null hypothesis and we will reject it.
-->

:numref:`fig_statistical_significance` thể hiện các giá trị quan sát và xác suất của một phân phối chuẩn trong một bài kiểm định thống kê hai mẫu.
Nếu các điểm dữ liệu quan sát nằm ngoài ngưỡng $95\%$, đó sẽ là một quan sát rất khó xảy ra dưới giả định của giả thuyết gốc.
Do đó, giả thuyết gốc có điều gì đó không đúng và chúng ta sẽ loại bỏ nó.

<!--
![Statistical significance.](../img/statistical_significance.svg)
-->

![Ý nghĩa thống kê](../img/statistical_significance.svg)
:label:`fig_statistical_significance`

<!-- =================== Kết thúc dịch Phần 7 ================================-->

<!-- =================== Bắt đầu dịch Phần 8 ================================-->

<!--
### Statistical Power
-->

### *dịch tiêu đề phía trên*

<!--
The *statistical power* (or *sensitivity*) measures the probability of reject the null hypothesis, $H_0$, when it should be rejected, i.e.,
-->

*dịch đoạn phía trên*

$$ \text{statistical power }= P(\text{reject } H_0  \mid H_0 \text{ is false} ).$$

<!--
Recall that a *type I error* is error caused by rejecting the null hypothesis when it is true, whereas a *type II error* is resulted from failing to reject the null hypothesis when it is false. 
A type II error is usually denoted as $\beta$, and hence the corresponding statistical power is $1-\beta$.
-->

*dịch đoạn phía trên*


<!--
Intuitively, statistical power can be interpreted as how likely our test will detect a real discrepancy of some minimum magnitude at a desired statistical significance level. 
$80\%$ is a commonly used statistical power threshold. The higher the statistical power, the more likely we are to detect true differences.
-->

*dịch đoạn phía trên*

<!--
One of the most common uses of statistical power is in determining the number of samples needed.  
The probability you reject the null hypothesis when it is false depends on the degree to which it is false (known as the *effect size*) and the number of samples you have.  
As you might expect, small effect sizes will require a very large number of samples to be detectable with high probability.  
While beyond the scope of this brief appendix to derive in detail, as an example, want to be able to reject a null hypothesis that our sample came from a mean zero variance one Gaussian, and we believe that our sample's mean is actually close to one, we can do so with acceptable error rates with a sample size of only $8$.  
However, if we think our sample population true mean is close to $0.01$, then we'd need a sample size of nearly $80000$ to detect the difference.
-->

*dịch đoạn phía trên*

<!--
We can imagine the power as a water filter. In this analogy, a high power hypothesis test is like a high quality water filtration system that will reduce harmful substances in the water as much as possible. 
On the other hand, a smaller discrepancy is like a low quality water filter, where some relative small substances may easily escape from the gaps. 
Similarly, if the statistical power is not of enough high power, then the test may not catch the smaller discrepancy.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 8 ================================-->

<!-- =================== Bắt đầu dịch Phần 9 ================================-->

<!--
### Test Statistic
-->

### *dịch tiêu đề phía trên*

<!--
A *test statistic* $T(x)$ is a scalar which summarizes some characteristic of the sample data.  
The goal of defining such a statistic is that it should allow us to distinguish between different distributions and conduct our hypothesis test.  
hinking back to our chemist example, if we wish to show that one population performs better than the other, it could be reasonable to take the mean as the test statistic.  
Different choices of test statistic can lead to statistical test with drastically different statistical power.
-->

*dịch đoạn phía trên*

<!--
Often, $T(X)$ (the distribution of the test statistic under our null hypothesis) will follow, at least approximately, a common probability distribution such as a normal distribution when considered under the null hypothesis. 
If we can derive explicitly such a distribution, and then measure our test statistic on our dataset, we can safely reject the null hypothesis if our statistic is far outside the range that we would expect.  Making this quantitative leads us to the notion of $p$-values.
-->

*dịch đoạn phía trên*


<!--
### $p$-value
-->

### *dịch tiêu đề phía trên*

<!--
The *$p$-value* (or the *probability value*) is the probability that $T(X)$ is at least as extreme as the observed test statistic $T(x)$ assuming that the null hypothesis is *true*, i.e.,
-->

*dịch đoạn phía trên*

$$ p\text{-value} = P_{H_0}(T(X) \geq T(x)).$$

<!--
If the $p$-value is smaller than or equal to a pre-defined and fixed statistical significance level $\alpha$, we may reject the null hypothesis. 
Otherwise, we will conclude that we are lack of evidence to reject the null hypothesis. 
For a given population distribution, the *region of rejection* will be the interval contained of all the points which has a $p$-value smaller than the statistical significance level $\alpha$.
-->

*dịch đoạn phía trên*


<!--
### One-side Test and Two-sided Test
-->

### *dịch tiêu đề phía trên*

<!--
Normally there are two kinds of significance test: the one-sided test and the two-sided test. 
The *one-sided test* (or *one-tailed test*) is applicable when the null hypothesis and the alternative hypothesis only have one direction. 
For example, the null hypothesis may state that the true parameter $\theta$ is less than or equal to a value $c$. 
The alternative hypothesis would be that $\theta$ is greater than $c$. 
That is, the region of rejection is on only one side of the sampling distribution.  
Contrary to the one-sided test, the *two-sided test* (or *two-tailed test*) is applicable when the region of rejection is on both sides of the sampling distribution. 
An example in this case may have a null hypothesis state that the true parameter $\theta$ is equal to a value $c$. 
The alternative hypothesis would be that $\theta$ is not equal to $c$.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 9 ================================-->

<!-- =================== Bắt đầu dịch Phần 10 ================================-->

<!--
### General Steps of Hypothesis Testing
-->

### *dịch tiêu đề phía trên*

<!--
After getting familiar with the above concepts, let's go through the general steps of hypothesis testing.
-->

*dịch đoạn phía trên*

<!--
1. State the question and establish a null hypotheses $H_0$.
2. Set the statistical significance level $\alpha$ and a statistical power ($1 - \beta$).
3. Obtain samples through experiments.  The number of samples needed will depend on the statistical power, and the expected effect size.
4. Calculate the test statistic and the $p$-value.
5. Make the decision to keep or reject the null hypothesis based on the $p$-value and the statistical significance level $\alpha$.
-->

*dịch đoạn phía trên*

<!--
To conduct a hypothesis test, we start by defining a null hypothesis and a level of risk that we are willing to take. 
Then we calculate the test statistic of the sample, taking an extreme value of the test statistic as evidence against the null hypothesis. 
If the test statistic falls within the reject region, we may reject the null hypothesis in favor of the alternative.
-->

*dịch đoạn phía trên*

<!--
Hypothesis testing is applicable in a variety of scenarios such as the clinical trails and A/B testing.
-->

*dịch đoạn phía trên*


<!--
## Constructing Confidence Intervals
-->

## *dịch tiêu đề phía trên*


<!--
When estimating the value of a parameter $\theta$, point estimators like $\hat \theta$ are of limited utility since they contain no notion of uncertainty. 
Rather, it would be far better if we could produce an interval that would contain the true parameter $\theta$ with high probability.  
If you were interested in such ideas a century ago, then you would have been excited to read "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability" by Jerzy Neyman :cite:`Neyman.1937`, who first introduced the concept of confidence interval in 1937.
-->

*dịch đoạn phía trên*

<!--
To be useful, a confidence interval should be as small as possible for a given degree of certainty. Let's see how to derive it.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 10 ================================-->

<!-- =================== Bắt đầu dịch Phần 11 ================================-->

<!--
### Definition
-->

### *dịch tiêu đề phía trên*

<!--
Mathematically, a *confidence interval* for the true parameter $\theta$ is an interval $C_n$ that computed from the sample data such that
-->

*dịch đoạn phía trên*

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

<!--
Here $\alpha \in (0, 1)$, and $1 - \alpha$ is called the *confidence level* or *coverage* of the interval. 
This is the same $\alpha$ as the significance level as we discussed about above.
-->

*dịch đoạn phía trên*

<!--
Note that :eqref:`eq_confidence` is about variable $C_n$, not about the fixed $\theta$. 
To emphasize this, we write $P_{\theta} (C_n \ni \theta)$ rather than $P_{\theta} (\theta \in C_n)$.
-->

*dịch đoạn phía trên*

<!--
### Interpretation
-->

### *dịch tiêu đề phía trên*

<!--
It is very tempting to interpret a $95\%$ confidence interval as an interval where you can be $95\%$ sure the true parameter lies, however this is sadly not true. 
 The true parameter is fixed, and it is the interval that is random.  
 Thus a better interpretation would be to say that if you generated a large number of confidence intervals by this procedure, $95\%$ of the generated intervals would contain the true parameter.
-->

*dịch đoạn phía trên*

<!--
This may seem pedantic, but it can have real implications for the interpretation of the results.  
In particular, we may satisfy :eqref:`eq_confidence` by constructing intervals that we are *almost certain* do not contain the true value, as long as we only do so rarely enough.  
We close this section by providing three tempting but false statements.  
An in-depth discussion of these points can be found in :cite:`Morey.Hoekstra.Rouder.ea.2016`.
-->

*dịch đoạn phía trên*

<!--
* **Fallacy 1**. Narrow confidence intervals mean we can estimate the parameter precisely.
* **Fallacy 2**. The values inside the confidence interval are more likely to be the true value than those outside the interval.
* **Fallacy 3**. The probability) that a particular observed $95\%$ confidence interval contains the true value is $95\%$.
-->

*dịch đoạn phía trên*

<!--
Sufficed to say, confidence intervals are subtle objects.  H
owever, if you keep the interpretation clear, they can be powerful tools.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 11 ================================-->

<!-- =================== Bắt đầu dịch Phần 12 ================================-->

<!--
### A Gaussian Example
-->

### *dịch tiêu đề phía trên*

<!--
Let's discuss the most classical example, the confidence interval for the mean of a Gaussian of unknown mean and variance.  
Suppose we collect $n$ samples $\{x_i\}_{i=1}^n$ from our Gaussian $\mathcal{N}(\mu, \sigma^2)$.  
We can compute estimators for the mean and standard deviation by taking
-->

*dịch đoạn phía trên*

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\text{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

<!--
If we now consider the random variable
-->

*dịch đoạn phía trên*

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

<!--
we obtain a random variable following a well-known distribution called the *Student's t-distribution on* $n-1$ *degrees of freedom*.
-->

*dịch đoạn phía trên*

<!--
This distribution is very well studied, and it is known, for instance, that as $n\rightarrow \infty$, it is approximately a standard Gaussian, and thus by looking up values of the Gaussian c.d.f. in a table, we may conclude that the value of $T$ is in the interval $[-1.96, 1.96]$ at least $95\%$ of the time.  
For finite values of $n$, the interval needs to be somewhat larger, but are well known and precomputed in tables.
-->

*dịch đoạn phía trên*

<!--
Thus, we may conclude that for large $n$,
-->

*dịch đoạn phía trên*

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

<!--
Rearranging this by multiplying both sides by $\hat\sigma_n/\sqrt{n}$ and then adding $\hat\mu_n$, we obtain
-->

*dịch đoạn phía trên*

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

<!--
Thus we know that we have found our $95\%$ confidence interval:
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

It is safe to say that :eqref:`eq_gauss_confidence` is one of the most used formula in statistics. 
Let's close our discussion of statistics by implementing it. 
For simplicity, we assume we are in the asymptotic regime. 
Small values of $N$ should include the correct value of `t_star` obtained either programmatically or from a $t$-table.
-->

*dịch đoạn phía trên*

```{.python .input}
# Number of samples
N = 1000

# Sample dataset
samples = np.random.normal(loc=0, scale=1, size=(N,))

# Lookup Students's t-distribution c.d.f.
t_star = 1.96

# Construct interval
mu_hat = np.mean(samples)
sigma_hat = samples.std(ddof=1)
(mu_hat - t_star*sigma_hat/np.sqrt(N), mu_hat + t_star*sigma_hat/np.sqrt(N))
```

<!-- =================== Kết thúc dịch Phần 12 ================================-->

<!-- =================== Bắt đầu dịch Phần 13 ================================-->

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* Statistics focuses on inference problems, whereas deep learning emphasizes on making accurate predictions without explicitly programming and understanding.
* There are three common statistics inference methods: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.
* There are three most common estimators: statistical bias, standard deviation, and mean square error.
* A confidence interval is an estimated range of a true population parameter that we can construct by given the samples.
* Hypothesis testing is a way of evaluating some evidence against the default statement about a population.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Let $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} \mathrm{Unif}(0, \theta)$, where "iid" stands for *independent and identically distributed*. Consider the following estimators of $\theta$:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * Find the statistical bias, standard deviation, and mean square error of $\hat{\theta}.$
    * Find the statistical bias, standard deviation, and mean square error of $\tilde{\theta}.$
    * Which estimator is better?
1. For our chemist example in introduction, can you derive the 5 steps to conduct a two-sided hypothesis testing? Given the statistical significance level $\alpha = 0.05$ and the statistical power $1 - \beta = 0.8$.
1. Run the confidence interval code with $N=2$ and $\alpha = 0.5$ for $100$ independently generated dataset, and plot the resulting intervals (in this case `t_star = 1.0`).  You will see several very short intervals which are very far from containing the true mean $0$.  Does this contradict the interpretation of the confidence interval?  Do you feel comfortable using short intervals to indicate high precision estimates?
-->

*dịch đoạn phía trên*

<!--
## [Discussions](https://discuss.mxnet.io/t/5156)
-->

## *dịch tiêu đề phía trên*

<!--
![](../img/qr_statistics.svg)
-->

![](../img/qr_statistics.svg)

<!-- ===================== Kết thúc dịch Phần 13 ==================== -->

### Những người thực hiện
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
* Phạm Hồng Vinh
* Vũ Hữu Tiệp

<!-- Phần 3 -->
*

<!-- Phần 4 -->
* Nguyễn Lê Quang Nhật
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc
* Vũ Hữu Tiệp

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*

<!-- Phần 7 -->
* Lê Khắc Hồng Phúc
* Vũ Hữu Tiệp
* Phạm Hồng Vinh
* Phạm Minh Đức

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

<!-- Phần 13 -->
*

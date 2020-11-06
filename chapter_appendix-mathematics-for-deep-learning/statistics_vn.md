<!--
# Statistics
-->

# Thống kê
:label:`sec_statistics`

<!--
Undoubtedly, to be a top deep learning practitioner, the ability to train the state-of-the-art and high accurate models is crucial.  
However, it is often unclear when improvements are significant, or only the result of random fluctuations in the training process.  
To be able to discuss uncertainty in estimated values, we must learn some statistics.
-->

Để trở thành chuyên gia Học sâu hàng đầu, điều kiện tiên quyết cần có là khả năng huấn luyện các mô hình hiện đại với độ chính xác cao.
Tuy nhiên, thường khó có thể biết được những cải tiến trong mô hình là đáng kể, hay chúng chỉ là kết quả của những biến động ngẫu nhiên trong quá trình huấn luyện.
Để có thể thảo luận về tính bất định trong các giá trị ước lượng, chúng ta cần có hiểu biết về thống kê.


<!--
The earliest reference of *statistics* can be traced back to an Arab scholar Al-Kindi in the $9^{\mathrm{th}}$-century, 
who gave a detailed description of how to use statistics and frequency analysis to decipher encrypted messages. 
After 800 years, the modern statistics arose from Germany in 1700s, when the researchers focused on the demographic and economic data collection and analysis. 
Today, statistics is the science subject that concerns the collection, processing, analysis, interpretation and visualization of data. 
What is more, the core theory of statistics has been widely used in the research within academia, industry, and government.
-->

Tài liệu tham khảo đầu tiên về *thống kê* có thể được truy ngược về học giả người Ả Rập Al-Kindi từ thế kỉ thứ chín. 
Ông đã đưa ra những mô tả chi tiết về cách sử dụng thống kê và phân tích tần suất để giải mã những thông điệp mã hóa. 
Sau 800 năm, thống kê hiện đại trỗi dậy ở Đức vào những năm 1700, khi các nhà nghiên cứu tập trung vào việc thu thập và phân tích các dữ liệu nhân khẩu học và kinh tế. 
Hiện nay, khoa học thống kê quan tâm đến việc thu thập, xử lý, phân tích, diễn giải và biểu diễn dữ liệu. 
Hơn nữa, lý thuyết cốt lõi của thống kê đã được sử dụng rộng rãi cho nghiên cứu trong giới học thuật, doanh nghiệp và chính phủ. 


<!--
More specifically, statistics can be divided to *descriptive statistics* and *statistical inference*. 
The former focus on summarizing and illustrating the features of a collection of observed data, which is referred to as a *sample*. 
The sample is drawn from a *population*, denotes the total set of similar individuals, items, or events of our experiment interests. 
Contrary to descriptive statistics, *statistical inference* further deduces the characteristics of a population from the given *samples*, 
based on the assumptions that the sample distribution can replicate the population distribution at some degree.
-->

Cụ thể hơn, thống kê có thể được chia thành *thống kê mô tả* (*descriptive statistic*) và *suy luận thống kê* (*statistical inference*).
Thống kê mô tả đặt trọng tâm vào việc tóm tắt và minh họa những đặc trưng của một tập hợp những dữ liệu đã được quan sát - được gọi là *mẫu*.
Mẫu được lấy ra từ một *tổng thể* (*population*), là biểu diễn của toàn bộ những cá thể, đồ vật hay sự kiện tương tự nhau mà thí nghiệm của ta quan tâm.
Trái với thống kê mô tả, *suy luận thống kê* (*statistical inference*) dự đoán những đặc điểm của một tổng thể qua những *mẫu* có sẵn, 
dựa theo giả định phân phối mẫu là một biểu diễn tương đối hợp lý của phân phối tổng thể.


<!--
You may wonder: “What is the essential difference between machine learning and statistics?” Fundamentally speaking, statistics focuses on the inference problem. 
This type of problems includes modeling the relationship between the variables, such as causal inference, and testing the statistically significance of model parameters, such as A/B testing. 
In contrast, machine learning emphasizes on making accurate predictions, without explicitly programming and understanding each parameter's functionality.
-->

Bạn có thể tự hỏi: "Sự khác biệt cơ bản giữa học máy và thống kê là gì?".
Về căn bản, thống kê tập trung vào các vấn đề suy luận.
Những vấn đề này bao gồm mô hình hóa mối quan hệ giữa các biến, ví dụ như suy luận nguyên nhân hoặc kiểm tra ý nghĩa thống kê của các tham số mô hình, ví dụ như phép thử A/B.
Ngược lại, học máy đề cao việc dự đoán chính xác mà không yêu cầu lập trình một cách tường minh và hiểu rõ chức năng của từng tham số.


<!--
In this section, we will introduce three types of statistics inference methods: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals. 
These methods can help us infer the characteristics of a given population, i.e., the true parameter $\theta$. 
For brevity, we assume that the true parameter $\theta$ of a given population is a scalar value. 
It is straightforward to extend to the case where $\theta$ is a vector or a tensor, thus we omit it in our discussion.
-->

Trong chương này, chúng tôi sẽ giới thiệu ba loại suy luận thống kê: đánh giá và so sánh các bộ ước lượng, tiến hành kiểm định giả thuyết và xây dựng khoảng tin cậy. 
Các phương pháp này có thể giúp chúng ta suy luận những đặc tính của một tổng thể, hay nói cách khác, tham số thực $\theta$. 
Nói ngắn gọn, chúng tôi giả sử tham số thực $\theta$ của một tổng thể cho trước là một số vô hướng. 
Việc mở rộng ra các trường hợp $\theta$ là một vector hoặc tensor là khá đơn giản nên chúng tôi sẽ không đề cập ở đây. 


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
If you have a number of samples from a Bernoulli random variable, then the maximum likelihood estimate for 
the probability the random variable is one can be obtained by counting the number of ones observed and dividing by the total number of samples.  
Similarly, an exercise asked you to show that the maximum likelihood estimate of the mean of a Gaussian given a number of samples is given by the average value of all the samples.  
These estimators will almost never give the true value of the parameter, but ideally for a large number of samples the estimate will be close.
-->

Ta đã thấy nhiều ví dụ đơn giản của bộ ước lượng trong phần :numref:`sec_maximum_likelihood`. 
Nếu bạn có một số mẫu ngẫu nhiên từ phân phối Bernoulli, thì ước lượng hợp lý cực đại (*maximum likelihood estimate*) 
cho xác xuất của biến ngẫu nhiên có thể được tính bằng cách đếm số lần biến cố xuất hiện rồi chia cho tổng số mẫu.
Tương tự, đã có một bài tập yêu cầu bạn chứng minh rằng ước lượng hợp lý cực đại của kỳ vọng phân phối Gauss với một số lượng mẫu cho trước là giá trị trung bình của tập mẫu đó. 
Các bộ ước lượng này dường như sẽ không bao giờ cho ra giá trị chính xác của tham số, nhưng với số lượng mẫu đủ lớn, ước lượng có được sẽ gần với giá trị thực. 


<!--
As an example, we show below the true density of a Gaussian random variable with mean zero and variance one, along with a collection samples from that Gaussian.  
We constructed the $y$ coordinate so every point is visible and the relationship to the original density is clearer.
-->

Xét ví dụ sau, chúng tôi biểu diễn mật độ của phân phối Gauss với kỳ vọng là không và phương sai là một, cùng với một tập các mẫu lấy ra từ phân phối đó. 
Tọa độ $y$ được xây dựng sao cho tất các điểm đều có thể nhìn thấy được và mối quan hệ giữa mật độ mẫu và mật độ gốc của phân phối có thể được nhìn thấy rõ hơn.  


```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
import random
npx.set_np()
# Sample datapoints and create y coordinate
epsilon = 0.1
random.seed(8675309)
xs = np.random.normal(loc=0, scale=1, size=(300,))
ys = [np.sum(np.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))
             / np.sqrt(2*np.pi*epsilon**2)) / len(xs) for i in range(len(xs))]
# Compute true density
xd = np.arange(np.min(xs), np.max(xs), 0.01)
yd = np.exp(-xd**2/2) / np.sqrt(2 * np.pi)
# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=np.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(np.mean(xs)):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
torch.pi = torch.acos(torch.zeros(1)) * 2  #define pi in torch
# Sample datapoints and create y coordinate
epsilon = 0.1
torch.manual_seed(8675309)
xs = torch.randn(size=(300,))
ys = torch.tensor(
    [torch.sum(torch.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2))\
               / torch.sqrt(2*torch.pi*epsilon**2)) / len(xs)\
     for i in range(len(xs))])
# Compute true density
xd = torch.arange(torch.min(xs), torch.max(xs), 0.01)
yd = torch.exp(-xd**2/2) / torch.sqrt(2 * torch.pi)
# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=torch.mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(torch.mean(xs).item()):.2f}')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
tf.pi = tf.acos(tf.zeros(1)) * 2  # define pi in TensorFlow
# Sample datapoints and create y coordinate
epsilon = 0.1
xs = tf.random.normal((300,))
ys = tf.constant(
    [(tf.reduce_sum(tf.exp(-(xs[:i] - xs[i])**2 / (2 * epsilon**2)) \
               / tf.sqrt(2*tf.pi*epsilon**2)) / tf.cast(
        tf.size(xs), dtype=tf.float32)).numpy() \
     for i in range(tf.size(xs))])
# Compute true density
xd = tf.range(tf.reduce_min(xs), tf.reduce_max(xs), 0.01)
yd = tf.exp(-xd**2/2) / tf.sqrt(2 * tf.pi)
# Plot the results
d2l.plot(xd, yd, 'x', 'density')
d2l.plt.scatter(xs, ys)
d2l.plt.axvline(x=0)
d2l.plt.axvline(x=tf.reduce_mean(xs), linestyle='--', color='purple')
d2l.plt.title(f'sample mean: {float(tf.reduce_mean(xs).numpy()):.2f}')
d2l.plt.show()
```


<!--
There can be many ways to compute an estimator of a parameter $\hat{\theta}_n$.  
In this section, we introduce three common methods to evaluate and compare estimators: the mean squared error, the standard deviation, and statistical bias.
-->

Có nhiều cách để tính toán một bộ ước lượng cho một tham số $\hat{\theta}_n$. 
Trong phần này, ta sẽ điểm qua ba phương thức phổ biến để đánh giá và so sánh các bộ ước lượng: trung bình bình phương sai số, độ lệch chuẩn và độ chệch thống kê. 


<!--
### Mean Squared Error
-->

### Trung bình Bình phương Sai số


<!--
Perhaps the simplest metric used to evaluate estimators is the *mean squared error (MSE)* (or *$l_2$ loss*) of an estimator can be defined as
-->

Có lẽ phép đo đơn giản nhất được sử dụng để đánh giá bộ ước lượng là *trung bình bình phương sai số (mean squared error -- MSE)* (hay *mất mát $l_2$*). 
Trung bình bình phương sai số của một bộ ước lượng được định nghĩa 


$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`


<!--
This allows us to quantify the average squared deviation from the true value.  
MSE is always non-negative. If you have read :numref:`sec_linear_regression`, you will recognize it as the most commonly used regression loss function. 
As a measure to evaluate an estimator, the closer its value to zero, the closer the estimator is close to the true parameter $\theta$.
-->

Phương pháp này cho phép ta định lượng trung bình bình phương độ lệch so với giá trị thực. 
MSE là một đại lượng không âm. 
Nếu đã đọc :numref:`sec_linear_regression`, bạn sẽ nhận ra đây là hàm mất mát được sử dụng phổ biến nhất trong bài toán hồi quy. 
Như một phép đo để đánh giá bộ ước lượng, giá trị của nó càng gần không thì bộ ước lượng càng gần với tham số thực $\theta$. 


<!--
### Statistical Bias
-->

### Độ chệch Thống kê

<!--
The MSE provides a natural metric, but we can easily imagine multiple different phenomena that might make it large.  
Two that we will see are fundamentally important are the fluctuation in the estimator due to randomness in the dataset, and systematic error in the estimator due to the estimation procedure.
-->

MSE cung cấp một phép đo tự nhiên, nhưng ta có thể dễ dàng nghĩ tới các trường hợp khác nhau mà ở đó giá trị MSE sẽ lớn.
Ta sẽ bàn tới hai trường hợp cơ bản đó là biến động của bộ ước lượng do sự ngẫu nhiên trong bộ dữ liệu, và sai số hệ thống của bộ ước lượng xảy ra trong quá trình ước lượng.


<!--
First, let's measure the systematic error. 
For an estimator $\hat{\theta}_n$, the mathematical illustration of *statistical bias* can be defined as
-->

Đầu tiên, ta hãy đo sai số hệ thống.
Với một bộ ước lượng $\hat{\theta}_n$, biểu diễn toán học của *độ chệch thống kê* được định nghĩa


$$\mathrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`


<!--
Note that when $\mathrm{bias}(\hat{\theta}_n) = 0$, the expectation of the estimator $\hat{\theta}_n$ is equal to the true value of parameter.  
In this case, we say $\hat{\theta}_n$ is an unbiased estimator.  
In general, an unbiased estimator is better than a biased estimator since its expectation is the same as the true parameter.
-->

Lưu ý rằng khi $\mathrm{bias}(\hat{\theta}_n) = 0$, kỳ vọng của bộ ước lượng $\hat{\theta}_n$ sẽ bằng với giá trị thực của tham số.
Trường hợp này, ta nói $\hat{\theta}_n$ là một bộ ước lượng không thiên lệch.
Nhìn chung, một bộ ước lượng không thiên lệch sẽ tốt hơn một bộ ước lượng thiên lệch vì kỳ vọng của nó sẽ bằng với tham số thực.


<!--
It is worth being aware, however, that biased estimators are frequently used in practice.  
There are cases where unbiased estimators do not exist without further assumptions, or are intractable to compute.  
This may seem like a significant flaw in an estimator, however the majority of estimators encountered in practice are at least asymptotically unbiased 
in the sense that the bias tends to zero as the number of available samples tends to infinity: $\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$.
-->

Tuy nhiên, những bộ ước lượng thiên lệch vẫn thường xuyên được sử dụng trong thực tế.  
Có những trường hợp không tồn tại các bộ ước lượng không thiên lệch nếu không có thêm giả định, hoặc rất khó để tính toán. 
Đây có thể xem như một khuyết điểm lớn trong bộ ước lượng, tuy nhiên phần lớn các bộ ước lượng gặp trong thực tiễn đều ít nhất tiệm cận 
không thiên lệch theo nghĩa độ chệch có xu hướng tiến về không khi số lượng mẫu có được tiến về vô cực: $\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$. 


<!--
### Variance and Standard Deviation
-->

### Phương sai và Độ lệch Chuẩn

<!--
Second, let's measure the randomness in the estimator.  
Recall from :numref:`sec_random_variables`, the *standard deviation* (or *standard error*) is defined as the squared root of the variance.  
We may measure the degree of fluctuation of an estimator by measuring the standard deviation or variance of that estimator.
-->

Tiếp theo, hãy cùng tính độ ngẫu nhiên trong bộ ước lượng.
Nhắc lại từ :numref:`sec_random_variables`, *độ lệch chuẩn* (*standard deviation*) (còn được gọi là *sai số chuẩn* -- *standard error*) được định nghĩa là căn bậc hai của phương sai.
Chúng ta có thể đo được độ dao động của bộ ước lượng bằng cách tính độ lệch chuẩn hoặc phương sai của bộ ước lượng đó.


$$\sigma_{\hat{\theta}_n} = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`


<!--
It is important to compare :eqref:`eq_var_est` to :eqref:`eq_mse_est`.  
In this equation we do not compare to the true population value $\theta$, but instead to $E(\hat{\theta}_n)$, the expected sample mean.  
Thus we are not measuring how far the estimator tends to be from the true value, but instead we measuring the fluctuation of the estimator itself.
-->

So sánh :eqref:`eq_var_est` và :eqref:`eq_mse_est` là một việc quan trọng.
Trong công thức này, thay vì so sánh với giá trị thực $\theta$ của tổng thể, chúng ta sử dụng $E(\hat{\theta}_n)$ là giá trị trung bình mẫu kỳ vọng.
Do đó chúng ta không đo độ lệch của bộ ước lượng so với giá trị thực mà là độ dao động của chính bộ ước lượng.


<!--
### The Bias-Variance Trade-off
-->

### Sự đánh đổi Độ chệch–Phương sai

<!--
It is intuitively clear that these two components contribute to the mean squared error.  
What is somewhat shocking is that we can show that this is actually a *decomposition* of the mean squared error into two contributions.  
That is to say that we can write the mean squared error as the sum of the variance and the square or the bias.
-->

Cả hai yếu tố trên rõ ràng đều ảnh hưởng đến trung bình bình phương sai số.
Một điều ngạc nhiên là chúng ta có thể chứng minh hai thành phần trên là *phân tách* của trung bình bình phương sai số.
Điều này có nghĩa là ta có thể viết trung bình bình phương sai số bằng tổng của phương sai và bình phương độ chệch.


$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - \theta)^2] \\
 &= E[(\hat{\theta}_n)^2] + E[\theta^2] - 2E[\hat{\theta}_n\theta] \\
 &= \mathrm{Var} [\hat{\theta}_n] + E[\hat{\theta}_n]^2 + \mathrm{Var} [\theta] + E[\theta]^2 - 2E[\hat{\theta}_n]E[\theta] \\
 &= (E[\hat{\theta}_n] - E[\theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (E[\hat{\theta}_n - \theta])^2 + \mathrm{Var} [\hat{\theta}_n] + \mathrm{Var} [\theta] \\
 &= (\mathrm{bias} [\hat{\theta}_n])^2 + \mathrm{Var} (\hat{\theta}_n) + \mathrm{Var} [\theta].\\
\end{aligned}
$$


<!--
We refer the above formula as *bias-variance trade-off*. 
The mean squared error can be divided into three sources of error: the error from high bias, the error from high variance and the irreducible error.
On the one hand, the bias error is commonly seen in a simple model (such as a linear regression model), which cannot extract high dimensional relations between the features and the outputs. 
If a model suffers from high bias error, we often say it is *underfitting* or lack of *flexibilty* as introduced in (:numref:`sec_model_selection`). 
On the flip side, the other error source---high variance usually results from a too complex model, which overfits the training data. 
As a result, an *overfitting* model is sensitive to small fluctuations in the data. 
If a model suffers from high variance, we often say it is *overfitting* and lack of *generalization* as introduced in (:numref:`sec_model_selection`).
The irreducible error is the result from noise in the $\theta$ itself.
-->

Chúng tôi gọi công thức trên là *sự đánh đổi độ chệch-phương sai*. 
Giá trị trung bình bình phương sai số có thể được phân tách chính xác thành ba nguồn sai số khác nhau: 
sai số từ độ chệch cao, sai số từ phương sai cao và sai số không tránh được (*irreducible error*). 
Sai số độ chệch thường xuất hiện ở các mô hình đơn giản (ví dụ như hồi quy tuyến tính), vì chúng không thể trích xuất những quan hệ đa chiều giữa các đặc trưng và đầu ra. 
Nếu một mô hình có độ chệch cao, chúng ta thường nói rằng nó *dưới khớp* (*underfitting*) hoặc là thiếu sự *uyển chuyển* như đã giới thiệu ở (:numref:`sec_model_selection`). 
Ngược lại, một mô hình *quá khớp* (*overfitting*) lại rất nhạy cảm với những dao động nhỏ trong dữ liệu. 
Nếu một mô hình có phương sai cao, chúng ta thường nói rằng nó *quá khớp* và thiếu *tổng quát hóa* như đã giới thiệu ở (:numref:`sec_model_selection`). 
Sai số không tránh được xuất phát từ nhiễu trong chính bản thân $\theta$.
<!--
### Evaluating Estimators in Code
-->

### Đánh giá các Bộ ước lượng qua Lập trình

<!--
Since the standard deviation of an estimator has been implementing in MXNet by simply calling `a.std()` for a `ndarray` "a", 
we will skip it but implement the statistical bias and the mean squared error in MXNet.
-->

Vì độ lệch chuẩn của bộ ước lượng đã được triển khai trong MXNet bằng cách gọi `a.std()` của đối tượng `ndarray` "a",
chúng ta sẽ bỏ qua bước này và thực hiện tính độ chệch thống kê và trung bình bình phương sai số trong MXNet.


```{.python .input}
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(np.mean(est_theta) - true_theta)
# Mean squared error
def mse(data, true_theta):
    return(np.mean(np.square(data - true_theta)))
```

```{.python .input}
#@tab pytorch
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(torch.mean(est_theta) - true_theta)
# Mean squared error
def mse(data, true_theta):
    return(torch.mean(torch.square(data - true_theta)))
```

```{.python .input}
#@tab tensorflow
# Statistical bias
def stat_bias(true_theta, est_theta):
    return(tf.reduce_mean(est_theta) - true_theta)
# Mean squared error
def mse(data, true_theta):
    return(tf.reduce_mean(tf.square(data - true_theta)))
```


<!--
To illustrate the equation of the bias-variance trade-off, let's simulate of normal distribution $\mathcal{N}(\theta, \sigma^2)$ with $10,000$ samples. 
Here, we use a $\theta = 1$ and $\sigma = 4$. 
As the estimator is a function of the given samples, here we use the mean of the samples as an estimator for true $\theta$ in this normal distribution $\mathcal{N}(\theta, \sigma^2)$ .
-->

Để minh họa cho phương trình sự đánh đổi độ chệch-phương sai, cùng giả lập một phân phối chuẩn $\mathcal{N}(\theta, \sigma^2)$ với $10,000$ mẫu.
Ở đây, ta sử dụng $\theta = 1$ và $\sigma = 4$.
Với bộ ước lượng là một hàm số từ các mẫu đã cho, ở đây chúng ta sử dụng trung bình của các mẫu như là bộ ước lượng
cho giá trị thực $\theta$ trong phân phối chuẩn này $\mathcal{N}(\theta, \sigma^2)$.


```{.python .input}
theta_true = 1
sigma = 4
sample_len = 10000
samples = np.random.normal(theta_true, sigma, sample_len)
theta_est = np.mean(samples)
theta_est
```

```{.python .input}
#@tab pytorch
theta_true = 1
sigma = 4
sample_len = 10000
samples = torch.normal(theta_true, sigma, size=(sample_len, 1))
theta_est = torch.mean(samples)
theta_est
```

```{.python .input}
#@tab tensorflow
theta_true = 1
sigma = 4
sample_len = 10000
samples = tf.random.normal((sample_len, 1), theta_true, sigma)
theta_est = tf.reduce_mean(samples)
theta_est
```


<!--
Let's validate the trade-off equation by calculating the summation of the squared bias and the variance of our estimator. First, calculate the MSE of our estimator.
-->

Cùng xác thực phương trình đánh đổi bằng cách tính tổng độ chệch bình phương và phương sai từ bộ ước lượng của chúng ta.
Đầu tiên, tính trung bình bình phương sai số của bộ ước lượng:


```{.python .input}
#@tab all
mse(samples, theta_true)
```


<!--
Next, we calculate $\mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2$ as below. As you can see, the two values agree to numerical precision.
-->

Tiếp theo, chúng ta tính $\mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2$ như dưới đây.
Bạn có thể thấy đại lượng này gần giống với trung bình bình phương sai số đã tính ở trên.


```{.python .input}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.python .input}
#@tab pytorch
bias = stat_bias(theta_true, theta_est)
torch.square(samples.std(unbiased=False)) + torch.square(bias)
```

```{.python .input}
#@tab tensorflow
bias = stat_bias(theta_true, theta_est)
tf.square(tf.math.reduce_std(samples)) + tf.square(bias)
```


<!--
## Conducting Hypothesis Tests
-->

## Tiến hành Kiểm định Giả thuyết


<!--
The most commonly encountered topic in statistical inference is hypothesis testing. 
While hypothesis testing was popularized in the early 20th century, the first use can be traced back to John Arbuthnot in the 1700s. 
John tracked 80-year birth records in London and concluded that more men were born than women each year. 
Following that, the modern significance testing is the intelligence heritage by Karl Pearson who invented $p$-value and Pearson's chi-squared test),
William Gosset who is the father of Student's t-distribution, and Ronald Fisher who initialed the null hypothesis and the significance test.
-->

Chủ đề thường gặp nhất trong suy luận thống kê là kiểm định giả thuyết.
Tuy kiểm định giả thuyết trở nên phổ biến từ đầu thế kỷ 20, trường hợp sử dụng đầu tiên được ghi nhận bởi John Arbuthnot từ tận những năm 1700.
John đã theo dõi hồ sơ khai sinh trong 80 năm ở London và kết luận rằng mỗi năm nhiều bé trai được sinh ra hơn so với bé gái.
Tiếp đó, phép thử nghiệm độ tin cậy ngày nay là di sản trí tuệ của Karl Pearson,
người đã phát minh ra $p$-value (*trị số p*) và bài kiểm định Chi bình phương Pearson (*Pearson's chi-squared test*), William Gosses,
cha đẻ của phân phối Student và Ronald Fisher, người đã khởi xướng giả thuyết gốc và kiểm định độ tin cậy.


<!--
A *hypothesis test* is a way of evaluating some evidence against the default statement about a population. 
We refer the default statement as the *null hypothesis* $H_0$, which we try to reject using the observed data. 
Here, we use $H_0$ as a starting point for the statistical significance testing. 
The *alternative hypothesis* $H_A$ (or $H_1$) is a statement that is contrary to the null hypothesis. 
A null hypothesis is often stated in a declarative form which posits a relationship between variables. 
It should reflect the brief as explicit as possible, and be testable by statistics theory.
-->

Một bài *kiểm định giả thuyết* sẽ đánh giá các bằng chứng chống lại mệnh đề mặc định của một tổng thể.
Chúng ta gọi các mệnh đề mặc định là *giả thuyết gốc - null hypothesis* $H_0$, giả thuyết mà chúng ta cố gắng bác bỏ thông qua các dữ liệu quan sát được.
Tại đây, chúng tả sử dụng $H_0$ là điểm bắt đầu cho việc thử nghiệm độ tin cậy thống kê.
*Giả thuyết đối - alternative hypothesis* $H_A$ (hay $H_1$) là mệnh đề đối lập với giả thuyết gốc.
Giả thuyết gốc thường được định nghĩa dưới dạng khai báo mà mà ở đó nó ấn định mối quan hệ giữa các biến.
Nó nên phản ánh mệnh đề một cách rõ ràng nhất, và có thể kiểm chứng được bằng lý thuyết thống kê.


<!--
Imagine you are a chemist. After spending thousands of hours in the lab, you develop a new medicine which can dramatically improve one's ability to understand math. 
To show its magic power, you need to test it. 
Naturally, you may need some volunteers to take the medicine and see whether it can help them learn math better. How do you get started?
-->

Tưởng tượng bạn là một nhà hóa học. Sau hàng ngàn giờ nghiên cứu trong phòng thí nghiệm,
bạn đã phát triển được một loại thuốc mới giúp cải thiện đáng kể khả năng hiểu về toán của con người.
Để chứng minh sức mạnh ma thuật của thuốc, bạn cần kiểm tra nó.
Thông thường, bạn cần một số tình nguyện viên sử dụng loại thuốc này để kiểm tra xem liệu nó có giúp họ học toán tốt hơn hay không.
Bạn sẽ bắt đầu điều này như thế nào?


<!--
First, you will need carefully random selected two groups of volunteers, so that there is no difference between their math understanding ability measured by some metrics. 
The two groups are commonly referred to as the test group and the control group. 
The *test group* (or *treatment group*) is a group of individuals who will experience the medicine, 
while the *control group* represents the group of users who are set aside as a benchmark, i.e., identical environment setups except taking this medicine. 
In this way, the influence of all the variables are minimized, except the impact of the independent variable in the treatment.
-->

Đầu tiên, bạn cần cẩn thận lựa chọn ngẫu nhiên hai nhóm tình nguyện viên để đảm bảo rằng không có sự khác biệt đáng kể dựa trên các tiêu chuẩn đo lường được về khả năng hiểu toán của họ.
Hai nhóm này thường được gọi là nhóm thử nghiệm và nhóm kiểm soát.
*Nhóm thử nghiệm* (hay *nhóm trị liệu*) là nhóm người được cho sử dụng thuốc, trong khi *nhóm kiểm soát* được đặt làm chuẩn so sánh;
tức là, họ có các yếu tố môi trường giống hệt với nhóm thử nghiệm trừ việc sử dụng thuốc.
Bằng cách này, sự ảnh hưởng của tất cả các biến được giảm thiểu, trừ sự tác động của biến độc lập trong quá trình điều trị.


<!--
Second, after a period of taking the medicine, you will need to measure the two groups' math understanding by the same metrics, 
such as letting the volunteers do the same tests after learning a new math formula.
Then, you can collect their performance and compare the results.  
In this case, our null hypothesis will be that there is no difference between the two groups, and our alternate will be that there is.
-->

Thứ hai, sau một thời gian sử dụng thuốc, bạn cần đo khả năng hiểu toán của hai nhóm trên bằng tiêu chuẩn đo lường chung,
ví dụ như cho các tình nguyện viên làm cùng một bài kiểm tra sau khi học một công thức toán mới.
Sau đó bạn có thể thu thập kết quả năng lực của họ và so sánh chúng.
Trong trường hợp này, giả thuyết gốc của chúng ta đó là không có sự khác biệt nào giữa hai nhóm, và giả thuyết đối là có sự khác biệt.


<!--
This is still not fully formal.  
There are many details you have to think of carefully. 
For example, what is the suitable metrics to test their math understanding ability? 
How many volunteers for your test so you can be confident to claim the effectiveness of your medicine? 
How long should you run the test? How do you decided if there is a difference between the two groups?  
Do you care about the average performance only, or do you also the range of variation of the scores. And so on.
-->

Quy trình trên vẫn chưa hoàn toàn chính quy.
Có rất nhiều chi tiết mà bạn phải suy nghĩ cẩn trọng.
Ví dụ, đâu là tiêu chuẩn đo lường thích hợp để kiểm tra khả năng hiểu toán?
Bao nhiêu tình nguyện viên thực hiện bài kiểm tra là đủ để bạn có thể tự tin khẳng định sự hiệu quả của thuốc?
Bài kiểm tra nên kéo dài trong bao lâu? Làm cách nào để bạn quyết định được có sự khác biệt rõ rệt giữa hai nhóm?
Bạn chỉ quan tâm đến kết quả trung bình hay cả phạm vi biến thiên của các điểm số, v.v.


<!--
In this way, hypothesis testing provides framework for experimental design and reasoning about certainty in observed results.  
If we can now show that the null hypothesis is very unlikely to be true, we may reject it with confidence.
-->

Bằng cách này, kiểm định giả thuyết cung cấp một khuôn khổ cho thiết kế thử nghiệm và cách suy luận về sự chắc chắn của những kết quả quan sát được.
Nếu chứng minh được giả thuyết gốc khả năng rất cao là không đúng, thì chúng ta có thể tự tin bác bỏ nó.


<!--
To complete the story of how to work with hypothesis testing, we need to now introduce some additional terminology and make some of our concepts above formal.
-->

Để hiểu rõ hơn về cách làm việc với kiểm định giả thuyết, chúng ta cần bổ sung thêm một số thuật ngữ và toán học hóa các khái niệm ở trên.


<!--
### Statistical Significance
-->

### Ý nghĩa Thống kê

<!--
The *statistical significance* measures the probability of erroneously reject the null hypothesis, $H_0$, when it should not be rejected, i.e.,
-->

*Ý nghĩa thống kê* (*statistical significance*) đo xác suất lỗi khi bác bỏ giả thuyết gốc, $H_0$, trong khi đúng ra không nên bác bỏ nó.


$$ \text{ ý nghĩa thống kê }= 1 - \alpha = 1 - P(\text{ bác bỏ } H_0 \mid H_0 \text{ là đúng } ).$$

<!--
It is also referred to as the *type I error* or *false positive*. 
The $\alpha$, is called as the *significance level* and its commonly used value is $5\%$, i.e., $1-\alpha = 95\%$. 
The level of statistical significance level can be explained as the level of risk that we are willing to take, when we reject a true null hypothesis.
-->

Đây còn được gọi là *lỗi loại I* hay *dương tính giả*.
$\alpha$ ở đây là *mức ý nghĩa* và thường được chọn ở giá trị $5\%$, tức là $1-\alpha = 95\%$.
Mức ý nghĩa thống kê còn có thể hiểu như mức độ rủi ro mà chúng ta chấp nhận khi bác bỏ nhầm một giả thuyết gốc chính xác.


<!--
:numref:`fig_statistical_significance` shows the the observations' values and probability of a given normal distribution in a two-sample hypothesis test. 
If the observation data point is located outsides the $95\%$ threshold, it will be a very unlikely observation under the null hypothesis assumption. 
Hence, there might be something wrong with the null hypothesis and we will reject it.
-->

:numref:`fig_statistical_significance` thể hiện các giá trị quan sát và xác suất của một phân phối chuẩn trong một bài kiểm định giả thuyết thống kê hai mẫu.
Nếu các điểm dữ liệu quan sát nằm ngoài ngưỡng $95\%$, chúng sẽ rất khó xảy ra dưới giả định của giả thuyết gốc.
Do đó, giả thuyết gốc có điều gì đó không đúng và chúng ta sẽ bác bỏ nó.


<!--
![Statistical significance.](../img/statistical-significance.svg)
-->

![Ý nghĩa thống kê.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`


<!--
### Statistical Power
-->

### Năng lực Thống kê

<!--
The *statistical power* (or *sensitivity*) measures the probability of reject the null hypothesis, $H_0$, when it should be rejected, i.e.,
-->

*Năng lực thống kê* (hay còn gọi là *độ nhạy*) là xác suất bác bỏ giả thuyết gốc, $H_0$, biết rằng nó nên bị bác bỏ, tức là: 

<!--
$$ \text{statistical power }= 1 - \beta = 1 - P(\text{ fail to reject } H_0  \mid H_0 \text{ is false} ).$$
-->

$$ \text{ năng lực thống kê }= 1 - \beta = 1 - P(\text{ không bác bỏ } H_0  \mid H_0 \text{ là sai } ).$$


<!--
Recall that a *type I error* is error caused by rejecting the null hypothesis when it is true, 
whereas a *type II error* is resulted from failing to reject the null hypothesis when it is false. 
A type II error is usually denoted as $\beta$, and hence the corresponding statistical power is $1-\beta$.
-->

Nhớ lại *lỗi loại I* là lỗi do bác bỏ giả thuyết gốc khi nó đúng, còn *lỗi loại II* xảy ra do không bác bỏ giả thuyết gốc khi nó sai.
Lỗi loại II thường được kí hiệu là $\beta$, vậy nên năng lực thống kê tương ứng là $1-\beta$.


<!--
Intuitively, statistical power can be interpreted as how likely our test will detect a real discrepancy of some minimum magnitude at a desired statistical significance level. 
$80\%$ is a commonly used statistical power threshold. The higher the statistical power, the more likely we are to detect true differences.
-->

Một cách trực quan, năng lực thống kê có thể được xem như khả năng phép kiểm định phát hiện được một sai lệch thực sự với độ lớn tối thiểu nào đó, ở một mức ý nghĩa thống kê mong muốn.
$80\%$ là một ngưỡng năng lực thống kê phổ biến. Năng lực thống kê càng cao, ta càng có nhiều khả năng phát hiện được những sai lệch thực sự.


<!--
One of the most common uses of statistical power is in determining the number of samples needed.  
The probability you reject the null hypothesis when it is false depends on the degree to which it is false (known as the *effect size*) and the number of samples you have.  
As you might expect, small effect sizes will require a very large number of samples to be detectable with high probability.  
While beyond the scope of this brief appendix to derive in detail, as an example, 
want to be able to reject a null hypothesis that our sample came from a mean zero variance one Gaussian, 
and we believe that our sample's mean is actually close to one, we can do so with acceptable error rates with a sample size of only $8$.  
However, if we think our sample population true mean is close to $0.01$, then we'd need a sample size of nearly $80000$ to detect the difference.
-->

Một trong những ứng dụng phổ biến nhất của năng lực thống kê là để xác định số lượng mẫu cần thiết.
Xác suất bạn bác bỏ giả thuyết gốc khi nó sai phụ thuộc vào mức độ sai của nó (hay còn gọi là *kích thước ảnh hưởng - effect size*) và số lượng mẫu bạn có.
Có thể đoán rằng sẽ cần một số lượng mẫu rất lớn để có thể phát hiện kích thước ảnh hưởng nhỏ với xác suất cao.
Việc đi sâu vào chi tiết nằm ngoài phạm vi của phần phụ lục ngắn gọn này, nhưng đây là một ví dụ.
Giả sử ta có giả thuyết gốc rằng các mẫu được lấy từ một phân phối Gauss với kỳ vọng là không và phương sai là một.
Nếu ta tin rằng giá trị trung bình của tập mẫu gần với một, ta chỉ cần $8$ mẫu là có thể bác bỏ giả thuyết gốc với tỷ lệ lỗi chấp nhận được.
Tuy nhiên, nếu ta cho rằng giá trị trung bình thực sự của tổng thể gần với $0.01$, thì ta cần cỡ khoảng $80000$ mẫu để có thể phát hiện được sự sai lệch.

<!--
We can imagine the power as a water filter. In this analogy, a high power hypothesis test is like a high quality water filtration system 
that will reduce harmful substances in the water as much as possible. 
On the other hand, a smaller discrepancy is like a low quality water filter, where some relative small substances may easily escape from the gaps. 
Similarly, if the statistical power is not of enough high power, then the test may not catch the smaller discrepancy.
-->

Ta có thể hình dung năng lực thống kê như một cái máy lọc nước.
Trong phép so sánh này, một kiểm định với năng lực cao giống như một hệ thống lọc nước chất lượng tốt, loại bỏ được các chất độc trong nước nhiều nhất có thể.
Ngược lại, các sai lệch nhỏ cũng giống các chất cặn bẩn nhỏ, một cái máy lọc chất lượng kém sẽ để lọt các chất bẩn nhỏ đó.
Tương tự, nếu năng lực thống kê không đủ cao, phép kiểm định có thể không phát hiện được các sai lệch nhỏ.

<!--
### Test Statistic
-->

### Tiêu chuẩn Kiểm định

<!--
A *test statistic* $T(x)$ is a scalar which summarizes some characteristic of the sample data.  
The goal of defining such a statistic is that it should allow us to distinguish between different distributions and conduct our hypothesis test.  
hinking back to our chemist example, if we wish to show that one population performs better than the other, it could be reasonable to take the mean as the test statistic.  
Different choices of test statistic can lead to statistical test with drastically different statistical power.
-->

*Tiêu chuẩn kiểm định* $T(x)$ là một số vô hướng có khả năng khái quát một đặc tính nào đó của dữ liệu mẫu.
Mục đích của việc đặt ra một tiêu chuẩn như vậy là để phân biệt các phân phối khác nhau và tiến hành kiểm định thống kê.
Nhìn lại ví dụ về nhà hóa học, nếu ta muốn chỉ ra rằng một tổng thể có chất lượng tốt hơn một tổng thể khác, việc lấy giá trị trung bình làm tiêu chuẩn kiểm định có vẻ hợp lý.
Các chọn lựa tiêu chuẩn kiểm định khác nhau có thể dẫn đến các phép kiểm định thống kê với năng lực thống kê khác nhau rõ rệt.


<!--
Often, $T(X)$ (the distribution of the test statistic under our null hypothesis) will follow, at least approximately, 
a common probability distribution such as a normal distribution when considered under the null hypothesis. 
If we can derive explicitly such a distribution, and then measure our test statistic on our dataset, 
we can safely reject the null hypothesis if our statistic is far outside the range that we would expect.
Making this quantitative leads us to the notion of $p$-values.
-->

Thường thì $T(X)$ (phân phối của tiêu chuẩn kiểm định dưới giả thuyết gốc) sẽ (xấp xỉ) 
tuân theo một phân phối phổ biến như phân phối chuẩn, khi được xem xét dưới giả thuyết gốc.
Nếu ta có thể chỉ rõ một phân phối như vậy, và sau đó tính tiêu chuẩn kiểm định trên tập dữ liệu,
ta có thể yên tâm bác bỏ giả thuyết gốc nếu thống kê đó nằm xa bên ngoài khoảng mong đợi.
Định lượng hóa ý tưởng này đưa ta đến với khái niệm trị số $p$ (*$p$-values*).


<!--
### $p$-value
-->

### Trị số $p$

<!--
The $p$-value (or the *probability value*) is the probability that $T(X)$ is at least as extreme as the observed test statistic $T(x)$ assuming that the null hypothesis is *true*, i.e.,
-->


Trị số $p$ (hay còn gọi là *trị số xác suất*) là xác suất mà $T(X)$ lớn hơn hoặc bằng tiêu chuẩn kiểm định ta thu được, giả sử rằng giả thuyết gốc đúng, tức là:


$$ p\text{-value} = P_{H_0}(T(X) \geq T(x)).$$


<!--
If the $p$-value is smaller than or equal to a predefined and fixed statistical significance level $\alpha$, we may reject the null hypothesis. 
Otherwise, we will conclude that we are lack of evidence to reject the null hypothesis. 
For a given population distribution, the *region of rejection* will be the interval contained of all the points which has a $p$-value smaller than the statistical significance level $\alpha$.
-->

Nếu trị số $p$ nhỏ hơn hoặc bằng một mức ý nghĩa thống kê cố định $\alpha$ cho trước, ta có thể bác bỏ giả thuyết gốc.
Còn nếu không, ta kết luận không có đủ bằng chứng để bác bỏ giả thuyết gốc.
Với một phân phối của tổng thể, *miền bác bỏ* là khoảng chứa tất cả các điểm có trị số $p$ nhỏ hơn mức ý nghĩa thống kê $\alpha$.


<!--
### One-side Test and Two-sided Test
-->

### Kiểm định Một phía và Kiểm định Hai phía


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

Thường thì có hai loại kiểm định ý nghĩa thống kê: kiểm định một phía và kiểm định hai phía.
*Kiểm định một phía* (hay *kiểm định một đuôi*) có thể được áp dụng khi giả thuyết gốc và giả thuyết đối chỉ đi theo một hướng.
Ví dụ, giả thuyết gốc có thể cho rằng tham số thực $\theta$ nhỏ hơn hoặc bằng một giá trị $c$.
Giả thuyết đối sẽ là $\theta$ lớn hơn $c$.
Nói cách khác, miền bác bỏ chỉ nằm ở một bên của phân phối mẫu.
Trái với kiểm định một phía, *kiểm định hai phía* (hay *kiểm định hai đuôi*) có thể được áp dụng khi miền bác bỏ nằm ở cả hai phía của phân phối mẫu.
Ví dụ cho trường hợp này có thể là một giả thuyết gốc cho rằng tham số thực $\theta$ bằng một giá trị $c$.
Giả thuyết đối lúc này sẽ là $\theta$ nhỏ hơn và lớn hơn $c$.


<!--
### General Steps of Hypothesis Testing
-->

### Các bước Thông thường trong Kiểm định Giả thuyết


<!--
After getting familiar with the above concepts, let's go through the general steps of hypothesis testing.
-->

Sau khi làm quen với các khái niệm ở trên, hãy cùng xem các bước kiểm định giả thuyết thông thường.


<!--
1. State the question and establish a null hypotheses $H_0$.
2. Set the statistical significance level $\alpha$ and a statistical power ($1 - \beta$).
3. Obtain samples through experiments. The number of samples needed will depend on the statistical power, and the expected effect size.
4. Calculate the test statistic and the $p$-value.
5. Make the decision to keep or reject the null hypothesis based on the $p$-value and the statistical significance level $\alpha$.
-->


1. Đặt câu hỏi và đưa ra giả thuyết gốc $H_0$.
2. Chọn mức ý nghĩa thống kê $\alpha$ và năng lực thống kê ($1 - \beta$).
3. Thu thập mẫu qua các thử nghiệm. Số lượng mẫu cần thiết sẽ phụ thuộc vào năng lực thống kê, và hệ số ảnh hưởng mong muốn.
4. Tính tiêu chuẩn kiểm định và trị số $p$.
5. Quyết định chấp nhận hoặc bác bỏ giả thuyết gốc dựa trên trị số $p$ và mức ý nghĩa thống kê $\alpha$.


<!--
To conduct a hypothesis test, we start by defining a null hypothesis and a level of risk that we are willing to take. 
Then we calculate the test statistic of the sample, taking an extreme value of the test statistic as evidence against the null hypothesis. 
If the test statistic falls within the reject region, we may reject the null hypothesis in favor of the alternative.
-->

Để tiến hành kiểm định giả thuyết, ta bắt đầu với việc định nghĩa giả thuyết gốc và mức rủi ro chấp nhận được.
Sau đó ta tính tiêu chuẩn kiểm định của mẫu, lấy cực trị của tiêu chuẩn kiểm định làm bằng chứng để phủ định giả thuyết gốc.
Nếu tiêu chuẩn kiểm định rơi vào miền bác bỏ, ta có thể bác bỏ giả thuyết gốc và ủng hộ giả thuyết đối.


<!--
Hypothesis testing is applicable in a variety of scenarios such as the clinical trails and A/B testing.
-->

Kiểm định giả thuyết áp dụng được trong nhiều tình huống như thử nghiệm lâm sàng (*clinical trials*) và kiểm định A/B.


<!--
## Constructing Confidence Intervals
-->

## Xây dựng khoảng Tin cậy


<!--
When estimating the value of a parameter $\theta$, point estimators like $\hat \theta$ are of limited utility since they contain no notion of uncertainty. 
Rather, it would be far better if we could produce an interval that would contain the true parameter $\theta$ with high probability.  
If you were interested in such ideas a century ago, then you would have been excited to 
read "Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability" by Jerzy Neyman :cite:`Neyman.1937`, 
who first introduced the concept of confidence interval in 1937.
-->

Khi ước lượng giá trị của tham số $\theta$, sử dụng bộ ước lượng điểm như $\hat \theta$ bị hạn chế vì chúng không bao hàm sự bất định.
Thay vào đó, sẽ tốt hơn nhiều nếu ta có thể tìm ra một khoảng chứa tham số $\theta$ thật sự với xác suất cao.
Nếu bạn hứng thú với những khái niệm từ một thế kỷ trước như thế này, có lẽ bạn nên đọc cuốn
"Outline of a Theory of Statistical Estimation Based on the Classical Theory of Probability" (*Đại cương về Lý thuyết Ước lượng Thống kê dựa trên Lý thuyết Xác suất Cổ điển*)
của Jerzy Neyman :cite:`Neyman.1937`, người đã đưa ra khái niệm về khoảng tin cậy vào năm 1937.

<!--
To be useful, a confidence interval should be as small as possible for a given degree of certainty. Let's see how to derive it.
-->

Để có tính hữu dụng, khoảng tin cậy nên càng bé càng tốt với một mức độ chắc chắn cho trước.
Hãy cùng xem xét cách tính khoảng tin cậy.


<!--
### Definition
-->

### Định nghĩa

<!--
Mathematically, a *confidence interval* for the true parameter $\theta$ is an interval $C_n$ that computed from the sample data such that
-->

Về mặt toán học, *khoảng tin cậy* $C_n$ của tham số thực $\theta$ được tính từ dữ liệu mẫu sao cho:


$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`


<!--
Here $\alpha \in (0, 1)$, and $1 - \alpha$ is called the *confidence level* or *coverage* of the interval. 
This is the same $\alpha$ as the significance level as we discussed about above.
-->

Với $\alpha \in (0, 1)$, và $1 - \alpha$ được gọi là *mức độ tin cậy* hoặc *độ phủ* của khoảng đó.
Nó cũng chính là hệ số $\alpha$ của mức ý nghĩa thống kê mà chúng ta đã bàn luận ở trên.


<!--
Note that :eqref:`eq_confidence` is about variable $C_n$, not about the fixed $\theta$. 
To emphasize this, we write $P_{\theta} (C_n \ni \theta)$ rather than $P_{\theta} (\theta \in C_n)$.
-->

Chú ý rằng :eqref:`eq_confidence` là về biến số $C_n$, chứ không phải giá trị cố định $\theta$.
Để nhấn mạnh điều này, chúng ta viết $P_{\theta} (C_n \ni \theta)$ thay cho $P_{\theta} (\theta \in C_n)$.


<!--
### Interpretation
-->

### Diễn giải

<!--
It is very tempting to interpret a $95\%$ confidence interval as an interval where you can be $95\%$ sure the true parameter lies, however this is sadly not true. 
The true parameter is fixed, and it is the interval that is random.  
Thus a better interpretation would be to say that if you generated a large number of confidence intervals by this procedure, 
$95\%$ of the generated intervals would contain the true parameter.
-->

Rất dễ để cho rằng khoảng tin cậy $95\%$ tương đương với việc chắc chắn $95\%$ giá trị thật phân bố trong khoảng đó, tuy nhiên đáng buồn thay điều này lại không chính xác.
Tham số thật là cố định và khoảng tin cậy mới là ngẫu nhiên.
Vậy nên một cách diễn giải tốt hơn đó là nếu bạn tạo ra một số lượng lớn các khoảng tin cậy theo quy trình này, thì $95\%$ các khoảng được tạo sẽ chứa tham số thật.


<!--
This may seem pedantic, but it can have real implications for the interpretation of the results.  
In particular, we may satisfy :eqref:`eq_confidence` by constructing intervals that we are *almost certain* do not contain the true value, as long as we only do so rarely enough.  
We close this section by providing three tempting but false statements.  
An in-depth discussion of these points can be found in :cite:`Morey.Hoekstra.Rouder.ea.2016`.
-->

Điều này nghe có vẻ tiểu tiết, nhưng lại có một ý nghĩa quan trọng trong việc diễn giải các kết quả.
Cụ thể, chúng ta có thể thỏa mãn :eqref:`eq_confidence` bằng cách tạo ra các khoảng *gần như chắc chắn* không chứa tham số thật, miễn là số lượng các khoảng này đủ nhỏ.
Chúng ta kết thúc mục này bằng ba mệnh đề nghe hợp lý nhưng lại không chính xác.
Thảo luận sâu hơn về các mệnh đề này có thể tham khảo thêm ở :cite:`Morey.Hoekstra.Rouder.ea.2016`.


<!--
* **Fallacy 1**. Narrow confidence intervals mean we can estimate the parameter precisely.
* **Fallacy 2**. The values inside the confidence interval are more likely to be the true value than those outside the interval.
* **Fallacy 3**. The probability that a particular observed $95\%$ confidence interval contains the true value is $95\%$.
-->

* **Sai lầm 1**: Khoảng tin cậy hẹp cho phép chúng ta dự đoán các giá trị một cách chính xác.
* **Sai lầm 2**: Các giá trị nằm trong khoảng tin cậy có nhiều khả năng là giá trị thực hơn là các giá trị nằm bên ngoài.
* **Sai lầm 3**: Xác xuất một khoảng tin cậy $95\%$ chứa các giá trị thực là $95\%$.


<!--
Sufficed to say, confidence intervals are subtle objects.  H
owever, if you keep the interpretation clear, they can be powerful tools.
-->

Có thể nói, các khoảng tin cậy là những đối tượng khó ước lượng.
Tuy nhiên nếu như ta diễn giải chúng một cách rõ ràng, thì chúng có thể trở thành những công cụ quyền năng.


<!--
### A Gaussian Example
-->

### Một ví dụ về Gaussian

<!--
Let's discuss the most classical example, the confidence interval for the mean of a Gaussian of unknown mean and variance.  
Suppose we collect $n$ samples $\{x_i\}_{i=1}^n$ from our Gaussian $\mathcal{N}(\mu, \sigma^2)$.  
We can compute estimators for the mean and standard deviation by taking
-->

Cùng bàn về ví dụ kinh điển nhất, khoảng tin cậy cho giá trị trung bình của một phân phối Gaussian với kỳ vọng và phương sai chưa xác định.
Giả sử chúng ta thu thập $n$ mẫu $\{x_i\}_{i=1}^n$ từ phân phối Gaussian $\mathcal{N}(\mu, \sigma^2)$.
Chúng ta có thể ước lượng kỳ vọng và độ lệch chuẩn bằng công thức:


$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\text{và}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$


<!--
If we now consider the random variable
-->

Nếu bây giờ chúng ta xem xét biến ngẫu nhiên: 


$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$


<!--
we obtain a random variable following a well-known distribution called the *Student's t-distribution on* $n-1$ *degrees of freedom*.
-->

Chúng ta có được một biến ngẫu nhiên theo *phân phối t Student trên* $n - 1$ *bậc tự do*.


<!--
This distribution is very well studied, and it is known, for instance, that as $n\rightarrow \infty$, 
it is approximately a standard Gaussian, and thus by looking up values of the Gaussian c.d.f. in a table, 
we may conclude that the value of $T$ is in the interval $[-1.96, 1.96]$ at least $95\%$ of the time.  
For finite values of $n$, the interval needs to be somewhat larger, but are well known and precomputed in tables.
-->

Phân phối này đã được nghiên cứu rất chi tiết, và đã được chứng minh là khi $n\rightarrow \infty$, 
nó xấp xỉ với một phân phối Gauss tiêu chuẩn, và do đó bằng cách nhìn vào bảng giá trị phân phối tích lũy Gauss, 
chúng ta có thể kết luận rằng giá trị $T$ nằm trong khoảng $[-1.96, 1.96]$ tối thiểu là $95\%$ các trường hợp.
Với giá trị $n$ hữu hạn, khoảng tin cậy sẽ lớn hơn, nhưng chúng vẫn rõ ràng và thường được tính sẵn và trình bày thành bảng.


<!--
Thus, we may conclude that for large $n$,
-->

Do đó, chúng ta có thể kết luận với giá trị $n$ lớn: 


$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$


<!--
Rearranging this by multiplying both sides by $\hat\sigma_n/\sqrt{n}$ and then adding $\hat\mu_n$, we obtain
-->

Sắp xếp lại công thức này bằng cách nhân hai vế với $\hat\sigma_n/\sqrt{n}$ và cộng thêm $\hat\mu_n$, ta có:


$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$


<!--
Thus we know that we have found our $95\%$ confidence interval:
-->

Như vậy chúng ta đã xác định được khoảng tin cậy $95\%$ cần tìm:


$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`


<!--
It is safe to say that :eqref:`eq_gauss_confidence` is one of the most used formula in statistics. 
Let's close our discussion of statistics by implementing it. 
For simplicity, we assume we are in the asymptotic regime. 
Small values of $N$ should include the correct value of `t_star` obtained either programmatically or from a $t$-table.
-->

Không quá khi nói rằng :eqref:`eq_gauss_confidence` là một trong những công thức sử dụng nhiều nhất trong thống kê.
Hãy kết thúc thảo luận về thống kê của chúng ta bằng cách lập trình tìm khoảng tin cậy.
Để đơn giản, giả sử chúng ta đang làm việc ở vùng tiệm cận.
Khi $N$ nhỏ, nên xác định giá trị chính xác của `t_star` bằng phương pháp lập trình hoặc từ bảng tra phân phối tích lũy $t$ Student.


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

```{.python .input}
#@tab pytorch
# PyTorch uses Bessel's correction by default, which means the use of ddof=1
# instead of default ddof=0 in numpy. We can use unbiased=False to imitate
# ddof=0.
# Number of samples
N = 1000
# Sample dataset
samples = torch.normal(0, 1, size=(N,))
# Lookup Students's t-distribution c.d.f.
t_star = 1.96
# Construct interval
mu_hat = torch.mean(samples)
sigma_hat = samples.std(unbiased=True)
(mu_hat - t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)),\
 mu_hat + t_star*sigma_hat/torch.sqrt(torch.tensor(N, dtype=torch.float32)))
```

```{.python .input}
#@tab tensorflow
# Number of samples
N = 1000
# Sample dataset
samples = tf.random.normal((N,), 0, 1)
# Lookup Students's t-distribution c.d.f.
t_star = 1.96
# Construct interval
mu_hat = tf.reduce_mean(samples)
sigma_hat = tf.math.reduce_std(samples)
(mu_hat - t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)), \
 mu_hat + t_star*sigma_hat/tf.sqrt(tf.constant(N, dtype=tf.float32)))
```


## Tóm tắt

<!--
* Statistics focuses on inference problems, whereas deep learning emphasizes on making accurate predictions without explicitly programming and understanding.
* There are three common statistics inference methods: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.
* There are three most common estimators: statistical bias, standard deviation, and mean square error.
* A confidence interval is an estimated range of a true population parameter that we can construct by given the samples.
* Hypothesis testing is a way of evaluating some evidence against the default statement about a population.
-->

* Thống kê tập trung vào các vấn đề suy luận, trong khi học sâu chú trọng vào đưa ra các dự đoán chuẩn xác mà không cần một phương pháp lập trình hay kiến thức rõ ràng.
* Ba phương pháp suy luận thống kê thông dụng nhất: đánh giá và so sánh các bộ ước lượng, tiến hành kiểm định giả thuyết, và tạo các khoảng tin cậy.
* Ba bộ ước lượng thông dụng nhất: độ chệch thống kê, độ lệch chuẩn, và trung bình bình phương sai số.
* Một khoảng tin cậy là khoảng ước tính của tập tham số thực mà chúng ta có thể tạo ra bằng các mẫu cho trước.
* Kiểm định giả thuyết là phương pháp để đánh giá các chứng cứ chống lại mệnh đề mặc định về một tổng thể.


## Bài tập

<!--
1. Let $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} \mathrm{Unif}(0, \theta)$, where "iid" stands for *independent and identically distributed*. Consider the following estimators of $\theta$:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * Find the statistical bias, standard deviation, and mean square error of $\hat{\theta}.$
    * Find the statistical bias, standard deviation, and mean square error of $\tilde{\theta}.$
    * Which estimator is better?
2. For our chemist example in introduction, can you derive the 5 steps to conduct a two-sided hypothesis testing? Given the statistical significance level $\alpha = 0.05$ and the statistical power $1 - \beta = 0.8$.
3. Run the confidence interval code with $N=2$ and $\alpha = 0.5$ for $100$ independently generated dataset, and plot the resulting intervals (in this case `t_star = 1.0`).  You will see several very short intervals which are very far from containing the true mean $0$.  Does this contradict the interpretation of the confidence interval?  Do you feel comfortable using short intervals to indicate high precision estimates?
-->

1. Cho $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} \mathrm{Unif}(0, \theta)$, 
với "iid" là viết tắt của *phân phối độc lập và giống nhau - independent and identically distributed*. 
Xét bộ ước lượng $\theta$ dưới đây: 
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$
$$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * Tìm độ chệch thống kê, độ lệch chuẩn, và trung bình bình phương sai số của $\hat{\theta}.$ 
    * Tìm độ chệch thống kê, độ lệch chuẩn, và trung bình bình phương sai số của $\tilde{\theta}.$ 
    * Bộ ước lượng nào tốt hơn? 
2. Trở lại ví dụ về nhà hóa học của chúng ta ở phần mở đầu, liệt kê 5 bước để tiến hành kiểm định giả thuyết hai chiều,
biết mức ý nghĩa thống kê $\alpha = 0.05$ và năng lực thống kê $1 - \beta = 0.8$. 
3. Chạy đoạn mã lập trình khoảng tin cậy biết $N=2$ và $\alpha = 0.5$ với $100$ dữ liệu được tạo độc lập, sau đó vẽ đồ thị các khoảng kết quả (trường hợp này `t_star = 1.0`). 
Ban sẽ thấy một vài khoảng rất nhỏ cách xa khoảng chứa giá trị kỳ vọng thực $0$.
Điều này có mâu thuẫn với việc diễn giải khoảng tin cậy không? Có đúng không khi sử dụng các khoảng nhỏ này để nói các ước lượng có độ chính xác cao? 


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/statistics/419), [Pytorch](https://discuss.d2l.ai/t/1102), [Tensorflow](https://discuss.d2l.ai/t/1103)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Ngô Thế Anh Khoa
* Phạm Hồng Vinh
* Vũ Hữu Tiệp
* Lê Khắc Hồng Phúc
* Đoàn Võ Duy Thanh
* Nguyễn Lê Quang Nhật
* Mai Sơn Hải
* Phạm Minh Đức
* Nguyễn Cảnh Thướng
* Nguyễn Văn Cường

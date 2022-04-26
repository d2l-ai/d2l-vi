# Thống kê
:label:`sec_statistics`

Không còn nghi ngờ gì nữa, để trở thành một học viên học sâu hàng đầu, khả năng đào tạo các mô hình hiện đại và chính xác cao là rất quan trọng. Tuy nhiên, người ta thường không rõ khi nào cải tiến là đáng kể, hoặc chỉ là kết quả của những biến động ngẫu nhiên trong quá trình đào tạo. Để có thể thảo luận về sự không chắc chắn về các giá trị ước tính, chúng ta phải tìm hiểu một số thống kê. 

Tài liệu tham khảo sớm nhất của *thống kê * có thể được bắt nguồn từ một học giả Ả Rập Al-Kindi trong thế kỷ $9^{\mathrm{th}}$, người đã đưa ra một mô tả chi tiết về cách sử dụng thống kê và phân tích tần số để giải mã các tin nhắn được mã hóa. Sau 800 năm, số liệu thống kê hiện đại phát sinh từ Đức vào những năm 1700, khi các nhà nghiên cứu tập trung vào việc thu thập và phân tích dữ liệu nhân khẩu học và kinh tế. Ngày nay, thống kê là môn khoa học liên quan đến việc thu thập, xử lý, phân tích, giải thích và trực quan hóa dữ liệu. Hơn nữa, lý thuyết cốt lõi của thống kê đã được sử dụng rộng rãi trong nghiên cứu trong học viện, công nghiệp và chính phủ. 

Cụ thể hơn, số liệu thống kê có thể được chia thành *thống kê mô tả* và * suy luận thống ký*. Tập trung trước đây vào việc tóm tắt và minh họa các tính năng của một bộ sưu tập dữ liệu quan sát được, được gọi là mẫu* *. Mẫu được rút ra từ một * dân số*, biểu thị tổng số các cá nhân, vật phẩm hoặc sự kiện tương tự của lợi ích thí nghiệm của chúng tôi. Trái ngược với thống kê mô tả, *suy luận thống ký* suy luận thêm các đặc điểm của một quần thể từ các mẫu* đã cho, dựa trên các giả định rằng phân phối mẫu có thể nhân rộng phân bố dân số ở một mức độ nào đó. 

Bạn có thể tự hỏi: “Sự khác biệt thiết yếu giữa học máy và thống kê là gì?” Nói về cơ bản, thống kê tập trung vào bài toán suy luận. Loại bài toán này bao gồm mô hình hóa mối quan hệ giữa các biến, chẳng hạn như suy luận nhân quả, và kiểm tra ý nghĩa thống kê của các tham số mô hình, chẳng hạn như thử nghiệm A/B. Ngược lại, machine learning nhấn mạnh vào việc đưa ra các dự đoán chính xác, mà không cần lập trình và hiểu rõ chức năng của từng tham số. 

Trong phần này, chúng tôi sẽ giới thiệu ba loại phương pháp suy luận thống kê: đánh giá và so sánh các ước lượng, tiến hành các bài kiểm tra giả thuyết và xây dựng khoảng thời gian tin cậy. Những phương pháp này có thể giúp chúng ta suy ra các đặc điểm của một quần thể nhất định, tức là tham số thực sự $\theta$. Đối với ngắn gọn, chúng ta giả định rằng tham số thật $\theta$ của một quần thể nhất định là một giá trị vô hướng. Thật đơn giản để mở rộng đến trường hợp $\theta$ là một vectơ hoặc tensor, do đó chúng tôi bỏ qua nó trong cuộc thảo luận của chúng tôi. 

## Đánh giá và so sánh ước tính

Trong thống kê, một *estimator* là một hàm của các mẫu đã cho được sử dụng để ước tính tham số đúng $\theta$. Chúng tôi sẽ viết $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ cho ước tính $\theta$ sau khi quan sát các mẫu {$x_1, x_2, \ldots, x_n$}. 

Chúng ta đã thấy các ví dụ đơn giản về các chứng thực trước đây trong phần :numref:`sec_maximum_likelihood`. Nếu bạn có một số mẫu từ một biến ngẫu nhiên Bernoulli, thì ước tính khả năng tối đa cho xác suất biến ngẫu nhiên là một có thể thu được bằng cách đếm số cái quan sát và chia cho tổng số mẫu. Tương tự, một bài tập yêu cầu bạn chỉ ra rằng ước tính khả năng tối đa của trung bình của một Gaussian được đưa ra một số mẫu được đưa ra bởi giá trị trung bình của tất cả các mẫu. Những ước lượng này hầu như sẽ không bao giờ cung cấp giá trị thực sự của tham số, nhưng lý tưởng cho một số lượng lớn các mẫu ước tính sẽ gần gũi. 

Ví dụ, chúng ta hiển thị bên dưới mật độ thực của một biến ngẫu nhiên Gaussian với trung bình 0 và phương sai một, cùng với một mẫu thu thập từ Gaussian đó. Chúng tôi xây dựng tọa độ $y$ để mọi điểm đều có thể nhìn thấy và mối quan hệ với mật độ ban đầu rõ ràng hơn.

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

Có thể có nhiều cách để tính toán một ước tính của một tham số $\hat{\theta}_n$. Trong phần này, chúng tôi giới thiệu ba phương pháp phổ biến để đánh giá và so sánh các ước lượng: lỗi bình phương trung bình, độ lệch chuẩn và thiên vị thống kê. 

### Lỗi bình phương trung bình

Có lẽ số liệu đơn giản nhất được sử dụng để đánh giá các ước lượng là lỗi bình phương * trung bình (MSE) * (hoặc mất $l_2$) của một ước tính có thể được định nghĩa là 

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$
:eqlabel:`eq_mse_est`

This allows us to quantify the average squared deviation from the true value.  MSE is always non-negative. If you have read :numref:`sec_linear_regression`, you will recognize it as the most commonly used regression loss function. As a measure to evaluate an estimator, the closer its value to zero, the closer the estimator is close to the true parameter $\theta$.

### Thiên vị thống kê

MSE cung cấp một số liệu tự nhiên, nhưng chúng ta có thể dễ dàng tưởng tượng nhiều hiện tượng khác nhau có thể làm cho nó lớn. Hai cơ bản quan trọng là biến động trong ước tính do tính ngẫu nhiên trong tập dữ liệu, và lỗi có hệ thống trong ước tính do quy trình ước tính. 

Đầu tiên, chúng ta hãy đo lỗi có hệ thống. Đối với một ước tính $\hat{\theta}_n$, minh họa toán học của *thiên vị thống ký* có thể được định nghĩa là 

$$\mathrm{bias}(\hat{\theta}_n) = E(\hat{\theta}_n - \theta) = E(\hat{\theta}_n) - \theta.$$
:eqlabel:`eq_bias`

Lưu ý rằng khi $\mathrm{bias}(\hat{\theta}_n) = 0$, kỳ vọng của ước tính $\hat{\theta}_n$ bằng giá trị thực của tham số. Trong trường hợp này, chúng tôi nói $\hat{\theta}_n$ là một ước tính không thiên vị. Nói chung, một ước tính không thiên vị tốt hơn một ước tính thiên vị vì kỳ vọng của nó giống như tham số thực sự. 

Tuy nhiên, điều đáng để nhận thức được rằng những người dự kiến thiên vị thường được sử dụng trong thực tế. Có những trường hợp mà các ước lượng không thiên vị không tồn tại mà không có giả định thêm, hoặc không thể chữa được để tính toán. Điều này có vẻ như là một lỗ hổng đáng kể trong một nhà ước tính, tuy nhiên phần lớn các ước lượng gặp phải trong thực tế ít nhất là không thiên vị theo nghĩa là sự thiên vị có xu hướng bằng không vì số lượng mẫu có sẵn có xu hướng vô cùng: $\lim_{n \rightarrow \infty} \mathrm{bias}(\hat{\theta}_n) = 0$. 

### Phương sai và độ lệch chuẩn

Thứ hai, chúng ta hãy đo lường sự ngẫu nhiên trong ước tính. Nhớ lại từ :numref:`sec_random_variables`, *độ lệch chuẩn* (hoặc *lỗi tiêu chuẩn*) được định nghĩa là gốc bình phương của phương sai. Chúng tôi có thể đo mức độ dao động của một ước tính bằng cách đo độ lệch chuẩn hoặc phương sai của ước tính đó. 

$$\sigma_{\hat{\theta}_n} = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$
:eqlabel:`eq_var_est`

Điều quan trọng là so sánh :eqref:`eq_var_est` đến :eqref:`eq_mse_est`. Trong phương trình này, chúng ta không so sánh với giá trị dân số thực sự $\theta$, mà thay vào đó là $E(\hat{\theta}_n)$, mẫu dự kiến có nghĩa là. Do đó, chúng tôi không đo lường mức độ ước tính có xu hướng từ giá trị thực sự, mà thay vào đó chúng ta đo lường sự dao động của chính ước tính. 

### Sự Bias-Variance Trade-off

Rõ ràng bằng trực giác rằng hai thành phần chính này góp phần vào lỗi bình phương trung bình. Điều hơi gây sốc là chúng ta có thể cho thấy rằng đây thực sự là một * decomposition* của lỗi bình phương trung bình vào hai đóng góp này cộng với một phần ba. Điều đó có nghĩa là chúng ta có thể viết sai số bình phương trung bình là tổng của bình phương của sự thiên vị, phương sai và lỗi không thể khắc phục được. 

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

Chúng tôi đề cập đến công thức trên là *bias-variance trade-off*. Lỗi bình phương trung bình có thể được chia thành ba nguồn error: the error from high bias, the error from high variance and the irreducible error. The bias error is commonly seen in a simple model (such as a linear regression model), which cannot extract high dimensional relations between the features and the outputs. If a model suffers from high bias error, we often say it is *underfitting* or lack of *flexibilty* as introduced in (:numref:`sec_model_selection`). Phương sai cao thường là kết quả từ một mô hình quá phức tạp, vượt quá dữ liệu đào tạo. Do đó, mô hình * overfitting* nhạy cảm với các biến động nhỏ trong dữ liệu. Nếu một mô hình bị phương sai cao, chúng ta thường nói rằng đó là * overfitting* và thiếu *khái quát hóa* như được giới thiệu trong (:numref:`sec_model_selection`). Lỗi không thể khắc phục là kết quả từ tiếng ồn trong chính $\theta$. 

### Đánh giá Ước tính trong Mã

Vì độ lệch chuẩn của một ước tính đã được thực hiện bằng cách gọi đơn giản `a.std()` cho tensor `a`, chúng tôi sẽ bỏ qua nó nhưng thực hiện thiên vị thống kê và sai số bình phương trung bình.

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

Để minh họa phương trình của sự cân bằng sai lệch, chúng ta hãy mô phỏng phân phối bình thường $\mathcal{N}(\theta, \sigma^2)$ với $10,000$ mẫu. Ở đây, chúng tôi sử dụng một $\theta = 1$ và $\sigma = 4$. Vì ước tính là một chức năng của các mẫu đã cho, ở đây chúng tôi sử dụng trung bình của các mẫu như một ước tính cho $\theta$ đúng trong phân phối bình thường này $\mathcal{N}(\theta, \sigma^2)$.

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

Chúng ta hãy xác nhận phương trình đánh đổi bằng cách tính tổng của sự thiên vị bình phương và phương sai của ước tính của chúng ta. Đầu tiên, tính toán MSE của ước tính của chúng tôi.

```{.python .input}
#@tab all
mse(samples, theta_true)
```

Tiếp theo, chúng tôi tính $\mathrm{Var} (\hat{\theta}_n) + [\mathrm{bias} (\hat{\theta}_n)]^2$ như dưới đây. Như bạn có thể thấy, hai giá trị đồng ý với độ chính xác số.

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

## Tiến hành các bài kiểm tra giả thuyết

Chủ đề thường gặp nhất trong suy luận thống kê là thử nghiệm giả thuyết. Trong khi thử nghiệm giả thuyết được phổ biến vào đầu thế kỷ 20, việc sử dụng đầu tiên có thể được truy trở lại John Arbuthnot vào những năm 1700. John đã theo dõi kỷ lục sinh 80 năm ở London và kết luận rằng nhiều nam giới được sinh ra hơn phụ nữ mỗi năm. Tiếp theo đó, thử nghiệm ý nghĩa hiện đại là di sản tình báo của Karl Pearson, người đã phát minh ra $p$ giá trị và bài kiểm tra chi bình phương của Pearson, William Gosset là cha đẻ của phân phối t của Sinh viên, và Ronald Fisher, người đã khởi xướng giả thuyết vô giá trị và bài kiểm tra ý nghĩa. 

A *giả thuyết test* là một cách để đánh giá một số bằng chứng chống lại tuyên bố mặc định về một dân số. Chúng tôi tham khảo câu lệnh mặc định là giả thuyết *null * $H_0$, mà chúng tôi cố gắng từ chối bằng cách sử dụng dữ liệu quan sát. Ở đây, chúng tôi sử dụng $H_0$ làm điểm khởi đầu cho việc kiểm tra ý nghĩa thống kê. Giả thuyết thay thế* $H_A$ (hoặc $H_1$) là một tuyên bố trái với giả thuyết null. Một giả thuyết null thường được nêu trong một hình thức khai báo đặt ra một mối quan hệ giữa các biến. Nó sẽ phản ánh ngắn gọn càng rõ ràng càng tốt, và được kiểm tra bởi lý thuyết thống kê. 

Hãy tưởng tượng bạn là một nhà hóa học. Sau khi dành hàng ngàn giờ trong phòng thí nghiệm, bạn phát triển một loại thuốc mới có thể cải thiện đáng kể khả năng hiểu toán của một người. Để thể hiện sức mạnh ma thuật của nó, bạn cần phải kiểm tra nó. Đương nhiên, bạn có thể cần một số tình nguyện viên uống thuốc và xem liệu nó có thể giúp họ học toán tốt hơn hay không. Làm thế nào để bạn bắt đầu? 

Đầu tiên, bạn sẽ cần hai nhóm tình nguyện viên được lựa chọn ngẫu nhiên cẩn thận, để không có sự khác biệt giữa khả năng hiểu biết toán học của họ được đo bằng một số số chỉ số. Hai nhóm thường được gọi là nhóm thử nghiệm và nhóm kiểm soát. *kiểm tra nhóm* (hoặc *nhóm điều trị*) là một nhóm các cá nhân sẽ trải nghiệm thuốc, trong khi nhóm kiểm soát * đại diện cho nhóm người dùng được đặt sang một bên như một chuẩn mực, tức là thiết lập môi trường giống hệt nhau trừ khi dùng thuốc này. Bằng cách này, ảnh hưởng của tất cả các biến được giảm thiểu, ngoại trừ tác động của biến độc lập trong điều trị. 

Thứ hai, sau một thời gian dùng thuốc, bạn sẽ cần đo lường sự hiểu biết toán của hai nhóm bằng các chỉ số tương tự, chẳng hạn như để các tình nguyện viên làm các bài kiểm tra tương tự sau khi học một công thức toán mới. Sau đó, bạn có thể thu thập hiệu suất của họ và so sánh kết quả. Trong trường hợp này, giả thuyết null của chúng ta sẽ là không có sự khác biệt giữa hai nhóm, và sự thay thế của chúng ta sẽ là có. 

Điều này vẫn chưa hoàn toàn chính thức. Có rất nhiều chi tiết bạn phải suy nghĩ cẩn thận. Ví dụ, các số liệu phù hợp để kiểm tra khả năng hiểu biết toán học của họ là gì? Có bao nhiêu tình nguyện viên cho bài kiểm tra của bạn để bạn có thể tự tin để khẳng định hiệu quả của thuốc của bạn? Bạn nên chạy bài kiểm tra trong bao lâu? Làm thế nào để bạn quyết định nếu có sự khác biệt giữa hai nhóm? Bạn có quan tâm đến hiệu suất trung bình chỉ, hoặc cũng là phạm vi biến thể của điểm số? Và như vậy. 

Bằng cách này, thử nghiệm giả thuyết cung cấp một khuôn khổ cho thiết kế thực nghiệm và lý luận về sự chắc chắn trong các kết quả quan sát được. Nếu bây giờ chúng ta có thể chỉ ra rằng giả thuyết null là rất khó có thể là đúng, chúng ta có thể từ chối nó với sự tự tin. 

Để hoàn thành câu chuyện về cách làm việc với thử nghiệm giả thuyết, bây giờ chúng ta cần giới thiệu một số thuật ngữ bổ sung và đưa ra một số khái niệm của chúng tôi ở trên chính thức. 

### Statistical Significance

The *statistical significance* measures the probability of erroneously rejecting the null hypothesis, $H_0$, when it should not be rejected, i.e.,

$$ \text{statistical significance }= 1 - \alpha = 1 - P(\text{reject } H_0 \mid H_0 \text{ is true} ).$$

It is also referred to as the *type I error* or *false positive*. The $\alpha$, is called as the *significance level* and its commonly used value is $5\%$, i.e., $1-\alpha = 95\%$. The significance level can be explained as the level of risk that we are willing to take, when we reject a true null hypothesis.

:numref:`fig_statistical_significance` shows the observations' values and probability of a given normal distribution in a two-sample hypothesis test. If the observation data example is located outsides the $95\%$ threshold, it will be a very unlikely observation under the null hypothesis assumption. Hence, there might be something wrong with the null hypothesis and we will reject it.

![Statistical significance.](../img/statistical-significance.svg)
:label:`fig_statistical_significance`

### Sức mạnh thống kê

The *statistical power* (or *sensitivity*) measures the probability of reject the null hypothesis, $H_0$, when it should be rejected, i.e.,

$$ \text{statistical power }= 1 - \beta = 1 - P(\text{ fail to reject } H_0  \mid H_0 \text{ is false} ).$$

Recall that a *type I error* is error caused by rejecting the null hypothesis when it is true, whereas a *type II error* is resulted from failing to reject the null hypothesis when it is false. A type II error is usually denoted as $\beta$, and hence the corresponding statistical power is $1-\beta$.


Intuitively, statistical power can be interpreted as how likely our test will detect a real discrepancy of some minimum magnitude at a desired statistical significance level. $80\%$ is a commonly used statistical power threshold. The higher the statistical power, the more likely we are to detect true differences.

Một trong những cách sử dụng phổ biến nhất của sức mạnh thống kê là xác định số lượng mẫu cần thiết. Xác suất bạn từ chối giả thuyết null khi nó là sai phụ thuộc vào mức độ sai (được gọi là *kích thước hiệu ứng*) và số lượng mẫu bạn có. Như bạn có thể mong đợi, kích thước hiệu ứng nhỏ sẽ yêu cầu một số lượng rất lớn các mẫu được phát hiện với xác suất cao. Trong khi vượt quá phạm vi của phụ lục ngắn gọn này để lấy ra chi tiết, như một ví dụ, muốn có thể từ chối một giả thuyết null rằng mẫu của chúng tôi đến từ một phương sai trung bình bằng 0 một Gaussian, và chúng tôi tin rằng trung bình mẫu của chúng tôi thực sự gần với một, chúng tôi có thể làm như vậy với tỷ lệ lỗi chấp nhận được với kích thước mẫu của chỉ $8$. Tuy nhiên, nếu chúng ta nghĩ rằng trung bình thực sự dân số mẫu của chúng ta gần $0.01$, thì chúng ta cần một kích thước mẫu gần $80000$ để phát hiện sự khác biệt. 

Chúng ta có thể tưởng tượng sức mạnh như một bộ lọc nước. Trong sự tương tự này, một bài kiểm tra giả thuyết công suất cao giống như một hệ thống lọc nước chất lượng cao sẽ làm giảm các chất độc hại trong nước càng nhiều càng tốt. Mặt khác, sự khác biệt nhỏ hơn giống như một bộ lọc nước chất lượng thấp, nơi một số chất nhỏ tương đối có thể dễ dàng thoát ra khỏi các khoảng trống. Tương tự, nếu sức mạnh thống kê không đủ công suất cao, thì thử nghiệm có thể không bắt được sự khác biệt nhỏ hơn. 

### Test Statistic

Một * thống kê thử nghiệm* $T(x)$ là một vô hướng tóm tắt một số đặc điểm của dữ liệu mẫu. Mục tiêu của việc xác định một thống kê như vậy là nó sẽ cho phép chúng ta phân biệt giữa các phân phối khác nhau và tiến hành kiểm tra giả thuyết của chúng tôi. Suy nghĩ lại ví dụ của nhà hóa học của chúng tôi, nếu chúng ta muốn chứng minh rằng một dân số hoạt động tốt hơn người kia, nó có thể là hợp lý để lấy trung bình như thống kê thử nghiệm. Các lựa chọn khác nhau của thống kê thử nghiệm có thể dẫn đến kiểm tra thống kê với sức mạnh thống kê khác nhau đáng kể. 

Thông thường, $T(X)$ (sự phân bố của thống kê thử nghiệm theo giả thuyết null của chúng tôi) sẽ theo sau, ít nhất là khoảng, một phân phối xác suất chung như phân phối bình thường khi được xem xét theo giả thuyết null. Nếu chúng ta có thể lấy được một phân phối rõ ràng như vậy, và sau đó đo thống kê thử nghiệm của chúng tôi về tập dữ liệu của chúng tôi, chúng ta có thể từ chối một cách an toàn giả thuyết null nếu thống kê của chúng tôi nằm ngoài phạm vi mà chúng ta mong đợi. Làm cho định lượng này dẫn chúng ta đến khái niệm $p$-giá trị. 

### $p$-value

Giá trị $p$-( hoặc giá trị xác suất *) là xác suất $T(X)$ ít nhất là cực đoan như thống kê thử nghiệm quan sát $T(x)$ giả định rằng giả thuyết null là * true*, tức là, 

$$ p\text{-value} = P_{H_0}(T(X) \geq T(x)).$$

Nếu giá trị $p$-nhỏ hơn hoặc bằng một mức ý nghĩa thống kê được xác định trước và cố định $\alpha$, chúng ta có thể từ chối giả thuyết null. Nếu không, chúng ta sẽ kết luận rằng chúng ta thiếu bằng chứng để từ chối giả thuyết null. Đối với một phân bố dân số nhất định, *khu vực từ khóa* sẽ là khoảng thời gian chứa tất cả các điểm có giá trị $p$ nhỏ hơn mức ý nghĩa thống kê $\alpha$. 

### Kiểm tra một mặt và thử nghiệm hai mặt

Thông thường có hai loại kiểm tra ý nghĩa: bài kiểm tra một mặt và thử nghiệm hai mặt. Kiểm tra * một mặt* (hoặc *kiểm tra một đuôi *) được áp dụng khi giả thuyết null và giả thuyết thay thế chỉ có một hướng. Ví dụ, giả thuyết null có thể nói rằng tham số thực $\theta$ nhỏ hơn hoặc bằng một giá trị $c$. Giả thuyết thay thế sẽ là $\theta$ lớn hơn $c$. Đó là, khu vực từ chối chỉ ở một bên của phân phối lấy mẫu. Trái ngược với thử nghiệm một mặt, thử nghiệm hai mặt* (hoặc * kiểm tra hai đuôi *) được áp dụng khi vùng từ chối nằm ở cả hai mặt của phân phối lấy mẫu. Một ví dụ trong trường hợp này có thể có một trạng thái giả thuyết null rằng tham số đúng $\theta$ bằng một giá trị $c$. Giả thuyết thay thế sẽ là $\theta$ không bằng $c$. 

### Các bước chung của thử nghiệm giả thuyết

Sau khi làm quen với các khái niệm trên, chúng ta hãy trải qua các bước chung của thử nghiệm giả thuyết. 

1. Nêu câu hỏi và thiết lập một giả thuyết vô giá trị $H_0$.
2. Đặt mức ý nghĩa thống kê $\alpha$ và một sức mạnh thống kê ($1 - \beta$).
3. Lấy mẫu thông qua các thí nghiệm. Số lượng mẫu cần thiết sẽ phụ thuộc vào sức mạnh thống kê và kích thước hiệu ứng mong đợi.
4. Tính số liệu thống kê thử nghiệm và giá trị $p$-.
5. Đưa ra quyết định giữ hoặc từ chối giả thuyết null dựa trên giá trị $p$-giá trị và mức ý nghĩa thống kê $\alpha$.

Để tiến hành một bài kiểm tra giả thuyết, chúng ta bắt đầu bằng cách xác định một giả thuyết null và mức độ rủi ro mà chúng tôi sẵn sàng thực hiện. Sau đó, chúng tôi tính toán thống kê thử nghiệm của mẫu, lấy một giá trị cực đoan của thống kê thử nghiệm làm bằng chứng chống lại giả thuyết null. Nếu thống kê thử nghiệm nằm trong khu vực từ chối, chúng ta có thể từ chối giả thuyết null có lợi cho sự thay thế. 

Xét nghiệm giả thuyết được áp dụng trong một loạt các tình huống như đường mòn lâm sàng và thử nghiệm A/B. 

## Xây dựng khoảng tự tin

Khi ước tính giá trị của một tham số $\theta$, các chứng thực điểm như $\hat \theta$ có tiện ích hạn chế vì chúng không chứa khái niệm về sự không chắc chắn. Thay vào đó, sẽ tốt hơn nhiều nếu chúng ta có thể tạo ra một khoảng thời gian có chứa tham số thực $\theta$ với xác suất cao. Nếu bạn quan tâm đến những ý tưởng như vậy một thế kỷ trước, thì bạn sẽ rất vui mừng khi đọc “Phác thảo của một lý thuyết ước tính thống kê dựa trên lý thuyết xác suất cổ điển” của Jerzy Neyman :cite:`Neyman.1937`, người lần đầu tiên giới thiệu khái niệm khoảng thời gian tin cậy vào năm 1937. 

Để hữu ích, một khoảng thời gian tin cậy nên càng nhỏ càng tốt cho một mức độ chắc chắn nhất định. Hãy để chúng tôi xem làm thế nào để lấy được nó. 

### Definition

Mathematically, a *confidence interval* for the true parameter $\theta$ is an interval $C_n$ that computed from the sample data such that

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta.$$
:eqlabel:`eq_confidence`

Here $\alpha \in (0, 1)$, and $1 - \alpha$ is called the *confidence level* or *coverage* of the interval. This is the same $\alpha$ as the significance level as we discussed about above.

Note that :eqref:`eq_confidence` is about variable $C_n$, not about the fixed $\theta$. To emphasize this, we write $P_{\theta} (C_n \ni \theta)$ rather than $P_{\theta} (\theta \in C_n)$.

### Interpretation

It is very tempting to interpret a $95\%$ confidence interval as an interval where you can be $95\%$ sure the true parameter lies, however this is sadly not true.  The true parameter is fixed, and it is the interval that is random.  Thus a better interpretation would be to say that if you generated a large number of confidence intervals by this procedure, $95\%$ of the generated intervals would contain the true parameter.

This may seem pedantic, but it can have real implications for the interpretation of the results.  In particular, we may satisfy :eqref:`eq_confidence` by constructing intervals that we are *almost certain* do not contain the true value, as long as we only do so rarely enough.  We close this section by providing three tempting but false statements.  An in-depth discussion of these points can be found in :cite:`Morey.Hoekstra.Rouder.ea.2016`.

* **Fallacy 1**. Narrow confidence intervals mean we can estimate the parameter precisely.
* **Fallacy 2**. The values inside the confidence interval are more likely to be the true value than those outside the interval.
* **Fallacy 3**. The probability that a particular observed $95\%$ confidence interval contains the true value is $95\%$.

Sufficed to say, confidence intervals are subtle objects.  However, if you keep the interpretation clear, they can be powerful tools.

### A Gaussian Example

Let us discuss the most classical example, the confidence interval for the mean of a Gaussian of unknown mean and variance.  Suppose we collect $n$ samples $\{x_i\}_{i=1}^n$ from our Gaussian $\mathcal{N}(\mu, \sigma^2)$.  We can compute estimators for the mean and standard deviation by taking

$$\hat\mu_n = \frac{1}{n}\sum_{i=1}^n x_i \;\text{and}\; \hat\sigma^2_n = \frac{1}{n-1}\sum_{i=1}^n (x_i - \hat\mu)^2.$$

If we now consider the random variable

$$
T = \frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}},
$$

we obtain a random variable following a well-known distribution called the *Student's t-distribution on* $n-1$ *degrees of freedom*.

This distribution is very well studied, and it is known, for instance, that as $n\rightarrow \infty$, it is approximately a standard Gaussian, and thus by looking up values of the Gaussian c.d.f. in a table, we may conclude that the value of $T$ is in the interval $[-1.96, 1.96]$ at least $95\%$ of the time.  For finite values of $n$, the interval needs to be somewhat larger, but are well known and precomputed in tables.

Thus, we may conclude that for large $n$,

$$
P\left(\frac{\hat\mu_n - \mu}{\hat\sigma_n/\sqrt{n}} \in [-1.96, 1.96]\right) \ge 0.95.
$$

Rearranging this by multiplying both sides by $\hat\sigma_n/\sqrt{n}$ and then adding $\hat\mu_n$, we obtain

$$
P\left(\mu \in \left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right]\right) \ge 0.95.
$$

Thus we know that we have found our $95\%$ confidence interval:
$$\left[\hat\mu_n - 1.96\frac{\hat\sigma_n}{\sqrt{n}}, \hat\mu_n + 1.96\frac{\hat\sigma_n}{\sqrt{n}}\right].$$
:eqlabel:`eq_gauss_confidence`

It is safe to say that :eqref:`eq_gauss_confidence` is one of the most used formula in statistics.  Let us close our discussion of statistics by implementing it.  For simplicity, we assume we are in the asymptotic regime.  Small values of $N$ should include the correct value of `t_star` obtained either programmatically or from a $t$-table.

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

* Thống kê tập trung vào các bài toán suy luận, trong khi học sâu nhấn mạnh vào việc đưa ra các dự đoán chính xác mà không cần lập trình và hiểu biết rõ ràng.
* Có ba phương pháp suy luận thống kê phổ biến: đánh giá và so sánh các ước lượng, tiến hành các bài kiểm tra giả thuyết, và xây dựng các khoảng tự tin.
* Có ba điều kiện phổ biến nhất: thiên vị thống kê, độ lệch chuẩn và sai số vuông trung bình.
* Khoảng thời gian tin cậy là một phạm vi ước tính của một tham số dân số thực sự mà chúng ta có thể xây dựng bằng cách đưa ra các mẫu.
* Xét nghiệm giả thuyết là một cách để đánh giá một số bằng chứng chống lại tuyên bố mặc định về một dân số.

## Bài tập

1. Hãy để $X_1, X_2, \ldots, X_n \overset{\text{iid}}{\sim} \mathrm{Unif}(0, \theta)$, trong đó “iid” là viết tắt của * độc lập và phân phối giống hệt nhau*. Hãy xem xét các chứng thực sau đây của $\theta$:
$$\hat{\theta} = \max \{X_1, X_2, \ldots, X_n \};$$ $$\tilde{\theta} = 2 \bar{X_n} = \frac{2}{n} \sum_{i=1}^n X_i.$$
    * Tìm thiên vị thống kê, độ lệch chuẩn và sai số vuông trung bình của $\hat{\theta}.$
    * Tìm thiên vị thống kê, độ lệch chuẩn và sai số vuông trung bình của $\tilde{\theta}.$
    * Ước tính nào tốt hơn?
1. Đối với ví dụ nhà hóa học của chúng tôi trong phần giới thiệu, bạn có thể lấy được 5 bước để tiến hành thử nghiệm giả thuyết hai mặt không? Với mức ý nghĩa thống kê $\alpha = 0.05$ và sức mạnh thống kê $1 - \beta = 0.8$.
1. Chạy mã khoảng thời gian tin cậy với $N=2$ và $\alpha = 0.5$ cho $100$ tập dữ liệu được tạo độc lập và vẽ các khoảng thời gian kết quả (trong trường hợp này là `t_star = 1.0`). Bạn sẽ thấy một số khoảng thời gian rất ngắn rất xa chứa trung bình thực sự $0$. Điều này có mâu thuẫn với việc giải thích khoảng thời gian tin cậy không? Bạn có cảm thấy thoải mái khi sử dụng khoảng thời gian ngắn để chỉ ra ước tính chính xác cao không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/419)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1102)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1103)
:end_tab:

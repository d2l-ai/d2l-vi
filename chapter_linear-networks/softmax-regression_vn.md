<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Softmax Regression
-->

# Hồi quy Softmax
:label:`sec_softmax`

<!--
In :numref:`sec_linear_regression`, we introduced linear regression, working through implementations from scratch in :numref:`sec_linear_scratch` 
and again using Gluon in :numref:`sec_linear_gluon` to do the heavy lifting.
-->

Trong :numref:`sec_linear_regression`, chúng ta đã giới thiệu về hồi quy tuyến tính, tự xây dựng mô hình hồi quy tuyến tính từ đầu trong :numref:`sec_linear_scratch` và xây dựng mô hình hồi quy tuyến tính một lần nữa sử dụng Gluon trong :numref:`sec_linear_gluon` để thực hiện phần việc nặng nhọc.

<!--
Regression is the hammer we reach for when we want to answer *how much?* or *how many?* questions.
If you want to predict the number of dollars (the *price*) at which a house will be sold, or the number of wins a baseball team might have,
or the number of days that a patient will remain hospitalized before being discharged, then you are probably looking for a regression model.
-->

Hồi quy là công cụ đắc lực có thể sử dụng khi ta muốn trả lời câu hỏi *bao nhiêu?*.
Nếu bạn muốn dự đoán một ngôi nhà sẽ được bán với giá bao nhiêu tiền (*Đô la*), hay số trận thắng mà một đội bóng có thể đạt được, hoặc số ngày một bệnh nhân phải điều trị nội trú trước khi được xuất viện, thì có lẽ bạn đang cần một mô hình hồi quy.

<!--
In practice, we are more often interested in classification: asking not *how much?* but *which one?*
-->

Trong thực tế, chúng ta thường quan tâm đến việc phân loại hơn: không phải câu hỏi *bao nhiêu?* mà là *loại nào?*

<!--
* Does this email belong in the spam folder or the inbox*?
* Is this customer more likely *to sign up* or *not to sign up* for a subscription service?*
* Does this image depict a donkey, a dog, a cat, or a rooster?
* Which movie is Aston most likely to watch next?
-->

* Email này có phải thư rác/lừa đảo hay không?
* Khách hàng này nhiều khả năng *đăng ký* hay *không đăng ký* một dịch vụ thuê bao?
* Hình ảnh này mô tả một con lừa, một con cún, một con mèo hay một con gà trống?
* Aston có khả năng xem bộ phim nào tiếp theo nhất?

<!--
Colloquially, machine learning practitioners overload the word *classification* to describe two subtly different problems:
(i) those where we are interested only in *hard* assignments of examples to categories; and (ii) those where we wish to make *soft assignments*,
i.e., to assess the *probability* that each category applies.
The distinction tends to get blurred, in part, because often, even when we only care about hard assignments, we still use models that make soft assignments.
-->

Thông thường, những người làm về học máy dùng từ *phân loại* để mô tả hai bài toán khác nhau đôi chút:
(i) ta chỉ quan tâm đến việc gán *cứng* một danh mục cho mỗi ví dụ: là cún, là gà, hay là mèo?; và (ii) ta muốn *gán mềm* tất cả các danh mục cho mỗi ví dụ, tức đánh giá *xác suất* một ví dụ rơi vào từng danh mục khả dĩ: là cún (92%), là gà (1%), là mèo (7%).
Sự khác biệt này thường không rõ ràng, một phần bởi vì thông thường, ngay cả khi chúng ta chỉ quan tâm đến việc gán cứng, chúng ta vẫn sử dụng các mô hình có khả năng thực hiện các phép gán mềm.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Classification Problems
-->

## Bài toán Phân loại

<!--
To get our feet wet, let's start off with a simple image classification problem.
Here, each input consists of a $2\times2$ grayscale image.
We can represent each pixel value with a single scalar, giving us four features $x_1, x_2, x_3, x_4$.
Further, let's assume that each image belongs to one among the categories "cat", "chicken" and "dog".
-->

Để khởi động, ta hãy bắt đầu với bài toán phân loại hình ảnh đơn giản.
Ở đây, mỗi đầu vào bao gồm một ảnh xám $2\times2$.
Bằng cách biểu diễn mỗi giá trị điểm ảnh bởi một số vô hướng, ta thu được bốn đặc trưng $x_1, x_2, x_3, x_4$.
Hơn nữa, hãy giả rử rằng mỗi hình ảnh đều thuộc về một trong các danh mục "mèo", "gà" và "chó".

<!--
Next, we have to choose how to represent the labels.
We have two obvious choices.
Perhaps the most natural impulse would be to choose $y \in \{1, 2, 3\}$, where the integers represent {dog, cat, chicken} respectively.
This is a great way of *storing* such information on a computer.
If the categories had some natural ordering among them, say if we were trying to predict {baby, toddler, adolescent, young adult, adult, geriatric},
then it might even make sense to cast this problem as regression and keep the labels in this format.
-->

Tiếp theo, ta cần phải chọn cách biểu diễn nhãn. 
Ta có hai cách làm hiển nhiên.
Cách tự nhiên nhất có lẽ là chọn $y \in \{1, 2, 3\}$ lần lượt ứng với {chó, mèo,  gà}.
Đây là một cách *lưu trữ* thông tin tuyệt vời trên máy tính.
Nếu các danh mục có một thứ tự tự nhiên giữa chúng, chẳng hạn như {trẻ sơ sinh, trẻ tập đi, thiếu niên, thanh niên, người trưởng thành, người cao tuổi}, sẽ là tự nhiên hơn nếu coi bài toán này là một bài toán hồi quy và nhãn sẽ được giữ nguyên dưới dạng số.

<!--
But general classification problems do not come with natural orderings among the classes.
Fortunately, statisticians long ago invented a simple way to represent categorical data: the *one hot encoding*.
A one-hot encoding is a vector with as many components as we have categories.
The component corresponding to particular instance's category is set to 1 and all other components are set to 0.
-->

Nhưng nhìn chung các bài toán phân loại không có các lớp tuân theo một trật tự tự nhiên nào.
May mắn thay, các nhà thông kê từ lâu đã tìm ra một cách đơn giản để có thể biểu diễn dữ liệu danh mục: *biểu diễn One-hot*.
Biểu diễn One-hot là một vector với số lượng thành phần bằng số danh mục mà ta có.
Thành phần tương ứng với từng danh mục cụ thể sẽ được gán là 1 và tất cả các thành phần khác sẽ được gán là 0.

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

<!--
In our case, $y$ would be a three-dimensional vector, with $(1, 0, 0)$ corresponding to "cat", $(0, 1, 0)$ to "chicken" and $(0, 0, 1)$ to "dog".
-->

Trong trường hợp này, $y$ sẽ là một vector 3 chiều, với $(1, 0, 0)$ tương ứng với "mèo", $(0, 1, 0)$ ứng với "gà" và $(0, 0, 1)$ ứng với "chó".

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Network Architecture
-->

### *dịch tiêu đề phía trên*

<!--
In order to estimate the conditional probabilities associated with each classes, we need a model with multiple outputs, one per class.
To address classification with linear models, we will need as many linear functions as we have outputs.
Each output will correspond to its own linear function.
In our case, since we have 4 features and 3 possible output categories, we will need 12 scalars to represent the weights, 
($w$ with subscripts) and 3 scalars to represent the biases ($b$ with subscripts).
We compute these three *logits*, $o_1, o_2$, and $o_3$, for each input:
-->

*dịch đoạn phía trên*

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

<!--
We can depict this calculation with the neural network diagram shown in :numref:`fig_softmaxreg`.
Just as in linear regression, softmax regression is also a single-layer neural network.
And since the calculation of each output, $o_1, o_2$, and $o_3$, depends on all inputs, $x_1$, $x_2$, $x_3$, and $x_4$,
the output layer of softmax regression can also be described as fully-connected layer.
-->

*dịch đoạn phía trên*

<!--
![Softmax regression is a single-layer neural network.  ](../img/softmaxreg.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

<!--
To express the model more compactly, we can use linear algebra notation.
In vector form, we arrive at $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$, a form better suited both for mathematics, and for writing code.
Note that we have gathered all of our weights into a $3\times4$ matrix and that for a given example $\mathbf{x}$, 
our outputs are given by a matrix-vector product of our weights by our inputs plus our biases $\mathbf{b}$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### Softmax Operation
-->

### *dịch tiêu đề phía trên*

<!--
The main approach that we are going to take here is to interpret the outputs of our model as probabilities.
We will optimize our parameters to produce probabilities that maximize the likelihood of the observed data.
Then, to generate predictions, we will set a threshold, for example, choosing the *argmax* of the predicted probabilities.
-->

*dịch đoạn phía trên*

<!--
Put formally, we would like outputs $\hat{y}_k$ that we can interpret as the probability that a given item belongs to class $k$.
Then we can choose the class with the largest output value as our prediction $\operatorname*{argmax}_k y_k$.
For example, if $\hat{y}_1$, $\hat{y}_2$, and $\hat{y}_3$ are $0.1$, $.8$, and $0.1$, respectively, then we predict category $2$, which (in our example) represents "chicken".
-->

*dịch đoạn phía trên*

<!--
You might be tempted to suggest that we interpret the logits $o$ directly as our outputs of interest.
However, there are some problems with directly interpreting the output of the linear layer as a probability.
Nothing constrains these numbers to sum to 1.
Moreover, depending on the inputs, they can take negative values.
These violate basic axioms of probability presented in :numref:`sec_prob`
-->

*dịch đoạn phía trên*

<!--
To interpret our outputs as probabilities, we must guarantee that (even on new data), they will be nonnegative and sum up to 1.
Moreover, we need a training objective that encourages the model to estimate faithfully *probabilities*.
Of all instances when a classifier outputs $.5$, we hope that half of those examples will *actually* belong to the predicted class.
This is a property called *calibration*.
-->

*dịch đoạn phía trên*

<!--
The *softmax function*, invented in 1959 by the social scientist R Duncan Luce in the context of *choice models* does precisely this.
To transform our logits such that they become nonnegative and sum to $1$, while requiring that the model remains differentiable, 
we first exponentiate each logit (ensuring non-negativity) and then divide by their sum (ensuring that they sum to $1$).
-->

*dịch đoạn phía trên*

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad
\hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}.
$$

<!--
It is easy to see $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ with $0 \leq \hat{y}_i \leq 1$ for all $i$.
Thus, $\hat{y}$ is a proper probability distribution and the values of $\hat{\mathbf{y}}$ can be interpreted accordingly.
Note that the softmax operation does not change the ordering among the logits, and thus we can still pick out the most likely class by:
-->

*dịch đoạn phía trên*

$$
\hat{\imath}(\mathbf{o}) = \operatorname*{argmax}_i o_i = \operatorname*{argmax}_i \hat y_i.
$$

<!--
The logits $\mathbf{o}$ then are simply the pre-softmax values that determining the probabilities assigned to each category.
Summarizing it all in vector notation we get ${\mathbf{o}}^{(i)} = \mathbf{W} {\mathbf{x}}^{(i)} + {\mathbf{b}}$, where ${\hat{\mathbf{y}}}^{(i)} = \mathrm{softmax}({\mathbf{o}}^{(i)})$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->


<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Vectorization for Minibatches
-->

### *dịch tiêu đề phía trên*

<!--
To improve computational efficiency and take advantage of GPUs, we typically carry out vector calculations for minibatches of data.
Assume that we are given a minibatch $\mathbf{X}$ of examples with dimensionality $d$ and batch size $n$.
Moreover, assume that we have $q$ categories (outputs).
Then the minibatch features $\mathbf{X}$ are in $\mathbb{R}^{n \times d}$, weights $\mathbf{W} \in \mathbb{R}^{d \times q}$, and the bias satisfies $\mathbf{b} \in \mathbb{R}^q$.
-->

*dịch đoạn phía trên*

$$
\begin{aligned}
\mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\
\hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}).
\end{aligned}
$$

<!--
This accelerates the dominant operation into a matrix-matrix product $\mathbf{W} \mathbf{X}$ vs the matrix-vector products we would be executing if we processed one example at a time.
The softmax itself can be computed by exponentiating all entries in $\mathbf{O}$ and then normalizing them by the sum.
-->

*dịch đoạn phía trên*

<!--
## Loss Function
-->

## *dịch tiêu đề phía trên*
:label:`section_cross_entropy`

<!--
Next, we need a *loss function* to measure the quality of our predicted probabilities.
We will rely on *likelihood maximization*, the very same concept that we encountered when providing a probabilistic justification for the least squares objective in linear regression 
(:numref:`sec_linear_regression`).
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
### Log-Likelihood
-->

### *dịch tiêu đề phía trên*

<!--
The softmax function gives us a vector $\hat{\mathbf{y}}$, which we can interpret as estimated conditional probabilities of each class given the input $x$, 
e.g., $\hat{y}_1$ = $\hat{P}(y=\mathrm{cat} \mid \mathbf{x})$.
We can compare the estimates with reality by checking how probable the *actual* classes are according to our model, given the features.
-->

*dịch đoạn phía trên*

$$
P(Y \mid X) = \prod_{i=1}^n P(y^{(i)} \mid x^{(i)})
\text{ and thus }
-\log P(Y \mid X) = \sum_{i=1}^n -\log P(y^{(i)} \mid x^{(i)}).
$$


<!--
Maximizing $P(Y \mid X)$ (and thus equivalently minimizing $-\log P(Y \mid X)$) corresponds to predicting the label well.
This yields the loss function (we dropped the superscript $(i)$ to avoid notation clutter):
-->

*dịch đoạn phía trên*

$$
l = -\log P(y \mid x) = - \sum_j y_j \log \hat{y}_j.
$$

<!--
For reasons explained later on, this loss function is commonly called the *cross-entropy* loss.
Here, we used that by construction $\hat{y}$ is a discrete probability distribution and that the vector $\mathbf{y}$ is a one-hot vector.
Hence the the sum over all coordinates $j$ vanishes for all but one term.
Since all $\hat{y}_j$ are probabilities, their logarithm is never larger than $0$.
Consequently, the loss function cannot be minimized any further if we correctly predict $y$ with *certainty*, i.e., if $P(y \mid x) = 1$ for the correct label.
Note that this is often not possible.
For example, there might be label noise in the dataset (some examples may be mislabeled).
It may also not be possible when the input features are not sufficiently informative to classify every example perfectly.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Softmax and Derivatives
-->

### *dịch tiêu đề phía trên*

<!--
Since the softmax and the corresponding loss are so common, it is worth while understanding a bit better how it is computed.
Plugging $o$ into the definition of the loss $l$ and using the definition of the softmax we obtain:
-->

*dịch đoạn phía trên*

$$
l = -\sum_j y_j \log \hat{y}_j = \sum_j y_j \log \sum_k \exp(o_k) - \sum_j y_j o_j
= \log \sum_k \exp(o_k) - \sum_j y_j o_j.
$$

<!--
To understand a bit better what is going on, consider the derivative with respect to $o$. We get
-->

*dịch đoạn phía trên*

$$
\partial_{o_j} l = \frac{\exp(o_j)}{\sum_k \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j = P(y = j \mid x) - y_j.
$$

<!--
In other words, the gradient is the difference between the probability assigned to the true class by our model, as expressed by the probability $P(y \mid x)$, and what actually happened, as expressed by $y$.
In this sense, it is very similar to what we saw in regression, where the gradient was the difference between the observation $y$ and estimate $\hat{y}$. This is not coincidence.
In any [exponential family](https://en.wikipedia.org/wiki/Exponential_family) model, the gradients of the log-likelihood are given by precisely this term.
This fact makes computing gradients easy in practice.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
### Cross-Entropy Loss
-->

### *dịch tiêu đề phía trên*

<!--
Now consider the case where we observe not just a single outcome but an entire distribution over outcomes.
We can use the same representation as before for $y$.
The only difference is that rather than a vector containing only binary entries, say $(0, 0, 1)$, we now have a generic probability vector, say $(0.1, 0.2, 0.7)$.
The math that we used previously to define the loss $l$ still works out fine, just that the interpretation is slightly more general.
It is the expected value of the loss for a distribution over labels.
-->

*dịch đoạn phía trên*

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_j y_j \log \hat{y}_j.
$$

<!--
This loss is called the cross-entropy loss and it is one of the most commonly used losses for multiclass classification.
We can demystify the name by introducing the basics of information theory.
-->

*dịch đoạn phía trên*

<!--
## Information Theory Basics
-->

## *dịch tiêu đề phía trên*

<!--
Information theory deals with the problem of encoding, decoding, transmitting and manipulating information (also known as data) in as concise form as possible.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!--
### Entropy
-->

### *dịch tiêu đề phía trên*

<!--
The central idea in information theory is to quantify the information content in data.
This quantity places a hard limit on our ability to compress the data.
In information theory, this quantity is called the [entropy](https://en.wikipedia.org/wiki/Entropy) of a distribution $p$, and it is captured by the following equation:
-->

*dịch đoạn phía trên*

$$
H[p] = \sum_j - p(j) \log p(j).
$$

<!--
One of the fundamental theorems of information theory states that in order to encode data drawn randomly from the distribution $p$, we need at least $H[p]$ "nats" to encode it.
If you wonder what a "nat" is, it is the equivalent of bit but when using a code with base $e$ rather than one with base 2.
One nat is $\frac{1}{\log(2)} \approx 1.44$ bit. 
$H[p] / 2$ is often also called the binary entropy.


<!--
### Surprisal
-->

### *dịch tiêu đề phía trên*

<!--
You might be wondering what compression has to do with prediction.
Imagine that we have a stream of data that we want to compress.
If it is always easy for us to predict the next token, then this data is easy to compress!
Take the extreme example where every token in the stream always takes the same value.
That is a very boring data stream!
And not only is it boring, but it is easy to predict.
Because they are always the same, we do not have to transmit any information to communicate the contents of the stream.
Easy to predict, easy to compress.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 10 ===================== -->

<!--
However if we cannot perfectly predict every event, then we might some times be surprised.
Our surprise is greater when we assigned an event lower probability.
For reasons that we will elaborate in the appendix,
Claude Shannon settled on $\log(1/p(j)) = -\log p(j)$ to quantify one's *surprisal* at observing an event $j$ having assigned it a (subjective) probability $p(j)$.
The entropy is then the *expected surprisal* when one assigned the correct probabilities (that truly match the data-generating process).
The entropy of the data is then the least surprised that one can ever be (in expectation).
-->

*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
### Cross-Entropy Revisited
-->

### *dịch tiêu đề phía trên*

<!--
So if entropy is level of surprise experienced by someone who knows the true probability, then you might be wondering, *what is cross-entropy?*
The cross-entropy *from $p$ to $q$*, denoted H(p, q), is the expected surprisal of an observer with subjective probabilities $q$ upon seeing data that was actually generated according to probabilities $p$.
The lowest possible cross-entropy is achieved when $p=q$.
In this case, the cross-entropy from $p$ to $q$ is $H(p, p)= H(p)$.
Relating this back to our classification objective, even if we get the best possible predictions, if the best possible possible, then we will never be perfect.
Our loss is lower-bounded by the entropy given by the actual conditional distributions $P(\mathbf{y} \mid \mathbf{x})$.
-->

*dịch đoạn phía trên*


<!--
### Kullback Leibler Divergence
-->

### *dịch tiêu đề phía trên*

<!--
Perhaps the most common way to measure the distance between two distributions is to calculate the *Kullback Leibler divergence* $D(p\|q)$.
This is simply the difference between the cross-entropy and the entropy, i.e., the additional cross-entropy incurred over the irreducible minimum value it could take:
-->

*dịch đoạn phía trên*

$$
D(p\|q) = H(p, q) - H[p] = \sum_j p(j) \log \frac{p(j)}{q(j)}.
$$

<!--
Note that in classification, we do not know the true $p$, so we cannot compute the entropy directly.
However, because the entropy is out of our control, minimizing $D(p\|q)$ with respect to $q$ is equivalent to minimizing the cross-entropy loss.
-->

*dịch đoạn phía trên*

<!--
In short, we can think of the cross-entropy classification objective in two ways: (i) as maximizing the likelihood of the observed data; 
and (ii) as minimizing our surprise (and thus the number of bits) required to communicate the labels.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 10 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 11 ===================== -->

<!--
## Model Prediction and Evaluation
-->

## *dịch tiêu đề phía trên*

<!--
After training the softmax regression model, given any example features,
we can predict the probability of each output category.
Normally, we use the category with the highest predicted probability as the output category. The prediction is correct if it is consistent with the actual category (label).
In the next part of the experiment, we will use accuracy to evaluate the model’s performance.
This is equal to the ratio between the number of correct predictions a nd the total number of predictions.
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* We introduced the softmax operation which takes a vector maps it into probabilities.
* Softmax regression applies to classification problems. It uses the probability distribution of the output category in the softmax operation.
* cross-entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Show that the Kullback-Leibler divergence $D(p\|q)$ is nonnegative for all distributions $p$ and $q$. Hint: use Jensen's inequality, i.e., use the fact that $-\log x$ is a convex function.
2. Show that $\log \sum_j \exp(o_j)$ is a convex function in $o$.
3. We can explore the connection between exponential families and the softmax in some more depth
    * Compute the second derivative of the cross-entropy loss $l(y,\hat{y})$ for the softmax.
    * Compute the variance of the distribution given by $\mathrm{softmax}(o)$ and show that it matches the second derivative computed above.
4. Assume that we three classes which occur with equal probability, i.e., the probability vector is $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    * What is the problem if we try to design a binary code for it? Can we match the entropy lower bound on the number of bits?
    * Can you design a better code. Hint: what happens if we try to encode two independent observations? What if we encode $n$ observations jointly?
5. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    * Prove that $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    * Prove that this holds for $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, provided that $\lambda > 0$.
    * Show that for $\lambda \to \infty$ we have $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    * What does the soft-min look like?
    * Extend this to more than two numbers.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 11 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2334)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2334)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Trần Thị Hồng Hạnh

<!-- Phần 2 -->
* Lý Phi Long
* Lê Khắc Hồng Phúc

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*

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

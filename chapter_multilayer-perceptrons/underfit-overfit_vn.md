<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Model Selection, Underfitting and Overfitting
-->

# Lựa Chọn Mô Hình, Dưới Khớp và Quá Khớp
:label:`sec_model_selection`

<!--
As machine learning scientists, our goal is to discover *patterns*.
But how can we be sure that we have truly discovered a *general* pattern and not simply memorized our data.
For example, imagine that we wanted to hunt for patterns among genetic markers linking patients to their dementia status, 
(let's the labels are drawn from the set {*dementia*, *mild cognitive impairment*, *healthy*}).
Because each person's genes identify them uniquely (ignoring identical siblings), it's possible to memorize the entire dataset.
-->

Là những nhà khoa học học máy, mục tiêu của chúng ta đó là khám phá ra các *khuôn mẫu*.
Nhưng làm sao chúng ta có thể chắc chắn rằng chúng ta đã thực sự khám phá ra một mẫu mà nó *phổ quát* và không chỉ đơn giản là ghi nhớ các dữ liệu của chúng ta.
Ví dụ, thử tưởng tượng rằng chúng ta muốn săn lùng các mẫu trong số các dấu hiệu di truyền liên kết các bệnh nhân và tình trạng mất trí của họ,
(hãy để các nhãn được trích ra từ bộ {*mất trí nhớ*, *suy giảm nhận thức mức độ nhẹ*, *khỏe mạnh*}).
Bởi vì các gene của mỗi người định dạng họ theo cách độc nhất vô nhị (bỏ qua các cặp song sinh giống hệt nhau), nên hoàn toàn có thể để ghi nhớ toàn bộ tập dữ liệu.

<!--
We don't want our model to say *"That's Bob! I remember him! He has dementia!*
The reason why is simple.
When we deploy the model in the future, we will encounter patients that the model has never seen before.
Our predictions will only be useful if our model has truly discovered a *general* pattern.
-->

Chúng ta không muốn mô hình của chúng ta nói rằng *"Bob kìa! Tôi nhớ anh ta! Anh ta bị mất trí nhớ!*
Lý do đơn giản.
Khi chúng ta triển khai mô hình trong tương lai, chúng ta sẽ chạm trán các bệnh nhân mà mô hình của chúng ta chưa bao giờ gặp trước đấy.
Sự dự toán của chúng ta sẽ chỉ có ích khi mô hình của chúng ta thực sự khám phá ra một mẫu *phổ quát*.

<!--
To recapitulate more formally, our goal is to discover patterns that capture regularities in the underlying population from which our training set was drawn.
If we are successfull in this endeavor, then we could successfully assess risk even for individuals that we have never encountered before.
This problem---how to discover patterns that *generalize*---is the fundamental problem of machine learning.
-->

Để tóm tắt một cách chính thức hơn, mục tiêu của chúng ta là khám phá các mẫu mà chúng nắm bắt được các quy tắc trong tập tổng thể nền tảng mà từ đó tập huấn luyện của chúng ta đã được trích ra. 
Nếu chúng ta thành công trong nỗ lực này, thì chúng ta có thể đánh giá thành công rủi ro ngay cả đối với các cá nhân mà chúng ta chưa bao giờ gặp phải trước đây.
Vấn đề này---làm cách nào để khám phá ra các mẫu mà *phổ quát hóa*---là vấn đề nền tảng của học máy.

<!--
The danger is that when we train models, we access just a small sample of data.
The largest public image datasets contain roughly one million images.
More often, we must learn from only thousands or tens of thousands of data points.
In a large hospital system, we might access hundreds of thousands of medical records.
When working with finite samples, we run the risk that we might discover *apparent* associations that turn out not to hold up when we collect more data.
-->

Nguy hiểm là khi huấn luyện các mô hình, chúng ta chỉ truy cập một tập dữ liệu nhỏ.
Các tệp dữ liệu hình ảnh công cộng lớn nhất chứa khoảng một triệu hình ảnh.
Thông thường hơn, chúng ta phải học chỉ từ hàng ngàn hoặc hàng chục ngàn điểm dữ liệu.
Trong một hệ thống bệnh viện lớn, chúng ta có thể truy cập hàng trăm ngàn hồ sơ y tế.
Khi làm việc với các mẫu hữu hạn, chúng ta gặp phải rủi ro mà chúng ta có thể khám phá các sự kết hợp *rõ ràng* mà hóa ra không giữ được khi chúng ta thu thập thêm dữ liệu.

<!--
The phenomena of fitting our training data more closely than we fit the underlying distribution is called overfitting, and the techniques used to combat overfitting are called regularization.
In the previous sections, you might have observed this effect while experimenting with the Fashion-MNIST dataset.
If you altered the model structure or the hyper-parameters during the experiment, 
you might have noticed that with enough nodes, layers, and training epochs, the model can eventually reach perfect accuracy on the training set, even as the accuracy on test data deteriorates.
-->

Hiện tượng việc mô hình khớp dữ liệu huấn luyện chặt chẽ hơn nhiều so với khớp phân phối nền tảng được gọi là quá khớp, và kỹ thuật sử dụng để chống lại quá khớp được gọi là điều chuẩn.
Trong các phần trước, bạn có thể đã quan sát hiệu ứng này khi thử nghiệm với tập dữ liệu Fashion-MNIST.
Nếu bạn đã sửa đổi cấu trúc mô hình hoặc siêu tham số trong suốt quá trình thử nghiệm, bạn có thể đã nhận ra rằng với đủ các nút, các tầng, và các epoch huấn luyện, mô hình ấy có thể cuối cùng cũng đạt đến sự chính xác hoàn hảo trên tập huấn luyện, ngay cả khi độ chính xác trên dữ liệu thử nghiệm giảm đi.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->


<!--
## Training Error and Generalization Error
-->

## *dịch tiêu đề phía trên*

<!--
In order to discuss this phenomenon more formally, we need to differentiate between *training error* and *generalization error*.
The training error is the error of our model as calculated on the training dataset, 
while generalization error is the expectation of our model's error 
were we to apply it to an infinite stream of additional data points drawn from the same underlying data distribution as our original sample.
-->

*dịch đoạn phía trên*

<!--
Problematically, *we can never calculate the generalization error exactly*.
That is because the imaginary stream of infinite data is an imaginary object.
In practice, we must *estimate* the generalization error by applying our model to an independent test set 
constituted of a random selection of data points that were withheld from our training set.
-->

*dịch đoạn phía trên*

<!--
The following three thought experiments will help illustrate this situation better.
Consider a college student trying to prepare for his final exam.
A diligent student will strive to practice well and test her abilities using exams from previous years.
Nonetheless, doing well on past exams is no guarantee that she will excel when it matters.
For instance, the student might try to prepare by rote learning the answers to the exam questions.
This requires the student to memorize many things.
She might even remember the answers for past exams perfectly.
Another student might prepare by trying to understand the reasons for giving certain answers.
In most cases, the latter student will do much better.
-->

*dịch đoạn phía trên*

<!--
Likewise, consider a model that simply uses a lookup table to answer questions. 
If the set of allowable inputs is discrete and reasonably small, then perhaps after viewing *many* training examples, this approach would perform well. 
Still this model has no ability to do better than random guessing when faced with examples that it has never seen before.
In reality the input spaces are far too large to memorize the answers corresponding to every conceivable input. 
For example, consider the black and white $28\times28$ images. 
If each pixel can take one among $256$ gray scale values, then there are $256^{784}$ possible images. 
That means that there are far more low-res grayscale thumbnail-sized images than there are atoms in the universe. 
Even if we could encounter this data, we could never afford to store the lookup table.
-->

*dịch đoạn phía trên*

<!--
Last, consider the problem of trying to classify the outcomes of coin tosses (class 0: heads, class 1: tails) based on some contextual features that might be available.
No matter what algorithm we come up with, because the generalization error will always be $\frac{1}{2}$.
However, for most algorithms, we should expect our training error to be considerably lower, depending on the luck of the draw, even if we did not have any features!
Consider the dataset {0, 1, 1, 1, 0, 1}.
Our feature-less would have to fall back on always predicting the *majority class*, which appears from our limited sample to be *1*.
In this case, the model that always predicts class 1 will incur an error of $\frac{1}{3}$, considerably better than our generalization error.
As we increase the amount of data, the probability that the fraction of heads will deviate significantly from $\frac{1}{2}$ diminishes, and our training error would come to match the generalization error.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Statistical Learning Theory
-->

### *dịch tiêu đề phía trên*

<!--
Since generalization is the fundamental problem in machine learning, you might not be surprised to learn 
that many mathematicians and theorists have dedicated their lives to developing formal theories to describe this phenomenon.
In their [eponymous theorem](https://en.wikipedia.org/wiki/Glivenko–Cantelli_theorem), Glivenko and Cantelli derived the rate at which the training error converges to the generalization error.
In a series of seminal papers, [Vapnik and Chervonenkis](https://en.wikipedia.org/wiki/Vapnik–Chervonenkis_theory) extended this theory to more general classes of functions.
This work laid the foundations of [Statistical Learning Theory](https://en.wikipedia.org/wiki/Statistical_learning_theory).
-->

*dịch đoạn phía trên*


<!--
In the *standard supervised learning setting*, which we have addressed up until now and will stick throughout most of this book,
we assume that both the training data and the test data are drawn *independently* from *identical* distributions (commonly called the i.i.d. assumption).
This means that the process that samples our data has no *memory*.
The $2^{\mathrm{nd}}$ example drawn and the $3^{\mathrm{rd}}$ drawn are no more correlated than the $2^{\mathrm{nd}}$ and the $2$-millionth sample drawn.
-->

*dịch đoạn phía trên*

<!--
Being a good machine learning scientist requires thinking critically, and already you should be poking holes in this assumption, coming up with common cases where the assumption fails.
What if we train a mortality risk predictor on data collected from patients at UCSF, and apply it on patients at Massachusetts General Hospital?
These distributions are simply not identical.
Moreover, draws might be correlated in time.
What if we are classifying the topics of Tweets.
The news cycle would create temporal dependencies in the topics being discussed violating any assumptions of independence.
-->

*dịch đoạn phía trên*

<!--
Sometimes we can get away with minor violations of the i.i.d. assumption and our models will continue to work remarkably well.
After all, nearly every real-world application involves at least some minor violation of the i.i.d. assumption, and yet we have useful tools for face recognition, speech recognition, language translation, etc.
-->

*dịch đoạn phía trên*

<!--
Other violations are sure to cause trouble.
Imagine, for example, if we tried to train a face recognition system by training it exclusively on university students and then want to deploy it as a tool for monitoring geriatrics in a nursing home population.
This is unlikely to work well since college students tend to look considerably different from the elderly.
-->

*dịch đoạn phía trên*

<!--
In subsequent chapters and volumes, we will discuss problems arising from violations of the i.i.d. assumption.
For now, even taking the i.i.d. assumption for granted, understanding generalization is a formidable problem.
Moreover, elucidating the precise theoretical foundations that might explain why deep neural networks generalize as well as they do continues to vexes the greatest minds in learning theory.
-->

*dịch đoạn phía trên*

<!--
When we train our models, we attempt searching for a function that fits the training data as well as possible.
If the function is so flexible that it can catch on to spurious patterns just as easily as to the true associations, 
then it might perform *too well* without producing a model that generalizes well to unseen data.
This is precisely what we want to avoid (or at least control).
Many of the techniques in deep learning are heuristics and tricks aimed at guarding against overfitting.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### Model Complexity
-->

### *dịch tiêu đề phía trên*

<!--
When we have simple models and abundant data, we expect the generalization error to resemble the training error.
When we work with more complex models and fewer examples, we expect the training error to go down but the generalization gap to grow.
What precisely constitutes model complexity is a complex matter.
Many factors govern whether a model will generalize well.
For example a model with more parameters might be considered more complex.
A model whose parameters can take a wider range of values might be more complex.
Often with neural networks, we think of a model that takes more training steps as more complex, and one subject to *early stopping* as less complex.
-->

*dịch đoạn phía trên*

<!--
It can be difficult to compare the complexity among members of substantially different model classes (say a decision tree versus a neural network).
For now, a simple rule of thumb is quite useful:
A model that can readily explain arbitrary facts is what statisticians view as complex, 
whereas one that has only a limited expressive power but still manages to explain the data well is probably closer to the truth.
In philosophy, this is closely related to Popper’s criterion of [falsifiability](https://en.wikipedia.org/wiki/Falsifiability) of a scientific theory: 
a theory is good if it fits data and if there are specific tests which can be used to disprove it.
This is important since all statistical estimation is [post hoc](https://en.wikipedia.org/wiki/Post_hoc), i.e., we estimate after we observe the facts, hence vulnerable to the associated fallacy.
For now, we will put the philosophy aside and stick to more tangible issues.
-->

*dịch đoạn phía trên*

<!--
In this section, to give you some intuition, we’ll focus on a few factors that tend to influence the generalizability of a model class:
-->

*dịch đoạn phía trên*

<!--
1. The number of tunable parameters. When the number of tunable parameters, sometimes called the *degrees of freedom*, is large, models tend to be more susceptible to overfitting.
2. The values taken by the parameters. When weights can take a wider range of values, models can be more susceptible to over fitting.
3. The number of training examples. It’s trivially easy to overfit a dataset containing only one or two examples even if your model is simple. 
But overfitting a dataset with millions of examples requires an extremely flexible model.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->


<!--
## Model Selection
-->

## *dịch tiêu đề phía trên*

<!--
In machine learning, we usually select our final model after evaluating several candidate models.
This process is called model selection.
Sometimes the models subject to comparison are fundamentally different in nature (say, decision trees vs linear models).
At other times, we are comparing members of the same class of models that have been trained with different hyperparameter settings.
-->

*dịch đoạn phía trên*

<!--
With multilayer perceptrons for example, we may wish to compare models with different numbers of hidden layers, 
different numbers of hidden units, and various choices of the activation functions applied to each hidden layer.
In order to determine the best among our candidate models, we will typically employ a validation set.
-->

*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Validation Dataset
-->

### *dịch tiêu đề phía trên*

<!--
In principle we should not touch our test set until after we have chosen all our hyper-parameters.
Were we to use the test data in the model selection process, there is a risk that we might overfit the test data.
Then we would be in serious trouble.
If we overfit our training data, there is always the evaluation on test data to keep us honest.
But if we overfit the test data, how would we ever know?
-->

*dịch đoạn phía trên*


<!--
Thus, we should never rely on the test data for model selection.
And yet we cannot rely solely on the training data for model selection either because we cannot estimate the generalization error on the very data that we use to train the model.
-->

*dịch đoạn phía trên*

<!--
The common practice to address this problem is to split our data three ways, incorporating a *validation set* in addition to the training and test sets.
-->

*dịch đoạn phía trên*


<!--
In practical applications, the picture gets muddier.
While ideally we would only touch the test data once, to assess the very best model or to compare a small number of models to each other, real-world test data is seldom discarded after just one use.
We can seldom afford a new test set for each round of experiments.
-->

*dịch đoạn phía trên*

<!--
The result is a murky practice where the boundaries between validation and test data are worryingly ambiguous.
Unless explicitly stated otherwise, in the experiments in this book we are really working with what should rightly be called training data and validation data, with no true test sets.
Therefore, the accuracy reported in each experiment is really the validation accuracy and not a true test set accuracy.
The good news is that we do not need too much data in the validation set.
The uncertainty in our estimates can be shown to be of the order of $\mathcal{O}(n^{-\frac{1}{2}})$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
### $K$-Fold Cross-Validation
-->

### *dịch tiêu đề phía trên*

<!--
When training data is scarce, we might not even be able to afford to hold out enough data to constitute a proper validation set.
One popular solution to this problem is to employ $K$*-fold cross-validation*.
Here, the original training data is split into $K$ non-overlapping subsets.
Then model training and validation are executed $K$ times, each time training on $K-1$ subsets and validating on a different subset (the one not used for training in that round).
Finally, the training and validation error rates are estimated by averaging over the results from the $K$ experiments.
-->

*dịch đoạn phía trên*


<!--
## Underfitting or Overfitting?
-->

## *dịch tiêu đề phía trên*

<!--
When we compare the training and validation errors, we want to be mindful of two common situations:
First, we want to watch out for cases when our training error and validation error are both substantial but there is a little gap between them.
If the model is unable to reduce the training error, that could mean that our model is too simple (i.e., insufficiently expressive) to capture the pattern that we are trying to model.
Moreover, since the *generalization gap* between our training and validation errors is small, we have reason to believe that we could get away with a more complex model.
This phenomenon is known as underfitting.
-->

*dịch đoạn phía trên*

<!--
On the other hand, as we discussed above, we want to watch out for the cases when our training error is significantly lower than our validation error, indicating severe overfitting.
Note that overfitting is not always a bad thing.
With deep learning especially, it is well known that the best predictive models often perform far better on training data than on holdout data.
Ultimately, we usually care more about the validation error than about the gap between the training and validation errors.
-->

*dịch đoạn phía trên*

<!--
Whether we overfit or underfit can depend both on the complexity of our model and the size of the available training datasets, two topics that we discuss below.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
### Model Complexity
-->

### *dịch tiêu đề phía trên*

<!--
To illustrate some classical intuition about overfitting and model complexity, we given an example using polynomials.
Given training data consisting of a single feature $x$ and a corresponding real-valued label $y$, we try to find the polynomial of degree $d$
-->

*dịch đoạn phía trên*

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

<!--
to estimate the labels $y$.
This is just a linear regression problem where our features are given by the powers of $x$, the $w_i$ given the model’s weights, and the bias is given by $w_0$ since $x^0 = 1$ for all $x$.
Since this is just a linear regression problem, we can use the squared error as our loss function.
-->

*dịch đoạn phía trên*


<!--
A higher-order polynomial function is more complex than a lower order polynomial function, since the higher-order polynomial has more parameters and the model function’s selection range is wider.
Fixing the training dataset, higher-order polynomial functions should always achieve lower (at worst, equal) training error relative to lower degree polynomials.
In fact, whenever the data points each have a distinct value of $x$, a polynomial function with degree equal to the number of data points can fit the training set perfectly.
We visualize the relationship between polynomial degree and under- vs over-fitting in :numref:`fig_capacity_vs_error`.
-->

*dịch đoạn phía trên*

<!--
![Influence of Model Complexity on Underfitting and Overfitting](../img/capacity_vs_error.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/capacity_vs_error.svg)
:label:`fig_capacity_vs_error`

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Dataset Size
-->

### *dịch tiêu đề phía trên*

<!--
The other big consideration to bear in mind is the dataset size.
Fixing our model, the fewer samples we have in the training dataset, the more likely (and more severely) we are to encounter overfitting.
As we increase the amount of training data, the generalization error typically decreases.
Moreover, in general, more data never hurts.
For a fixed task and data *distribution*, there is typically a relationship between model complexity and dataset size.
Given more data, we might profitably attempt to fit a more complex model.
Absent sufficient data, simpler models may be difficult to beat.
For many tasks, deep learning only outperforms linear models when many thousands of training examples are available.
In part, the current success of deep learning owes to the current abundance of massive datasets due to internet companies, cheap storage, connected devices, and the broad digitization of the economy.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
## Polynomial Regression
-->

## *dịch tiêu đề phía trên*

<!--
We can now explore these concepts interactively by fitting polynomials to data.
To get started we will import our usual packages.
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
import d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

<!--
### Generating the Dataset
-->

### *dịch tiêu đề phía trên*

<!--
First we need data. Given $x$, we will use the following cubic polynomial to generate the labels on training and test data:
-->

*dịch đoạn phía trên*

$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1).$$
-->

*dịch đoạn phía trên*

<!--
The noise term $\epsilon$ obeys a normal distribution with a mean of 0 and a standard deviation of 0.1.
We will synthesize 100 samples each for the training set and test set.
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
maxdegree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(maxdegree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
features = np.random.shuffle(features)
poly_features = np.power(features, np.arange(maxdegree).reshape(1, -1))
poly_features = poly_features / (
    npx.gamma(np.arange(maxdegree) + 1).reshape(1, -1))
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

<!--
For optimization, we typically want to avoid very large values of gradients, losses, etc.
This is why the monomials stored in `poly_features` are rescaled from $x^i$ to $\frac{1}{i!} x^i$.
It allows us to avoid very large values for large exponents $i$.
Factorials are implemented in Gluon using the Gamma function,
where $n! = \Gamma(n+1)$.
-->

*dịch đoạn phía trên*

<!--
Take a look at the first 2 samples from the generated dataset.
The value 1 is technically a feature, namely the constant feature corresponding to the bias.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
features[:2], poly_features[:2], labels[:2]
```

<!--
### Training and Testing Model
-->

### *dịch tiêu đề phía trên*

<!--
Let's first implement a function to evaluate the loss on a given data.
-->

*dịch đoạn phía trên*

```{.python .input}
# Saved in the d2l package for later use
def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # sum_loss, num_examples
    for X, y in data_iter:
        metric.add(loss(net(X), y).sum(), y.size)
    return metric[0] / metric[1]
```

<!--
Now define the training function.
-->

*dịch đoạn phía trên*

```{.python .input  n=5}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=1000):
    loss = gluon.loss.L2Loss()
    net = nn.Sequential()
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(1, num_epochs+1):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch % 50 == 0:
            animator.add(epoch, (evaluate_loss(net, train_iter, loss),
                                 evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
### Third-Order Polynomial Function Fitting (Normal)
-->

### *dịch tiêu đề phía trên*

<!--
We will begin by first using a third-order polynomial function with the same order as the data generation function.
The results show that this model’s training error rate when using the testing dataset is low.
The trained model parameters are also close to the true values $w = [5, 1.2, -3.4, 5.6]$.
-->

*dịch đoạn phía trên*

```{.python .input  n=6}
# Pick the first four dimensions, i.e., 1, x, x^2, x^3 from the polynomial
# features
train(poly_features[:n_train, 0:4], poly_features[n_train:, 0:4],
      labels[:n_train], labels[n_train:])
```

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!--
### Linear Function Fitting (Underfitting)
-->

### *dịch tiêu đề phía trên*

<!--
Let’s take another look at linear function fitting.
After the decline in the early epoch, it becomes difficult to further decrease this model’s training error rate.
After the last epoch iteration has been completed, the training error rate is still high.
When used to fit non-linear patterns (like the third-order polynomial function here) linear models are liable to underfit.
-->

*dịch đoạn phía trên*

```{.python .input  n=7}
# Pick the first four dimensions, i.e., 1, x from the polynomial features
train(poly_features[:n_train, 0:3], poly_features[n_train:, 0:3],
      labels[:n_train], labels[n_train:])
```

<!--
### Insufficient Training (Overfitting)
-->

### *dịch tiêu đề phía trên*

<!--
Now let's try to train the model using a polynomial of too high degree.
Here, there is insufficient data to learn that the higher-degree coefficients should have values close to zero.
As a result, our overly-complex model is far too susceptible to being influenced by noise in the training data.
Of course, our training error will now be low (even lower than if we had the right model!) but our test error will be high.
-->

*dịch đoạn phía trên*

<!--
Try out different model complexities (`n_degree`) and training set sizes (`n_subset`) to gain some intuition of what is happening.
-->

*dịch đoạn phía trên*

```{.python .input  n=8}
n_subset = 100  # Subset of data to train on
n_degree = 20  # Degree of polynomials
train(poly_features[1:n_subset, 0:n_degree],
      poly_features[n_train:, 0:n_degree], labels[1:n_subset],
      labels[n_train:])
```

<!--
In later chapters, we will continue to discuss overfitting problems and methods for dealing with them, such as weight decay and dropout.
-->

*dịch đoạn phía trên*


<!--
## Summary
-->

## Tóm tắt

<!--
* Since the generalization error rate cannot be estimated based on the training error rate, simply minimizing the training error rate will not necessarily mean a reduction in the generalization error rate. Machine learning models need to be careful to safeguard against overfitting such as to minimize the generalization error.
* A validation set can be used for model selection (provided that it is not used too liberally).
* Underfitting means that the model is not able to reduce the training error rate while overfitting is a result of the model training error rate being much lower than the testing dataset rate.
* We should choose an appropriately complex model and avoid using insufficient training samples.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
1. Can you solve the polynomial regression problem exactly? Hint: use linear algebra.
2. Model selection for polynomials
    * Plot the training error vs. model complexity (degree of the polynomial). What do you observe?
    * Plot the test error in this case.
    * Generate the same graph as a function of the amount of data?
3. What happens if you drop the normalization of the polynomial features $x^i$ by $1/i!$. Can you fix this in some other way?
4. What degree of polynomial do you need to reduce the training error to 0?
5. Can you ever expect to see 0 generalization error?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2341)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2341)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)
* 

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
*

<!-- Phần 7 -->
*

<!-- Phần 8 -->
*

<!-- Phần 9 -->
*

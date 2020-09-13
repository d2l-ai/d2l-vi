<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->

<!--
# Generative Adversarial Networks
-->

# Mạng đối sinh
:label:`sec_basic_gan`


<!--
Throughout most of this book, we have talked about how to make predictions.
In some form or another, we used deep neural networks learned mappings from data examples to labels.
This kind of learning is called discriminative learning,
as in, we'd like to be able to discriminate between photos cats and photos of dogs.
Classifiers and regressors are both examples of discriminative learning.
And neural networks trained by backpropagation have upended everything 
we thought we knew about discriminative learning on large complicated datasets.
Classification accuracies on high-res images has gone from useless to human-level (with some caveats) in just 5-6 years.
We will spare you another spiel about all the other discriminative tasks where deep neural networks do astoundingly well.
-->

Xuyên suốt gần toàn bộ cuốn sách này, ta đã nói về việc làm thế nào để thực hiện các dự đoán.
Dưới dạng nào đi nữa, ta đã sử dụng mạng nơ-rôn sâu học ánh xạ từ các mẫu ví dụ sang các nhãn.
Loại học này được gọi là học phân biệt,
như là ta muốn có thể phân biệt giữa ảnh của các con chó và các con mèo.
Phân loại và hồi quy là hai ví dụ của việc học phân biệt.
Và mạng nơ-rôn được huấn luyện dùng lan truyền ngược đã đảo lộn mọi thứ 
ta từng nghĩ là ta đã biết về học phân biệt trên các tập dữ liệu lớn phức tạp. 
Độ chính xác phân loại ảnh có độ phân giải cao đã đạt tới mức độ như người (với một số điều kiện) từ chỗ không thể sử dụng được chỉ trong 5-6 năm gần đây.
Chúng tôi sẽ đem đến cho các bạn một câu chuyện khác về tất cả các tác vụ phân biệt khác mà ở đó mạng nơ-rôn sâu thực hiện tốt đáng kinh ngạc.

<!--
But there is more to machine learning than just solving discriminative tasks.
For example, given a large dataset, without any labels, 
we might want to learn a model that concisely captures the characteristics of this data.
Given such a model, we could sample synthetic data examples that resemble the distribution of the training data.
For example, given a large corpus of photographs of faces,
we might want to be able to generate a new photorealistic image that looks like it might plausibly have come from the same dataset.
This kind of learning is called generative modeling.
-->
Nhưng có nhiều thứ để học máy hơn là chỉ giải các tác vụ phân biệt.
Chẳng hạn, với một tập dữ liệu cho trước, không có bất kỳ nhãn nào,
ta có lẽ muốn học một mô hình thu chính xác các đặc tính của tập dữ liệu này.
Với một mô hình như vậy, ta có thể lấy các mẫu dữ liệu tạo ra giống như phân phối của dữ liệu dùng huấn luyện.
Ví dụ, với một kho lớn dữ liệu ảnh khuôn mặt cho trước,
có thể ta muốn có khả năng tạo ra được một ảnh như thật mà trông giống như nó được lấy ra từ cùng tập dữ liệu.
Kiểu học này được gọi là tạo mô hình sinh.


<!--
Until recently, we had no method that could synthesize novel photorealistic images.
But the success of deep neural networks for discriminative learning opened up new possibilities.
One big trend over the last three years has been the application of
discriminative deep nets to overcome challenges in problems that we do not generally think of as supervised learning problems.
The recurrent neural network language models are one example of using a discriminative network 
(trained to predict the next character) that once trained can act as a generative model.
-->

Cho đến gần đây, ta không có phương cách nào để có thể tổng hợp các ảnh như thật mới.
Nhưng thành công của mạng nơ-rôn sâu với học phân biệt đã mở ra những khả năng mới.
Một hướng lớn trong hơn ba năm vừa qua là đã áp dụng 
mạng sâu phân biệt để vượt qua các thách thức trong các bài toán mà ta nhìn chung không nghĩ nó như là bài toán học có giám sát.
Các mô hình ngôn ngữ mạng nơ-ron hồi tiếp là một ví dụ về việc sử dụng một mạng phân biệt
(được huấn luyện để dự đoán ký tự kế tiếp) mà ngay khi được huấn luyện có thể vận hành như một mô hình sinh.


<!--
In 2014, a breakthrough paper introduced Generative adversarial networks (GANs) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`, 
a clever new way to leverage the power of discriminative models to get good generative models.
At their heart, GANs rely on the idea that a data generator is good if we cannot tell fake data apart from real data.
In statistics, this is called a two-sample test - a test to answer 
the question whether datasets $X=\{x_1,\ldots, x_n\}$ and $X'=\{x'_1,\ldots, x'_n\}$ were drawn from the same distribution.
The main difference between most statistics papers and GANs is that the latter use this idea in a constructive way.
In other words, rather than just training a model to say "hey, these two datasets do not look like they came from the same distribution",
they use the [two-sample test](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) to provide training signals to a generative model.
This allows us to improve the data generator until it generates something that resembles the real data.
At the very least, it needs to fool the classifier. Even if our classifier is a state of the art deep neural network.
-->

Trong năm 2014, một bài báo đột phá đưa ra các mạng đối sinh (GAN) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`,
một cách mới khôn khéo để nâng sức mạnh của các mô hình phân biệt đến các mô hình sinh tốt.
Ở trái tim của chúng, các GAN dựa trên ý tưởng là một bộ sinh dữ liệu gọi là tốt nếu ta không thể chỉ ra đâu dữ liệu giả đâu là dữ liệu thật.
Trong thống kê, điều này được gọi là bài kiểm tra từ hai tập mẫu - một bài kiểm tra để trả lời
câu hỏi liệu tập dữ liệu $X=\{x_1,\ldots, x_n\}$ và $X'=\{x'_1,\ldots, x'_n\}$  có được rút ra từ cùng tập phân phối.
Sự khác biệt chính giữa hầu hết những bài nghiên cứu thống kê và các GAN là loại sau sử dụng ý tưởng này theo một cách có cấu trúc.
Nói cách khác, thay vì chỉ huấn luyện một mô hình để nói "này, có hai tập dữ liệu không giống như chúng đến cùng từ một tập phân phối",
chúng sử dụng [kiểm tra trên hai tập mẫu](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) để cung cấp tín hiệu cho việc huấn luyện cho một mô hình sinh.
Điều này cho phép ta cải thiện bộ sinh dữ liệu tới khi nó sinh ra thứ gì đó giống như dữ liệu thực.
Ở mức tối thiểu nhất, nó cần lừa được bộ phân loại. Thậm chí nếu bộ phân loại của ta là mạng nơ-rôn sâu tân tiến nhất.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
![Generative Adversarial Networks](../img/gan.svg)
-->

![*dịch mô tả phía trên*](../img/gan.svg)
:label:`fig_gan`


<!--
The GAN architecture is illustrated in :numref:`fig_gan`.
As you can see, there are two pieces in GAN architecture - first off, we need a device
(say, a deep network but it really could be anything, such as a game rendering engine) 
that might potentially be able to generate data that looks just like the real thing.
If we are dealing with images, this needs to generate images.
If we are dealing with speech, it needs to generate audio sequences, and so on.
We call this the generator network. The second component is the discriminator network.
It attempts to distinguish fake and real data from each other.
Both networks are in competition with each other.
The generator network attempts to fool the discriminator network.
At that point, the discriminator network adapts to the new fake data.
This information, in turn is used to improve the generator network, and so on.
-->

*dịch đoạn phía trên*


<!--
The discriminator is a binary classifier to distinguish if the input $x$ is real (from real data) or fake (from the generator).
Typically, the discriminator outputs a scalar prediction $o\in\mathbb R$ for input $\mathbf x$, 
such as using a dense layer with hidden size 1, 
and then applies sigmoid function to obtain the predicted probability $D(\mathbf x) = 1/(1+e^{-o})$.
Assume the label $y$ for the true data is $1$ and $0$ for the fake data.
We train the discriminator to minimize the cross-entropy loss, *i.e.*,
-->

*dịch đoạn phía trên*


$$ \min_D \{ - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)) \},$$


<!--
For the generator, it first draws some parameter $\mathbf z\in\mathbb R^d$ from a source of randomness,
*e.g.*, a normal distribution $\mathbf z \sim \mathcal{N} (0, 1)$.
We often call $\mathbf z$ as the latent variable.
It then applies a function to generate $\mathbf x'=G(\mathbf z)$.
The goal of the generator is to fool the discriminator to classify $\mathbf x'=G(\mathbf z)$ 
as true data, *i.e.*, we want $D( G(\mathbf z)) \approx 1$.
In other words, for a given discriminator $D$, 
we update the parameters of the generator $G$ to maximize the cross-entropy loss when $y=0$, *i.e.*,
-->

*dịch đoạn phía trên*


$$ \max_G \{ - (1-y) \log(1-D(G(\mathbf z))) \} = \max_G \{ - \log(1-D(G(\mathbf z))) \}.$$


<!--
If the generator does a perfect job, then $D(\mathbf x')\approx 1$ so the above loss near 0, 
which results the gradients are too small to make a good progress for the discriminator.
So commonly we minimize the following loss:
-->

*dịch đoạn phía trên*


$$ \min_G \{ - y \log(D(G(\mathbf z))) \} = \min_G \{ - \log(D(G(\mathbf z))) \}, $$


<!--
which is just feed $\mathbf x'=G(\mathbf z)$ into the discriminator but giving label $y=1$.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
To sum up, $D$ and $G$ are playing a "minimax" game with the comprehensive objective function:
-->

*dịch đoạn phía trên*


$$min_D max_G \{ -E_{x \sim \text{Data}} log D(\mathbf x) - E_{z \sim \text{Noise}} log(1 - D(G(\mathbf z))) \}.$$


<!--
Many of the GANs applications are in the context of images.
As a demonstration purpose, we are going to content ourselves with fitting a much simpler distribution first.
We will illustrate what happens if we use GANs to build the world's most inefficient estimator of parameters for a Gaussian. 
Let us get started.
-->

*dịch đoạn phía trên*


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```


<!--
## Generate some "real" data
-->

## *dịch tiêu đề trên*


<!--
Since this is going to be the world's lamest example, we simply generate data drawn from a Gaussian.
-->

*dịch đoạn phía trên*


```{.python .input}
#@tab all
X = d2l.normal(0.0, 1, (1000, 2))
A = d2l.tensor([[1, 2], [-0.1, 0.5]])
b = d2l.tensor([1, 2])
data = d2l.matmul(X, A) + b
```


<!--
Let us see what we got.
This should be a Gaussian shifted in some rather arbitrary way with mean $b$ and covariance matrix $A^TA$.
-->

*dịch đoạn phía trên*


```{.python .input}
#@tab all
d2l.set_figsize()
d2l.plt.scatter(d2l.numpy(data[:100, 0]), d2l.numpy(data[:100, 1]));
print(f'The covariance matrix is\n{d2l.matmul(A.T, A)}')
```

```{.python .input}
#@tab all
batch_size = 8
data_iter = d2l.load_array((data,), batch_size)
```


<!--
## Generator
-->

## *dịch tiêu đề trên*


<!--
Our generator network will be the simplest network possible - a single layer linear model.
This is since we will be driving that linear network with a Gaussian data generator.
Hence, it literally only needs to learn the parameters to fake things perfectly.
-->

*dịch đoạn phía trên*


```{.python .input}
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

```{.python .input}
#@tab pytorch
net_G = nn.Sequential(nn.Linear(2, 2))
```


<!--
## Discriminator
-->

## *dịch tiêu đề trên*


<!--
For the discriminator we will be a bit more discriminating: 
we will use an MLP with 3 layers to make things a bit more interesting.
-->

*dịch đoạn phía trên*


```{.python .input}
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

```{.python .input}
#@tab pytorch
net_D = nn.Sequential(
    nn.Linear(2, 5), nn.Tanh(),
    nn.Linear(5, 3), nn.Tanh(),
    nn.Linear(3, 1))
```


<!--
## Training
-->

## *dịch tiêu đề trên*


<!--
First we define a function to update the discriminator.
-->

*dịch đoạn phía trên*


```{.python .input}
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = np.ones((batch_size,), ctx=X.ctx)
    zeros = np.zeros((batch_size,), ctx=X.ctx)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Do not need to compute gradient for `net_G`, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return float(loss_D.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator."""
    batch_size = X.shape[0]
    ones = torch.ones((batch_size,), device=X.device)
    zeros = torch.zeros((batch_size,), device=X.device)
    trainer_D.zero_grad()
    real_Y = net_D(X)
    fake_X = net_G(Z)
    # Do not need to compute gradient for `net_G`, detach it from
    # computing gradients.
    fake_Y = net_D(fake_X.detach())
    loss_D = (loss(real_Y, ones.reshape(real_Y.shape)) + 
              loss(fake_Y, zeros.reshape(fake_Y.shape))) / 2
    loss_D.backward()
    trainer_D.step()
    return loss_D
```


<!--
The generator is updated similarly.
Here we reuse the cross-entropy loss but change the label of the fake data from $0$ to $1$.
-->

*dịch đoạn phía trên*


```{.python .input}
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = np.ones((batch_size,), ctx=Z.ctx)
    with autograd.record():
        # We could reuse `fake_X` from `update_D` to save computation
        fake_X = net_G(Z)
        # Recomputing `fake_Y` is needed since `net_D` is changed
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return float(loss_G.sum())
```

```{.python .input}
#@tab pytorch
#@save
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator."""
    batch_size = Z.shape[0]
    ones = torch.ones((batch_size,), device=Z.device)
    trainer_G.zero_grad()
    # We could reuse `fake_X` from `update_D` to save computation
    fake_X = net_G(Z)
    # Recomputing `fake_Y` is needed since `net_D` is changed
    fake_Y = net_D(fake_X)
    loss_G = loss(fake_Y, ones.reshape(fake_Y.shape))
    loss_G.backward()
    trainer_G.step()
    return loss_G
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
Both the discriminator and the generator performs a binary logistic regression with the cross-entropy loss.
We use Adam to smooth the training process.
In each iteration, we first update the discriminator and then the generator.
We visualize both losses and generated examples.
-->

*dịch đoạn phía trên*


```{.python .input}
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = gluon.loss.SigmoidBCELoss()
    net_D.initialize(init=init.Normal(0.02), force_reinit=True)
    net_G.initialize(init=init.Normal(0.02), force_reinit=True)
    trainer_D = gluon.Trainer(net_D.collect_params(),
                              'adam', {'learning_rate': lr_D})
    trainer_G = gluon.Trainer(net_G.collect_params(),
                              'adam', {'learning_rate': lr_G})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for X in data_iter:
            batch_size = X.shape[0]
            Z = np.random.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = np.random.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).asnumpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```

```{.python .input}
#@tab pytorch
def train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim, data):
    loss = nn.BCEWithLogitsLoss(reduction='sum')
    for w in net_D.parameters():
        nn.init.normal_(w, 0, 0.02)
    for w in net_G.parameters():
        nn.init.normal_(w, 0, 0.02)
    trainer_D = torch.optim.Adam(net_D.parameters(), lr=lr_D)
    trainer_G = torch.optim.Adam(net_G.parameters(), lr=lr_G)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs], nrows=2, figsize=(5, 5),
                            legend=['discriminator', 'generator'])
    animator.fig.subplots_adjust(hspace=0.3)
    for epoch in range(num_epochs):
        # Train one epoch
        timer = d2l.Timer()
        metric = d2l.Accumulator(3)  # loss_D, loss_G, num_examples
        for (X,) in data_iter:
            batch_size = X.shape[0]
            Z = torch.normal(0, 1, size=(batch_size, latent_dim))
            metric.add(update_D(X, Z, net_D, net_G, loss, trainer_D),
                       update_G(Z, net_D, net_G, loss, trainer_G),
                       batch_size)
        # Visualize generated examples
        Z = torch.normal(0, 1, size=(100, latent_dim))
        fake_X = net_G(Z).detach().numpy()
        animator.axes[1].cla()
        animator.axes[1].scatter(data[:, 0], data[:, 1])
        animator.axes[1].scatter(fake_X[:, 0], fake_X[:, 1])
        animator.axes[1].legend(['real', 'generated'])
        # Show the losses
        loss_D, loss_G = metric[0]/metric[2], metric[1]/metric[2]
        animator.add(epoch + 1, (loss_D, loss_G))
    print(f'loss_D {loss_D:.3f}, loss_G {loss_G:.3f}, '
          f'{metric[2] / timer.stop():.1f} examples/sec')
```


<!--
Now we specify the hyperparameters to fit the Gaussian distribution.
-->

*dịch đoạn phía trên*


```{.python .input}
#@tab all
lr_D, lr_G, latent_dim, num_epochs = 0.05, 0.005, 2, 20
train(net_D, net_G, data_iter, num_epochs, lr_D, lr_G,
      latent_dim, d2l.numpy(data[:100]))
```

## Tóm tắt

<!--
* Generative adversarial networks (GANs) composes of two deep networks, the generator and the discriminator.
* The generator generates the image as much closer to the true image as possible to fool the discriminator, via maximizing the cross-entropy loss, *i.e.*, $\max \log(D(\mathbf{x'}))$.
* The discriminator tries to distinguish the generated images from the true images, via minimizing the cross-entropy loss, *i.e.*, $\min - y \log D(\mathbf{x}) - (1-y)\log(1-D(\mathbf{x}))$.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
Does an equilibrium exist where the generator wins, *i.e.* the discriminator ends up unable to distinguish the two distributions on finite samples?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/408)
* [Tiếng Anh - PyTorch](https://discuss.d2l.ai/t/776)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


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

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

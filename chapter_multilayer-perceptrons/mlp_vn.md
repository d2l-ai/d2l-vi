<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Multilayer Perceptrons
-->

# *dịch tiêu đề phía trên*
:label:`sec_mlp`

<!--
In the previous chapter, we introduced softmax regression (:numref:`sec_softmax`), 
implementing the algorithm from scratch (:numref:`sec_softmax_scratch`) and in gluon (:numref:`sec_softmax_gluon`) 
and training classifiers to recognize 10 categories of clothing from low-resolution images.
Along the way, we learned how to wrangle data, coerce our outputs into a valid probability distribution (via `softmax`), 
apply an appropriate loss function, and to minimize it with respect to our model's parameters.
Now that we have mastered these mechanics in the context of simple linear models, 
we can launch our exploration of deep neural networks, the comparatively rich class of models with which this book is primarily concerned.
-->

*dịch đoạn phía trên*

<!--
## Hidden Layers
-->

## *dịch tiêu đề phía trên*

<!--
To begin, recall the model architecture corresponding to our softmax regression example, illustrated in  :numref:`fig_singlelayer` below.
This model mapped our inputs directly to our outputs via a single linear transformation:
-->

*dịch đoạn phía trên*

$$
\hat{\mathbf{o}} = \mathrm{softmax}(\mathbf{W} \mathbf{x} + \mathbf{b}).
$$

<!--
![Single layer perceptron with 5 output units.](../img/singlelayer.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/singlelayer.svg)
:label:`fig_singlelayer`

<!--
If our labels truly were related to our input data by a linear function, then this approach would be sufficient.
But linearity is a *strong assumption*.
-->

*dịch đoạn phía trên*

<!--
For example, linearity implies the *weaker* assumption of *monotonicity*: 
that any increase in our feature must either always cause an increase in our model's output (if the corresponding weight is positive), 
or always always cause a decrease in our model's output (if the corresponding weight is negative).
Sometimes that makes sense.
For example, if we were trying to predict whether an individual will repay a loan, 
we might reasonably imagine that holding all else equal, an applicant with a higher income would always be more likely to repay than one with a lower income.
While monotonic this relationship likely
A increase in income from $0 to $50k likely corresponds to a bigger increase in likelihood of repayment than an increase from $1M to $1.05M.
One way to handle this might be to pre-process our data such that linearity becomes more plausible, say, by using the logarithm of income as our feature.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
Note that we can easily come up with examples that violate *monotonicity*.
Say for example that we want to predict probability of death based on body temperature.
For individuals with a body temperature above 37°C (98.6°F), higher termperatures indicate greater risk.
However, for individuals with body termperatures below 37° C, higher temperatures indicate *lower* risk!
In this case too, we might resolve the problem with some clever preprocessing.
Namely, we might use the *distance* from 37°C as our feature.
-->

*dịch đoạn phía trên*


<!--
But what about classifying images of cats and dogs?
Should increasing the intensity of the pixel at location (13, 17) always increase (or always decrease) the likelihood that the image depicts a dog?
Reliance on a linear model corrsponds to the (implicit) assumption that the only requirement for differentiating cats vs. dogs is to assess the brightness of individual pixels.
This approach is doomed to fail in a world where inverting an image preserves the category.
-->

*dịch đoạn phía trên*


<!--
And yet despite the apparent aburdity of linearity here, as compared to our previous examples, it's less obvious that we could address the problem with a simple preprocessing fix.
That is because the significance of any pixel depends in complex ways on its context (the values of the surrounding pixels).
While there might exist a representation of our data that would take into account the relevant interactions among our features (and on top of which a linear model would be suitable), 
we simply do not know how to calculate it by hand.
With deep neural networks, we used observational data to jointly learn both a representation (via hidden layers) and a linear predictor that acts upon that representation.
-->

*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU =================================== -->

<!--
### Incorporating Hidden Layers
-->

### *dịch tiêu đề phía trên*

<!--
We can over come these limitations of linear models and handle a more general class of functions by incorporating one or more hidden layers.
The easiest way to do this is to stack many fully-connected layers on top of each other.
Each layer feeds into the layer above it, until we generate an output.
We can think of the first $L-1$ layers as our representation and the final layer as our linear predictor.
This architecture is commonly called a *multilayer perceptron*, often abbreviated as *MLP*.
Below, we depict an MLP diagramtically (:numref:`fig_nlp`).
-->

*dịch đoạn phía trên*

<!--
![Multilayer perceptron with hidden layers. This example contains a hidden layer with 5 hidden units in it. ](../img/mlp.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/mlp.svg)
:label:`fig_nlp`

<!--
This multilayer perceptron has 4 inputs, 3 outputs, and its hidden layer contains 5 hidden units.
Since the input layer does not involve any calculations, producing outputs with this network requires implementing the computations for each of the 2 layers (hidden and output).
Note that these layers are both fully connected.
Every input influences every neuron in the hidden layer, and each of these in turn influences every neuron in the output layer.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
### From Linear to Nonlinear
-->

### *dịch tiêu đề phía trên*

<!--
Formally, we calculate the each layer in this one-hidden-layer MLP as follows:
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
    \mathbf{h} & = \mathbf{W}_1 \mathbf{x} + \mathbf{b}_1, \\
    \mathbf{o} & = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2, \\
    \hat{\mathbf{y}} & = \mathrm{softmax}(\mathbf{o}).
\end{aligned}
$$

<!--
Note that after adding this layer, our model now requires us to track and update two additional sets of parameters.
So what have we gained in exchange?
You might be surprised to find out that---in the model defined above---*we gain nothing for our troubles!*
The reason is plain.
The hidden units above are given by a linear function of the inputs, and the outputs (pre-softmax) are just a linear function of the hidden units.
A linear function of a linear function is itself a linear function.
Moreover, our linear model was already capable of representing any linear function.
-->

*dịch đoạn phía trên*


<!--
We can view the equivalence formally by proving that for any values of the weights, we can just collapse out the hidden layer, yielding an equivalent single-layer model with paramters
-->

*dịch đoạn phía trên*

$\mathbf{W} = \mathbf{W}_2 \mathbf{W}_1$ and $\mathbf{b} = \mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2$.

$$\mathbf{o} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2 = \mathbf{W}_2 (\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 = (\mathbf{W}_2 \mathbf{W}_1) \mathbf{x} + (\mathbf{W}_2 \mathbf{b}_1 + \mathbf{b}_2) = \mathbf{W} \mathbf{x} + \mathbf{b}.$$


<!--
In order to realize the potential of multilayer architectures,
we need one more key ingredient---an
elementwise *nonlinear activation function* $\sigma$
to be applied to each hidden unit
(following the linear transformation).
The most popular choice for the nonlinearity
these days is the rectified linear unit (ReLU)
$\mathrm{max}(x, 0)$.
In general, with these activation functions in place,
it is no longer possible to collapse our MLP into a linear model.
-->

*dịch đoạn phía trên*

$$
\begin{aligned}
    \mathbf{h} & = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1), \\
    \mathbf{o} & = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2, \\
    \hat{\mathbf{y}} & = \mathrm{softmax}(\mathbf{o}).
\end{aligned}
$$

<!--
To build more general MLPs, we can continue stacking such hidden layers, 
e.g., $\mathbf{h}_1 = \sigma(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)$ and $\mathbf{h}_2 = \sigma(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)$, 
one atop another, yielding ever more expressive models (assuming fixed width).
-->

*dịch đoạn phía trên*

<!--
MLPs can capture complex interactions among our inputs via their hidden neurons, which depend on the values of each of the inputs.
We can easily design hidden nodes to perform arbitrary computation, for instance, basic logic operations on a pair of inputs.
Moreover, for certain choices of the activation function, it is widely known that MLPs are universal approximators.
Even a single-hidden-layer network, given enough nodes (possibly absurdly manby), and the right set of weights, we can model any function at all.
*Actually learning that function is the hard part.* You might think of your neural network as being a bit like the C programming language.
The language, like any other modern language, is capable of expressing any computable program.
But actually coming up with a program that meets your specifications is the hard part.
-->

*dịch đoạn phía trên*

<!--
Moreover, just because a single-layer network *can* learn any function does not mean that you should try to solve all of your problems with single-layer networks.
In fact, we can approximate many functions much more compactly by using deeper (vs wider) networks.
We'll touch upon more rigorous arguments in subsequent chapters, but first let's actually build an MLP in code.
In this example, we’ll implement an MLP with two hidden layers and one output layer.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU =================================== -->

<!--
### Vectorization and Minibatch
-->

### Vector hoá và Minibatch

<!--
As before, by the matrix $\mathbf{X}$, we denote a minibatch of inputs.
The calculations to produce outputs from an MLP with two hidden layers can thus be expressed:
-->

Giống như trước, chúng ta dùng ma trận $\mathbf{X}$ để ký hiệu một minibatch các giá trị đầu vào.
Các phép tính toán dẫn đến các giá trị đầu ra từ một MLP với hai tầng ẩn khi đó có thể được biểu diễn như sau:

$$
\begin{aligned}
    \mathbf{H}_1 & = \sigma(\mathbf{W}_1 \mathbf{X} + \mathbf{b}_1), \\
    \mathbf{H}_2 & = \sigma(\mathbf{W}_2 \mathbf{H}_1 + \mathbf{b}_2), \\
    \mathbf{O} & = \mathrm{softmax}(\mathbf{W}_3 \mathbf{H}_2 + \mathbf{b}_3).
\end{aligned}
$$

<!--
With some abuse of notation, we define the nonlinearity $\sigma$ to apply to its inputs in a row-wise fashion, i.e., one observation at a time.
Note that we are also using the notation for *softmax* in the same way to denote a row-wise operation.
Often, as in this section, the activation functions that we apply to hidden layers are not merely row-wise, but component wise.
That means that after computing the linear portion of the layer, we can calculate each nodes activation without looking at the values taken by the other hidden units.
This is true for most activation functions (the batch normalization operation will be introduced in :numref:`sec_batch_norm` is a notable exception to that rule).
-->

Khác với các quy ước thông thường, chúng ta định nghĩa hàm phi tuyến $\sigma$ áp dụng cho các giá trị đầu vào theo từng hàng, tức lần lượt từng điểm dữ liệu một.
Cần chú ý rằng ta cũng sử dụng quy ước này cho *softmax* để ký hiệu toán tử tính theo từng dòng.
Thông thường, như trong mục này, các hàm kích hoạt mà chúng ta áp dụng vào tầng ẩn không chỉ đơn thuần theo từng dòng mà còn theo từng phần.
Điều đó có nghĩa là sau khi tính toán các phần tuyến tính của tầng, chúng ta có thể tính kích hoạt cho từng nút mà không cần quan tâm đến giá trị của các nút còn lại.
Điều này cũng đúng đối với hầu hết các hàm kích hoạt (toán tử chuẩn hoá batch sẽ được giới thiệu trong :numref:`sec_batch_norm` là một trường hợp ngoại lệ của quy tắc này).

```{.python .input  n=1}
%matplotlib inline
import d2l
from mxnet import autograd, np, npx
npx.set_np()
```

<!--
## Activation Functions
-->

## Các hàm Kích hoạt

<!--
Activation functions decide whether a neuron should be activated or not by calculating the weighted sum and further adding bias with it.
They are differentiable operators to transform input signals to outputs, while most of them add non-linearity.
Because activation functions are fundamental to deep learning, let's briefly survey some common activation functions.
-->

Các hàm kích hoạt quyết định một nơ-ron có được kích hoạt hay không bằng cách tính tổng có trọng số và cộng thêm hệ số điều chỉnh vào nó.
Chúng là các toán tử khả vi biến đổi các tín hiệu đầu vào thành các tín hiệu đầu ra, mặc dù hầu hết các hàm kích hoạt này đều thêm vào phần phi tuyến tính.
Bởi vì các hàm kích hoạt rất quan trọng trong học sâu, chúng ta hãy cùng tìm hiểu sơ lược một số hàm kích hoạt thông dụng.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### ReLU Function
-->

### *dịch tiêu đề phía trên*

<!--
As stated above, the most popular choice, due to both simplicity of implementation its performance on a variety of predictive tasks is the rectified linear unit (ReLU).
ReLUs provide a very simple nonlinear transformation.
Given the element $z$, the function is defined as the maximum of that element and 0.
-->

*dịch đoạn phía trên*

$$\mathrm{ReLU}(z) = \max(z, 0).$$

<!--
Informally, the ReLU function retains only positive elements and discards all negative elements (setting the corresponding activations to 0).
To gain some intuition, we can plot the function.
Because it is used so commonly, NDarray supports the `relu` function as a native operator.
As you can see, the activation function is piecewise linear.
-->

*dịch đoạn phía trên*

```{.python .input  n=2}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.set_figsize((4, 2.5))
d2l.plot(x, y, 'x', 'relu(x)')
```

<!--
When the input is negative, the derivative of ReLU function is 0 and when the input is positive, the derivative of ReLU function is 1.
Note that the ReLU function is not differentiable when the input takes value precisely equal to 0.
In these cases, we default to the left-hand-side (LHS) derivative and say that the derivative is 0 when the input is 0.
We can get away with this because the input may never actually be zero.
There is an old adage that if subtle boundary conditions matter, we are probably doing (*real*) mathematics, not engineering.
That conventional wisdom may apply here.
We plot the derivative of the ReLU function plotted below.
-->

*dịch đoạn phía trên*

```{.python .input  n=3}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu')
```

<!--
Note that there are many variants to the ReLU function, including the parameterized ReLU (pReLU) of [He et al., 2015](https://arxiv.org/abs/1502.01852).
This variation adds a linear term to the ReLU, so some information still gets through, even when the argument is negative.
-->

*dịch đoạn phía trên*

$$\mathrm{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

<!--
The reason for using the ReLU is that its derivatives are particularly well behaved: either they vanish or they just let the argument through.
This makes optimization better behaved and it mitigated well-documented problem of *vanishing gradients* that plagued previous versions of neural networks (more on this later).
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU =================================== -->

<!--
### Sigmoid Function
-->

### Hàm Sigmoid

<!--
The sigmoid function transforms its inputs, which values in the domain $\mathbb{R}$, to outputs that lie the interval $(0, 1)$.
For that reason, the sigmoid is often called a *squashing* function: it *squashes* any input in the range (-inf, inf) to some value in the range (0, 1).
-->

Hàm sigmoid biến đổi các giá trị đầu vào có miền giá trị thuộc $\mathbb{R}$ thành các giá trị đầu ra nằm trong khoảng $(0, 1)$.
Vì vậy, hàm sigmoid thường được gọi là hàm *nén*: nó *nén* một giá trị đầu vào bất kỳ nằm trong khoảng (âm vô cùng, dương vô cùng) thành một giá trị đầu ra nằm trong khoảng (0, 1).

$$\mathrm{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

<!--
In the earliest neural networks, scientists were interested in modeling biological neurons which either *fire* or *do not fire*.
Thus the pioneers of this field, going all the way back to McCulloch and Pitts, the inventors of the artificial neuron, focused on thresholding units.
A thresholding activation takes value $0$ when its input is below some threshold and value $1$ when the input exceeds the threshold.
-->

Từ những nghiên cứu đầu tiên về mạng nơ-ron, các nhà khoa học đã quan tâm đến việc mô hình hóa các nơ-ron sinh học, thứ có thể ở một trong hai trạng thái *kích hoạt* hoặc *không kích hoạt*.
Vì vậy mà những người tiên phong trong lĩnh vực này, bao gồm [McCulloch](https://en.wikipedia.org/wiki/Warren_Sturgis_McCulloch) và [Pitts](https://en.wikipedia.org/wiki/Walter_Pitts), những người phát minh ra nơ-ron nhân tạo, tập trung nghiên cứu các đơn vị ngưỡng.
Một kích hoạt ngưỡng có giá trị là $0$ khi đầu vào của nó ở mức dưới ngưỡng và giá trị là $1$ khi đầu vào ở mức vượt ngưỡng.

<!--
When attention shifted to gradient based learning, the sigmoid function was a natural choice because it is a smooth, differentiable approximation to a thresholding unit.
Sigmoids are still widely used as activation functions on the output units, 
when we want to interpret the outputs as probabilities for binary classification problems (you can think of the sigmoid as a special case of the softmax).
However, the sigmoid has mostly been replaced by the simpler and more easily trainable ReLU for most use in hidden layers.
In the "Recurrent Neural Network" chapter (:numref:`sec_plain_rnn`), we will describe architectures that leverage sigmoid units to control the flow of information across time.
-->

Khi phương pháp học dựa trên gradient trở nên thu hút, hàm sigmoid là một lựa chọn phù hợp cho việc xấp xỉ đơn vị ngưỡng bởi vì nó liên tục và khả vi.
Hàm sigmoid vẫn là hàm kích hoạt được sử dụng rộng rãi ở các đơn vị đầu ra,
khi ta muốn biểu diễn kết quả đầu ra như các xác suất cho bài toán phân loại nhị phân (bạn có thể xem sigmoid như một trường hợp đặc biệt của softmax).
Tuy nhiên, trong các tầng ẩn, hàm sigmoid đã gần như bị thay thế bằng hàm ReLU vì nó đơn giản hơn và giúp cho việc huấn luyện trở nên dễ dàng hơn.
Trong chương "Mạng nơ-ron truy hồi" (:numref:`sec_plain_rnn`), chúng tôi sẽ mô tả các mô hình sử dụng đơn vị sigmoid để kiểm soát luồng thông tin theo thời gian.

<!--
Below, we plot the sigmoid function.
Note that when the input is close to 0, the sigmoid function approaches a linear transformation.
-->

Dưới đây, ta vẽ đồ thị hàm sigmoid.
Cần chú ý rằng, khi đầu vào có giá trị gần bằng 0, hàm sigmoid tiến tới một phép biến đổi tuyến tính.

```{.python .input  n=4}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)')
```

<!--
The derivative of sigmoid function is given by the following equation:
-->

Đạo hàm của hàm sigmoid được tính bởi phương trình sau:

$$\frac{d}{dx} \mathrm{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \mathrm{sigmoid}(x)\left(1-\mathrm{sigmoid}(x)\right).$$


<!--
The derivative of sigmoid function is plotted below.
Note that when the input is 0, the derivative of the sigmoid function reaches a maximum of 0.25.
As the input diverges from 0 in either direction, the derivative approaches 0.
-->

Đồ thị đạo hàm của hàm sigmoid được vẽ ở dưới.
Chú ý rằng khi đầu vào là 0, đạo hàm của hàm sigmoid đạt giá trị lớn nhất là 0.25.
Khi đầu vào phân kỳ từ 0 theo một trong hai hướng, đạo hàm sẽ tiến tới 0.

```{.python .input  n=5}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid')
```

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
### Tanh Function
-->

### Hàm "Tanh"

<!--
Like the sigmoid function, the tanh (Hyperbolic Tangent) function also squashes its inputs, transforms them into elements on the interval between -1 and 1:
-->

Tương tự như hàm sigmoid, hàm tanh (Hyperbolic Tangent) cũng nén các biến đầu vào và biến đổi chúng thành các phần tử nằm trong khoảng -1 và 1:

$$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

<!--
We plot the tanh function blow. Note that as the input nears 0, the tanh function approaches a linear transformation. 
Although the shape of the function is similar to the sigmoid function, the tanh function exhibits point symmetry about the origin of the coordinate system.
-->

Chúng ta sẽ vẽ hàm tanh như sau. Chú ý rằng nếu đầu vào có giá trị gần bằng 0, hàm tanh sẽ tiến đến một phép biến đổi tuyến tính.
Mặc dù hình dạng của hàm tanh trông khá giống hàm sigmoid, hàm tanh lại thể hiện tính đối xứng tâm qua gốc của hệ trục tọa độ.

```{.python .input  n=6}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)')
```

<!--
The derivative of the Tanh function is:
-->

Đạo hàm của hàm Tanh là:

$$\frac{d}{dx} \mathrm{tanh}(x) = 1 - \mathrm{tanh}^2(x).$$

<!--
The derivative of tanh function is plotted below.
As the input nears 0, the derivative of the tanh function approaches a maximum of 1.
And as we saw with the sigmoid function, as the input moves away from 0 in either direction, the derivative of the tanh function approaches 0.
-->

Đạo hàm của hàm tanh được vẽ như sau.
Khi đầu vào có giá trị gần bằng 0, đạo hàm của hàm tanh tiến tới giá trị tối đa là 1.
Tương tự như hàm sigmoid, khi đầu vào tiến xa khỏi 0 theo bất kỳ hướng nào, đạo hàm của hàm tanh sẽ tiến đến 0.

```{.python .input  n=7}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh')
```

<!--
In summary, we now know how to incorporate nonlinearities to build expressive multilayer neural network architectures.
As a side note, your knowledge already puts you in command of a similar toolkit to a practitioner circa 1990.
In some ways, you have an advantage over anyone working the 1990s, because you can leverage powerful open-source deep learning frameworks to build models rapidly, using only a few lines of code.
Previously, getting these nets training required researchers to code up thousands of lines of C and Fortran.
-->

Tóm lại, bây giờ chúng ta biết cách kết hợp các hàm phi tuyến để xây dựng các mô hình mạng nơ-ron đa tầng.
Ngoài ra, kiến thức bây giờ của bạn đã cung cấp một bộ công cụ tương đương như của một người làm học sâu vào khoảng năm 1990.
Xét theo một khía cạnh nào đó, bạn còn có lợi thế hơn bất kỳ ai làm việc trong những năm 1990, bởi vì bạn có thể tận dụng triệt để các framework học sâu nguồn mở để xây dựng các mô hình một cách nhanh chóng, chỉ với một vài dòng mã.
Trước đây, việc huấn luyện các mạng nơ-ron đòi hỏi các nhà nghiên cứu phải viết đến hàng ngàn dòng mã C và Fortran.

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
## Summary
-->

## Tóm tắt

<!--
* The multilayer perceptron adds one or multiple fully-connected hidden layers between the output and input layers and transforms the output of the hidden layer via an activation function.
* Commonly-used activation functions include the ReLU function, the sigmoid function, and the tanh function.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
1. Compute the derivative of the tanh and the pReLU activation function.
2. Show that a multilayer perceptron using only ReLU (or pReLU) constructs a continuous piecewise linear function.
3. Show that $\mathrm{tanh}(x) + 1 = 2 \mathrm{sigmoid}(2x)$.
4. Assume we have a multilayer perceptron *without* nonlinearities between the layers. 
In particular, assume that we have $d$ input dimensions, $d$ output dimensions and that one of the layers had only $d/2$ dimensions. 
Show that this network is less expressive (powerful) than a single layer perceptron.
5. Assume that we have a nonlinearity that applies to one minibatch at a time. What kinds of problems do you expect this to cause?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2338)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2338)
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
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
* Lê Cao Thăng

<!-- Phần 5 -->
*

<!-- Phần 6 -->
* Lâm Ngọc Tâm
* Phạm Minh Đức
* Phạm Hồng Vinh

<!-- Phần 7 -->
* Lâm Ngọc Tâm
* Phạm Minh Đức

<!-- Phần 8 -->
*

<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Numerical Stability and Initialization
-->

# Sự ổn định Số và Sự khởi tạo
:label:`sec_numerical_stability`


<!--
Thus far, every model that we have implemented required that initialize its parameters according to some pre-specified distribution.
Until now, we took the initialization scheme for granted, glossed over the details of how these these choices are made.
You might have even gotten the impression that these choices are not especially important.
However, the choice of initialization scheme plays a significant role in neural network learning, and can be crucial for maintaining numerical stability.
Moreover, these choices can be tied up in interesting ways with the choice of the nonlinear activation function.
Which function we choose and how we initialize parameters can determine how quickly our optimization algorithm converges.
Poor choices here can cause us to encounter exploding or vanishing gradients while training.
In this section, we delve into these topics with greater detail and discuss some useful heuristics that you will frequently useful throughout your career in deep learning.
-->

Cho đến nay, đối với mọi mô hình mà ta đã lập trình, ta đều phải khởi tạo các tham số theo một phân phối cụ thể nào đó.
Tuy nhiên, ta mới chỉ lướt qua các chi tiết thực hiện và không để tâm tới việc khởi tạo các siêu tham số.
Bạn thậm chí có thể nghĩ rằng các lựa chọn này không đặc biệt quan trọng.
Tuy nhiên, việc lựa chọn cơ chế khởi tạo đóng vai trò rất lớn trong quá trình học của mạng nơ-ron và có thể là yếu tố quyết định để duy trì sự ổn định số học.
Hơn nữa, các lựa chọn cách khởi tạo cũng có thể có một vài liên kết thú vị tới sự lựa chọn các hàm kích hoạt phi tuyến.
Việc lựa chọn hàm kích hoạt và cách khởi tạo tham số có thể ảnh hưởng tới tốc độ hội tụ của thuật toán tối ưu.
Nếu ta lựa chọn không hợp lý, việc bùng nổ hoặc tiêu biến gradient có thể sẽ xảy ra.
Trong phần này, ta sẽ đi sâu vào các chủ đề trên một cách chi tiết hơn và thảo luận một số phương pháp hữu dụng dựa trên thực nghiêm mà bạn có thể sử dụng thường xuyên trong suốt sự nghiệp học sâu.

<!--
## Vanishing and Exploding Gradients
-->

## Tiêu biến và Bùng nổ Gradient

<!--
Consider a deep network with $L$ layers, input $\mathbf{x}$ and output $\mathbf{o}$.
Each layer satisfies:
With each layer $l$ defined by a transformation $f_l$ parameterized by weights $\mathbf{W}_l$ our network can be expressed as:
-->

Xét một mạng nơ-ron sâu với $L$ tầng, đầu vào $\mathbf{x}$ và đầu ra $\mathbf{o}$.
Mỗi tầng $l$ được định nghĩa bởi một phép biến đổi $f_l$ với tham số là trọng số $\mathbf{W}_l$.
Mạng nơ-ron này có thể được biểu diễn như sau:

$$\mathbf{h}^{l+1} = f_l (\mathbf{h}^l) \text{ và vì vậy } \mathbf{o} = f_L \circ \ldots, \circ f_1(\mathbf{x}).$$

<!--
If all activations and inputs are vectors, wwe can write the gradient of $\mathbf{o}$ with respect to any set of parameters $\mathbf{W}_l$ as follows:
-->

Nếu tất cả giá trị kích hoạt và đầu vào là vector, ta có thể viết lại gradient của $\mathbf{o}$ theo một tập tham số $\mathbf{W}_l$ bất kỳ như sau:

$$\partial_{\mathbf{W}_l} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{L-1}} \mathbf{h}^L}_{:= \mathbf{M}_L} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{l}} \mathbf{h}^{l+1}}_{:= \mathbf{M}_l} \underbrace{\partial_{\mathbf{W}_l} \mathbf{h}^l}_{:= \mathbf{v}_l}.$$

<!--
In other words, it is the product of $d-t$ matrices $\mathbf{M}_d \cdot \ldots, \cdot \mathbf{M}_t$ and the gradient vector $\mathbf{v}_t$.
What happens is similar to the situation when we experienced numerical underflow
when multiplying too many probabilities.
At the time, we were able to mitigate the problem by switching from into log-space, 
i.e., by shifting the problem from the mantissa to the exponent of the numerical representation. 
Unfortunately the problem outlined in the equation above is much more serious: initially the matrices $M_t$ may well have a wide variety of eigenvalues.
They might be small, they might be large, and in particular, their product might well be *very large* or *very small*.
This is not (only) a problem of numerical representation but it means that the optimization algorithm is bound to fail.
It receives gradients that are either excessively large or excessively small.
As a result the steps taken are either (i) excessively large (the *exploding* gradient problem), in which case the parameters blow up in magnitude rendering the model useless,
or (ii) excessively small, (the *vanishing gradient problem*), in which case the parameters hardly move at all, and thus the learning process makes no progress.
-->

<!-- UPDATE
In other words, this gradient is the product of $L-l$ matrices $\mathbf{M}_L \cdot \ldots, \cdot \mathbf{M}_l$ and the gradient vector $\mathbf{v}_l$.
Thus we are susceptible to the same  problems of numerical underflow that often crop up  when multiplying together too many probabilities.
When dealing with probabilities, a common trick is to switch into log-space, i.e., shifting  pressure from the mantissa to the exponent  of the numerical representation. 
Unfortunately, our problem above is more serious: initially the matrices $M_l$ may have a wide variety of eigenvalues.
They might be small or large, and  their product might be *very large* or *very small*.
The risks posed by unstable gradients  goes beyond numerical representation.
Gradients of unpredictable magnitude  also threaten the stability of our optimization algorithms.
We may facing parameter updates that are either (i) excessively large, destroying our model (the *exploding* gradient problem); 
or (ii) excessively small, (the *vanishing gradient problem*), rendering learning impossible as parameters hardly move on each update.
-->

Nói cách khác, gradient này là tích của $L-l$ ma trận $\mathbf{M}_L \cdot \ldots, \cdot \mathbf{M}_l$ với vector gradient $\mathbf{v}_l$.
Vì vậy ta sẽ dễ gặp phải vấn đề tràn số dưới, một hiện tượng thường xảy ra khi nhân quá nhiều xác suất lại với nhau.
Khi làm việc với các xác suất, một mánh phổ biến là chuyển về giá trị log, đồng nghĩa với việc đưa các bit từ phần định trị sang phần mũ nếu nhìn từ góc độ biểu diễn số học.
Thật không may, bài toán trên lại nghiêm trọng hơn nhiều: các ma trận $M_l$ ban đầu có thể có nhiều trị riêng khác nhau.
Các trị riêng có thể nhỏ hoặc lớn, và do đó tích của chúng có thể *rất lớn* hoặc *rất nhỏ*.
Rủi ro của việc gradient bất ổn không chỉ dừng lại ở vấn đề biểu diễn số học.
Nếu ta không kiểm soát được độ lớn của gradient, sự ổn định của các thuật toán tối ưu cũng không được đảm bảo.
Lúc đó ta sẽ quan sát được các bước cập nhật hoặc (i) quá lớn và phá hỏng mô hình (vấn đề *bùng nổ* gradient); hoặc (ii) quá nhỏ (vấn đề *tiêu biến* gradient), khiến việc học trở nên bất khả thi, khi mà các tham số hầu như không thay đổi ở mỗi bước cập nhật.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
### Vanishing Gradients
-->

### Tiêu biến Gradient

<!--
One major culprit in the vanishing gradient problem is the choices of the activation functions $\sigma$ that are interleaved with the linear operations in each layer.
Historically, the sigmoid function $(1 + \exp(-x))$ (introduced in :numref:`sec_mlp`) was a popular choice owing to its similarity to a thresholding function.
Since early artificial neural networks were inspired by biological neural networks, the idea of neurons that either fire or do not fire (biological neurons do not partially fire) seemed appealing.
Let's take a closer look at the function to see why picking it might be problematic vis-a-vis vanishing gradients.
-->

<!-- UPDATE
One frequent culprit causing the vanishing gradient problem is the choice of the activation function $\sigma$ that is appended following each layer's linear operations.
Historically, the sigmoid function  $1/(1 + \exp(-x))$ (introduced in :numref:`sec_mlp`) was popular because it resembles a thresholding function.
Since early artificial neural networks were inspired by biological neural networks, the idea of neurons that either fire either *fully* or *not at all* (like biological neurons) seemed appealing.
Let's take a closer look at the sigmoid to see why it can cause vanishing gradients.
-->

Thông thường, thủ phạm gây ra vấn đề tiêu biến gradient là hàm kích hoạt $\sigma$ được chọn để đặt nối tiếp phép toán tuyến tính tại mỗi tầng.
Trước đây, hàm kích hoạt sigmoid $(1 + \exp(-x))$ (đã giới thiệu trong :numref:`sec_mlp`) là lựa chọn phổ biến bởi nó hoạt động giống với một hàm lấy ngưỡng.
Bởi các mạng nơ-ron nhân tạo thời kỳ đầu lấy cảm hứng từ mạng nơ-ron sinh học, ý tưởng các nơ-ron được kích hoạt *hoàn toàn* hoặc *không hề* kích hoạt (giống như nơ-ron sinh học) có vẻ rất hấp dẫn.
Hãy cùng xem xét hàm sigmoid kỹ lưỡng hơn để thấy tại sao nó có thể gây ra vấn đề tiêu biến gradient.

```{.python .input}
%matplotlib inline
import d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

<!--
As we can see, the gradient of the sigmoid vanishes both when its inputs are large and when they are small.
Moreover, when we execute backward propagation, due to the chain rule, this means that unless we are in the Goldilocks zone, 
where the inputs to most of the sigmoids are in the range of, say $[-4, 4]$, the gradients of the overall product may vanish.
When we have many layers, unless we are especially careful, we are likely to find that our gradient is cut off at *some* layer.
Before ReLUs ($\max(0, x)$) were proposed as an alternative to squashing functions, this problem used to plague deep network training.
As a consequence, ReLUs have become the default choice when designing activation functions in deep networks.
-->

<!-- UPDATE
As you can see, the sigmoid's gradient vanishes both when its inputs are large and when they are small.
Moreover, when backpropagating through many layers, unless we are in the Goldilocks zone---where 
the inputs to many of the sigmoids are close to zero, the gradients of the overall product may vanish.
When our network boasts many layers, unless we are careful, the gradient will likely be cut off at *some* layer.
Indeed, this problem used to plague deep network training.
Consequently, ReLUs which are more stable (but less neurally plausible) have emerged as the default choice for practitioners.
-->

Như ta có thể thấy, gradient của hàm sigmoid tiêu biến khi đầu vào của nó quá lớn hoặc quá nhỏ.
Hơn nữa, khi thực hiện lan truyền ngược qua nhiều tầng, trừ khi giá trị nằm trong vùng Goldilocks, tại đó đầu vào của hầu hết các hàm sigmoid có giá trị xấp xỉ không, gradient của cả phép nhân có thể bị tiêu biến.
Khi mạng nơ-ron có nhiều tầng, trừ khi ta cẩn trọng, nhiều khả năng luồng gradient sẽ bị ngắt tại *một* tầng nào đó.
Vấn đề này đã từng gây nhiều khó khăn cho quá trình huấn luyện mạng nơ-ron sâu.
Do đó, ReLU, một hàm số ổn định hơn (nhưng lại không hợp lý lắm từ khía cạnh khoa học thần kinh) đã và đang dần trở thành lựa chọn mặc định của những người làm học sâu. 

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Exploding Gradients
-->

### Bùng nổ Gradient

<!--
The opposite problem, when gradients explode, can be similarly vexing.
To illustrate this a bit better, we draw $100$ Gaussian random matrices and multiply them with some initial matrix.
For the scale that we picked (the choice of the variance $\sigma^2=1$), the matrix product explodes.
When this happens due to the initialization of a deep network, we have no chance of getting a gradient descent optimizer to converge.
-->

Một vấn đề đối lập, bùng nổ gradient, cũng có thể gây phiền toái không kém.
Để giải thích việc này rõ hơn, chúng ta lấy $100$ ma trận ngẫu nhiên Gaussian và nhân chúng với một ma trận khởi tạo.
Với khoảng giá trị mà ta đã chọn (phương sai $\sigma^2=1$), tích các ma trận bị bùng nổ số học.
Nếu điều này xảy ra trong các mạng học sâu, các bộ tối ưu dựa trên hạ gradient sẽ không thể hội tụ được.

```{.python .input  n=5}
M = np.random.normal(size=(4, 4))
print('A single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('After multiplying 100 matrices', M)
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
### Symmetry
-->

### Tính Đối xứng

<!--
Another problem in deep network design is the symmetry inherent in their parametrization.
Assume that we have a deep network with one hidden layer with two units, say $h_1$ and $h_2$.
In this case, we could permute the weights $\mathbf{W}_1$ of the first layer and likewise permute the weights of the output layer to obtain the same function.
There is nothing special differentiating the first hidden unit vs the second hidden unit.
In other words, we have permutation symmetry among the hidden units of each layer.
-->

Một vấn đề khác trong thiết kế mạng học sâu là tính đối xứng vốn có trong quá trình tham số hóa.
Giả sử ta có một mạng học sâu với một tầng ẩn gồm hai nút, $h_1$ và $h_2$. 
Trong trường hợp này, ta có thể hoán vị trọng số $\mathbf{W}_1$ của lớp đầu tiên cũng như các trọng số của tầng đầu ra để đạt được một hàm tương tự.
Không có gì đặc biệt khác nhau giữa việc vi phân nút ẩn đầu tiên với nút ẩn thứ hai.
Nói cách khác, ta có tính đối xứng hoán vị giữa các nút ẩn của từng tầng.

<!--
This is more than just a theoretical nuisance.
Imagine what would happen if we initialized all of the parameters of some layer as $\mathbf{W}_l = c$ for some constant $c$.
In this case, the gradients for all dimensions are identical: thus not only would each unit take the same value, but it would receive the same update.
Stochastic gradient descent would  never break the symmetry on its own and we might never be able to realize the network's expressive power.
The hidden layer would behave as if it had only a single unit.
Note that while SGD would not break this symmetry, dropout regularization would!
-->

Đây không chỉ là phiền toái về mặt lý thuyết.
Tưởng tượng điều gì sẽ xảy ra nếu ta đặt giá trị ban đầu cho tất cả các thông số của các tầng theo cách $\mathbf{W}_l = c$ với hằng số $c$ nào đó.
Trong trường hợp này, các gradient cho tất cả các chiều là giống hệt nhau: nên mỗi nút không chỉ có cùng giá trị mà cũng sẽ có bước cập nhật giống nhau.
Hạ gradient ngẫu nhiên sẽ không bao giờ phá vỡ tính đối xứng sẵn có và ta có thể sẽ không hiện thực được sức mạnh biểu diễn của mạng.
Tầng ẩn sẽ hoạt động như thể nó chỉ có một nút duy nhất.
Bên cạnh đó, lưu ý rằng hạ gradient ngẫu nhiên sẽ không phá vỡ cân đối này thì nó sẽ bị phá vỡ bởi điều chuẩn hóa dropout!

<!--
## Parameter Initialization
-->

## Khởi tạo Tham số

<!--
One way of addressing, or at least mitigating the issues raised above is through careful initialization of the weight vectors.
This way we can ensure that (at least initially) the gradients do not vanish and that they maintain a reasonable scale where the network weights do not diverge.
Additional care during optimization and suitable regularization ensures that things never get too bad.
-->

<!-- UPDATE
One way of addressing---or at least mitigating---the issues raised above is through careful initialization.
Additional care during optimization and suitable regularization can further enhance stability.
-->

Một cách giải quyết, hay ít nhất là giảm nhẹ các vấn đề được nêu ra ở phía trên được thực hiện thông qua việc khởi tạo cẩn thận các vector trọng số. 
Bằng cách này ta có thể chắc chắn rằng (ít nhất là ban đầu) các gradient không biến mất và chúng duy trì ở một tỉ lệ hợp lí trong đó trọng số mạng không phân kỳ.
Cộng thêm sự chăm sóc đặc biệt trong quá trình tối ưu hóa và điều chuẩn phù hợp sẽ đảm bảo mọi thứ không bao giờ trở nên quá tệ.

<!--
### Default Initialization
-->

### Khởi tạo Mặc định

<!--
In the previous sections, e.g., in :numref:`sec_linear_gluon`, we used `net.initialize(init.Normal(sigma=0.01))` to initialize the values of our weights.
If the initialization method is not specified, such as `net.initialize()`, 
MXNet will use the default random initialization method, sampling each weight parameter from  the uniform distribution $U[-0.07, 0.07]$ and setting the bias parameters to $0$.
Both choices tend to work well in practice for moderate problem sizes.
-->

Trong các phần trước, ví dụ, trong :numref:`sec_linear_gluon`, ta đã sử dụng `net.initialize(init.Normal(sigma=0.01))` để khởi tạo các giá trị cho trọng số.
Nếu phương thức khởi tạo không được xác định rõ, như là `net.initialize()`, MNXet sẽ sử dụng phương thức khởi tạo mặc định ngẫu nhiên: mỗi thành tố của trọng tham số được lấy mẫu ngẫu nhiên với phân phối đồng đều $U[-0.07, 0.07]$ và các tham số điều chỉnh đều được đưa về giá trị $0$.
Cả hai lựa chọn đều hoạt động tốt trong thực tế cho các vấn đề cỡ trung. 

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Xavier Initialization
-->

### Khởi tạo Xavier

<!--
Let's look at the scale distribution of the activations of the hidden units $h_{i}$ for some layer. 
They are given by
-->

Ta hãy xem phân phối tỉ lệ của giá trị kích hoạt các nút ẩn $h_{i}$ cho một vài tầng được đưa ra bởi

$$h_{i} = \sum_{j=1}^{n_\mathrm{in}} W_{ij} x_j.$$

<!--
The weights $W_{ij}$ are all drawn independently from the same distribution.
Furthermore, let's assume that this distribution has zero mean and variance $\sigma^2$ (this does not mean that the distribution has to be Gaussian, just that mean and variance need to exist).
For now, let's assume that the inputs to layer $x_j$ also have zero mean and variance $\gamma^2$ and that they are independent of $\mathbf{W}$.
In this case, we can compute mean and variance of $h_i$ as follows:
-->

Các trọng số $W_{ij}$ đều được lấy mẫu độc lập từ cùng một phân phối.
Hơn nữa, ta giả sử rằng phân phối này có trung bình bằng 0 và phương sai $\sigma^2$ (điều này không có nghĩa là phân phối đấy phải là dạng Gaussian, chỉ có nghĩa số trung bình và phương sai cần phải tồn tại).
Ta không thực sự có nhiều kiểm soát trên các đầu vào trên tầng $x_j$ nhưng hãy tiếp tục với giả định hơi phi thực tế là chúng có trung bình bằng 0, phương sai là $\gamma^2$ và độc lập với $\mathbf{W}$.
Trong trường hợp này, ta có thể tính toán số trung bình và phương sai của $h_i$ theo cách sau:

$$
\begin{aligned}
    E[h_i] & = \sum_{j=1}^{n_\mathrm{in}} E[W_{ij} x_j] = 0, \\
    E[h_i^2] & = \sum_{j=1}^{n_\mathrm{in}} E[W^2_{ij} x^2_j] \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[W^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

<!--
One way to keep the variance fixed is to set $n_\mathrm{in} \sigma^2 = 1$.
Now consider backpropagation.
There we face a similar problem, albeit with gradients being propagated from the top layers.
That is, instead of $\mathbf{W} \mathbf{w}$, we need to deal with $\mathbf{W}^\top \mathbf{g}$, where $\mathbf{g}$ is the incoming gradient from the layer above.
Using the same reasoning as for forward propagation, we see that the gradients' variance can blow up unless $n_\mathrm{out} \sigma^2 = 1$.
This leaves us in a dilemma: we cannot possibly satisfy both conditions simultaneously.
Instead, we simply try to satisfy:
-->

Một cách để giữ phương sai cố định là đặt $n_\mathrm{in} \sigma^2 = 1$.
Bây giờ hãy xem xét lan truyền ngược.
Ở đó ta phải đối mặt với vấn đề tương tự, mặc dù gradients được truyền từ các lớp trên cùng.
Đó là, thay vì $\mathbf{W} \mathbf{w}$, ta cần đối phó với $\mathbf{W}^\top \mathbf{g}$, trong đó $\mathbf{g}$ là gradient đến từ lớp phía trên.
Sử dụng lý do tương tự như thế để lan truyền xuôi, ta thấy phương sai của các gradient có thể tăng lên trừ khi $n_\mathrm{out} \sigma^2 = 1$.
Điều này làm ta rơi vào tình trạng khó xử: ta không thể thỏa mãn hai điều kiện cùng một lúc.
Thay vào đó, ta chỉ đơn giản cố gắng thỏa mãn:

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

<!--
This is the reasoning underlying the eponymous Xavier initialization :cite:`Glorot.Bengio.2010`.
It works well enough in practice.
For Gaussian random variables, the Xavier initialization picks a normal distribution with zero mean and variance $\sigma^2 = 2/(n_\mathrm{in} + n_\mathrm{out})$.
For uniformly distributed random variables $U[-a, a]$, note that their variance is given by $a^2/3$.
Plugging $a^2/3$ into the condition on $\sigma^2$ yields that we should initialize uniformly with $U\left[-\sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}, \sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}\right]$.
-->

<!-- UPDATE
This is the reasoning underlying the now-standard and practically beneficial *Xavier* initialization, named for its creator :cite:`Glorot.Bengio.2010`.
Typically, the Xavier initialization samples weights from a Gaussian distribution with zero mean and variance $\sigma^2 = 2/(n_\mathrm{in} + n_\mathrm{out})$.
We can also adapt Xavier's intuition to choose the variance when sampling weightsfrom a uniform distribution.
Note the distribution $U[-a, a]$ has variance $a^2/3$.
Plugging $a^2/3$ into our condition on $\sigma^2$, yields the suggestion to initialize according to $U\left[-\sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}, \sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}\right]$.

-->

Đây là lý do làm cơ sở cho cách khởi tạo Xavier :cite:`Glorot.Bengio.2010`.
Nó hoạt động đủ tốt trong thực tế.
Đối với các biến ngẫu nhiên Gaussian, khởi tạo Xavier chọn phân phối chuẩn với trung bình bằng 0 và phương sai $\sigma^2 = 2/(n_\mathrm{in} + n_\mathrm{out})$.  
Dành cho các biến ngẫu nhiên được phân bố đồng đều $U[-a, a]$, chú ý rằng phương sai của chúng được cho bởi $a^2/3$.
Thay $a^2/3$ vào điều kiện trên $\sigma^2$, ta có lý do nên khởi tạo đồng đều với $U\left[-\sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}, \sqrt{6/(n_\mathrm{in} + n_\mathrm{out})}\right]$. 

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### Beyond
-->

### Xa hơn nữa

<!--
The reasoning above barely scratches the surface of modern approaches to parameter initialization.
In fact, MXNet has an entire [`mxnet.initializer`](https://mxnet.apache.org/api/python/docs/api/initializer/index.html) module implementing over a dozen different heuristics.
Moreover, initialization continues to be a hot area of inquiry within research into the fundamental theory of neural network optimization.
Some of these heuristics are especially suited for when parameters are tied 
(i.e., when parameters of in different parts the network are shared), for super-resolution, sequence models, and related problems.
We recommend that the interested reader take a closer look at what is offered as part of this module, and investigate the recent research on parameter initialization.
Perhaps you may come across a recent clever idea and contribute its implementation to MXNet, or you may even invent your own scheme!
-->

<!-- UPDATE
The reasoning above barely scratches the surfaceof modern approaches to parameter initialization.
In fact, MXNet has an entire `mxnet.initializer` moduleimplementing over a dozen different heuristics.
Moreover, parameter initialization continues to bea hot area of fundamental research in deep learning.
Among these are heuristics specialized for tied (shared) parameters, super-resolution, sequence models, and other situations.
If the topic interests you we suggest a deep dive into this module's offerings, reading the papers that proposed and analyzed each heuristic, and then exploring the latest publications on the topic.
Perhaps you will stumble across (or even invent!) a clever idea and contribute an implementation to MXNet.
-->

Các lập luận đưa ra ở trên mới chỉ chạm tới bề mặt của những kỹ thuật khởi tạo tham số hiện đại.
Trên thực tế, MXNet có nguyên một mô-đun [`mxnet.initializer`](https://mxnet.apache.org/api/python/docs/api/initializer/index.html) với hàng chục các phương pháp khởi tạo dựa theo thực nghiệm khác nhau đã được lập trình sẵn.
Hơn nữa, cách khởi tạo vẫn đang là một chủ đề rất được quan tâm trong các nghiên cứu lý thuyết căn bản về tối ưu hóa mạng nơ-ron.
Một số phương pháp thực nghiệm này đặc biệt phù hợp khi tham số bị ràng buộc (tức tham số của các phần khác nhau trong mạng được chia sẻ với nhau), trong nhiệm vụ siêu phân giải, mô hình chuỗi và những vấn đề liên quan. 
Chúng tôi gợi ý với những độc giả quan tâm có thể tìm xem kĩ hơn về các kỹ thuật có trong mô-đun này và tìm hiểu thêm những nghiên cứu gần đây về vấn đề khởi tạo tham số.
Có thể bạn sẽ gặp được một ý tưởng hay và đóng góp cách lập trình chúng vào MXNet, hoặc thậm chí là tự phát minh ra phương pháp của riêng mình.


<!--
## Summary
-->

## Tóm tắt

<!--
* Vanishing and exploding gradients are common issues in very deep networks, unless great care is taking to ensure that gradients and parameters remain well controlled.
* Initialization heuristics are needed to ensure that at least the initial gradients are neither too large nor too small.
* The ReLU addresses one of the vanishing gradient problems, namely that gradients vanish for very large inputs. This can accelerate convergence significantly.
* Random initialization is key to ensure that symmetry is broken before optimization.
-->

<!-- UPDATE
* Vanishing and exploding gradients are common issues in deep networks. Great care in parameter initialization is required to ensure that gradients and parameters remain well controlled.
* Initialization heuristics are needed to ensure that the initial gradients are neither too large nor too small.
* ReLU activation functions mitigate the vanishing gradient problem. This can accelerate convergence.
* Random initialization is key to ensure that symmetry is broken before optimization.
-->

* Tiêu biến hay bùng nổ gradient đều là những vấn đề phổ biến trong những mạng rất sâu, trừ khi ta có nhiều sự quan tâm nhằm đảm bảo gradient và các tham số vẫn được kiểm soát tốt.
* Các kĩ thuật khởi tạo tham số dựa trên kinh nghiệm là cần thiết để đảm bảo ít nhất rằng gradient ban đầu không bị quá lớn hay quá nhỏ.
* ReLU giải quyết một trong những vấn đề về tiêu biến gradient, cụ thể là việc tiêu biến gradient cho các đầu vào rất lớn. Điều này có thể tăng tốc độ hội tụ đáng kể.
* Khởi tạo ngẫu nhiên là chìa khóa để đảm bảo tính đối xứng bị phá vỡ trước khi tối ưu hóa.

<!--
## Exercises
-->

## Bài tập

<!--
1. Can you design other cases where a neural network might exhibit symmetry requiring breaking besides the permutation symmetry in a multilayer pereceptron's layers?
2. Can we initialize all weight parameters in linear regression or in softmax regression to the same value?
3. Look up analytic bounds on the eigenvalues of the product of two matrices. What does this tell you about ensuring that gradients are well conditioned?
4. If we know that some terms diverge, can we fix this after the fact? Look at the paper on LARS for inspiration :cite:`You.Gitman.Ginsburg.2017`.
-->

1. Bạn có thể thiết kế các trường hợp phá vỡ đối xứng khác bên cạnh đối xứng hoán vị?
2. Ta có thể khởi tạo tất cả trọng số ở trong mạng hồi quy tuyến tính hoặc trong hồi quy softmax cùng một giá trị hay không?
3. Hãy tìm hiểu thêm về phân cách tích ràng buộc trị riêng của phép nhân 2 ma trận. Nó cho ta biết được gì về điều kiện đảm bảo gradient có độ lớn vừa phải?
4. Nếu biết rằng mô hình có một vài số hạng phân kỳ, bạn có thể khắc phục vấn đề này không? Bạn có thể tìm cảm hứng từ bài báo LARS :cite:`You.Gitman.Ginsburg.2017`.

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2345)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2345)
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
* Lý Phi Long
* Lê Khắc Hồng Phúc
* Phạm Minh Đức

<!-- Phần 2 -->
* Nguyễn Văn Tâm
* Lê Khắc Hồng Phúc
* Phạm Minh Đức

<!-- Phần 3 -->
* Trần Yến Thy
* Lê Khắc Hồng Phúc

<!-- Phần 4 -->
* Trần Yến Thy
* Lê Khắc Hồng Phúc

<!-- Phần 5 -->
* Bùi Chí Minh
* Đoàn Võ Duy Thanh
* Lê Khắc Hồng Phúc
* Lý Phi Long

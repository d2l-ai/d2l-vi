<!--
# RMSProp
-->

# RMSProp
:label:`sec_rmsprop`

<!--
One of the key issues in :numref:`sec_adagrad` is that the learning rate decreases at a predefined schedule of effectively $\mathcal{O}(t^{-\frac{1}{2}})$.
While this is generally appropriate for convex problems, it might not be ideal for nonconvex ones, such as those encountered in deep learning.
Yet, the coordinate-wise adaptivity of Adagrad is highly desirable as a preconditioner.
-->

Một trong những vấn đề then chốt trong :numref:`sec_adagrad` là tốc độ học thực tế được giảm theo một thời điểm được định nghĩa sẵn $\mathcal{O}(t^{-\frac{1}{2}})$.
Nhìn chung, cách này thích hợp với các bài toán lồi nhưng có thể không phải giải pháp lý tưởng cho những bài toán không lồi, chẳng hạn những bài toán gặp phải trong học sâu. 
Tuy vậy, khả năng thích ứng theo tọa độ của Adagrad là rất tuyệt vời cho một bộ tiền điều kiện (*preconditioner*). 

<!--
:cite:`Tieleman.Hinton.2012` proposed the RMSProp algorithm as a simple fix to decouple rate scheduling from coordinate-adaptive learning rates.
The issue is that Adagrad accumulates the squares of the gradient $\mathbf{g}_t$ into a state vector $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$.
As a result $\mathbf{s}_t$ keeps on growing without bound due to the lack of normalization, essentially linearly as the algorithm converges.
-->


:cite:`Tieleman.Hinton.2012` đề xuất thuật toán RMSProp như một bản vá đơn giản để tách rời tốc độ định thời ra khỏi tốc độ học thay đổi theo tọa độ (*coordinate-adaptive*). 
Vấn đề ở đây là Adagrad cộng dồn tổng bình phương của gradient $\mathbf{g}_t$ vào vector trạng thái $\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2$. 
Kết quả là, do không có phép chuẩn hóa, $\mathbf{s}_t$ vẫn tiếp tục tăng tuyến tính không ngừng trong quá trình hội tụ của thuật toán. 

<!--
One way of fixing this problem would be to use $\mathbf{s}_t / t$.
For reasonable distributions of $\mathbf{g}_t$ this will converge.
Unfortunately it might take a very long time until the limit behavior starts to matter since the procedure remembers the full trajectory of values.
An alternative is to use a leaky average in the same way we used in the momentum method, i.e., $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ for some parameter $\gamma > 0$.
Keeping all other parts unchanged yields RMSProp.
-->

Vấn đề này có thể được giải quyết bằng cách sử dụng $\mathbf{s}_t / t$.
Với phân phối $\mathbf{g}_t$ hợp lý, thuật toán sẽ hội tụ.
Đáng tiếc là có thể mất rất nhiều thời gian cho đến khi các tính chất tại giới hạn bắt đầu có ảnh hưởng, bởi thuật toán này ghi nhớ toàn bộ quỹ đạo của các giá trị.
Một cách khác là sử dụng trung bình rò rỉ tương tự như trong phương pháp động lượng, tức là $\mathbf{s}_t \leftarrow \gamma \mathbf{s}_{t-1} + (1-\gamma) \mathbf{g}_t^2$ cho các tham số $\gamma > 0$.
Giữ nguyên tất cả các phần khác và ta có thuật toán RMSProp.

<!--
## The Algorithm
-->

## Thuật toán

<!--
Let us write out the equations in detail.
-->

Chúng ta hãy viết các phương trình ra một cách chi tiết. 


$$\begin{aligned}
    \mathbf{s}_t & \leftarrow \gamma \mathbf{s}_{t-1} + (1 - \gamma) \mathbf{g}_t^2, \\
    \mathbf{x}_t & \leftarrow \mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t.
\end{aligned}$$


<!--
The constant $\epsilon > 0$ is typically set to $10^{-6}$ to ensure that we do not suffer from division by zero or overly large step sizes.
Given this expansion we are now free to control the learning rate $\eta$ independently of the scaling that is applied on a per-coordinate basis.
In terms of leaky averages we can apply the same reasoning as previously applied in the case of the momentum method.
Expanding the definition of $\mathbf{s}_t$ yields
-->


Hằng số $\epsilon > 0$ thường được đặt bằng $10^{-6}$ để đảm bảo rằng chúng ta sẽ không gặp vấn đề khi chia cho 0 hoặc kích thước bước quá lớn.
Với khai triển này, bây giờ chúng ta có thể tự do kiểm soát tốc độ học $\eta$ độc lập với phép biến đổi tỉ lệ được áp dụng cho từng tọa độ.
Về mặt trung bình rò rỉ, chúng ta có thể áp dụng các lập luận tương tự như trước trong phương pháp động lượng.
Khai triển định nghĩa $\mathbf{s}_t$ ta có

$$
\begin{aligned}
\mathbf{s}_t & = (1 - \gamma) \mathbf{g}_t^2 + \gamma \mathbf{s}_{t-1} \\
& = (1 - \gamma) \left(\mathbf{g}_t^2 + \gamma \mathbf{g}_{t-1}^2 + \gamma^2 \mathbf{g}_{t-2} + \ldots, \right).
\end{aligned}
$$


<!--
As before in :numref:`sec_momentum` we use $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$.
Hence the sum of weights is normalized to $1$ with a half-life time of an observation of $\gamma^{-1}$.
Let us visualize the weights for the past 40 timesteps for various choices of $\gamma$.
-->

Tương tự như :numref:`sec_momentum`, ta có $1 + \gamma + \gamma^2 + \ldots, = \frac{1}{1-\gamma}$. 
Do đó, tổng trọng số được chuẩn hóa bằng $1$ và chu kỳ bán rã của một quan sát là $\gamma^{-1}$. 
Hãy cùng minh họa trực quan các trọng số này trong vòng 40 bước thời gian trước đó với các giá trị $\gamma$ khác nhau. 


```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx

npx.set_np()

d2l.set_figsize()
gammas = [0.95, 0.9, 0.8, 0.7]
for gamma in gammas:
    x = np.arange(40).asnumpy()
    d2l.plt.plot(x, (1-gamma) * gamma ** x, label=f'gamma = {gamma:.2f}')
d2l.plt.xlabel('time');
```


<!--
## Implementation from Scratch
-->

## Lập trình Từ đầu

<!--
As before we use the quadratic function $f(\mathbf{x})=0.1x_1^2+2x_2^2$ to observe the trajectory of RMSProp.
Recall that in :numref:`sec_adagrad`, when we used Adagrad with a learning rate of 0.4, 
the variables moved only very slowly in the later stages of the algorithm since the learning rate decreased too quickly.
Since $\eta$ is controlled separately this does not happen with RMSProp.
-->

Như trước đây, chúng ta sử dụng hàm bậc hai $f(\mathbf{x})=0.1x_1^2+2x_2^2$ để quan sát quỹ đạo của RMSProp.
Nhớ lại trong :numref:`sec_adagrad`, khi chúng ta sử dụng Adagrad với tốc độ học bằng 0.4,
các biến di chuyển rất chậm trong các giai đoạn sau của thuật toán do tốc độ học giảm quá nhanh.
Do $\eta$ được kiểm soát riêng biệt, nên điều này không xảy ra với RMSProp.

```{.python .input}
def rmsprop_2d(x1, x2, s1, s2):
    g1, g2, eps = 0.2 * x1, 4 * x2, 1e-6
    s1 = gamma * s1 + (1 - gamma) * g1 ** 2
    s2 = gamma * s2 + (1 - gamma) * g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta, gamma = 0.4, 0.9
d2l.show_trace_2d(f_2d, d2l.train_2d(rmsprop_2d))
```


<!--
Next, we implement RMSProp to be used in a deep network.
This is equally straightforward.
-->

Tiếp theo, chúng ta hãy lập trình thuật toán RMSProp để sử dụng trong một mạng nơ-ron sâu. 
Cách lập trình không quá phức tạp. 


```{.python .input  n=22}
def init_rmsprop_states(feature_dim):
    s_w = np.zeros((feature_dim, 1))
    s_b = np.zeros(1)
    return (s_w, s_b)

def rmsprop(params, states, hyperparams):
    gamma, eps = hyperparams['gamma'], 1e-6
    for p, s in zip(params, states):
        s[:] = gamma * s + (1 - gamma) * np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```


<!--
We set the initial learning rate to 0.01 and the weighting term $\gamma$ to 0.9.
That is, $\mathbf{s}$ aggregates on average over the past $1/(1-\gamma) = 10$ observations of the square gradient.
-->

Chúng ta khởi tạo tốc độ học ban đầu bằng 0.01 và trọng số $\gamma$ bằng 0.9. 
Nghĩa là, $\mathbf{s}$ là tổng trung bình của $1/(1-\gamma) = 10$ quan sát bình phương gradient trong quá khứ.   


```{.python .input  n=24}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(rmsprop, init_rmsprop_states(feature_dim),
               {'lr': 0.01, 'gamma': 0.9}, data_iter, feature_dim);
```


<!--
## Concise Implementation
-->

## Lập trình Súc tích

<!--
Since RMSProp is a rather popular algorithm it is also available in the `Trainer` instance.
All we need to do is instantiate it using an algorithm named `rmsprop`, assigning $\gamma$ to the parameter `gamma1`.
-->

Do RMSProp là thuật toán khá phổ biến, nó cũng được tích hợp sẵn trong thực thể `Trainer`. 
Những gì ta cần phải làm là khởi tạo thuật toán có tên là `rmsprop`, với $\gamma$ được gán cho tham số `gamma1`. 


```{.python .input  n=29}
d2l.train_concise_ch11('rmsprop', {'learning_rate': 0.01, 'gamma1': 0.9},
                       data_iter)
```


<!--
## Summary
-->

## Tóm tắt

<!--
* RMSProp is very similar to Adagrad insofar as both use the square of the gradient to scale coefficients.
* RMSProp shares with momentum the leaky averaging. However, RMSProp uses the technique to adjust the coefficient-wise preconditioner.
* The learning rate needs to be scheduled by the experimenter in practice.
* The coefficient $\gamma$ determines how long the history is when adjusting the per-coordinate scale.
-->

* Thuật toán RMSProp rất giống với Adagrad ở chỗ cả hai đều sử dụng bình phương của gradient để thay đổi tỉ lệ hệ số. 
* RMSProp có điểm chung với phương pháp động lượng là chúng đều sử dụng trung bình rò rỉ. Tuy nhiên, RMSProp sử dụng kỹ thuật này để điều chỉnh tiền điều kiện theo hệ số. 
* Trong thực tế, tốc độ học cần được định thời bởi người lập trình.  
* Hệ số $\gamma$ xác định độ dài thông tin quá khứ được sử dụng khi điều chỉnh tỉ lệ theo từng tọa độ. 

<!--
## Exercises
-->

## Bài tập

<!--
1. What happens experimentally if we set $\gamma = 1$? Why?
2. Rotate the optimization problem to minimize $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. What happens to the convergence?
3. Try out what happens to RMSProp on a real machine learning problem, such as training on Fashion-MNIST. Experiment with different choices for adjusting the learning rate.
4. Would you want to adjust $\gamma$ as optimization progresses? How sensitive is RMSProp to this?
-->

1. Điều gì sẽ xảy ra nếu ta đặt $\gamma = 1$? Giải thích tại sao? 
2. Biến đổi bài toán tối ưu thành cực tiểu hóa $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Sự hội tụ sẽ diễn ra như thế nào? 
3. Hãy thử áp dụng RMSProp cho một bài toán học máy cụ thể, chẳng hạn như huấn luyện trên tập Fashion-MNIST. Hãy thí nghiệm với các tốc độ học khác nhau. 
4. Bạn có muốn điều chỉnh $\gamma$ khi việc tối ưu đang tiến triển không? Hãy cho biết độ nhạy của RMSProp với việc điều chỉnh này? 



## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/356)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:


* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Lê Quang Nhật
* Phạm Minh Đức
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh


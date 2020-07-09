<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Adadelta
-->

# Adadelta
:label:`sec_adadelta`

<!--
Adadelta is yet another variant of AdaGrad.
The main difference lies in the fact that it decreases the amount by which the learning rate is adaptive to coordinates.
Moreover, traditionally it referred to as not having a learning rate since it uses the amount of change itself as calibration for future change.
The algorithm was proposed in :cite:`Zeiler.2012`.
It is fairly straightforward, given the discussion of previous algorithms so far.
-->

Adadelta là một biến thể khác của AdaGrad.
Điểm khác biệt chính là Adadelta giảm mức độ thay đổi của tốc độ học theo toạ độ.
Hơn nữa, Adadelta thường được biết đến là thuật toán không sử dụng tốc độ học vì nó dựa trên chính lượng thay đổi hiện tại để căn chỉnh lượng thay đổi trong tương lai.
Thuật toán Adadelta được đề xuất trong :cite:`Zeiler.2012`.
Nó khá đơn giản dựa trên những thảo luận về các thuật toán trước đây.

<!--
## The Algorithm
-->

## Thuật toán


<!--
In a nutshell Adadelta uses two state variables, $\mathbf{s}_t$ to store a leaky average of the second moment of the gradient 
and $\Delta\mathbf{x}_t$ to store a leaky average of the second moment of the change of parameters in the model itself.
Note that we use the original notation and naming of the authors for compatibility with other publications and implementations 
(there is no other real reason why one should use different Greek variables to indicate a parameter serving the same purpose in momentum, Adagrad, RMSProp, and Adadelta).
The parameter du jour is $\rho$. We obtain the following leaky updates:
-->

Nói ngắn gọn, Adadelta sử dụng hai biến trạng thái, $\mathbf{s}_t$ để lưu trữ trung bình rò rỉ mô-men bậc hai của gradient
và $\Delta\mathbf{x}_t$ để lưu trữ trung bình rò rỉ mô-men bậc hai các thay đổi của các tham số trong mô hình.
Lưu ý rằng chúng ta sử dụng các ký hiệu và cách đặt tên nguyên bản của chính tác giả để nhất quán với các nghiên cứu khác và cách lập trình
(không có lý do nào để sử dụng các kí hiệu La Mã khác cho tham số có cùng mục đích trong các thuật toán động lượng, Adagrad, RMSProp, và Adadelta).
Tham số suy giảm là $\rho$. <!-- note for reviser: $rho$ trong bài báo gốc là `decay rate` -->
Chúng ta có được các cập nhật rò rỉ sau:


$$\begin{aligned}
    \mathbf{s}_t & = \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t^2, \\
    \mathbf{g}_t' & = \sqrt{\frac{\Delta\mathbf{x}_{t-1} + \epsilon}{\mathbf{s}_t + \epsilon}} \odot \mathbf{g}_t, \\
    \mathbf{x}_t  & = \mathbf{x}_{t-1} - \mathbf{g}_t', \\
    \Delta \mathbf{x}_t & = \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) \mathbf{x}_t^2.
\end{aligned}$$


<!--
The difference to before is that we perform updates with the rescaled gradient $\mathbf{g}_t'$ which is computed by taking the ratio between 
the average squared rate of change and the average second moment of the gradient.
The use of $\mathbf{g}_t'$ is purely for notational convenience.
In practice we can implement this algorithm without the need to use additional temporary space for $\mathbf{g}_t'$.
As before $\eta$ is a parameter ensuring nontrivial numerical results, i.e., avoiding zero step size or infinite variance. Typically we set this to $\eta = 10^{-5}$.
-->

Điểm khác biệt so với trước là ta thực hiện các cập nhật với gradient $\mathbf{g}_t'$ được chuyển đổi giá trị bằng cách lấy căn bậc hai tỷ lệ giữa trung bình bình phương của tốc độ thay đổi và trung bình mô-men bậc hai của gradient.
Sử dụng $\mathbf{g}_t'$ chỉ đơn thuần với mục đích thuận tiện cho việc ký hiệu.
Trong thực tế chúng ta có thể lập trình thuật toán này mà không cần phải sử dụng thêm bộ nhớ tạm cho $\mathbf{g}_t'$.
Như trước đây $\epsilon$ là tham số đảm bảo ta sẽ thu được kết quả xấp xỉ ý nghĩa, tức để tránh kích thước bước bằng $0$ hoặc phương sai vô hạn. Thông thường, chúng ta đặt $\epsilon = 10^{-5}$.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Implementation
-->

## Lập trình

<!--
Adadelta needs to maintain two state variables for each variable, $\mathbf{s}_t$ and $\Delta\mathbf{x}_t$. This yields the following implementation.
-->

Thuật toán Adadelta cần duy trì hai biến trạng thái cho từng biến $\mathbf{s}_t$ và $\Delta\mathbf{x}_t$. Do đó ta lập trình như sau.


```{.python .input  n=11}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = np.zeros((feature_dim, 1)), np.zeros(1)
    delta_w, delta_b = np.zeros((feature_dim, 1)), np.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        # In-place updates via [:]
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```


<!--
Choosing $\rho = 0.9$ amounts to a half-life time of 10 for each parameter update. This tends to work quite well. We get the following behavior.
-->

Chọn $\rho = 0.9$ bằng thời gian một nửa chu kỳ bán rã của 10 cho mỗi lần cập nhật tham số. Cách này thường hoạt động khá tốt. Hoạt động của thuật toán thu được như sau.


```{.python .input  n=12}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```


<!--
For a concise implementation we simply use the `adadelta` algorithm from the `Trainer` class. This yields the following one-liner for a much more compact invocation.
-->

Để lập trình súc tích, ta chỉ đơn giản sử dụng thuật toán `adadelta` trực tiếp từ lớp `Trainer`. Nhờ vậy mà thuật toán được gọi chỉ với một dòng lệnh khá ngắn gọn. 


```{.python .input  n=9}
d2l.train_gluon_ch11('adadelta', {'rho': 0.9}, data_iter)
```


<!--
## Summary
-->

## Tóm tắt

<!--
* Adadelta has no learning rate parameter. Instead, it uses the rate of change in the parameters itself to adapt the learning rate.
* Adadelta requires two state variables to store the second moments of gradient and the change in parameters.
* Adadelta uses leaky averages to keep a running estimate of the appropriate statistics.
-->

* Adadelta không sử dụng tham số tốc độ học. Thay vào đó, nó sử dụng tốc độ thay đổi của chính các tham số của nó để điều chỉnh tốc độ học.
* Adadelta cần sử dụng hai biến trạng thái để lưu trữ các mô-men bậc hai của gradient và của thay đổi trong các tham số.
* Adadelta sử dụng trung bình rò rỉ để lưu ước lượng thống kê động thích hợp.

<!--
## Exercises
-->

## Bài tập

<!--
1. Adjust the value of $\rho$. What happens?
2. Show how to implement the algorithm without the use of $\mathbf{g}_t'$. Why might this be a good idea?
3. Is Adadelta really learning rate free? Could you find optimization problems that break Adadelta?
4. Compare Adadelta to Adagrad and RMS prop to discuss their convergence behavior.
-->

1. Điều gì xảy ra khi điều chỉnh giá trị của $\rho$?
2. Hãy lập trình thuật toán trên mà không cần dùng biến $\mathbf{g}_t'$. Giải thích tại sao đây có thể là một ý tưởng tốt?
3. Adadelta có thực sự không cần tốc độ học? Bạn đọc có thể chỉ ra các bài toán tối ưu mà không thoả mãn Adadelta?
4. Hãy so sánh Adadelta với Adagrad và RMSprop để thảo luận về sự hội tụ của từng thuật toán.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC =================================== -->



## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2377)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Văn Quang
* Nguyễn Văn Cường

<!-- Phần 2 -->
* Nguyễn Văn Quang

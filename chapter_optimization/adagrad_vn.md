<!--
# Adagrad
-->

# Adagrad
:label:`sec_adagrad`

<!--
Let us begin by considering learning problems with features that occur infrequently.
-->

Để khởi động, hãy cùng xem xét các bài toán với những đặc trưng xuất hiện không thường xuyên.

<!--
## Sparse Features and Learning Rates
-->

## Đặc trưng Thưa và Tốc độ Học

<!--
Imagine that we are training a language model.
To get good accuracy we typically want to decrease the learning rate as we keep on training, usually at a rate of $\mathcal{O}(t^{-\frac{1}{2}})$ or slower.
Now consider a model training on sparse features, i.e., features that occur only infrequently.
This is common for natural language, e.g., it is a lot less likely that we will see the word *preconditioning* than *learning*.
However, it is also common in other areas such as computational advertising and personalized collaborative filtering.
After all, there are many things that are of interest only for a small number of people.
-->

Hãy tưởng tượng ta đang huấn luyện một mô hình ngôn ngữ.
Để đạt độ chính xác cao ta thường muốn giảm dần tốc độ học trong quá trình huấn luyện, thường là với tỉ lệ $\mathcal{O}(t^{-\frac{1}{2}})$ hoặc chậm hơn.
Xét một mô hình huấn luyện dựa trên những đặc trưng thưa, tức là các đặc trưng hiếm khi xuất hiện.
Đây là điều thường gặp trong ngôn ngữ tự nhiên, ví dụ từ *preconditioning* hiếm gặp hơn nhiều so với *learning*.
Tuy nhiên, đây cũng là vấn đề thường gặp trong nhiều mảng khác như quảng cáo điện toán (*computational advertising*) và lọc cộng tác (*collaborative filtering*).
Xét cho cùng, có rất nhiều thứ mà chỉ có một nhóm nhỏ người chú ý đến.

<!--
Parameters associated with infrequent features only receive meaningful updates whenever these features occur.
Given a decreasing learning rate we might end up in a situation where the parameters for common features converge rather quickly to their optimal values, 
whereas for infrequent features we are still short of observing them sufficiently frequently before their optimal values can be determined.
In other words, the learning rate either decreases too slowly for frequent features or too quickly for infrequent ones.
-->

Các tham số liên quan đến các đặc trưng thưa chỉ được cập nhật khi những đặc trưng này xuất hiện.
Đối với tốc độ học giảm dần, ta có thể gặp phải trường hợp các tham số của những đặc trưng phổ biến hội tụ khá nhanh đến giá trị tối ưu,
trong khi đối với các đặc trưng thưa, ta không có đủ số lượng dữ liệu thích đáng để xác định giá trị tối ưu của chúng.
Nói một cách khác, tốc độ học hoặc là giảm quá chậm đối với các đặc trưng phổ biến hoặc là quá nhanh đối với các đặc trưng hiếm.

<!--
A possible hack to redress this issue would be to count the number of times we see a particular feature and to use this as a clock for adjusting learning rates.
That is, rather than choosing a learning rate of the form $\eta = \frac{\eta_0}{\sqrt{t + c}}$ we could use $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$.
Here $s(i, t)$ counts the number of nonzeros for feature $i$ that we have observed up to time $t$.
This is actually quite easy to implement at no meaningful overhead.
However, it fails whenever we do not quite have sparsity but rather just data where the gradients are often very small and only rarely large.
After all, it is unclear where one would draw the line between something that qualifies as an observed feature or not.
-->

Một mẹo để khắc phục vấn đề này là đếm số lần ta gặp một đặc trưng nhất định và sử dụng nó để điều chỉnh tốc độ học.
Tức là thay vì chọn tốc độ học theo công thức $\eta = \frac{\eta_0}{\sqrt{t + c}}$ ta có thể sử dụng $\eta_i = \frac{\eta_0}{\sqrt{s(i, t) + c}}$.
Trong đó $s(i, t)$ là số giá trị khác không của đặc trưng $i$ ta quan sát được đến thời điểm $t$.
Công thức này khá dễ để lập trình và không tốn thêm bao nhiêu công sức.
Tuy nhiên, cách này thất bại trong trường hợp khi đặc trưng không hẳn là thưa, chỉ là có gradient nhỏ và hiếm khi đạt giá trị lớn.
Xét cho cùng, ta khó có thể phân định rõ ràng khi nào thì một đặc trưng là đã được quan sát hay chưa.

<!--
Adagrad by :cite:`Duchi.Hazan.Singer.2011` addresses this by replacing the rather crude counter $s(i, t)$ by an aggregate of the squares of previously observed gradients.
In particular, it uses $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ as a means to adjust the learning rate.
This has two benefits: first, we no longer need to decide just when a gradient is large enough.
Second, it scales automatically with the magnitude of the gradients.
Coordinates that routinely correspond to large gradients are scaled down significantly, whereas others with small gradients receive a much more gentle treatment.
In practice this leads to a very effective optimization procedure for computational advertising and related problems.
But this hides some of the additional benefits inherent in Adagrad that are best understood in the context of preconditioning.
-->

Adagrad được đề xuất trong :cite:`Duchi.Hazan.Singer.2011` đã giải quyết vấn đề này bằng cách thay đổi bộ đếm thô $s(i, t)$ bởi tổng bình phương của tất cả các gradient được quan sát trước đó.
Cụ thể, nó sử dụng $s(i, t+1) = s(i, t) + \left(\partial_i f(\mathbf{x})\right)^2$ làm công cụ để điều chỉnh tốc độ học.
Việc này đem lại hai lợi ích: trước tiên ta không cần phải quyết định khi nào thì gradient được coi là đủ lớn.
Thứ hai, nó tự động thay đổi giá trị tuỳ theo độ lớn của gradient.
Các tọa độ thường xuyên có gradient lớn bị giảm đi đáng kể, trong khi các tọa độ khác với gradient nhỏ được xử lý nhẹ nhàng hơn nhiều.
Phương pháp này trong thực tế đưa ra một quy trình tối ưu hoạt động rất hiệu quả trong quảng cáo điện toán và các bài toán liên quan.
Tuy nhiên, Adagrad vẫn còn ẩn chứa một vài lợi ích khác mà ta sẽ hiểu rõ nhất khi xét đến bối cảnh tiền điều kiện.


<!--
## Preconditioning
-->

## Tiền điều kiện

<!--
Convex optimization problems are good for analyzing the characteristics of algorithms.
After all, for most nonconvex problems it is difficult to derive meaningful theoretical guarantees, but *intuition* and *insight* often carry over.
Let us look at the problem of minimizing $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.
-->

Các bài toán tối ưu lồi rất phù hợp để phân tích đặc tính của các thuật toán.
Suy cho cùng, với đa số các bài toán không lồi ta khó có thể tìm được các chứng minh lý thuyết vững chắc. Tuy nhiên, *trực giác* và *ý nghĩa hàm chứa* suy ra từ các bài toán tối ưu lồi vẫn có thể được áp dụng.
Xét bài toán cực tiểu hóa $f(\mathbf{x}) = \frac{1}{2} \mathbf{x}^\top \mathbf{Q} \mathbf{x} + \mathbf{c}^\top \mathbf{x} + b$.

<!--
As we saw in :numref:`sec_momentum`, it is possible to rewrite this problem in terms of its eigendecomposition 
$\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ to arrive at a much simplified problem where each coordinate can be solved individually:
-->

Như ta đã thấy ở :numref:`sec_momentum`, ta có thể biến đổi bài toán sử dụng phép phân tích trị riêng $\mathbf{Q} = \mathbf{U}^\top \boldsymbol{\Lambda} \mathbf{U}$ nhằm biến đổi nó về dạng đơn giản hơn mà ta có thể xử lý trên từng tọa độ một:


$$f(\mathbf{x}) = \bar{f}(\bar{\mathbf{x}}) = \frac{1}{2} \bar{\mathbf{x}}^\top \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}}^\top \bar{\mathbf{x}} + b.$$


<!--
Here we used $\mathbf{x} = \mathbf{U} \mathbf{x}$ and consequently $\mathbf{c} = \mathbf{U} \mathbf{c}$.
The modified problem has as its minimizer $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ 
and minimum value $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$.
This is much easier to compute since $\boldsymbol{\Lambda}$ is a diagonal matrix containing the eigenvalues of $\mathbf{Q}$.
-->

Ở đây ta sử dụng $\mathbf{x} = \mathbf{U} \mathbf{x}$ và theo đó $\mathbf{c} = \mathbf{U} \mathbf{c}$.
Bài toán sau khi được biến đổi có các nghiệm cực tiểu (*minimizer*) $\bar{\mathbf{x}} = -\boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}}$ và giá trị nhỏ nhất $-\frac{1}{2} \bar{\mathbf{c}}^\top \boldsymbol{\Lambda}^{-1} \bar{\mathbf{c}} + b$.
Việc tính toán trở nên dễ dàng hơn nhiều do $\boldsymbol{\Lambda}$ là một ma trận đường chéo chứa các trị riêng của $\mathbf{Q}$.

<!--
If we perturb $\mathbf{c}$ slightly we would hope to find only slight changes in the minimizer of $f$.
Unfortunately this is not the case.
While slight changes in $\mathbf{c}$ lead to equally slight changes in $\bar{\mathbf{c}}$, this is not the case for the minimizer of $f$ (and of $\bar{f}$ respectively).
Whenever the eigenvalues $\boldsymbol{\Lambda}_i$ are large we will see only small changes in $\bar{x}_i$ and in the minimum of $\bar{f}$.
Conversely, for small $\boldsymbol{\Lambda}_i$ changes in $\bar{x}_i$ can be dramatic.
The ratio between the largest and the smallest eigenvalue is called the condition number of an optimization problem.
-->

Nếu ta làm nhiễu $\mathbf{c}$ một chút, ta sẽ mong rằng các nghiệm cực tiểu của $f$ cũng chỉ thay đổi không đáng kể.
Đáng tiếc thay, điều đó lại không xảy ra.
Mặc dù thay đổi $\mathbf{c}$ một chút dẫn đến $\bar{\mathbf{c}}$ cũng thay đổi một lượng tương ứng, các nghiệm cực tiểu của $f$ (cũng như $\bar{f}$) lại không như vậy.
Mỗi khi các trị riêng $\boldsymbol{\Lambda}_i$ mang giá trị lớn, ta sẽ thấy $\bar{x}_i$ và cực tiểu của $f$ thay đổi khá nhỏ.
Ngược lại, với $\boldsymbol{\Lambda}_i$ nhỏ, sự thay đổi $\bar{x}_i$ có thể là đáng kể.
Tỉ lệ giữa trị riêng lớn nhất và nhỏ nhất được gọi là hệ số điều kiện (*condition number*) của bài toán tối ưu.


$$\kappa = \frac{\boldsymbol{\Lambda}_1}{\boldsymbol{\Lambda}_d}.$$


<!--
If the condition number $\kappa$ is large, it is difficult to solve the optimization problem accurately.
We need to ensure that we are careful in getting a large dynamic range of values right.
Our analysis leads to an obvious, albeit somewhat naive question: couldn't we simply "fix" the problem by distorting the space such that all eigenvalues are $1$.
In theory this is quite easy: we only need the eigenvalues and eigenvectors of $\mathbf{Q}$ to rescale the problem 
from $\mathbf{x}$ to one in $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$.
In the new coordinate system $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ could be simplified to $\|\mathbf{z}\|^2$.
Alas, this is a rather impractical suggestion.
Computing eigenvalues and eigenvectors is in general *much more* expensive than solving the actual problem.
-->

Nếu hệ số điều kiện $\kappa$ lớn, việc giải bài toán tối ưu một cách chính xác trở nên khá khó khăn.
Ta cần đảm bảo việc lựa chọn một khoảng động lớn các giá trị phù hợp.
Quá trình phân tích dẫn đến một câu hỏi hiển nhiên dù có phần ngây thơ rằng: chẳng phải ta có thể "sửa chữa" bài toán bằng cách biến đổi không gian sao cho tất cả các trị riêng đều có giá trị bằng $1$.
Điều này khá đơn giản trên lý thuyết: ta chỉ cần tính các trị riêng và các vector riêng của $\mathbf{Q}$ nhằm biến đổi bài toán 
từ $\mathbf{x}$ sang $\mathbf{z} := \boldsymbol{\Lambda}^{\frac{1}{2}} \mathbf{U} \mathbf{x}$.
Trong hệ toạ độ mới, $\mathbf{x}^\top \mathbf{Q} \mathbf{x}$ có thể được đơn giản hóa thành $\|\mathbf{z}\|^2$.
Nhưng có vẻ hướng giải quyết này không thực tế.
Việc tính toán các trị riêng và các vector riêng thường tốn kém hơn *rất nhiều* so với việc tìm lời giải cho bài toán thực tế.

<!--
While computing eigenvalues exactly might be expensive, guessing them and computing them even somewhat approximately may already be a lot better than not doing anything at all.
In particular, we could use the diagonal entries of $\mathbf{Q}$ and rescale it accordingly.
This is *much* cheaper than computing eigenvalues.
-->

Trong khi việc tính toán chính xác các trị riêng có thể có chi phí cao, việc ước chừng và tính toán xấp xỉ chúng đã là tốt hơn nhiều so với không làm gì cả.
Trong thực tế, ta có thể sử dụng các phần tử trên đường chéo của $\mathbf{Q}$ và tái tỉ lệ chúng một cách tương ứng.
Việc này có chi phí tính toán thấp hơn *nhiều* so với tính các trị riêng.


$$\tilde{\mathbf{Q}} = \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}) \mathbf{Q} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{Q}).$$


<!--
In this case we have $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ and specifically $\tilde{\mathbf{Q}}_{ii} = 1$ for all $i$.
In most cases this simplifies the condition number considerably.
For instance, the cases we discussed previously, this would entirely eliminate the problem at hand since the problem is axis aligned.
-->

Trong trường hợp này ta có $\tilde{\mathbf{Q}}_{ij} = \mathbf{Q}_{ij} / \sqrt{\mathbf{Q}_{ii} \mathbf{Q}_{jj}}$ và cụ thể $\tilde{\mathbf{Q}}_{ii} = 1$ với mọi $i$.
Trong đa số các trường hợp, cách làm này sẽ đơn giản hóa đáng kể hệ số điều kiện. 
Ví dụ đối với các trường hợp ta đã thảo luận ở phần trước, việc này sẽ triệt tiêu hoàn toàn vấn đề đang có do các bài toán đều có cấu trúc hình học với các cạnh song song trục toạ độ (*axis aligned*).

<!--
Unfortunately we face yet another problem: in deep learning we typically do not even have access to the second derivative of the objective function: 
for $\mathbf{x} \in \mathbb{R}^d$ the second derivative even on a minibatch may require $\mathcal{O}(d^2)$ space and work to compute, thus making it practically infeasible.
The ingenious idea of Adagrad is to use a proxy for that elusive diagonal of the Hessian that is both relatively cheap to compute and effective---the magnitude of the gradient itself.
-->

Đáng tiếc rằng ta phải tiếp tục đối mặt với một vấn đề khác: trong học sâu, ta thường không tính được ngay cả đạo hàm bậc hai của hàm mục tiêu.
Đối với $\mathbf{x} \in \mathbb{R}^d$, đạo hàm bậc hai thậm chí với một minibatch có thể yêu cầu không gian và độ phức tạp lên đến $\mathcal{O}(d^2)$ để tính toán, do đó khiến cho vấn đề không thể thực hiện được trong thực tế.
Sự khéo léo của Adagrad nằm ở việc sử dụng một biến đại diện (*proxy*) để tính toán đường chéo của ma trận Hessian một cách hiệu quả và đơn giản---đó là độ lớn của chính gradient.

<!--
In order to see why this works, let us look at $\bar{f}(\bar{\mathbf{x}})$. We have that
-->

Để tìm hiểu tại sao cách này lại có hiệu quả, hãy cùng xét $\bar{f}(\bar{\mathbf{x}})$. Ta có:


$$\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}}) = \boldsymbol{\Lambda} \bar{\mathbf{x}} + \bar{\mathbf{c}} = \boldsymbol{\Lambda} \left(\bar{\mathbf{x}} - \bar{\mathbf{x}}_0\right),$$


<!--
where $\bar{\mathbf{x}}_0$ is the minimizer of $\bar{f}$.
Hence the magnitude of the gradient depends both on $\boldsymbol{\Lambda}$ and the distance from optimality.
If $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ didn't change, this would be all that's needed.
After all, in this case the magnitude of the gradient $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ suffices.
Since AdaGrad is a stochastic gradient descent algorithm, we will see gradients with nonzero variance even at optimality.
As a result we can safely use the variance of the gradients as a cheap proxy for the scale of the Hessian.
A thorough analysis is beyond the scope of this section (it would be several pages).
We refer the reader to :cite:`Duchi.Hazan.Singer.2011` for details.
-->

trong đó $\bar{\mathbf{x}}_0$ là nghiệm cực tiểu của $\bar{f}$.
Do đó độ lớn của gradient phụ thuộc vào cả $\boldsymbol{\Lambda}$ và khoảng cách đến điểm tối ưu.
Nếu $\bar{\mathbf{x}} - \bar{\mathbf{x}}_0$ không đổi thì đây chính là tất cả các giá trị ta cần tính.
Suy cho cùng, trong trường hợp này độ lớn của gradient $\partial_{\bar{\mathbf{x}}} \bar{f}(\bar{\mathbf{x}})$ là đủ.
Do AdaGrad là một thuật toán hạ gradient ngẫu nhiên, ta sẽ thấy các gradient có phương sai khác không ngay cả tại điểm tối ưu. 
Chính vì thế ta có thể yên tâm sử dụng phương sai của các gradient như một biến đại diện dễ tính cho độ lớn của ma trận Hessian.
Việc phân tích chi tiết nằm ngoài phạm vi của phần này (có thể lên đến nhiều trang).
Độc giả có thể tham khảo :cite:`Duchi.Hazan.Singer.2011` để biết thêm chi tiết.


<!--
## The Algorithm
-->

## Thuật toán

<!--
Let us formalize the discussion from above.
We use the variable $\mathbf{s}_t$ to accumulate past gradient variance as follows.
-->

Hãy cùng công thức hóa phần thảo luận ở trên.
Ta sử dụng biến $\mathbf{s}_t$ để tích luỹ phương sai của các gradient trong quá khứ như sau:


$$\begin{aligned}
    \mathbf{g}_t & = \partial_{\mathbf{w}} l(y_t, f(\mathbf{x}_t, \mathbf{w})), \\
    \mathbf{s}_t & = \mathbf{s}_{t-1} + \mathbf{g}_t^2, \\
    \mathbf{w}_t & = \mathbf{w}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t + \epsilon}} \cdot \mathbf{g}_t.
\end{aligned}$$



<!--
Here the operation are applied coordinate wise.
That is, $\mathbf{v}^2$ has entries $v_i^2$.
Likewise $\frac{1}{\sqrt{v}}$ has entries $\frac{1}{\sqrt{v_i}}$ and $\mathbf{u} \cdot \mathbf{v}$ has entries $u_i v_i$.
As before $\eta$ is the learning rate and $\epsilon$ is an additive constant that ensures that we do not divide by $0$.
Last, we initialize $\mathbf{s}_0 = \mathbf{0}$.
-->

Ở đây các phép toán được thực hiện theo từng tọa độ.
Nghĩa là, $\mathbf{v}^2$ có các phần tử $v_i^2$.
Tương tự, $\frac{1}{\sqrt{v}}$ cũng có các phần tử $\frac{1}{\sqrt{v_i}}$ và $\mathbf{u} \cdot \mathbf{v}$ có các phần tử $u_i v_i$. 
Như phần trước $\eta$ là tốc độ học và $\epsilon$ là hằng số cộng thêm đảm bảo rằng ta không bị lỗi chia cho $0$.
Cuối cùng, ta khởi tạo $\mathbf{s}_0 = \mathbf{0}$.

<!--
Just like in the case of momentum we need to keep track of an auxiliary variable, in this case to allow for an individual learning rate per coordinate.
This does not increase the cost of Adagrad significantly relative to SGD, simply since the main cost is typically to compute $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ and its derivative.
-->

Tương tự như trường hợp sử dụng động lượng, ta cần phải theo dõi các biến bổ trợ để mỗi toạ độ có một tốc độ học độc lập.
Cách này không làm tăng chi phí của Adagrad so với SGD, lý do đơn giản là bởi chi phí chính yếu thường nằm ở bước tính $l(y_t, f(\mathbf{x}_t, \mathbf{w}))$ và đạo hàm của nó. 

<!--
Note that accumulating squared gradients in $\mathbf{s}_t$ means that $\mathbf{s}_t$ grows essentially at linear rate (somewhat slower than linearly in practice, since the gradients initially diminish).
This leads to an $\mathcal{O}(t^{-\frac{1}{2}})$ learning rate, albeit adjusted on a per coordinate basis.
For convex problems this is perfectly adequate.
In deep learning, though, we might want to decrease the learning rate rather more slowly.
This led to a number of Adagrad variants that we will discuss in the subsequent chapters.
For now let us see how it behaves in a quadratic convex problem.
We use the same problem as before:
-->

Cần lưu ý, tổng bình phương các gradient trong $\mathbf{s}_t$ có thể hiểu là về cơ bản $\mathbf{s}_t$ tăng một cách tuyến tính (có phần chậm hơn so với tuyến tính trong thực tế, do gradient lúc ban đầu bị co lại).
Điều này dẫn đến tốc độ học là $\mathcal{O}(t^{-\frac{1}{2}})$, mặc dù được điều chỉnh theo từng toạ độ một.
Đối với các bài toán lồi, như vậy là hoàn toàn đủ.
Tuy nhiên trong học sâu, có lẽ ta sẽ muốn giảm tốc độ học chậm hơn một chút.
Việc này dẫn đến một số biến thể của Adagrad mà ta sẽ thảo luận trong các phần tới.
Còn bây giờ hãy cùng xét cách thức hoạt động của Adagrad trong một bài toán lồi bậc hai.
Ta vẫn giữ nguyên bài toán như cũ:


$$f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2.$$


<!--
We are going to implement Adagrad using the same learning rate previously, i.e., $\eta = 0.4$.
As we can see, the iterative trajectory of the independent variable is smoother.
However, due to the cumulative effect of $\boldsymbol{s}_t$, the learning rate continuously decays, so the independent variable does not move as much during later stages of iteration.
-->

Ta sẽ lập trình Adagrad với tốc độ học giữ nguyên như phần trước, tức $\eta = 0.4$.
Có thể thấy quỹ đạo của biến độc lập mượt hơn nhiều.
Tuy nhiên, do ta tính tổng $\boldsymbol{s}_t$, tốc độ học liên tục suy giảm khiến cho các biến độc lập không thay đổi nhiều ở các giai đoạn về sau của vòng lặp.


```{.python .input  n=6}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np, npx
npx.set_np()

def adagrad_2d(x1, x2, s1, s2):
    eps = 1e-6
    g1, g2 = 0.2 * x1, 4 * x2
    s1 += g1 ** 2
    s2 += g2 ** 2
    x1 -= eta / math.sqrt(s1 + eps) * g1
    x2 -= eta / math.sqrt(s2 + eps) * g2
    return x1, x2, s1, s2

def f_2d(x1, x2):
    return 0.1 * x1 ** 2 + 2 * x2 ** 2

eta = 0.4
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```


<!--
As we increase the learning rate to $2$ we see much better behavior.
This already indicates that the decrease in learning rate might be rather aggressive, even in the noise-free case and we need to ensure that parameters converge appropriately.
-->

Nếu tăng tốc độ học lên $2$, ta có thể thấy quá trình học tốt hơn đáng kể.
Điều này chứng tỏ rằng tốc độ học giảm khá mạnh, ngay cả trong trường hợp không có nhiễu và ta cần phải đảm bảo rằng các tham số hội tụ một cách thích hợp.



```{.python .input  n=10}
eta = 2
d2l.show_trace_2d(f_2d, d2l.train_2d(adagrad_2d))
```


<!--
## Implementation from Scratch
-->

## Lập trình từ đầu

<!--
Just like the momentum method, Adagrad needs to maintain a state variable of the same shape as the parameters.
-->

Giống như phương pháp động lượng, Adagrad cần duy trì một biến trạng thái có cùng kích thước với các tham số.


```{.python .input  n=8}
def init_adagrad_states(feature_dim):
    s_w = np.zeros((feature_dim, 1))
    s_b = np.zeros(1)
    return (s_w, s_b)

def adagrad(params, states, hyperparams):
    eps = 1e-6
    for p, s in zip(params, states):
        s[:] += np.square(p.grad)
        p[:] -= hyperparams['lr'] * p.grad / np.sqrt(s + eps)
```


<!--
Compared to the experiment in :numref:`sec_minibatch_sgd` we use a
larger learning rate to train the model.
-->

Ta sử dụng tốc độ học lớn hơn so với thí nghiệm ở :numref:`sec_minibatch_sgd` để huấn luyện mô hình.


```{.python .input  n=9}
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adagrad, init_adagrad_states(feature_dim),
               {'lr': 0.1}, data_iter, feature_dim);
```


<!--
## Concise Implementation
-->

## Lập trình Súc tích

<!--
Using the `Trainer` instance of the algorithm `adagrad`, we can invoke the Adagrad algorithm in Gluon.
-->

Sử dụng đối tượng `Trainer` trong thuật toán `adagrad`, ta có thể gọi thuật toán Adagrad trong Gluon.


```{.python .input  n=5}
d2l.train_concise_ch11('adagrad', {'learning_rate': 0.1}, data_iter)
```


<!--
## Summary
-->

## Tóm tắt

<!--
* Adagrad decreases the learning rate dynamically on a per-coordinate basis.
* It uses the magnitude of the gradient as a means of adjusting how quickly progress is achieved - coordinates with large gradients are compensated with a smaller learning rate.
* Computing the exact second derivative is typically infeasible in deep learning problems due to memory and computational constraints. The gradient can be a useful proxy.
* If the optimization problem has a rather uneven uneven structure Adagrad can help mitigate the distortion.
* Adagrad is particularly effective for sparse features where the learning rate needs to decrease more slowly for infrequently occurring terms.
* On deep learning problems Adagrad can sometimes be too aggressive in reducing learning rates. We will discuss strategies for mitigating this in the context of :numref:`sec_adam`.
-->

* Adagrad liên tục giảm giá trị của tốc độ học theo từng toạ độ.
* Thuật toán sử dụng độ lớn của gradient như một phương thức để điều chỉnh tiến độ học - các tọa độ với gradient lớn được cân bằng bởi tốc độ học nhỏ.
* Tính đạo hàm bậc hai một cách chính xác thường không khả thi trong các bài toán học sâu do hạn chế về bộ nhớ và khả năng tính toán. Do đó, gradient có thể trở thành một biến đại diện hữu ích.
* Nếu bài toán tối ưu có cấu trúc không được đồng đều, Adagrad có thể làm giảm bớt sự biến dạng đó.
* Adagrad thường khá hiệu quả đối với các đặc trưng thưa, trong đó tốc độ học cần giảm chậm hơn cho các tham số hiếm khi xảy ra.
* Trong các bài toán học sâu, Adagrad đôi khi làm giảm tốc độ học quá mạnh. Ta sẽ thảo luận các chiến lược nhằm giảm bớt vấn đề này trong ngữ cảnh của :numref:`sec_adam`.

<!--
## Exercises
-->

## Bài tập

<!--
1. Prove that for an orthogonal matrix $\mathbf{U}$ and a vector $\mathbf{c}$ the following holds: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$.
Why does this mean that the magnitude of perturbations does not change after an orthogonal change of variables?
2. Try out Adagrad for $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ and also for the objective function was rotated by 45 degrees, 
i.e., $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Does it behave differently?
3. Prove [Gerschgorin's circle theorem](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem) which states that eigenvalues $\lambda_i$ of 
a matrix $\mathbf{M}$ satisfy $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$ for at least one choice of $j$.
4. What does Gerschgorin's theorem tell us about the eigenvalues of the diagonally preconditioned matrix $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
5. Try out Adagrad for a proper deep network, such as :numref:`sec_lenet` when applied to Fashion MNIST.
6. How would you need to modify Adagrad to achieve a less aggressive decay in learning rate?
-->

1. Chứng minh rằng một ma trận trực giao $\mathbf{U}$ và một vector $\mathbf{c}$ thoả mãn điều kiện: $\|\mathbf{c} - \mathbf{\delta}\|_2 = \|\mathbf{U} \mathbf{c} - \mathbf{U} \mathbf{\delta}\|_2$.
Tại sao biểu thức trên lại biểu thị rằng độ nhiễu loạn không thay đổi khi biến đổi trực giao các biến?
2. Thử áp dụng Adagrad đối với $f(\mathbf{x}) = 0.1 x_1^2 + 2 x_2^2$ và đối với hàm mục tiêu được quay 45 độ,
tức là $f(\mathbf{x}) = 0.1 (x_1 + x_2)^2 + 2 (x_1 - x_2)^2$. Adagrad có hoạt động khác đi hay không?
3. Chứng minh [Định lý Gerschgorin](https://en.wikipedia.org/wiki/Gershgorin_circle_theorem), định lý phát biểu rằng với các trị riêng $\lambda_i$ của
ma trận $\mathbf{M}$, tồn tại $j$ thoả mãn $|\lambda_i - \mathbf{M}_{jj}| \leq \sum_{k \neq j} |\mathbf{M}_{jk}|$. 
4. Từ định lý Gerschgorin, ta có thể chỉ ra điều gì về các trị riêng của ma trận đường chéo tiền điều kiện (*diagonally preconditioned matrix*) $\mathrm{diag}^{-\frac{1}{2}}(\mathbf{M}) \mathbf{M} \mathrm{diag}^{-\frac{1}{2}}(\mathbf{M})$?
5. Hãy thử áp dụng Adagrad cho một mạng thực sự sâu như :numref:`sec_lenet` khi sử dụng Fashion MNIST.
6. Bạn sẽ thay đổi Adagrad như thế nào để tốc độ học không suy giảm quá mạnh?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/355)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Đỗ Trường Giang
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc
* Nguyễn Văn Quang
* Phạm Hồng Vinh

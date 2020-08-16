<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Approximate Training
-->

# Huấn luyện gần đúng
:label:`sec_approx_train`

<!--
Recall content of the last section.  The core feature of the skip-gram model is the use of softmax operations to compute the conditional probability of generating context word $w_o$ based on the given central target word $w_c$.
-->

Nhắc lại nội dung của phần trước. Đặc điểm cốt lõi của mô hình skip-gram là sử dụng các toán tử softmax để tính xác suất có điều kiện sinh ra từ ngữ cảnh $w_o$ dựa trên từ đích trung tâm cho trước $w_c$.


$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}.$$


<!--
The logarithmic loss corresponding to the conditional probability is given as
-->

Mất mát log tương ứng với xác suất có điều kiện trên được tính như sau


$$-\log P(w_o \mid w_c) =
-\mathbf{u}_o^\top \mathbf{v}_c + \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$


<!--
Because the softmax operation has considered that the context word could be any word in the dictionary $\mathcal{V}$, 
the loss mentioned above actually includes the sum of the number of items in the dictionary size.
From the last section, we know that for both the skip-gram model and CBOW model, 
because they both get the conditional probability using a softmax operation, 
the gradient computation for each step contains the sum of the number of items in the dictionary size.
For larger dictionaries with hundreds of thousands or even millions of words, the overhead for computing each gradient may be too high.
In order to reduce such computational complexity, we will introduce two approximate training methods in this section: negative sampling and hierarchical softmax.
Since there is no major difference between the skip-gram model and the CBOW model, 
we will only use the skip-gram model as an example to introduce these two training methods in this section.
-->

Do toán tử softmax xem xét từ ngữ cảnh có thể bất kỳ từ nào trong từ điển $\mathcal{V}$, 
nên mất mát được đề cập ở trên, thực tế, bao gồm tổng số lượng các phần tử trong từ điển.
Ở phần trước, ta đã biết rằng đối với mô hình skip-gram và mô hình CBOW,
vì cả hai đều tính được xác suất có điều kiện bằng cách sử dụng toán tử softmax,
nên tính toán gradient cho mỗi bước bao gồm tổng số lượng các phần tử trong từ điển.
Đối với các từ điển lớn hơn với hàng trăm nghìn hoặc thậm chí hàng triệu từ, chi phí tính toán cho mỗi gradient có thể rất cao.
Để giảm độ phức tạp tính toán này, chúng tôi sẽ giới thiệu hai phương pháp huấn luyện gần đúng trong phần này, đó là lấy mẫu âm tính và toán tử softmax phân cấp.
Do không có sự khác biệt lớn giữa mô hình skip-gram và mô hình CBOW,
nên ta chỉ sử dụng mô hình skip-gram làm ví dụ để giới thiệu hai phương pháp huấn luyện trên trong phần này.


<!--
## Negative Sampling
-->

## Lấy Mẫu Âm tính
:label:`subsec_negative-sampling`


<!--
Negative sampling modifies the original objective function.
Given a context window for the central target word $w_c$, we will treat it as an event for context word $w_o$ to appear in the context window and compute the probability of this event from
-->


Phương pháp lấy mẫu tính sửa đổi hàm mục tiêu ban đầu.
Cho một cửa sổ ngữ cảnh cho từ đích trung tâm $w_c$, ta coi nó như một sự kiện cho từ ngữ cảnh $w_o$ xuất hiện trong cửa sổ ngữ cảnh và tính xác suất của sự kiện này theo


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$


<!--
Here, the $\sigma$ function has the same definition as the sigmoid activation function:
-->

Ở đây, hàm $\sigma$ có cùng định nghĩa với hàm kích hoạt sigmoid:


$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$


<!--
We will first consider training the word vector by maximizing the joint probability of all events in the text sequence.
Given a text sequence of length $T$, we assume that the word at timestep $t$ is $w^{(t)}$ and the context window size is $m$.
Now we consider maximizing the joint probability
-->

Đầu tiên, ta sẽ xem xét việc huấn luyện vector từ bằng cách cực đại hóa xác suất kết hợp của tất cả các sự kiện trong chuỗi văn bản.
Cho một chuỗi văn bản có độ dài $T$, ta giả sử rằng từ tại bước thời gian $t$ là $w^{(t)}$ và kích thước cửa sổ ngữ cảnh là $m$.
Bây giờ, ta sẽ xem xét việc cực đại hóa xác suất kết hợp


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->


<!--
However, the events included in the model only consider positive examples.
In this case, only when all the word vectors are equal and their values approach infinity can the joint probability above be maximized to 1.
Obviously, such word vectors are meaningless.
Negative sampling makes the objective function more meaningful by sampling with an addition of negative examples.
Assume that event $P$ occurs when context word $w_o$ appears in the context window of central target word $w_c$, 
and we sample $K$ words that do not appear in the context window according to the distribution $P(w)$ to act as noise words.
We assume the event for noise word $w_k$($k=1, \ldots, K$) to not appear in the context window of central target word $w_c$ is $N_k$.
Suppose that events $P$ and $N_1, \ldots, N_K$ for both positive and negative examples are independent of each other.
By considering negative sampling, we can rewrite the joint probability above, which only considers the positive examples, as
-->

*dịch đoạn phía trên*


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$


<!--
Here, the conditional probability is approximated to be
-->

*dịch đoạn phía trên*


$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$


<!--
Let the text sequence index of word $w^{(t)}$ at timestep $t$ be $i_t$ and $h_k$ for noise word $w_k$ in the dictionary.
The logarithmic loss for the conditional probability above is
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$


<!--
Here, the gradient computation in each step of the training is no longer related to the dictionary size, but linearly related to $K$. When $K$ takes a smaller constant, the negative sampling has a lower computational overhead for each step.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Hierarchical Softmax
-->

## *dịch tiêu đề phía trên*


<!--
Hierarchical softmax is another type of approximate training method.
It uses a binary tree for data structure as illustrated in :numref:`fig_hi_softmax`, 
with the leaf nodes of the tree representing every word in the dictionary $\mathcal{V}$.
-->

*dịch đoạn phía trên*


<!--
![Hierarchical Softmax. Each leaf node of the tree represents a word in the dictionary.](../img/hi-softmax.svg)
-->

![*dịch mô tả phía trên*](../img/hi-softmax.svg)
:label:`fig_hi_softmax`


<!--
We assume that $L(w)$ is the number of nodes on the path (including the root and leaf nodes) from the root node of the binary tree to the leaf node of word $w$.
Let $n(w, j)$ be the $j^\mathrm{th}$ node on this path, with the context word vector $\mathbf{u}_{n(w, j)}$.
We use Figure 10.3 as an example, so $L(w_3) = 4$.
Hierarchical softmax will approximate the conditional probability in the skip-gram model as
-->

*dịch đoạn phía trên*


$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$


<!--
Here the $\sigma$ function has the same definition as the sigmoid activation function, and $\text{leftChild}(n)$ is the left child node of node $n$.
If $x$ is true, $[\![x]\!] = 1$; otherwise $[\![x]\!] = -1$.
Now, we will compute the conditional probability of generating word $w_3$ based on the given word $w_c$ in Figure 10.3.
We need to find the inner product of word vector $\mathbf{v}_c$ (for word $w_c$) and each non-leaf node vector on the path from the root node to $w_3$.
Because, in the binary tree, the path from the root node to leaf node $w_3$ needs to be traversed left, right, and left again (the path with the bold line in Figure 10.3), we get
-->

*dịch đoạn phía trên*


$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
Because $\sigma(x)+\sigma(-x) = 1$, the condition that the sum of the conditional probability of any word generated 
based on the given central target word $w_c$ in dictionary $\mathcal{V}$ be 1 will also suffice:
-->

*dịch đoạn phía trên*


$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$


<!--
In addition, because the order of magnitude for $L(w_o)-1$ is $\mathcal{O}(\text{log}_2|\mathcal{V}|)$, when the size of dictionary $\mathcal{V}$ is large, the computational overhead for each step in the hierarchical softmax training is greatly reduced compared to situations where we do not use approximate training.
-->

*dịch đoạn phía trên*


## Tóm tắt

<!--
* Negative sampling constructs the loss function by considering independent events that contain both positive and negative examples.
The gradient computational overhead for each step in the training process is linearly related to the number of noise words we sample.
* Hierarchical softmax uses a binary tree and constructs the loss function based on the path from the root node to the leaf node.
The gradient computational overhead for each step in the training process is related to the logarithm of the dictionary size.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. Before reading the next section, think about how we should sample noise words in negative sampling.
2. What makes the last formula in this section hold?
3. How can we apply negative sampling and hierarchical softmax in the skip-gram model?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh](https://discuss.d2l.ai/t/382)
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
* Nguyễn Văn Quang

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 


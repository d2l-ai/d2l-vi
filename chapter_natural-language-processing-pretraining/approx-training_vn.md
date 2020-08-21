<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Approximate Training
-->

# *dịch tiêu đề phía trên*
:label:`sec_approx_train`

<!--
Recall content of the last section.  The core feature of the skip-gram model is the use of softmax operations to compute the conditional probability of generating context word $w_o$ based on the given central target word $w_c$.
-->

*dịch đoạn phía trên*


$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}.$$


<!--
The logarithmic loss corresponding to the conditional probability is given as
-->

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


<!--
## Negative Sampling
-->

## *dịch tiêu đề phía trên*
:label:`subsec_negative-sampling`


<!--
Negative sampling modifies the original objective function.
Given a context window for the central target word $w_c$, we will treat it as an event for context word $w_o$ to appear in the context window and compute the probability of this event from
-->

*dịch đoạn phía trên*


$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$


<!--
Here, the $\sigma$ function has the same definition as the sigmoid activation function:
-->

*dịch đoạn phía trên*


$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$


<!--
We will first consider training the word vector by maximizing the joint probability of all events in the text sequence.
Given a text sequence of length $T$, we assume that the word at timestep $t$ is $w^{(t)}$ and the context window size is $m$.
Now we consider maximizing the joint probability
-->

*dịch đoạn phía trên*


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

Tuy nhiên, các sự kiện trong mô hình chỉ xem xét các mẫu dương.
Trong trường hợp này, chỉ khi tất cả các vector từ bằng nhau và giá trị của chúng tiến tới vô cùng, thì xác suất kết hợp trên mới có thể đạt giá trị cực đại bằng 1.
Rõ ràng, các vector từ như vậy là vô nghĩa.
Phương pháp lấy mẫu âm khiến hàm mục tiêu có ý nghĩa hơn bằng cách lấy mẫu có bổ sung các mẫu âm.
Giả sử sự kiện $P$ xảy ra khi từ ngữ cảnh $w_o$ xuất hiện trong cửa sổ ngữ cảnh của từ đích trung tâm $w_c$,
và ta lấy mẫu $K$ từ không xuất hiện trong cửa sổ ngữ cảnh, đóng vai trò là các từ nhiễu, theo phân phối $P(w)$.
Ta giả sử sự kiện cho từ nhiễu $w_k$($k=1, \ldots, K$) không xuất hiện trong cửa sổ ngữ cảnh của từ đích trung tâm $w_c$ là $N_k$.
Giả sử các sự kiện $P$ và $N_1, \ldots, N_K$ cho cả mẫu dương lẫn và mẫu âm là độc lập với nhau.
Bằng cách xem xét phương pháp lấy mẫu âm, ta có thể viết lại xác suất kết hợp chỉ xem xét các mẫu dương ở trên như sau


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$


<!--
Here, the conditional probability is approximated to be
-->

Ở đây, xác suất có điều kiện được tính gần đúng bằng


$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$


<!--
Let the text sequence index of word $w^{(t)}$ at timestep $t$ be $i_t$ and $h_k$ for noise word $w_k$ in the dictionary.
The logarithmic loss for the conditional probability above is
-->

Đặt chỉ số của từ $w^{(t)}$ trong chuỗi văn bản tại bước thời gian $t$ là $i_t$ và chỉ số của từ nhiễu $w_k$ trong từ điển là $h_k$.
Mất mát log cho xác suất có điều kiện ở trên là


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

Ở đây, tính toán gradient trong mỗi bước huấn luyện không còn liên quan đến kích thước từ điển, mà liên quan tuyến tính với $K$. Khi $K$ có giá trị nhỏ hơn, thì phương pháp lấy mẫu âm có chi phí tính toán cho mỗi bước thấp hơn.


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
* 

<!-- Phần 2 -->
* Nguyễn Văn Quang

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

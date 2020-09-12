<!--
# Word Embedding with Global Vectors (GloVe)
-->

# Embedding từ với Vector Toàn cục (GloVe)
:label:`sec_glove`


<!--
First, we should review the skip-gram model in word2vec.
The conditional probability $P(w_j\mid w_i)$ expressed in the skip-gram model using the softmax operation will be recorded as $q_{ij}$, that is:
-->

Trước tiên, ta sẽ xem lại mô hình skip-gram trong word2vec. 
Xác suất có điều kiện $P(w_j\mid w_i)$ được biểu diễn trong mô hình skip-gram bằng hàm kích hoạt softmax sẽ được gọi là $q_{ij}$ như sau: 


$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$


<!--
where $\mathbf{v}_i$ and $\mathbf{u}_i$ are the vector representations of word $w_i$ of index $i$ as the center word and context word respectively, 
and $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ is the vocabulary index set.
-->

Ở đây $\mathbf{v}_i$ và $\mathbf{u}_i$ là các biểu diễn vector từ $w_i$ với chỉ số $i$, lần lượt khi nó là từ trung tâm và từ ngữ cảnh, 
và $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ là tập chứa các chỉ số của bộ từ vựng. 


<!--
For word $w_i$, it may appear in the dataset for multiple times.
We collect all the context words every time when $w_i$ is a center word and keep duplicates, denoted as multiset $\mathcal{C}_i$.
The number of an element in a multiset is called the multiplicity of the element.
For instance, suppose that word $w_i$ appears twice in the dataset: 
the context windows when these two $w_i$ become center words in the text sequence contain context word indices $2, 1, 5, 2$ and $2, 3, 2, 1$.
Then, multiset $\mathcal{C}_i = \{1, 1, 2, 2, 2, 2, 3, 5\}$, where multiplicity of element 1 is 2, multiplicity of element 2 is 4, and multiplicities of elements 3 and 5 are both 1.
Denote multiplicity of element $j$ in multiset $\mathcal{C}_i$ as $x_{ij}$: it is the number of word $w_j$ in all the context windows for center word $w_i$ in the entire dataset.
As a result, the loss function of the skip-gram model can be expressed in a different way:
-->

Từ $w_i$ có thể xuất hiện trong tập dữ liệu nhiều lần. 
Ta gom tất cả các từ ngữ cảnh mỗi khi $w_i$ là từ trung tâm và giữ các lần trùng lặp, rồi ký hiệu đó là tập bội $\mathcal{C}_i$. 
Số lượng của một phần tử trong tập bội được gọi là bội số của phần tử đó.
Chẳng hạn, giả sử rằng từ $w_i$ xuất hiện hai lần trong tập dữ liệu: 
khi hai từ $w_i$ đó là từ trung tâm trong chuỗi văn bản, hai cửa sổ ngữ cảnh tương ứng chứa các chỉ số từ ngữ cảnh $2, 1, 5, 2$ và $2, 3, 2, 1$. 
Khi đó, ta sẽ có tập bội $\mathcal{C}_i = \{1, 1, 2, 2, 2, 2, 3, 5\}$, trong đó bội số của phần tử 1 là 2, bội số của phần tử 2 là 4, và bội số của phần tử 3 và 5 đều là 1. 
Ta ký hiệu bội số của phần tử $j$ trong tập bội $\mathcal{C}_i$ là $x{ij}$: nó là số lần từ $w_j$ xuất hiện trong cửa sổ ngữ cảnh khi từ trung tâm là $w_i$ trong toàn bộ tập dữ liệu. 
Kết quả là hàm mất mát của mô hình skip-gram có thể được biểu diễn theo một cách khác: 

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$


<!--
We add up the number of all the context words for the central target word $w_i$ to get $x_i$, 
and record the conditional probability $x_{ij}/x_i$ for generating context word $w_j$ based on central target word $w_i$ as $p_{ij}$.
We can rewrite the loss function of the skip-gram model as
-->

Ta tính tổng số lượng tất cả các từ ngữ cảnh đối với từ trung tâm $w_i$ để có $x_i$, 
rồi thu được xác suất có điều kiện để sinh ra từ ngữ cảnh $w_j$ dựa trên từ trung tâm $w_i$ là $p_{ij}$ bằng $x_{ij}/x_i$. 
Ta có thể viết lại hàm mất mất của mô hình skip-gram như sau 


$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$


<!--
In the formula above, $\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ computes the conditional probability distribution $p_{ij}$ for context word 
generation based on the central target word $w_i$ and the cross-entropy of conditional probability distribution $q_{ij}$ predicted by the model.
The loss function is weighted using the sum of the number of context words with the central target word $w_i$.
If we minimize the loss function from the formula above, we will be able to allow the predicted conditional probability distribution 
to approach as close as possible to the true conditional probability distribution.
-->

Trong công thức trên, $\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ tính toán phân phối xác suất có điều kiện $p_{ij}$ của việc sinh từ ngữ cảnh 
dựa trên từ đích trung tâm $w_i$ và entropy chéo với phân phối xác suất có điều kiện $q_{ij}$ được dự đoán bởi mô hình. 
Hàm mất mát được đánh trọng số bằng cách sử dụng tổng số từ ngữ cảnh cho từ đích trung tâm $w_i$. 
Việc cực tiểu hóa hàm mất mát theo công thức trên cho phép phân phối xác suất có điều kiện được dự đoán một cách 
gần nhất có thể tới phân phối xác suất có điều kiện thật sự. 


<!--
However, although the most common type of loss function, the cross-entropy loss function is sometimes not a good choice.
On the one hand, as we mentioned in :numref:`sec_approx_train` the cost of letting the model prediction $q_{ij}$ become
the legal probability distribution has the sum of all items in the entire dictionary in its denominator.
This can easily lead to excessive computational overhead.
On the other hand, there are often a lot of uncommon words in the dictionary, and they appear rarely in the dataset.
In the cross-entropy loss function, the final prediction of the conditional probability distribution on a large number of uncommon words is likely to be inaccurate.
-->

Tuy nhiên, mặc dù là hàm mất mát phổ biến nhất, đôi khi hàm mất mát entropy chéo lại không phải là một lựa chọn phù hợp. 
Một mặt, như ta đã đề cập trong :numref:`sec_approx_train`, chi phí để mô hình đưa ra dự đoán $q_{ij}$ trở thành phân phối xác suất hợp lệ 
gồm phép lấy tổng qua toàn bộ các từ trong từ điển ở mẫu số của nó. 
Điều này có thể dễ dàng khiến tổng chi phí tính toán trở nên quá lớn. 
Mặt khác, thường sẽ có rất nhiều từ hiếm gặp trong từ điển, và chúng ít khi xuất hiện trong tập dữ liệu. 
Trong hàm mất mát entropy chéo, dự đoán cuối cùng cho phân phối xác suất có điều kiện trên một lượng lớn các từ hiếm gặp rất có thể sẽ không được chính xác. 


<!--
## The GloVe Model
-->

## Mô hình GloVe


<!--
To address this, GloVe :cite:`Pennington.Socher.Manning.2014`, a word embedding model that came after word2vec, adopts
square loss and makes three changes to the skip-gram model based on this loss.
-->

Để giải quyết vấn đề trên, GloVe :cite:`Pennington.Socher.Manning.2014`, một mô hình embedding từ xuất hiện sau word2vec
đã áp dụng mất mát bình phương và đề xuất ba thay đổi trong mô hình skip-gram dựa theo mất mát này.


<!--
1. Here, we use the non-probability distribution variables $p'_{ij}=x_{ij}$ and $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ and take their logs.
Therefore, we get the square loss $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. We add two scalar model parameters for each word $w_i$: the bias terms $b_i$ (for central target words) and $c_i$(for context words).
3. Replace the weight of each loss with the function $h(x_{ij})$. The weight function $h(x)$ is a monotone increasing function with the range $[0, 1]$.
-->

1. Ở đây, ta sử dụng các biến phân phối phi xác suất $p'_{ij}=x_{ij}$ và $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ rồi tính log của chúng.
Do đó, ta có mất mát bình phương $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$. 
2. Ta thêm hai tham số mô hình cho mỗi từ $w_i$: hệ số điều chỉnh $b_i$ (cho các từ trung tâm) và $c_i$ (cho các từ ngữ cảnh). 
3. Thay thế trọng số của mỗi giá trị mất mát bằng hàm $h(x_{ij})$. Hàm trọng số $h(x)$ là hàm đơn điệu tăng trong khoảng $[0, 1]$. 


<!--
Therefore, the goal of GloVe is to minimize the loss function.
-->

Do đó, mục tiêu của GloVe là cực tiểu hóa hàm mất mát.


$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$


<!--
Here, we have a suggestion for the choice of weight function $h(x)$: when $x < c$ (e.g $c = 100$), make $h(x) = (x/c) ^\alpha$ (e.g $\alpha = 0.75$), otherwise make $h(x) = 1$.
Because $h(0)=0$, the squared loss term for $x_{ij}=0$ can be simply ignored.
When we use minibatch SGD for training, we conduct random sampling to get a non-zero minibatch $x_{ij}$ from each timestep and compute the gradient to update the model parameters.
These non-zero $x_{ij}$ are computed in advance based on the entire dataset and they contain global statistics for the dataset.
Therefore, the name GloVe is taken from "Global Vectors".
-->


Ở đây, chúng tôi có một đề xuất đối với việc lựa chọn hàm trọng số $h(x)$: khi $x < c$ (ví dụ $c = 100$) thì $h(x) = (x/c) ^\alpha$ (ví dụ $\alpha = 0.75$), nếu không thì $h(x) = 1$. 
Do $h(0)=0$, ta có thể đơn thuần bỏ qua mất mát bình phương tại $x_{ij}=0$.
Khi sử dụng minibatch SGD trong quá trình huấn luyện, ta tiến hành lấy mẫu ngẫu nhiên để được một minibatch $x_{ij}$ khác không 
tại mỗi bước thời gian và tính toán gradient để cập nhật các tham số mô hình.
Các giá trị $x_{ij}$ khác không trên được tính trước trên toàn bộ tập dữ liệu và là thống kê toàn cục của tập dữ liệu. 
Do đó, tên gọi GloVe được lấy từ "Global Vectors (*Vector Toàn cục*)". 


<!--
Notice that if word $w_i$ appears in the context window of word $w_j$, then word $w_j$ will also appear in the context window of word $w_i$. Therefore, $x_{ij}=x_{ji}$.
Unlike word2vec, GloVe fits the symmetric $\log\, x_{ij}$ in lieu of the asymmetric conditional probability $p_{ij}$.
Therefore, the central target word vector and context word vector of any word are equivalent in GloVe.
However, the two sets of word vectors that are learned by the same word may be different in the end due to different initialization values.
After learning all the word vectors, GloVe will use the sum of the central target word vector and the context word vector as the final word vector for the word.
-->

Chú ý rằng nếu từ $w_i$ xuất hiện trong cửa sổ ngữ cảnh của từ $w_j$ thì từ $w_j$ cũng sẽ xuất hiện trong cửa sổ ngữ cảnh của từ $w_i$. Do đó, $x_{ij}=x_{ji}$. 
Không như word2vec, GloVe khớp $\log\, x_{ij}$ đối xứng thay vì xác suất có điều kiện $p_{ij}$ bất đối xứng. 
Do đó, vector từ đích trung tâm và vector từ ngữ cảnh của bất kì từ nào đều tương đương nhau trong GloVe. 
Tuy vậy, hai tập vector từ được học bởi cùng một mô hình về cuối có thể sẽ khác nhau do giá trị khởi tạo khác nhau. 
Sau khi học tất cả các vector từ, GloVe sẽ sử dụng tổng của vector từ đích trung tâm và vector từ ngữ cảnh để làm vector từ cuối cùng cho từ đó. 


<!--
## Understanding GloVe from Conditional Probability Ratios
-->

## Lý giải GloVe bằng Tỷ số Xác suất Có điều kiện


<!--
We can also try to understand GloVe word embedding from another perspective.
We will continue the use of symbols from earlier in this section, $P(w_j \mid w_i)$ represents 
the conditional probability of generating context word $w_j$ with central target word $w_i$ in the dataset, and it will be recorded as $p_{ij}$.
From a real example from a large corpus, here we have the following two sets of conditional probabilities with "ice" and "steam" as the central target words and the ratio between them:
-->

Ta cũng có thể cố gắng lý giải embedding từ GloVe theo một cách nhìn khác. 
Ta sẽ tiếp tục sử dụng các ký hiệu như ở trên, $P(w_j \mid w_i)$ biểu diễn
xác suất có điều kiện sinh từ ngữ cảnh $w_j$ với từ tâm đích $w_i$ trong tập dữ liệu, và xác suất này được ghi lại bằng $p_{ij}$. 
Xét ví dụ thực tế từ một kho ngữ liệu lớn, ở đây ta có hai tập các xác suất có điều kiện với "ice" và "steam" là các từ tâm đích và tỷ số giữa chúng: 


| $w_k$=                       |  solid   |  gas     |  water  |  fashion  |
|-----------------------------:|:--------:|:--------:|:-------:|:---------:|
| $p_1=P(w_k\mid \text{ice})$  | 0.00019  | 0.000066 | 0.003   | 0.000017  |
| $p_2=P(w_k\mid\text{steam})$ | 0.000022 | 0.00078  | 0.0022  | 0.000018  |
| $p_1/p_2$                    | 8.9      | 0.085    | 1.36    | 0.96      |


<!--
We will be able to observe phenomena such as:
-->

Ta có thể quan sát thấy các hiện tượng như sau: 


<!--
* For a word $w_k$ that is related to "ice" but not to "steam", such as $w_k=$"solid", 
we would expect a larger conditional probability ratio, like the value 8.9 in the last row of the table above.
* For a word $w_k$ that is related to "steam" but not to "ice", such as $w_k=$"gas", 
we would expect a smaller conditional probability ratio, like the value 0.085 in the last row of the table above.
* For a word $w_k$ that is related to both "ice" and "steam", such as $w_k=$"water", 
we would expect a conditional probability ratio close to 1, like the value 1.36 in the last row of the table above.
* For a word $w_k$ that is related to neither "ice" or "steam", such as $w_k=$"fashion", 
we would expect a conditional probability ratio close to 1, like the value 0.96 in the last row of the table above.
-->

Với từ $w_k$ liên quan tới từ "ice (đá)" nhưng không liên quan đến từ "steam (hơi nước)", như là $w_k=\text{solid (rắn)}$, 
ta kỳ vọng là tỷ số xác suất có điều kiện sẽ lớn hơn, như trường hợp này là 8.9 ở hàng cuối cùng của bảng trên.
Với từ $w_k$ liên quan tới từ "steam (hơi nước)" mà không có liên quan nào với từ "ice (đá)", như là $w_k=\text{gas (khí)}$, 
ta kỳ vọng là tỷ số xác suất có điều kiện sẽ nhỏ hơn, như trường hợp này là 0.085 ở hàng cuối cùng của bảng trên. 
Với từ $w_k$ liên quan tới cả hai từ "steam (hơi nước)" và từ "ice (đá)", như là $w_k=\text{water (nước)}$, 
ta kỳ vọng là tỷ số xác suất có điều kiện sẽ gần với 1, như trường hợp này là 1.36 ở hàng cuối cùng của bảng trên. 
Với từ $w_k$ không liên quan tới cả hai từ "steam (hơi)" và từ "ice (đá)", như là $w_k=\text{fashion (thời trang)}$, 
ta kỳ vọng là tỷ số xác suất có điều kiện sẽ gần với 1, như trường hợp này là 0.96 ở hàng cuối cùng của bảng trên. 


<!--
We can see that the conditional probability ratio can represent the relationship between different words more intuitively.
We can construct a word vector function to fit the conditional probability ratio more effectively.
As we know, to obtain any ratio of this type requires three words $w_i$, $w_j$, and $w_k$.
The conditional probability ratio with $w_i$ as the central target word is ${p_{ij}}/{p_{ik}}$.
We can find a function that uses word vectors to fit this conditional probability ratio.
-->

Có thể thấy rằng tỷ số xác suất có điều kiện thể hiện mối quan hệ giữa các từ khác nhau trực quan hơn. 
Ta có thể tạo một hàm vector của từ để khớp tỷ số xác suất có điều kiện một cách hiệu quả hơn. 
Như đã biết, để thu được bất cứ tỷ số nào loại này đòi hỏi phải có ba từ $w_i$, $w_j$, và $w_k$. 
tỷ số xác suất có điều kiện với $w_i$ làm từ trung tâm là ${p_{ij}}/{p_{ik}}$.
Ta có thể tìm một hàm dùng các vector từ để khớp với tỷ số xác suất có điều kiện này. 


$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$


<!--
The possible design of function $f$ here will not be unique.
We only need to consider a more reasonable possibility.
Notice that the conditional probability ratio is a scalar, we can limit $f$ to be a scalar function: 
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$.
After exchanging index $j$ with $k$, we will be able to see that function $f$ satisfies the condition $f(x)f(-x)=1$, so one possibility could be $f(x)=\exp(x)$. Thus:
-->

Thiết kế khả dĩ của hàm $f$ ở đây không phải duy nhất. 
Ta chỉ cần quan tâm một lựa chọn hợp lý hơn. 
Do tỷ số xác suất có điều kiện là một số vô hướng, ta có thể giới hạn $f$ vào một hàm vô hướng: 
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$. 
Sau khi hóan đổi chỉ số $j$ và $k$, ta có thể thấy rằng hàm $f$ thỏa mãn điều kiện $f(x)f(-x)=1$, do đó một lựa chọn có thể là $f(x)=\exp(x)$. Ta có: 


$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$


<!--
One possibility that satisfies the right side of the approximation sign is $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$, where $\alpha$ is a constant.
Considering that $p_{ij}=x_{ij}/x_i$, after taking the logarithm we get $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$.
We use additional bias terms to fit $- \log\, \alpha + \log\, x_i$, such as the central target word bias term $b_i$ and context word bias term $c_j$:
-->

Một xác suất thỏa mãn vế phải biểu thức xấp xỉ là $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$, ở đây $\alpha$ là một hằng số. 
Xét $p_{ij}=x_{ij}/x_i$, sau khi lấy logarit ta được $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$.
Ta sử dụng thêm hệ số điều chỉnh để khớp $- \log\, \alpha + \log\, x_i$, cụ thể là hệ số điều chỉnh từ trung tâm $b_i$ và hệ số điều chỉnh từ ngữ cảnh $c_j$: 


$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log(x_{ij}).$$


<!--
By taking the square error and weighting the left and right sides of the formula above, we can get the loss function of GloVe.
-->

Bằng cách lấy sai số bình phương và đặt trọng số vào vế trái và vế phải của biểu thức trên, ta tính được hàm mất mát của GloVe. 


## Tóm tắt

<!--
* In some cases, the cross-entropy loss function may have a disadvantage.
GloVe uses squared loss and the word vector to fit global statistics computed in advance based on the entire dataset.
* The central target word vector and context word vector of any word are equivalent in GloVe.
-->

* Trong một số trường hợp, hàm mất mát entropy chéo có sự hạn chế.
GloVe sử dụng mất mát bình phương và vector từ để khớp các thống kê toàn cục được tính trước dựa trên toàn bộ dữ liệu.
* Vector từ đích trung tâm và vector từ ngữ cảnh của bất kì từ nào là như nhau trong GloVe.

## Bài tập

<!--
1. If a word appears in the context window of another word, 
how can we use the distance between them in the text sequence to redesign the method for computing the conditional probability $p_{ij}$?
Hint: See section 4.2 from the paper GloVe :cite:`Pennington.Socher.Manning.2014`.
2. For any word, will its central target word bias term and context word bias term be equivalent to each other in GloVe? Why?
-->

1. Nếu một từ xuất hiện trong cửa sổ ngữ cảnh của từ khác, 
làm thế nào để sử dụng khoảng cách giữa hai từ này trong chuỗi văn bản để thiết kế lại phương pháp tính toán xác suất có điều kiện $p_{ij}$?
Gợi ý: Tham khảo phần 4.2 trong bài báo GloVe :cite:`Pennington.Socher.Manning.2014`. 
2. Với một từ bất kỳ, liệu hệ số điều chỉnh của từ đích trung tâm và từ ngữ cảnh là như nhau trong GloVe không? Tại sao? 


## Thảo luận
* Tiếng Anh: [Main Forum](https://discuss.d2l.ai/t/385)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Lê Khắc Hồng Phúc
* Đỗ Trường Giang
* Nguyễn Văn Cường
* Nguyễn Mai Hoàng Long
* Nguyễn Văn Quang
* Phạm Minh Đức
* Nguyễn Lê Quang Nhật
* Phạm Hồng Vinh

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 29/08/2020)*

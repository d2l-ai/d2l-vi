<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Word Embedding (word2vec)
-->

# *dịch đoạn phía trên*
:label:`sec_word2vec`


<!--
A natural language is a complex system that we use to express meanings.
In this system, words are the basic unit of linguistic meaning.
As its name implies, a word vector is a vector used to represent a word.
It can also be thought of as the feature vector of a word.
The technique of mapping words to vectors of real numbers is also known as word embedding.
Over the last few years, word embedding has gradually become basic knowledge in natural language processing.
-->

*dịch đoạn phía trên*


<!--
## Why Not Use One-hot Vectors?
-->

## *dịch đoạn phía trên*


<!--
We used one-hot vectors to represent words (characters are words) in :numref:`sec_rnn_scratch`.
Recall that when we assume the number of different words in a dictionary (the dictionary size) is $N$, each word can correspond one-to-one with consecutive integers from 0 to $N-1$.
These integers that correspond to words are called the indices of the words.
We assume that the index of a word is $i$.
In order to get the one-hot vector representation of the word, we create a vector of all 0s with a length of $N$ and set element $i$ to 1.
In this way, each word is represented as a vector of length $N$ that can be used directly by the neural network.
-->

*dịch đoạn phía trên*


<!--
Although one-hot word vectors are easy to construct, they are usually not a good choice.
One of the major reasons is that the one-hot word vectors cannot accurately express the similarity between different words, such as the cosine similarity that we commonly use.
For the vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, their cosine similarities are the cosines of the angles between them:
-->

*dịch đoạn phía trên*


$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$


<!--
Since the cosine similarity between the one-hot vectors of any two different words is 0, 
it is difficult to use the one-hot vector to accurately represent the similarity between multiple different words.
-->

*dịch đoạn phía trên*


<!--
[Word2vec](https://code.google.com/archive/p/word2vec/) is a tool that we came up with to solve the problem above.
It represents each word with a fixed-length vector and uses these vectors to better indicate the similarity and analogy relationships between different words.
The Word2vec tool contains two models: skip-gram :cite:`Mikolov.Sutskever.Chen.ea.2013` and continuous bag of words (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`.
Next, we will take a look at the two models and their training methods.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## The Skip-Gram Model
-->

## *dịch đoạn phía trên*


<!--
The skip-gram model assumes that a word can be used to generate the words that surround it in a text sequence.
For example, we assume that the text sequence is "the", "man", "loves", "his", and "son".
We use "loves" as the central target word and set the context window size to 2.
As shown in :numref:`fig_skip_gram`, given the central target word "loves", the skip-gram model is concerned with the conditional probability 
for generating the context words, "the", "man", "his" and "son", that are within a distance of no more than 2 words, which is
-->

*dịch đoạn phía trên*



$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$


<!--
We assume that, given the central target word, the context words are generated independently of each other.
In this case, the formula above can be rewritten as
-->

*dịch đoạn phía trên*


$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$


<!--
![The skip-gram model cares about the conditional probability of generating context words for a given central target word.](../img/skip-gram.svg)
-->


![*dịch mô tả phía trên*](../img/skip-gram.svg)
:label:`fig_skip_gram`


<!--
In the skip-gram model, each word is represented as two $d$-dimension vectors, which are used to compute the conditional probability.
We assume that the word is indexed as $i$ in the dictionary, its vector is represented as $\mathbf{v}_i\in\mathbb{R}^d$ 
when it is the central target word, and $\mathbf{u}_i\in\mathbb{R}^d$ when it is a context word.
Let the central target word $w_c$ and context word $w_o$ be indexed as $c$ and $o$ respectively in the dictionary.
The conditional probability of generating the context word for the given central target word can be obtained by performing a softmax operation on the vector inner product:
-->

*dịch đoạn phía trên*


$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$


<!--
where vocabulary index set $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$.
Assume that a text sequence of length $T$ is given, where the word at timestep $t$ is denoted as $w^{(t)}$.
Assume that context words are independently generated given center words.
When context window size is $m$, the likelihood function of the skip-gram model is the joint probability of generating all the context words given any center word
-->

*dịch đoạn phía trên*


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$


<!--
Here, any timestep that is less than 1 or greater than $T$ can be ignored.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->


<!--
### Skip-Gram Model Training
-->

### *dịch đoạn phía trên*


<!--
The skip-gram model parameters are the central target word vector and context word vector for each individual word.
In the training process, we are going to learn the model parameters by maximizing the likelihood function, which is also known as maximum likelihood estimation.
This is equivalent to minimizing the following loss function:
-->

*dịch đoạn phía trên*


$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$


<!--
If we use the SGD, in each iteration we are going to pick a shorter subsequence through random sampling to compute the loss for that subsequence, 
and then compute the gradient to update the model parameters.
The key of gradient computation is to compute the gradient of the logarithmic conditional probability for the central word vector and the context word vector.
By definition, we first have
-->

*dịch đoạn phía trên*


$$\log P(w_o \mid w_c) =
\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$


<!--
Through differentiation, we can get the gradient $\mathbf{v}_c$ from the formula above.
-->

*dịch đoạn phía trên*


$$
\begin{aligned}
\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}
&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\
&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.
\end{aligned}
$$


<!--
Its computation obtains the conditional probability for all the words in the dictionary given the central target word $w_c$.
We then use the same method to obtain the gradients for other word vectors.
-->

*dịch đoạn phía trên*


<!--
After the training, for any word in the dictionary with index $i$, we are going to get its two word vector sets $\mathbf{v}_i$ and $\mathbf{u}_i$.
In applications of natural language processing, the central target word vector in the skip-gram model is generally used as the representation vector of a word.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## The Continuous Bag of Words (CBOW) Model
-->

## *dịch đoạn phía trên*


<!--
The continuous bag of words (CBOW) model is similar to the skip-gram model.
The biggest difference is that the CBOW model assumes that the central target word is generated based on the context words before and after it in the text sequence.
With the same text sequence "the", "man", "loves", "his" and "son", in which "loves" is the central target word, given a context window size of 2, 
the CBOW model is concerned with the conditional probability of generating 
the target word "loves" based on the context words "the", "man", "his" and "son"(as shown in :numref:`fig_cbow`), such as
-->

*dịch đoạn phía trên*


$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$


<!--
![The CBOW model cares about the conditional probability of generating the central target word from given context words.](../img/cbow.svg)
-->

![*dịch mô tả phía trên*](../img/cbow.svg)
:label:`fig_cbow`


<!--
Since there are multiple context words in the CBOW model, we will average their word vectors and then use the same method as the skip-gram model to compute the conditional probability.
We assume that $\mathbf{v_i}\in\mathbb{R}^d$ and $\mathbf{u_i}\in\mathbb{R}^d$ are the context word vector 
and central target word vector of the word with index $i$ in the dictionary (notice that the symbols are opposite to the ones in the skip-gram model).
Let central target word $w_c$ be indexed as $c$, and context words $w_{o_1}, \ldots, w_{o_{2m}}$ be indexed as $o_1, \ldots, o_{2m}$ in the dictionary.
Thus, the conditional probability of generating a central target word from the given context word is
-->

*dịch đoạn phía trên*


$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$


<!--
For brevity, denote $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$, and $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$.
The equation above can be simplified as
-->

*dịch đoạn phía trên*


$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$


<!--
Given a text sequence of length $T$, we assume that the word at timestep $t$ is $w^{(t)}$, and the context window size is $m$.
The likelihood function of the CBOW model is the probability of generating any central target word from the context words.
-->

*dịch đoạn phía trên*


$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### CBOW Model Training
-->

### Huấn luyện Mô hình CBOW


<!--
CBOW model training is quite similar to skip-gram model training.
The maximum likelihood estimation of the CBOW model is equivalent to minimizing the loss function.
-->

Quá trình huấn luyện mô hình CBOW khá giống với quá trình huấn luyện mô hình skip-gram.
Uớc lượng hợp lý cực đại của mô hình CBOW tương đương với việc cực tiểu hoá hàm mất mát.


$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$


<!--
Notice that
-->

Lưu ý rằng


$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$


<!--
Through differentiation, we can compute the logarithm of the conditional probability of the gradient of any context word vector $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$) in the formula above.
-->

Thông qua phép vi phân, ta có thể tính log của xác suất có điều kiện của gradient của bất kỳ vector từ ngữ cảnh nào $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$) trong công thức trên.


$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$


<!--
We then use the same method to obtain the gradients for other word vectors.
Unlike the skip-gram model, we usually use the context word vector as the representation vector for a word in the CBOW model.
-->

Sau đó, ta sử dụng cùng một phương pháp để tính gradient cho các vector của từ khác.
Không giống như mô hình skip-gam, ta thường sử dụng vector từ ngữ cảnh làm vector đại diện cho một từ trong mô hình CBOW.

## Tóm tắt

<!--
* A word vector is a vector used to represent a word. 
The technique of mapping words to vectors of real numbers is also known as word embedding.
* Word2vec includes both the continuous bag of words (CBOW) and skip-gram models. 
The skip-gram model assumes that context words are generated based on the central target word. 
The CBOW model assumes that the central target word is generated based on the context words.
-->

* Vector từ là một vector được sử dụng để biểu diễn một từ.
Kỹ thuật ánh xạ từ sang vector các số thực còn được gọi là kỹ thuật embedding từ.
* Word2vec bao gồm cả mô hình túi từ liên tục (CBOW) và mô hình skip-gam.
Mô hình skip-gam giả định rằng các từ ngữ cảnh được sinh ra dựa trên từ đích trung tâm.
Mô hình CBOW giả định rằng từ đích trung tâm được sinh ra dựa trên các từ ngữ cảnh.


## Bài tập

<!--
1. What is the computational complexity of each gradient? If the dictionary contains a large volume of words, what problems will this cause?
2. There are some fixed phrases in the English language which consist of multiple words, such as "new york".
How can you train their word vectors? Hint: See section 4 in the Word2vec paper[2].
3. Use the skip-gram model as an example to think about the design of a word2vec model. 
What is the relationship between the inner product of two word vectors and the cosine similarity in the skip-gram model? 
For a pair of words with close semantical meaning, why it is likely for their word vector cosine similarity to be high?
-->

1. Độ phức tạp tính toán của mỗi gradient là gì? Nếu từ điển chứa một lượng lớn các từ, điều này sẽ gây ra vấn đề gì?
2. Có một số cụm từ cố định trong tiếng Anh bao gồm nhiều từ, chẳng hạn như "new york".
Bạn sẽ huấn luyện các vector từ của chúng như thế nào? Gợi ý: Xem phần 4 trong bài báo Word2vec[2].
3. Sử dụng mô hình skip-gam làm ví dụ để tìm hiểu về thiết kế của mô hình word2vec.
Mối quan hệ giữa tích vô hướng của hai vector từ và độ tương tự cosine trong mô hình skip-gam là gì?
Đối với một cặp từ có ngữ nghĩa gần nhau, tại sao khả năng độ tương tự cosine giữa hai vector từ này lại cao?


<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/381)
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
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* Nguyễn Văn Quang



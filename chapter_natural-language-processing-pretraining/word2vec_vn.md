<!--
# Word Embedding (word2vec)
-->

# Embedding Từ (word2vec)
:label:`sec_word2vec`


<!--
A natural language is a complex system that we use to express meanings.
In this system, words are the basic unit of linguistic meaning.
As its name implies, a word vector is a vector used to represent a word.
It can also be thought of as the feature vector of a word.
The technique of mapping words to vectors of real numbers is also known as word embedding.
Over the last few years, word embedding has gradually become basic knowledge in natural language processing.
-->

Ngôn ngữ tự nhiên là một hệ thống phức tạp mà con người sử dụng để diễn đạt ngữ nghĩa. 
Trong hệ thống này, từ là đơn vị cơ bản của ngữ nghĩa.
Như tên gọi của nó, một vector từ (*word vector*) là một vector được sử dụng để biểu diễn một từ.
Vector từ cũng có thể được xem là vector đặc trưng của một từ.
Kỹ thuật ánh xạ từ ngữ sang vector số thực còn được gọi là kỹ thuật embedding từ (*word embedding*).
Trong vài năm gần đây, embedding từ dần trở thành kiến thức cơ bản trong xử lý ngôn ngữ tự nhiên.

<!--
## Why Not Use One-hot Vectors?
-->

## Tại sao không Sử dụng Vector One-hot?


<!--
We used one-hot vectors to represent words (characters are words) in :numref:`sec_rnn_scratch`.
Recall that when we assume the number of different words in a dictionary (the dictionary size) is $N$, each word can correspond one-to-one with consecutive integers from 0 to $N-1$.
These integers that correspond to words are called the indices of the words.
We assume that the index of a word is $i$.
In order to get the one-hot vector representation of the word, we create a vector of all 0s with a length of $N$ and set element $i$ to 1.
In this way, each word is represented as a vector of length $N$ that can be used directly by the neural network.
-->


Chúng ta đã sử dụng vector one-hot để đại diện cho từ (thực chất là ký tự) trong :numref:`sec_rnn_scratch`.
Nhớ lại rằng khi giả sử số lượng các từ riêng biệt trong từ điển (tức kích thước từ điển) là $N$, 
mỗi từ có thể tương ứng một-một với các số nguyên liên tiếp từ 0 đến $N-1$, được gọi là chỉ số của từ.
Giả sử chỉ số của một từ là $i$.
Để thu được biểu diễn vector one-hot của từ đó, ta tạo một vector có $N$ phần tử có giá trị là 0 và đặt phần tử thứ $i$ bằng 1.
Theo đó, mỗi từ được biểu diễn dưới dạng vector có độ dài $N$ có thể được trực tiếp đưa vào mạng nơ-ron.


<!--
Although one-hot word vectors are easy to construct, they are usually not a good choice.
One of the major reasons is that the one-hot word vectors cannot accurately express the similarity between different words, such as the cosine similarity that we commonly use.
For the vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, their cosine similarities are the cosines of the angles between them:
-->

Mặc dù rất dễ xây dựng các vector one-hot, nhưng chúng thường không phải là lựa chọn tốt.
Một trong những lý do chính là các vector one-hot không thể biểu diễn một cách chính xác độ tương tự giữa các từ khác nhau, chẳng hạn như độ tương tự cô-sin mà ta thường sử dụng.
Độ tương tự cô-sin của hai vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$ là giá trị cô-sin của góc giữa chúng:


$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$


<!--
Since the cosine similarity between the one-hot vectors of any two different words is 0, 
it is difficult to use the one-hot vector to accurately represent the similarity between multiple different words.
-->

Do độ tương tự cô-sin giữa các vector one-hot của bất kỳ hai từ khác nhau nào đều bằng 0, 
nên rất khó sử dụng vector one-hot để biểu diễn độ tương tự giữa các từ khác nhau.


<!--
[Word2vec](https://code.google.com/archive/p/word2vec/) is a tool that we came up with to solve the problem above.
It represents each word with a fixed-length vector and uses these vectors to better indicate the similarity and analogy relationships between different words.
The Word2vec tool contains two models: skip-gram :cite:`Mikolov.Sutskever.Chen.ea.2013` and continuous bag of words (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`.
Next, we will take a look at the two models and their training methods.
-->


[Word2vec](https://code.google.com/archive/p/word2vec/) là một công cụ được phát minh để giải quyết vấn đề trên.
Nó biểu diễn mỗi từ bằng một vector có độ dài cố định và sử dụng những vector này để biểu thị tốt hơn độ tương tự và và các quan hệ loại suy (*analogy relationship*) giữa các từ.
Công cụ Word2vec gồm hai mô hình: skip-gam :cite:`Mikolov.Sutskever.Chen.ea.2013` và túi từ liên tục (*continuous bag of words* – CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`.
Tiếp theo, ta sẽ xem xét hai mô hình này và phương pháp huấn luyện chúng.


<!--
## The Skip-Gram Model
-->

## Mô hình Skip-Gram


<!--
The skip-gram model assumes that a word can be used to generate the words that surround it in a text sequence.
For example, we assume that the text sequence is "the", "man", "loves", "his", and "son".
We use "loves" as the central target word and set the context window size to 2.
As shown in :numref:`fig_skip_gram`, given the central target word "loves", the skip-gram model is concerned with the conditional probability 
for generating the context words, "the", "man", "his" and "son", that are within a distance of no more than 2 words, which is
-->


Mô hình skip-gam giả định rằng một từ có thể được sử dụng để sinh ra các từ xung quanh nó trong một chuỗi văn bản.
Ví dụ, giả sử chuỗi văn bản là "the", "man", "loves", "his" và "son".
Ta sử dụng "loves" làm từ đích trung tâm và đặt kích thước cửa sổ ngữ cảnh bằng 2.
Như mô tả trong :numref:`fig_skip_gram`, với từ đích trung tâm "loves", mô hình skip-gram quan tâm đến xác suất có điều kiện 
sinh ra các từ ngữ cảnh ("the", "man", "his" và "son") nằm trong khoảng cách không quá 2 từ:


$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$


<!--
We assume that, given the central target word, the context words are generated independently of each other.
In this case, the formula above can be rewritten as
-->


Ta giả định rằng, với từ đích trung tâm cho trước, các từ ngữ cảnh được sinh ra độc lập với nhau.
Trong trường hợp này, công thức trên có thể được viết lại thành


$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$


<!--
![The skip-gram model cares about the conditional probability of generating context words for a given central target word.](../img/skip-gram.svg)
-->


![Mô hình skip-gram quan tâm đến xác suất có điều kiện sinh ra các từ ngữ cảnh với một từ đích trung tâm cho trước.](../img/skip-gram.svg)
:label:`fig_skip_gram`


<!--
In the skip-gram model, each word is represented as two $d$-dimension vectors, which are used to compute the conditional probability.
We assume that the word is indexed as $i$ in the dictionary, its vector is represented as $\mathbf{v}_i\in\mathbb{R}^d$ 
when it is the central target word, and $\mathbf{u}_i\in\mathbb{R}^d$ when it is a context word.
Let the central target word $w_c$ and context word $w_o$ be indexed as $c$ and $o$ respectively in the dictionary.
The conditional probability of generating the context word for the given central target word can be obtained by performing a softmax operation on the vector inner product:
-->


Trong mô hình skip-gam, mỗi từ được biểu diễn bằng hai vector $d$-chiều để tính xác suất có điều kiện.
Giả sử chỉ số của một từ trong từ điển là $i$, vector của từ được biểu diễn là $\mathbf{v}_i\in\mathbb{R}^d$ 
khi từ này là từ đích trung tâm và là $\mathbf{u}_i\in\mathbb{R}^d$ khi từ này là một từ ngữ cảnh.
Gọi $c$ và $o$ lần lượt là chỉ số của từ đích trung tâm $w_c$ và từ ngữ cảnh $w_o$ trong từ điển.
Có thể thu được xác suất có điều kiện sinh ra từ ngữ cảnh cho một từ đích trung tâm cho trước bằng phép toán softmax trên tích vô hướng của vector:


$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$


<!--
where vocabulary index set $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$.
Assume that a text sequence of length $T$ is given, where the word at timestep $t$ is denoted as $w^{(t)}$.
Assume that context words are independently generated given center words.
When context window size is $m$, the likelihood function of the skip-gram model is the joint probability of generating all the context words given any center word
-->


trong đó, tập chỉ số trong bộ từ vựng là $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$.
Giả sử trong một chuỗi văn bản có độ dài $T$, từ tại bước thời gian $t$ được ký hiệu là $w^{(t)}$.
Giả sử rằng các từ ngữ cảnh được sinh độc lập với từ trung tâm cho trước.
Khi kích thước cửa sổ ngữ cảnh là $m$, hàm hợp lý (*likelihood*) của mô hình skip-gam là xác suất kết hợp sinh ra tất cả các từ ngữ cảnh với bất kỳ từ trung tâm cho trước nào


$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$


<!--
Here, any timestep that is less than 1 or greater than $T$ can be ignored.
-->


Ở đây, bất kỳ bước thời gian nào nhỏ hơn 1 hoặc lớn hơn $T$ đều có thể được bỏ qua.


<!--
### Skip-Gram Model Training
-->

### Huấn luyện Mô hình Skip-Gram


<!--
The skip-gram model parameters are the central target word vector and context word vector for each individual word.
In the training process, we are going to learn the model parameters by maximizing the likelihood function, which is also known as maximum likelihood estimation.
This is equivalent to minimizing the following loss function:
-->

Các tham số trong mô hình skip-gram là vector từ đích trung tâm và vector từ ngữ cảnh cho từng từ riêng lẻ.
Trong quá trình huấn luyện, chúng ta sẽ học các tham số mô hình bằng cách cực đại hóa hàm hợp lý, còn gọi là ước lượng hợp lý cực đại.
Việc này tương tự với việc giảm thiểu hàm mất mát sau đây:


$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$


<!--
If we use the SGD, in each iteration we are going to pick a shorter subsequence through random sampling to compute the loss for that subsequence, 
and then compute the gradient to update the model parameters.
The key of gradient computation is to compute the gradient of the logarithmic conditional probability for the central word vector and the context word vector.
By definition, we first have
-->

Nếu ta dùng SGD, thì trong mỗi vòng lặp, ta chọn ra một chuỗi con nhỏ hơn bằng việc lấy mẫu ngẫu nhiên để tính toán mất mát cho chuỗi con đó,
rồi sau đó tính gradient để cập nhật các tham số mô hình.
Điểm then chốt của việc tính toán gradient là tính gradient của logarit xác suất có điều kiện cho vector từ trung tâm và vector từ ngữ cảnh.
Đầu tiên, theo định nghĩa ta có


$$\log P(w_o \mid w_c) =
\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$


<!--
Through differentiation, we can get the gradient $\mathbf{v}_c$ from the formula above.
-->

Thông qua phép tính đạo hàm, ta nhận được giá trị gradient $\mathbf{v}_c$ từ công thức trên.


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

Phép tính cho ra xác suất có điều kiện cho mọi từ có trong từ điển với từ đích trung tâm $w_c$ cho trước.
Sau đó, ta lại sử dụng phương pháp đó để tìm gradient cho các vector từ khác.


<!--
After the training, for any word in the dictionary with index $i$, we are going to get its two word vector sets $\mathbf{v}_i$ and $\mathbf{u}_i$.
In applications of natural language processing, the central target word vector in the skip-gram model is generally used as the representation vector of a word.
-->

Sau khi huấn luyện xong, với từ bất kỳ có chỉ số là $i$ trong từ điển, ta sẽ nhận được tập hai vector từ $\mathbf{v}_i$ và $\mathbf{u}_i$.
Trong các ứng dụng xử lý ngôn ngữ tự nhiên, vector từ đích trung tâm trong mô hình skip-gram thường được sử dụng để làm vector biểu diễn một từ.


<!--
## The Continuous Bag of Words (CBOW) Model
-->

## Mô hình Túi từ Liên tục (CBOW)


<!--
The continuous bag of words (CBOW) model is similar to the skip-gram model.
The biggest difference is that the CBOW model assumes that the central target word is generated based on the context words before and after it in the text sequence.
With the same text sequence "the", "man", "loves", "his" and "son", in which "loves" is the central target word, given a context window size of 2, 
the CBOW model is concerned with the conditional probability of generating 
the target word "loves" based on the context words "the", "man", "his" and "son"(as shown in :numref:`fig_cbow`), such as
-->

Mô hình túi từ liên tục (*Continuous bag of words* - CBOW) tương tự như mô hình skip-gram.
Khác biệt lớn nhất là mô hình CBOW giả định rằng từ đích trung tâm được tạo ra dựa trên các từ ngữ cảnh phía trước và sau nó trong một chuỗi văn bản.
Với cùng một chuỗi văn bản gồm các từ "the", "man", "loves", "his" và "son", trong đó "love" là từ đích trung tâm, với kích thước cửa sổ ngữ cảnh bằng 2,
mô hình CBOW quan tâm đến xác suất có điều kiện để sinh ra
từ đích "love" dựa trên các từ ngữ cảnh "the", "man", "his" và "son" (minh họa ở :numref:`fig_cbow`) như sau:


$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$


<!--
![The CBOW model cares about the conditional probability of generating the central target word from given context words.](../img/cbow.svg)
-->

![Mô hình CBOW quan tâm đến xác suất có điều kiện tạo ra từ đích trung tâm dựa trên các từ ngữ cảnh cho trước.](../img/cbow.svg)
:label:`fig_cbow`


<!--
Since there are multiple context words in the CBOW model, we will average their word vectors and then use the same method as the skip-gram model to compute the conditional probability.
We assume that $\mathbf{v_i}\in\mathbb{R}^d$ and $\mathbf{u_i}\in\mathbb{R}^d$ are the context word vector 
and central target word vector of the word with index $i$ in the dictionary (notice that the symbols are opposite to the ones in the skip-gram model).
Let central target word $w_c$ be indexed as $c$, and context words $w_{o_1}, \ldots, w_{o_{2m}}$ be indexed as $o_1, \ldots, o_{2m}$ in the dictionary.
Thus, the conditional probability of generating a central target word from the given context word is
-->

Vì có quá nhiều từ ngữ cảnh trong mô hình CBOW, ta sẽ lấy trung bình các vector từ của chúng và sau đó sử dụng phương pháp tương tự như trong mô hình skip-gram để tính xác suất có điều kiện.
Giả sử $\mathbf{v_i}\in\mathbb{R}^d$ và $\mathbf{u_i}\in\mathbb{R}^d$ là vector từ ngữ cảnh
và vector từ đích trung tâm của từ có chỉ số $i$ trong từ điển (lưu ý rằng các ký hiệu này ngược với các ký hiệu trong mô hình skip-gram).
Gọi $c$ là chỉ số của từ đích trung tâm $w_c$, và $o_1, \ldots, o_{2m}$ là chỉ số các từ ngữ cảnh $w_{o_1}, \ldots, w_{o_{2m}}$ trong từ điển.
Do đó, xác suất có điều kiện sinh ra từ đích trung tâm dựa vào các từ ngữ cảnh cho trước là 


$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$


<!--
For brevity, denote $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$, and $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$.
The equation above can be simplified as
-->

Để rút gọn, ký hiệu $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$, và $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$.
Phương trình trên được đơn giản hóa thành


$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$


<!--
Given a text sequence of length $T$, we assume that the word at timestep $t$ is $w^{(t)}$, and the context window size is $m$.
The likelihood function of the CBOW model is the probability of generating any central target word from the context words.
-->

Cho một chuỗi văn bản có độ dài $T$, ta giả định rằng từ xuất hiện tại bước thời gian $t$ là $w^{(t)}$, và kích thước của cửa sổ ngữ cảnh là $m$.
Hàm hợp lý của mô hình CBOW là xác suất sinh ra bất kỳ từ đích trung tâm nào dựa vào những từ ngữ cảnh.


$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$


<!--
### CBOW Model Training
-->

### Huấn luyện Mô hình CBOW


<!--
CBOW model training is quite similar to skip-gram model training.
The maximum likelihood estimation of the CBOW model is equivalent to minimizing the loss function.
-->

Quá trình huấn luyện mô hình CBOW khá giống với quá trình huấn luyện mô hình skip-gram.
Uớc lượng hợp lý cực đại của mô hình CBOW tương đương với việc cực tiểu hóa hàm mất mát:


$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$


<!--
Notice that
-->

Lưu ý rằng


$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$


<!--
Through differentiation, we can compute the logarithm of the conditional probability of the gradient of any context word vector $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$) in the formula above.
-->

Thông qua phép đạo hàm, ta có thể tính log của xác suất có điều kiện của gradient của bất kỳ vector từ ngữ cảnh nào $\mathbf{v}_{o_i}$($i = 1, \ldots, 2m$) trong công thức trên.


$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$


<!--
We then use the same method to obtain the gradients for other word vectors.
Unlike the skip-gram model, we usually use the context word vector as the representation vector for a word in the CBOW model.
-->

Sau đó, ta sử dụng cùng phương pháp đó để tính gradient cho các vector của từ khác.
Không giống như mô hình skip-gam, trong mô hình CBOW ta thường sử dụng vector từ ngữ cảnh làm vector biểu diễn một từ.


## Tóm tắt

<!--
* A word vector is a vector used to represent a word. 
The technique of mapping words to vectors of real numbers is also known as word embedding.
* Word2vec includes both the continuous bag of words (CBOW) and skip-gram models. 
The skip-gram model assumes that context words are generated based on the central target word. 
The CBOW model assumes that the central target word is generated based on the context words.
-->

* Vector từ là một vector được sử dụng để biểu diễn một từ.
Kỹ thuật ánh xạ các từ sang vector số thực còn được gọi là kỹ thuật embedding từ. 
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

1. Độ phức tạp tính toán của mỗi gradient là bao nhiêu? Nếu từ điển chứa một lượng lớn các từ, điều này sẽ gây ra vấn đề gì?
2. Có một số cụm từ cố định trong tiếng Anh bao gồm nhiều từ, chẳng hạn như "new york".
Bạn sẽ huấn luyện các vector từ của chúng như thế nào? Gợi ý: Xem phần 4 trong bài báo Word2vec[2].
3. Sử dụng mô hình skip-gam làm ví dụ để tìm hiểu về thiết kế của mô hình word2vec.
Mối quan hệ giữa tích vô hướng của hai vector từ và độ tương tự cô-sin trong mô hình skip-gam là gì?
Đối với một cặp từ có ngữ nghĩa gần nhau, tại sao hai vector từ này lại thường có độ tương tự cô-sin cao?


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/381)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Quang
* Nguyễn Văn Cường
* Phạm Đăng Khoa
* Lê Khắc Hồng Phúc

*Lần cập nhật gần nhất: 12/09/2020. (Cập nhật lần cuối từ nội dung gốc: 29/08/2020)*

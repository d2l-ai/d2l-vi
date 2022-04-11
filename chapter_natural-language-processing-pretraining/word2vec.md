# Từ nhúng (word2vec)
:label:`sec_word2vec`

Ngôn ngữ tự nhiên là một hệ thống phức tạp dùng để thể hiện ý nghĩa. Trong hệ thống này, từ ngữ là đơn vị cơ bản của nghĩa. Như tên của nó,
*vectơ từ* là các vectơ được sử dụng để biểu diễn các từ,
và cũng có thể được coi là vectơ tính năng hoặc biểu diễn của các từ. Kỹ thuật lập bản đồ từ với vectơ thực được gọi là *word embedding*. Trong những năm gần đây, nhúng từ đã dần trở thành kiến thức cơ bản về xử lý ngôn ngữ tự nhiên. 

## Vectơ một nóng là một lựa chọn tồi

Chúng tôi sử dụng vectơ một nóng để biểu diễn các từ (ký tự là từ) trong :numref:`sec_rnn_scratch`. Giả sử số từ khác nhau trong từ điển (cỡ từ điển) là $N$, và mỗi từ tương ứng với một số nguyên (index) khác nhau từ $0$ đến $N−1$. Để có được biểu diễn vectơ một nóng cho bất kỳ từ nào có chỉ số $i$, chúng ta tạo một vector chiều dài-$N$ với tất cả 0s và đặt phần tử ở vị trí $i$ thành 1. Bằng cách này, mỗi từ được biểu diễn dưới dạng vectơ có chiều dài $N$, và nó có thể được sử dụng trực tiếp bởi các mạng thần kinh. 

Mặc dù vectơ từ một nóng rất dễ xây dựng, chúng thường không phải là một lựa chọn tốt. Một lý do chính là các vectơ từ một nóng không thể diễn tả chính xác sự tương đồng giữa các từ khác nhau, chẳng hạn như tương tự * cosine* mà chúng ta thường sử dụng. Đối với vectơ $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, sự tương đồng cosin của chúng là cosin của góc giữa chúng: 

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} \in [-1, 1].$$

Vì sự tương đồng cosin giữa các vectơ một nóng của hai từ khác nhau bất kỳ là 0, các vectơ một nóng không thể mã hóa các điểm tương đồng giữa các từ. 

## Word2vec tự giám sát

Công cụ [word2vec](https://code.google.com/archive/p/word2vec/) đã được đề xuất để giải quyết vấn đề trên. Nó ánh xạ từng từ thành một vectơ có độ dài cố định, và các vectơ này có thể thể hiện tốt hơn mối quan hệ tương đồng và tương tự giữa các từ khác nhau. Công cụ word2vec chứa hai mô hình, cụ thể là *skip-gram* :cite:`Mikolov.Sutskever.Chen.ea.2013` và * túi liên từ* (CBOW) :cite:`Mikolov.Chen.Corrado.ea.2013`. Đối với các đại diện có ý nghĩa về mặt ngữ nghĩa, đào tạo của họ dựa vào xác suất có điều kiện có thể được xem như dự đoán một số từ bằng cách sử dụng một số từ xung quanh của họ trong thể. Vì sự giám sát đến từ dữ liệu không có nhãn, cả bỏ qua gram và túi từ liên tục đều là các mô hình tự giám sát. 

Sau đây, chúng tôi sẽ giới thiệu hai mô hình này và phương pháp đào tạo của họ. 

## Mô hình Skip-Gram
:label:`subsec_skip-gram`

Mô hình *skip-gram* giả định rằng một từ có thể được sử dụng để tạo ra các từ xung quanh của nó trong một chuỗi văn bản. Lấy chuỗi văn bản “các”, “người đàn ông”, “yêu thương”, “anh ấy”, “con trai” làm ví dụ. Hãy để chúng tôi chọn “loves” làm từ *center* và đặt kích thước cửa sổ ngữ cảnh thành 2. Như thể hiện trong :numref:`fig_skip_gram`, với từ trung tâm “yêu thương”, mô hình skip-gram xem xét xác suất có điều kiện để tạo ra các từ ngữ cảnh *: “the”, “man”, “his”, và “con trai”, không quá 2 từ cách từ trung tâm: 

$$P(\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}\mid\textrm{"loves"}).$$

Giả sử rằng các từ ngữ cảnh được tạo độc lập cho từ trung tâm (tức là độc lập có điều kiện). Trong trường hợp này, xác suất có điều kiện ở trên có thể được viết lại dưới dạng 

$$P(\textrm{"the"}\mid\textrm{"loves"})\cdot P(\textrm{"man"}\mid\textrm{"loves"})\cdot P(\textrm{"his"}\mid\textrm{"loves"})\cdot P(\textrm{"son"}\mid\textrm{"loves"}).$$

![The skip-gram model considers the conditional probability of generating the surrounding context words given a center word.](../img/skip-gram.svg)
:label:`fig_skip_gram`

Trong mô hình skip-gram, mỗi từ có hai biểu diễn $d$-dimensional-vector để tính xác suất có điều kiện. Cụ thể hơn, đối với bất kỳ từ nào có chỉ mục $i$ trong từ điển, biểu thị bằng $\mathbf{v}_i\in\mathbb{R}^d$ và $\mathbf{u}_i\in\mathbb{R}^d$ hai vectơ của nó khi được sử dụng như một từ *center* và một từ *context*, tương ứng. Xác suất có điều kiện tạo ra bất kỳ từ ngữ cảnh nào $w_o$ (với chỉ số $o$ trong từ điển) cho từ trung tâm $w_c$ (với chỉ số $c$ trong từ điển) có thể được mô hình hóa bằng một hoạt động softmax trên các sản phẩm chấm vector: 

$$P(w_o \mid w_c) = \frac{\text{exp}(\mathbf{u}_o^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)},$$
:eqlabel:`eq_skip-gram-softmax`

nơi chỉ số từ vựng đặt $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$. Cho một chuỗi văn bản có độ dài $T$, trong đó từ lúc bước $t$ được ký hiệu là $w^{(t)}$. Giả sử rằng các từ ngữ cảnh được tạo độc lập cho bất kỳ từ trung tâm nào. Đối với kích thước cửa sổ ngữ cảnh $m$, hàm khả năng của mô hình skip-gram là xác suất tạo ra tất cả các từ ngữ cảnh cho bất kỳ từ trung tâm nào: 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

trong đó bất kỳ bước thời gian nào nhỏ hơn $1$ hoặc lớn hơn $T$ có thể được bỏ qua. 

### Đào tạo

Các tham số mô hình skip-gram là vector từ trung tâm và vector từ ngữ cảnh cho mỗi từ trong từ vựng. Trong đào tạo, chúng ta tìm hiểu các tham số mô hình bằng cách tối đa hóa hàm khả năng (tức là ước tính khả năng tối đa). Điều này tương đương với việc giảm thiểu hàm mất sau: 

$$ - \sum_{t=1}^{T} \sum_{-m \leq j \leq m,\ j \neq 0} \text{log}\, P(w^{(t+j)} \mid w^{(t)}).$$

Khi sử dụng stochastic gradient descent để giảm thiểu tổn thất, trong mỗi lần lặp lại, chúng ta có thể lấy mẫu ngẫu nhiên một dãy con ngắn hơn để tính toán gradient (stochastic) cho dãy tiếp theo này để cập nhật các tham số mô hình. Để tính toán gradient (stochastic) này, chúng ta cần phải có được gradient của xác suất điều kiện nhật ký đối với vector từ trung tâm và vector từ ngữ cảnh. Nói chung, theo :eqref:`eq_skip-gram-softmax` xác suất có điều kiện nhật ký liên quan đến bất kỳ cặp nào của từ trung tâm $w_c$ và từ ngữ cảnh $w_o$ là 

$$\log P(w_o \mid w_c) =\mathbf{u}_o^\top \mathbf{v}_c - \log\left(\sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)\right).$$
:eqlabel:`eq_skip-gram-log`

Thông qua sự khác biệt, chúng ta có thể có được gradient của nó đối với vector từ trung tâm $\mathbf{v}_c$ như 

$$\begin{aligned}\frac{\partial \text{log}\, P(w_o \mid w_c)}{\partial \mathbf{v}_c}&= \mathbf{u}_o - \frac{\sum_{j \in \mathcal{V}} \exp(\mathbf{u}_j^\top \mathbf{v}_c)\mathbf{u}_j}{\sum_{i \in \mathcal{V}} \exp(\mathbf{u}_i^\top \mathbf{v}_c)}\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} \left(\frac{\text{exp}(\mathbf{u}_j^\top \mathbf{v}_c)}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \mathbf{v}_c)}\right) \mathbf{u}_j\\&= \mathbf{u}_o - \sum_{j \in \mathcal{V}} P(w_j \mid w_c) \mathbf{u}_j.\end{aligned}$$
:eqlabel:`eq_skip-gram-grad`

Lưu ý rằng việc tính toán trong :eqref:`eq_skip-gram-grad` đòi hỏi xác suất có điều kiện của tất cả các từ trong từ điển với $w_c$ là từ trung tâm. Các gradient cho các vectơ từ khác có thể thu được theo cùng một cách. 

Sau khi đào tạo, đối với bất kỳ từ nào có chỉ mục $i$ trong từ điển, chúng tôi có được cả hai vectơ từ $\mathbf{v}_i$ (là từ trung tâm) và $\mathbf{u}_i$ (như từ ngữ cảnh). Trong các ứng dụng xử lý ngôn ngữ tự nhiên, các vectơ từ trung tâm của mô hình skip-gram thường được sử dụng làm biểu diễn từ. 

## Mô hình túi từ liên tục (CBOW)

Mô hình túi từ* (CBOW) liên tục tương tự như mô hình bỏ qua gram. Sự khác biệt lớn so với mô hình skip-gram là mô hình túi từ liên tục giả định rằng một từ trung tâm được tạo ra dựa trên các từ ngữ cảnh xung quanh của nó trong chuỗi văn bản. Ví dụ, trong cùng một chuỗi văn bản “the”, “man”, “loves”, “his”, và “son”, với “yêu” là từ trung tâm và kích thước cửa sổ ngữ cảnh là 2, túi liên tục của mô hình từ xem xét xác suất có điều kiện tạo ra từ trung tâm “yêu” dựa trên các từ ngữ cảnh “the”, “man”, “his” và “son “(như thể hiện trong :numref:`fig_cbow`), đó là 

$$P(\textrm{"loves"}\mid\textrm{"the"},\textrm{"man"},\textrm{"his"},\textrm{"son"}).$$

![The continuous bag of words model considers the conditional probability of generating the center word given its surrounding context words.](../img/cbow.svg)
:eqlabel:`fig_cbow`

Vì có nhiều từ ngữ cảnh trong mô hình túi từ liên tục, các vectơ từ ngữ cảnh này được tính trung bình trong việc tính xác suất có điều kiện. Cụ thể, đối với bất kỳ từ nào có chỉ số $i$ trong từ điển, biểu thị bằng $\mathbf{v}_i\in\mathbb{R}^d$ và $\mathbf{u}_i\in\mathbb{R}^d$ hai vectơ của nó khi được sử dụng như một từ *context* và một từ *center* (nghĩa được chuyển trong mô hình skip-gram), tương ứng. Xác suất có điều kiện tạo ra bất kỳ từ trung tâm nào $w_c$ (với chỉ số $c$ trong từ điển) cho các từ ngữ cảnh xung quanh của nó $w_{o_1}, \ldots, w_{o_{2m}}$ (với chỉ số $o_1, \ldots, o_{2m}$ trong từ điển) có thể được mô hình hóa bởi 

$$P(w_c \mid w_{o_1}, \ldots, w_{o_{2m}}) = \frac{\text{exp}\left(\frac{1}{2m}\mathbf{u}_c^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}{ \sum_{i \in \mathcal{V}} \text{exp}\left(\frac{1}{2m}\mathbf{u}_i^\top (\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}}) \right)}.$$
:eqlabel:`fig_cbow-full`

Đối với ngắn gọn, hãy để $\mathcal{W}_o= \{w_{o_1}, \ldots, w_{o_{2m}}\}$ và $\bar{\mathbf{v}}_o = \left(\mathbf{v}_{o_1} + \ldots, + \mathbf{v}_{o_{2m}} \right)/(2m)$. Sau đó, :eqref:`fig_cbow-full` có thể được đơn giản hóa như 

$$P(w_c \mid \mathcal{W}_o) = \frac{\exp\left(\mathbf{u}_c^\top \bar{\mathbf{v}}_o\right)}{\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)}.$$

Cho một chuỗi văn bản có chiều dài $T$, trong đó từ lúc bước $t$ được ký hiệu là $w^{(t)}$. Đối với kích thước cửa sổ ngữ cảnh $m$, chức năng khả năng của túi liên tục của mô hình từ là xác suất tạo ra tất cả các từ trung tâm cho các từ ngữ cảnh của chúng: 

$$ \prod_{t=1}^{T}  P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

### Đào tạo

Đào tạo túi liên tục của các mô hình từ gần giống như đào tạo mô hình skip-gram. Ước tính khả năng tối đa của mô hình túi từ liên tục tương đương với việc giảm thiểu chức năng mất sau: 

$$  -\sum_{t=1}^T  \text{log}\, P(w^{(t)} \mid  w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}).$$

Chú ý rằng 

$$\log\,P(w_c \mid \mathcal{W}_o) = \mathbf{u}_c^\top \bar{\mathbf{v}}_o - \log\,\left(\sum_{i \in \mathcal{V}} \exp\left(\mathbf{u}_i^\top \bar{\mathbf{v}}_o\right)\right).$$

Thông qua sự khác biệt, chúng ta có thể có được gradient của nó đối với bất kỳ vector từ ngữ cảnh $\mathbf{v}_{o_i}$ ($i = 1, \ldots, 2m$) như 

$$\frac{\partial \log\, P(w_c \mid \mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m} \left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} \frac{\exp(\mathbf{u}_j^\top \bar{\mathbf{v}}_o)\mathbf{u}_j}{ \sum_{i \in \mathcal{V}} \text{exp}(\mathbf{u}_i^\top \bar{\mathbf{v}}_o)} \right) = \frac{1}{2m}\left(\mathbf{u}_c - \sum_{j \in \mathcal{V}} P(w_j \mid \mathcal{W}_o) \mathbf{u}_j \right).$$
:eqlabel:`eq_cbow-gradient`

Các gradient cho các vectơ từ khác có thể thu được theo cùng một cách. Không giống như mô hình skip-gram, mô hình túi từ liên tục thường sử dụng vectơ từ ngữ cảnh làm biểu diễn từ. 

## Tóm tắt

* Vectơ từ là các vectơ dùng để biểu diễn các từ, và cũng có thể được coi là vectơ đặc trưng hoặc biểu diễn các từ. Kỹ thuật lập bản đồ từ với vectơ thực được gọi là nhúng từ.
* Công cụ word2vec chứa cả mô hình bỏ qua gram và túi liên tục.
* Mô hình skip-gram giả định rằng một từ có thể được sử dụng để tạo ra các từ xung quanh của nó trong một chuỗi văn bản; trong khi mô hình túi từ liên tục giả định rằng một từ trung tâm được tạo ra dựa trên các từ ngữ cảnh xung quanh của nó.

## Bài tập

1. Độ phức tạp tính toán để tính từng gradient là gì? Điều gì có thể là vấn đề nếu kích thước từ điển là rất lớn?
1. Một số cụm từ cố định trong tiếng Anh bao gồm nhiều từ, chẳng hạn như “new york”. Làm thế nào để đào tạo vectơ từ của họ? Hint: see Section 4 in the word2vec paper :cite:`Mikolov.Sutskever.Chen.ea.2013`.
1. Chúng ta hãy suy ngẫm về thiết kế word2vec bằng cách lấy mô hình skip-gram làm ví dụ. Mối quan hệ giữa tích chấm của hai vectơ từ trong mô hình skip-gram và sự tương đồng cosin là gì? Đối với một cặp từ có ngữ nghĩa tương tự, tại sao sự tương đồng cosin của vectơ từ của chúng (được đào tạo bởi mô hình skip-gram) có thể cao?

[Discussions](https://discuss.d2l.ai/t/381)

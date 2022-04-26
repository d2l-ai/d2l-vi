# Đào tạo gần đúng
:label:`sec_approx_train`

Nhớ lại các cuộc thảo luận của chúng tôi trong :numref:`sec_word2vec`. Ý tưởng chính của mô hình skip-gram là sử dụng các phép toán softmax để tính xác suất có điều kiện tạo ra một từ ngữ cảnh $w_o$ dựa trên từ trung tâm đã cho $w_c$ trong :eqref:`eq_skip-gram-softmax`, có tổn thất logarit tương ứng được đưa ra bởi ngược lại :eqref:`eq_skip-gram-log`. 

Do tính chất của hoạt động softmax, vì một từ ngữ cảnh có thể là bất cứ ai trong từ điển $\mathcal{V}$, ngược lại với :eqref:`eq_skip-gram-log` chứa tổng các mục nhiều như toàn bộ kích thước của từ vựng. Do đó, tính toán gradient cho mô hình skip-gram trong :eqref:`eq_skip-gram-grad` và cho mô hình túi-từ liên tục trong :eqref:`eq_cbow-gradient` cả hai đều chứa tổng. Thật không may, chi phí tính toán cho gradient như vậy tổng hợp trên một từ điển lớn (thường với hàng trăm ngàn hoặc hàng triệu từ) là rất lớn! 

Để giảm độ phức tạp tính toán nói trên, phần này sẽ giới thiệu hai phương pháp đào tạo gần đúng:
*lấy mẫu tiêu cực* và * phân cấp softmax*.
Do sự giống nhau giữa mô hình skip-gram và mô hình túi từ liên tục, chúng tôi sẽ chỉ lấy mô hình skip-gram làm ví dụ để mô tả hai phương pháp đào tạo gần đúng này. 

## Lấy mẫu âm
:label:`subsec_negative-sampling`

Lấy mẫu âm sửa đổi chức năng mục tiêu ban đầu. Với cửa sổ ngữ cảnh của một từ trung tâm $w_c$, thực tế là bất kỳ từ nào (ngữ cảnh) $w_o$ xuất phát từ cửa sổ ngữ cảnh này được coi là một sự kiện với xác suất được mô hình hóa bởi 

$$P(D=1\mid w_c, w_o) = \sigma(\mathbf{u}_o^\top \mathbf{v}_c),$$

trong đó $\sigma$ sử dụng định nghĩa của hàm kích hoạt sigmoid: 

$$\sigma(x) = \frac{1}{1+\exp(-x)}.$$
:eqlabel:`eq_sigma-f`

Chúng ta hãy bắt đầu bằng cách tối đa hóa xác suất chung của tất cả các sự kiện như vậy trong chuỗi văn bản để đào tạo nhúng từ. Cụ thể, đưa ra một chuỗi văn bản chiều dài $T$, biểu thị bằng $w^{(t)}$ từ tại bước thời điểm $t$ và để cho kích thước cửa sổ ngữ cảnh là $m$, xem xét tối đa hóa xác suất chung 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(D=1\mid w^{(t)}, w^{(t+j)}).$$
:eqlabel:`eq-negative-sample-pos`

Tuy nhiên, :eqref:`eq-negative-sample-pos` chỉ xem xét những sự kiện liên quan đến các ví dụ tích cực. Kết quả là, xác suất chung trong :eqref:`eq-negative-sample-pos` được tối đa hóa đến 1 chỉ khi tất cả các vectơ từ bằng vô cực. Tất nhiên, kết quả như vậy là vô nghĩa. To make the objective mục tiêu function chức năng more meaningfulý nghĩa,
*lấy mẫu tiêu cực*
thêm các ví dụ tiêu cực lấy mẫu từ một phân phối được xác định trước. 

Biểu thị bởi $S$ sự kiện mà một từ ngữ cảnh $w_o$ xuất phát từ cửa sổ ngữ cảnh của một từ trung tâm $w_c$. Đối với sự kiện này liên quan đến $w_o$, từ một phân phối được xác định trước $P(w)$ mẫu $K$ * tiếng ồn từ * không phải từ cửa sổ ngữ cảnh này. Biểu thị bởi $N_k$ sự kiện rằng một từ tiếng ồn $w_k$ ($k=1, \ldots, K$) không đến từ cửa sổ ngữ cảnh của $w_c$. Giả sử rằng những sự kiện này liên quan đến cả ví dụ tích cực và các ví dụ tiêu cực $S, N_1, \ldots, N_K$ là độc lập lẫn nhau. Lấy mẫu âm viết lại xác suất chung (chỉ liên quan đến các ví dụ tích cực) trong :eqref:`eq-negative-sample-pos` như 

$$ \prod_{t=1}^{T} \prod_{-m \leq j \leq m,\ j \neq 0} P(w^{(t+j)} \mid w^{(t)}),$$

trong đó xác suất có điều kiện được xấp xỉ thông qua các sự kiện $S, N_1, \ldots, N_K$: 

$$ P(w^{(t+j)} \mid w^{(t)}) =P(D=1\mid w^{(t)}, w^{(t+j)})\prod_{k=1,\ w_k \sim P(w)}^K P(D=0\mid w^{(t)}, w_k).$$
:eqlabel:`eq-negative-sample-conditional-prob`

Biểu thị bởi $i_t$ và $h_k$ các chỉ số của một từ $w^{(t)}$ tại bước thời gian $t$ của một chuỗi văn bản và một từ tiếng ồn $w_k$, tương ứng. Sự mất mát logarit đối với xác suất có điều kiện trong :eqref:`eq-negative-sample-conditional-prob` là 

$$
\begin{aligned}
-\log P(w^{(t+j)} \mid w^{(t)})
=& -\log P(D=1\mid w^{(t)}, w^{(t+j)}) - \sum_{k=1,\ w_k \sim P(w)}^K \log P(D=0\mid w^{(t)}, w_k)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\left(1-\sigma\left(\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right)\right)\\
=&-  \log\, \sigma\left(\mathbf{u}_{i_{t+j}}^\top \mathbf{v}_{i_t}\right) - \sum_{k=1,\ w_k \sim P(w)}^K \log\sigma\left(-\mathbf{u}_{h_k}^\top \mathbf{v}_{i_t}\right).
\end{aligned}
$$

Chúng ta có thể thấy rằng bây giờ chi phí tính toán cho gradient ở mỗi bước đào tạo không liên quan gì đến kích thước từ điển, nhưng tuyến tính phụ thuộc vào $K$. Khi đặt siêu tham số $K$ thành một giá trị nhỏ hơn, chi phí tính toán cho gradient ở mỗi bước đào tạo với lấy mẫu âm nhỏ hơn. 

## Phân cấp Softmax

Như một phương pháp đào tạo gần đúng thay thế,
*softmaxphân học*
sử dụng cây nhị phân, một cấu trúc dữ liệu minh họa trong :numref:`fig_hi_softmax`, trong đó mỗi nút lá của cây đại diện cho một từ trong từ điển $\mathcal{V}$. 

![Hierarchical softmax for approximate training, where each leaf node of the tree represents a word in the dictionary.](../img/hi-softmax.svg)
:label:`fig_hi_softmax`

Biểu thị bằng $L(w)$ số nút (bao gồm cả hai đầu) trên đường dẫn từ nút gốc đến nút lá biểu diễn từ $w$ trong cây nhị phân. Hãy để $n(w,j)$ là nút $j^\mathrm{th}$ trên đường dẫn này, với vector từ ngữ cảnh của nó là $\mathbf{u}_{n(w, j)}$. Ví dụ, $L(w_3) = 4$ trong :numref:`fig_hi_softmax`. Softmax phân cấp xấp xỉ xác suất có điều kiện trong :eqref:`eq_skip-gram-softmax` như 

$$P(w_o \mid w_c) = \prod_{j=1}^{L(w_o)-1} \sigma\left( [\![  n(w_o, j+1) = \text{leftChild}(n(w_o, j)) ]\!] \cdot \mathbf{u}_{n(w_o, j)}^\top \mathbf{v}_c\right),$$

trong đó hàm $\sigma$ được định nghĩa trong :eqref:`eq_sigma-f`, và $\text{leftChild}(n)$ là nút con trái của nút $n$: nếu $x$ là đúng, $ [\! [x]\!] = 1$; otherwise $ [\! [x]\!] = -1$. 

Để minh họa, chúng ta hãy tính toán xác suất có điều kiện tạo từ $w_3$ cho từ $w_c$ trong :numref:`fig_hi_softmax`. Điều này đòi hỏi các sản phẩm chấm giữa từ vector $\mathbf{v}_c$ của $w_c$ và vectơ nút không lá trên đường đi (đường dẫn in đậm trong :numref:`fig_hi_softmax`) từ gốc đến $w_3$, được đi qua trái, phải, sau đó trái: 

$$P(w_3 \mid w_c) = \sigma(\mathbf{u}_{n(w_3, 1)}^\top \mathbf{v}_c) \cdot \sigma(-\mathbf{u}_{n(w_3, 2)}^\top \mathbf{v}_c) \cdot \sigma(\mathbf{u}_{n(w_3, 3)}^\top \mathbf{v}_c).$$

Kể từ $\sigma(x)+\sigma(-x) = 1$, nó cho rằng xác suất có điều kiện của việc tạo ra tất cả các từ trong từ điển $\mathcal{V}$ dựa trên bất kỳ từ nào $w_c$ tổng hợp lên đến một: 

$$\sum_{w \in \mathcal{V}} P(w \mid w_c) = 1.$$
:eqlabel:`eq_hi-softmax-sum-one`

May mắn thay, vì $L(w_o)-1$ theo thứ tự $\mathcal{O}(\text{log}_2|\mathcal{V}|)$ do cấu trúc cây nhị phân, khi kích thước từ điển $\mathcal{V}$ rất lớn, chi phí tính toán cho mỗi bước đào tạo sử dụng softmax phân cấp được giảm đáng kể so với điều đó mà không cần đào tạo gần đúng. 

## Tóm tắt

* Lấy mẫu âm xây dựng hàm mất bằng cách xem xét các sự kiện độc lập lẫn nhau liên quan đến cả ví dụ tích cực và tiêu cực. Chi phí tính toán để đào tạo phụ thuộc tuyến tính vào số lượng từ tiếng ồn ở mỗi bước.
* Phân cấp softmax cấu tạo hàm mất bằng cách sử dụng đường dẫn từ nút gốc đến nút lá trong cây nhị phân. Chi phí tính toán cho đào tạo phụ thuộc vào logarit của kích thước từ điển ở mỗi bước.

## Bài tập

1. Làm thế nào chúng ta có thể lấy mẫu từ tiếng ồn trong lấy mẫu tiêu cực?
1. Xác minh rằng :eqref:`eq_hi-softmax-sum-one` nắm giữ.
1. Làm thế nào để đào tạo mô hình túi từ liên tục bằng cách sử dụng lấy mẫu âm và softmax phân cấp, tương ứng?

[Discussions](https://discuss.d2l.ai/t/382)

# Word Nhúng với Vectơ toàn cầu (Glove)
:label:`sec_glove`

Các trường hợp đồng từ trong các cửa sổ ngữ cảnh có thể mang thông tin ngữ nghĩa phong phú. Ví dụ, trong một từ corpus lớn “rắn” có nhiều khả năng đồng xảy ra với “băng” hơn là “hơi nước”, nhưng từ “khí” có lẽ đồng xảy ra với “hơi nước” thường xuyên hơn “băng”. Bên cạnh đó, số liệu thống kê cơ thể toàn cầu của các lần đồng xuất hiện như vậy có thể được tính toán trước: điều này có thể dẫn đến đào tạo hiệu quả hơn. Để tận dụng thông tin thống kê trong toàn bộ corpus để nhúng từ, trước tiên chúng ta hãy xem lại mô hình skip-gram trong :numref:`subsec_skip-gram`, nhưng giải thích nó bằng cách sử dụng số liệu thống kê cơ thể toàn cầu như số lượng đồng xuất hiện. 

## Skip-Gram với Thống kê Corpus toàn cầu
:label:`subsec_skipgram-global`

Biểu thị bởi $q_{ij}$ xác suất có điều kiện $P(w_j\mid w_i)$ của từ $w_j$ cho từ $w_i$ trong mô hình skip-gram, chúng tôi có 

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

trong đó cho bất kỳ chỉ số $i$ vectơ $\mathbf{v}_i$ và $\mathbf{u}_i$ đại diện cho từ $w_i$ là từ trung tâm và từ ngữ cảnh, tương ứng, và $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ là tập hợp chỉ mục của từ vựng. 

Xem xét từ $w_i$ có thể xảy ra nhiều lần trong corpus. Trong toàn bộ cơ thể, tất cả các từ ngữ cảnh bất cứ nơi nào $w_i$ được lấy làm từ trung tâm của chúng tạo thành một * multiset* $\mathcal{C}_i$ các chỉ số từ * cho phép nhiều trường hợp của cùng một phần tố*. Đối với bất kỳ phần tử nào, số phiên bản của nó được gọi là *multiplicity* của nó. Để minh họa với một ví dụ, giả sử rằng từ $w_i$ xảy ra hai lần trong corpus và chỉ số của các từ ngữ cảnh lấy $w_i$ làm từ trung tâm của chúng trong hai cửa sổ ngữ cảnh là $k, j, m, k$ và $k, l, k, j$. Do đó, multiset $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$, trong đó nhân của các yếu tố $j, k, l, m$ là 2, 4, 1, 1, tương ứng. 

Bây giờ chúng ta hãy biểu thị sự đa dạng của yếu tố $j$ trong multiset $\mathcal{C}_i$ như $x_{ij}$. Đây là số lượng đồng xuất hiện toàn cầu của từ $w_j$ (như từ ngữ cảnh) và từ $w_i$ (là từ trung tâm) trong cùng một cửa sổ ngữ cảnh trong toàn bộ corpus. Sử dụng số liệu thống kê cơ thể toàn cầu như vậy, chức năng mất mát của mô hình skip-gram tương đương với 

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-x_ij`

Chúng tôi tiếp tục biểu thị bằng $x_i$ số lượng của tất cả các từ ngữ cảnh trong các cửa sổ ngữ cảnh nơi $w_i$ xảy ra như là từ trung tâm của chúng, tương đương với $|\mathcal{C}_i|$. Cho phép $p_{ij}$ là xác suất có điều kiện $x_{ij}/x_i$ để tạo từ ngữ cảnh $w_j$ cho từ trung tâm $w_i$, :eqref:`eq_skipgram-x_ij` có thể được viết lại như 

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$
:eqlabel:`eq_skipgram-p_ij`

Năm :eqref:`eq_skipgram-p_ij`, $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ tính toán ngẫu nhiên chéo của phân phối có điều kiện $p_{ij}$ số liệu thống kê thể toàn cầu và phân phối có điều kiện $q_{ij}$ dự đoán mô hình. Sự mất mát này cũng được trọng số bởi $x_i$ như đã giải thích ở trên. Giảm thiểu hàm mất trong :eqref:`eq_skipgram-p_ij` sẽ cho phép phân phối có điều kiện dự đoán đến gần với phân phối có điều kiện từ số liệu thống kê corpus toàn cầu. 

Mặc dù thường được sử dụng để đo khoảng cách giữa các phân phối xác suất, chức năng mất ngẫu nhiên chéo có thể không phải là một lựa chọn tốt ở đây. Một mặt, như chúng tôi đã đề cập trong :numref:`sec_approx_train`, chi phí bình thường hóa đúng $q_{ij}$ dẫn đến tổng trên toàn bộ từ vựng, có thể tốn kém về mặt tính toán. Mặt khác, một số lượng lớn các sự kiện hiếm hoi từ một thể lớn thường được mô hình hóa bởi sự mất mát chéo entropy được chỉ định với quá nhiều trọng lượng. 

## Mô hình Glove

Theo quan điểm này, mô hình *GloVe* thực hiện ba thay đổi đối với mô hình skip-gram dựa trên tổn thất bình phương :cite:`Pennington.Socher.Manning.2014`: 

1. Sử dụng các biến $p'_{ij}=x_{ij}$ và $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ 
đó không phải là phân phối xác suất và lấy logarit của cả hai, vì vậy thuật ngữ tổn thất bình phương là $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Thêm hai tham số mô hình vô hướng cho mỗi từ $w_i$: thiên vị từ trung tâm $b_i$ và thiên vị từ ngữ cảnh $c_i$.
3. Thay thế trọng lượng của mỗi thời hạn giảm với chức năng trọng lượng $h(x_{ij})$, trong đó $h(x)$ đang tăng trong khoảng thời gian $[0, 1]$.

Kết hợp tất cả mọi thứ lại với nhau, đào tạo Glove là để giảm thiểu các chức năng mất mát sau: 

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$
:eqlabel:`eq_glove-loss`

Đối với chức năng trọng lượng, một lựa chọn được đề xuất là: $h(x) = (x/c) ^\alpha$ (ví dụ $\alpha = 0.75$) nếu $x < c$ (ví dụ, $c = 100$); nếu không thì $h(x) = 1$. Trong trường hợp này, vì $h(0)=0$, thời hạn tổn thất bình phương cho bất kỳ $x_{ij}=0$ nào có thể được bỏ qua cho hiệu quả tính toán. Ví dụ: khi sử dụng minibatch stochastic gradient descent để đào tạo, tại mỗi lần lặp lại, chúng tôi lấy mẫu ngẫu nhiên một minibatch của * non-zero* $x_{ij}$ để tính toán gradient và cập nhật các tham số mô hình. Lưu ý rằng những $x_{ij}$ không phải bằng không này là số liệu thống kê cơ thể toàn cầu được tính toán trước; do đó, mô hình được gọi là Glove cho *Global Vectors*. 

Cần nhấn mạnh rằng nếu từ $w_i$ xuất hiện trong cửa sổ ngữ cảnh của từ $w_j$, thì *vice versa*. Do đó, $x_{ij}=x_{ji}$. Không giống như word2vec phù hợp với xác suất điều kiện bất đối xứng $p_{ij}$, Glove phù hợp với đối xứng $\log \, x_{ij}$. Do đó, vector từ trung tâm và vector từ ngữ cảnh của bất kỳ từ nào tương đương về mặt toán học trong mô hình Glove. Tuy nhiên trong thực tế, do các giá trị khởi tạo khác nhau, cùng một từ vẫn có thể nhận được các giá trị khác nhau trong hai vectơ này sau khi đào tạo: Glove tổng chúng làm vectơ đầu ra. 

## Giải thích găng tay từ Tỷ lệ xác suất đồng xuất hiện

Chúng ta cũng có thể diễn giải mô hình Glove từ một góc độ khác. Sử dụng ký hiệu tương tự trong :numref:`subsec_skipgram-global`, hãy để $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$ là xác suất có điều kiện tạo ra từ ngữ cảnh $w_j$ cho $w_i$ làm từ trung tâm trong corpus. :numref:`tab_glove` liệt kê một số xác suất đồng xuất hiện cho từ “băng” và “hơi nước” và tỷ lệ của chúng dựa trên thống kê từ một thể lớn. 

:Word-word co-occurrence probabilities and their ratios from a large corpus (adapted from Table 1 in :cite:`Pennington.Socher.Manning.2014`:) 

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove`

Chúng ta có thể quan sát những điều sau đây từ :numref:`tab_glove`: 

* Đối với một từ $w_k$ có liên quan đến “băng” nhưng không liên quan đến “hơi nước”, chẳng hạn như $w_k=\text{solid}$, chúng tôi mong đợi tỷ lệ xác suất đồng xảy ra lớn hơn, chẳng hạn như 8.9.
* Đối với một từ $w_k$ có liên quan đến “hơi nước” nhưng không liên quan đến “băng”, chẳng hạn như $w_k=\text{gas}$, chúng tôi mong đợi tỷ lệ xác suất đồng xảy ra nhỏ hơn, chẳng hạn như 0,085.
* Đối với một từ $w_k$ có liên quan đến cả “băng” và “hơi nước”, chẳng hạn như $w_k=\text{water}$, chúng tôi mong đợi một tỷ lệ xác suất đồng chiếm gần 1, chẳng hạn như 1.36.
* Đối với một từ $w_k$ không liên quan đến cả “băng” và “hơi nước”, chẳng hạn như $w_k=\text{fashion}$, chúng tôi mong đợi một tỷ lệ xác suất đồng chiếm gần 1, chẳng hạn như 0,96.

Có thể thấy rằng tỷ lệ xác suất đồng xuất hiện có thể thể hiện trực giác mối quan hệ giữa các từ. Do đó, chúng ta có thể thiết kế một hàm của ba vectơ từ để phù hợp với tỷ lệ này. Đối với tỷ lệ xác suất đồng xuất hiện ${p_{ij}}/{p_{ik}}$ với $w_i$ là từ trung tâm và $w_j$ và $w_k$ là các từ ngữ cảnh, chúng tôi muốn phù hợp với tỷ lệ này bằng cách sử dụng một số hàm $f$: 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$
:eqlabel:`eq_glove-f`

Trong số nhiều thiết kế có thể có cho $f$, chúng tôi chỉ chọn một sự lựa chọn hợp lý trong những điều sau đây. Vì tỷ lệ xác suất đồng xuất hiện là vô hướng, chúng tôi yêu cầu $f$ là một hàm vô hướng, chẳng hạn như $f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$. Chuyển đổi các chỉ số từ $j$ và $k$ trong :eqref:`eq_glove-f`, nó phải giữ đó $f(x)f(-x)=1$, vì vậy một khả năng là $f(x)=\exp(x)$, tức là,  

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Bây giờ chúng ta hãy chọn $\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$, trong đó $\alpha$ là một hằng số. Kể từ $p_{ij}=x_{ij}/x_i$, sau khi lấy logarit ở cả hai bên, chúng tôi nhận được $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$. Chúng tôi có thể sử dụng các thuật ngữ thiên vị bổ sung để phù hợp với $- \log\, \alpha + \log\, x_i$, chẳng hạn như thiên vị từ trung tâm $b_i$ và thiên vị từ ngữ cảnh $c_j$: 

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$
:eqlabel:`eq_glove-square`

Đo sai số bình phương :eqref:`eq_glove-square` với trọng lượng, chức năng mất Glove trong :eqref:`eq_glove-loss` thu được. 

## Tóm tắt

* Mô hình skip-gram có thể được giải thích bằng cách sử dụng thống kê corpus toàn cầu như số lượng đồng xuất hiện từ.
* Sự mất mát chéo entropy có thể không phải là một lựa chọn tốt để đo sự khác biệt của hai phân phối xác suất, đặc biệt là đối với một thể lớn. Găng tay sử dụng tổn thất bình phương để phù hợp với số liệu thống kê cơ thể toàn cầu được tính toán trước.
* Vector từ trung tâm và vector từ ngữ cảnh tương đương về mặt toán học cho bất kỳ từ nào trong Glove.
* Găng tay có thể được giải thích từ tỷ lệ xác suất đồng xuất hiện từ.

## Bài tập

1. Nếu các từ $w_i$ và $w_j$ đồng xảy ra trong cùng một cửa sổ ngữ cảnh, làm thế nào chúng ta có thể sử dụng khoảng cách của chúng trong chuỗi văn bản để thiết kế lại phương pháp để tính xác suất có điều kiện $p_{ij}$? Hint: see Section 4.2 of the GloVe paper :cite:`Pennington.Socher.Manning.2014`.
1. Đối với bất kỳ từ nào, thiên vị từ trung tâm của nó và thiên vị từ ngữ cảnh tương đương về mặt toán học trong Glove không? Tại sao?

[Discussions](https://discuss.d2l.ai/t/385)

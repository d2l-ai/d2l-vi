# Tìm kiếm chùm
:label:`sec_beam-search`

Năm :numref:`sec_seq2seq`, chúng tôi dự đoán mã thông báo chuỗi đầu ra bằng token cho đến khi token "<eos>" end-of-sequence đặc biệt được dự đoán. Trong phần này, chúng ta sẽ bắt đầu bằng cách chính thức hóa chiến lược tìm kiếm* tham lam này* và khám phá các vấn đề với nó, sau đó so sánh chiến lược này với các lựa chọn thay thế khác:
*tìm kiếm đầy đủ* và * chùm tìm kiếm*.

Trước khi giới thiệu chính thức về tìm kiếm tham lam, chúng ta hãy chính thức hóa bài toán tìm kiếm bằng cách sử dụng cùng một ký hiệu toán học từ :numref:`sec_seq2seq`. Bất cứ lúc nào bước $t'$, xác suất của đầu ra bộ giải mã $y_{t'}$ có điều kiện trên dãy thứ tự đầu ra $y_1, \ldots, y_{t'-1}$ trước $t'$ và biến ngữ cảnh $\mathbf{c}$ mã hóa thông tin của chuỗi đầu vào. Để định lượng chi phí tính toán, biểu thị bằng $\mathcal{Y}$ (nó chứa "<eos>“) từ vựng đầu ra. Vì vậy, tính cardinality $\left|\mathcal{Y}\right|$ của bộ từ vựng này là kích thước từ vựng. Chúng ta cũng chỉ định số lượng thẻ tối đa của một chuỗi đầu ra là $T'$. Kết quả là, mục tiêu của chúng tôi là tìm kiếm một đầu ra lý tưởng từ tất cả các chuỗi đầu ra có thể có $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$. Tất nhiên, đối với tất cả các chuỗi đầu ra này, các phần bao gồm và sau <eos>"" sẽ bị loại bỏ trong đầu ra thực tế. 

## Tìm kiếm tham lam

Đầu tiên, chúng ta hãy xem xét một chiến lược đơn giản: * tìm kiếm tham lam*. Chiến lược này đã được sử dụng để dự đoán các trình tự trong :numref:`sec_seq2seq`. Trong tìm kiếm tham lam, bất cứ lúc nào bước $t'$ của chuỗi đầu ra, chúng tôi tìm kiếm mã thông báo có xác suất có điều kiện cao nhất từ $\mathcal{Y}$, tức là,  

$$y_{t'} = \operatorname*{argmax}_{y \in \mathcal{Y}} P(y \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$

như đầu ra. Khi "<eos>" được xuất ra hoặc chuỗi đầu ra đã đạt đến độ dài tối đa $T'$, chuỗi đầu ra được hoàn thành. 

Vì vậy, những gì có thể đi sai với tìm kiếm tham lam? Trên thực tế, trình tự tối ưu * phải là chuỗi đầu ra với $\prod_{t'=1}^{T'} P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c})$ tối đa, là xác suất có điều kiện tạo ra một chuỗi đầu ra dựa trên chuỗi đầu vào. Thật không may, không có gì đảm bảo rằng trình tự tối ưu sẽ thu được bằng cách tìm kiếm tham lam. 

![At each time step, greedy search selects the token with the highest conditional probability.](../img/s2s-prob1.svg)
:label:`fig_s2s-prob1`

Hãy để chúng tôi minh họa nó với một ví dụ. Giả sử có bốn mã thông báo “A”, “B”, “C”, và "<eos>" trong từ điển đầu ra. Năm :numref:`fig_s2s-prob1`, bốn số dưới mỗi bước thời gian đại diện cho xác suất có điều kiện tạo ra “A”, “B”, “C”, và "<eos>" tại bước thời điểm đó, tương ứng. Tại mỗi bước thời gian, tìm kiếm tham lam chọn mã thông báo có xác suất có điều kiện cao nhất. Do đó, chuỗi đầu ra “A”, “B”, “C”, và "<eos>" sẽ được dự đoán vào năm :numref:`fig_s2s-prob1`. Xác suất có điều kiện của chuỗi đầu ra này là $0.5\times0.4\times0.4\times0.6 = 0.048$. 

![The four numbers under each time step represent the conditional probabilities of generating "A", "B", "C", and "&lt;eos&gt;" at that time step.  At time step 2, the token "C", which has the second highest conditional probability, is selected.](../img/s2s-prob2.svg)
:label:`fig_s2s-prob2`

Tiếp theo, chúng ta hãy xem xét một ví dụ khác trong :numref:`fig_s2s-prob2`. Không giống như trong :numref:`fig_s2s-prob1`, tại thời điểm bước 2 chúng tôi chọn token “C” trong :numref:`fig_s2s-prob2`, có xác suất điều kiện cao nhất * giây*. Vì các dãy con đầu ra tại thời điểm bước 1 và 2, trên đó bước thời gian 3 dựa trên, đã thay đổi từ “A” và “B” trong :numref:`fig_s2s-prob1` thành “A” và “C” trong :numref:`fig_s2s-prob2`, xác suất có điều kiện của mỗi mã thông báo tại thời điểm bước 3 cũng đã thay đổi trong :numref:`fig_s2s-prob2`. Giả sử rằng chúng ta chọn token “B” tại bước thời điểm 3. Bây giờ bước thời gian 4 là điều kiện trên dãy con đầu ra ở ba bước thời gian đầu tiên “A”, “C”, và “B”, khác với “A”, “B”, và “C” trong :numref:`fig_s2s-prob1`. Do đó, xác suất có điều kiện tạo ra từng token tại bước thời điểm 4 trong :numref:`fig_s2s-prob2` cũng khác với xác suất có điều kiện trong :numref:`fig_s2s-prob1`. Kết quả là xác suất có điều kiện của chuỗi đầu ra “A”, “C”, “B” và "<eos>" trong :numref:`fig_s2s-prob2` là $0.5\times0.3 \times0.6\times0.6=0.054$, lớn hơn so với tìm kiếm tham lam vào năm :numref:`fig_s2s-prob1`. Trong ví dụ này, chuỗi đầu ra “A”, “B”, “C”, và "<eos>" thu được bằng cách tìm kiếm tham lam không phải là một chuỗi tối ưu. 

## Tìm kiếm đầy đủ

Nếu mục tiêu là để có được trình tự tối ưu, chúng tôi có thể xem xét sử dụng * tìm kiếm đầy đủ*: liệt kê toàn bộ tất cả các chuỗi đầu ra có thể có với xác suất có điều kiện của chúng, sau đó xuất ra một trong những xác suất có điều kiện cao nhất. 

Mặc dù chúng ta có thể sử dụng tìm kiếm đầy đủ để có được trình tự tối ưu, nhưng chi phí tính toán của nó $\mathcal{O}(\left|\mathcal{Y}\right|^{T'})$ có thể là quá cao. Ví dụ: khi $|\mathcal{Y}|=10000$ và $T'=10$, chúng ta sẽ cần đánh giá trình tự $10000^{10} = 10^{40}$. Đây là bên cạnh không thể! Mặt khác, chi phí tính toán của tìm kiếm tham lam là $\mathcal{O}(\left|\mathcal{Y}\right|T')$: nó thường nhỏ hơn đáng kể so với tìm kiếm đầy đủ. Ví dụ, khi $|\mathcal{Y}|=10000$ và $T'=10$, chúng ta chỉ cần đánh giá $10000\times10=10^5$ trình tự. 

## Tìm kiếm chùm

Các quyết định về chiến lược tìm kiếm trình tự nằm trên một phổ, với những câu hỏi dễ dàng ở một trong hai cực đoan. Điều gì sẽ xảy ra nếu chỉ chính xác quan trọng? Rõ ràng, tìm kiếm đầy đủ. Điều gì sẽ xảy ra nếu chỉ có chi phí tính toán quan trọng? Rõ ràng, tìm kiếm tham lam. Một ứng dụng trong thế giới thực thường đặt ra một câu hỏi phức tạp, ở đâu đó giữa hai thái cực đó. 

*Chùm tìm kiếm* là một phiên bản cải tiến của tìm kiếm tham lam. Nó có một siêu tham số có tên là kích thước chùm tia *, $k$. 
Vào thời điểm bước 1, chúng tôi chọn $k$ token có xác suất có điều kiện cao nhất. Mỗi người trong số họ sẽ là mã thông báo đầu tiên của $k$ trình tự đầu ra ứng cử viên, tương ứng. Tại mỗi bước thời gian tiếp theo, dựa trên trình tự đầu ra ứng cử viên $k$ ở bước thời gian trước, chúng tôi tiếp tục chọn trình tự đầu ra ứng cử viên $k$ với xác suất có điều kiện cao nhất từ $k\left|\mathcal{Y}\right|$ các lựa chọn có thể. 

![The process of beam search (beam size: 2, maximum length of an output sequence: 3). The candidate output sequences are $A$, $C$, $AB$, $CE$, $ABD$, and $CED$.](../img/beam-search.svg)
:label:`fig_beam-search`

:numref:`fig_beam-search` thể hiện quá trình tìm kiếm chùm tia với một ví dụ. Giả sử rằng từ vựng đầu ra chỉ chứa năm phần tử: $\mathcal{Y} = \{A, B, C, D, E\}$, trong đó một trong số chúng là “<eos>”. Hãy để kích thước chùm là 2 và chiều dài tối đa của một chuỗi đầu ra là 3. Vào thời điểm bước 1, giả sử rằng các mã thông báo có xác suất có điều kiện cao nhất $P(y_1 \mid \mathbf{c})$ là $A$ và $C$. Tại thời điểm bước 2, cho tất cả $y_2 \in \mathcal{Y},$ chúng tôi tính toán  

$$\begin{aligned}P(A, y_2 \mid \mathbf{c}) = P(A \mid \mathbf{c})P(y_2 \mid A, \mathbf{c}),\\ P(C, y_2 \mid \mathbf{c}) = P(C \mid \mathbf{c})P(y_2 \mid C, \mathbf{c}),\end{aligned}$$  

và chọn hai giá trị lớn nhất trong số mười giá trị này, nói $P(A, B \mid \mathbf{c})$ và $P(C, E \mid \mathbf{c})$. Sau đó, tại thời điểm bước 3, cho tất cả $y_3 \in \mathcal{Y}$, chúng tôi tính  

$$\begin{aligned}P(A, B, y_3 \mid \mathbf{c}) = P(A, B \mid \mathbf{c})P(y_3 \mid A, B, \mathbf{c}),\\P(C, E, y_3 \mid \mathbf{c}) = P(C, E \mid \mathbf{c})P(y_3 \mid C, E, \mathbf{c}),\end{aligned}$$ 

và chọn hai lớn nhất trong số mười giá trị này, nói $P(A, B, D \mid \mathbf{c})$ và $P(C, E, D \mid  \mathbf{c}).$ Kết quả là, chúng tôi nhận được sáu trình tự đầu ra ứng cử viên: (i) $A$; (ii) $C$; (iii) $A$, $A$, $A$, $A$, $A$ 3617, $D$; và (vi) $C$, $E$, $D$.  

Cuối cùng, chúng tôi có được tập hợp các chuỗi đầu ra ứng viên cuối cùng dựa trên sáu chuỗi này (ví dụ, loại bỏ các phần bao gồm và sau “<eos>”). Sau đó, chúng ta chọn dãy có điểm cao nhất sau làm chuỗi đầu ra: 

$$ \frac{1}{L^\alpha} \log P(y_1, \ldots, y_{L}\mid \mathbf{c}) = \frac{1}{L^\alpha} \sum_{t'=1}^L \log P(y_{t'} \mid y_1, \ldots, y_{t'-1}, \mathbf{c}),$$
:eqlabel:`eq_beam-search-score`

trong đó $L$ là độ dài của trình tự ứng cử viên cuối cùng và $\alpha$ thường được đặt thành 0,75. Vì một chuỗi dài hơn có nhiều thuật ngữ logarit hơn trong tổng kết :eqref:`eq_beam-search-score`, thuật ngữ $L^\alpha$ trong mẫu số phạt các chuỗi dài. 

Chi phí tính toán của tìm kiếm chùm tia là $\mathcal{O}(k\left|\mathcal{Y}\right|T')$. Kết quả này nằm giữa tìm kiếm tham lam và tìm kiếm đầy đủ. Trên thực tế, tìm kiếm tham lam có thể được coi là một loại tìm kiếm chùm đặc biệt với kích thước chùm tia là 1. Với sự lựa chọn linh hoạt về kích thước chùm tia, tìm kiếm chùm tia cung cấp sự cân bằng giữa độ chính xác so với chi phí tính toán. 

## Tóm tắt

* Các chiến lược tìm kiếm trình tự bao gồm tìm kiếm tham lam, tìm kiếm đầy đủ và tìm kiếm chùm tia.
* Tìm kiếm chùm tia cung cấp sự cân bằng giữa độ chính xác so với chi phí tính toán thông qua sự lựa chọn linh hoạt của kích thước chùm tia.

## Bài tập

1. Chúng ta có thể coi tìm kiếm toàn diện như một loại tìm kiếm chùm đặc biệt không? Tại sao hoặc tại sao không?
1. Áp dụng tìm kiếm chùm tia trong vấn đề dịch máy trong :numref:`sec_seq2seq`. Kích thước chùm tia ảnh hưởng đến kết quả dịch thuật và tốc độ dự đoán như thế nào?
1. Chúng tôi đã sử dụng mô hình hóa ngôn ngữ để tạo văn bản sau tiền tố do người dùng cung cấp trong :numref:`sec_rnn_scratch`. Nó sử dụng loại chiến lược tìm kiếm nào? Bạn có thể cải thiện nó?

[Discussions](https://discuss.d2l.ai/t/338)

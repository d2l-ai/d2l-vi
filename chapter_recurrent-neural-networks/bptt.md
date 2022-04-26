# Backpropagation qua thời gian
:label:`sec_bptt`

Cho đến nay chúng tôi đã nhiều lần ám chỉ những thứ như
*bùng nổ độ dạng*,
*biến mất độ dạng*,
và sự cần thiết phải
*tách gradient* cho RNNs.
Ví dụ, trong :numref:`sec_rnn_scratch`, chúng tôi đã gọi hàm `detach` trên trình tự. Không ai trong số này thực sự được giải thích đầy đủ, vì lợi ích của việc có thể xây dựng một mô hình một cách nhanh chóng và xem nó hoạt động như thế nào. Trong phần này, chúng ta sẽ nghiên cứu sâu hơn một chút về các chi tiết về truyền ngược cho các mô hình trình tự và tại sao (và cách thức) toán học hoạt động. 

Chúng tôi gặp phải một số hiệu ứng của vụ nổ gradient khi lần đầu tiên chúng tôi triển khai RNN s (:numref:`sec_rnn_scratch`). Đặc biệt, nếu bạn giải quyết các bài tập, bạn sẽ thấy rằng cắt gradient là rất quan trọng để đảm bảo sự hội tụ thích hợp. Để hiểu rõ hơn về vấn đề này, phần này sẽ xem xét cách tính toán độ dốc cho các mô hình trình tự. Lưu ý rằng không có gì về mặt khái niệm mới trong cách thức hoạt động của nó. Rốt cuộc, chúng ta vẫn chỉ đơn thuần áp dụng quy tắc chuỗi để tính toán độ dốc. Tuy nhiên, nó là giá trị trong khi xem xét backpropagation (:numref:`sec_backprop`) một lần nữa. 

Chúng tôi đã mô tả về phía trước và lạc hậu tuyên truyền và đồ thị tính toán trong MLP s trong :numref:`sec_backprop`. Chuyển tiếp tuyên truyền trong một RNN là tương đối đơn giản.
*Backpropagation thông qua thời giới* thực sự là một cụ thể
ứng dụng của backpropagation trong RNNs :cite:`Werbos.1990`. Nó đòi hỏi chúng ta phải mở rộng biểu đồ tính toán của một RNN một bước một lần tại một thời điểm để có được các phụ thuộc giữa các biến mô hình và tham số. Sau đó, dựa trên quy tắc chuỗi, chúng tôi áp dụng backpropagation để tính toán và lưu trữ gradient. Vì các trình tự có thể khá dài, sự phụ thuộc có thể khá dài. Ví dụ, đối với một chuỗi 1000 ký tự, token đầu tiên có khả năng có thể có ảnh hưởng đáng kể đến mã thông báo ở vị trí cuối cùng. Điều này không thực sự khả thi về mặt tính toán (mất quá nhiều thời gian và đòi hỏi quá nhiều bộ nhớ) và nó đòi hỏi hơn 1000 sản phẩm ma trận trước khi chúng tôi sẽ đến gradient rất khó nắm bắt đó. Đây là một quá trình đầy sự không chắc chắn tính toán và thống kê. Trong phần sau đây, chúng tôi sẽ làm sáng tỏ những gì xảy ra và cách giải quyết vấn đề này trong thực tế. 

## Phân tích Gradient trong RNNs
:label:`subsec_bptt_analysis`

Chúng tôi bắt đầu với một mô hình đơn giản hóa về cách thức hoạt động của một RNN. Mô hình này bỏ qua chi tiết về các chi tiết cụ thể của trạng thái ẩn và cách nó được cập nhật. Ký hiệu toán học ở đây không phân biệt rõ ràng vô hướng, vectơ, và ma trận như trước đây. Những chi tiết này không quan trọng đối với phân tích và sẽ chỉ phục vụ để làm lộn xộn ký hiệu trong tiểu mục này. 

Trong mô hình đơn giản hóa này, chúng tôi biểu thị $h_t$ là trạng thái ẩn, $x_t$ làm đầu vào và $o_t$ là đầu ra tại bước thời gian $t$. Nhớ lại các cuộc thảo luận của chúng tôi trong :numref:`subsec_rnn_w_hidden_states` rằng đầu vào và trạng thái ẩn có thể được nối để được nhân với một biến trọng lượng trong lớp ẩn. Do đó, chúng tôi sử dụng $w_h$ và $w_o$ để chỉ ra trọng lượng của lớp ẩn và lớp đầu ra, tương ứng. Kết quả là, các trạng thái ẩn và đầu ra tại mỗi thời điểm các bước có thể được giải thích là 

$$\begin{aligned}h_t &= f(x_t, h_{t-1}, w_h),\\o_t &= g(h_t, w_o),\end{aligned}$$
:eqlabel:`eq_bptt_ht_ot`

trong đó $f$ và $g$ là sự biến đổi của lớp ẩn và lớp đầu ra, tương ứng. Do đó, chúng ta có một chuỗi các giá trị $\{\ldots, (x_{t-1}, h_{t-1}, o_{t-1}), (x_{t}, h_{t}, o_t), \ldots\}$ phụ thuộc vào nhau thông qua tính toán lặp lại. Việc tuyên truyền về phía trước là khá đơn giản. Tất cả những gì chúng ta cần là vòng lặp qua $(x_t, h_t, o_t)$ ba lần một bước tại một thời điểm. Sự khác biệt giữa đầu ra $o_t$ và nhãn mong muốn $y_t$ sau đó được đánh giá bởi một chức năng khách quan trên tất cả các bước thời gian $T$ như 

$$L(x_1, \ldots, x_T, y_1, \ldots, y_T, w_h, w_o) = \frac{1}{T}\sum_{t=1}^T l(y_t, o_t).$$

Đối với truyền ngược, các vấn đề phức tạp hơn một chút, đặc biệt là khi chúng ta tính toán các gradient liên quan đến các tham số $w_h$ của hàm khách quan $L$. Để được cụ thể, theo quy tắc chuỗi, 

$$\begin{aligned}\frac{\partial L}{\partial w_h}  & = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial w_h}  \\& = \frac{1}{T}\sum_{t=1}^T \frac{\partial l(y_t, o_t)}{\partial o_t} \frac{\partial g(h_t, w_o)}{\partial h_t}  \frac{\partial h_t}{\partial w_h}.\end{aligned}$$
:eqlabel:`eq_bptt_partial_L_wh`

Yếu tố thứ nhất và thứ hai của sản phẩm trong :eqref:`eq_bptt_partial_L_wh` rất dễ tính toán. Yếu tố thứ ba $\partial h_t/\partial w_h$ là nơi mọi thứ trở nên khó khăn, vì chúng ta cần tính toán lại hiệu ứng của tham số $w_h$ trên $h_t$. Theo tính toán tái phát vào năm :eqref:`eq_bptt_ht_ot`, $h_t$ phụ thuộc vào cả $h_{t-1}$ và $w_h$, trong đó tính toán $h_{t-1}$ cũng phụ thuộc vào $w_h$. Do đó, sử dụng sản lượng quy tắc chuỗi 

$$\frac{\partial h_t}{\partial w_h}= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_recur`

Để lấy được gradient trên, giả sử rằng chúng ta có ba chuỗi $\{a_{t}\},\{b_{t}\},\{c_{t}\}$ đáp ứng $a_{0}=0$ và $a_{t}=b_{t}+c_{t}a_{t-1}$ cho $t=1, 2,\ldots$. Sau đó, đối với $t\geq 1$, thật dễ dàng để hiển thị 

$$a_{t}=b_{t}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t}c_{j}\right)b_{i}.$$
:eqlabel:`eq_bptt_at`

Bằng cách thay thế $a_t$, $b_t$, và $c_t$ theo 

$$\begin{aligned}a_t &= \frac{\partial h_t}{\partial w_h},\\
b_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}, \\
c_t &= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}},\end{aligned}$$

tính toán gradient trong :eqref:`eq_bptt_partial_ht_wh_recur` thỏa mãn $a_{t}=b_{t}+c_{t}a_{t-1}$. Do đó, mỗi :eqref:`eq_bptt_at`, chúng ta có thể loại bỏ tính toán tái phát trong :eqref:`eq_bptt_partial_ht_wh_recur` với 

$$\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^{t} \frac{\partial f(x_{j},h_{j-1},w_h)}{\partial h_{j-1}} \right) \frac{\partial f(x_{i},h_{i-1},w_h)}{\partial w_h}.$$
:eqlabel:`eq_bptt_partial_ht_wh_gen`

Mặc dù chúng ta có thể sử dụng quy tắc chuỗi để tính toán $\partial h_t/\partial w_h$ đệ quy, chuỗi này có thể nhận được rất dài bất cứ khi nào $t$ lớn. Hãy để chúng tôi thảo luận về một số chiến lược để đối phó với vấn đề này. 

### Full tính toán ### 

Rõ ràng, chúng ta chỉ có thể tính toán toàn bộ tổng trong :eqref:`eq_bptt_partial_ht_wh_gen`. Tuy nhiên, điều này rất chậm và độ dốc có thể nổ tung, vì những thay đổi tinh tế trong điều kiện ban đầu có khả năng ảnh hưởng đến kết quả rất nhiều. Đó là, chúng ta có thể thấy những thứ tương tự như hiệu ứng butterfly trong đó những thay đổi tối thiểu trong điều kiện ban đầu dẫn đến những thay đổi không cân xứng trong kết quả. Điều này thực sự là khá không mong muốn về mô hình mà chúng tôi muốn ước tính. Rốt cuộc, chúng tôi đang tìm kiếm những người dự đoán mạnh mẽ mà khái quát hóa tốt. Do đó chiến lược này hầu như không bao giờ được sử dụng trong thực tế. 

### Cắt ngắn thời gian bước ###

Ngoài ra, chúng ta có thể cắt ngắn tổng trong :eqref:`eq_bptt_partial_ht_wh_gen` sau $\tau$ bước. Đây là những gì chúng ta đã thảo luận cho đến nay, chẳng hạn như khi chúng ta tách các gradient trong :numref:`sec_rnn_scratch`. Điều này dẫn đến một *xấp xỉ * của gradient thật, đơn giản bằng cách chấm dứt tổng tại $\partial h_{t-\tau}/\partial w_h$. Trong thực tế, điều này hoạt động khá tốt. Nó là những gì thường được gọi là backpropgation cắt ngắn qua thời gian :cite:`Jaeger.2002`. Một trong những hậu quả của việc này là mô hình tập trung chủ yếu vào ảnh hưởng ngắn hạn chứ không phải là hậu quả lâu dài. Điều này thực sự là * mong muốn*, vì nó thiên vị ước tính đối với các mô hình đơn giản và ổn định hơn. 

### Ngẫu nhiên Cắt ngắn ### 

Cuối cùng, chúng ta có thể thay thế $\partial h_t/\partial w_h$ bằng một biến ngẫu nhiên đó là chính xác trong kỳ vọng nhưng cắt ngắn chuỗi. Điều này đạt được bằng cách sử dụng một chuỗi $\xi_t$ với $0 \leq \pi_t \leq 1$ được xác định trước, trong đó $P(\xi_t = 0) = 1-\pi_t$ và $P(\xi_t = \pi_t^{-1}) = \pi_t$, do đó $E[\xi_t] = 1$. Chúng tôi sử dụng điều này để thay thế gradient $\partial h_t/\partial w_h$ trong :eqref:`eq_bptt_partial_ht_wh_recur` bằng 

$$z_t= \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial w_h} +\xi_t \frac{\partial f(x_{t},h_{t-1},w_h)}{\partial h_{t-1}} \frac{\partial h_{t-1}}{\partial w_h}.$$

Nó theo định nghĩa của $\xi_t$ rằng $E[z_t] = \partial h_t/\partial w_h$. Bất cứ khi nào $\xi_t = 0$ tính toán tái phát chấm dứt tại thời điểm đó bước $t$. Điều này dẫn đến một tổng trọng số của các chuỗi có độ dài khác nhau trong đó các chuỗi dài rất hiếm nhưng quá nặng một cách thích hợp. Ý tưởng này được đề xuất bởi Tallec và Ollivier :cite:`Tallec.Ollivier.2017`. 

### So sánh chiến lược

![Comparing strategies for computing gradients in RNNs. From top to bottom: randomized truncation, regular truncation, and full computation.](../img/truncated-bptt.svg)
:label:`fig_truncated_bptt`

:numref:`fig_truncated_bptt` minh họa ba chiến lược khi phân tích vài ký tự đầu tiên của *The Time Machine* cuốn sách sử dụng backpropagation qua thời gian cho RNNs: 

* Hàng đầu tiên là sự cắt ngắn ngẫu nhiên phân vùng văn bản thành các phân đoạn có độ dài khác nhau.
* Hàng thứ hai là sự cắt ngắn thông thường phá vỡ văn bản thành các dãy con có cùng độ dài. Đây là những gì chúng tôi đã làm trong các thí nghiệm RNN.
* Hàng thứ ba là sự lan truyền ngược đầy đủ thông qua thời gian dẫn đến một biểu thức không khả thi về mặt tính toán.

Thật không may, trong khi hấp dẫn về lý thuyết, cắt ngắn ngẫu nhiên không hoạt động tốt hơn nhiều so với cắt ngắn thông thường, rất có thể là do một số yếu tố. Đầu tiên, hiệu quả của một quan sát sau một số bước lan truyền ngược vào quá khứ là khá đủ để nắm bắt các phụ thuộc trong thực tế. Thứ hai, phương sai tăng chống lại thực tế là gradient chính xác hơn với nhiều bước hơn. Thứ ba, chúng tôi thực sự * muốn* mô hình chỉ có một phạm vi tương tác ngắn. Do đó, thường xuyên cắt ngắn backpropagation thông qua thời gian có một hiệu ứng thường xuyên nhẹ có thể được mong muốn. 

## Backpropagation qua thời gian chi tiết

Sau khi thảo luận về nguyên tắc chung, chúng ta hãy thảo luận về tuyên truyền ngược qua thời gian một cách chi tiết. Khác với phân tích trong :numref:`subsec_bptt_analysis`, sau đây chúng ta sẽ chỉ ra cách tính độ dốc của hàm mục tiêu đối với tất cả các tham số mô hình bị phân hủy. Để giữ cho mọi thứ đơn giản, chúng ta xem xét một RNN không có tham số thiên vị, có chức năng kích hoạt trong lớp ẩn sử dụng ánh xạ danh tính ($\phi(x)=x$). Đối với bước thời gian $t$, hãy để đầu vào ví dụ duy nhất và nhãn lần lượt là $\mathbf{x}_t \in \mathbb{R}^d$ và $y_t$. Trạng thái ẩn $\mathbf{h}_t \in \mathbb{R}^h$ và đầu ra $\mathbf{o}_t \in \mathbb{R}^q$ được tính là 

$$\begin{aligned}\mathbf{h}_t &= \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},\\
\mathbf{o}_t &= \mathbf{W}_{qh} \mathbf{h}_{t},\end{aligned}$$

trong đó $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$, $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$, và $\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$ là các thông số trọng lượng. Biểu thị bởi $l(\mathbf{o}_t, y_t)$ sự mất mát tại bước thời gian $t$. Chức năng mục tiêu của chúng tôi, sự mất mát trên $T$ bước thời gian từ đầu chuỗi là do đó 

$$L = \frac{1}{T} \sum_{t=1}^T l(\mathbf{o}_t, y_t).$$

Để hình dung các phụ thuộc giữa các biến mô hình và tham số trong quá trình tính toán RNN, chúng ta có thể vẽ một biểu đồ tính toán cho mô hình, như thể hiện trong :numref:`fig_rnn_bptt`. Ví dụ, tính toán các trạng thái ẩn của bước thời gian 3, $\mathbf{h}_3$, phụ thuộc vào các thông số mô hình $\mathbf{W}_{hx}$ và $\mathbf{W}_{hh}$, trạng thái ẩn của bước thời gian cuối cùng $\mathbf{h}_2$ và đầu vào của bước thời gian hiện tại $\mathbf{x}_3$. 

![Computational graph showing dependencies for an RNN model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators.](../img/rnn-bptt.svg)
:label:`fig_rnn_bptt`

Như vừa đề cập, các thông số mô hình trong :numref:`fig_rnn_bptt` là $\mathbf{W}_{hx}$, $\mathbf{W}_{hh}$ và $\mathbf{W}_{qh}$. Nói chung, đào tạo mô hình này yêu cầu tính toán gradient đối với các thông số này $\partial L/\partial \mathbf{W}_{hx}$, $\partial L/\partial \mathbf{W}_{hh}$ và $\partial L/\partial \mathbf{W}_{qh}$. Theo các phụ thuộc trong :numref:`fig_rnn_bptt`, chúng ta có thể đi qua theo hướng ngược lại của các mũi tên để tính toán và lưu trữ các gradient lần lượt. Để thể hiện linh hoạt phép nhân của ma trận, vectơ và vô hướng của các hình dạng khác nhau trong quy tắc chuỗi, chúng ta tiếp tục sử dụng toán tử $\text{prod}$ như được mô tả trong :numref:`sec_backprop`. 

Trước hết, việc phân biệt chức năng khách quan đối với đầu ra mô hình bất cứ lúc nào $t$ là khá đơn giản: 

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial l (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t} \in \mathbb{R}^q.$$
:eqlabel:`eq_bptt_partial_L_ot`

Bây giờ, chúng ta có thể tính toán gradient của hàm mục tiêu đối với tham số $\mathbf{W}_{qh}$ trong lớp đầu ra: $\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$. Dựa trên :numref:`fig_rnn_bptt`, hàm khách quan $L$ phụ thuộc vào $\mathbf{W}_{qh}$ qua $\mathbf{o}_1, \ldots, \mathbf{o}_T$. Sử dụng sản lượng quy tắc chuỗi 

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top,
$$

trong đó $\partial L/\partial \mathbf{o}_t$ được đưa ra bởi :eqref:`eq_bptt_partial_L_ot`. 

Tiếp theo, như thể hiện trong :numref:`fig_rnn_bptt`, tại bước thời gian cuối cùng $T$ các chức năng khách quan $L$ phụ thuộc vào trạng thái ẩn $\mathbf{h}_T$ chỉ qua $\mathbf{o}_T$. Do đó, chúng ta có thể dễ dàng tìm thấy gradient $\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$ bằng quy tắc chuỗi: 

$$\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.$$
:eqlabel:`eq_bptt_partial_L_hT_final_step`

Nó trở nên phức tạp hơn cho bất kỳ bước thời gian $t < T$, nơi chức năng khách quan $L$ phụ thuộc vào $\mathbf{h}_t$ qua $\mathbf{h}_{t+1}$ và $\mathbf{o}_t$. Theo quy tắc chuỗi, gradient của trạng thái ẩn $\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$ bất cứ lúc nào bước $t < T$ có thể được tính toán một cách lặp lại như: 

$$\frac{\partial L}{\partial \mathbf{h}_t} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right) + \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right) = \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.$$
:eqlabel:`eq_bptt_partial_L_ht_recur`

Để phân tích, mở rộng tính toán định kỳ cho bất kỳ bước thời gian $1 \leq t \leq T$ cho 

$$\frac{\partial L}{\partial \mathbf{h}_t}= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.$$
:eqlabel:`eq_bptt_partial_L_ht`

Chúng ta có thể thấy từ :eqref:`eq_bptt_partial_L_ht` rằng ví dụ tuyến tính đơn giản này đã thể hiện một số vấn đề chính của các mô hình chuỗi dài: nó liên quan đến sức mạnh có khả năng rất lớn của $\mathbf{W}_{hh}^\top$. Trong đó, eigenvalues nhỏ hơn 1 biến mất và eigenvalues lớn hơn 1 phân kỳ. Điều này không ổn định về số lượng, biểu hiện dưới dạng biến mất và bùng nổ gradient. Một cách để giải quyết vấn đề này là cắt ngắn các bước thời gian ở kích thước thuận tiện về mặt tính toán như đã thảo luận trong :numref:`subsec_bptt_analysis`. Trong thực tế, sự cắt ngắn này được thực hiện bằng cách tách gradient sau một số bước thời gian nhất định. Sau đó, chúng ta sẽ thấy các mô hình trình tự phức tạp hơn như bộ nhớ ngắn hạn dài có thể làm giảm bớt điều này hơn nữa.  

Cuối cùng, :numref:`fig_rnn_bptt` cho thấy hàm khách quan $L$ phụ thuộc vào các tham số mô hình $\mathbf{W}_{hx}$ và $\mathbf{W}_{hh}$ trong lớp ẩn thông qua các trạng thái ẩn $\mathbf{h}_1, \ldots, \mathbf{h}_T$. Để tính toán gradient liên quan đến các tham số như vậy $\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ và $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$, chúng tôi áp dụng quy tắc chuỗi cung cấp 

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top,
\end{aligned}
$$

trong đó $\partial L/\partial \mathbf{h}_t$ được tính toán lặp đi lặp lại bởi :eqref:`eq_bptt_partial_L_hT_final_step` và :eqref:`eq_bptt_partial_L_ht_recur` là đại lượng chính ảnh hưởng đến sự ổn định số. 

Kể từ khi tuyên truyền ngược qua thời gian là việc áp dụng tuyên truyền ngược trong RNN s, như chúng tôi đã giải thích trong :numref:`sec_backprop`, đào tạo RNN s xen kẽ tuyên truyền về phía trước với truyền ngược qua thời gian. Bên cạnh đó, backpropagation thông qua thời gian tính toán và lưu trữ các gradient trên lần lượt. Cụ thể, các giá trị trung gian được lưu trữ được tái sử dụng để tránh tính toán trùng lặp, chẳng hạn như lưu trữ $\partial L/\partial \mathbf{h}_t$ được sử dụng trong tính toán cả $\partial L / \partial \mathbf{W}_{hx}$ và $\partial L / \partial \mathbf{W}_{hh}$. 

## Tóm tắt

* Backpropagation qua thời gian chỉ đơn thuần là một ứng dụng của backpropagation cho các mô hình trình tự với một trạng thái ẩn.
* Cắt ngắn là cần thiết để thuận tiện tính toán và ổn định số, chẳng hạn như cắt ngắn thường xuyên và cắt ngắn ngẫu nhiên.
* Sức mạnh cao của ma trận có thể dẫn đến sự khác biệt hoặc biến mất eigenvalues. Điều này thể hiện chính nó dưới dạng bùng nổ hoặc biến mất gradient.
* Để tính toán hiệu quả, các giá trị trung gian được lưu trữ trong quá trình truyền ngược qua thời gian.

## Bài tập

1. Giả sử rằng chúng ta có một ma trận đối xứng $\mathbf{M} \in \mathbb{R}^{n \times n}$ với eigenvalues $\lambda_i$ có eigenvectors tương ứng là $\mathbf{v}_i$ ($i = 1, \ldots, n$). Không mất tính tổng quát, giả định rằng chúng được đặt hàng theo thứ tự $|\lambda_i| \geq |\lambda_{i+1}|$. 
   1. Cho thấy $\mathbf{M}^k$ có eigenvalues $\lambda_i^k$.
   1. Chứng minh rằng đối với một vector ngẫu nhiên $\mathbf{x} \in \mathbb{R}^n$, với xác suất cao $\mathbf{M}^k \mathbf{x}$ sẽ rất phù hợp với eigenvector $\mathbf{v}_1$ 
của $\mathbf{M}$. Chính thức hóa tuyên bố này.
   1. Kết quả trên có ý nghĩa gì đối với gradient trong RNNs?
1. Bên cạnh việc cắt gradient, bạn có thể nghĩ về bất kỳ phương pháp nào khác để đối phó với sự bùng nổ gradient trong các mạng thần kinh tái phát không?

[Discussions](https://discuss.d2l.ai/t/334)

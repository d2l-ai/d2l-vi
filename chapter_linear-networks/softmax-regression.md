# Hồi quy Softmax
:label:`sec_softmax`

Trong :numref:`sec_linear_regression`, chúng tôi giới thiệu hồi quy tuyến tính, làm việc thông qua việc triển khai từ đầu vào năm :numref:`sec_linear_scratch` và một lần nữa sử dụng API cấp cao của một khung học sâu trong :numref:`sec_linear_concise` để thực hiện nâng nặng. 

Hồi quy là cái búa chúng ta đạt được khi chúng ta muốn trả lời * bao nhiêu? * hoặc *bao nhiêu? * câu hỏi. Nếu bạn muốn dự đoán số đô la (giá cả) mà tại đó một ngôi nhà sẽ được bán, hoặc số chiến thắng một đội bóng chày có thể có, hoặc số ngày mà một bệnh nhân sẽ vẫn nhập viện trước khi được xuất viện, sau đó bạn có thể đang tìm kiếm một mô hình hồi quy. 

Trong thực tế, chúng tôi thường quan tâm đến *phân loại *: hỏi không phải “bao nhiêu” mà là “cái nào”: 

* Email này có thuộc về thư mục spam hay hộp thư đến không?
* Khách hàng này có nhiều khả năng * để đăng ký* hoặc * không đăng ký* cho một dịch vụ đăng ký?
* Hình ảnh này mô tả một con lừa, chó, một con mèo, hoặc một con gà trống?
* Aston có thể xem bộ phim nào nhất tiếp theo?

Thông thường, các học viên học máy làm quá tải từ * phân loại * để mô tả hai vấn đề tinh tế khác nhau: (i) những vấn đề mà chúng tôi chỉ quan tâm đến các bài tập cứng của các ví dụ cho các loại (lớp); và (ii) những vấn đề mà chúng tôi muốn thực hiện các bài tập mềm, tức là, để đánh giá xác suất mà mỗi loại áp dụng. Sự khác biệt có xu hướng bị mờ, một phần, bởi vì thường xuyên, ngay cả khi chúng ta chỉ quan tâm đến các bài tập cứng, chúng ta vẫn sử dụng các mô hình tạo ra các bài tập mềm. 

## Phân loại vấn đề
:label:`subsec_classification-problem`

Để có được bàn chân của chúng tôi ướt, chúng ta hãy bắt đầu với một vấn đề phân loại hình ảnh đơn giản. Ở đây, mỗi đầu vào bao gồm một hình ảnh thang màu xám $2\times2$. Chúng ta có thể đại diện cho mỗi giá trị pixel với một vô hướng duy nhất, cho chúng ta bốn tính năng $x_1, x_2, x_3, x_4$. Hơn nữa, chúng ta hãy giả định rằng mỗi hình ảnh thuộc về một trong số các loại “mèo”, “gà” và “chó”. 

Tiếp theo, chúng ta phải chọn cách đại diện cho các nhãn. Chúng tôi có hai lựa chọn rõ ràng. Có lẽ xung tự nhiên nhất sẽ là chọn $y \in \{1, 2, 3\}$, trong đó các số nguyên đại diện cho $\{\text{dog}, \text{cat}, \text{chicken}\}$ tương ứng. Đây là một cách tuyệt vời để * lưu trữ* thông tin như vậy trên máy tính. Nếu các danh mục có một số thứ tự nhiên trong số đó, hãy nói nếu chúng ta đang cố gắng dự đoán $\{\text{baby}, \text{toddler}, \text{adolescent}, \text{young adult}, \text{adult}, \text{geriatric}\}$, thì thậm chí có thể khiến vấn đề này làm hồi quy và giữ nhãn ở định dạng này. 

Nhưng các vấn đề phân loại chung không đi kèm với trật tự tự nhiên giữa các lớp học. May mắn thay, các nhà thống kê từ lâu đã phát minh ra một cách đơn giản để đại diện cho dữ liệu phân loại: mã hóa* một*. Mã hóa một nóng là một vector với nhiều thành phần như chúng ta có các loại. Thành phần tương ứng với thể loại của phiên bản cụ thể được đặt thành 1 và tất cả các thành phần khác được đặt thành 0. Trong trường hợp của chúng tôi, một nhãn $y$ sẽ là một vector ba chiều, với $(1, 0, 0)$ tương ứng với “mèo”, $(0, 1, 0)$ đến “gà” và $(0, 0, 1)$ thành “chó”: 

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}.$$

## Kiến trúc mạng

Để ước tính xác suất có điều kiện liên quan đến tất cả các lớp có thể, chúng ta cần một mô hình với nhiều đầu ra, một cho mỗi lớp. Để giải quyết phân loại với các mô hình tuyến tính, chúng ta sẽ cần nhiều hàm affine như chúng ta có đầu ra. Mỗi đầu ra sẽ tương ứng với chức năng affine riêng của nó. Trong trường hợp của chúng tôi, vì chúng tôi có 4 tính năng và 3 danh mục đầu ra có thể, chúng tôi sẽ cần 12 vô hướng để đại diện cho trọng lượng ($w$ với các chỉ mục) và 3 vô hướng để đại diện cho các thành kiến ($b$ với chỉ mục). Chúng tôi tính toán ba *logits* này, $o_1, o_2$, và $o_3$, cho mỗi đầu vào: 

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

Chúng ta có thể mô tả tính toán này với sơ đồ mạng thần kinh được hiển thị trong :numref:`fig_softmaxreg`. Cũng giống như trong hồi quy tuyến tính, hồi quy softmax cũng là một mạng thần kinh một lớp. Và kể từ khi tính toán của mỗi đầu ra, $o_1, o_2$ và $o_3$, phụ thuộc vào tất cả các đầu vào, $x_1$, $x_2$, $x_3$ và $x_4$, lớp đầu ra của hồi quy softmax cũng có thể được mô tả là lớp kết nối hoàn toàn. 

![Softmax regression is a single-layer neural network.](../img/softmaxreg.svg)
:label:`fig_softmaxreg`

Để thể hiện mô hình nhỏ gọn hơn, chúng ta có thể sử dụng ký hiệu đại số tuyến tính. Ở dạng vector, chúng tôi đến $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$, một hình thức phù hợp hơn cho cả toán học và viết mã. Lưu ý rằng chúng tôi đã thu thập tất cả các trọng lượng của chúng tôi thành ma trận $3 \times 4$ và đối với các tính năng của một ví dụ dữ liệu nhất định $\mathbf{x}$, đầu ra của chúng tôi được đưa ra bởi một sản phẩm ma thuật-vector có trọng lượng của chúng tôi bởi các tính năng đầu vào của chúng tôi cộng với thiên vị của chúng tôi $\mathbf{b}$. 

## Chi phí tham số của các lớp được kết nối hoàn toàn
:label:`subsec_parameterization-cost-fc-layers`

Như chúng ta sẽ thấy trong các chương tiếp theo, các lớp được kết nối hoàn toàn có mặt khắp nơi trong học sâu. Tuy nhiên, như tên cho thấy, các lớp được kết nối hoàn toàn là * đầy đủ* kết nối với nhiều tham số có khả năng học được. Cụ thể, đối với bất kỳ lớp kết nối hoàn toàn nào với đầu vào $d$ và đầu ra $q$, chi phí tham số hóa là $\mathcal{O}(dq)$, có thể rất cao trong thực tế. May mắn thay, chi phí chuyển đổi đầu vào $d$ này thành đầu ra $q$ có thể được giảm xuống còn $\mathcal{O}(\frac{dq}{n})$, trong đó siêu tham số $n$ có thể được chúng tôi xác định linh hoạt để cân bằng giữa tiết kiệm thông số và hiệu quả mô hình trong các ứng dụng thế giới thực :cite:`Zhang.Tay.Zhang.ea.2021`. 

## Softmax hoạt động
:label:`subsec_softmax_operation`

Cách tiếp cận chính mà chúng ta sẽ thực hiện ở đây là giải thích các kết quả đầu ra của mô hình của chúng tôi là xác suất. Chúng tôi sẽ tối ưu hóa các thông số của mình để tạo ra xác suất tối đa hóa khả năng của dữ liệu quan sát. Sau đó, để tạo ra dự đoán, chúng tôi sẽ đặt một ngưỡng, ví dụ, chọn nhãn với xác suất dự đoán tối đa. 

Đặt chính thức, chúng tôi muốn bất kỳ đầu ra $\hat{y}_j$ được hiểu là xác suất mà một mục nhất định thuộc về lớp $j$. Sau đó, chúng ta có thể chọn lớp có giá trị đầu ra lớn nhất như dự đoán của chúng ta $\operatorname*{argmax}_j y_j$. Ví dụ: nếu $\hat{y}_1$, $\hat{y}_2$ và $\hat{y}_3$ lần lượt là 0,1, 0,8 và 0,1, thì chúng tôi dự đoán loại 2, trong đó (trong ví dụ của chúng tôi) đại diện cho “gà”. 

Bạn có thể bị cám dỗ để đề nghị rằng chúng tôi giải thích các bản ghi $o$ trực tiếp như đầu ra của chúng tôi quan tâm. Tuy nhiên, có một số vấn đề với việc giải thích trực tiếp đầu ra của lớp tuyến tính là một xác suất. Một mặt, không có gì hạn chế những con số này để tổng cộng với 1. Mặt khác, tùy thuộc vào đầu vào, chúng có thể lấy giá trị âm. Những vi phạm tiên đề cơ bản của xác suất được trình bày trong :numref:`sec_prob` 

Để hiểu kết quả đầu ra của chúng tôi là xác suất, chúng tôi phải đảm bảo rằng (ngay cả trên dữ liệu mới), chúng sẽ không âm và tổng hợp lên đến 1. Hơn nữa, chúng ta cần một mục tiêu đào tạo khuyến khích mô hình ước tính xác suất trung thực. Trong tất cả các trường hợp khi một phân loại đầu ra 0.5, chúng tôi hy vọng rằng một nửa trong số các ví dụ đó sẽ thực sự thuộc về lớp dự đoán. Đây là một thuộc tính gọi là *calibration*. 

Chức năng *softmax*, được phát minh vào năm 1959 bởi nhà khoa học xã hội R.Duncan Luce trong bối cảnh mô hình *lựa chọn*, thực hiện chính xác điều này. Để chuyển đổi nhật ký của chúng tôi sao cho chúng trở nên không âm và tổng thành 1, trong khi yêu cầu mô hình vẫn có thể khác biệt, trước tiên chúng ta cấp mũ mỗi logit (đảm bảo không tiêu cực) và sau đó chia cho tổng của chúng (đảm bảo rằng chúng tổng thành 1): 

$$\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad \hat{y}_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}. $$
:eqlabel:`eq_softmax_y_and_o`

Thật dễ dàng để xem $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ với $0 \leq \hat{y}_j \leq 1$ cho tất cả $j$. Do đó, $\hat{\mathbf{y}}$ là một phân phối xác suất thích hợp mà các giá trị phần tử có thể được diễn giải cho phù hợp. Lưu ý rằng thao tác softmax không thay đổi thứ tự trong số các bản ghi $\mathbf{o}$, đơn giản là các giá trị pre-softmax xác định xác suất được gán cho mỗi lớp. Do đó, trong quá trình dự đoán, chúng ta vẫn có thể chọn ra lớp có khả năng nhất bằng 

$$
\operatorname*{argmax}_j \hat y_j = \operatorname*{argmax}_j o_j.
$$

Mặc dù softmax là một hàm phi tuyến, các đầu ra của hồi quy softmax vẫn là * quyết định* bởi một biến đổi affine của các tính năng đầu vào; do đó, hồi quy softmax là một mô hình tuyến tính. 

## Vectorization cho Minibatches
:label:`subsec_softmax_vectorization`

Để cải thiện hiệu quả tính toán và tận dụng lợi thế của GPU, chúng tôi thường thực hiện các phép tính vector cho các minibatches dữ liệu. Giả sử rằng chúng ta được đưa ra một minibatch $\mathbf{X}$ ví dụ với kích thước tính năng (số lượng đầu vào) $d$ và kích thước lô $n$. Hơn nữa, giả sử rằng chúng tôi có $q$ danh mục trong đầu ra. Sau đó, các tính năng minibatch $\mathbf{X}$ là trong $\mathbb{R}^{n \times d}$, trọng lượng $\mathbf{W} \in \mathbb{R}^{d \times q}$ và sự thiên vị đáp ứng $\mathbf{b} \in \mathbb{R}^{1\times q}$. 

$$ \begin{aligned} \mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b}, \\ \hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O}). \end{aligned} $$
:eqlabel:`eq_minibatch_softmax_reg`

Điều này tăng tốc hoạt động thống trị thành một sản phẩm ma trận ma trận $\mathbf{X} \mathbf{W}$ so với các sản phẩm ma trận vector mà chúng tôi sẽ thực hiện nếu chúng tôi xử lý một ví dụ tại một thời điểm. Vì mỗi hàng trong $\mathbf{X}$ đại diện cho một ví dụ dữ liệu, nên bản thân thao tác softmax có thể được tính toán * rowwise*: cho mỗi hàng $\mathbf{O}$, cấp số mũ tất cả các mục nhập và sau đó bình thường hóa chúng bằng tổng. Kích hoạt phát sóng trong tổng kết $\mathbf{X} \mathbf{W} + \mathbf{b}$ trong :eqref:`eq_minibatch_softmax_reg`, cả hai minibatch đăng nhập $\mathbf{O}$ và xác suất đầu ra $\hat{\mathbf{Y}}$ là $n \times q$ ma trận. 

## Chức năng mất

Tiếp theo, chúng ta cần một chức năng mất mát để đo lường chất lượng xác suất dự đoán của chúng tôi. Chúng ta sẽ dựa vào ước tính khả năng tối đa, khái niệm rất giống nhau mà chúng ta gặp phải khi cung cấp một biện minh xác suất cho mục tiêu sai số bình phương trung bình trong hồi quy tuyến tính (:numref:`subsec_normal_distribution_and_squared_loss`). 

### Log-Likelihood

Hàm softmax cung cấp cho chúng ta một vector $\hat{\mathbf{y}}$, mà chúng ta có thể giải thích như xác suất có điều kiện ước tính của mỗi lớp cho bất kỳ đầu vào $\mathbf{x}$, ví dụ, $\hat{y}_1$ = $P(y=\text{cat} \mid \mathbf{x})$. Giả sử rằng toàn bộ dữ liệu $\{\mathbf{X}, \mathbf{Y}\}$ có $n$ ví dụ, trong đó ví dụ được lập chỉ mục bởi $i$ bao gồm một vector tính năng $\mathbf{x}^{(i)}$ và một vector nhãn một nóng $\mathbf{y}^{(i)}$. Chúng ta có thể so sánh các ước tính với thực tế bằng cách kiểm tra xem các lớp thực tế có thể xảy ra như thế nào theo mô hình của chúng tôi, với các tính năng: 

$$
P(\mathbf{Y} \mid \mathbf{X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)}).
$$

Theo ước tính khả năng tối đa, chúng tôi tối đa hóa $P(\mathbf{Y} \mid \mathbf{X})$, tương đương với việc giảm thiểu khả năng log âm: 

$$
-\log P(\mathbf{Y} \mid \mathbf{X}) = \sum_{i=1}^n -\log P(\mathbf{y}^{(i)} \mid \mathbf{x}^{(i)})
= \sum_{i=1}^n l(\mathbf{y}^{(i)}, \hat{\mathbf{y}}^{(i)}),
$$

nơi đối với bất kỳ cặp nhãn $\mathbf{y}$ và dự đoán mô hình $\hat{\mathbf{y}}$ trên các lớp $q$, hàm mất $l$ là 

$$ l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_{j=1}^q y_j \log \hat{y}_j. $$
:eqlabel:`eq_l_cross_entropy`

Vì lý do được giải thích sau này, hàm mất trong :eqref:`eq_l_cross_entropy` thường được gọi là mất *cross-entropy*. Kể từ $\mathbf{y}$ là một vectơ một nóng có chiều dài $q$, tổng trên tất cả các tọa độ của nó $j$ biến mất cho tất cả trừ một thuật ngữ. Vì tất cả $\hat{y}_j$ được dự đoán xác suất, logarit của chúng không bao giờ lớn hơn $0$. Do đó, hàm mất không thể được giảm thiểu thêm nữa nếu chúng ta dự đoán chính xác nhãn thực tế với *chắc chắn*, tức là, nếu xác suất dự đoán $P(\mathbf{y} \mid \mathbf{x}) = 1$ cho nhãn thực tế $\mathbf{y}$. Lưu ý rằng điều này thường là không thể. Ví dụ: có thể có nhiễu nhãn trong tập dữ liệu (một số ví dụ có thể bị dán nhãn sai). Cũng có thể không thể thực hiện được khi các tính năng đầu vào không đủ thông tin để phân loại mọi ví dụ một cách hoàn hảo. 

### Softmax và các dẫn xuất
:label:`subsec_softmax_and_derivatives`

Vì softmax và tổn thất tương ứng rất phổ biến, nên hiểu rõ hơn một chút về cách nó được tính toán. Cắm :eqref:`eq_softmax_y_and_o` vào định nghĩa về sự mất mát trong :eqref:`eq_l_cross_entropy` và sử dụng định nghĩa của softmax chúng ta có được: 

$$
\begin{aligned}
l(\mathbf{y}, \hat{\mathbf{y}}) &=  - \sum_{j=1}^q y_j \log \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} \\
&= \sum_{j=1}^q y_j \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j\\
&= \log \sum_{k=1}^q \exp(o_k) - \sum_{j=1}^q y_j o_j.
\end{aligned}
$$

Để hiểu rõ hơn một chút những gì đang xảy ra, hãy xem xét phái sinh liên quan đến bất kỳ logit $o_j$ nào. Chúng tôi nhận được 

$$
\partial_{o_j} l(\mathbf{y}, \hat{\mathbf{y}}) = \frac{\exp(o_j)}{\sum_{k=1}^q \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j.
$$

Nói cách khác, đạo hàm là sự khác biệt giữa xác suất được gán bởi mô hình của chúng ta, như thể hiện bằng phép toán softmax, và những gì thực sự đã xảy ra, như thể hiện bằng các phần tử trong vectơ nhãn một nóng. Theo nghĩa này, nó rất giống với những gì chúng ta đã thấy trong hồi quy, trong đó gradient là sự khác biệt giữa quan sát $y$ và ước tính $\hat{y}$. Đây không phải là sự trùng hợp ngẫu nhiên. Trong bất kỳ họ cấp số nhân nào (xem mô hình [online appendix on distributions](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/distributions.html)), độ dốc của khả năng log được đưa ra bởi chính xác thuật ngữ này. Thực tế này làm cho gradient tính toán dễ dàng trong thực tế. 

### Mất Cross-Entropy

Bây giờ hãy xem xét trường hợp mà chúng ta quan sát không chỉ là một kết quả duy nhất mà còn là toàn bộ phân phối trên kết quả. Chúng ta có thể sử dụng đại diện tương tự như trước cho nhãn $\mathbf{y}$. Sự khác biệt duy nhất là thay vì một vector chỉ chứa các mục nhị phân, giả sử $(0, 0, 1)$, bây giờ chúng ta có một vector xác suất chung, nói $(0.1, 0.2, 0.7)$. Toán học mà chúng tôi đã sử dụng trước đây để xác định mất $l$ trong :eqref:`eq_l_cross_entropy` vẫn hoạt động tốt, chỉ là việc giải thích là tổng quát hơn một chút. Đó là giá trị dự kiến của sự mất mát cho một phân phối trên nhãn. Tổn thất này được gọi là mất *cross-entropy* và nó là một trong những tổn thất được sử dụng phổ biến nhất cho các vấn đề phân loại. Chúng ta có thể demystify tên bằng cách giới thiệu chỉ những điều cơ bản của lý thuyết thông tin. Nếu bạn muốn hiểu thêm chi tiết về lý thuyết thông tin, bạn có thể tham khảo thêm [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html). 

## Thông tin cơ bản về lý thuyết
:label:`subsec_info_theory_basics`

*Lý lịch thông tin* đề cập đến vấn đề mã hóa, giải mã, truyền tải,
và thao tác thông tin (còn được gọi là dữ liệu) ở dạng ngắn gọn nhất có thể. 

### Entropy

Ý tưởng trung tâm trong lý thuyết thông tin là định lượng nội dung thông tin trong dữ liệu. Số lượng này đặt một giới hạn cứng về khả năng nén dữ liệu của chúng tôi. Trong lý thuyết thông tin, đại lượng này được gọi là *entropy* của một phân phối $P$, và nó được bắt bởi phương trình sau: 

$$H[P] = \sum_j - P(j) \log P(j).$$
:eqlabel:`eq_softmax_reg_entropy`

Một trong những định lý cơ bản của lý thuyết thông tin nói rằng để mã hóa dữ liệu được rút ra ngẫu nhiên từ phân phối $P$, chúng ta cần ít nhất $H[P]$ “net” để mã hóa nó. Nếu bạn tự hỏi một “nat” là gì, nó tương đương với bit nhưng khi sử dụng một mã với base $e$ chứ không phải là một với base 2. Như vậy, một nat là $\frac{1}{\log(2)} \approx 1.44$ bit. 

### Surprisal

Bạn có thể tự hỏi nén có liên quan gì đến dự đoán. Hãy tưởng tượng rằng chúng ta có một luồng dữ liệu mà chúng ta muốn nén. Nếu chúng ta luôn dễ dàng dự đoán token tiếp theo, thì dữ liệu này rất dễ nén! Lấy ví dụ cực đoan trong đó mọi token trong luồng luôn có cùng giá trị. Đó là một luồng dữ liệu rất nhàm chán! Và không chỉ nó là nhàm chán, nhưng nó cũng dễ dàng để dự đoán. Bởi vì chúng luôn giống nhau, chúng tôi không phải truyền tải bất kỳ thông tin nào để truyền đạt nội dung của luồng. Dễ dự đoán, dễ nén. 

Tuy nhiên, nếu chúng ta không thể dự đoán hoàn hảo mọi sự kiện, thì đôi khi chúng ta có thể ngạc nhiên. Sự ngạc nhiên của chúng tôi là lớn hơn khi chúng tôi chỉ định một sự kiện xác suất thấp hơn. Claude Shannon định cư vào $\log \frac{1}{P(j)} = -\log P(j)$ để định lượng của một người *ngạc nhiên* khi quan sát một sự kiện $j$ đã gán cho nó một xác suất (chủ quan) $P(j)$. Entropy được định nghĩa trong :eqref:`eq_softmax_reg_entropy` sau đó là sự ngạc nhiên dự kiến * khi người ta gán xác suất chính xác thực sự phù hợp với quá trình tạo dữ liệu. 

### Cross-Entropy Revisited

Vì vậy, nếu entropy là mức độ bất ngờ trải qua bởi một người biết xác suất thực sự, thì bạn có thể tự hỏi, cross-entropy là gì? Các cross-entropy* từ* $P$ * đến* $Q$, được ký hiệu là $H(P, Q)$, là sự ngạc nhiên dự kiến của một người quan sát với xác suất chủ quan $Q$ khi nhìn thấy dữ liệu thực sự được tạo ra theo xác suất $P$. Entropy chéo thấp nhất có thể đạt được khi $P=Q$. Trong trường hợp này, entropy chéo từ $P$ đến $Q$ là $H(P, P)= H(P)$. 

Nói tóm lại, chúng ta có thể nghĩ đến mục tiêu phân loại chéo entropy theo hai cách: (i) là tối đa hóa khả năng của dữ liệu quan sát; và (ii) là giảm thiểu sự ngạc nhiên của chúng ta (và do đó số lượng bit) cần thiết để truyền đạt các nhãn. 

## Dự đoán và đánh giá mô hình

Sau khi đào tạo mô hình hồi quy softmax, cho bất kỳ tính năng ví dụ nào, chúng ta có thể dự đoán xác suất của mỗi lớp đầu ra. Thông thường, chúng ta sử dụng class có xác suất dự đoán cao nhất làm class output. Dự đoán là chính xác nếu nó phù hợp với lớp thực tế (label). Trong phần tiếp theo của thử nghiệm, chúng tôi sẽ sử dụng *accuracy* để đánh giá hiệu suất của mô hình. Điều này bằng tỷ lệ giữa số dự đoán chính xác và tổng số dự đoán. 

## Tóm tắt

* Thao tác softmax lấy một vector và ánh xạ nó thành xác suất.
* Hồi quy Softmax áp dụng cho các bài toán phân loại. Nó sử dụng phân phối xác suất của lớp đầu ra trong hoạt động softmax.
* Cross-entropy là một thước đo tốt về sự khác biệt giữa hai phân phối xác suất. Nó đo số lượng bit cần thiết để mã hóa dữ liệu cho mô hình của chúng tôi.

## Bài tập

1. Chúng ta có thể khám phá mối liên hệ giữa các gia đình cấp số nhân và softmax ở một số chiều sâu hơn.
    1. Tính toán đạo hàm thứ hai của tổn thất chéo entropy $l(\mathbf{y},\hat{\mathbf{y}})$ cho softmax.
    1. Tính toán phương sai của phân phối được đưa ra bởi $\mathrm{softmax}(\mathbf{o})$ và cho thấy nó khớp với đạo hàm thứ hai được tính toán ở trên.
1. Giả sử rằng chúng ta có ba lớp xảy ra với xác suất bằng nhau, tức là vector xác suất là $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    1. Vấn đề là gì nếu chúng ta cố gắng thiết kế một mã nhị phân cho nó?
    1. Bạn có thể thiết kế một mã tốt hơn? Gợi ý: điều gì sẽ xảy ra nếu chúng ta cố gắng mã hóa hai quan sát độc lập? Điều gì sẽ xảy ra nếu chúng ta mã hóa $n$ quan sát cùng nhau?
1. Softmax là một sai lầm cho bản đồ được giới thiệu ở trên (nhưng tất cả mọi người trong deep learning đều sử dụng nó). Softmax thực được định nghĩa là $\mathrm{RealSoftMax}(a, b) = \log (\exp(a) + \exp(b))$.
    1. Chứng minh rằng $\mathrm{RealSoftMax}(a, b) > \mathrm{max}(a, b)$.
    1. Chứng minh rằng điều này giữ cho $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, với điều kiện là $\lambda > 0$.
    1. Cho thấy rằng đối với $\lambda \to \infty$ chúng tôi có $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a, b)$.
    1. Soft-min trông như thế nào?
    1. Mở rộng điều này đến hơn hai số.

[Discussions](https://discuss.d2l.ai/t/46)

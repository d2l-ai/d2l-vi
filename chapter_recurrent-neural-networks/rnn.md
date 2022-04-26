# Mạng nơ-ron tái phát
:label:`sec_rnn`

Trong :numref:`sec_language_model` chúng tôi đã giới thiệu các mô hình $n$-gram, trong đó xác suất có điều kiện của từ $x_t$ tại bước thời gian $t$ chỉ phụ thuộc vào $n-1$ từ trước. Nếu chúng ta muốn kết hợp hiệu ứng có thể có của các từ sớm hơn bước thời gian $t-(n-1)$ trên $x_t$, chúng ta cần tăng $n$. Tuy nhiên, số lượng tham số mô hình cũng sẽ tăng theo cấp số nhân với nó, vì chúng ta cần lưu trữ $|\mathcal{V}|^n$ số cho một bộ từ vựng $\mathcal{V}$. Do đó, thay vì mô hình hóa $P(x_t \mid x_{t-1}, \ldots, x_{t-n+1})$, tốt hơn là sử dụng một mô hình biến tiềm ẩn: 

$$P(x_t \mid x_{t-1}, \ldots, x_1) \approx P(x_t \mid h_{t-1}),$$

trong đó $h_{t-1}$ là trạng thái * ẩn* (còn được gọi là biến ẩn) lưu trữ thông tin trình tự lên đến bước thời gian $t-1$. Nói chung, trạng thái ẩn bất cứ lúc nào bước $t$ có thể được tính toán dựa trên cả đầu vào hiện tại $x_{t}$ và trạng thái ẩn trước đó $h_{t-1}$: 

$$h_t = f(x_{t}, h_{t-1}).$$
:eqlabel:`eq_ht_xt`

Đối với một chức năng đủ mạnh $f$ trong :eqref:`eq_ht_xt`, mô hình biến tiềm ẩn không phải là một xấp xỉ. Rốt cuộc, $h_t$ có thể chỉ đơn giản là lưu trữ tất cả dữ liệu mà nó đã quan sát cho đến nay. Tuy nhiên, nó có khả năng có thể làm cho cả tính toán và lưu trữ đắt tiền. 

Nhớ lại rằng chúng ta đã thảo luận về các lớp ẩn với các đơn vị ẩn trong :numref:`chap_perceptrons`. Đáng chú ý là các lớp ẩn và trạng thái ẩn đề cập đến hai khái niệm rất khác nhau. Các lớp ẩn, như giải thích, các lớp được ẩn khỏi chế độ xem trên đường dẫn từ đầu vào đến đầu ra. Các trạng thái ẩn đang nói về mặt kỹ thuật * đầu vào* cho bất cứ điều gì chúng ta làm ở một bước nhất định và chúng chỉ có thể được tính toán bằng cách xem dữ liệu ở các bước thời gian trước. 

*Mạng thần kinh định kỳ* (RNN) là các mạng thần kinh với các trạng thái ẩn. Trước khi giới thiệu mô hình RNN, lần đầu tiên chúng tôi xem lại mô hình MLP được giới thiệu trong :numref:`sec_mlp`.

## Mạng thần kinh không có trạng thái ẩn

Chúng ta hãy nhìn vào một MLP với một lớp ẩn duy nhất. Hãy để chức năng kích hoạt của lớp ẩn là $\phi$. Với một minibatch các ví dụ $\mathbf{X} \in \mathbb{R}^{n \times d}$ với kích thước lô $n$ và $d$ đầu vào, đầu ra $\mathbf{H} \in \mathbb{R}^{n \times h}$ của lớp ẩn được tính như 

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$
:eqlabel:`rnn_h_without_state`

Trong :eqref:`rnn_h_without_state`, chúng ta có tham số trọng lượng $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}$, tham số thiên vị $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ và số lượng đơn vị ẩn $h$, cho lớp ẩn. Như vậy, phát sóng (xem :numref:`subsec_broadcasting`) được áp dụng trong quá trình tổng kết. Tiếp theo, biến ẩn $\mathbf{H}$ được sử dụng làm đầu vào của lớp đầu ra. Lớp đầu ra được đưa ra bởi 

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q,$$

trong đó $\mathbf{O} \in \mathbb{R}^{n \times q}$ là biến đầu ra, $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ là tham số trọng lượng và $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ là tham số thiên vị của lớp đầu ra. Nếu đó là một bài toán phân loại, chúng ta có thể sử dụng $\text{softmax}(\mathbf{O})$ để tính toán phân phối xác suất của các loại đầu ra. 

Điều này hoàn toàn tương tự như bài toán hồi quy mà chúng tôi đã giải quyết trước đây trong :numref:`sec_sequence`, do đó chúng tôi bỏ qua các chi tiết. Đủ để nói rằng chúng ta có thể chọn các cặp nhãn tính năng một cách ngẫu nhiên và tìm hiểu các thông số của mạng của chúng tôi thông qua sự phân biệt tự động và gốc gradient ngẫu nhiên. 

## Mạng thần kinh định kỳ với các trạng thái ẩn
:label:`subsec_rnn_w_hidden_states`

Các vấn đề hoàn toàn khác nhau khi chúng ta có các trạng thái ẩn. Chúng ta hãy nhìn vào cấu trúc một số chi tiết hơn. 

Giả sử rằng chúng ta có một minibatch của đầu vào $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ tại bước thời gian $t$. Nói cách khác, đối với một minibatch gồm $n$ ví dụ chuỗi, mỗi hàng $\mathbf{X}_t$ tương ứng với một ví dụ tại bước thời điểm $t$ từ chuỗi. Tiếp theo, biểu thị bằng $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ biến ẩn của bước thời gian $t$. Không giống như MLP, ở đây chúng ta lưu biến ẩn $\mathbf{H}_{t-1}$ từ bước thời gian trước và giới thiệu một tham số trọng lượng mới $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ để mô tả cách sử dụng biến ẩn của bước thời gian trước đó trong bước thời gian hiện tại. Cụ thể, việc tính toán biến ẩn của bước thời gian hiện tại được xác định bởi đầu vào của bước thời gian hiện tại cùng với biến ẩn của bước thời gian trước đó: 

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$
:eqlabel:`rnn_h_with_state`

So với :eqref:`rnn_h_without_state`, :eqref:`rnn_h_with_state` bổ sung thêm một thuật ngữ $\mathbf{H}_{t-1} \mathbf{W}_{hh}$ và do đó khởi tạo :eqref:`eq_ht_xt`. Từ mối quan hệ giữa các biến ẩn $\mathbf{H}_t$ và $\mathbf{H}_{t-1}$ của các bước thời gian liền kề, chúng ta biết rằng các biến này đã thu thập và giữ lại thông tin lịch sử của trình tự lên đến bước thời gian hiện tại của chúng, giống như trạng thái hoặc bộ nhớ của bước thời gian hiện tại của mạng thần kinh. Do đó, một biến ẩn như vậy được gọi là trạng thái * hidden*. Vì trạng thái ẩn sử dụng cùng một định nghĩa của bước thời gian trước đó trong bước thời gian hiện tại, việc tính toán :eqref:`rnn_h_with_state` là* đệ quy*. Do đó, các mạng thần kinh với các trạng thái ẩn dựa trên tính toán tái phát được đặt tên
*mạng thần kinh định kỳ*.
Các lớp thực hiện tính toán :eqref:`rnn_h_with_state` trong RNN s được gọi là * lớp* lặp đi lặp lại*. 

Có nhiều cách khác nhau để xây dựng RNNs. RNNs với trạng thái ẩn được xác định bởi :eqref:`rnn_h_with_state` là rất phổ biến. Đối với bước thời gian $t$, đầu ra của lớp đầu ra tương tự như tính toán trong MLP: 

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

Các thông số của RNN bao gồm các trọng lượng $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$, và thiên vị $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$ của lớp ẩn, cùng với trọng lượng $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ và thiên vị $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ của lớp đầu ra. Điều đáng nói là ngay cả ở các bước thời gian khác nhau, RNN s luôn sử dụng các tham số mô hình này. Do đó, chi phí tham số hóa của một RNN không tăng lên khi số bước thời gian tăng lên. 

:numref:`fig_rnn` minh họa logic tính toán của một RNN tại ba bước thời gian liền kề. Bất cứ lúc nào bước $t$, tính toán trạng thái ẩn có thể được coi là: (i) nối đầu vào $\mathbf{X}_t$ ở bước thời gian hiện tại $t$ và trạng thái ẩn $\mathbf{H}_{t-1}$ ở bước thời gian trước $t-1$; (ii) cho kết quả nối vào một lớp kết nối đầy đủ với sự kích hoạt chức năng $\phi$. Đầu ra của một lớp được kết nối hoàn toàn như vậy là trạng thái ẩn $\mathbf{H}_t$ của bước thời gian hiện tại $t$. Trong trường hợp này, các thông số mô hình là nối $\mathbf{W}_{xh}$ và $\mathbf{W}_{hh}$, và một sự thiên vị của $\mathbf{b}_h$, tất cả từ :eqref:`rnn_h_with_state`. Trạng thái ẩn của bước thời gian hiện tại $t$, $\mathbf{H}_t$, sẽ tham gia vào việc tính toán trạng thái ẩn $\mathbf{H}_{t+1}$ của bước thời gian tiếp theo $t+1$. Hơn nữa, $\mathbf{H}_t$ cũng sẽ được đưa vào lớp đầu ra được kết nối hoàn toàn để tính toán đầu ra $\mathbf{O}_t$ của bước thời gian hiện tại $t$. 

![An RNN with a hidden state.](../img/rnn.svg)
:label:`fig_rnn`

Chúng tôi vừa đề cập rằng việc tính $\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$ cho trạng thái ẩn tương đương với phép nhân ma trận của nối $\mathbf{X}_t$ và $\mathbf{H}_{t-1}$ và nối $\mathbf{H}_{t-1}$ và nối $\mathbf{W}_{xh}$ và $\mathbf{W}_{hh}$. Mặc dù điều này có thể được chứng minh trong toán học, sau đây chúng ta chỉ sử dụng một đoạn mã đơn giản để hiển thị điều này. Để bắt đầu, chúng tôi xác định ma trận `X`, `W_xh`, `H` và `W_hh`, có hình dạng tương ứng (3, 1), (1, 4), (3, 4) và (4, 4). Nhân `X` với `W_xh`, và `H` `W_hh`, tương ứng, và sau đó thêm hai nhân này, chúng tôi có được một ma trận hình dạng (3, 4).

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

```{.python .input}
#@tab mxnet, pytorch
X, W_xh = d2l.normal(0, 1, (3, 1)), d2l.normal(0, 1, (1, 4))
H, W_hh = d2l.normal(0, 1, (3, 4)), d2l.normal(0, 1, (4, 4))
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

```{.python .input}
#@tab tensorflow
X, W_xh = d2l.normal((3, 1), 0, 1), d2l.normal((1, 4), 0, 1)
H, W_hh = d2l.normal((3, 4), 0, 1), d2l.normal((4, 4), 0, 1)
d2l.matmul(X, W_xh) + d2l.matmul(H, W_hh)
```

Bây giờ chúng ta nối các ma trận `X` và `H` dọc theo các cột (trục 1), và ma trận `W_xh` và `W_hh` dọc theo hàng (trục 0). Hai kết nối này dẫn đến ma trận của hình dạng (3, 5) và hình dạng (5, 4), tương ứng. Nhân hai ma trận nối này, chúng ta có được ma trận đầu ra giống nhau của hình dạng (3, 4) như trên.

```{.python .input}
#@tab all
d2l.matmul(d2l.concat((X, H), 1), d2l.concat((W_xh, W_hh), 0))
```

## Mô hình ngôn ngữ cấp ký tự dựa trên RNN

Nhớ lại rằng đối với mô hình hóa ngôn ngữ trong :numref:`sec_language_model`, chúng tôi đặt mục tiêu dự đoán token tiếp theo dựa trên các mã thông báo hiện tại và quá khứ, do đó chúng tôi chuyển chuỗi gốc bằng một mã thông báo làm nhãn. Bengio et al. lần đầu tiên đề xuất sử dụng mạng thần kinh để mô hình hóa ngôn ngữ :cite:`Bengio.Ducharme.Vincent.ea.2003`. Sau đây, chúng tôi minh họa cách RNN có thể được sử dụng để xây dựng một mô hình ngôn ngữ. Hãy để kích thước minibatch là một, và trình tự của văn bản là “máy”. Để đơn giản hóa việc đào tạo trong các phần tiếp theo, chúng tôi mã hóa văn bản thành các ký tự thay vì từ và xem xét mô hình ngôn ngữ cấp ký tự*. :numref:`fig_rnn_train` thể hiện cách dự đoán ký tự tiếp theo dựa trên các ký tự hiện tại và trước đó thông qua RNN cho mô hình ngôn ngữ cấp ký tự. 

![A character-level language model based on the RNN. The input and label sequences are "machin" and "achine", respectively.](../img/rnn-train.svg)
:label:`fig_rnn_train`

Trong quá trình đào tạo, chúng tôi chạy một thao tác softmax trên đầu ra từ lớp đầu ra cho mỗi bước thời gian, sau đó sử dụng tổn thất chéo entropy để tính toán lỗi giữa đầu ra mô hình và nhãn. Do tính toán tái phát của trạng thái ẩn trong lớp ẩn, đầu ra của bước thời gian 3 trong :numref:`fig_rnn_train`, $\mathbf{O}_3$, được xác định bởi chuỗi văn bản “m”, “a”, và “c”. Vì ký tự tiếp theo của dãy trong dữ liệu đào tạo là “h”, việc mất thời gian bước 3 sẽ phụ thuộc vào phân bố xác suất của ký tự tiếp theo được tạo ra dựa trên dãy tính năng “m”, “a”, “c” và nhãn “h” của bước thời gian này. 

Trong thực tế, mỗi mã thông báo được đại diện bởi một vector $d$ chiều, và chúng tôi sử dụng kích thước lô $n>1$. Do đó, đầu vào $\mathbf X_t$ tại bước thời gian $t$ sẽ là ma trận $n\times d$, giống hệt với những gì chúng ta đã thảo luận trong :numref:`subsec_rnn_w_hidden_states`. 

## Bối rối
:label:`subsec_perplexity`

Cuối cùng, chúng ta hãy thảo luận về cách đo lường chất lượng mô hình ngôn ngữ, sẽ được sử dụng để đánh giá các mô hình dựa trên RN của chúng tôi trong các phần tiếp theo. Một cách là kiểm tra văn bản đáng ngạc nhiên như thế nào. Một mô hình ngôn ngữ tốt có thể dự đoán với các mã thông báo có độ chính xác cao mà những gì chúng ta sẽ thấy tiếp theo. Hãy xem xét các liên tục sau của cụm từ “Trời mưa”, như được đề xuất bởi các mô hình ngôn ngữ khác nhau: 

1. “Trời đang mưa bên ngoài”
1. “Trời đang mưa cây chuối”
1. “It is raining mưa; kcj pwepoiut”

Về chất lượng, ví dụ 1 rõ ràng là tốt nhất. Các từ là hợp lý và hợp lý mạch lạc. Mặc dù nó có thể không hoàn toàn phản ánh chính xác từ nào theo ngữ nghĩa (“ở San Francisco” và “vào mùa đông” sẽ là phần mở rộng hoàn toàn hợp lý), mô hình có thể nắm bắt loại từ nào theo sau. Ví dụ 2 tồi tệ hơn đáng kể bằng cách tạo ra một phần mở rộng vô nghĩa. Tuy nhiên, ít nhất mô hình đã học cách đánh vần các từ và một mức độ tương quan giữa các từ. Cuối cùng, ví dụ 3 chỉ ra một mô hình được đào tạo kém không phù hợp với dữ liệu đúng cách. 

Chúng ta có thể đo lường chất lượng của mô hình bằng cách tính toán khả năng của trình tự. Thật không may đây là một con số khó hiểu và khó so sánh. Rốt cuộc, các trình tự ngắn hơn có nhiều khả năng xảy ra hơn nhiều so với các trình tự dài hơn, do đó đánh giá mô hình trên magnum opus của Tolstoy
*Chiến tranh và Hòa bình* chắc chắn sẽ tạo ra một khả năng nhỏ hơn nhiều so với, nói, trên tiểu thuyết của Saint-Exupery* The Little Prince*. Những gì còn thiếu là tương đương với mức trung bình.

Lý thuyết thông tin có ích ở đây. Chúng tôi đã xác định entropy, ngạc nhiên và cross-entropy khi chúng tôi giới thiệu hồi quy softmax (:numref:`subsec_info_theory_basics`) và nhiều lý thuyết thông tin được thảo luận trong [online appendix on information theory](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/information-theory.html). Nếu chúng ta muốn nén văn bản, chúng ta có thể hỏi về việc dự đoán token tiếp theo cho bộ mã thông báo hiện tại. Một mô hình ngôn ngữ tốt hơn sẽ cho phép chúng ta dự đoán token tiếp theo chính xác hơn. Do đó, nó sẽ cho phép chúng ta chi tiêu ít bit hơn trong việc nén chuỗi. Vì vậy, chúng ta có thể đo lường nó bằng cách mất chéo entropy trung bình trên tất cả các mã thông báo $n$ của một chuỗi: 

$$\frac{1}{n} \sum_{t=1}^n -\log P(x_t \mid x_{t-1}, \ldots, x_1),$$
:eqlabel:`eq_avg_ce_for_lm`

trong đó $P$ được đưa ra bởi một mô hình ngôn ngữ và $x_t$ là mã thông báo thực tế quan sát tại bước thời gian $t$ từ chuỗi. Điều này làm cho hiệu suất trên các tài liệu có độ dài khác nhau tương đương. Vì lý do lịch sử, các nhà khoa học trong xử lý ngôn ngữ tự nhiên thích sử dụng một số lượng gọi là *perplexity*. Tóm lại, nó là hàm mũ của :eqref:`eq_avg_ce_for_lm`: 

$$\exp\left(-\frac{1}{n} \sum_{t=1}^n \log P(x_t \mid x_{t-1}, \ldots, x_1)\right).$$

Sự bối rối có thể được hiểu rõ nhất là trung bình hài hòa của số lượng lựa chọn thực sự mà chúng ta có khi quyết định token nào sẽ chọn tiếp theo. Chúng ta hãy xem xét một số trường hợp: 

* Trong trường hợp tốt nhất, mô hình luôn ước tính hoàn hảo xác suất của mã thông báo nhãn là 1. Trong trường hợp này, sự bối rối của mô hình là 1.
* Trong trường hợp xấu nhất, mô hình luôn dự đoán xác suất của token nhãn là 0. Trong tình huống này, sự bối rối là vô cùng tích cực.
* Tại đường cơ sở, mô hình dự đoán phân phối thống nhất trên tất cả các mã thông báo có sẵn của từ vựng. Trong trường hợp này, sự bối rối bằng số lượng mã thông báo duy nhất của từ vựng. Trên thực tế, nếu chúng ta lưu trữ trình tự mà không cần bất kỳ nén nào, đây sẽ là điều tốt nhất chúng ta có thể làm để mã hóa nó. Do đó, điều này cung cấp một ràng buộc trên không tầm thường mà bất kỳ mô hình hữu ích nào cũng phải đánh bại.

Trong các phần sau, chúng tôi sẽ triển khai RNNcho các mô hình ngôn ngữ cấp ký tự và sử dụng sự bối rối để đánh giá các mô hình như vậy. 

## Tóm tắt

* Một mạng thần kinh sử dụng tính toán định kỳ cho các trạng thái ẩn được gọi là mạng thần kinh tái phát (RNN).
* Trạng thái ẩn của một RNN có thể nắm bắt thông tin lịch sử của chuỗi lên đến bước thời gian hiện tại.
* Số lượng tham số mô hình RNN không tăng lên khi số bước thời gian tăng lên.
* Chúng ta có thể tạo mô hình ngôn ngữ cấp ký tự bằng cách sử dụng RNN.
* Chúng ta có thể sử dụng sự bối rối để đánh giá chất lượng của các mô hình ngôn ngữ.

## Bài tập

1. Nếu chúng ta sử dụng một RNN để dự đoán ký tự tiếp theo trong một chuỗi văn bản, kích thước cần thiết cho bất kỳ đầu ra nào là gì?
1. Tại sao RNN s có thể thể hiện xác suất có điều kiện của một mã thông báo tại một bước thời gian dựa trên tất cả các token trước đó trong chuỗi văn bản?
1. Điều gì xảy ra với gradient nếu bạn backpropagate thông qua một chuỗi dài?
1. Một số vấn đề liên quan đến mô hình ngôn ngữ được mô tả trong phần này là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/337)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1050)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1051)
:end_tab:

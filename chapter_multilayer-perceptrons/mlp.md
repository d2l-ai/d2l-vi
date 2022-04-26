# Multilayer Perceptrons
:label:`sec_mlp`

Trong :numref:`chap_linear`, chúng tôi giới thiệu hồi quy softmax (:numref:`sec_softmax`), thực hiện thuật toán từ đầu (:numref:`sec_softmax_scratch`) và sử dụng API cấp cao (:numref:`sec_softmax_concise`), và đào tạo phân loại để nhận ra 10 loại quần áo từ hình ảnh có độ phân giải thấp. Trên đường đi, chúng tôi đã học cách wrangle dữ liệu, ép buộc đầu ra của chúng tôi vào một phân phối xác suất hợp lệ, áp dụng một chức năng mất mát thích hợp và giảm thiểu nó đối với các tham số của mô hình của chúng tôi. Bây giờ chúng ta đã thành thạo các cơ chế này trong bối cảnh các mô hình tuyến tính đơn giản, chúng ta có thể khởi động khám phá các mạng thần kinh sâu, lớp mô hình tương đối phong phú mà cuốn sách này chủ yếu quan tâm. 

## Các lớp ẩn

Chúng tôi đã mô tả sự biến đổi affine trong :numref:`subsec_linear_model`, là một biến đổi tuyến tính được thêm vào bởi một sự thiên vị. Để bắt đầu, hãy nhớ lại kiến trúc mô hình tương ứng với ví dụ hồi quy softmax của chúng tôi, minh họa trong :numref:`fig_softmaxreg`. Mô hình này ánh xạ đầu vào của chúng tôi trực tiếp đến đầu ra của chúng tôi thông qua một biến đổi affine duy nhất, tiếp theo là một hoạt động softmax. Nếu nhãn của chúng tôi thực sự có liên quan đến dữ liệu đầu vào của chúng tôi bằng cách chuyển đổi affine, thì cách tiếp cận này sẽ là đủ. Nhưng tuyến tính trong biến đổi affine là một giả định * mạnh mẽ*. 

### Mô hình tuyến tính có thể đi sai

Ví dụ, tuyến tính ngụ ý giả định *yếu hơn* về *monotonicity*: rằng bất kỳ sự gia tăng tính năng nào của chúng tôi phải luôn gây ra sự gia tăng sản lượng của mô hình của chúng tôi (nếu trọng lượng tương ứng là dương) hoặc luôn làm giảm sản lượng của mô hình của chúng tôi (nếu trọng lượng tương ứng là âm). Đôi khi điều đó có ý nghĩa. Ví dụ, nếu chúng tôi đang cố gắng dự đoán liệu một cá nhân có trả nợ hay không, chúng ta có thể tưởng tượng một cách hợp lý rằng giữ tất cả đều bằng nhau, một người nộp đơn có thu nhập cao hơn sẽ luôn có nhiều khả năng trả nợ hơn một người có thu nhập thấp hơn. Trong khi đơn điệu, mối quan hệ này có khả năng không liên quan đến xác suất trả nợ. Tăng thu nhập từ 0 lên 50 nghìn có khả năng tương ứng với mức tăng khả năng trả nợ lớn hơn mức tăng từ 1 triệu lên 1.05 triệu. Một cách để xử lý điều này có thể là xử lý trước dữ liệu của chúng tôi sao cho tuyến tính trở nên hợp lý hơn, ví dụ, bằng cách sử dụng logarit thu nhập làm tính năng của chúng tôi. 

Lưu ý rằng chúng ta có thể dễ dàng đưa ra các ví dụ vi phạm tính đơn điệu. Nói ví dụ rằng chúng ta muốn dự đoán xác suất tử vong dựa trên nhiệt độ cơ thể. Đối với những người có nhiệt độ cơ thể trên 37° C (98,6° F), nhiệt độ cao hơn cho thấy nguy cơ cao hơn. Tuy nhiên, đối với những người có nhiệt độ cơ thể dưới 37° C, nhiệt độ cao hơn cho thấy nguy cơ thấp hơn! Trong trường hợp này, chúng tôi cũng có thể giải quyết vấn đề với một số tiền xử lý thông minh. Cụ thể, chúng ta có thể sử dụng khoảng cách từ 37° C làm tính năng của chúng tôi. 

Nhưng những gì về việc phân loại hình ảnh của mèo và chó? Có nên tăng cường độ của pixel tại vị trí (13, 17) luôn tăng (hoặc luôn giảm) khả năng hình ảnh mô tả một chú chó? Sự phụ thuộc vào một mô hình tuyến tính tương ứng với giả định ngầm rằng yêu cầu duy nhất để phân biệt mèo so với chó là đánh giá độ sáng của từng pixel. Cách tiếp cận này cam chịu thất bại trong một thế giới nơi đảo ngược một hình ảnh bảo tồn danh mục. 

Tuy nhiên, mặc dù sự vô lý rõ ràng của tuyến tính ở đây, so với các ví dụ trước đây của chúng tôi, nó là ít rõ ràng rằng chúng tôi có thể giải quyết vấn đề với một sửa chữa tiền xử lý đơn giản. Đó là do ý nghĩa của bất kỳ pixel nào phụ thuộc theo những cách phức tạp vào bối cảnh của nó (các giá trị của các pixel xung quanh). Mặc dù có thể tồn tại một đại diện của dữ liệu của chúng tôi có thể tính đến các tương tác có liên quan giữa các tính năng của chúng tôi, trên đó một mô hình tuyến tính sẽ phù hợp, chúng tôi chỉ đơn giản là không biết cách tính toán nó bằng tay. Với các mạng nơ-ron sâu, chúng tôi đã sử dụng dữ liệu quan sát để cùng tìm hiểu cả một biểu diễn thông qua các lớp ẩn và một bộ dự đoán tuyến tính hoạt động dựa trên biểu diễn đó. 

### Kết hợp các lớp ẩn

Chúng ta có thể khắc phục những hạn chế này của các mô hình tuyến tính và xử lý một lớp hàm tổng quát hơn bằng cách kết hợp một hoặc nhiều lớp ẩn. Cách dễ nhất để làm điều này là xếp chồng nhiều lớp được kết nối hoàn toàn lên nhau. Mỗi lớp nạp vào lớp phía trên nó, cho đến khi chúng ta tạo ra các đầu ra. Chúng ta có thể nghĩ về $L-1$ lớp đầu tiên là đại diện của chúng ta và lớp cuối cùng là bộ dự đoán tuyến tính của chúng ta. Kiến trúc này thường được gọi là một *multilayer perceptron*, thường được viết tắt là *MLP*. Dưới đây, chúng tôi mô tả một MLP diagrammatically (:numref:`fig_mlp`). 

![An MLP with a hidden layer of 5 hidden units. ](../img/mlp.svg)
:label:`fig_mlp`

MLP này có 4 đầu vào, 3 đầu ra và lớp ẩn của nó chứa 5 đơn vị ẩn. Vì lớp đầu vào không liên quan đến bất kỳ tính toán nào, việc tạo ra kết quả đầu ra với mạng này đòi hỏi phải thực hiện các tính toán cho cả hai lớp ẩn và đầu ra; do đó, số lớp trong MLP này là 2. Lưu ý rằng các lớp này đều được kết nối hoàn toàn. Mỗi đầu vào ảnh hưởng đến mọi tế bào thần kinh trong lớp ẩn, và mỗi lần lượt ảnh hưởng đến mọi tế bào thần kinh trong lớp đầu ra. Tuy nhiên, theo đề xuất của :numref:`subsec_parameterization-cost-fc-layers`, chi phí tham số hóa của MLP với các lớp được kết nối hoàn toàn có thể cao, điều này có thể thúc đẩy sự cân bằng giữa tiết kiệm tham số và hiệu quả mô hình ngay cả khi không thay đổi kích thước đầu vào hoặc đầu ra :cite:`Zhang.Tay.Zhang.ea.2021`. 

### Từ tuyến tính đến phi tuyến

Như trước đây, bởi ma trận $\mathbf{X} \in \mathbb{R}^{n \times d}$, chúng tôi biểu thị một minibatch gồm $n$ ví dụ trong đó mỗi ví dụ có $d$ đầu vào (tính năng). Đối với một MLP một lớp ẩn có lớp ẩn có $h$ đơn vị ẩn, biểu thị bằng $\mathbf{H} \in \mathbb{R}^{n \times h}$ các đầu ra của lớp ẩn, đó là
*đại diện ẩn*.
Trong toán học hoặc mã, $\mathbf{H}$ còn được gọi là một biến thể * ẩn lớp* hoặc một biến * hidden*. Vì các lớp ẩn và đầu ra đều được kết nối hoàn toàn, chúng ta có trọng lượng lớp ẩn $\mathbf{W}^{(1)} \in \mathbb{R}^{d \times h}$ và thiên vị $\mathbf{b}^{(1)} \in \mathbb{R}^{1 \times h}$ và trọng lượng lớp đầu ra $\mathbf{W}^{(2)} \in \mathbb{R}^{h \times q}$ và thiên vị $\mathbf{b}^{(2)} \in \mathbb{R}^{1 \times q}$. Chính thức, chúng tôi tính toán các đầu ra $\mathbf{O} \in \mathbb{R}^{n \times q}$ của MLP một lớp ẩn như sau: 

$$
\begin{aligned}
    \mathbf{H} & = \mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}, \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.
\end{aligned}
$$

Lưu ý rằng sau khi thêm lớp ẩn, mô hình của chúng tôi bây giờ yêu cầu chúng tôi theo dõi và cập nhật các bộ tham số bổ sung. Vì vậy, những gì chúng ta đã đạt được trong trao đổi? Bạn có thể ngạc nhiên khi tìm hiểu điều đó - trong mô hình được xác định ở trên —* chúng tôi không đạt được gì cho sự cố của chúng tôi*! Lý do là đơn giản. Các đơn vị ẩn ở trên được đưa ra bởi một hàm affine của các đầu vào, và các đầu ra (pre-softmax) chỉ là một hàm affine của các đơn vị ẩn. Một hàm affine của một hàm affine tự nó là một hàm affine. Hơn nữa, mô hình tuyến tính của chúng tôi đã có khả năng đại diện cho bất kỳ chức năng affine nào. 

Chúng ta có thể xem sự tương đương chính thức bằng cách chứng minh rằng đối với bất kỳ giá trị nào của trọng lượng, chúng ta chỉ có thể thu gọn lớp ẩn, mang lại một mô hình một lớp tương đương với các tham số $\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}$ và $\mathbf{b} = \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}$: 

$$
\mathbf{O} = (\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{X} \mathbf{W} + \mathbf{b}.
$$

Để nhận ra tiềm năng của các kiến trúc đa lớp, chúng ta cần thêm một thành phần quan trọng: chức năng kích hoạt* phi tuyến * $\sigma$ được áp dụng cho mỗi đơn vị ẩn sau khi chuyển đổi affine. Các đầu ra của các chức năng kích hoạt (ví dụ, $\sigma(\cdot)$) được gọi là *kích hoạt*. Nói chung, với các chức năng kích hoạt tại chỗ, không còn có thể thu gọn MLP của chúng ta thành một mô hình tuyến tính: 

$$
\begin{aligned}
    \mathbf{H} & = \sigma(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)}), \\
    \mathbf{O} & = \mathbf{H}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}.\\
\end{aligned}
$$

Vì mỗi hàng trong $\mathbf{X}$ tương ứng với một ví dụ trong minibatch, với một số lạm dụng ký hiệu, chúng tôi xác định tính phi tuyến $\sigma$ để áp dụng cho các đầu vào của nó theo kiểu hàng, tức là một ví dụ tại một thời điểm. Lưu ý rằng chúng ta đã sử dụng ký hiệu cho softmax theo cùng một cách để biểu thị một hoạt động theo hàng trong :numref:`subsec_softmax_vectorization`. Thông thường, như trong phần này, các chức năng kích hoạt mà chúng ta áp dụng cho các lớp ẩn không chỉ đơn thuần là hàng, mà là elementwise. Điều đó có nghĩa là sau khi tính toán phần tuyến tính của lớp, chúng ta có thể tính toán từng kích hoạt mà không cần nhìn vào các giá trị được thực hiện bởi các đơn vị ẩn khác. Điều này đúng đối với hầu hết các chức năng kích hoạt. 

Để xây dựng các MLP chung hơn, chúng ta có thể tiếp tục xếp chồng các lớp ẩn như vậy, ví dụ, $\mathbf{H}^{(1)} = \sigma_1(\mathbf{X} \mathbf{W}^{(1)} + \mathbf{b}^{(1)})$ và $\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)})$, một trên đỉnh khác, mang lại các mô hình biểu cảm hơn bao giờ hết. 

### Phổ Approximators

MLP có thể nắm bắt các tương tác phức tạp giữa các đầu vào của chúng ta thông qua các tế bào thần kinh ẩn của chúng, phụ thuộc vào các giá trị của từng đầu vào. Chúng ta có thể dễ dàng thiết kế các nút ẩn để thực hiện tính toán tùy ý, ví dụ, các phép toán logic cơ bản trên một cặp đầu vào. Hơn nữa, đối với một số lựa chọn nhất định của chức năng kích hoạt, người ta biết rằng MLPlà gần đúng phổ quát. Ngay cả với một mạng lưới một lớp ẩn, cho đủ nút (có thể vô lý nhiều), và tập hợp đúng trọng lượng, chúng ta có thể mô hình hóa bất kỳ chức năng, mặc dù thực sự học chức năng đó là phần cứng. Bạn có thể nghĩ về mạng thần kinh của bạn như là một chút giống như ngôn ngữ lập trình C. Ngôn ngữ, giống như bất kỳ ngôn ngữ hiện đại nào khác, có khả năng thể hiện bất kỳ chương trình có thể tính toán nào. Nhưng thực sự đến với một chương trình đáp ứng thông số kỹ thuật của bạn là phần khó khăn. 

Hơn nữa, chỉ vì một mạng lưới một lớp ẩn
*có thể* học bất kỳ chức năng nào
không có nghĩa là bạn nên cố gắng giải quyết tất cả các vấn đề của mình với các mạng một lớp ẩn. Trên thực tế, chúng ta có thể xấp xỉ nhiều chức năng nhỏ gọn hơn nhiều bằng cách sử dụng các mạng sâu hơn (so với rộng hơn). Chúng tôi sẽ đề cập đến những lập luận nghiêm ngặt hơn trong các chương tiếp theo. 

## Chức năng kích hoạt
:label:`subsec_activation-functions`

Các chức năng kích hoạt quyết định xem một tế bào thần kinh có nên được kích hoạt hay không bằng cách tính tổng trọng số và thêm thiên vị với nó. Chúng là các toán tử khác biệt để chuyển đổi tín hiệu đầu vào thành đầu ra, trong khi hầu hết chúng thêm tính phi tuyến tính. Bởi vì các chức năng kích hoạt là nền tảng cho việc học sâu, (** hãy để chúng tôi khảo sát ngắn gọn một số chức năng kích hoạt phổ biến**).

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

### Chức năng ReLU

Sự lựa chọn phổ biến nhất, do cả sự đơn giản của việc thực hiện và hiệu suất tốt của nó trên một loạt các nhiệm vụ dự đoán, là đơn vị tuyến tính * sửa chữ* (* Relu*). [**ReLU cung cấp một biến đổi phi tuyến rất đơn giản**]. Cho một phần tử $x$, hàm được định nghĩa là tối đa của phần tử đó và $0$: 

$$\operatorname{ReLU}(x) = \max(x, 0).$$

Không chính thức, hàm ReLU chỉ giữ lại các phần tử dương và loại bỏ tất cả các yếu tố âm bằng cách đặt các kích hoạt tương ứng thành 0. Để đạt được một số trực giác, chúng ta có thể vẽ hàm. Như bạn có thể thấy, chức năng kích hoạt là piecewise tuyến tính.

```{.python .input}
x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.relu(x)
d2l.plot(x, y, 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
d2l.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(tf.range(-8.0, 8.0, 0.1), dtype=tf.float32)
y = tf.nn.relu(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'relu(x)', figsize=(5, 2.5))
```

Khi đầu vào là âm, đạo hàm của hàm ReLU là 0, và khi đầu vào là dương, đạo hàm của hàm ReLU là 1. Lưu ý rằng hàm ReLU không phân biệt được khi đầu vào lấy giá trị chính xác bằng 0. Trong những trường hợp này, chúng ta mặc định là đạo hàm bên trái và nói rằng đạo hàm là 0 khi đầu vào là 0. Chúng ta có thể nhận được đi với điều này bởi vì đầu vào có thể không bao giờ thực sự bằng không. Có một câu ngạn ngữ cũ rằng nếu điều kiện ranh giới tinh tế quan trọng, chúng ta có thể đang làm (* real*) toán học, chứ không phải kỹ thuật. Trí tuệ thông thường đó có thể áp dụng ở đây. Chúng tôi vẽ đạo hàm của hàm ReLU được vẽ bên dưới.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y.backward(torch.ones_like(x), retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.relu(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of relu',
         figsize=(5, 2.5))
```

Lý do để sử dụng ReLU là các dẫn xuất của nó được cư xử đặc biệt tốt: hoặc chúng biến mất hoặc họ chỉ để cho cuộc tranh luận thông qua. Điều này làm cho tối ưu hóa hoạt động tốt hơn và nó giảm thiểu vấn đề được tài liệu tốt về độ dốc biến mất cản các phiên bản trước của mạng thần kinh (thêm về điều này sau). 

Lưu ý rằng có nhiều biến thể cho hàm ReLU, bao gồm hàm ReLU* (*pReLU*) được tham số hóa :cite:`He.Zhang.Ren.ea.2015`. Biến thể này thêm một thuật ngữ tuyến tính vào ReLU, vì vậy một số thông tin vẫn được thông qua, ngay cả khi đối số là âm: 

$$\operatorname{pReLU}(x) = \max(0, x) + \alpha \min(0, x).$$

### Chức năng Sigmoid

[**Chức năng * sigmoid* biến đổi đầu vào của nó**], mà các giá trị nằm trong miền $\mathbb{R}$, (** để đầu ra nằm trong khoảng thời gian (0, 1) .**) Vì lý do đó, sigmoid thường được gọi là hàm *squashing*: nó đè bẹp bất kỳ đầu vào trong phạm vi (-inf, inf) đến một số giá trị trong phạm vi (0, 1): 

$$\operatorname{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$$

Trong các mạng thần kinh sớm nhất, các nhà khoa học quan tâm đến việc mô hình hóa các tế bào thần kinh sinh học mà * cháy* hoặc * không cháy*. Do đó, những người tiên phong của lĩnh vực này, đi tất cả các con đường trở lại McCulloch và Pitt, những nhà phát minh của tế bào thần kinh nhân tạo, tập trung vào các đơn vị ngưỡng. Kích hoạt ngưỡng có giá trị 0 khi đầu vào của nó nằm dưới ngưỡng và giá trị 1 khi đầu vào vượt quá ngưỡng. 

Khi sự chú ý chuyển sang học tập dựa trên gradient, chức năng sigmoid là một lựa chọn tự nhiên bởi vì nó là một xấp xỉ trơn tru, khác biệt với một đơn vị ngưỡng. Sigmoids vẫn được sử dụng rộng rãi như các hàm kích hoạt trên các đơn vị đầu ra, khi chúng ta muốn diễn giải các đầu ra như là xác suất cho các bài toán phân loại nhị phân (bạn có thể nghĩ về sigmoid như một trường hợp đặc biệt của softmax). Tuy nhiên, sigmoid chủ yếu được thay thế bằng ReLU đơn giản và dễ dàng hơn để sử dụng hầu hết trong các lớp ẩn. Trong các chương sau trên các mạng thần kinh định kỳ, chúng tôi sẽ mô tả các kiến trúc tận dụng các đơn vị sigmoid để kiểm soát luồng thông tin theo thời gian. 

Dưới đây, chúng tôi vẽ hàm sigmoid. Lưu ý rằng khi đầu vào gần 0, hàm sigmoid tiếp cận một biến đổi tuyến tính.

```{.python .input}
with autograd.record():
    y = npx.sigmoid(x)
d2l.plot(x, y, 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.sigmoid(x)
d2l.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```

Đạo hàm của hàm sigmoid được cho bởi phương trình sau: 

$$\frac{d}{dx} \operatorname{sigmoid}(x) = \frac{\exp(-x)}{(1 + \exp(-x))^2} = \operatorname{sigmoid}(x)\left(1-\operatorname{sigmoid}(x)\right).$$

Đạo hàm của hàm sigmoid được vẽ bên dưới. Lưu ý rằng khi đầu vào là 0, đạo hàm của hàm sigmoid đạt tối đa 0,25. Khi đầu vào phân kỳ từ 0 theo một trong hai hướng, đạo hàm tiếp cận 0.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of sigmoid',
         figsize=(5, 2.5))
```

### Chức năng Tanh

Giống như hàm sigmoid, [**hàm tanh (hyperbol tangent) cũng đè bẹp các đầu vào của nó**], biến chúng thành các phần tử trong khoảng thời gian (**giữa -1 và 1**): 

$$\operatorname{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$$

Chúng tôi vẽ hàm tanh dưới đây. Lưu ý rằng khi đầu vào gần 0, hàm tanh tiếp cận một phép biến đổi tuyến tính. Mặc dù hình dạng của hàm tương tự như của hàm sigmoid, hàm tánh thể hiện tính đối xứng điểm về nguồn gốc của hệ tọa độ.

```{.python .input}
with autograd.record():
    y = np.tanh(x)
d2l.plot(x, y, 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
y = torch.tanh(x)
d2l.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
y = tf.nn.tanh(x)
d2l.plot(x.numpy(), y.numpy(), 'x', 'tanh(x)', figsize=(5, 2.5))
```

Đạo hàm của hàm tánh là: 

$$\frac{d}{dx} \operatorname{tanh}(x) = 1 - \operatorname{tanh}^2(x).$$

Đạo hàm của hàm tánh được vẽ bên dưới. Khi đầu vào gần 0, đạo hàm của hàm tanh tiếp cận tối đa là 1. Và như chúng ta đã thấy với hàm sigmoid, khi đầu vào di chuyển ra khỏi 0 theo một trong hai hướng, đạo hàm của hàm tanh tiếp cận 0.

```{.python .input}
y.backward()
d2l.plot(x, x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab pytorch
# Clear out previous gradients.
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
d2l.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.nn.tanh(x)
d2l.plot(x.numpy(), t.gradient(y, x).numpy(), 'x', 'grad of tanh',
         figsize=(5, 2.5))
```

Tóm lại, bây giờ chúng ta biết làm thế nào để kết hợp phi tuyến tính để xây dựng kiến trúc mạng thần kinh đa lớp biểu cảm. Như một lưu ý phụ, kiến thức của bạn đã đặt bạn trong chỉ huy của một bộ công cụ tương tự cho một học viên khoảng 1990. Theo một số cách nào đó, bạn có lợi thế hơn bất kỳ ai làm việc trong những năm 1990, bởi vì bạn có thể tận dụng các khuôn khổ deep learning mã nguồn mở mạnh mẽ để xây dựng các mô hình nhanh chóng, chỉ sử dụng một vài dòng mã. Trước đây, đào tạo các mạng này đòi hỏi các nhà nghiên cứu phải mã hóa hàng ngàn dòng C và Fortran. 

## Tóm tắt

* MLP thêm một hoặc nhiều lớp ẩn được kết nối hoàn toàn giữa các lớp đầu ra và đầu vào và biến đổi đầu ra của lớp ẩn thông qua chức năng kích hoạt.
* Các chức năng kích hoạt được sử dụng phổ biến bao gồm hàm ReLU, hàm sigmoid và hàm tanh.

## Bài tập

1. Tính toán đạo hàm của hàm kích hoạt pReLU.
1. Cho thấy một MLP chỉ sử dụng ReLU (hoặc pReLU) xây dựng một hàm tuyến tính piecewise liên tục.
1. Cho thấy rằng $\operatorname{tanh}(x) + 1 = 2 \operatorname{sigmoid}(2x)$.
1. Giả sử rằng chúng ta có một phi tuyến tính áp dụng cho một minibatch tại một thời điểm. Những loại vấn đề nào bạn mong đợi điều này gây ra?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/90)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/91)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/226)
:end_tab:

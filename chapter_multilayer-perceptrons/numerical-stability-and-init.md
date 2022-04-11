# Tính ổn định số và khởi tạo
:label:`sec_numerical_stability`

Cho đến nay, mọi mô hình mà chúng tôi đã thực hiện đều yêu cầu chúng tôi khởi tạo các tham số của nó theo một số phân phối được chỉ định trước. Cho đến bây giờ, chúng tôi đã thực hiện sơ đồ khởi tạo là điều hiển nhiên, nêu rõ các chi tiết về cách các lựa chọn này được thực hiện. Bạn thậm chí có thể đã nhận được ấn tượng rằng những lựa chọn này không đặc biệt quan trọng. Ngược lại, việc lựa chọn sơ đồ khởi tạo đóng một vai trò quan trọng trong việc học mạng thần kinh, và nó có thể rất quan trọng để duy trì sự ổn định số. Hơn nữa, những lựa chọn này có thể được gắn lên theo những cách thú vị với sự lựa chọn của chức năng kích hoạt phi tuyến. Chức năng nào chúng ta chọn và cách chúng ta khởi tạo các tham số có thể xác định thuật toán tối ưu hóa của chúng ta hội tụ nhanh như thế nào. Những lựa chọn kém ở đây có thể khiến chúng ta gặp phải sự bùng nổ hoặc biến mất gradient trong khi đào tạo. Trong phần này, chúng tôi đi sâu vào các chủ đề này với chi tiết hơn và thảo luận về một số heuristics hữu ích mà bạn sẽ thấy hữu ích trong suốt sự nghiệp của mình trong việc học sâu. 

## Biến mất và bùng nổ Gradient

Hãy xem xét một mạng sâu với $L$ lớp, đầu vào $\mathbf{x}$ và đầu ra $\mathbf{o}$. Với mỗi lớp $l$ được xác định bởi một biến đổi $f_l$ tham số hóa bởi trọng lượng $\mathbf{W}^{(l)}$, có biến ẩn là $\mathbf{h}^{(l)}$ (để $\mathbf{h}^{(0)} = \mathbf{x}$), mạng của chúng tôi có thể được biểu thị như: 

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

Nếu tất cả các biến ẩn và đầu vào là vectơ, chúng ta có thể viết gradient của $\mathbf{o}$ đối với bất kỳ tập hợp các tham số $\mathbf{W}^{(l)}$ như sau: 

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$

Nói cách khác, gradient này là sản phẩm của ma trận $L-l$ $\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$ và vector gradient $\mathbf{v}^{(l)}$. Do đó, chúng ta dễ bị các vấn đề tương tự của dòng chảy số thường cắt lên khi nhân với nhau quá nhiều xác suất. Khi đối phó với xác suất, một mẹo phổ biến là chuyển sang không gian đăng nhập, tức là, chuyển áp lực từ mantissa sang số mũ của biểu diễn số. Thật không may, vấn đề của chúng tôi ở trên nghiêm trọng hơn: ban đầu các ma trận $\mathbf{M}^{(l)}$ có thể có nhiều eigenvalues. Chúng có thể nhỏ hoặc lớn, và sản phẩm của họ có thể là * rất lớn* hoặc * rất nhỏ*. 

Các rủi ro gây ra bởi gradient không ổn định vượt ra ngoài biểu diễn số. Độ dốc có độ lớn không thể đoán trước cũng đe dọa sự ổn định của các thuật toán tối ưu hóa của chúng tôi. Chúng ta có thể phải đối mặt với các bản cập nhật tham số (i) quá lớn, phá hủy mô hình của chúng tôi (vấn đề * exploding gradient*); hoặc (ii) quá nhỏ (vấn đề * vanishing gradient*), khiến việc học không thể như các tham số hầu như không di chuyển trên mỗi bản cập nhật. 

### (** Vanishing Gradients**)

Một thủ phạm thường xuyên gây ra vấn đề gradient biến mất là sự lựa chọn của hàm kích hoạt $\sigma$ được nối sau các phép toán tuyến tính của mỗi lớp. Trong lịch sử, hàm sigmoid $1/(1 + \exp(-x))$ (được giới thiệu vào năm :numref:`sec_mlp`) rất phổ biến vì nó giống như một hàm ngưỡng. Vì các mạng thần kinh nhân tạo ban đầu được lấy cảm hứng từ các mạng thần kinh sinh học, ý tưởng về các tế bào thần kinh bắn * đầy đủ* hoặc * không* (như tế bào thần kinh sinh học) dường như hấp dẫn. Chúng ta hãy xem xét kỹ hơn sigmoid để xem lý do tại sao nó có thể gây ra gradient biến mất.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Như bạn có thể thấy, (** gradient của sigmoid biến mất cả khi đầu vào của nó lớn và khi chúng nhỏ**). Hơn nữa, khi truyền ngược qua nhiều lớp, trừ khi chúng ta ở trong vùng Goldilocks, nơi các đầu vào cho nhiều sigmoids gần bằng 0, độ dốc của sản phẩm tổng thể có thể biến mất. Khi mạng của chúng tôi tự hào có nhiều lớp, trừ khi chúng tôi cẩn thận, gradient có thể sẽ bị cắt ở một số lớp. Thật vậy, vấn đề này được sử dụng để đào tạo mạng sâu bệnh dịch hạch. Do đó, ReLUs, ổn định hơn (nhưng ít hợp lý về mặt thần kinh), đã nổi lên như là lựa chọn mặc định cho các học viên. 

### [**Bùng nổ Gradients**]

Vấn đề ngược lại, khi gradient nổ tung, có thể tương tự như vexing. Để minh họa điều này tốt hơn một chút, chúng tôi vẽ 100 ma trận ngẫu nhiên Gaussian và nhân chúng với một số ma trận ban đầu. Đối với thang đo mà chúng tôi đã chọn (sự lựa chọn của phương sai $\sigma^2=1$), sản phẩm ma trận phát nổ. Khi điều này xảy ra do khởi tạo một mạng sâu, chúng ta không có cơ hội nhận được trình tối ưu hóa gradient descent để hội tụ.

```{.python .input}
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)
```

```{.python .input}
#@tab pytorch
M = torch.normal(0, 1, size=(4,4))
print('a single matrix \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('after multiplying 100 matrices\n', M)
```

```{.python .input}
#@tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('after multiplying 100 matrices\n', M.numpy())
```

### Phá vỡ đối xứng

Một vấn đề khác trong thiết kế mạng thần kinh là sự đối xứng vốn có trong tham số hóa của chúng. Giả sử rằng chúng ta có một MLP đơn giản với một lớp ẩn và hai đơn vị. Trong trường hợp này, chúng ta có thể hoán vị trọng lượng $\mathbf{W}^{(1)}$ của lớp đầu tiên và tương tự như vậy hoán vị trọng lượng của lớp đầu ra để có được chức năng tương tự. Không có gì đặc biệt phân biệt đơn vị ẩn đầu tiên so với đơn vị ẩn thứ hai. Nói cách khác, chúng ta có sự đối xứng hoán vị giữa các đơn vị ẩn của mỗi lớp. 

Đây không chỉ là một phiền toái lý thuyết. Hãy xem xét MLP một lớp ẩn đã nói ở trên với hai đơn vị ẩn. Để minh họa, giả sử lớp đầu ra biến đổi hai đơn vị ẩn thành chỉ một đơn vị đầu ra. Hãy tưởng tượng điều gì sẽ xảy ra nếu chúng ta khởi tạo tất cả các tham số của lớp ẩn là $\mathbf{W}^{(1)} = c$ cho một số hằng số $c$. Trong trường hợp này, trong quá trình tuyên truyền chuyển tiếp hoặc đơn vị ẩn có cùng một đầu vào và tham số, tạo ra cùng một kích hoạt, được đưa vào đơn vị đầu ra. Trong quá trình lan truyền ngược, phân biệt đơn vị đầu ra đối với các tham số $\mathbf{W}^{(1)}$ cho một gradient có các phần tử có cùng giá trị. Do đó, sau khi lặp lại dựa trên gradient (ví dụ, minibatch stochastic gradient descent), tất cả các phần tử của $\mathbf{W}^{(1)}$ vẫn có cùng giá trị. Các lần lặp như vậy sẽ không bao giờ * tự phá vỡ đối xưởng* và chúng ta có thể không bao giờ có thể nhận ra sức mạnh biểu cảm của mạng. Lớp ẩn sẽ hoạt động như thể nó chỉ có một đơn vị duy nhất. Lưu ý rằng trong khi minibatch stochastic gradient gốc sẽ không phá vỡ đối xứng này, việc bỏ học thường xuyên sẽ! 

## Khởi tạo tham số

Một cách để giải quyết - hoặc ít nhất là giảm thiểu - các vấn đề nêu trên là thông qua việc khởi tạo cẩn thận. Chăm sóc bổ sung trong quá trình tối ưu hóa và chính quy hóa phù hợp có thể tăng cường hơn nữa sự ổn định. 

### Khởi tạo mặc định

Trong các phần trước, ví dụ, trong :numref:`sec_linear_concise`, chúng tôi đã sử dụng một phân phối bình thường để khởi tạo các giá trị của trọng lượng của chúng tôi. Nếu chúng ta không chỉ định phương thức khởi tạo, framework sẽ sử dụng phương thức khởi tạo ngẫu nhiên mặc định, thường hoạt động tốt trong thực tế cho kích thước vấn đề vừa phải. 

### Khởi tạo Xavier
:label:`subsec_xavier`

Chúng ta hãy xem xét phân phối tỷ lệ của một đầu ra (ví dụ, một biến ẩn) $o_{i}$ cho một số lớp được kết nối hoàn toàn
*không có phi tuyến tính*.
Với $n_\mathrm{in}$ đầu vào $x_j$ và trọng lượng liên quan của chúng $w_{ij}$ cho lớp này, một đầu ra được đưa ra bởi 

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$

Các trọng lượng $w_{ij}$ đều được vẽ độc lập từ cùng một phân phối. Hơn nữa, chúng ta hãy giả định rằng phân phối này có 0 trung bình và phương sai $\sigma^2$. Lưu ý rằng điều này không có nghĩa là phân phối phải là Gaussian, chỉ là trung bình và phương sai cần phải tồn tại. Hiện tại, chúng ta hãy giả định rằng các đầu vào cho lớp $x_j$ cũng có 0 trung bình và phương sai $\gamma^2$ và chúng độc lập với $w_{ij}$ và độc lập với nhau. Trong trường hợp này, chúng ta có thể tính toán trung bình và phương sai của $o_i$ như sau: 

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Một cách để giữ phương sai cố định là đặt $n_\mathrm{in} \sigma^2 = 1$. Bây giờ hãy xem xét backpropagation. Ở đó chúng ta phải đối mặt với một vấn đề tương tự, mặc dù với gradient được truyền từ các lớp gần đầu ra hơn. Sử dụng lý luận tương tự như đối với tuyên truyền chuyển tiếp, chúng ta thấy rằng phương sai của gradient có thể thổi lên trừ khi $n_\mathrm{out} \sigma^2 = 1$, trong đó $n_\mathrm{out}$ là số lượng đầu ra của lớp này. Điều này khiến chúng ta gặp khó xử: chúng ta không thể thỏa mãn cả hai điều kiện cùng một lúc. Thay vào đó, chúng tôi chỉ đơn giản là cố gắng thỏa mãn: 

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

Đây là lý do dựa trên chuẩn hiện nay và thực tế có lợi * Xavier khởi hóa*, được đặt tên theo tác giả đầu tiên của người tạo ra nó :cite:`Glorot.Bengio.2010`. Thông thường, các mẫu khởi tạo Xavier trọng lượng từ phân phối Gaussian với 0 trung bình và phương sai $\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$. Chúng ta cũng có thể điều chỉnh trực giác của Xavier để chọn phương sai khi lấy mẫu trọng lượng từ phân phối đồng đều. Lưu ý rằng sự phân bố thống nhất $U(-a, a)$ có phương sai $\frac{a^2}{3}$. Cắm $\frac{a^2}{3}$ vào điều kiện của chúng tôi trên $\sigma^2$ mang lại gợi ý khởi tạo theo 

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$

Mặc dù giả định cho sự không tồn tại của phi tuyến tính trong lý luận toán học trên có thể dễ dàng vi phạm trong các mạng thần kinh, phương pháp khởi tạo Xavier hóa ra hoạt động tốt trong thực tế. 

### Beyond

Lý do trên hầu như không làm trầy xước bề mặt của các phương pháp tiếp cận hiện đại để khởi tạo tham số. Một khuôn khổ học sâu thường thực hiện hơn một chục heuristics khác nhau. Hơn nữa, khởi tạo tham số tiếp tục là một lĩnh vực nóng của nghiên cứu cơ bản trong học sâu. Trong số này có heuristics chuyên cho các tham số gắn (chia sẻ), siêu phân giải, mô hình trình tự và các tình huống khác. Ví dụ, Xiao et al. đã chứng minh khả năng đào tạo mạng thần kinh 10000 lớp mà không cần thủ thuật kiến trúc bằng cách sử dụng phương pháp khởi tạo được thiết kế cẩn thận :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`. 

Nếu chủ đề mà bạn quan tâm, chúng tôi đề nghị đi sâu vào các dịch vụ của mô-đun này, đọc các bài báo đề xuất và phân tích từng heuristic, sau đó khám phá các ấn phẩm mới nhất về chủ đề này. Có lẽ bạn sẽ vấp ngã hoặc thậm chí phát minh ra một ý tưởng thông minh và đóng góp thực hiện cho các khuôn khổ học sâu. 

## Tóm tắt

* Biến mất và bùng nổ gradient là những vấn đề phổ biến trong các mạng sâu. Chăm sóc tuyệt vời trong khởi tạo tham số là cần thiết để đảm bảo rằng độ dốc và tham số vẫn được kiểm soát tốt.
* Khởi tạo heuristics là cần thiết để đảm bảo rằng các gradient ban đầu không quá lớn cũng không quá nhỏ.
* Chức năng kích hoạt ReLU giảm thiểu vấn đề gradient biến mất. Điều này có thể đẩy nhanh sự hội tụ.
* Khởi tạo ngẫu nhiên là chìa khóa để đảm bảo rằng đối xứng bị phá vỡ trước khi tối ưu hóa.
* Khởi tạo Xavier gợi ý rằng, đối với mỗi lớp, phương sai của bất kỳ đầu ra nào không bị ảnh hưởng bởi số lượng đầu vào và phương sai của bất kỳ gradient nào không bị ảnh hưởng bởi số lượng đầu ra.

## Bài tập

1. Bạn có thể thiết kế các trường hợp khác trong đó một mạng thần kinh có thể thể hiện đối xứng đòi hỏi phải phá vỡ bên cạnh sự đối xứng hoán vị trong các lớp của MLP?
1. Chúng ta có thể khởi tạo tất cả các tham số trọng lượng trong hồi quy tuyến tính hoặc hồi quy softmax đến cùng một giá trị?
1. Tra cứu giới hạn phân tích trên eigenvalues của sản phẩm của hai ma trận. Điều này cho bạn biết gì về việc đảm bảo rằng độ dốc được điều hòa tốt?
1. Nếu chúng ta biết rằng một số thuật ngữ phân kỳ, chúng ta có thể khắc phục điều này sau khi thực tế? Nhìn vào bài báo về tỷ lệ thích ứng layerwise tỷ lệ cho cảm hứng :cite:`You.Gitman.Ginsburg.2017`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:

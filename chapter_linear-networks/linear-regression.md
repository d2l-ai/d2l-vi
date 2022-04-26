# Hồi quy tuyến tính
:label:`sec_linear_regression`

*Hồi quy* đề cập đến một tập hợp các phương pháp để mô hình hóa
mối quan hệ giữa một hoặc nhiều biến độc lập và một biến phụ thuộc. Trong khoa học tự nhiên và khoa học xã hội, mục đích của hồi quy thường là
*characterize* mối quan hệ giữa các đầu vào và đầu ra.
Mặt khác, học máy thường liên quan đến * dự đoán *. 

Vấn đề hồi quy bật lên bất cứ khi nào chúng ta muốn dự đoán một giá trị số. Các ví dụ phổ biến bao gồm dự đoán giá (nhà cửa, cổ phiếu, v.v.), dự đoán thời gian lưu trú (đối với bệnh nhân trong bệnh viện), dự báo nhu cầu (đối với doanh số bán lẻ), trong số vô số những người khác. Không phải mọi vấn đề dự đoán đều là một vấn đề hồi quy cổ điển. Trong các phần tiếp theo, chúng tôi sẽ giới thiệu các vấn đề phân loại, trong đó mục tiêu là dự đoán tư cách thành viên trong một tập hợp các loại. 

## Các yếu tố cơ bản của hồi quy tuyến tính

*Hồi quy tuyến tính* có thể là đơn giản nhất
và phổ biến nhất trong số các công cụ tiêu chuẩn để hồi quy. Có niên đại từ bình minh của thế kỷ 19, hồi quy tuyến tính chảy từ một vài giả định đơn giản. Đầu tiên, chúng ta giả định rằng mối quan hệ giữa các biến độc lập $\mathbf{x}$ và biến phụ thuộc $y$ là tuyến tính, tức là $y$ có thể được biểu thị dưới dạng tổng trọng số của các phần tử trong $\mathbf{x}$, cho một số tiếng ồn trên các quan sát. Thứ hai, chúng tôi giả định rằng bất kỳ tiếng ồn nào cũng được cư xử tốt (theo phân phối Gaussian). 

Để thúc đẩy cách tiếp cận, chúng ta hãy bắt đầu với một ví dụ đang chạy. Giả sử rằng chúng tôi muốn ước tính giá nhà ở (tính bằng đô la) dựa trên diện tích của họ (tính bằng feet vuông) và tuổi (tính bằng năm). Để thực sự phát triển một mô hình để dự đoán giá nhà, chúng ta sẽ cần phải có được bàn tay của chúng tôi trên một bộ dữ liệu bao gồm doanh số bán hàng mà chúng tôi biết giá bán, diện tích và tuổi cho mỗi ngôi nhà. Trong thuật ngữ học máy, tập dữ liệu được gọi là tập dữ liệu đào tạo* hoặc * bộ đào tạo* và mỗi hàng (ở đây dữ liệu tương ứng với một lần bán) được gọi là *ví dụ* (hoặc điểm dữ liệu*, *dữ liệu instance*, *mẫu*). Điều chúng tôi đang cố gắng dự đoán (giá) được gọi là * nhãy* (hoặc *mục tiêu *). Các biến độc lập (tuổi và khu vực) dựa trên dự đoán được gọi là *features* (hoặc *covariates*). 

Thông thường, chúng ta sẽ sử dụng $n$ để biểu thị số ví dụ trong tập dữ liệu của chúng ta. Chúng tôi lập chỉ mục các ví dụ dữ liệu bằng $i$, biểu thị mỗi đầu vào là $\mathbf{x}^{(i)} = [x_1^{(i)}, x_2^{(i)}]^\top$ và nhãn tương ứng là $y^{(i)}$. 

### Mô hình tuyến tính
:label:`subsec_linear_model`

Giả định tuyến tính chỉ nói rằng mục tiêu (giá) có thể được biểu thị dưới dạng tổng trọng số của các tính năng (diện tích và tuổi): 

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$
:eqlabel:`eq_price-area`

Trong :eqref:`eq_price-area`, $w_{\mathrm{area}}$ và $w_{\mathrm{age}}$ được gọi là *trọng lượng*, và $b$ được gọi là *bias* (còn được gọi là *offset* hoặc *intercept*). Trọng lượng xác định ảnh hưởng của từng tính năng đối với dự đoán của chúng tôi và sự thiên vị chỉ nói giá trị mà giá dự đoán nên mất khi tất cả các tính năng có giá trị 0. Ngay cả khi chúng ta sẽ không bao giờ nhìn thấy bất kỳ ngôi nhà nào có diện tích bằng 0, hoặc chính xác là 0 năm tuổi, chúng ta vẫn cần sự thiên vị nếu không chúng ta sẽ hạn chế tính biểu cảm của mô hình của mình. Nói đúng ra, :eqref:`eq_price-area` là một biến đổi * affine* của các tính năng đầu vào, được đặc trưng bởi sự biến đổi tuyến tính* của các tính năng thông qua tổng trọng số, kết hợp với một * dịch* thông qua sự thiên vị được thêm vào. 

Với một tập dữ liệu, mục tiêu của chúng tôi là chọn trọng lượng $\mathbf{w}$ và thiên vị $b$ sao cho trung bình, các dự đoán được thực hiện theo mô hình của chúng tôi phù hợp nhất với giá thực sự quan sát thấy trong dữ liệu. Các mô hình có dự đoán đầu ra được xác định bởi sự biến đổi affine của các tính năng đầu vào là * mô hình tuyến tính*, trong đó chuyển đổi affine được chỉ định bởi các trọng lượng và thiên vị đã chọn. 

Trong các ngành mà người ta thường tập trung vào các bộ dữ liệu chỉ với một vài tính năng, thể hiện rõ ràng các mô hình dạng dài như thế này là phổ biến. Trong học máy, chúng ta thường làm việc với các bộ dữ liệu chiều cao, vì vậy thuận tiện hơn khi sử dụng ký hiệu đại số tuyến tính. Khi đầu vào của chúng tôi bao gồm các tính năng $d$, chúng tôi thể hiện dự đoán của chúng tôi $\hat{y}$ (nói chung là biểu tượng “mũ” biểu thị ước tính) như 

$$\hat{y} = w_1  x_1 + ... + w_d  x_d + b.$$

Thu thập tất cả các tính năng thành một vector $\mathbf{x} \in \mathbb{R}^d$ và tất cả các trọng lượng thành một vector $\mathbf{w} \in \mathbb{R}^d$, chúng ta có thể thể hiện mô hình của mình một cách nhỏ gọn bằng cách sử dụng một sản phẩm chấm: 

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b.$$
:eqlabel:`eq_linreg-y`

Trong :eqref:`eq_linreg-y`, vector $\mathbf{x}$ tương ứng với các tính năng của một ví dụ dữ liệu duy nhất. Chúng ta thường sẽ thấy thuận tiện khi tham khảo các tính năng của toàn bộ dữ liệu của chúng tôi $n$ ví dụ thông qua ma trận thiết kế*$\mathbf{X} \in \mathbb{R}^{n \times d}$. Ở đây, $\mathbf{X}$ chứa một hàng cho mỗi ví dụ và một cột cho mọi tính năng. 

Đối với một bộ sưu tập các tính năng $\mathbf{X}$, các dự đoán $\hat{\mathbf{y}} \in \mathbb{R}^n$ có thể được thể hiện thông qua sản phẩm ma trục-vector: 

$${\hat{\mathbf{y}}} = \mathbf{X} \mathbf{w} + b,$$

nơi phát sóng (xem :numref:`subsec_broadcasting`) được áp dụng trong quá trình tổng kết. Với các tính năng của tập dữ liệu đào tạo $\mathbf{X}$ và các nhãn tương ứng (đã biết) $\mathbf{y}$, mục tiêu của hồi quy tuyến tính là tìm vector trọng lượng $\mathbf{w}$ và thuật ngữ thiên vị $b$ đưa ra các tính năng của một ví dụ dữ liệu mới được lấy mẫu từ cùng một phân phối như $\mathbf{X}$, nhãn của ví dụ mới sẽ (trong kỳ vọng) được dự đoán với lỗi thấp nhất. 

Ngay cả khi chúng tôi tin rằng mô hình tốt nhất để dự đoán $y$ cho $\mathbf{x}$ là tuyến tính, chúng tôi sẽ không mong đợi để tìm thấy một bộ dữ liệu thế giới thực của $n$ ví dụ trong đó $y^{(i)}$ chính xác bằng $\mathbf{w}^\top \mathbf{x}^{(i)}+b$ cho tất cả $1 \leq i \leq n$. Ví dụ, bất kỳ công cụ nào chúng tôi sử dụng để quan sát các tính năng $\mathbf{X}$ và nhãn $\mathbf{y}$ có thể bị sai số đo nhỏ. Do đó, ngay cả khi chúng tôi tự tin rằng mối quan hệ cơ bản là tuyến tính, chúng tôi sẽ kết hợp một thuật ngữ tiếng ồn để tính đến các lỗi như vậy. 

Trước khi chúng ta có thể tìm kiếm các thông số* tốt nhất* (hoặc tham số *model *) $\mathbf{w}$ và $b$, chúng ta sẽ cần thêm hai điều nữa: (i) một biện pháp chất lượng cho một số mô hình nhất định; và (ii) một quy trình để cập nhật mô hình để cải thiện chất lượng của nó. 

### Chức năng mất

Trước khi chúng ta bắt đầu suy nghĩ về cách * phù hợp* dữ liệu với mô hình của chúng tôi, chúng ta cần xác định thước đo* phù hợp*. Chức năng *loss* định lượng khoảng cách giữa giá trị *real* và *dự đoán * của mục tiêu. Sự mất mát thường sẽ là một số không âm trong đó các giá trị nhỏ hơn là dự đoán tốt hơn và hoàn hảo sẽ bị mất 0. Chức năng mất phổ biến nhất trong các bài toán hồi quy là lỗi bình phương. Khi dự đoán của chúng ta cho một ví dụ $i$ là $\hat{y}^{(i)}$ và nhãn đúng tương ứng là $y^{(i)}$, lỗi bình phương được đưa ra bởi: 

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$
:eqlabel:`eq_mse`

Hằng số $\frac{1}{2}$ không tạo ra sự khác biệt thực sự nhưng sẽ chứng minh một cách rõ ràng thuận tiện, hủy bỏ khi chúng ta lấy dẫn xuất của sự mất mát. Kể từ khi tập dữ liệu đào tạo được cung cấp cho chúng tôi, và do đó ngoài tầm kiểm soát của chúng tôi, lỗi thực nghiệm chỉ là một chức năng của các tham số mô hình. Để làm cho mọi thứ cụ thể hơn, hãy xem xét ví dụ dưới đây nơi chúng ta vẽ một bài toán hồi quy cho một trường hợp một chiều như thể hiện trong :numref:`fig_fit_linreg`. 

![Fit data with a linear model.](../img/fit-linreg.svg)
:label:`fig_fit_linreg`

Lưu ý rằng sự khác biệt lớn giữa các ước tính $\hat{y}^{(i)}$ và quan sát $y^{(i)}$ dẫn đến những đóng góp thậm chí còn lớn hơn cho sự mất mát, do sự phụ thuộc bậc hai. Để đo lường chất lượng của một mô hình trên toàn bộ dữ liệu $n$ ví dụ, chúng tôi chỉ đơn giản là trung bình (hoặc tương đương, tổng hợp) các tổn thất trên bộ đào tạo. 

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

Khi đào tạo mô hình, chúng tôi muốn tìm các tham số ($\mathbf{w}^*, b^*$) giúp giảm thiểu tổng tổn thất trên tất cả các ví dụ đào tạo: 

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$

### Giải pháp phân tích

Hồi quy tuyến tính xảy ra là một vấn đề tối ưu hóa đơn giản bất thường. Không giống như hầu hết các mô hình khác mà chúng ta sẽ gặp phải trong cuốn sách này, hồi quy tuyến tính có thể được giải quyết một cách phân tích bằng cách áp dụng một công thức đơn giản. Để bắt đầu, chúng ta có thể đặt sai lệch $b$ vào tham số $\mathbf{w}$ bằng cách thêm một cột vào ma trận thiết kế bao gồm tất cả các cột. Sau đó, vấn đề dự đoán của chúng tôi là giảm thiểu $\|\mathbf{y} - \mathbf{X}\mathbf{w}\|^2$. Chỉ có một điểm quan trọng trên bề mặt mất mát và nó tương ứng với mức tối thiểu tổn thất trên toàn bộ miền. Lấy đạo hàm của tổn thất đối với $\mathbf{w}$ và đặt nó bằng 0 mang lại giải pháp phân tích (dạng đóng): 

$$\mathbf{w}^* = (\mathbf X^\top \mathbf X)^{-1}\mathbf X^\top \mathbf{y}.$$

Trong khi các vấn đề đơn giản như hồi quy tuyến tính có thể thừa nhận các giải pháp phân tích, bạn không nên quen với may mắn như vậy. Mặc dù các giải pháp phân tích cho phép phân tích toán học tốt đẹp, yêu cầu của một giải pháp phân tích rất hạn chế đến mức nó sẽ loại trừ tất cả các học sâu. 

### Minibatch Stochastic Gradient Descent

Ngay cả trong trường hợp chúng ta không thể giải quyết các mô hình một cách phân tích, hóa ra chúng ta vẫn có thể đào tạo các mô hình hiệu quả trong thực tế. Hơn nữa, đối với nhiều nhiệm vụ, những mô hình khó tối ưu hóa đó hóa ra trở nên tốt hơn nhiều đến mức tìm ra cách đào tạo chúng sẽ rất đáng để gặp rắc rối. 

Kỹ thuật chính để tối ưu hóa gần như bất kỳ mô hình học sâu nào và chúng ta sẽ kêu gọi trong suốt cuốn sách này, bao gồm việc giảm lỗi lặp đi lặp lại bằng cách cập nhật các tham số theo hướng làm giảm chức năng mất dần. Thuật toán này được gọi là *gradient descent*. 

Ứng dụng ngây thơ nhất của gradient descent bao gồm lấy đạo hàm của hàm mất, là trung bình của các tổn thất được tính toán trên mỗi ví dụ duy nhất trong tập dữ liệu. Trong thực tế, điều này có thể cực kỳ chậm: chúng ta phải vượt qua toàn bộ tập dữ liệu trước khi thực hiện một bản cập nhật duy nhất. Do đó, chúng ta thường sẽ giải quyết để lấy mẫu một minibatch ngẫu nhiên các ví dụ mỗi khi chúng ta cần tính toán bản cập nhật, một biến thể được gọi là * minibatch stochastic gradient descent*. 

Trong mỗi lần lặp lại, lần đầu tiên chúng ta lấy mẫu ngẫu nhiên một minibatch $\mathcal{B}$ bao gồm một số ví dụ đào tạo cố định. Sau đó chúng ta tính toán đạo hàm (gradient) của tổn thất trung bình trên minibatch liên quan đến các tham số mô hình. Cuối cùng, chúng ta nhân gradient với giá trị dương được xác định trước $\eta$ và trừ thuật ngữ kết quả từ các giá trị tham số hiện tại. 

Chúng ta có thể thể hiện bản cập nhật về mặt toán học như sau ($\partial$ biểu thị đạo hàm một phần): 

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b).$$

Tóm lại, các bước của thuật toán như sau: (i) chúng ta khởi tạo các giá trị của các tham số mô hình, thường là ngẫu nhiên; (ii) chúng ta lặp lại lấy mẫu minibatches ngẫu nhiên từ dữ liệu, cập nhật các tham số theo hướng của gradient âm. Đối với tổn thất bậc hai và biến đổi affine, chúng ta có thể viết ra điều này một cách rõ ràng như sau: 

$$\begin{aligned} \mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) = \mathbf{w} - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\ b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  = b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right). \end{aligned}$$
:eqlabel:`eq_linreg_batch_update`

Lưu ý rằng $\mathbf{w}$ và $\mathbf{x}$ là vectơ trong :eqref:`eq_linreg_batch_update`. Ở đây, ký hiệu vector thanh lịch hơn làm cho toán học dễ đọc hơn nhiều so với thể hiện mọi thứ về hệ số, nói $w_1, w_2, \ldots, w_d$. Tập cardinality $|\mathcal{B}|$ đại diện cho số lượng ví dụ trong mỗi minibatch (* batch size*) và $\eta$ biểu thị tỷ lệ học tập *. Chúng tôi nhấn mạnh rằng các giá trị của quy mô lô và tốc độ học tập được chỉ định trước bằng tay và thường không được học thông qua đào tạo mô hình. Các tham số này có thể điều chỉnh nhưng không được cập nhật trong vòng đào tạo được gọi là *hyperparameters*.
*Điều chỉnh siêu tham số* là quá trình mà các siêu tham số được chọn,
và thường yêu cầu chúng tôi điều chỉnh chúng dựa trên kết quả của vòng lặp đào tạo như được đánh giá trên một bộ dữ liệu xác thực * riêng biệt* (hoặc bộ xác thực *). 

Sau khi đào tạo cho một số lần lặp lại được xác định trước (hoặc cho đến khi đáp ứng một số tiêu chí dừng khác), chúng tôi ghi lại các tham số mô hình ước tính, ký hiệu là $\hat{\mathbf{w}}, \hat{b}$. Lưu ý rằng ngay cả khi chức năng của chúng ta thực sự tuyến tính và không ồn ào, các tham số này sẽ không phải là bộ giảm thiểu chính xác của sự mất mát bởi vì, mặc dù thuật toán hội tụ chậm về phía các bộ giảm thiểu, nó không thể đạt được chính xác trong một số bước hữu hạn. 

Hồi quy tuyến tính xảy ra là một vấn đề học tập mà chỉ có một mức tối thiểu so với toàn bộ miền. Tuy nhiên, đối với các mô hình phức tạp hơn, như mạng sâu, các bề mặt mất mát chứa nhiều minima. May mắn thay, vì những lý do chưa được hiểu đầy đủ, các học viên học sâu hiếm khi đấu tranh để tìm ra các thông số giảm thiểu tổn thất * trong bộ đào tạo*. Nhiệm vụ ghê gớm hơn là tìm các tham số sẽ đạt được tổn thất thấp đối với dữ liệu mà chúng ta chưa thấy trước đây, một thách thức được gọi là *generalization*. Chúng tôi quay trở lại các chủ đề này trong suốt cuốn sách. 

### Đưa ra dự đoán với mô hình đã học

Với mô hình hồi quy tuyến tính đã học $\hat{\mathbf{w}}^\top \mathbf{x} + \hat{b}$, bây giờ chúng ta có thể ước tính giá của một ngôi nhà mới (không chứa trong dữ liệu đào tạo) cho khu vực của nó $x_1$ và $x_2$ tuổi. Ước tính các mục tiêu được đưa ra các tính năng thường được gọi là *prediction* hoặc *inference*. 

Chúng tôi sẽ cố gắng gắn bó với *prediction* vì gọi bước này* suy tiế*, mặc dù nổi lên như là thuật ngữ tiêu chuẩn trong học sâu, nhưng có phần là một sự sai lầm. Trong thống kê, *inference* thường biểu thị các tham số ước tính dựa trên tập dữ liệu. Việc sử dụng sai thuật ngữ này là một nguồn gây nhầm lẫn phổ biến khi các học viên học sâu nói chuyện với các nhà thống kê. 

## Vectorization cho tốc độ

Khi đào tạo các mô hình của chúng tôi, chúng tôi thường muốn xử lý toàn bộ minibatches ví dụ cùng một lúc. Làm điều này một cách hiệu quả đòi hỏi rằng (**we**) (~~should~~) (** vectorize các phép tính và tận dụng các thư viện đại số tuyến tính nhanh hơn là viết tốn kém cho - vòng lặp trong Python.**)

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import np
import time
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
import numpy as np
import time
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
import numpy as np
import time
```

Để minh họa lý do tại sao điều này quan trọng rất nhiều, chúng ta có thể (** xem xét hai phương pháp để thêm vectors.**) Để bắt đầu chúng ta khởi tạo hai vectơ 10000 chiều chứa tất cả các vectơ. Trong một phương pháp, chúng ta sẽ lặp lại các vectơ bằng Python for-loop. Trong phương pháp khác, chúng tôi sẽ dựa vào một cuộc gọi duy nhất đến `+`.

```{.python .input}
#@tab all
n = 10000
a = d2l.ones(n)
b = d2l.ones(n)
```

Vì chúng ta sẽ chuẩn hóa thời gian chạy thường xuyên trong cuốn sách này, [** hãy để chúng tôi định nghĩa một timer**].

```{.python .input}
#@tab all
class Timer:  #@save
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()
```

Bây giờ chúng ta có thể chuẩn khối lượng công việc. Đầu tiên, [**chúng tôi thêm chúng, một tọa độ tại một thời điểm, sử dụng một vòng lặp cho.**]

```{.python .input}
#@tab mxnet, pytorch
c = d2l.zeros(n)
timer = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
f'{timer.stop():.5f} sec'
```

```{.python .input}
#@tab tensorflow
c = tf.Variable(d2l.zeros(n))
timer = Timer()
for i in range(n):
    c[i].assign(a[i] + b[i])
f'{timer.stop():.5f} sec'
```

(**Ngoài ra, chúng tôi dựa vào toán tử `+` nạp lại để tính toán tổng elementwise**)

```{.python .input}
#@tab all
timer.start()
d = a + b
f'{timer.stop():.5f} sec'
```

Bạn có thể nhận thấy rằng phương pháp thứ hai nhanh hơn đáng kể so với phương pháp đầu tiên. Mã vector hóa thường mang lại tốc độ lớn theo thứ tự. Hơn nữa, chúng tôi đẩy nhiều toán học vào thư viện và không cần phải tự viết nhiều tính toán, giảm khả năng xảy ra lỗi. 

## Phân phối bình thường và tổn thất bình phương
:label:`subsec_normal_distribution_and_squared_loss`

Mặc dù bạn đã có thể làm bẩn tay chỉ bằng cách sử dụng thông tin ở trên, nhưng sau đây chúng ta có thể chính thức thúc đẩy mục tiêu mất bình phương thông qua các giả định về sự phân bố tiếng ồn. 

Hồi quy tuyến tính được phát minh bởi Gauss vào năm 1795, người cũng phát hiện ra sự phân bố bình thường (còn gọi là *Gaussian*). Nó chỉ ra rằng kết nối giữa phân bố bình thường và hồi quy tuyến tính chạy sâu hơn cha mẹ chung. Để làm mới bộ nhớ của bạn, mật độ xác suất của một phân phối bình thường với trung bình $\mu$ và phương sai $\sigma^2$ (độ lệch chuẩn $\sigma$) được đưa ra như 

$$p(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (x - \mu)^2\right).$$

Bên dưới [**chúng tôi định nghĩa một hàm Python để tính toán bản phân phối bình thường**].

```{.python .input}
#@tab all
def normal(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)
```

Bây giờ chúng ta có thể (** hình dung các bản phân phối bình thường**).

```{.python .input}
#@tab mxnet
# Use numpy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x.asnumpy(), [normal(x, mu, sigma).asnumpy() for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

```{.python .input}
#@tab pytorch, tensorflow
# Use numpy again for visualization
x = np.arange(-7, 7, 0.01)

# Mean and standard deviation pairs
params = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
```

Như chúng ta có thể thấy, việc thay đổi trung bình tương ứng với sự dịch chuyển dọc theo trục $x$-và tăng phương sai lan rộng phân phối ra, hạ thấp đỉnh của nó. 

Một cách để thúc đẩy hồi quy tuyến tính với hàm mất lỗi bình phương trung bình (hoặc đơn giản là mất bình phương) là chính thức cho rằng các quan sát phát sinh từ các quan sát ồn ào, trong đó tiếng ồn thường được phân phối như sau: 

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Do đó, bây giờ chúng ta có thể viết ra * likelihood* của việc nhìn thấy một $y$ cụ thể cho $\mathbf{x}$ qua 

$$P(y \mid \mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right).$$

Bây giờ, theo nguyên tắc khả năng tối đa, các giá trị tốt nhất của các tham số $\mathbf{w}$ và $b$ là những giá trị tối đa hóa * likelihood* của toàn bộ dữ liệu: 

$$P(\mathbf y \mid \mathbf X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)}).$$

Ước tính được lựa chọn theo nguyên tắc khả năng tối đa được gọi là * dự kiến khả năng tối đa*. Trong khi, tối đa hóa sản phẩm của nhiều hàm mũ, có thể trông khó khăn, chúng ta có thể đơn giản hóa mọi thứ đáng kể, mà không thay đổi mục tiêu, bằng cách tối đa hóa nhật ký của khả năng thay thế. Vì lý do lịch sử, tối ưu hóa thường được thể hiện dưới dạng giảm thiểu hơn là tối đa hóa. Vì vậy, mà không thay đổi bất cứ điều gì chúng ta có thể giảm thiểu * âm log-likelihood* $-\log P(\mathbf y \mid \mathbf X)$. Làm việc ra toán học cho chúng ta: 

$$-\log P(\mathbf y \mid \mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2.$$

Bây giờ chúng ta chỉ cần thêm một giả định rằng $\sigma$ là một số hằng số cố định. Do đó chúng ta có thể bỏ qua thuật ngữ đầu tiên vì nó không phụ thuộc vào $\mathbf{w}$ hoặc $b$. Bây giờ thuật ngữ thứ hai giống hệt với tổn thất lỗi bình phương được giới thiệu trước đó, ngoại trừ hằng số nhân $\frac{1}{\sigma^2}$. May mắn thay, giải pháp không phụ thuộc vào $\sigma$. Theo đó, việc giảm thiểu sai số bình phương trung bình tương đương với ước tính khả năng tối đa của một mô hình tuyến tính theo giả định của tiếng ồn Gaussian phụ gia. 

## Từ hồi quy tuyến tính đến mạng sâu

Cho đến nay chúng ta chỉ nói về các mô hình tuyến tính. Trong khi các mạng thần kinh bao gồm một họ mô hình phong phú hơn nhiều, chúng ta có thể bắt đầu nghĩ về mô hình tuyến tính như một mạng thần kinh bằng cách thể hiện nó bằng ngôn ngữ của mạng thần kinh. Để bắt đầu, chúng ta hãy bắt đầu bằng cách viết lại mọi thứ trong một ký hiệu “lớp”. 

### Sơ đồ mạng thần kinh

Các học viên học sâu thích vẽ sơ đồ để hình dung những gì đang xảy ra trong mô hình của họ. Trong :numref:`fig_single_neuron`, chúng tôi mô tả mô hình hồi quy tuyến tính của chúng tôi như một mạng thần kinh. Lưu ý rằng các sơ đồ này làm nổi bật mô hình kết nối như cách mỗi đầu vào được kết nối với đầu ra, nhưng không phải các giá trị được thực hiện bởi trọng lượng hoặc thành kiến. 

![Linear regression is a single-layer neural network.](../img/singleneuron.svg)
:label:`fig_single_neuron`

Đối với mạng nơ-ron được hiển thị trong :numref:`fig_single_neuron`, các đầu vào là $x_1, \ldots, x_d$, do đó, *số đầu vào* (hoặc *tính năng dimensionality*) trong lớp đầu vào là $d$. Đầu ra của mạng trong :numref:`fig_single_neuron` là $o_1$, do đó * số đầu ra* trong lớp đầu ra là 1. Lưu ý rằng các giá trị đầu vào là tất cả * given* và chỉ có một tế bào thần kinh * computed* duy nhất. Tập trung vào nơi tính toán diễn ra, thông thường chúng ta không xem xét lớp đầu vào khi đếm các lớp. Điều đó có nghĩa là, * số lớp* cho mạng thần kinh trong :numref:`fig_single_neuron` là 1. Chúng ta có thể nghĩ về các mô hình hồi quy tuyến tính như các mạng thần kinh chỉ bao gồm một tế bào thần kinh nhân tạo duy nhất, hoặc như các mạng thần kinh một lớp. 

Vì đối với hồi quy tuyến tính, mọi đầu vào được kết nối với mọi đầu ra (trong trường hợp này chỉ có một đầu ra), chúng ta có thể coi chuyển đổi này (lớp đầu ra trong :numref:`fig_single_neuron`) như một lớp * kết nối đầy đủ* hoặc * lớp dày dài*. Chúng ta sẽ nói nhiều hơn về các mạng bao gồm các lớp như vậy trong chương tiếp theo. 

### Sinh học

Kể từ khi hồi quy tuyến tính (được phát minh vào năm 1795) có trước khoa học thần kinh tính toán, có vẻ như dị ứng để mô tả hồi quy tuyến tính như một mạng lưới thần kinh. Để xem tại sao các mô hình tuyến tính là một nơi tự nhiên để bắt đầu khi các nhà mạng mạch/nhà sinh lý thần kinh Warren McCulloch và Walter Pitt bắt đầu phát triển các mô hình tế bào thần kinh nhân tạo, hãy xem xét hình ảnh hoạt hình của một tế bào thần kinh sinh học trong :numref:`fig_Neuron`, bao gồm
*dendrites* (thiết bị đầu cuối đầu vào),
*nucleus* (CPU), * axon* (dây đầu ra) và thiết bị đầu cuối * axon* (thiết bị đầu cuối đầu ra), cho phép kết nối với các tế bào thần kinh khác thông qua * synapses*. 

![The real neuron.](../img/neuron.svg)
:label:`fig_Neuron`

Thông tin $x_i$ đến từ các tế bào thần kinh khác (hoặc cảm biến môi trường như võng mạc) được nhận trong các dendrites. Đặc biệt, thông tin đó được cân bằng * synaptic trọng lượng* $w_i$ xác định ảnh hưởng của các đầu vào (ví dụ: kích hoạt hoặc ức chế thông qua sản phẩm $x_i w_i$). Các đầu vào có trọng số đến từ nhiều nguồn được tổng hợp trong hạt nhân dưới dạng tổng trọng số $y = \sum_i x_i w_i + b$ và thông tin này sau đó được gửi để xử lý thêm trong axon $y$, thường là sau một số xử lý phi tuyến thông qua $\sigma(y)$. Từ đó nó hoặc đến đích của nó (ví dụ, một cơ bắp) hoặc được đưa vào một tế bào thần kinh khác thông qua các dendrites của nó. 

Chắc chắn, ý tưởng cấp cao rằng nhiều đơn vị như vậy có thể được rải sỏi cùng với kết nối phù hợp và thuật toán học tập đúng đắn, để tạo ra hành vi thú vị và phức tạp hơn nhiều so với bất kỳ tế bào thần kinh nào một mình có thể thể hiện nợ nghiên cứu của chúng tôi về các hệ thống thần kinh sinh học thực sự. 

Đồng thời, hầu hết các nghiên cứu trong học sâu ngày nay thu hút rất ít cảm hứng trực tiếp trong khoa học thần kinh. Chúng tôi gọi Stuart Russell và Peter Norvig, người trong cuốn sách giáo khoa AI cổ điển của họ
*Artificial Intelligence: A Modern Approach* :cite:`Russell.Norvig.2016`,
chỉ ra rằng mặc dù máy bay có thể đã được truyền cảm hứng * bởi các loài chim, điểu học đã không phải là động lực chính của đổi mới hàng không trong một số thế kỷ. Tương tự như vậy, cảm hứng trong học sâu những ngày này có thước đo bằng hoặc lớn hơn từ toán học, thống kê và khoa học máy tính. 

## Tóm tắt

* Các thành phần chính trong mô hình học máy là dữ liệu đào tạo, chức năng mất mát, thuật toán tối ưu hóa và khá rõ ràng là bản thân mô hình.
* Vector hóa làm cho mọi thứ tốt hơn (chủ yếu là toán học) và nhanh hơn (chủ yếu là mã).
* Giảm thiểu một chức năng khách quan và thực hiện ước tính khả năng tối đa có thể có nghĩa là điều tương tự.
* Các mô hình hồi quy tuyến tính cũng là mạng thần kinh.

## Bài tập

1. Giả sử rằng chúng ta có một số dữ liệu $x_1, \ldots, x_n \in \mathbb{R}$. Mục tiêu của chúng tôi là tìm một hằng số $b$ sao cho $\sum_i (x_i - b)^2$ được giảm thiểu.
    1. Tìm một giải pháp phân tích cho giá trị tối ưu là $b$.
    1. Làm thế nào để vấn đề này và giải pháp của nó liên quan đến phân phối bình thường?
1. Lấy được giải pháp phân tích cho bài toán tối ưu hóa cho hồi quy tuyến tính với lỗi bình phương. Để giữ cho mọi thứ đơn giản, bạn có thể bỏ qua sự thiên vị $b$ khỏi vấn đề (chúng ta có thể làm điều này theo kiểu nguyên tắc bằng cách thêm một cột vào $\mathbf X$ bao gồm tất cả các cột).
    1. Viết ra bài toán tối ưu hóa trong ký hiệu ma trận và vector (coi tất cả dữ liệu như một ma trận duy nhất, và tất cả các giá trị đích như một vectơ duy nhất).
    1. Tính toán độ dốc của sự mất mát đối với $w$.
    1. Tìm giải pháp phân tích bằng cách đặt gradient bằng 0 và giải phương trình ma trận.
    1. Khi nào điều này có thể tốt hơn so với sử dụng stochastic gradient descent? Khi nào phương pháp này có thể phá vỡ?
1. Giả sử rằng mô hình tiếng ồn điều chỉnh tiếng ồn phụ gia $\epsilon$ là phân phối theo cấp số nhân. Đó là, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    1. Viết ra khả năng log âm của dữ liệu theo mô hình $-\log P(\mathbf y \mid \mathbf X)$.
    1. Bạn có thể tìm thấy một giải pháp hình thức khép kín?
    1. Đề xuất một thuật toán gốc gradient ngẫu nhiên để giải quyết vấn đề này. Điều gì có thể xảy ra sai (gợi ý: điều gì xảy ra gần điểm cố định khi chúng ta tiếp tục cập nhật các tham số)? Bạn có thể sửa chữa điều này?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/40)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/258)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/259)
:end_tab:

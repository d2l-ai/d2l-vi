# Thao tác dữ liệu
:label:`sec_ndarray`

Để hoàn thành bất cứ điều gì, chúng ta cần một số cách để lưu trữ và thao tác dữ liệu. Nói chung, có hai điều quan trọng chúng ta cần làm với dữ liệu: (i) có được chúng; và (ii) xử lý chúng một khi chúng ở trong máy tính. Không có điểm nào trong việc thu thập dữ liệu mà không cần một cách nào đó để lưu trữ nó, vì vậy hãy để chúng tôi làm bẩn tay trước bằng cách chơi với dữ liệu tổng hợp. Để bắt đầu, chúng tôi giới thiệu mảng $n$ chiều, còn được gọi là * tensor*. 

Nếu bạn đã làm việc với NumPy, gói máy tính khoa học được sử dụng rộng rãi nhất trong Python, thì bạn sẽ thấy phần này quen thuộc. Cho dù bạn sử dụng khung nào, lớp tensor* của nó* (`ndarray` trong MXNet, `Tensor` trong cả PyTorch và TensorFlow) tương tự như `ndarray` của NumPy với một vài tính năng giết người. Đầu tiên, GPU được hỗ trợ tốt để tăng tốc tính toán trong khi NumPy chỉ hỗ trợ tính toán CPU. Thứ hai, lớp tensor hỗ trợ sự khác biệt tự động. Những tính chất này làm cho lớp tensor phù hợp với học sâu. Trong suốt cuốn sách, khi chúng tôi nói hàng chục, chúng tôi đang đề cập đến các trường hợp của lớp tensor trừ khi có quy định khác. 

## Bắt đầu

Trong phần này, chúng tôi mong muốn giúp bạn hoạt động, trang bị cho bạn các công cụ toán học và tính toán số cơ bản mà bạn sẽ xây dựng khi bạn tiến bộ qua cuốn sách. Đừng lo lắng nếu bạn đấu tranh để grok một số khái niệm toán học hoặc chức năng thư viện. Các phần sau đây sẽ xem xét lại tài liệu này trong bối cảnh các ví dụ thực tế và nó sẽ chìm vào. Mặt khác, nếu bạn đã có một số nền tảng và muốn đi sâu hơn vào nội dung toán học, chỉ cần bỏ qua phần này.

:begin_tab:`mxnet`
Để bắt đầu, chúng tôi nhập các mô-đun `np` (`numpy`) và `npx` (`numpy_extension`) từ MXNet. Ở đây, mô-đun `np` bao gồm các chức năng được hỗ trợ bởi NumPy, trong khi mô-đun `npx` chứa một tập hợp các phần mở rộng được phát triển để trao quyền cho việc học sâu trong một môi trường giống như NumPy-like. Khi sử dụng hàng chục, chúng tôi hầu như luôn gọi hàm `set_np`: đây là để tương thích xử lý tensor bởi các thành phần khác của MXNet.
:end_tab:

:begin_tab:`pytorch`
(**Để bắt đầu, chúng tôi nhập `torch`. Lưu ý rằng mặc dù nó được gọi là PyTorch, chúng ta nên nhập `torch` thay vì `pytorch`.**)
:end_tab:

:begin_tab:`tensorflow`
Để bắt đầu, chúng tôi nhập `tensorflow`. Vì tên dài một chút, chúng ta thường nhập nó với một bí danh ngắn `tf`.
:end_tab:

```{.python .input}
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
import torch
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
```

[**A tensor đại diện cho một mảng (có thể đa chiều) của các giá trị số.**] Với một trục, một tensor được gọi là *vector*. Với hai trục, một tensor được gọi là *matri*. Với $k > 2$ trục, chúng tôi thả các tên chuyên biệt và chỉ cần tham khảo đối tượng dưới dạng $k^\mathrm{th}$ *căng đơn hàng*.

:begin_tab:`mxnet`
MXNet cung cấp một loạt các chức năng để tạo ra các hàng chục mới prepopulated với các giá trị. Ví dụ, bằng cách gọi `arange(n)`, chúng ta có thể tạo một vectơ có giá trị cách đều nhau, bắt đầu từ 0 (bao gồm) và kết thúc ở `n` (không bao gồm). Theo mặc định, kích thước khoảng thời gian là $1$. Trừ khi có quy định khác, hàng chục mới được lưu trữ trong bộ nhớ chính và được chỉ định cho tính toán dựa trên CPU.
:end_tab:

:begin_tab:`pytorch`
PyTorch cung cấp một loạt các chức năng để tạo ra các hàng chục mới prepopulated với các giá trị. Ví dụ, bằng cách gọi `arange(n)`, chúng ta có thể tạo một vectơ có giá trị cách đều nhau, bắt đầu từ 0 (bao gồm) và kết thúc ở `n` (không bao gồm). Theo mặc định, kích thước khoảng thời gian là $1$. Trừ khi có quy định khác, hàng chục mới được lưu trữ trong bộ nhớ chính và được chỉ định cho tính toán dựa trên CPU.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow cung cấp một loạt các chức năng để tạo ra các hàng chục mới prepopulated với các giá trị. Ví dụ, bằng cách gọi `range(n)`, chúng ta có thể tạo một vectơ có giá trị cách đều nhau, bắt đầu từ 0 (bao gồm) và kết thúc ở `n` (không bao gồm). Theo mặc định, kích thước khoảng thời gian là $1$. Trừ khi có quy định khác, hàng chục mới được lưu trữ trong bộ nhớ chính và được chỉ định cho tính toán dựa trên CPU.
:end_tab:

```{.python .input}
x = np.arange(12)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

(**Chúng tôi có thể truy cập một tensor*shape***) (~~ và tổng số phần tử ~~) (chiều dài dọc theo mỗi trục) bằng cách kiểm tra thuộc tính `shape` của nó.

```{.python .input}
#@tab all
x.shape
```

Nếu chúng ta chỉ muốn biết tổng số phần tử trong một tensor, tức là, sản phẩm của tất cả các yếu tố hình dạng, chúng ta có thể kiểm tra kích thước của nó. Bởi vì chúng ta đang xử lý một vector ở đây, phần tử duy nhất của `shape` của nó giống hệt với kích thước của nó.

```{.python .input}
x.size
```

```{.python .input}
#@tab pytorch
x.numel()
```

```{.python .input}
#@tab tensorflow
tf.size(x)
```

Để [**thay đổi hình dạng của tensor mà không thay đổi số phần tử hoặc giá trị của chúng **], chúng ta có thể gọi hàm `reshape`. Ví dụ, chúng ta có thể biến đổi tensor của mình, `x`, từ một vectơ hàng có hình dạng (12,) thành ma trận có hình dạng (3, 4). Tensor mới này chứa các giá trị chính xác giống nhau, nhưng xem chúng như một ma trận được tổ chức thành 3 hàng và 4 cột. Để nhắc lại, mặc dù hình dạng đã thay đổi, các yếu tố không có. Lưu ý rằng kích thước không bị thay đổi bằng cách định hình lại.

```{.python .input}
#@tab mxnet, pytorch
X = x.reshape(3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Định hình lại bằng cách chỉ định thủ công mọi chiều là không cần thiết. Nếu hình dạng mục tiêu của chúng ta là một ma trận có hình dạng (chiều cao, chiều rộng), thì sau khi chúng ta biết chiều rộng, chiều cao được đưa ra ngầm. Tại sao chúng ta phải tự thực hiện việc phân chia? Trong ví dụ trên, để có được một ma trận với 3 hàng, chúng tôi đã chỉ định cả hai rằng nó phải có 3 hàng và 4 cột. May mắn thay, hàng chục có thể tự động làm việc ra một chiều cho phần còn lại. Chúng tôi gọi khả năng này bằng cách đặt `-1` cho kích thước mà chúng tôi muốn hàng chục tự động suy ra. Trong trường hợp của chúng tôi, thay vì gọi `x.reshape(3, 4)`, chúng tôi có thể gọi tương đương `x.reshape(-1, 4)` hoặc `x.reshape(3, -1)`. 

Thông thường, chúng ta sẽ muốn ma trận của chúng tôi khởi tạo hoặc với số không, một số hằng số khác, hoặc số được lấy mẫu ngẫu nhiên từ một phân phối cụ thể. [**Chúng ta có thể tạo một tensor đại diện cho một tensor với tất cả các phần tử được đặt là 0**](~~hoặc 1~~) và một hình dạng của (2, 3, 4) như sau:

```{.python .input}
np.zeros((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.zeros((2, 3, 4))
```

Tương tự, ta có thể tạo tenors với mỗi phần tử đặt thành 1 như sau:

```{.python .input}
np.ones((2, 3, 4))
```

```{.python .input}
#@tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
#@tab tensorflow
tf.ones((2, 3, 4))
```

Thông thường, chúng ta muốn [** lấy mẫu ngẫu nhiên các giá trị cho mỗi phần tử trong một tensor**] từ một số phân phối xác suất. Ví dụ, khi chúng ta xây dựng mảng để đóng vai trò là tham số trong mạng thần kinh, chúng ta thường sẽ khởi tạo các giá trị của chúng một cách ngẫu nhiên. Đoạn mã sau đây tạo ra một tensor với hình dạng (3, 4). Mỗi phần tử của nó được lấy mẫu ngẫu nhiên từ một phân bố Gaussian (bình thường) chuẩn với trung bình 0 và độ lệch chuẩn là 1.

```{.python .input}
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
#@tab pytorch
torch.randn(3, 4)
```

```{.python .input}
#@tab tensorflow
tf.random.normal(shape=[3, 4])
```

Chúng ta cũng có thể [** chỉ định các giá trị chính xác cho mỗi element**] trong tensor mong muốn bằng cách cung cấp một danh sách Python (hoặc danh sách các danh sách) chứa các giá trị số. Ở đây, danh sách ngoài cùng tương ứng với trục 0 và danh sách bên trong đến trục 1.

```{.python .input}
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
#@tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Hoạt động

Cuốn sách này không phải là về kỹ thuật phần mềm. Sở thích của chúng tôi không giới hạn chỉ đơn giản là đọc và viết dữ liệu từ/đến mảng. Chúng tôi muốn thực hiện các hoạt động toán học trên những mảng. Một số thao tác đơn giản và hữu ích nhất là các hoạt động *elementwise*. Chúng áp dụng một phép toán vô hướng tiêu chuẩn cho mỗi phần tử của một mảng. Đối với các hàm lấy hai mảng làm đầu vào, các phép toán elementwise áp dụng một số toán tử nhị phân chuẩn trên mỗi cặp phần tử tương ứng từ hai mảng. Chúng ta có thể tạo ra một hàm elementwise từ bất kỳ hàm nào mà bản đồ từ vô hướng đến vô hướng. 

Trong ký hiệu toán học, chúng ta sẽ biểu thị một toán tử vô hướng * unary* như vậy (lấy một đầu vào) bằng chữ ký $f: \mathbb{R} \rightarrow \mathbb{R}$. Điều này chỉ có nghĩa là chức năng đang ánh xạ từ bất kỳ số thực nào ($\mathbb{R}$) lên một số khác. Tương tự như vậy, chúng tôi biểu thị một toán tử vô hướng * binary* (lấy hai đầu vào thực và mang lại một đầu ra) bằng chữ ký $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$. Với bất kỳ hai vectơ $\mathbf{u}$ và $\mathbf{v}$ * có cùng hình dạng*, và một toán tử nhị phân $f$, chúng ta có thể tạo ra một vector $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ bằng cách đặt $c_i \gets f(u_i, v_i)$ cho tất cả $i$, trong đó $c_i, u_i$, và $v_i$ là các yếu tố $i^\mathrm{th}$ của vectơ $\mathbf{c}, \mathbf{u}$, và $v_i$ 25. Ở đây, chúng tôi đã tạo ra các vectơ có giá trị $F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$ bằng *lifting* hàm vô hướng cho một hoạt động vectơ elementwise. 

Các toán tử số học tiêu chuẩn phổ biến (`+`, `-`, `*`, `/` và `**`) đều được nâng lên * để hoạt động yếu tố cho bất kỳ hàng chục hình dạng giống hệt nhau nào có hình dạng tùy ý. Chúng ta có thể gọi các hoạt động elementwise trên bất kỳ hai hàng chục có cùng hình dạng. Trong ví dụ sau, chúng ta sử dụng dấu phẩy để xây dựng một tuple 5 phần tử, trong đó mỗi phần tử là kết quả của một phép toán elementwise. 

### Hoạt động

[**Các toán tử số học tiêu chuẩn phổ biến (`+`, `-`, `*`, `/` và `**`) đều đã được nâng lên * thành các hoạt động elementwise**]

```{.python .input}
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # The ** operator is exponentiation
```

Nhiều (** nhiều thao tác hơn có thể được áp dụng elementwise**), bao gồm các toán tử đơn nguyên như hàm mũ.

```{.python .input}
np.exp(x)
```

```{.python .input}
#@tab pytorch
torch.exp(x)
```

```{.python .input}
#@tab tensorflow
tf.exp(x)
```

Ngoài tính toán elementwise, chúng ta cũng có thể thực hiện các phép toán đại số tuyến tính, bao gồm các sản phẩm chấm vector và phép nhân ma trận. Chúng tôi sẽ giải thích các bit quan trọng của đại số tuyến tính (không có kiến thức nào được giả định trước) trong :numref:`sec_linear-algebra`. 

Chúng ta cũng có thể [*** nối nhiều chục với nhau, **] xếp chúng từ đầu đến cuối để tạo thành một tensor lớn hơn. Chúng ta chỉ cần cung cấp một danh sách các hàng chục và nói với hệ thống dọc theo trục nào để nối. Ví dụ dưới đây cho thấy những gì xảy ra khi chúng ta nối hai ma trận dọc theo các hàng (trục 0, phần tử đầu tiên của hình dạng) so với các cột (trục 1, phần tử thứ hai của hình dạng). Chúng ta có thể thấy rằng độ dài trục 0 của tensor đầu ra đầu tiên ($6$) là tổng của chiều dài trục 0 của hai căng đầu vào ($3 + 3$); trong khi độ dài trục của bộ căng đầu ra thứ hai ($8$) là tổng của chiều dài trục 1 của hai căng đầu vào ($4 + 4$).

```{.python .input}
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
#@tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

Đôi khi, chúng ta muốn [** xây dựng một tensor nhị phân thông qua *trạng thái logic *.**] Lấy `X == Y` làm ví dụ. Đối với mỗi vị trí, nếu `X` và `Y` bằng nhau tại vị trí đó thì mục nhập tương ứng trong tensor mới lấy giá trị là 1, nghĩa là câu lệnh logic `X == Y` đúng tại vị trí đó; nếu không vị trí đó mất 0.

```{.python .input}
#@tab all
X == Y
```

[** Tổng hợp tất cả các yếu tố trong tensor**] mang lại một tensor chỉ với một phần tử.

```{.python .input}
#@tab mxnet, pytorch
X.sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(X)
```

## Cơ chế phát sóng
:label:`subsec_broadcasting`

Trong phần trên, chúng ta đã thấy cách thực hiện các hoạt động elementwise trên hai hàng chục có cùng hình dạng. Trong một số điều kiện nhất định, ngay cả khi hình dạng khác nhau, chúng ta vẫn có thể [** thực hiện các hoạt động elementwise bằng cách gọi cơ chế phát sóng *.**] Cơ chế này hoạt động theo cách sau: Đầu tiên, mở rộng một hoặc cả hai mảng bằng cách sao chép các phần tử một cách thích hợp để sau khi chuyển đổi này, hai hàng chục có cùng một hình dạng. Thứ hai, thực hiện các hoạt động elementwise trên các mảng kết quả. 

Trong hầu hết các trường hợp, chúng tôi phát dọc theo một trục mà một mảng ban đầu chỉ có chiều dài 1, chẳng hạn như trong ví dụ sau:

```{.python .input}
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
#@tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
#@tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

Vì `a` và `b` tương ứng là $3\times1$ và $1\times2$ ma trận, hình dạng của chúng không khớp nếu chúng ta muốn thêm chúng. Chúng tôi * phát sóng * các mục của cả hai ma trận thành ma trận $3\times2$ lớn hơn như sau: đối với ma trận `a`, nó sao chép các cột và cho ma trận `b` nó sao chép các hàng trước khi thêm cả hai elementwise.

```{.python .input}
#@tab all
a + b
```

## Lập chỉ mục và cắt

Cũng giống như trong bất kỳ mảng Python nào khác, các phần tử trong một tensor có thể được truy cập bằng chỉ mục. Như trong bất kỳ mảng Python nào, phần tử đầu tiên có chỉ số 0 và phạm vi được chỉ định để bao gồm phần tử đầu tiên nhưng * trước* phần tử cuối cùng. Như trong danh sách Python tiêu chuẩn, chúng ta có thể truy cập các phần tử theo vị trí tương đối của chúng đến cuối danh sách bằng cách sử dụng các chỉ số âm. 

Do đó, [**`[-1]` chọn phần tử cuối cùng và `[1:3]` chọn phần tử thứ hai và thứ ba**] như sau:

```{.python .input}
#@tab all
X[-1], X[1:3]
```

:begin_tab:`mxnet, pytorch`
Ngoài việc đọc, (** chúng ta cũng có thể viết các phần tử của ma trận bằng cách chỉ định các chỉ số.**)
:end_tab:

:begin_tab:`tensorflow`
`Tensors` trong TensorFlow là bất biến và không thể được gán cho. `Variables` trong TensorFlow là các thùng chứa có thể thay đổi của trạng thái hỗ trợ các bài tập. Hãy nhớ rằng độ dốc trong TensorFlow không chảy ngược qua các bài tập `Variable`. 

Ngoài việc gán một giá trị cho toàn bộ `Variable`, chúng ta có thể viết các phần tử của `Variable` bằng cách chỉ định các chỉ số.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
X[1, 2] = 9
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

Nếu chúng ta muốn [** để gán nhiều phần tử cùng một giá trị, chúng ta chỉ cần lập chỉ mục tất cả chúng và sau đó gán cho họ giá trị.**] Ví dụ, `[0:2, :]` truy cập các hàng đầu tiên và thứ hai, trong đó `:` lấy tất cả các phần tử dọc theo trục 1 (cột). Trong khi chúng tôi thảo luận về lập chỉ mục cho ma trận, điều này rõ ràng cũng hoạt động cho vectơ và cho hàng chục hơn 2 chiều.

```{.python .input}
#@tab mxnet, pytorch
X[0:2, :] = 12
X
```

```{.python .input}
#@tab tensorflow
X_var = tf.Variable(X)
X_var[0:2, :].assign(tf.ones(X_var[0:2,:].shape, dtype = tf.float32) * 12)
X_var
```

## Tiết kiệm bộ nhớ

[**Các thao tác chạy có thể khiến bộ nhớ mới được phân bổ cho kết quả máy chủ**] Ví dụ, nếu chúng ta viết `Y = X + Y`, chúng ta sẽ hủy bỏ tensor mà `Y` dùng để trỏ đến và thay vào đó chỉ `Y` vào bộ nhớ mới được phân bổ. Trong ví dụ sau, chúng ta chứng minh điều này với hàm `id()` của Python, cung cấp cho chúng ta địa chỉ chính xác của đối tượng tham chiếu trong bộ nhớ. Sau khi chạy `Y = Y + X`, chúng tôi sẽ thấy rằng `id(Y)` chỉ đến một vị trí khác. Đó là do Python lần đầu tiên đánh giá `Y + X`, phân bổ bộ nhớ mới cho kết quả và sau đó làm cho `Y` trỏ đến vị trí mới này trong bộ nhớ.

```{.python .input}
#@tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

Điều này có thể là không mong muốn vì hai lý do. Đầu tiên, chúng tôi không muốn chạy xung quanh phân bổ bộ nhớ một cách không cần thiết tất cả các thời gian. Trong machine learning, chúng ta có thể có hàng trăm megabyte tham số và cập nhật tất cả chúng nhiều lần mỗi giây. Thông thường, chúng tôi sẽ muốn thực hiện các bản cập nhật này * tại chỗ *. Thứ hai, chúng ta có thể chỉ vào các tham số tương tự từ nhiều biến. Nếu chúng ta không cập nhật tại chỗ, các tham chiếu khác vẫn sẽ trỏ đến vị trí bộ nhớ cũ, làm cho nó có thể cho các phần của mã của chúng ta vô tình tham chiếu các tham số cũ.

:begin_tab:`mxnet, pytorch`
May mắn thay, (** thực hiện hoạt động tại chỗ **) rất dễ dàng. Chúng ta có thể gán kết quả của một hoạt động cho một mảng được phân bổ trước đó với ký hiệu slice, ví dụ, `Y[:] = <expression>`. Để minh họa khái niệm này, trước tiên chúng ta tạo ra một ma trận mới `Z` với hình dạng tương tự như `Y` khác, sử dụng `zeros_like` để phân bổ một khối $0$ mục nhập.
:end_tab:

:begin_tab:`tensorflow`
`Variables` là các thùng chứa có thể thay đổi trạng thái trong TensorFlow. Họ cung cấp một cách để lưu trữ các thông số mô hình của bạn. Chúng ta có thể gán kết quả của một hoạt động cho một `Variable` với `assign`. Để minh họa khái niệm này, chúng tôi tạo ra một `Variable` `Z` với hình dạng tương tự như một tensor `Y`, sử dụng `zeros_like` để phân bổ một khối $0$ mục nhập.
:end_tab:

```{.python .input}
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
#@tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

:begin_tab:`mxnet, pytorch`
[**Nếu giá trị của `X` không được sử dụng lại trong các tính toán tiếp theo, chúng ta cũng có thể sử dụng `X[:] = X + Y` hoặc `X += Y` để giảm chi phí bộ nhớ của hoạt động.**]
:end_tab:

:begin_tab:`tensorflow`
Ngay cả khi bạn lưu trữ trạng thái liên tục trong `Variable`, bạn có thể muốn giảm mức sử dụng bộ nhớ của mình hơn nữa bằng cách tránh phân bổ dư thừa cho hàng chục không phải là tham số mô hình của bạn. 

Bởi vì TensorFlow `Tensors` là bất biến và gradient không chảy qua các bài tập `Variable`, TensorFlow không cung cấp một cách rõ ràng để chạy một hoạt động riêng lẻ tại chỗ. 

Tuy nhiên, TensorFlow cung cấp trình trang trí `tf.function` để gói tính toán bên trong đồ thị TensorFlow được biên dịch và tối ưu hóa trước khi chạy. Điều này cho phép TensorFlow cắt tỉa các giá trị chưa sử dụng và sử dụng lại các phân bổ trước đó không còn cần thiết nữa. Điều này giảm thiểu chi phí bộ nhớ của tính toán TensorFlow.
:end_tab:

```{.python .input}
#@tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
#@tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be re-used when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Chuyển đổi sang các đối tượng Python khác

:begin_tab:`mxnet, tensorflow`
[**Chuyển đổi sang tensor NumPy (`ndarray`) **], hoặc ngược lại, rất dễ dàng. Kết quả được chuyển đổi không chia sẻ bộ nhớ. Sự bất tiện nhỏ này thực sự khá quan trọng: khi bạn thực hiện các thao tác trên CPU hoặc trên GPU, bạn không muốn tạm dừng tính toán, chờ xem liệu gói NumPy của Python có thể muốn làm một cái gì đó khác với cùng một đoạn bộ nhớ hay không.
:end_tab:

:begin_tab:`pytorch`
[**Chuyển đổi sang tensor NumPy (`ndarray`) **], hoặc ngược lại, rất dễ dàng. Ngọn đuốc Tensor và mảng numpy sẽ chia sẻ vị trí bộ nhớ cơ bản của chúng, và thay đổi một thông qua một hoạt động tại chỗ cũng sẽ thay đổi vị trí khác.
:end_tab:

```{.python .input}
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
#@tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
#@tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

Để (**chuyển đổi một tensor size-1 thành một scalar Python **), chúng ta có thể gọi hàm `item` hoặc các hàm tích hợp sẵn của Python.

```{.python .input}
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
#@tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

## Tóm tắt

* Giao diện chính để lưu trữ và thao tác dữ liệu cho học sâu là tensor (mảng $n$ chiều). Nó cung cấp một loạt các chức năng bao gồm các hoạt động toán học cơ bản, phát sóng, lập chỉ mục, cắt lát, tiết kiệm bộ nhớ và chuyển đổi sang các đối tượng Python khác.

## Bài tập

1. Chạy mã trong phần này. Thay đổi câu lệnh có điều kiện `X == Y` trong phần này thành `X < Y` hoặc `X > Y`, và sau đó xem loại tensor bạn có thể nhận được.
1. Thay thế hai hàng chục hoạt động theo phần tử trong cơ chế phát sóng bằng các hình dạng khác, ví dụ, các dụng cụ 3 chiều. Kết quả có giống như mong đợi không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/26)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/27)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/187)
:end_tab:

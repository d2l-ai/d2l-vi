# Đại số tuyến tính
:label:`sec_linear-algebra`

Bây giờ bạn có thể lưu trữ và thao tác dữ liệu, chúng ta hãy xem xét ngắn gọn tập hợp con của đại số tuyến tính cơ bản mà bạn sẽ cần hiểu và thực hiện hầu hết các mô hình được đề cập trong cuốn sách này. Dưới đây, chúng tôi giới thiệu các đối tượng toán học cơ bản, số học, và phép toán trong đại số tuyến tính, thể hiện từng đối tượng trong số chúng thông qua ký hiệu toán học và việc thực hiện tương ứng trong mã. 

## Vô hướng

Nếu bạn chưa bao giờ học đại số tuyến tính hoặc học máy, thì kinh nghiệm trong quá khứ của bạn với toán học có thể bao gồm suy nghĩ về một số tại một thời điểm. Và, nếu bạn đã bao giờ cân bằng một sổ séc hoặc thậm chí trả tiền cho bữa tối tại một nhà hàng thì bạn đã biết làm thế nào để làm những việc cơ bản như thêm và nhân các cặp số. Ví dụ, nhiệt độ ở Palo Alto là $52$ độ F. Chính thức, chúng ta gọi các giá trị chỉ bao gồm một số lượng * vô số*. Nếu bạn muốn chuyển đổi giá trị này sang Celsius (thang nhiệt độ hợp lý hơn của hệ mét), bạn sẽ đánh giá biểu thức $c = \frac{5}{9}(f - 32)$, đặt $f$ thành $52$. Trong phương trình này, mỗi thuật ngữ — $5$, $9$, và $32$—là những giá trị vô hướng. Các giữ chỗ $c$ và $f$ được gọi là *variables* và chúng đại diện cho các giá trị vô hướng không xác định. 

Trong cuốn sách này, chúng ta áp dụng ký hiệu toán học trong đó các biến vô hướng được biểu thị bằng các chữ cái thường thấp hơn (ví dụ: $x$, $y$ và $z$). Chúng tôi biểu thị không gian của tất cả (liên tục) * có giá trị thực* vô hướng bởi $\mathbb{R}$. Đối với sự nhanh chóng, chúng ta sẽ xem xét các định nghĩa nghiêm ngặt về chính xác *không gian* là gì, nhưng chỉ cần nhớ rằng biểu thức $x \in \mathbb{R}$ là một cách chính thức để nói rằng $x$ là một vô hướng có giá trị thực. Ký hiệu $\in$ có thể được phát âm “trong” và chỉ đơn giản là biểu thị thành viên trong một bộ. Tương tự, chúng ta có thể viết $x, y \in \{0, 1\}$ để nói rằng $x$ và $y$ là những con số có giá trị chỉ có thể là $0$ hoặc $1$. 

(**Một vô hướng được biểu diễn bằng một tensor chỉ với một yếu tố. **) Trong đoạn tiếp theo, chúng ta khởi tạo hai vô hướng và thực hiện một số phép toán số học quen thuộc với chúng, cụ thể là cộng, nhân, chia và hàm mũ.

```{.python .input}
from mxnet import np, npx
npx.set_np()

x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
#@tab pytorch
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

## Vectơ

[**Bạn có thể nghĩ một vector chỉ đơn giản là một danh sách các giá trị vô hướng. **] Chúng ta gọi các giá trị này là * element* (*entries* hoặc *components*) của vector. Khi vectơ của chúng ta đại diện cho các ví dụ từ tập dữ liệu của chúng ta, các giá trị của chúng giữ một số ý nghĩa trong thế giới thực. Ví dụ: nếu chúng ta đang đào tạo một mô hình để dự đoán rủi ro mà một khoản vay mặc định, chúng ta có thể liên kết mỗi ứng viên với một vectơ có thành phần tương ứng với thu nhập, thời gian làm việc, số lượng mặc định trước đó và các yếu tố khác. Nếu chúng ta đang nghiên cứu nguy cơ đau tim bệnh nhân bệnh viện có khả năng phải đối mặt, chúng ta có thể đại diện cho mỗi bệnh nhân bằng một vectơ có thành phần nắm bắt các dấu hiệu quan trọng gần đây nhất của họ, mức cholesterol, phút tập thể dục mỗi ngày, vv Trong ký hiệu toán học, chúng ta thường sẽ biểu thị vectơ là mặt đậm, thấp hơn cased các chữ cái (ví dụ: $\mathbf{x}$, $\mathbf{y}$ và $\mathbf{z})$. 

Chúng tôi làm việc với các vectơ thông qua các hàng chục một chiều. Trong hàng chục nói chung có thể có độ dài tùy ý, tùy thuộc vào giới hạn bộ nhớ của máy của bạn.

```{.python .input}
x = np.arange(4)
x
```

```{.python .input}
#@tab pytorch
x = torch.arange(4)
x
```

```{.python .input}
#@tab tensorflow
x = tf.range(4)
x
```

Chúng ta có thể tham khảo bất kỳ phần tử nào của vectơ bằng cách sử dụng chỉ số dưới. Ví dụ: chúng ta có thể tham khảo phần tử $i^\mathrm{th}$ của $\mathbf{x}$ bởi $x_i$. Lưu ý rằng phần tử $x_i$ là vô hướng, vì vậy chúng ta không phải đối mặt với phông chữ khi đề cập đến nó. Văn học mở rộng coi vectơ cột là định hướng mặc định của vectơ, cuốn sách này cũng vậy. Trong toán học, một vector $\mathbf{x}$ có thể được viết là 

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\x_{2}  \\ \vdots  \\x_{n}\end{bmatrix},$$
:eqlabel:`eq_vec_def`

trong đó $x_1, \ldots, x_n$ là các yếu tố của vectơ. Trong code, chúng ta (**truy cập bất kỳ phần tử nào bằng cách lập chỉ mục vào tensor.**)

```{.python .input}
x[3]
```

```{.python .input}
#@tab pytorch
x[3]
```

```{.python .input}
#@tab tensorflow
x[3]
```

### Chiều dài, kích thước và hình dạng

Chúng ta hãy xem lại một số khái niệm từ :numref:`sec_ndarray`. Một vectơ chỉ là một mảng các số. Và cũng giống như mỗi mảng có một chiều dài, vì vậy không mỗi vector. Trong ký hiệu toán học, nếu chúng ta muốn nói rằng một vector $\mathbf{x}$ bao gồm $n$ vô hướng có giá trị thực, chúng ta có thể thể hiện điều này là $\mathbf{x} \in \mathbb{R}^n$. Độ dài của một vectơ thường được gọi là *dimension* của vectơ. 

Như với một mảng Python thông thường, chúng ta [** có thể truy cập độ dài của một tensor**] bằng cách gọi hàm `len()` tích hợp của Python.

```{.python .input}
len(x)
```

```{.python .input}
#@tab pytorch
len(x)
```

```{.python .input}
#@tab tensorflow
len(x)
```

Khi một tensor đại diện cho một vectơ (với chính xác một trục), chúng ta cũng có thể truy cập độ dài của nó thông qua thuộc tính `.shape`. Hình dạng là một tuple liệt kê chiều dài (chiều) dọc theo mỗi trục của tensor. (**Đối với hàng chục chỉ với một trục, hình dạng chỉ có một yếu tố. **)

```{.python .input}
x.shape
```

```{.python .input}
#@tab pytorch
x.shape
```

```{.python .input}
#@tab tensorflow
x.shape
```

Lưu ý rằng từ “chiều” có xu hướng bị quá tải trong các bối cảnh này và điều này có xu hướng gây nhầm lẫn cho mọi người. Để làm rõ, chúng ta sử dụng chiều của một * vector* hoặc một * axis* để chỉ chiều dài của nó, tức là số phần tử của một vectơ hoặc một trục. Tuy nhiên, chúng ta sử dụng chiều của một tensor để chỉ số trục mà một tensor có. Theo nghĩa này, chiều của một số trục của một tensor sẽ là chiều dài của trục đó. 

## Ma trận

Cũng giống như các vectơ tổng quát hóa vô hướng từ thứ tự số 0 để đặt hàng một, ma trận khái quát hóa vectơ từ thứ tự một đến thứ tự hai. Ma trận, mà chúng ta thường sẽ biểu thị bằng chữ in hoa, có mặt đậm (ví dụ: $\mathbf{X}$, $\mathbf{Y}$ và $\mathbf{Z}$), được biểu diễn bằng mã dưới dạng hàng chục với hai trục. 

Trong ký hiệu toán, ta sử dụng $\mathbf{A} \in \mathbb{R}^{m \times n}$ để thể hiện rằng ma trận $\mathbf{A}$ bao gồm $m$ hàng và $n$ cột vô hướng có giá trị thực. Trực quan, chúng ta có thể minh họa bất kỳ ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ như một bảng, trong đó mỗi phần tử $a_{ij}$ thuộc về hàng $i^{\mathrm{th}}$ và cột $j^{\mathrm{th}}$: 

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`

Đối với bất kỳ $\mathbf{A} \in \mathbb{R}^{m \times n}$, hình dạng của $\mathbf{A}$ là ($m$, $n$) hoặc $m \times n$. Cụ thể, khi một ma trận có cùng số hàng và cột, hình dạng của nó trở thành một hình vuông; do đó, nó được gọi là matrận *vuông*. 

Chúng ta có thể [** tạo một ma trận $m \times n$**] bằng cách chỉ định một hình dạng với hai thành phần $m$ và $n$ khi gọi bất kỳ chức năng yêu thích nào của chúng tôi để khởi tạo một tensor.

```{.python .input}
A = np.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab pytorch
A = torch.arange(20).reshape(5, 4)
A
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20), (5, 4))
A
```

Chúng ta có thể truy cập phần tử vô hướng $a_{ij}$ của ma trận $\mathbf{A}$ trong :eqref:`eq_matrix_def` bằng cách chỉ định các chỉ số cho hàng ($i$) và cột ($j$), chẳng hạn như $[\mathbf{A}]_{ij}$. Khi các yếu tố vô hướng của ma trận $\mathbf{A}$, chẳng hạn như trong :eqref:`eq_matrix_def`, không được đưa ra, chúng ta có thể chỉ đơn giản sử dụng chữ thường của ma trận $\mathbf{A}$ với chỉ số dưới, $a_{ij}$, để tham khảo $[\mathbf{A}]_{ij}$. Để giữ ký hiệu đơn giản, dấu phẩy chỉ được chèn vào các chỉ số riêng biệt khi cần thiết, chẳng hạn như $a_{2, 3j}$ và $[\mathbf{A}]_{2i-1, 3}$. 

Đôi khi, chúng tôi muốn lật các trục. Khi chúng ta trao đổi các hàng và cột của ma trận, kết quả được gọi là *transpose* của ma trận. Chính thức, chúng tôi biểu thị một ma trận $\mathbf{A}$ của transpose bởi $\mathbf{A}^\top$ và nếu $\mathbf{B} = \mathbf{A}^\top$, sau đó $b_{ij} = a_{ji}$ cho bất kỳ $i$ và $j$. Như vậy, transpose của $\mathbf{A}$ trong :eqref:`eq_matrix_def` là một ma trận $n \times m$: 

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

Bây giờ chúng ta truy cập vào một (** matrix của transpose**) trong mã.

```{.python .input}
A.T
```

```{.python .input}
#@tab pytorch
A.T
```

```{.python .input}
#@tab tensorflow
tf.transpose(A)
```

Là một loại đặc biệt của ma trận vuông, [** a *matrix đối xứng * $\mathbf{A}$ bằng chuyển vị của nó: $\mathbf{A} = \mathbf{A}^\top$.**] Ở đây chúng ta xác định ma trận đối xứng `B`.

```{.python .input}
B = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab pytorch
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

```{.python .input}
#@tab tensorflow
B = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
```

Bây giờ chúng tôi so sánh `B` với transpose của nó.

```{.python .input}
B == B.T
```

```{.python .input}
#@tab pytorch
B == B.T
```

```{.python .input}
#@tab tensorflow
B == tf.transpose(B)
```

Ma trận là cấu trúc dữ liệu hữu ích: chúng cho phép chúng ta tổ chức dữ liệu có các phương thức biến thể khác nhau. Ví dụ: các hàng trong ma trận của chúng ta có thể tương ứng với các ngôi nhà khác nhau (ví dụ dữ liệu), trong khi các cột có thể tương ứng với các thuộc tính khác nhau. Điều này nghe có vẻ quen thuộc nếu bạn đã từng sử dụng phần mềm bảng tính hoặc đã đọc :numref:`sec_pandas`. Do đó, mặc dù định hướng mặc định của một vectơ duy nhất là một vectơ cột, trong một ma trận đại diện cho một tập dữ liệu dạng bảng, nó là thông thường hơn để coi mỗi ví dụ dữ liệu như một vectơ hàng trong ma trận. Và, như chúng ta sẽ thấy trong các chương sau, quy ước này sẽ cho phép các thực hành học sâu phổ biến. Ví dụ, dọc theo trục ngoài cùng của một tensor, chúng ta có thể truy cập hoặc liệt kê minibatches ví dụ dữ liệu, hoặc chỉ các ví dụ dữ liệu nếu không có minibatch tồn tại. 

## Tensors

Cũng giống như vectơ tổng quát hóa vô hướng, và ma trận tổng quát hóa vectơ, chúng ta có thể xây dựng cấu trúc dữ liệu với nhiều trục hơn nữa. [**Tensors**](“tensors” trong tiểu mục này đề cập đến các đối tượng đại số) (** cung cấp cho chúng ta một cách chung để mô tả các mảng $n$ chiều với một số trục tùy ý.**) Vectơ, ví dụ, là các hàng chục bậc nhất, và ma trận là hàng chục bậc hai. Tensors được ký hiệu bằng chữ in hoa của một khuôn mặt phông chữ đặc biệt (ví dụ, $\mathsf{X}$, $\mathsf{Y}$, và $\mathsf{Z}$) và cơ chế lập chỉ mục của chúng (ví dụ, $x_{ijk}$ và $[\mathsf{X}]_{1, 2i-1, 3}$) tương tự như của ma trận. 

Tensors sẽ trở nên quan trọng hơn khi chúng ta bắt đầu làm việc với hình ảnh, đến các mảng $n$ chiều với 3 trục tương ứng với chiều cao, chiều rộng và trục * kênh* để xếp các kênh màu (đỏ, xanh lá cây và xanh dương). Hiện tại, chúng ta sẽ bỏ qua hàng chục bậc cao hơn và tập trung vào những điều cơ bản.

```{.python .input}
X = np.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab pytorch
X = torch.arange(24).reshape(2, 3, 4)
X
```

```{.python .input}
#@tab tensorflow
X = tf.reshape(tf.range(24), (2, 3, 4))
X
```

## Tính chất cơ bản của Tensor Arithmetic

Vô hướng, vectơ, ma trận, và hàng chục (“tensors” trong tiểu mục này đề cập đến các đối tượng đại số) của một số trục tùy ý có một số tính chất tốt đẹp thường có ích. Ví dụ, bạn có thể nhận thấy từ định nghĩa của một phép toán elementwise rằng bất kỳ hoạt động đơn nguyên tố nào không thay đổi hình dạng của toán hạng của nó. Tương tự như vậy, [**với bất kỳ hai hàng chục có cùng hình dạng, kết quả của bất kỳ phép toán phần tử nhị phân nào sẽ là một tensor của cùng một hình dạng đó**] Ví dụ, thêm hai ma trận có cùng hình dạng thực hiện phép cộng elementwise trên hai ma trận này.

```{.python .input}
A = np.arange(20).reshape(5, 4)
B = A.copy()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab pytorch
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # Assign a copy of `A` to `B` by allocating new memory
A, A + B
```

```{.python .input}
#@tab tensorflow
A = tf.reshape(tf.range(20, dtype=tf.float32), (5, 4))
B = A  # No cloning of `A` to `B` by allocating new memory
A, A + B
```

Cụ thể, [** elementwise phép nhân của hai ma trận được gọi là sản phẩm * Hadamard***] của họ (ký hiệu toán $\odot$). Xem xét ma trận $\mathbf{B} \in \mathbb{R}^{m \times n}$ có phần tử của hàng $i$ và cột $j$ là $b_{ij}$. Sản phẩm Hadamard của ma trận $\mathbf{A}$ (được định nghĩa trong :eqref:`eq_matrix_def`) và $\mathbf{B}$ 

$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
A * B
```

```{.python .input}
#@tab pytorch
A * B
```

```{.python .input}
#@tab tensorflow
A * B
```

[**Nhân hoặc thêm một tensor với vô hướng **] cũng không thay đổi hình dạng của tensor, trong đó mỗi phần tử của tensor toán hạng sẽ được thêm hoặc nhân với vô hướng.

```{.python .input}
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
#@tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

## Giảm
:label:`subseq_lin-alg-reduction`

Một thao tác hữu ích mà chúng ta có thể thực hiện với các hàng chục tùy ý là tính toán [** tổng các phần tử của chúng.**] Trong ký hiệu toán học, chúng ta thể hiện các khoản tiền bằng ký hiệu $\sum$. Để thể hiện tổng của các phần tử trong một vector $\mathbf{x}$ chiều dài $d$, chúng tôi viết $\sum_{i=1}^d x_i$. Trong code, chúng ta chỉ có thể gọi hàm để tính tổng.

```{.python .input}
x = np.arange(4)
x, x.sum()
```

```{.python .input}
#@tab pytorch
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
#@tab tensorflow
x = tf.range(4, dtype=tf.float32)
x, tf.reduce_sum(x)
```

Chúng ta có thể diễn đạt [** tổng trên các phần tử của hàng chục hình dạng tùy ý.**] Ví dụ, tổng các phần tử của ma trận $m \times n$ $\mathbf{A}$ có thể được viết $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
A.shape, A.sum()
```

```{.python .input}
#@tab pytorch
A.shape, A.sum()
```

```{.python .input}
#@tab tensorflow
A.shape, tf.reduce_sum(A)
```

Theo mặc định, gọi hàm để tính tổng
*giảm* một tensor dọc theo tất cả các trục của nó đến một vô hướng.
Chúng ta cũng có thể [** chỉ định các trục dọc theo đó tensor được giảm thông qua tổng hợp.**] Lấy ma trận làm ví dụ. Để giảm kích thước hàng (trục 0) bằng cách tổng hợp các phần tử của tất cả các hàng, chúng tôi chỉ định `axis=0` khi gọi hàm. Vì ma trận đầu vào giảm dọc theo trục 0 để tạo ra vectơ đầu ra, kích thước của trục 0 của đầu vào bị mất trong hình dạng đầu ra.

```{.python .input}
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis0 = tf.reduce_sum(A, axis=0)
A_sum_axis0, A_sum_axis0.shape
```

Chỉ định `axis=1` sẽ giảm kích thước cột (trục 1) bằng cách tổng hợp các phần tử của tất cả các cột. Do đó, kích thước của trục 1 của đầu vào bị mất trong hình dạng đầu ra.

```{.python .input}
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab pytorch
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
```

```{.python .input}
#@tab tensorflow
A_sum_axis1 = tf.reduce_sum(A, axis=1)
A_sum_axis1, A_sum_axis1.shape
```

Giảm một ma trận dọc theo cả hàng và cột thông qua tổng tương đương với việc tổng hợp tất cả các phần tử của ma trận.

```{.python .input}
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab pytorch
A.sum(axis=[0, 1])  # Same as `A.sum()`
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(A, axis=[0, 1])  # Same as `tf.reduce_sum(A)`
```

[**Một số lượng liên quan là *mean*, còn được gọi là *trung bình*.**] Chúng tôi tính trung bình bằng cách chia tổng cho tổng số phần tử. Trong mã, chúng ta chỉ có thể gọi hàm để tính toán trung bình trên hàng chục hình dạng tùy ý.

```{.python .input}
A.mean(), A.sum() / A.size
```

```{.python .input}
#@tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

Tương tự như vậy, chức năng tính trung bình cũng có thể làm giảm tensor dọc theo các trục được chỉ định.

```{.python .input}
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab pytorch
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
#@tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

### Tổng không giảm
:label:`subseq_lin-alg-non-reduction`

Tuy nhiên, đôi khi có thể hữu ích khi [** giữ số trục không thay đổi**] khi gọi hàm tính tổng hoặc trung bình.

```{.python .input}
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab pytorch
sum_A = A.sum(axis=1, keepdims=True)
sum_A
```

```{.python .input}
#@tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A
```

Ví dụ, vì `sum_A` vẫn giữ hai trục của nó sau khi tổng hợp mỗi hàng, chúng ta có thể (** chia `A` cho `sum_A` với phát sóng. **)

```{.python .input}
A / sum_A
```

```{.python .input}
#@tab pytorch
A / sum_A
```

```{.python .input}
#@tab tensorflow
A / sum_A
```

Nếu chúng ta muốn tính toán [** tổng tích lũy của các phần tử của `A` dọc theo một số trục **], giả sử `axis=0` (từng hàng), chúng ta có thể gọi hàm `cumsum`. Chức năng này sẽ không làm giảm tensor đầu vào dọc theo bất kỳ trục nào.

```{.python .input}
A.cumsum(axis=0)
```

```{.python .input}
#@tab pytorch
A.cumsum(axis=0)
```

```{.python .input}
#@tab tensorflow
tf.cumsum(A, axis=0)
```

## Sản phẩm Dot

Cho đến nay, chúng tôi chỉ thực hiện các hoạt động elementwise, tổng và trung bình. Và nếu đây là tất cả những gì chúng ta có thể làm, đại số tuyến tính có lẽ sẽ không xứng đáng với phần riêng của nó. Tuy nhiên, một trong những hoạt động cơ bản nhất là sản phẩm chấm. Với hai vectơ $\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, sản phẩm *dot của họ* $\mathbf{x}^\top \mathbf{y}$ (hoặc $\langle \mathbf{x}, \mathbf{y}  \rangle$) là một tổng so với các sản phẩm của các phần tử ở cùng một vị trí: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$. 

[~~Các *dot product* của hai vectơ là một tổng so với các sản phẩm của các phần tử ở cùng một vị trí ~~]

```{.python .input}
y = np.ones(4)
x, y, np.dot(x, y)
```

```{.python .input}
#@tab pytorch
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
#@tab tensorflow
y = tf.ones(4, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

Lưu ý rằng (**chúng ta có thể diễn đạt tích chấm của hai vectơ tương đương bằng cách thực hiện phép nhân elementwise và sau đó là tổng: **)

```{.python .input}
np.sum(x * y)
```

```{.python .input}
#@tab pytorch
torch.sum(x * y)
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(x * y)
```

Các sản phẩm Dot rất hữu ích trong một loạt các bối cảnh. Ví dụ, cho một số tập hợp các giá trị, được biểu thị bằng một vectơ $\mathbf{x}  \in \mathbb{R}^d$ và một tập hợp trọng số được ký hiệu bởi $\mathbf{w} \in \mathbb{R}^d$, tổng trọng số của các giá trị trong $\mathbf{x}$ theo trọng lượng $\mathbf{w}$ có thể được biểu thị dưới dạng sản phẩm chấm $\mathbf{x}^\top \mathbf{w}$. Khi trọng lượng không âm và tổng thành một (tức là $\left(\sum_{i=1}^{d} {w_i} = 1\right)$), sản phẩm chấm biểu thị trung bình * có trọng lự*. Sau khi bình thường hóa hai vectơ để có chiều dài đơn vị, các sản phẩm chấm thể hiện cosin của góc giữa chúng. Chúng tôi sẽ chính thức giới thiệu khái niệm này là * chiều dài* sau trong phần này. 

## Matrix-Vector Sản phẩm

Bây giờ chúng ta đã biết cách tính toán các sản phẩm chấm, chúng ta có thể bắt đầu hiểu các sản phẩm *ma thuật-vectơ *. Nhớ lại ma trận $\mathbf{A} \in \mathbb{R}^{m \times n}$ và vector $\mathbf{x} \in \mathbb{R}^n$ được xác định và hình dung trong :eqref:`eq_matrix_def` và :eqref:`eq_vec_def` tương ứng. Hãy để chúng tôi bắt đầu bằng cách hình dung ma trận $\mathbf{A}$ về vectơ hàng của nó 

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

trong đó mỗi $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ là một vectơ hàng đại diện cho hàng $i^\mathrm{th}$ của ma trận $\mathbf{A}$. 

[**Sản phẩm ma trận-vector $\mathbf{A}\mathbf{x}$ chỉ đơn giản là một vector cột có chiều dài $m$, có $i^\mathrm{th}$ phần tử là sản phẩm chấm $\mathbf{a}^\top_i \mathbf{x}$: **] 

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Chúng ta có thể nghĩ đến phép nhân của một ma trận $\mathbf{A}\in \mathbb{R}^{m \times n}$ như một biến đổi dự án vectơ từ $\mathbb{R}^{n}$ đến $\mathbb{R}^{m}$. Những biến đổi này trở nên hữu ích đáng kể. Ví dụ, chúng ta có thể biểu diễn các vòng quay dưới dạng phép nhân bằng một ma trận vuông. Như chúng ta sẽ thấy trong các chương tiếp theo, chúng ta cũng có thể sử dụng các sản phẩm ma thuật-vector để mô tả các phép tính chuyên sâu nhất cần thiết khi tính toán từng lớp trong mạng thần kinh với các giá trị của lớp trước đó.

:begin_tab:`mxnet`
Thể hiện các sản phẩm ma thuật-vector trong mã với hàng chục, chúng tôi sử dụng chức năng `dot` tương tự như đối với các sản phẩm chấm. Khi chúng ta gọi `np.dot(A, x)` với ma trận `A` và một vector `x`, sản phẩm ma thuật-vector được thực hiện. Lưu ý rằng kích thước cột `A` (chiều dài của nó dọc theo trục 1) phải giống với kích thước `x` (chiều dài của nó).
:end_tab:

:begin_tab:`pytorch`
Thể hiện các sản phẩm ma thuật-vector trong mã với hàng chục, chúng tôi sử dụng hàm `mv`. Khi chúng ta gọi `torch.mv(A, x)` với ma trận `A` và một vector `x`, sản phẩm ma thuật-vector được thực hiện. Lưu ý rằng kích thước cột của `A` (chiều dài của nó dọc theo trục 1) phải giống với kích thước `x` (chiều dài của nó).
:end_tab:

:begin_tab:`tensorflow`
Thể hiện các sản phẩm ma thuật-vector trong mã với hàng chục, chúng tôi sử dụng hàm `matvec`. Khi chúng ta gọi `tf.linalg.matvec(A, x)` với ma trận `A` và một vector `x`, sản phẩm ma thuật-vector được thực hiện. Lưu ý rằng kích thước cột `A` (chiều dài của nó dọc theo trục 1) phải giống với kích thước `x` (chiều dài của nó).
:end_tab:

```{.python .input}
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
#@tab pytorch
A.shape, x.shape, torch.mv(A, x)
```

```{.python .input}
#@tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

## Phép nhân ma trận ma trận

Nếu bạn đã nhận được sự treo của các sản phẩm chấm và các sản phẩm ma thuật-vector, thì nhân * ma trận ma trận * phải đơn giản. 

Nói rằng chúng ta có hai ma trận $\mathbf{A} \in \mathbb{R}^{n \times k}$ và $\mathbf{B} \in \mathbb{R}^{k \times m}$: 

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$

Biểu thị bằng $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ vector hàng đại diện cho hàng $i^\mathrm{th}$ của ma trận $\mathbf{A}$, và để cho $\mathbf{b}_{j} \in \mathbb{R}^k$ là vectơ cột từ cột $j^\mathrm{th}$ của ma trận $\mathbf{B}$. Để sản xuất sản phẩm ma trận $\mathbf{C} = \mathbf{A}\mathbf{B}$, dễ nhất là nghĩ đến $\mathbf{A}$ về vectơ hàng và $\mathbf{B}$ về vectơ cột của nó: 

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$

Sau đó, sản phẩm ma trận $\mathbf{C} \in \mathbb{R}^{n \times m}$ được sản xuất như chúng ta chỉ đơn giản là tính toán từng phần tử $c_{ij}$ như là sản phẩm chấm $\mathbf{a}^\top_i \mathbf{b}_j$: 

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

[**Chúng ta có thể nghĩ đến phép nhân ma trận ma trận $\mathbf{AB}$ chỉ đơn giản là thực hiện $m$ các sản phẩm ma trận vector và khâu kết quả lại với nhau để tạo thành ma trận $n \times m$**] Trong đoạn mã sau, chúng tôi thực hiện phép nhân ma trận trên `A` và `B`. Ở đây, `A` là một ma trận với 5 hàng và 4 cột, và `B` là một ma trận với 4 hàng và 3 cột. Sau khi nhân, chúng ta có được một ma trận với 5 hàng và 3 cột.

```{.python .input}
B = np.ones(shape=(4, 3))
np.dot(A, B)
```

```{.python .input}
#@tab pytorch
B = torch.ones(4, 3)
torch.mm(A, B)
```

```{.python .input}
#@tab tensorflow
B = tf.ones((4, 3), tf.float32)
tf.matmul(A, B)
```

Phép nhân ma trận ma trận có thể được gọi đơn giản là nhân *ma trận *, và không nên nhầm lẫn với sản phẩm Hadamard. 

## Định mức
:label:`subsec_lin-algebra-norms`

Một số toán tử hữu ích nhất trong đại số tuyến tính là * chỉ tiêu *. Một cách không chính thức, định mức của một vector cho chúng ta biết cách * lớn* một vectơ là. Khái niệm về kích thước* đang được xem xét ở đây không liên quan đến chiều mà là độ lớn của các thành phần. 

Trong đại số tuyến tính, một định mức vectơ là một hàm $f$ ánh xạ một vectơ đến vô hướng, thỏa mãn một số ít các thuộc tính. Với bất kỳ vector $\mathbf{x}$ nào, thuộc tính đầu tiên nói rằng nếu chúng ta quy mô tất cả các phần tử của một vectơ bằng một hệ số không đổi $\alpha$, định mức của nó cũng quy mô theo giá trị * tuyệt đối * của cùng một yếu tố không đổi: 

$$f(\alpha \mathbf{x}) = |\alpha| f(\mathbf{x}).$$

Tính chất thứ hai là bất đẳng thức tam giác quen thuộc: 

$$f(\mathbf{x} + \mathbf{y}) \leq f(\mathbf{x}) + f(\mathbf{y}).$$

Tài sản thứ ba chỉ đơn giản nói rằng định mức phải không tiêu cực: 

$$f(\mathbf{x}) \geq 0.$$

Điều đó có ý nghĩa, như trong hầu hết các bối cảnh, * kích thước* nhỏ nhất cho bất cứ điều gì là 0. Thuộc tính cuối cùng yêu cầu định mức nhỏ nhất đạt được và chỉ đạt được bằng một vectơ bao gồm tất cả các số không. 

$$\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(\mathbf{x})=0.$$

Bạn có thể nhận thấy rằng các định mức âm thanh rất giống với các biện pháp khoảng cách. Và nếu bạn nhớ khoảng cách Euclide (nghĩ rằng định lý Pythagoras) từ trường lớp, thì các khái niệm về không tiêu cực và bất bình đẳng tam giác có thể rung chuông. Trên thực tế, khoảng cách Euclide là một tiêu chuẩn: cụ thể nó là định mức $L_2$. Giả sử rằng các yếu tố trong vector $n$ chiều $\mathbf{x}$ là $x_1, \ldots, x_n$. 

[**$L_2$ *norm* của $\mathbf{x}$ là căn bậc hai của tổng các ô vuông của các phần tử vectơ: **] 

(**$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2},$$**) 

trong đó chỉ số dưới $2$ thường bị bỏ qua trong $L_2$ chỉ tiêu, tức là $\|\mathbf{x}\|$ tương đương với $\|\mathbf{x}\|_2$. Trong mã, chúng ta có thể tính định mức $L_2$ của một vectơ như sau.

```{.python .input}
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
#@tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
#@tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

Trong học sâu, chúng tôi làm việc thường xuyên hơn với định mức $L_2$ bình phương. 

Bạn cũng sẽ thường xuyên gặp phải [**the $L_1$ *norm***], được biểu thị dưới dạng tổng các giá trị tuyệt đối của các phần tử vectơ: 

(**$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$**) 

So với định mức $L_2$, nó ít bị ảnh hưởng bởi outliers. Để tính định mức $L_1$, ta soạn hàm giá trị tuyệt đối với một tổng trên các phần tử.

```{.python .input}
np.abs(u).sum()
```

```{.python .input}
#@tab pytorch
torch.abs(u).sum()
```

```{.python .input}
#@tab tensorflow
tf.reduce_sum(tf.abs(u))
```

Cả định mức $L_2$ và định mức $L_1$ đều là những trường hợp đặc biệt của $L_p$ * tiêu chuẩn* chung hơn: 

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Tương tự như định mức $L_2$ của vectơ, [**chuẩn *Frobenius* của ma trận $\mathbf{X} \in \mathbb{R}^{m \times n}$**] là căn bậc hai của tổng bình phương của các phần tử ma trận: 

[**$$\|\mathbf{X}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$**] 

Định mức Frobenius thỏa mãn tất cả các thuộc tính của các chỉ tiêu vector. Nó hoạt động như thể nó là một định mức $L_2$ của một vectơ hình ma trận. Gọi hàm sau sẽ tính định mức Frobenius của một ma trận.

```{.python .input}
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
#@tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
#@tab tensorflow
tf.norm(tf.ones((4, 9)))
```

### Định mức và mục tiêu
:label:`subsec_norms_and_objectives`

Mặc dù chúng ta không muốn vượt quá xa bản thân, chúng ta có thể trồng một số trực giác về lý do tại sao những khái niệm này hữu ích. Trong học sâu, chúng ta thường cố gắng giải quyết các vấn đề tối ưu hóa:
*tối đa hóa* xác suất được gán cho dữ liệu quan sát;
*giảm xuống* khoảng cách giữa các dự đoán
and the ground-truth đất sự thật observations quan sát. Gán biểu diễn vector cho các mục (như từ, sản phẩm hoặc bài báo tin tức) sao cho khoảng cách giữa các mục tương tự được giảm thiểu và khoảng cách giữa các mục khác nhau được tối đa hóa. Thông thường, các mục tiêu, có lẽ là các thành phần quan trọng nhất của các thuật toán học sâu (bên cạnh dữ liệu), được thể hiện dưới dạng định mức. 

## Tìm hiểu thêm về Linear Algebra

Chỉ trong phần này, chúng tôi đã dạy bạn tất cả các đại số tuyến tính mà bạn sẽ cần phải hiểu một phần đáng chú ý của học sâu hiện đại. Có rất nhiều hơn để đại số tuyến tính và rất nhiều toán học đó rất hữu ích cho việc học máy. Ví dụ, ma trận có thể bị phân hủy thành các yếu tố, và những phân hủy này có thể tiết lộ cấu trúc chiều thấp trong các bộ dữ liệu thế giới thực. Có toàn bộ trường con của machine learning tập trung vào việc sử dụng các phân hủy ma trận và khái quát hóa của chúng cho các hàng chục bậc cao để khám phá cấu trúc trong bộ dữ liệu và giải quyết các vấn đề dự đoán. Nhưng cuốn sách này tập trung vào việc học sâu. Và chúng tôi tin rằng bạn sẽ có xu hướng tìm hiểu thêm toán học hơn một khi bạn đã nhận được bàn tay của bạn bẩn triển khai các mô hình máy học hữu ích trên các bộ dữ liệu thực. Vì vậy, trong khi chúng tôi có quyền giới thiệu nhiều toán học hơn nhiều sau này, chúng tôi sẽ kết thúc phần này ở đây. 

Nếu bạn muốn tìm hiểu thêm về đại số tuyến tính, bạn có thể tham khảo [online appendix on linear algebraic operations](https://d2l.ai/chapter_appendix-mathematics-for-deep-learning/geometry-linear-algebraic-ops.html) hoặc các tài nguyên tuyệt vời khác :cite:`Strang.1993,Kolter.2008,Petersen.Pedersen.ea.2008`. 

## Tóm tắt

* Vô hướng, vectơ, ma trận, và hàng chục là các đối tượng toán học cơ bản trong đại số tuyến tính.
* Vectơ tổng quát hóa vô hướng, và ma trận tổng quát hóa vectơ.
* Vô hướng, vectơ, ma trận và hàng chục có số không, một, hai và một số trục tùy ý, tương ứng.
* Một tensor có thể được giảm dọc theo các trục được chỉ định bởi `sum` và `mean`.
* Nhân Elementwise của hai ma trận được gọi là sản phẩm Hadamard của họ. Nó khác với phép nhân ma trận.
* Trong học sâu, chúng ta thường làm việc với các tiêu chuẩn như định mức $L_1$, định mức $L_2$ và định mức Frobenius.
* Chúng ta có thể thực hiện một loạt các hoạt động trên vô hướng, vectơ, ma trận và hàng chục.

## Bài tập

1. Chứng minh rằng sự chuyển vị của một ma trận $\mathbf{A}$ transpose là $\mathbf{A}$:$(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Cho hai ma trận $\mathbf{A}$ và $\mathbf{B}$, cho thấy tổng các transposes bằng chuyển vị của một tổng: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Cho bất kỳ ma trận vuông $\mathbf{A}$, là $\mathbf{A} + \mathbf{A}^\top$ luôn đối xứng? Tại sao?
1. Chúng tôi xác định tensor `X` của hình dạng (2, 3, 4) trong phần này. Sản lượng của `len(X)` là gì?
1. Đối với một tensor `X` có hình dạng tùy ý, liệu `len(X)` luôn tương ứng với chiều dài của một trục nhất định là `X` không? Trục đó là gì?
1. Chạy `A / A.sum(axis=1)` và xem những gì sẽ xảy ra. Bạn có thể phân tích lý do?
1. Khi đi du lịch giữa hai điểm ở Manhattan, khoảng cách mà bạn cần bao gồm về tọa độ, tức là về con đường và đường phố là bao nhiêu? Bạn có thể đi du lịch theo đường chéo?
1. Xem xét một tensor với hình dạng (2, 3, 4). Các hình dạng của đầu ra tổng kết dọc theo trục 0, 1 và 2 là gì?
1. Nạp một tensor với 3 trục trở lên đến chức năng `linalg.norm` và quan sát đầu ra của nó. Chức năng này tính toán gì cho hàng chục hình dạng tùy ý?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/30)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/31)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/196)
:end_tab:

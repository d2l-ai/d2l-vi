# Sự khác biệt tự động
:label:`sec_autograd`

Như chúng tôi đã giải thích trong :numref:`sec_calculus`, sự khác biệt là một bước quan trọng trong gần như tất cả các thuật toán tối ưu hóa học tập sâu. Trong khi các tính toán để lấy các dẫn xuất này rất đơn giản, chỉ đòi hỏi một số phép tính cơ bản, đối với các mô hình phức tạp, làm việc các bản cập nhật bằng tay có thể là một nỗi đau (và thường dễ bị lỗi). 

Các khung học sâu đẩy nhanh công việc này bằng cách tự động tính toán các dẫn xuất, tức là *phân biệt tự động*. Trong thực tế, dựa trên mô hình được thiết kế của chúng tôi, hệ thống xây dựng một biểu đồ tính toán *, theo dõi dữ liệu kết hợp thông qua đó các hoạt động để tạo ra đầu ra. Sự khác biệt tự động cho phép hệ thống để sau đó backpropagate gradients. Ở đây, * backpropagate* chỉ đơn giản là có nghĩa là theo dõi thông qua biểu đồ tính toán, điền vào các dẫn xuất từng phần đối với mỗi tham số. 

## Một ví dụ đơn giản

Như một ví dụ đồ chơi, nói rằng chúng ta quan tâm đến (** phân biệt chức năng $y = 2\mathbf{x}^{\top}\mathbf{x}$ đối với vector cột $\mathbf{x}$.**) Để bắt đầu, chúng ta hãy tạo biến `x` và gán cho nó một giá trị ban đầu.

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input}
#@tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**Trước khi chúng tôi thậm chí tính toán gradient của $y$ đối với $\mathbf{x}$, chúng ta sẽ cần một nơi để lưu trữ nó.**] Điều quan trọng là chúng ta không phân bổ bộ nhớ mới mỗi khi chúng ta lấy một dẫn xuất đối với một tham số vì chúng ta thường sẽ cập nhật các tham số tương tự hàng ngàn hoặc hàng triệu lần và could quickly Nhanh chóng run chạy out of memory bộ nhớ. Lưu ý rằng một gradient của một hàm có giá trị vô hướng đối với một vector $\mathbf{x}$ chính nó có giá trị vectơ và có hình dạng tương tự như $\mathbf{x}$.

```{.python .input}
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input}
#@tab pytorch
x.requires_grad_(True)  # Same as `x = torch.arange(4.0, requires_grad=True)`
x.grad  # The default value is None
```

```{.python .input}
#@tab tensorflow
x = tf.Variable(x)
```

(**Bây giờ chúng ta hãy tính $y$.**)

```{.python .input}
# Place our code inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input}
#@tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
#@tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

Vì `x` là một vectơ có chiều dài 4, một tích chấm của `x` và `x` được thực hiện, mang lại đầu ra vô hướng mà chúng ta gán cho `y`. Tiếp theo, [**chúng ta có thể tự động tính toán gradient của `y` đối với mỗi thành phần của `x`**] bằng cách gọi hàm để truyền ngược và in gradient.

```{.python .input}
y.backward()
x.grad
```

```{.python .input}
#@tab pytorch
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(**Độ dốc của hàm $y = 2\mathbf{x}^{\top}\mathbf{x}$ đối với $\mathbf{x}$ nên là $4\mathbf{x}$.**) Hãy để chúng tôi nhanh chóng xác minh rằng gradient mong muốn của chúng tôi đã được tính chính xác.

```{.python .input}
x.grad == 4 * x
```

```{.python .input}
#@tab pytorch
x.grad == 4 * x
```

```{.python .input}
#@tab tensorflow
x_grad == 4 * x
```

[**Bây giờ chúng ta hãy tính một hàm khác của `x`.**]

```{.python .input}
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input}
#@tab pytorch
# PyTorch accumulates the gradient in default, we need to clear the previous
# values
x.grad.zero_()
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## Lùi cho các biến không vô hướng

Về mặt kỹ thuật, khi `y` không phải là vô hướng, cách giải thích tự nhiên nhất về sự khác biệt của một vectơ `y` đối với một vectơ `x` là một ma trận. Đối với bậc cao hơn và chiều cao hơn `y` và `x`, kết quả khác biệt có thể là một tensor bậc cao. 

Tuy nhiên, trong khi những đối tượng kỳ lạ hơn này xuất hiện trong học máy nâng cao (bao gồm [** trong học sâu **]), thường xuyên hơn (** khi chúng ta gọi ngược trên một vectơ, **), chúng ta đang cố gắng tính toán các dẫn xuất của các hàm mất mát cho mỗi thành phần của một * lô* ví dụ đào tạo. Ở đây, (**ý định của chúng tôi là**) không tính toán ma trận phân biệt mà thay vào đó (** tổng của các dẫn xuất từng phần được tính riêng cho mỗi ví dụ**) trong lô.

```{.python .input}
# When we invoke `backward` on a vector-valued variable `y` (function of `x`),
# a new scalar variable is created by summing the elements in `y`. Then the
# gradient of that scalar variable with respect to `x` is computed
with autograd.record():
    y = x * x  # `y` is a vector
y.backward()
x.grad  # Equals to y = sum(x * x)
```

```{.python .input}
#@tab pytorch
# Invoking `backward` on a non-scalar requires passing in a `gradient` argument
# which specifies the gradient of the differentiated function w.r.t `self`.
# In our case, we simply want to sum the partial derivatives, so passing
# in a gradient of ones is appropriate
x.grad.zero_()
y = x * x
# y.backward(torch.ones(len(x))) equivalent to the below
y.sum().backward()
x.grad
```

```{.python .input}
#@tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## Tách tính toán

Đôi khi, chúng tôi muốn [** di chuyển một số tính toán bên ngoài đồ thị tính toán đã ghi lại.**] Ví dụ, nói rằng `y` được tính như một hàm `x`, và sau đó `z` được tính như một hàm của cả `y` và `x`. Bây giờ, hãy tưởng tượng rằng chúng tôi muốn tính toán gradient của `z` đối với `x`, nhưng muốn vì một lý do nào đó để đối xử với `y` như một hằng số, và chỉ tính đến vai trò mà `x` chơi sau khi `y` được tính toán. 

Ở đây, chúng ta có thể tách `y` để trả về một biến mới `u` có cùng giá trị như `y` nhưng loại bỏ bất kỳ thông tin nào về cách `y` được tính toán trong biểu đồ tính toán. Nói cách khác, gradient sẽ không chảy ngược qua `u` đến `x`. Như vậy, hàm truyền ngược sau đây tính toán đạo hàm từng phần của `z = u * x` đối với `x` trong khi xử lý `u` như một hằng số, thay vì đạo hàm từng phần của `z = x * x * x` đối với `x`.

```{.python .input}
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
#@tab tensorflow
# Set `persistent=True` to run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

Kể từ khi tính toán `y` đã được ghi lại, sau đó chúng ta có thể gọi truyền ngược trên `y` để có được dẫn xuất của `y = x * x` đối với `x`, đó là `2 * x`.

```{.python .input}
y.backward()
x.grad == 2 * x
```

```{.python .input}
#@tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
#@tab tensorflow
t.gradient(y, x) == 2 * x
```

## Tính toán Gradient của Python Control Flow

Một lợi ích của việc sử dụng sự khác biệt tự động là [** thậm chí nếu**] xây dựng biểu đồ tính toán của (** một hàm yêu cầu đi qua một mê cung của dòng điều khiển Python **) (ví dụ, điều kiện, vòng lặp và các cuộc gọi hàm tùy ý), (** chúng ta vẫn có thể tính toán gradient của biến thể kết quả. **) Trong đoạn sau, lưu ý rằng số lần lặp của vòng lặp `while` và đánh giá câu lệnh `if` cả phụ thuộc vào giá trị của đầu vào `a`.

```{.python .input}
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
#@tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Hãy để chúng tôi tính toán gradient.

```{.python .input}
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
#@tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
#@tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

Bây giờ chúng ta có thể phân tích hàm `f` được xác định ở trên. Lưu ý rằng nó là piecewise tuyến tính trong đầu vào của nó `a`. Nói cách khác, đối với bất kỳ `a` có tồn tại một số vô hướng không đổi `k` sao cho `f(a) = k * a`, trong đó giá trị của `k` phụ thuộc vào `a` đầu vào. Do đó `d / a` cho phép chúng tôi xác minh rằng gradient là chính xác.

```{.python .input}
a.grad == d / a
```

```{.python .input}
#@tab pytorch
a.grad == d / a
```

```{.python .input}
#@tab tensorflow
d_grad == d / a
```

## Tóm tắt

* Các khuôn khổ học sâu có thể tự động hóa việc tính toán các dẫn xuất. Để sử dụng nó, trước tiên chúng ta đính kèm gradient vào các biến đó đối với mà chúng ta mong muốn các dẫn xuất một phần. Sau đó, chúng tôi ghi lại tính toán giá trị mục tiêu của chúng tôi, thực hiện chức năng của nó để truyền ngược và truy cập gradient kết quả.

## Bài tập

1. Tại sao đạo hàm thứ hai đắt hơn nhiều để tính toán so với đạo hàm đầu tiên?
1. Sau khi chạy chức năng để truyền ngược, ngay lập tức chạy lại và xem điều gì sẽ xảy ra.
1. Trong ví dụ dòng điều khiển, nơi chúng ta tính toán đạo hàm của `d` đối với `a`, điều gì sẽ xảy ra nếu chúng ta thay đổi biến `a` thành vectơ hoặc ma trận ngẫu nhiên. Tại thời điểm này, kết quả của phép tính `f(a)` không còn là vô hướng. Điều gì xảy ra với kết quả? Làm thế nào để chúng ta phân tích điều này?
1. Thiết kế lại một ví dụ về việc tìm gradient của luồng điều khiển. Chạy và phân tích kết quả.
1. Hãy để $f(x) = \sin(x)$. Lô $f(x)$ và $\frac{df(x)}{dx}$, nơi sau này được tính toán mà không cần khai thác $f'(x) = \cos(x)$ đó.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:

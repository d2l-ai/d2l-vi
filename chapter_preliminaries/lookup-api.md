# Documentation

:begin_tab:`mxnet`
Do những hạn chế về độ dài của cuốn sách này, chúng tôi không thể giới thiệu mọi chức năng và lớp MXNet duy nhất (và có lẽ bạn sẽ không muốn chúng tôi). Các tài liệu API và các hướng dẫn bổ sung và ví dụ cung cấp nhiều tài liệu ngoài cuốn sách. Trong phần này, chúng tôi cung cấp cho bạn một số hướng dẫn để khám phá API MXNet.
:end_tab:

:begin_tab:`pytorch`
Do những hạn chế về độ dài của cuốn sách này, chúng ta không thể giới thiệu mọi hàm và lớp PyTorch (và có lẽ bạn sẽ không muốn chúng tôi). Các tài liệu API và các hướng dẫn bổ sung và ví dụ cung cấp nhiều tài liệu ngoài cuốn sách. Trong phần này, chúng tôi cung cấp cho bạn một số hướng dẫn để khám phá API PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Do những hạn chế về độ dài của cuốn sách này, chúng ta không thể giới thiệu mọi hàm và lớp TensorFlow (và có lẽ bạn sẽ không muốn chúng tôi). Các tài liệu API và các hướng dẫn bổ sung và ví dụ cung cấp nhiều tài liệu ngoài cuốn sách. Trong phần này, chúng tôi cung cấp cho bạn một số hướng dẫn để khám phá API TensorFlow.
:end_tab:

## Tìm tất cả các hàm và lớp học trong một mô-đun

Để biết các hàm và lớp nào có thể được gọi trong một mô-đun, chúng ta gọi hàm `dir`. Ví dụ, chúng ta có thể (** truy vấn tất cả các thuộc tính trong mô-đun để tạo ra số ngẫu nhiên**):

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

Nói chung, chúng ta có thể bỏ qua các hàm bắt đầu và kết thúc bằng `__` (các đối tượng đặc biệt trong Python) hoặc các hàm bắt đầu bằng một `_` duy nhất (thường là hàm nội bộ). Dựa trên hàm hoặc tên thuộc tính còn lại, chúng ta có thể nguy hiểm đoán rằng mô-đun này cung cấp các phương pháp khác nhau để tạo ra các số ngẫu nhiên, bao gồm lấy mẫu từ phân phối thống nhất (`uniform`), phân phối bình thường (`normal`), và phân phối đa phương thức (`multinomial`). 

## Tìm cách sử dụng các hàm và lớp cụ thể

Đối với các hướng dẫn cụ thể hơn về cách sử dụng một hàm hoặc lớp nhất định, chúng ta có thể gọi hàm `help`. Ví dụ, chúng ta hãy [** khám phá các hướng dẫn sử dụng cho hàm `ones` của tensor**].

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

Từ tài liệu, chúng ta có thể thấy rằng hàm `ones` tạo ra một tensor mới với hình dạng được chỉ định và đặt tất cả các phần tử thành giá trị của 1. Bất cứ khi nào có thể, bạn nên (** chạy một bài kiểm tra nhanh**) để xác nhận giải thích của bạn:

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

Trong sổ ghi chép Jupyter, chúng ta có thể sử dụng `? `để hiển thị tài liệu trong một cửa sổ khác. Ví dụ: `list? `sẽ tạo nội dung gần giống với `help(list)`, hiển thị nó trong một cửa sổ trình duyệt mới. Ngoài ra, nếu chúng ta sử dụng hai dấu hỏi, chẳng hạn như `list?? `, mã Python thực hiện hàm cũng sẽ được hiển thị. 

## Tóm tắt

* Tài liệu chính thức cung cấp rất nhiều mô tả và ví dụ vượt ra ngoài cuốn sách này.
* Chúng ta có thể tra cứu tài liệu cho việc sử dụng API bằng cách gọi các hàm `dir` và `help`, hoặc `? ` and `?? `trong máy tính xách tay Jupyter.

## Bài tập

1. Tra cứu tài liệu cho bất kỳ chức năng hoặc lớp học nào trong khuôn khổ học sâu. Bạn cũng có thể tìm thấy tài liệu trên trang web chính thức của khuôn khổ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

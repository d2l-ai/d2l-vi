# Khởi tạo hoãn lại
:label:`sec_deferred_init`

Cho đến nay, có vẻ như chúng ta đã thoát khỏi sự cẩu thả trong việc thiết lập mạng của mình. Cụ thể, chúng tôi đã làm những điều không trực quan sau đây, có vẻ như chúng không nên hoạt động: 

* Chúng tôi xác định các kiến trúc mạng mà không chỉ định kích thước đầu vào.
* Chúng tôi đã thêm các lớp mà không chỉ định kích thước đầu ra của lớp trước đó.
* Chúng tôi thậm chí “khởi tạo” các tham số này trước khi cung cấp đủ thông tin để xác định có bao nhiêu tham số mô hình của chúng tôi nên chứa.

Bạn có thể ngạc nhiên rằng mã của chúng tôi chạy ở tất cả. Rốt cuộc, không có cách nào khung học sâu có thể cho biết kích thước đầu vào của mạng sẽ là gì. Bí quyết ở đây là framework * defers khởi hóa*, chờ cho đến lần đầu tiên chúng ta truyền dữ liệu thông qua mô hình, để suy ra kích thước của mỗi lớp một cách nhanh chóng. 

Sau đó, khi làm việc với các mạng thần kinh phức tạp, kỹ thuật này sẽ trở nên thuận tiện hơn nữa vì kích thước đầu vào (tức là độ phân giải của một hình ảnh) sẽ ảnh hưởng đến kích thước của mỗi lớp tiếp theo. Do đó, khả năng thiết lập các tham số mà không cần biết, tại thời điểm viết mã, kích thước là gì có thể đơn giản hóa rất nhiều nhiệm vụ chỉ định và sau đó sửa đổi các mô hình của chúng tôi. Tiếp theo, chúng ta đi sâu hơn vào cơ chế khởi tạo. 

## Khởi tạo một mạng

Để bắt đầu, chúng ta hãy khởi tạo một MLP.

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    return net

net = get_net()
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])
```

Tại thời điểm này, mạng không thể biết được kích thước của trọng lượng của lớp đầu vào vì kích thước đầu vào vẫn chưa được biết. Do đó, khung chưa khởi tạo bất kỳ tham số nào. Chúng tôi xác nhận bằng cách cố gắng truy cập các tham số bên dưới.

```{.python .input}
print(net.collect_params)
print(net.collect_params())
```

```{.python .input}
#@tab tensorflow
[net.layers[i].get_weights() for i in range(len(net.layers))]
```

:begin_tab:`mxnet`
Lưu ý rằng trong khi các đối tượng tham số tồn tại, kích thước đầu vào cho mỗi lớp được liệt kê là -1. MXNet sử dụng giá trị đặc biệt -1 để chỉ ra rằng kích thước tham số vẫn chưa được biết. Tại thời điểm này, các nỗ lực truy cập `net[0].weight.data()` sẽ kích hoạt lỗi thời gian chạy nói rằng mạng phải được khởi tạo trước khi các tham số có thể được truy cập. Bây giờ chúng ta hãy xem những gì xảy ra khi chúng ta cố gắng khởi tạo các tham số thông qua chức năng `initialize`.
:end_tab:

:begin_tab:`tensorflow`
Lưu ý rằng mỗi đối tượng lớp tồn tại nhưng trọng lượng trống. Sử dụng `net.get_weights()` sẽ gây ra lỗi vì trọng lượng chưa được khởi tạo.
:end_tab:

```{.python .input}
net.initialize()
net.collect_params()
```

:begin_tab:`mxnet`
Như chúng ta có thể thấy, không có gì thay đổi. Khi không xác định kích thước đầu vào, các cuộc gọi để khởi tạo không thực sự khởi tạo các tham số. Thay vào đó, cuộc gọi này đăng ký vào MXNet mà chúng tôi muốn (và tùy chọn, theo phân phối nào) để khởi tạo các tham số.
:end_tab:

Tiếp theo chúng ta hãy truyền dữ liệu qua mạng để làm cho framework cuối cùng khởi tạo các tham số.

```{.python .input}
X = np.random.uniform(size=(2, 20))
net(X)

net.collect_params()
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]
```

Ngay khi chúng ta biết kích thước đầu vào, 20, khung có thể xác định hình dạng của ma trận trọng lượng của lớp đầu tiên bằng cách cắm giá trị 20. Sau khi nhận ra hình dạng của lớp đầu tiên, khung tiến tới lớp thứ hai, v.v. Thông qua biểu đồ tính toán cho đến khi tất cả các hình dạng được biết đến. Lưu ý rằng trong trường hợp này, chỉ có lớp đầu tiên yêu cầu khởi tạo hoãn lại, nhưng khung khởi tạo tuần tự. Khi tất cả các hình dạng tham số được biết, khung cuối cùng có thể khởi tạo các tham số. 

## Tóm tắt

* Khởi tạo hoãn lại có thể thuận tiện, cho phép khung tự động suy ra các hình dạng tham số, giúp dễ dàng sửa đổi kiến trúc và loại bỏ một nguồn lỗi phổ biến.
* Chúng ta có thể truyền dữ liệu thông qua mô hình để làm cho framework cuối cùng khởi tạo các tham số.

## Bài tập

1. Điều gì xảy ra nếu bạn chỉ định kích thước đầu vào cho lớp đầu tiên nhưng không cho các lớp tiếp theo? Bạn có được khởi tạo ngay lập tức không?
1. Điều gì sẽ xảy ra nếu bạn chỉ định kích thước không phù hợp?
1. Bạn sẽ cần làm gì nếu bạn có đầu vào của chiều chiều khác nhau? Gợi ý: nhìn vào tham số buộc.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/280)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/281)
:end_tab:

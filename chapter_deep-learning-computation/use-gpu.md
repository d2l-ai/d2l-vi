# GPU
:label:`sec_use_gpu`

Năm :numref:`tab_intro_decade`, chúng tôi đã thảo luận về sự tăng trưởng nhanh chóng của tính toán trong hai thập kỷ qua. Tóm lại, hiệu suất GPU đã tăng hệ số 1000 mỗi thập kỷ kể từ năm 2000. Điều này mang lại những cơ hội tuyệt vời nhưng nó cũng cho thấy một nhu cầu đáng kể để cung cấp hiệu suất như vậy. 

Trong phần này, chúng tôi bắt đầu thảo luận về cách khai thác hiệu suất tính toán này cho nghiên cứu của bạn. Đầu tiên bằng cách sử dụng GPU duy nhất và tại một thời điểm sau đó, làm thế nào để sử dụng nhiều GPU và nhiều máy chủ (với nhiều GPU). 

Cụ thể, chúng tôi sẽ thảo luận về cách sử dụng một GPU NVIDIA duy nhất để tính toán. Trước tiên, hãy chắc chắn rằng bạn đã cài đặt ít nhất một GPU NVIDIA. Sau đó, tải xuống [NVIDIA driver and CUDA](https://developer.nvidia.com/cuda-downloads) và làm theo lời nhắc để đặt đường dẫn thích hợp. Khi các chế phẩm này hoàn tất, lệnh `nvidia-smi` có thể được sử dụng để (**xem thông tin card đồ hoạ**).

```{.python .input}
#@tab all
!nvidia-smi
```

:begin_tab:`mxnet`
Bạn có thể nhận thấy rằng tensor MXNet trông gần giống với NumPy `ndarray`. Nhưng có một vài sự khác biệt quan trọng. Một trong những tính năng chính phân biệt MXNet với NumPy là hỗ trợ cho các thiết bị phần cứng đa dạng. 

Trong MXNet, mỗi mảng có một ngữ cảnh. Cho đến nay, theo mặc định, tất cả các biến và tính toán liên quan đã được gán cho CPU. Thông thường, các ngữ cảnh khác có thể là các GPU khác nhau. Mọi thứ có thể nhận được thậm chí hairier khi chúng tôi triển khai công việc trên nhiều máy chủ. Bằng cách gán mảng cho ngữ cảnh một cách thông minh, chúng ta có thể giảm thiểu thời gian truyền dữ liệu giữa các thiết bị. Ví dụ: khi đào tạo mạng thần kinh trên máy chủ có GPU, chúng tôi thường thích các tham số của mô hình sống trên GPU. 

Tiếp theo, chúng ta cần xác nhận rằng phiên bản GPU của MXNet được cài đặt. Nếu phiên bản CPU của MXNet đã được cài đặt, chúng ta cần gỡ cài đặt nó trước. Ví dụ: sử dụng lệnh `pip uninstall mxnet`, sau đó cài đặt phiên bản MXNet tương ứng theo phiên bản CIDA của bạn. Giả sử bạn đã cài đặt CDA 10.0, bạn có thể cài đặt phiên bản MXNet hỗ trợ CDA 10.0 qua `pip install mxnet-cu100`.
:end_tab:

:begin_tab:`pytorch`
Trong PyTorch, mỗi mảng có một thiết bị, chúng ta thường gọi nó như là một bối cảnh. Cho đến nay, theo mặc định, tất cả các biến và tính toán liên quan đã được gán cho CPU. Thông thường, các ngữ cảnh khác có thể là các GPU khác nhau. Mọi thứ có thể nhận được thậm chí hairier khi chúng tôi triển khai công việc trên nhiều máy chủ. Bằng cách gán mảng cho ngữ cảnh một cách thông minh, chúng ta có thể giảm thiểu thời gian truyền dữ liệu giữa các thiết bị. Ví dụ: khi đào tạo mạng thần kinh trên máy chủ có GPU, chúng tôi thường thích các tham số của mô hình sống trên GPU. 

Tiếp theo, chúng ta cần xác nhận rằng phiên bản GPU của PyTorch được cài đặt. Nếu phiên bản CPU của PyTorch đã được cài đặt, chúng ta cần gỡ cài đặt nó trước. Ví dụ: sử dụng lệnh `pip uninstall torch`, sau đó cài đặt phiên bản PyTorch tương ứng theo phiên bản CIDA của bạn. Giả sử bạn đã cài đặt CDA 10.0, bạn có thể cài đặt phiên bản PyTorch hỗ trợ CDA 10.0 qua `pip install torch-cu100`.
:end_tab:

Để chạy các chương trình trong phần này, bạn cần ít nhất hai GPU. Lưu ý rằng điều này có thể là xa hoa đối với hầu hết các máy tính để bàn nhưng nó có thể dễ dàng có sẵn trên đám mây, ví dụ, bằng cách sử dụng các phiên bản đa GPU AWS EC2. Hầu như tất cả các phần khác làm * không* yêu cầu nhiều GPU. Thay vào đó, điều này chỉ đơn giản là để minh họa cách dữ liệu chảy giữa các thiết bị khác nhau. 

## [**Thiết bị vi tính**]

Chúng tôi có thể chỉ định các thiết bị, chẳng hạn như CPU và GPU, để lưu trữ và tính toán. Theo mặc định, hàng chục được tạo trong bộ nhớ chính và sau đó sử dụng CPU để tính toán nó.

:begin_tab:`mxnet`
Trong MXNet, CPU và GPU có thể được chỉ định bởi `cpu()` và `gpu()`. Cần lưu ý rằng `cpu()` (hoặc bất kỳ số nguyên nào trong ngoặc đơn) có nghĩa là tất cả các CPU vật lý và bộ nhớ. Điều này có nghĩa là tính toán của MXNet sẽ cố gắng sử dụng tất cả các lõi CPU. Tuy nhiên, `gpu()` chỉ đại diện cho một thẻ và bộ nhớ tương ứng. Nếu có nhiều GPU, chúng tôi sử dụng `gpu(i)` để đại diện cho GPU $i^\mathrm{th}$ ($i$ bắt đầu từ 0). Ngoài ra, `gpu(0)` và `gpu()` là tương đương.
:end_tab:

:begin_tab:`pytorch`
Trong PyTorch, CPU và GPU có thể được chỉ định bởi `torch.device('cpu')` và `torch.device('cuda')`. Cần lưu ý rằng thiết bị `cpu` có nghĩa là tất cả các CPU vật lý và bộ nhớ. Điều này có nghĩa là các tính toán của PyTorch sẽ cố gắng sử dụng tất cả các lõi CPU. Tuy nhiên, một thiết bị `gpu` chỉ đại diện cho một thẻ và bộ nhớ tương ứng. Nếu có nhiều GPU, chúng tôi sử dụng `torch.device(f'cuda:{i}')` để đại diện cho GPU $i^\mathrm{th}$ ($i$ bắt đầu từ 0). Ngoài ra, `gpu:0` và `gpu` là tương đương.
:end_tab:

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

```{.python .input}
#@tab pytorch
import torch
from torch import nn

torch.device('cpu'), torch.device('cuda'), torch.device('cuda:1')
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')
```

Chúng tôi có thể (** truy vấn số lượng GPU có sẵn.**)

```{.python .input}
npx.num_gpus()
```

```{.python .input}
#@tab pytorch
torch.cuda.device_count()
```

```{.python .input}
#@tab tensorflow
len(tf.config.experimental.list_physical_devices('GPU'))
```

Bây giờ chúng ta [** xác định hai hàm tiện lợi cho phép chúng tôi chạy mã ngay cả khi GPU được yêu cầu không tồn tại.**]

```{.python .input}
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu()] if no GPU exists."""
    devices = [npx.gpu(i) for i in range(npx.num_gpus())]
    return devices if devices else [npx.cpu()]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab pytorch
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

try_gpu(), try_gpu(10), try_all_gpus()
```

```{.python .input}
#@tab tensorflow
def try_gpu(i=0):  #@save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():  #@save
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()
```

## Tensors và GPU

Theo mặc định, hàng chục được tạo trên CPU. Chúng ta có thể [** truy vấn thiết bị nơi tensor được đặt. **]

```{.python .input}
x = np.array([1, 2, 3])
x.ctx
```

```{.python .input}
#@tab pytorch
x = torch.tensor([1, 2, 3])
x.device
```

```{.python .input}
#@tab tensorflow
x = tf.constant([1, 2, 3])
x.device
```

Điều quan trọng cần lưu ý là bất cứ khi nào chúng tôi muốn hoạt động theo nhiều điều khoản, chúng cần phải ở trên cùng một thiết bị. Ví dụ, nếu chúng ta tổng hợp hai chục, chúng ta cần đảm bảo rằng cả hai đối số đều sống trên cùng một thiết bị—nếu không khung sẽ không biết lưu trữ kết quả ở đâu hoặc thậm chí làm thế nào để quyết định nơi thực hiện tính toán. 

### Lưu trữ trên GPU

Có một số cách để [** lưu trữ một tensor trên GPU.**] Ví dụ: chúng ta có thể chỉ định một thiết bị lưu trữ khi tạo tensor. Tiếp theo, chúng ta tạo biến tensor `X` trên `gpu` đầu tiên. Tensor được tạo trên GPU chỉ tiêu thụ bộ nhớ của GPU này. Chúng ta có thể sử dụng lệnh `nvidia-smi` để xem mức sử dụng bộ nhớ GPU. Nói chung, chúng ta cần đảm bảo rằng chúng ta không tạo dữ liệu vượt quá giới hạn bộ nhớ GPU.

```{.python .input}
X = np.ones((2, 3), ctx=try_gpu())
X
```

```{.python .input}
#@tab pytorch
X = torch.ones(2, 3, device=try_gpu())
X
```

```{.python .input}
#@tab tensorflow
with try_gpu():
    X = tf.ones((2, 3))
X
```

Giả sử rằng bạn có ít nhất hai GPU, mã sau sẽ (** tạo tensor ngẫu nhiên trên GPU.** thứ hai)

```{.python .input}
Y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
Y
```

```{.python .input}
#@tab pytorch
Y = torch.rand(2, 3, device=try_gpu(1))
Y
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y
```

### Sao chép

[**Nếu chúng ta muốn tính `X + Y`, chúng ta cần quyết định nơi thực hiện thao tác này.**] Ví dụ, như thể hiện trong :numref:`fig_copyto`, chúng ta có thể chuyển `X` sang GPU thứ hai và thực hiện thao tác ở đó.
*Làm không* chỉ cần thêm `X` và `Y`,
vì điều này sẽ dẫn đến một ngoại lệ. Công cụ thời gian chạy sẽ không biết phải làm gì: nó không thể tìm thấy dữ liệu trên cùng một thiết bị và nó không thành công. Vì `Y` sống trên GPU thứ hai, chúng ta cần phải di chuyển `X` ở đó trước khi chúng ta có thể thêm hai. 

![Copy data to perform an operation on the same device.](../img/copyto.svg)
:label:`fig_copyto`

```{.python .input}
Z = X.copyto(try_gpu(1))
print(X)
print(Z)
```

```{.python .input}
#@tab pytorch
Z = X.cuda(1)
print(X)
print(Z)
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z = X
print(X)
print(Z)
```

Bây giờ [** dữ liệu nằm trên cùng một GPU (cả `Z` và `Y` đều là), chúng ta có thể thêm chúng lên.**]

```{.python .input}
#@tab all
Y + Z
```

:begin_tab:`mxnet`
Hãy tưởng tượng rằng biến `Z` của bạn đã sống trên GPU thứ hai của bạn. Điều gì xảy ra nếu chúng ta vẫn gọi `Z.copyto(gpu(1))`? Nó sẽ tạo một bản sao và phân bổ bộ nhớ mới, mặc dù biến đó đã sống trên thiết bị mong muốn. Có những lúc, tùy thuộc vào môi trường mà mã của chúng tôi đang chạy, hai biến có thể đã sống trên cùng một thiết bị. Vì vậy, chúng tôi muốn tạo một bản sao chỉ khi các biến hiện đang sống trong các thiết bị khác nhau. Trong những trường hợp này, chúng ta có thể gọi `as_in_ctx`. Nếu biến đã sống trong thiết bị được chỉ định thì đây là một no-op. Trừ khi bạn đặc biệt muốn tạo một bản sao, `as_in_ctx` là phương pháp lựa chọn.
:end_tab:

:begin_tab:`pytorch`
Hãy tưởng tượng rằng biến `Z` của bạn đã sống trên GPU thứ hai của bạn. Điều gì sẽ xảy ra nếu chúng ta vẫn gọi `Z.cuda(1)`? Nó sẽ trả về `Z` thay vì tạo một bản sao và phân bổ bộ nhớ mới.
:end_tab:

:begin_tab:`tensorflow`
Hãy tưởng tượng rằng biến `Z` của bạn đã sống trên GPU thứ hai của bạn. Điều gì xảy ra nếu chúng ta vẫn gọi `Z2 = Z` trong cùng một phạm vi thiết bị? Nó sẽ trả về `Z` thay vì tạo một bản sao và phân bổ bộ nhớ mới.
:end_tab:

```{.python .input}
Z.as_in_ctx(try_gpu(1)) is Z
```

```{.python .input}
#@tab pytorch
Z.cuda(1) is Z
```

```{.python .input}
#@tab tensorflow
with try_gpu(1):
    Z2 = Z
Z2 is Z
```

### Ghi chú bên

Mọi người sử dụng GPU để học máy vì họ mong đợi chúng sẽ nhanh chóng. Nhưng chuyển các biến giữa các thiết bị chậm. Vì vậy, chúng tôi muốn bạn chắc chắn 100% rằng bạn muốn làm điều gì đó chậm trước khi chúng tôi cho phép bạn làm điều đó. Nếu khung học sâu chỉ thực hiện bản sao tự động mà không bị rơi sau đó bạn có thể không nhận ra rằng bạn đã viết một số mã chậm. 

Ngoài ra, truyền dữ liệu giữa các thiết bị (CPU, GPU và các máy khác) là thứ chậm hơn nhiều so với tính toán. Nó cũng làm cho việc song song trở nên khó khăn hơn rất nhiều, vì chúng ta phải đợi dữ liệu được gửi (hay đúng hơn là được nhận) trước khi chúng ta có thể tiến hành nhiều thao tác hơn. Đây là lý do tại sao các thao tác sao chép nên được thực hiện cẩn thận. Theo nguyên tắc chung, nhiều hoạt động nhỏ tồi tệ hơn nhiều so với một hoạt động lớn. Hơn nữa, một số hoạt động tại một thời điểm tốt hơn nhiều so với nhiều thao tác đơn lẻ xen kẽ trong mã trừ khi bạn biết những gì bạn đang làm. Đây là trường hợp vì các hoạt động như vậy có thể chặn nếu một thiết bị phải đợi thiết bị kia trước khi nó có thể làm một cái gì đó khác. Nó là một chút giống như đặt hàng cà phê của bạn trong một hàng đợi thay vì đặt hàng trước nó qua điện thoại và phát hiện ra rằng nó đã sẵn sàng khi bạn đang có. 

Cuối cùng, khi chúng ta in hàng chục hoặc chuyển đổi hàng chục sang định dạng NumPy, nếu dữ liệu không nằm trong bộ nhớ chính, khung sẽ sao chép nó vào bộ nhớ chính trước, dẫn đến chi phí truyền bổ sung. Thậm chí tệ hơn, bây giờ nó phải tuân theo khóa thông dịch viên toàn cầu đáng sợ khiến mọi thứ chờ đợi Python hoàn thành. 

## [**Mạng nơ-ron và GPU **]

Tương tự, một mô hình mạng thần kinh có thể chỉ định các thiết bị. Mã sau đặt các tham số mô hình trên GPU.

```{.python .input}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
```

```{.python .input}
#@tab tensorflow
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)])
```

Chúng ta sẽ thấy nhiều ví dụ khác về cách chạy các mô hình trên GPU trong các chương sau, đơn giản vì chúng sẽ trở nên chuyên sâu hơn về mặt tính toán. 

Khi đầu vào là một tensor trên GPU, mô hình sẽ tính toán kết quả trên cùng một GPU.

```{.python .input}
#@tab all
net(X)
```

Hãy để chúng tôi (** xác nhận rằng các tham số mô hình được lưu trữ trên cùng một GPU.**)

```{.python .input}
net[0].weight.data().ctx
```

```{.python .input}
#@tab pytorch
net[0].weight.data.device
```

```{.python .input}
#@tab tensorflow
net.layers[0].weights[0].device, net.layers[0].weights[1].device
```

Nói tóm lại, miễn là tất cả dữ liệu và tham số đều trên cùng một thiết bị, chúng ta có thể học các mô hình một cách hiệu quả. Trong các chương sau, chúng ta sẽ thấy một số ví dụ như vậy. 

## Tóm tắt

* Chúng tôi có thể chỉ định các thiết bị để lưu trữ và tính toán, chẳng hạn như CPU hoặc GPU. Theo mặc định, dữ liệu được tạo trong bộ nhớ chính và sau đó sử dụng CPU để tính toán.
* Khung học sâu yêu cầu tất cả dữ liệu đầu vào để tính toán phải nằm trên cùng một thiết bị, có thể là CPU hoặc cùng một GPU.
* Bạn có thể mất hiệu suất đáng kể bằng cách di chuyển dữ liệu mà không cần quan tâm. Một sai lầm điển hình như sau: tính toán tổn thất cho mỗi minibatch trên GPU và báo cáo lại cho người dùng trên dòng lệnh (hoặc đăng nhập nó trong NumPy `ndarray`) sẽ kích hoạt một khóa thông dịch viên toàn cầu ngăn chặn tất cả các GPU. Tốt hơn là phân bổ bộ nhớ để đăng nhập vào GPU và chỉ di chuyển các bản ghi lớn hơn.

## Bài tập

1. Hãy thử một nhiệm vụ tính toán lớn hơn, chẳng hạn như phép nhân của ma trận lớn và xem sự khác biệt về tốc độ giữa CPU và GPU. Còn một nhiệm vụ với một lượng nhỏ tính toán thì sao?
1. Làm thế nào chúng ta nên đọc và viết các tham số mô hình trên GPU?
1. Đo thời gian cần thiết để tính toán 1000 phép nhân ma trận ma trận của $100 \times 100$ ma trận và ghi lại định mức Frobenius của ma trận đầu ra một kết quả tại một thời điểm so với giữ nhật ký trên GPU và chỉ chuyển kết quả cuối cùng.
1. Đo lường thời gian cần thiết để thực hiện hai phép nhân ma trận ma trận trên hai GPU cùng một lúc so với theo thứ tự trên một GPU. Gợi ý: bạn sẽ thấy tỷ lệ gần như tuyến tính.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/62)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/63)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/270)
:end_tab:

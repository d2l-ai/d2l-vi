# Song song tự động
:label:`sec_auto_para`

Các khung học sâu (ví dụ: MXNet và PyTorch) tự động xây dựng các đồ thị tính toán ở phụ trợ. Sử dụng đồ thị tính toán, hệ thống nhận thức được tất cả các phụ thuộc, và có thể thực hiện một cách chọn lọc nhiều tác vụ không phụ thuộc lẫn nhau song song để cải thiện tốc độ. Ví dụ, :numref:`fig_asyncgraph` trong :numref:`sec_async` khởi tạo hai biến một cách độc lập. Do đó, hệ thống có thể chọn để thực hiện chúng song song. 

Thông thường, một toán tử duy nhất sẽ sử dụng tất cả các tài nguyên tính toán trên tất cả các CPU hoặc trên một GPU duy nhất. Ví dụ, nhà điều hành `dot` sẽ sử dụng tất cả các lõi (và luồng) trên tất cả các CPU, ngay cả khi có nhiều bộ xử lý CPU trên một máy duy nhất. Điều tương tự cũng áp dụng cho một GPU duy nhất. Do đó song song không phải là khá hữu ích cho các máy tính đơn thiết bị. Với nhiều thiết bị mọi thứ quan trọng hơn. Mặc dù song song thường có liên quan nhất giữa nhiều GPU, việc thêm CPU cục bộ sẽ tăng hiệu suất một chút. Ví dụ: xem :cite:`Hadjis.Zhang.Mitliagkas.ea.2016` tập trung vào đào tạo các mô hình thị giác máy tính kết hợp GPU và CPU. Với sự tiện lợi của một khung song song tự động, chúng ta có thể thực hiện cùng một mục tiêu trong một vài dòng mã Python. Rộng hơn, cuộc thảo luận của chúng tôi về tính toán song song tự động tập trung vào tính toán song song bằng cách sử dụng cả CPU và GPU, cũng như sự song song của tính toán và giao tiếp. 

Lưu ý rằng chúng ta cần ít nhất hai GPU để chạy các thí nghiệm trong phần này.

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

## Tính toán song song trên GPU

Chúng ta hãy bắt đầu bằng cách xác định khối lượng công việc tham chiếu để kiểm tra: hàm `run` dưới đây thực hiện 10 phép nhân ma trận trên thiết bị mà chúng tôi lựa chọn bằng cách sử dụng dữ liệu được phân bổ thành hai biến: `x_gpu1` và `x_gpu2`.

```{.python .input}
devices = d2l.try_all_gpus()
def run(x):
    return [x.dot(x) for _ in range(50)]

x_gpu1 = np.random.uniform(size=(4000, 4000), ctx=devices[0])
x_gpu2 = np.random.uniform(size=(4000, 4000), ctx=devices[1])
```

```{.python .input}
#@tab pytorch
devices = d2l.try_all_gpus()
def run(x):
    return [x.mm(x) for _ in range(50)]

x_gpu1 = torch.rand(size=(4000, 4000), device=devices[0])
x_gpu2 = torch.rand(size=(4000, 4000), device=devices[1])
```

:begin_tab:`mxnet`
Bây giờ chúng ta áp dụng chức năng cho dữ liệu. Để đảm bảo rằng bộ nhớ đệm không đóng vai trò trong kết quả, chúng tôi làm nóng các thiết bị bằng cách thực hiện một lần vượt qua một trong hai trong số chúng trước khi đo.
:end_tab:

:begin_tab:`pytorch`
Bây giờ chúng ta áp dụng chức năng cho dữ liệu. Để đảm bảo rằng bộ nhớ đệm không đóng vai trò trong kết quả, chúng tôi làm nóng các thiết bị bằng cách thực hiện một lần vượt qua một trong hai trong số chúng trước khi đo. `torch.cuda.synchronize()` chờ tất cả các hạt nhân trong tất cả các luồng trên thiết bị CIDA hoàn thành. Phải mất một đối số `device`, thiết bị mà chúng ta cần đồng bộ hóa. Nó sử dụng thiết bị hiện tại, được đưa ra bởi `current_device()`, nếu đối số thiết bị là `None` (mặc định).
:end_tab:

```{.python .input}
run(x_gpu1)  # Warm-up both devices
run(x_gpu2)
npx.waitall()  

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
run(x_gpu1)
run(x_gpu2)  # Warm-up all devices
torch.cuda.synchronize(devices[0])
torch.cuda.synchronize(devices[1])

with d2l.Benchmark('GPU1 time'):
    run(x_gpu1)
    torch.cuda.synchronize(devices[0])

with d2l.Benchmark('GPU2 time'):
    run(x_gpu2)
    torch.cuda.synchronize(devices[1])
```

:begin_tab:`mxnet`
Nếu chúng ta loại bỏ câu lệnh `waitall` giữa cả hai tác vụ, hệ thống sẽ tự do song song tính toán trên cả hai thiết bị một cách tự động.
:end_tab:

:begin_tab:`pytorch`
Nếu chúng ta loại bỏ câu lệnh `synchronize` giữa cả hai tác vụ, hệ thống sẽ tự do song song tính toán trên cả hai thiết bị một cách tự động.
:end_tab:

```{.python .input}
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('GPU1 & GPU2'):
    run(x_gpu1)
    run(x_gpu2)
    torch.cuda.synchronize()
```

Trong trường hợp trên, tổng thời gian thực thi nhỏ hơn tổng các bộ phận của nó, vì khung học sâu sẽ tự động lên lịch tính toán trên cả hai thiết bị GPU mà không cần mã phức tạp thay mặt cho người dùng. 

## Tính toán song song và truyền thông

Trong nhiều trường hợp, chúng ta cần di chuyển dữ liệu giữa các thiết bị khác nhau, nói giữa CPU và GPU hoặc giữa các GPU khác nhau. Ví dụ, điều này xảy ra khi chúng ta muốn thực hiện tối ưu hóa phân tán, nơi chúng ta cần tổng hợp các gradient trên nhiều thẻ gia tốc. Chúng ta hãy mô phỏng điều này bằng cách tính toán trên GPU và sau đó sao chép kết quả trở lại CPU.

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
def copy_to_cpu(x, non_blocking=False):
    return [y.to('cpu', non_blocking=non_blocking) for y in x]

with d2l.Benchmark('Run on GPU1'):
    y = run(x_gpu1)
    torch.cuda.synchronize()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    torch.cuda.synchronize()
```

:begin_tab:`mxnet`
Điều này có phần không hiệu quả. Lưu ý rằng chúng ta đã có thể bắt đầu sao chép các phần của `y` vào CPU trong khi phần còn lại của danh sách vẫn đang được tính toán. Tình huống này xảy ra, ví dụ, khi chúng ta tính toán gradient trên một minibatch. Độ dốc của một số tham số sẽ có sẵn sớm hơn so với các tham số khác. Do đó, nó hoạt động để lợi thế của chúng tôi để bắt đầu sử dụng băng thông bus PCI-Express trong khi GPU vẫn đang chạy. Loại bỏ `waitall` giữa cả hai phần cho phép chúng tôi mô phỏng kịch bản này.
:end_tab:

:begin_tab:`pytorch`
Điều này có phần không hiệu quả. Lưu ý rằng chúng ta đã có thể bắt đầu sao chép các phần của `y` vào CPU trong khi phần còn lại của danh sách vẫn đang được tính toán. Tình huống này xảy ra, ví dụ, khi chúng ta tính toán gradient (backprop) trên một minibatch. Độ dốc của một số tham số sẽ có sẵn sớm hơn so với các tham số khác. Do đó, nó hoạt động để lợi thế của chúng tôi để bắt đầu sử dụng băng thông bus PCI-Express trong khi GPU vẫn đang chạy. Trong PyTorch, một số chức năng như `to()` và `copy_()` thừa nhận một đối số `non_blocking` rõ ràng, cho phép đồng bộ hóa người gọi bỏ qua khi không cần thiết. Đặt `non_blocking=True` cho phép chúng tôi mô phỏng kịch bản này.
:end_tab:

```{.python .input}
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```

```{.python .input}
#@tab pytorch
with d2l.Benchmark('Run on GPU1 and copy to CPU'):
    y = run(x_gpu1)
    y_cpu = copy_to_cpu(y, True)
    torch.cuda.synchronize()
```

Tổng thời gian cần thiết cho cả hai hoạt động là (như mong đợi) ít hơn tổng các bộ phận của chúng. Lưu ý rằng tác vụ này khác với tính toán song song vì nó sử dụng một tài nguyên khác: bus giữa CPU và GPU. Trên thực tế, chúng ta có thể tính toán trên cả hai thiết bị và giao tiếp, tất cả cùng một lúc. Như đã nói ở trên, có sự phụ thuộc giữa tính toán và giao tiếp: `y[i]` phải được tính toán trước khi nó có thể được sao chép vào CPU. May mắn thay, hệ thống có thể sao chép `y[i-1]` trong khi tính toán `y[i]` để giảm tổng thời gian chạy. 

Chúng tôi kết luận với một minh họa của biểu đồ tính toán và phụ thuộc của nó cho một MLP hai lớp đơn giản khi đào tạo trên CPU và hai GPU, như được mô tả trong :numref:`fig_twogpu`. Sẽ khá đau đớn khi lên lịch chương trình song song kết quả từ việc này bằng tay. Đây là nơi thuận lợi để có một phụ trợ điện toán dựa trên đồ thị để tối ưu hóa. 

![The computational graph and its dependencies of a two-layer MLP on a CPU and two GPUs.](../img/twogpu.svg)
:label:`fig_twogpu`

## Tóm tắt

* Các hệ thống hiện đại có nhiều thiết bị khác nhau, chẳng hạn như nhiều GPU và CPU. Chúng có thể được sử dụng song song, không đồng bộ. 
* Các hệ thống hiện đại cũng có nhiều tài nguyên khác nhau để liên lạc, chẳng hạn như PCI Express, lưu trữ (điển hình là ổ đĩa trạng thái rắn hoặc qua mạng), và băng thông mạng. Chúng có thể được sử dụng song song cho hiệu quả cao điểm. 
* Các phụ trợ có thể cải thiện hiệu suất thông qua tính toán song song tự động và giao tiếp. 

## Bài tập

1. Tám thao tác đã được thực hiện trong hàm `run` được xác định trong phần này. Không có sự phụ thuộc giữa chúng. Thiết kế một thí nghiệm để xem khung học sâu sẽ tự động thực hiện chúng song song hay không.
1. Khi khối lượng công việc của một nhà điều hành cá nhân đủ nhỏ, song song có thể giúp đỡ ngay cả trên một CPU hoặc GPU duy nhất. Thiết kế một thí nghiệm để xác minh điều này. 
1. Thiết kế một thí nghiệm sử dụng tính toán song song trên CPU, GPU và giao tiếp giữa cả hai thiết bị.
1. Sử dụng trình gỡ lỗi như [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) của NVIDIA để xác minh rằng mã của bạn có hiệu quả. 
1. Thiết kế các tác vụ tính toán bao gồm các phụ thuộc dữ liệu phức tạp hơn và chạy thử nghiệm để xem liệu bạn có thể có được kết quả chính xác trong khi cải thiện hiệu suất hay không.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/362)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1681)
:end_tab:

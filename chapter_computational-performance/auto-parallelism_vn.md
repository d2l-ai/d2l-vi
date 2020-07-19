<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Automatic Parallelism
-->

# *dịch tiêu đề phía trên*
:label:`sec_auto_para`

<!--
MXNet automatically constructs computational graphs at the backend.
Using a computational graph, the system is aware of all the dependencies, and can selectively execute multiple non-interdependent tasks in parallel to improve speed.
For instance, :numref:`fig_asyncgraph` in :numref:`sec_async` initializes two variables independently.
Consequently the system can choose to execute them in parallel.
-->

*dịch đoạn phía trên*


<!--
Typically, a single operator will use all the computational resources on all CPUs or on a single GPU.
For example, the `dot` operator will use all cores (and threads) on all CPUs, even if there are multiple CPU processors on a single machine.
The same applies to a single GPU.
Hence parallelization is not quite so useful single-device computers.
With multiple devices things matter more.
While parallelization is typically most relevant between multiple GPUs, adding the local CPU will increase performance slightly.
See e.g., :cite:`Hadjis.Zhang.Mitliagkas.ea.2016` for a paper that focuses on training computer vision models combining a GPU and a CPU.
With the convenience of an automatically parallelizing framework we can accomplish the same goal in a few lines of Python code.
More broadly, our discussion of automatic parallel computation focuses on parallel computation using both CPUs and GPUs, as well as the parallelization of computation and communication.
We begin by importing the required packages and modules. Note that we need at least one GPU to run the experiments in this section.
-->

*dịch đoạn phía trên*


```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Parallel Computation on CPUs and GPUs
-->

## Tính toán song song trên CPU và GPU

<!--
Let us start by defining a reference workload to test - the `run` function below performs 10 matrix-matrix multiplications 
on the device of our choosing using data allocated into two variables, `x_cpu` and `x_gpu`.
-->

Ta hãy bắt đầu bằng việc định nghĩa một khối lượng công việc tham khảo để kiểm thử. 
Hàm `run` dưới đây thực hiện 10 phép nhân ma trận trên thiết bị mà chúng ta lựa chọn bằng cách sử dụng dữ liệu được lưu ở hai biến `x_cpu` và `x_gpu`.


```{.python .input}
def run(x):
    return [x.dot(x) for _ in range(10)]

x_cpu = np.random.uniform(size=(2000, 2000))
x_gpu = np.random.uniform(size=(6000, 6000), ctx=d2l.try_gpu())
```


<!--
Now we apply the function to the data.
To ensure that caching does not play a role in the results we warm up the devices by performing a single pass on each of them prior to measuring.
-->

Bây giờ ta sẽ gọi hàm với dữ liệu.
Để chắc chắn rằng bộ nhớ đệm không ảnh hưởng đến kết quả, ta khởi động các thiết bị bằng việc thực hiện một lượt tính cho mỗi biến trước khi bắt đầu đo lường.

```{.python .input}
run(x_cpu)  # Warm-up both devices
run(x_gpu)
npx.waitall()  

with d2l.Benchmark('CPU time'):
    run(x_cpu)
    npx.waitall()

with d2l.Benchmark('GPU time'):
    run(x_gpu)
    npx.waitall()
```


<!--
If we remove the `waitall()` between both tasks the system is free to parallelize computation on both devices automatically.
-->

Nếu ta bỏ `waitall()` giữa hai tác vụ thì hệ thống sẽ tự động song song hóa việc tính toán trên cả hai thiết bị.


```{.python .input}
with d2l.Benchmark('CPU & GPU'):
    run(x_cpu)
    run(x_gpu)
    npx.waitall()
```


<!--
In the above case the total execution time is less than the sum of its parts, since MXNet automatically schedules computation on 
both CPU and GPU devices without the need for sophisticated code on behalf of the user. 
-->

Trong trường hợp phía trên, thời gian thi hành toàn bộ các tác vụ ít hơn tổng thời gian thi hành từng tác vụ riêng lẻ, bởi vì MXNet tự động định thời việc tính toán trên cả CPU và GPU mà không đòi hỏi người dùng phải cung cấp các đoạn mã phức tạp.


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Parallel Computation and Communication
-->

## Tính toán và Giao tiếp Song song


<!--
In many cases we need to move data between different devices, say between CPU and GPU, or between different GPUs.
This occurs e.g., when we want to perform distributed optimization where we need to aggregate the gradients over multiple accelerator cards.
Let us simulate this by computing on the GPU and then copying the results back to the CPU.
-->

Trong nhiều trường hợp ta cần di chuyển dữ liệu giữa các thiết bị như CPU và GPU, hoặc giữa các GPU với nhau.
Điều này xảy ra, chẳng hạn như khi cần phải tổng hợp các gradient trên nhiều GPU khi ta muốn thực hiện tối ưu hóa phân tán.
Hãy cùng mô phỏng điều này bằng việc tính toán trên GPU và sau đó sao chép kết quả trở lại CPU.


```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

with d2l.Benchmark('Run on GPU'):
    y = run(x_gpu)
    npx.waitall()

with d2l.Benchmark('Copy to CPU'):
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```


<!--
This is somewhat inefficient. Note that we could already start copying parts of `y` to the CPU while the remainder of the list is still being computed.
This situatio occurs, e.g., when we compute the (backprop) gradient on a minibatch.
The gradients of some of the parameters will be available earlier than that of others.
Hence it works to our advantage to start using PCI-Express bus bandwidth while the GPU is still running.
Removing `waitall` between both parts allows us to simulate this scenario.
-->

Điều này hơi kém hiệu quả một chút. Lưu ý rằng ta có thể bắt đầu sao chép một vài phần đã tính xong của `y` đến CPU trong khi các phần còn lại của `y` vẫn đang được tính toán.
Tình huống này có thể xảy ra khi ta tính toán gradient (lan truyền ngược) trên một minibatch.
Gradient của một vài tham số sẽ được tính xong sớm hơn so với các tham số khác.
Do đó sẽ có lợi nếu ta bắt đầu truyền dữ liệu về bằng bus băng thông PCI-Express trong khi GPU vẫn còn đang chạy.
Việc bỏ đi `waitall` giữa các phần cho phép ta mô phỏng tình huống này.


```{.python .input}
with d2l.Benchmark('Run on GPU and copy to CPU'):
    y = run(x_gpu)
    y_cpu = copy_to_cpu(y)
    npx.waitall()
```


<!--
The total time required for both operations is (as expected) significantly less than the sum of their parts.
Note that this task is different from parallel computation as it uses a different resource: the bus between CPU and GPUs.
In fact, we could compute on both devices and communicate, all at the same time.
As noted above, there is a dependency between computation and communication: `y[i]` must be computed before it can be copied to the CPU.
Fortunately, the system can copy `y[i-1]` while computing `y[i]` to reduce the total running time.
-->

Thời gian cần cho cả hai thao tác thì (như mong đợi) ít hơn hẳn so với tổng thời gian thực hiện từng thao tác đơn lẻ.
Lưu ý rằng tác vụ này khác với việc tính toán song song bởi nó sử dụng một tài nguyên khác: bus giữa CPU và GPU.
Thực tế, ta có thể vừa tính toán và giao tiếp trên cả hai thiết bị cùng một lúc.
Như đã lưu ý phía trên, có một sự phụ thuộc giữa việc tính toán và giao tiếp: `y[i]` phải được tính xong trước khi ta có thể sao chép nó qua CPU.
May mắn thay, hệ thống có thể sao chép `y[i-1]` trong khi tính toán `y[i]` để giảm thiểu tổng thời gian chạy.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
We conclude with an illustration of the computational graph and its dependencies for a simple two-layer MLP when training on a CPU and two GPUs, as depicted in :numref:`fig_twogpu`.
It would be quite painful to schedule the parallel program resulting from this manually.
This is where it is advantageous to have a graph based compute backend for optimization.
-->

Để tổng kết phần này, ta xét một ví dụ minh hoạ đồ thị tính toán và các quan hệ phụ thuộc của nó trong một mạng MLP hai tầng đơn giản khi huấn luyện trên một CPU và hai GPU, như miêu tả trong :numref:`fig_twogpu`.
Việc tự định thời chương trình tính toán song song từ mô tả trên là khá vất vả.
Do đó, đây chính là cơ hội thuận lợi để sử dụng back-end tính toán dựa trên đồ thị để tối ưu.

<!--
![Two layer MLP on a CPU and 2 GPUs.](../img/twogpu.svg)
-->

![Mạng MLP hai tầng trên một CPU và hai GPU](../img/twogpu.svg)
:label:`fig_twogpu`


## Tóm tắt


<!--
* Modern systems have a variety of devices, such as multiple GPUs and CPUs. They can be used in parallel, asynchronously. 
* Modern systems also have a variety of resources for communication, such as PCI Express, storage (typically SSD or via network), and network bandwidth. They can be used in parallel for peak efficiency. 
* The backend can improve performance through through automatic parallel computation and communication. 
-->

* Các hệ thống hiện đại thường bao gồm nhiều thiết bị, ví dụ như nhiều GPU và CPU. Các thiết bị này có thể được sử dụng song song, một cách bất đồng bộ.
* Các hệ thống hiện đại thường cũng có nhiều tài nguyên để giao tiếp, ví dụ như kết nối PCI Express, bộ nhớ (thường là SSD hoặc thông qua mạng), và băng thông mạng. Chúng có thể được sử dụng song song để đạt hiệu năng tối đa.
* Back-end có thể cải thiện hiệu năng thông qua việc tự động tính toán và giao tiếp song song.


## Bài tập


<!--
1. 10 operations were performed in the `run` function defined in this section. There are no dependencies between them. Design an experiment to see if MXNet will automatically execute them in parallel.
2. When the workload of an individual operator is sufficiently small, parallelization can help even on a single CPU or GPU. Design an experiment to verify this. 
3. Design an experiment that uses parallel computation on CPU, GPU and communication between both devices.
4. Use a debugger such as NVIDIA's Nsight to verify that your code is efficient. 
5. Designing computation tasks that include more complex data dependencies, and run experiments to see if you can obtain the correct results while improving performance.
-->

1. Có 10 thao tác được thực hiện trong hàm `run` đã được định nghĩa trong phần này. Giữa chúng không có bất cứ quan hệ phụ thuộc nào. Thiết kế một thí nghiệm để xem liệu MXNet có tự động thực thi các thao tác này một cách song song.
2. Khi khối lượng công việc của một thao tác đủ nhỏ, song song hoá có thể hữu ích ngay cả khi chạy trên CPU hay GPU đơn. Thiết kế một thí nghiệm để kiểm chứng.
3. Thiết kế một thí nghiệm sử dụng tính toán song song trên CPU, GPU và giao tiếp giữa cả hai thiết bị.
4. Sử dụng một trình gỡ lỗi (*debugger*) như Nsight của NVIDIA để kiểm chứng rằng đoạn mã của bạn hoạt động hiệu quả.
5. Thiết kế các tác vụ tính toán chứa nhiều dữ liệu có quan hệ phụ thuộc phức tạp hơn nữa, và thực hiện thí nghiệm để xem rằng liệu bạn có thể thu lại kết quả đúng trong khi vẫn cải thiện hiệu năng.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2382)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.
Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* 

<!-- Phần 2 -->
* Trần Yến Thy
* Lê Khắc Hồng Phúc
* Phạm Minh Đức

<!-- Phần 3 -->
* Trần Yến Thy
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Phạm Minh Đức
 
<!-- Phần 4 -->
* Đỗ Trường Giang

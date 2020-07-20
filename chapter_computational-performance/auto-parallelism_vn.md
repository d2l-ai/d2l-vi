<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Automatic Parallelism
-->

# Song song hóa Tự động
:label:`sec_auto_para`

<!--
MXNet automatically constructs computational graphs at the backend.
Using a computational graph, the system is aware of all the dependencies, and can selectively execute multiple non-interdependent tasks in parallel to improve speed.
For instance, :numref:`fig_asyncgraph` in :numref:`sec_async` initializes two variables independently.
Consequently the system can choose to execute them in parallel.
-->

MXNet tự động xây dựng các đồ thị tính toán ở back-end.
Sử dụng đồ thị tính toán, hệ thống biết được tất cả các thành phần phụ thuộc và có thể thực hiện song song có chọn lọc các tác vụ không liên quan đến nhau để cải thiện tốc độ.
Chẳng hạn, :numref:`fig_asyncgraph` in :numref:`sec_async` khởi tạo hai biến độc lập.
Do đó hệ thống có thể chọn để thực hiện chúng song song với nhau.

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

Thông thường, một toán tử đơn sẽ sử dụng toàn bộ tài nguyên tính toán trên tất cả các CPU hoặc trên một CPU đơn.
Chẳng hạn như toán tử `dot` sẽ sử dụng tất cả các nhân (và các luồng) của toàn bộ các CPUs, thậm chí là nhiều bộ vi xử lý trên một máy tính nếu có.
Điều tương tự cũng xảy ra trên một bộ GPU đơn.
Do đó việc song song hóa không thật sự hữu dụng mấy với các máy tính đơn xử lý. 
VỚi các thiết bị đa xử lý thì nó lại thật sự có giá trị hơn nhiều.
Trong khi thực hiện song song hóa thường liên quan nhất giữa nhiều GPU, sử dụng thêm các vi xử lý CPU cục bộ trên máy sẽ tăng hiệu năng tính toán lên chút đỉnh.
Tham khảo :cite:`Hadjis.Zhang.Motliagkas.ea.2016`, một bài báo tập trung về việc huấn luyện mô hình thị giác máy tính kết hợp một GPU và một CPU.
Với sự thuận tiện của một framework cho phép song song hóa một cách tự động, ta có thể thực hiện cùng mục tiêu đó chỉ với vài dòng mã lệnh Python.
Mở rộng hơn, thảo luận của chúng ta về tính toán song song tự động tập trung vào tính toán song song sử dụng cả CPUs và GPUs, cũng như tính toán và truyền thông tin song song.
Chúng ta bắt đầu bằng việc nhập các gói thư viện và mô-đun cần thiết. Lưu ý rằng chúng ta cần ít nhất một GPU để chạy các thử nghiệm trong phần này.

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

## *dịch tiêu đề phía trên*

<!--
Let us start by defining a reference workload to test - the `run` function below performs 10 matrix-matrix multiplications 
on the device of our choosing using data allocated into two variables, `x_cpu` and `x_gpu`.
-->

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Parallel Computation and Communication
-->

## *dịch tiêu đề phía trên*


<!--
In many cases we need to move data between different devices, say between CPU and GPU, or between different GPUs.
This occurs e.g., when we want to perform distributed optimization where we need to aggregate the gradients over multiple accelerator cards.
Let us simulate this by computing on the GPU and then copying the results back to the CPU.
-->

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
We conclude with an illustration of the computational graph and its dependencies for a simple two-layer MLP when training on a CPU and two GPUs, as depicted in :numref:`fig_twogpu`.
It would be quite painful to schedule the parallel program resulting from this manually.
This is where it is advantageous to have a graph based compute backend for optimization.
-->

*dịch đoạn phía trên*

<!--
![Two layer MLP on a CPU and 2 GPUs.](../img/twogpu.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/twogpu.svg)
:label:`fig_twogpu`


## Tóm tắt


<!--
* Modern systems have a variety of devices, such as multiple GPUs and CPUs. They can be used in parallel, asynchronously. 
* Modern systems also have a variety of resources for communication, such as PCI Express, storage (typically SSD or via network), and network bandwidth. They can be used in parallel for peak efficiency. 
* The backend can improve performance through through automatic parallel computation and communication. 
-->

*dịch đoạn phía trên*


## Bài tập


<!--
1. 10 operations were performed in the `run` function defined in this section. There are no dependencies between them. Design an experiment to see if MXNet will automatically execute them in parallel.
2. When the workload of an individual operator is sufficiently small, parallelization can help even on a single CPU or GPU. Design an experiment to verify this. 
3. Design an experiment that uses parallel computation on CPU, GPU and communication between both devices.
4. Use a debugger such as NVIDIA's Nsight to verify that your code is efficient. 
5. Designing computation tasks that include more complex data dependencies, and run experiments to see if you can obtain the correct results while improving performance.
-->

*dịch đoạn phía trên*

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
* Nguyễn Mai Hoàng Long
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

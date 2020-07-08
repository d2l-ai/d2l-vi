<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Asynchronous Computation
-->

# Tính toán Bất đồng bộ
:label:`sec_async`

<!--
Today's computers are highly parallel systems, consisting of multiple CPU cores (often multiple threads per core), multiple processing elements per GPU and often multiple GPUs per device.
In short, we can process many different things at the same time, often on different devices.
Unfortunately Python is not a great way of writing parallel and asynchronous code, at least not with some extra help.
After all, Python is single-threaded and this is unlikely to change in the future.
Deep learning frameworks such as MXNet and TensorFlow utilize an asynchronous programming model to improve performance (PyTorch uses Python's own scheduler leading to a different performance trade-off).
Hence, understanding how asynchronous programming works helps us to develop more efficient programs, by proactively reducing computational requirements and mutual dependencies.
This allows us to reduce memory overhead and increase processor utilization.
We begin by importing the necessary libraries.
-->

Máy tính ngày nay là các hệ thống song song, bao gồm nhiều lõi CPU (mỗi lõi thường có nhiều luồng),
mỗi GPU chứa nhiều thành phần xử lý và mỗi máy thường bao gồm nhiều GPU.
Nói ngắn gọn, ta có thể xử lý nhiều việc cùng một lúc, trên nhiều thiết bị khác nhau.
Tiếc thay, Python không phải là một ngôn ngữ phù hợp để viết mã tính toán song song và bất đồng bộ khi không có sự trợ giúp từ bên ngoài.
Xét cho cùng, Python là ngôn ngữ đơn luồng, và có lẽ trong tương lai sẽ không có gì thay đổi.
Các framework học sâu như MXNet và TensorFlow tận dụng mô hình lập trình bất đồng bộ để cải thiện hiệu năng (PyTorch sử dụng tính năng định thời của chính Python, dẫn tới việc đánh đổi hiệu năng).
Do đó, hiểu cách lập trình bất đồng bộ hoạt động giúp ta phát triển các chương trình hiệu quả hơn bằng cách chủ động giảm thiểu yêu cầu tính toán và các quan hệ phụ thuộc tương hỗ.
Việc này cho phép ta giảm tổng chi phí và tăng khả năng sử dụng khối xử lý.
Ta bắt đầu bằng việc nhập các thư viện cần thiết.


```{.python .input  n=1}
from d2l import mxnet as d2l
import numpy, os, subprocess
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

<!--
## Asynchrony via Backend
-->

## Bất đồng bộ qua Back-end

<!--
For a warmup consider the following toy problem - we want to generate a random matrix and multiply it.
Let us do that both in NumPy and in MXNet NP to see the difference.
-->

Để khởi động, hãy cùng xét một bài toán nhỏ - ta muốn sinh ra một ma trận ngẫu nhiên và nhân nó lên nhiều lần.
Hãy thực hiện trên cả Numpy và trên MXNet NP để xem xét sự khác nhau.


```{.python .input  n=2}
with d2l.Benchmark('numpy'):
    for _ in range(10):
        a = numpy.random.normal(size=(1000, 1000))
        b = numpy.dot(a, a)

with d2l.Benchmark('mxnet.np'):
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
```


<!--
This is orders of magnitude faster.
At least it seems to be so.
Since both are executed on the same processor something else must be going on.
Forcing MXNet to finish all computation prior to returning shows what happened previously: computation is being executed by the backend while the frontend returns control to Python.
-->

Kết quả trên được sắp xếp theo tốc độ.
Ít nhất có vẻ là như vậy.
Do cả hai thư viện đều được thực hiện trên một bộ xử lý, chắc hẳn phải có gì đó ảnh hướng đến kết quả.
Nếu bắt buộc MXNet phải hoàn thành toàn bộ tính toán trước khi trả về kết quả, ta có thể thấy rõ điều gì đã xảy ra ở trên: phần tính toán được thực hiện bởi back-end trong khi front-end trả lại quyền điều khiển cho Python.

```{.python .input  n=3}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```


<!--
Broadly speaking, MXNet has a frontend for direct interaction with the users, e.g., via Python, as well as a backend used by the system to perform the computation.
The backend possesses its own threads that continuously collect and execute queued tasks.
Note that for this to work the backend must be able to keep track of the dependencies between various steps in the computational graph.
Hence it is ony possible to parallelize operations that do not depend on each other.
-->

Nhìn chung, MXNet có front-end cho phép tương tác trực tiếp với người dùng thông qua Python, cũng như back-end được sử dụng bởi hệ thống nhằm thực hiện nhiệm vụ tính toán.
Back-end có các luồng xử lý riêng liên tục tập hợp và thực hiện các tác vụ trong hàng đợi.
Chú ý rằng, back-end cần có khả năng theo dõi quan hệ phụ thuộc giữa nhiều bước khác nhau trong đồ thị tính toán để có thể hoạt động.
Do đó ta chỉ có thể song song hoá các thao tác không phụ thuộc lẫn nhau.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
As shown in :numref:`fig_frontends`, users can write MXNet programs in various frontend languages, such as Python, R, Scala and C++.
Regardless of the front-end programming language used, the execution of MXNet programs occurs primarily in the back-end of C++ implementations.
Operations issued by the frontend language are passed on to the backend for execution.
The backend manages its own threads that continuously collect and execute queued tasks.
Note that for this to work the backend must be able to keep track of the dependencies between various steps in the computational graph.
That is, it is not possible to parallelize operations that depend on each other.
-->

*dịch đoạn phía trên*

<!--
![Programming Frontends.](../img/frontends.png)
-->

![*dịch chú thích ảnh phía trên*](../img/frontends.png)
:width:`300px`
:label:`fig_frontends`


<!--
Let us look at another toy example to understand the dependency graph a bit better.
-->

*dịch đoạn phía trên*


```{.python .input  n=4}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

<!--
![Dependencies.](../img/asyncgraph.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/asyncgraph.svg)
:label:`fig_asyncgraph`


<!--
The code snippet above is also illustrated in :numref:`fig_asyncgraph`.
Whenever the Python frontend thread executes one of the first three statements, it simply returns the task to the backend queue.
When the last statement’s results need to be printed, the Python frontend thread will wait for the C++ backend thread to finish computing result of the variable `z`.
One benefit of this design is that the Python frontend thread does not need to perform actual computations.
Thus, there is little impact on the program’s overall performance, regardless of Python’s performance.
:numref:`fig_threading` illustrates how frontend and backend interact.
-->

*dịch đoạn phía trên*

<!--
![Frontend and Backend.](../img/threading.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/threading.svg)
:label:`fig_threading`

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Barriers and Blockers
-->

## *dịch tiêu đề phía trên*

<!--
There are a number of operations that will force Python to wait for completion:
* Most obviously `npx.waitall()` waits until all computation has completed, regardless of when the compute instructions were issued.
In practice it is a bad idea to use this operator unless absolutely necessary since it can lead to poor performance.
* If we just want to wait until a specific variable is available we can call `z.wait_to_read()`.
In this case MXNet blocks return to Python until the variable `z` has been computed. Other computation may well continue afterwards.
-->

*dịch đoạn phía trên*

<!--
Let us see how this works in practice:
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
with d2l.Benchmark('waitall'):
    b = np.dot(a, a)
    npx.waitall()

with d2l.Benchmark('wait_to_read'):
    b = np.dot(a, a)
    b.wait_to_read()
```


<!--
Both operations take approximately the same time to complete.
Besides the obvious blocking operations we recommend that the reader is aware of *implicit* blockers.
Printing a variable clearly requires the variable to be available and is thus a blocker.
Lastly, conversions to NumPy via `z.asnumpy()` and conversions to scalars via `z.item()` are blocking, since NumPy has no notion of asynchrony.
It needs access to the values just like the `print` function.
Copying small amounts of data frequently from MXNet's scope to NumPy and back can destroy performance of an otherwise efficient code, 
since each such operation requires the compute graph to evaluate all intermediate results needed to get the relevant term *before* anything else can be done.
-->

*dịch đoạn phía trên*


```{.python .input  n=7}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
## Improving Computation
-->

## *dịch tiêu đề phía trên*

<!--
On a heavily multithreaded system (even regular laptops have 4 threads or more and on multi-socket servers this number can exceed 256) the overhead of scheduling operations can become significant.
This is why it is highly desirable to have computation and scheduling occur asynchronously and in parallel.
To illustrate the benefit of doing this let us see what happens if we increment a variable by 1 multiple times, both in sequence or asynchronously.
We simulate synchronous execution by inserting a `wait_to_read()` barrier in between each addition.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
with d2l.Benchmark('synchronous'):
    for _ in range(1000):
        y = x + 1
        y.wait_to_read()

with d2l.Benchmark('asynchronous'):
    for _ in range(1000):
        y = x + 1
    y.wait_to_read()
```


<!--
A slightly simplified interaction between the Python front-end thread and the C++ back-end thread can be summarized as follows:
-->

*dịch đoạn phía trên*

<!--
1. The front-end orders the back-end to insert the calculation task `y = x + 1` into the queue.
2. The back-end then receives the computation tasks from the queue and performs the actual computations.
3. The back-end then returns the computation results to the front-end.
-->

*dịch đoạn phía trên*

<!--
Assume that the durations of these three stages are $t_1, t_2$ and $t_3$, respectively.
If we do not use asynchronous programming, the total time taken to perform 1000 computations is approximately $1000 (t_1+ t_2 + t_3)$.
If asynchronous programming is used, the total time taken to perform 1000 computations can be reduced to $t_1 + 1000 t_2 + t_3$ (assuming $1000 t_2 > 999t_1$), 
since the front-end does not have to wait for the back-end to return computation results for each loop.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Improving Memory Footprint
-->

## *dịch tiêu đề phía trên*

<!--
Imagine a situation where we keep on inserting operations into the backend by executing Python code on the frontend.
For instance, the frontend might insert a large number of minibatch tasks within a very short time.
After all, if no meaningful computation happens in Python this can be done quite quickly.
If each of these tasks can be launched quickly at the same time this may cause a spike in memory usage.
Given a finite amount of memory available on GPUs (and even on CPUs) this can lead to resource contention or even program crashes.
Some readers might have noticed that previous training routines made use of synchronization methods such as `item` or even `asnumpy`.
-->

*dịch đoạn phía trên*

<!--
We recommend to use these operations carefully, e.g., for each minibatch, such as to balance computational efficiency and memory footprint.
To illustrate what happens let us implement a simple training loop for a deep network and measure its memory consumption and timing.
Below is the mock data generator and deep network.
-->

*dịch đoạn phía trên*


```{.python .input  n=10}
def data_iter():
    timer = d2l.Timer()
    num_batches, batch_size = 150, 1024
    for i in range(num_batches):
        X = np.random.normal(size=(batch_size, 512))
        y = np.ones((batch_size,))
        yield X, y
        if (i + 1) % 50 == 0:
            print(f'batch {i + 1}, time {timer.stop():.4f} sec')

net = nn.Sequential()
net.add(nn.Dense(2048, activation='relu'),
        nn.Dense(512, activation='relu'), nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd')
loss = gluon.loss.L2Loss()
```


<!--
Next we need a tool to measure the memory footprint of our code. We use a relatively primitive `ps` call to accomplish this (note that the latter only works on Linux and MacOS).
For a much more detailed analysis of what is going on here use e.g., Nvidia's [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) or Intel's [vTune](https://software.intel.com/en-us/vtune).
-->

*dịch đoạn phía trên*


```{.python .input  n=12}
def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3
```


<!--
Before we can begin testing we need to initialize the parameters of the network and process one batch.
Otherwise it would be tricky to see what the additional memory consumption is.
See :numref:`sec_deferred_init` for further details related to initialization.
-->

*dịch đoạn phía trên*


```{.python .input  n=13}
for X, y in data_iter():
    break
loss(y, net(X)).wait_to_read()
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
To ensure that we do not overflow the task buffer on the backend we insert a `wait_to_read` call for the loss function at the end of each loop.
This forces the forward pass to complete before a new forward pass is commenced.
Note that a (possibly more elegant) alternative would have been to track the loss in a scalar variable and to force a barrier via the `item` call.
-->

*dịch đoạn phía trên*


```{.python .input  n=14}
mem = get_mem()
with d2l.Benchmark('time per epoch'):
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
        l.backward()
        trainer.step(X.shape[0])
        l.wait_to_read()  # Barrier before a new batch
    npx.waitall()
print(f'increased memory: {get_mem() - mem:f} MB')
```


<!--
As we see, the timing of the minibatches lines up quite nicely with the overall runtime of the optimization code.
Moreover, memory footprint only increases slightly.
Now let us see what happens if we drop the barrier at the end of each minibatch.
-->

*dịch đoạn phía trên*


```{.python .input  n=14}
mem = get_mem()
with d2l.Benchmark('time per epoch'):
    for X, y in data_iter():
        with autograd.record():
            l = loss(y, net(X))
        l.backward()
        trainer.step(X.shape[0])
    npx.waitall()
print(f'increased memory: {get_mem() - mem:f} MB')
```


<!--
Even though the time to issue instructions for the backend is an order of magnitude smaller, we still need to perform computation.
Consequently a large amount of intermediate results cannot be released and may pile up in memory.
While this didn't cause any issues in the toy example above, it might well have resulted in out of memory situations when left unchecked in real world scenarios.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->


## Tóm tắt

<!--
* MXNet decouples the Python frontend from an execution backend. This allows for fast asynchronous insertion of commands into the backend and associated parallelism.
* Asynchrony leads to a rather responsive frontend. However, use caution not to overfill the task queue since it may lead to excessive memory consumption.
* It is recommended to synchronize for each minibatch to keep frontend and backend approximately synchronized.
* Be aware of the fact that conversions from MXNet's memory management to Python will force the backend to wait until  the specific variable is ready. 
`print`, `asnumpy` and `item` all have this effect. This can be desirable but a carless use of synchronization can ruin performance.
* Chip vendors offer sophisticated performance analysis tools to obtain a much more fine-grained insight into the efficiency of deep learning.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. We mentioned above that using asynchronous computation can reduce the total amount of time needed to perform $1000$ computations to $t_1 + 1000 t_2 + t_3$. Why do we have to assume $1000 t_2 > 999 t_1$ here?
2. How would you need to modify the training loop if you wanted to have an overlap of one minibatch each? I.e., if you wanted to ensure that batch $b_t$ finishes before batch $b_{t+2}$ commences?
3. What might happen if we want to execute code on CPUs and GPUs simultaneously? Should you still insist on synchronizing after every minibatch has been issued?
4. Measure the difference between `waitall` and `wait_to_read`. Hint: perform a number of instructions and synchronize for an intermediate result.
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2381)
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
* Đỗ Trường Giang
* Đoàn Võ Duy Thanh

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* 

<!-- Phần 6 -->
* 

<!-- Phần 7 -->
* 

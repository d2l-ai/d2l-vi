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

Máy tính ngày nay là các hệ thống có tính song song cao, được cấu thành từ nhiều lõi CPU (mỗi lõi thường có nhiều luồng),
nhiều phần tử xử lý trong mỗi GPU và thường có nhiều GPU trong mỗi máy.
Nói ngắn gọn, ta có thể xử lý nhiều tác vụ cùng một lúc, thường là trên nhiều thiết bị khác nhau.
Tiếc thay, Python không phải là một ngôn ngữ phù hợp để viết mã tính toán song song và bất đồng bộ, nhất là khi không có sự trợ giúp từ bên ngoài.
Xét cho cùng, Python là ngôn ngữ đơn luồng, và có lẽ trong tương lai sẽ không có gì thay đổi.
Các framework học sâu như MXNet và TensorFlow tận dụng mô hình lập trình bất đồng bộ để cải thiện hiệu năng (PyTorch sử dụng bộ định thời của chính Python nên có tiêu chí đánh đổi hiệu năng khác).
Do đó, việc hiểu rõ cách lập trình bất đồng bộ giúp ta phát triển các chương trình hiệu quả hơn bằng cách chủ động giảm thiểu yêu cầu tính toán và các quan hệ phụ thuộc tương hỗ.
Việc này cho phép ta giảm chi phí tính toán phụ trợ và tăng khả năng tận dụng vi xử lý.
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
Hãy thực hiện việc này bằng cả NumPy và NumPy của MXNet để xem xét sự khác nhau.


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

NumPy của MXNet nhanh hơn tới cả hàng trăm hàng ngàn lần.
Ít nhất là có vẻ là như vậy.
Do cả hai thư viện đều được thực hiện trên cùng một bộ xử lý, chắc hẳn phải có gì đó ảnh hướng đến kết quả.
Nếu ta ép MXNet phải hoàn thành tất cả phép tính trước khi trả về kết quả, ta có thể thấy rõ điều gì đã xảy ra ở trên: phần tính toán được thực hiện bởi back-end trong khi front-end đã trả lại quyền điều khiển cho Python.


```{.python .input  n=3}
with d2l.Benchmark():
    for _ in range(10):
        a = np.random.normal(size=(1000, 1000))
        b = np.dot(a, a)
    npx.waitall()
```


<!--
Broadly speaking, MXNet has a frontend for direct interaction with the users, e.g., via Python, as well as a backend used by the system to perform the computation.
As shown in :numref:`fig_frontends`, users can write MXNet programs in various frontend languages, such as Python, R, Scala, and C++.
Regardless of the frontend programming language used, the execution of MXNet programs occurs primarily in the backend of C++ implementations.
Operations issued by the frontend language are passed on to the backend for execution.
The backend manages its own threads that continuously collect and execute queued tasks.
Note that for this to work the backend must be able to keep track of the dependencies between various steps in the computational graph.
That is, it is not possible to parallelize operations that depend on each other.
-->

Nhìn chung, MXNet có front-end cho phép tương tác trực tiếp với người dùng thông qua Python, cũng như back-end được sử dụng bởi hệ thống nhằm thực hiện nhiệm vụ tính toán.
Như ở :numref:`fig_frontends`, người dùng có thể viết chương trình MXNet bằng nhiều ngôn ngữ front-end như Python, R, Scala và C++.
Dù sử dụng ngôn ngữ front-end nào, chương trình MXNet chủ yếu thực thi trên back-end lập trình bằng C++.
Các thao tác đưa ra bởi ngôn ngữ front-end được truyền vào back-end để thực thi.
Back-end tự quản lý các luồng xử lý bằng việc liên tục tập hợp và thực thi các tác vụ trong hàng đợi.
Chú ý rằng, back-end cần phải có khả năng theo dõi quan hệ phụ thuộc giữa các bước trong đồ thị tính toán để có thể hoạt động.
Nghĩa là ta không thể song song hóa các thao tác phụ thuộc lẫn nhau.

<!--
![Programming Frontends.](../img/frontends.png)
-->

![Lập trình Front-end.](../img/frontends.png)
:width:`500px`
:label:`fig_frontends`


<!--
Let us look at another toy example to understand the dependency graph a bit better.
-->

Hãy xét một ví dụ đơn giản để có thể hiểu rõ hơn đồ thị quan hệ phụ thuộc (*dependency graph*).


```{.python .input  n=4}
x = np.ones((1, 2))
y = np.ones((1, 2))
z = x * y + 2
z
```

<!--
![Dependencies.](../img/asyncgraph.svg)
-->

![Quan hệ phụ thuộc.](../img/asyncgraph.svg)
:label:`fig_asyncgraph`


<!--
The code snippet above is also illustrated in :numref:`fig_asyncgraph`.
Whenever the Python frontend thread executes one of the first three statements, it simply returns the task to the backend queue.
When the last statement’s results need to be printed, the Python frontend thread will wait for the C++ backend thread to finish computing result of the variable `z`.
One benefit of this design is that the Python frontend thread does not need to perform actual computations.
Thus, there is little impact on the program’s overall performance, regardless of Python’s performance.
:numref:`fig_threading` illustrates how frontend and backend interact.
-->

Đoạn mã trên cũng được mô tả trong :numref:`fig_asyncgraph`.
Mỗi khi luồng front-end của Python thực thi một trong ba câu lệnh đầu tiên, nó sẽ chỉ đưa tác vụ đó vào hàng chờ của back-end.
Khi kết quả của câu lệnh cuối cùng cần được in ra, luồng front-end của Python sẽ chờ luồng xử lý back-end C++ tính toán xong kết quả của biến `z`.
Lợi ích của thiết kế này nằm ở việc luồng front-end Python không cần phải đích thân thực hiện việc tính toán.
Do đó, hiệu năng tổng thể của chương trình cũng ít bị ảnh hưởng bởi hiệu năng của Python.
:numref:`fig_threading` mô tả cách front-end và back-end tương tác với nhau.

<!--
![Frontend and Backend.](../img/threading.svg)
-->

![Front-end và Back-end](../img/threading.svg)
:label:`fig_threading`


<!--
## Barriers and Blockers
-->

## Lớp cản và Bộ chặn

<!--
There are a number of operations that will force Python to wait for completion:
* Most obviously `npx.waitall()` waits until all computation has completed, regardless of when the compute instructions were issued.
In practice it is a bad idea to use this operator unless absolutely necessary since it can lead to poor performance.
* If we just want to wait until a specific variable is available we can call `z.wait_to_read()`.
In this case MXNet blocks return to Python until the variable `z` has been computed. Other computation may well continue afterwards.
-->


Có khá nhiều thao tác buộc Python phải chờ cho đến khi nó hoàn thành:

* Hiển nhiên nhất là lệnh `npx.waitall()` chờ đến khi toàn bộ phép toán đã hoàn
  thành, bất chấp thời điểm câu lệnh tính toán được đưa ra. Trong thực tế, trừ
  khi thực sự cần thiết, việc sử dụng thao tác này là một ý tưởng tồi do nó có
  thể làm giảm hiệu năng.
 
* Nếu ta chỉ muốn chờ đến khi một biến cụ thể nào đó sẵn sàng, ta có thể gọi
  `z.wait_to_read()`. Trong trường hợp này MXNet chặn việc trả luồng điều khiển
  về Python cho đến khi biến `z` đã được tính xong. Các thao tác khác sau đó
  mới có thể tiếp tục.


<!--
Let us see how this works in practice:
-->

Hãy xem cách các lệnh chờ trên hoạt động trong thực tế:


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

Cả hai thao tác hoàn thành với thời gian xấp xỉ nhau.
Ngoài các thao tác chặn (*blocking operation*) tường minh, bạn đọc cũng nên biết về việc chặn *ngầm*.
Rõ ràng việc in một biến ra yêu cầu biến đó phải sẵn sàng và do đó nó là một bộ chặn.
Cuối cùng, ép kiểu sang NumPy bằng `z.asnumpy()` và ép kiểu sang số vô hướng bằng `z.item()` cũng là bộ chặn, do trong NumPy không có khái niệm bất đồng bộ.
Có thể thấy việc ép kiểu cũng cần truy cập giá trị, giống như hàm `print`. 
Việc thường xuyên sao chép một lượng nhỏ dữ liệu từ phạm vi của MXNet sang NumPy và ngược lại có thể làm giảm đáng kể hiệu năng của một đoạn mã đáng lẽ sẽ có hiệu năng tốt,
do mỗi thao tác như vậy buộc đồ thị tính toán phải tính toàn bộ các giá trị trung gian để suy ra các số hạng cần thiết *trước khi* thực hiện bất cứ thao tác nào khác.


```{.python .input  n=7}
with d2l.Benchmark('numpy conversion'):
    b = np.dot(a, a)
    b.asnumpy()

with d2l.Benchmark('scalar conversion'):
    b = np.dot(a, a)
    b.sum().item()
```

<!--
## Improving Computation
-->

## Cải thiện Năng lực Tính toán

<!--
On a heavily multithreaded system (even regular laptops have 4 threads or more and on multi-socket servers this number can exceed 256) the overhead of scheduling operations can become significant.
This is why it is highly desirable to have computation and scheduling occur asynchronously and in parallel.
To illustrate the benefit of doing this let us see what happens if we increment a variable by 1 multiple times, both in sequence or asynchronously.
We simulate synchronous execution by inserting a `wait_to_read()` barrier in between each addition.
-->

Trong một hệ thống đa luồng lớn (ngay cả laptop phổ thông cũng có 4 luồng hoặc hơn, và trên các máy trạm đa socket, số luồng có thể vượt quá 256), chi phí phụ trợ từ việc định thời các thao tác có thể trở nên khá lớn.
Đó là lý do tại sao hai quá trình tính toán và định thời nên xảy ra song song và bất đồng bộ.
Để minh hoạ cho lợi ích của việc này, hãy so sánh khi liên tục cộng 1 vào một biến theo cách đồng bộ và bất đồng bộ.
Ta mô phỏng quá trình thực thi đồng bộ bằng cách chèn một lớp cản `wait_to_read()` giữa mỗi phép cộng.



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
A slightly simplified interaction between the Python frontend thread and the C++ backend thread can be summarized as follows:
-->

Ta có thể tóm tắt đơn giản sự tương tác giữa luồng front-end Python và luồng back-end C++ như sau: 

<!--
1. The frontend orders the backend to insert the calculation task `y = x + 1` into the queue.
2. The backend then receives the computation tasks from the queue and performs the actual computations.
3. The backend then returns the computation results to the frontend.
-->

1. Front-end ra lệnh cho back-end đưa tác vụ tính `y = x + 1` vào hàng đợi.
2. Back-end sau đó nhận các tác vụ tính toán từ hàng đợi và thực hiện các phép tính.
3. Back-end trả kết quả tính toán về cho front-end.

<!--
Assume that the durations of these three stages are $t_1, t_2$ and $t_3$, respectively.
If we do not use asynchronous programming, the total time taken to perform 1000 computations is approximately $1000 (t_1+ t_2 + t_3)$.
If asynchronous programming is used, the total time taken to perform 1000 computations can be reduced to $t_1 + 1000 t_2 + t_3$ (assuming $1000 t_2 > 999t_1$), 
since the front-end does not have to wait for the back-end to return computation results for each loop.
-->

Giả sử thời gian thực hiện mỗi giai đoạn trên lần lượt là $t_1, t_2$ và $t_3$.
Nếu ta không áp dụng lập trình bất đồng bộ, tổng thời gian để thực hiện 1000 phép tính xấp xỉ bằng $1000 (t_1+ t_2 + t_3)$.
Còn nếu ta áp dụng lập trình bất đồng bộ, tổng thời gian để thực hiện 1000 phép tính có thể giảm xuống còn $t_1 + 1000 t_2 + t_3$ (giả sử $1000 t_2 > 999t_1$),
do front-end không cần phải chờ back-end trả về kết quả tính toán sau mỗi vòng lặp.


<!--
## Improving Memory Footprint
-->

## Cải thiện Mức chiếm dụng Bộ nhớ

<!--
Imagine a situation where we keep on inserting operations into the backend by executing Python code on the frontend.
For instance, the frontend might insert a large number of minibatch tasks within a very short time.
After all, if no meaningful computation happens in Python this can be done quite quickly.
If each of these tasks can be launched quickly at the same time this may cause a spike in memory usage.
Given a finite amount of memory available on GPUs (and even on CPUs) this can lead to resource contention or even program crashes.
Some readers might have noticed that previous training routines made use of synchronization methods such as `item` or even `asnumpy`.
-->

Cùng hình dung với trường hợp ta liên tục thêm các tính toán vào back-end bằng cách thực thi mã Python trên front-end.
Ví dụ, trong một khoảng thời gian rất ngắn, front-end liên tục thêm vào một lượng lớn các tác vụ trên minibatch.
Xét cho cùng, công việc trên có thể hoàn thành nhanh chóng nếu không có phép tính nào thật sự diễn ra trên Python.
Nếu tất cả tác vụ trên cùng được khởi động một cách nhanh chóng thì có thể dẫn đến dung lượng bộ nhớ sử dụng tăng đột ngột.
Do dung lượng bộ nhớ có sẵn trên GPU (và ngay cả CPU) là có hạn, điều này có thể gây ra sự tranh chấp tài nguyên hoặc thậm chí làm sập chương trình.
Độc giả có lẽ đã nhận ra rằng ở các quy trình huấn luyện trước, ta áp dụng các thao tác đồng bộ như `item` hay ngay cả `asnumpy`.

<!--
We recommend to use these operations carefully, e.g., for each minibatch, such as to balance computational efficiency and memory footprint.
To illustrate what happens let us implement a simple training loop for a deep network and measure its memory consumption and timing.
Below is the mock data generator and deep network.
-->

Chúng tôi khuyến nghị nên sử dụng các thao tác này một cách cẩn thận, ví dụ như với từng minibatch, ta cần đảm bảo sao cho hiệu năng tính toán và mức chiếm dụng bộ nhớ (*memory footprint*) được cân bằng.
Để minh họa, hãy cùng lập trình một vòng lặp huấn luyện đơn giản, đo lượng bộ nhớ tiêu hao và thời gian thực thi,
sử dụng hàm sinh dữ liệu và mạng học sâu dưới đây.


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

Tiếp theo, ta cần công cụ để đo lường mức chiếm dụng bộ nhớ của đoạn mã trên. Để có thể xây dựng công cụ này, ta sử dụng lệnh `ps` của hệ điều hành (chỉ hoạt động trên Linux và macOS).
Để phân tích chi tiết hoạt động của đoạn mã trên, bạn có thể sử dụng [Nsight](https://developer.nvidia.com/nsight-compute-2019_5) của Nvidia hoặc [vTune](https://software.intel.com/en-us/vtune) của Intel.


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

Trước khi bắt đầu kiểm tra, ta cần khởi tạo các tham số của mạng và xử lý một batch.
Nếu không, việc kiểm tra dung lượng bộ nhớ sử dụng thêm sẽ là khá rắc rối.
Bạn đọc có thể tham khảo :numref:`sec_deferred_init` để hiểu rõ chi tiết việc khởi tạo.


```{.python .input  n=13}
for X, y in data_iter():
    break
loss(y, net(X)).wait_to_read()
```


<!--
To ensure that we do not overflow the task buffer on the backend we insert a `wait_to_read` call for the loss function at the end of each loop.
This forces the forward pass to complete before a new forward pass is commenced.
Note that a (possibly more elegant) alternative would have been to track the loss in a scalar variable and to force a barrier via the `item` call.
-->

Để đảm bảo bộ đệm tác vụ tại back-end không bị tràn, ta chèn phương thức `wait_to_read` vào back-end cho hàm mất mát ở cuối mỗi vòng lặp.
Điều này buộc mỗi lượt truyền xuôi phải hoàn thành trước khi lượt truyền xuôi tiếp theo được bắt đầu.
Chú ý rằng có một phương án thay thế khác (có lẽ tinh tế hơn) là theo dõi lượng mất mát ở biến vô hướng và buộc đi qua một lớp chặn (*barrier*) qua việc gọi phương thức `item`.


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

Như ta có thể thấy, thời gian thực hiện từng minibatch khá khớp so với tổng thời gian chạy của đoạn mã tối ưu.
Hơn nữa, lượng bộ nhớ sử dụng tăng không đáng kể.
Giờ hãy cùng xem chuyện gì sẽ xảy ra nếu ta bỏ lớp chặn ở cuối mỗi minibatch.


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

Mặc dù thời gian để đưa ra chỉ dẫn cho back-end nhỏ hơn đến hàng chục lần, ta vẫn cần thực hiện các bước tính toán.
Hậu quả là một lượng lớn các kết quả trung gian không được đưa ra sử dụng và có thể chất đống trong bộ nhớ.
Dù rằng việc này không gây ra bất cứ vấn đề nào trong ví dụ nhỏ trên, nó có thể dẫn đến tình trạng cạn kiệt bộ nhớ nếu không được kiểm tra trong viễn cảnh thực tế.



## Tóm tắt

<!--
* MXNet decouples the Python frontend from an execution backend. This allows for fast asynchronous insertion of commands into the backend and associated parallelism.
* Asynchrony leads to a rather responsive frontend. However, use caution not to overfill the task queue since it may lead to excessive memory consumption.
* It is recommended to synchronize for each minibatch to keep frontend and backend approximately synchronized.
* Be aware of the fact that conversions from MXNet's memory management to Python will force the backend to wait until  the specific variable is ready. 
`print`, `asnumpy` and `item` all have this effect. This can be desirable but a carless use of synchronization can ruin performance.
* Chip vendors offer sophisticated performance analysis tools to obtain a much more fine-grained insight into the efficiency of deep learning.
-->


* MXNet tách riêng khối front-end Python khỏi khối back-end thực thi. Điều này cho phép nhanh chóng chèn các câu lệnh một cách bất đồng bộ vào khối back-end và kết hợp tính toán song song.
* Sự bất đồng bộ giúp front-end phản ứng nhanh hơn. Tuy nhiên, cần phải áp dụng cẩn thận để không làm tràn các tác vụ ở trạng thái đợi, gây chiếm dụng bộ nhớ.
* Nên đồng bộ theo từng minibatch một để giữ cho front-end và back-end được đồng bộ tương đối.
* Nên nhớ rằng việc chuyển quản lý bộ nhớ từ MXNet sang Python sẽ buộc back-end phải chờ cho đến khi biến đó sẵn sàng.
`print`, `asnumpy` và `item` đều gây ra hiệu ứng trên. Điều này có thể có ích đôi lúc, tuy nhiên lạm dụng chúng có thể làm sụt giảm hiệu năng.
* Nhà sản xuất vi xử lý cung cấp các công cụ phân tích hiệu năng tinh vi, cho phép đánh giá hiệu năng của học sâu một cách chi tiết hơn rất nhiều.


## Bài tập

<!--
1. We mentioned above that using asynchronous computation can reduce the total amount of time needed to perform $1000$ computations to $t_1 + 1000 t_2 + t_3$. Why do we have to assume $1000 t_2 > 999 t_1$ here?
2. How would you need to modify the training loop if you wanted to have an overlap of one minibatch each? I.e., if you wanted to ensure that batch $b_t$ finishes before batch $b_{t+2}$ commences?
3. What might happen if we want to execute code on CPUs and GPUs simultaneously? Should you still insist on synchronizing after every minibatch has been issued?
4. Measure the difference between `waitall` and `wait_to_read`. Hint: perform a number of instructions and synchronize for an intermediate result.
-->

1. Như đã đề cập ở trên, sử dụng tính toán bất đồng bộ có thể giảm tổng thời gian cần thiết để thực hiện $1000$ phép tính xuống $t_1 + 1000 t_2 + t_3$. Tại sao ở đó ta lại phải giả sử $1000 t_2 > 999 t_1$?
2. Bạn có thể chỉnh sửa vòng lặp huấn luyện như thế nào nếu muốn xử lý 2 batch cùng lúc (đảm bảo batch $b_t$ hoàn thành trước khi batch $b_{t+2}$ bắt đầu)? 
3. Chuyện gì sẽ xảy ra nếu thực thi mã nguồn đồng thời trên cả CPU và GPU? Liệu có nên tiếp tục đồng bộ sau khi xử lý mỗi minibatch?
4. So sánh sự khác nhau giữa `waitall` và `wait_to_read`. Gợi ý: thực hiện một số lệnh và đồng bộ theo kết quả trung gian.



## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/361)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Đỗ Trường Giang
* Nguyễn Văn Cường
* Phạm Minh Đức
* Nguyễn Lê Quang Nhật
* Phạm Hồng Vinh
* Lê Khắc Hồng Phúc

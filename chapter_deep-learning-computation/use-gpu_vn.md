<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# GPUs
-->

# *dịch tiêu đề phía trên*
:label:`sec_use_gpu`

<!--
In the introduction to this book we discussed the rapid growth of computation over the past two decades. 
In a nutshell, GPU performance has increased by a factor of 1000 every decade since 2000. 
This offers great opportunity but it also suggests a significant need to provide such performance.
-->

*dịch đoạn phía trên*

<!--
|Decade|Dataset|Memory|Floating Point Calculations per Second|
|:--|:-|:-|:-|
|1970|100 (Iris)|1 KB|100 KF (Intel 8080)|
|1980|1 K (House prices in Boston)|100 KB|1 MF (Intel 80186)|
|1990|10 K (optical character recognition)|10 MB|10 MF (Intel 80486)|
|2000|10 M (web pages)|100 MB|1 GF (Intel Core)|
|2010|10 G (advertising)|1 GB|1 TF (NVIDIA C2050)|
|2020|1 T (social network)|100 GB|1 PF (NVIDIA DGX-2)|
-->

*dịch đoạn phía trên*

<!--
In this section we begin to discuss how to harness this compute performance for your research. 
First by using single GPUs and at a later point, how to use multiple GPUs and multiple servers (with multiple GPUs). 
You might have noticed that MXNet `ndarray` looks almost identical to NumPy. But there are a few crucial differences. 
One of the key features that differentiates MXNet from NumPy is its support for diverse hardware devices.
-->

*dịch đoạn phía trên*

<!--
In MXNet, every array has a context. 
In fact, whenever we displayed an `ndarray` so far, it added a cryptic `@cpu(0)` notice to the output which remained unexplained so far. 
As we will discover, this just indicates that the computation is being executed on the CPU. 
Other contexts might be various GPUs. 
Things can get even hairier when we deploy jobs across multiple servers. 
By assigning arrays to contexts intelligently, we can minimize the time spent transferring data between devices. 
For example, when training neural networks on a server with a GPU, we typically prefer for the model’s parameters to live on the GPU.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
In short, for complex neural networks and large-scale data, using only CPUs for computation may be inefficient. 
In this section, we will discuss how to use a single NVIDIA GPU for calculations. 
First, make sure you have at least one NVIDIA GPU installed. 
Then, [download CUDA](https://developer.nvidia.com/cuda-downloads) and follow the prompts to set the appropriate path. 
Once these preparations are complete, the `nvidia-smi` command can be used to view the graphics card information.
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
!nvidia-smi
```

<!--
Next, we need to confirm that the GPU version of MXNet is installed. 
If a CPU version of MXNet is already installed, we need to uninstall it first. 
For example, use the `pip uninstall mxnet` command, then install the corresponding MXNet version according to the CUDA version. 
Assuming you have CUDA 9.0 installed, you can install the MXNet version that supports CUDA 9.0 by `pip install mxnet-cu90`. 
To run the programs in this section, you need at least two GPUs.
-->

*dịch đoạn phía trên*

<!--
Note that this might be extravagant for most desktop computers but it is easily available in the cloud, 
e.g., by using the AWS EC2 multi-GPU instances. Almost all other sections do *not* require multiple GPUs. 
Instead, this is simply to illustrate how data flows between different devices.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Computing Devices
-->

## *dịch tiêu đề phía trên*

<!--
MXNet can specify devices, such as CPUs and GPUs, for storage and calculation. 
By default, MXNet creates data in the main memory and then uses the CPU to calculate it. 
In MXNet, the CPU and GPU can be indicated by `cpu()` and `gpu()`. 
It should be noted that `cpu()` (or any integer in the parentheses) means all physical CPUs and memory. 
This means that MXNet's calculations will try to use all CPU cores. However, `gpu()` only represents one graphic card and the corresponding graphic memory. 
If there are multiple GPUs, we use `gpu(i)` to represent the $i^\mathrm{th}$ GPU ($i$ starts from 0). 
Also, `gpu(0)` and `gpu()` are equivalent.
-->

*dịch đoạn phía trên*

```{.python .input}
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

npx.cpu(), npx.gpu(), npx.gpu(1)
```

<!--
We can query the number of available GPUs through `num_gpus()`.
-->

*dịch đoạn phía trên*

```{.python .input}
npx.num_gpus()
```

<!--
Now we define two convenient functions that allows us to run codes even if the requested GPUs do not exist.
-->

*dịch đoạn phía trên*


```{.python .input}
# Saved in the d2l package for later use
def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu()."""
    return npx.gpu(i) if npx.num_gpus() >= i + 1 else npx.cpu()

# Saved in the d2l package for later use
def try_all_gpus():
    """Return all available GPUs, or [cpu(),] if no GPU exists."""
    ctxes = [npx.gpu(i) for i in range(npx.num_gpus())]
    return ctxes if ctxes else [npx.cpu()]

try_gpu(), try_gpu(3), try_all_gpus()
```

<!--
## `ndarray` and GPUs
-->

## *dịch tiêu đề phía trên*

<!--
By default, `ndarray` objects are created on the CPU. 
Therefore, we will see the `@cpu(0)` identifier each time we print an `ndarray`.
-->

*dịch đoạn phía trên*

```{.python .input  n=4}
x = np.array([1, 2, 3])
x
```

<!--
We can use the `ctx` property of `ndarray` to view the device where the `ndarray` is located. 
It is important to note that whenever we want to operate on multiple terms they need to be in the same context. 
For instance, if we sum two variables, we need to make sure that both arguments are on the same device---otherwise MXNet 
would not know where to store the result or even how to decide where to perform the computation.
-->

*dịch đoạn phía trên*

```{.python .input}
x.ctx
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### Storage on the GPU
-->

### Lưu trữ trên GPU

<!--
There are several ways to store an `ndarray` on the GPU. 
For example, we can specify a storage device with the `ctx` parameter when creating an `ndarray`. 
Next, we create the `ndarray` variable `a` on `gpu(0)`. 
Notice that when printing `a`, the device information becomes `@gpu(0)`. 
The `ndarray` created on a GPU only consumes the memory of this GPU. 
We can use the `nvidia-smi` command to view GPU memory usage. 
In general, we need to make sure we do not create data that exceeds the GPU memory limit.
-->

Có một số cách để lưu trữ một `ndarray` trên GPU.
Ví dụ: ta có thể chỉ định một thiết bị lưu trữ với tham số `ctx` khi tạo một` ndarray`.
Tiếp theo, ta tạo biến cho `ndarray` là `a` trên `gpu(0)`.
Lưu ý rằng khi in `a`, thông tin thiết bị sẽ trở thành `@gpu(0)`.
`ndarray` được tạo trên GPU nào chỉ chiếm bộ nhớ của GPU đó.
Ta có thể sử dụng lệnh `nvidia-smi` để xem việc sử dụng bộ nhớ GPU.
Nói chung, ta cần đảm bảo rằng ta không tạo dữ liệu vượt quá giới hạn bộ nhớ GPU.

```{.python .input  n=5}
x = np.ones((2, 3), ctx=try_gpu())
x
```

<!--
Assuming you have at least two GPUs, the following code will create a random array on `gpu(1)`.
-->

Giả sử bạn có ít nhất hai GPU, đoạn mã sau sẽ tạo ra một mảng ngẫu nhiên trên `gpu(1)`.

```{.python .input}
y = np.random.uniform(size=(2, 3), ctx=try_gpu(1))
y
```

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Copying
-->

### Sao chép

<!--
If we want to compute $\mathbf{x} + \mathbf{y}$ we need to decide where to perform this operation. 
For instance, as shown in :numref:`fig_copyto`, we can transfer $\mathbf{x}$ to `gpu(1)` and perform the operation there. 
*Do not* simply add `x + y` since this will result in an exception. 
The runtime engine would not know what to do, it cannot find data on the same device and it fails.
-->

Nếu ta muốn tính $\mathbf{x} + \mathbf{y}$ thì ta cần quyết định nơi thực hiện phép tính này.
Chẳng hạn, như trong :numref:`fig_copyto`, ta có thể chuyển $\mathbf{x}$ sang `gpu(1)` và thực hiện phép tính ở đó.
*Đừng* chỉ thêm `x + y` vì điều này sẽ dẫn đến một ngoại lệ.
Hệ thống thời gian chạy sẽ không biết phải làm gì và gặp lỗi bởi nó không thể tìm thấy dữ liệu trên cùng một thiết bị.

<!--
![Copyto copies arrays to the target device](../img/copyto.svg)
-->

![Lệnh copyto sao chép các mảng đến thiết bị mục tiêu](../img/copyto.svg)
:label:`fig_copyto`

<!--
`copyto` copies the data to another device such that we can add them. Since $\mathbf{y}$ lives on the second GPU we need to move $\mathbf{x}$ there before we can add the two.
-->

Lệnh `copyto` sao chép dữ liệu sang một thiết bị khác để ta có thể cộng chúng. Vì $\mathbf{y}$ tồn tại trên GPU thứ hai, ta cần di chuyển $\mathbf{x}$ trước khi ta có thể cộng chúng lại.


```{.python .input  n=7}
z = x.copyto(try_gpu(1))
print(x)
print(z)
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
Now that the data is on the same GPU (both $\mathbf{z}$ and $\mathbf{y}$ are), we can add them up. 
In such cases MXNet places the result on the same device as its constituents. In our case that is `@gpu(1)`.
-->

*dịch đoạn phía trên*

```{.python .input}
y + z
```

<!--
Imagine that your variable z already lives on your second GPU (gpu(1)). 
What happens if we call z.copyto(gpu(1))? It will make a copy and allocate new memory, even though that variable already lives on the desired device!
There are times where depending on the environment our code is running in, two variables may already live on the same device. 
So we only want to make a copy if the variables currently lives on different contexts. 
In these cases, we can call `as_in_ctx()`. If the variable is already the specified context then this is a no-op. 
In fact, unless you specifically want to make a copy, `as_in_ctx()` is the method of choice.
-->

*dịch đoạn phía trên*

```{.python .input}
z = x.as_in_ctx(try_gpu(1))
z
```

<!--
It is important to note that, if the `ctx` of the source variable and the target variable are consistent, 
then the `as_in_ctx` function causes the target variable and the source variable to share the memory of the source variable.
-->

*dịch đoạn phía trên*

```{.python .input  n=8}
y.as_in_ctx(try_gpu(1)) is y
```

<!--
The `copyto` function always creates new memory for the target variable.
-->

*dịch đoạn phía trên*

```{.python .input}
y.copyto(try_gpu(1)) is y
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
### Side Notes
-->

### *dịch tiêu đề phía trên*

<!--
People use GPUs to do machine learning because they expect them to be fast. 
But transferring variables between contexts is slow. 
So we want you to be 100% certain that you want to do something slow before we let you do it. 
If MXNet just did the copy automatically without crashing then you might not realize that you had written some slow code.
-->

*dịch đoạn phía trên*

<!--
Also, transferring data between devices (CPU, GPUs, other machines) is something that is *much slower* than computation. 
It also makes parallelization a lot more difficult, since we have to wait for data to be sent (or rather to be received) before we can proceed with more operations. 
This is why copy operations should be taken with great care. As a rule of thumb, many small operations are much worse than one big operation. 
Moreover, several operations at a time are much better than many single operations interspersed in the code (unless you know what you are doing). 
This is the case since such operations can block if one device has to wait for the other before it can do something else. 
It is a bit like ordering your coffee in a queue rather than pre-ordering it by phone and finding out that it is ready when you are.
-->

*dịch đoạn phía trên*

<!--
Last, when we print `ndarray`s or convert `ndarray`s to the NumPy format, if the data is not in main memory, 
MXNet will copy it to the main memory first, resulting in additional transmission overhead. 
Even worse, it is now subject to the dreaded Global Interpreter Lock which makes everything wait for Python to complete.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## Gluon and GPUs
-->

## *dịch tiêu đề phía trên*

<!--
Similarly, Gluon's model can specify devices through the `ctx` parameter during initialization. 
The following code initializes the model parameters on the GPU (we will see many more examples of 
how to run models on GPUs in the following, simply since they will become somewhat more compute intensive).
-->

*dịch đoạn phía trên*

```{.python .input  n=12}
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=try_gpu())
```

<!--
When the input is an `ndarray` on the GPU, Gluon will calculate the result on the same GPU.
-->

*dịch đoạn phía trên*

```{.python .input  n=13}
net(x)
```

<!--
Let's confirm that the model parameters are stored on the same GPU.
-->

*dịch đoạn phía trên*

```{.python .input  n=14}
net[0].weight.data()
```

<!--
In short, as long as all data and parameters are on the same device, we can learn models efficiently. 
In the following we will see several such examples.
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## Tóm tắt

<!--
* MXNet can specify devices for storage and calculation, such as CPU or GPU. 
By default, MXNet creates data in the main memory and then uses the CPU to calculate it.
* MXNet requires all input data for calculation to be *on the same device*, be it CPU or the same GPU.
* You can lose significant performance by moving data without care. 
A typical mistake is as follows: computing the loss for every minibatch on the GPU and reporting it back to the user 
on the command line (or logging it in a NumPy array) will trigger a global interpreter lock which stalls all GPUs. 
It is much better to allocate memory for logging inside the GPU and only move larger logs.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. Try a larger computation task, such as the multiplication of large matrices, and see the difference in speed between the CPU and GPU. 
What about a task with a small amount of calculations?
2. How should we read and write model parameters on the GPU?
3. Measure the time it takes to compute 1000 matrix-matrix multiplications of $100 \times 100$ matrices 
and log the matrix norm $\mathrm{tr} M M^\top$ one result at a time vs. keeping a log on the GPU and transferring only the final result.
4. Measure how much time it takes to perform two matrix-matrix multiplications on two GPUs at the same time vs. in sequence on one GPU (hint: you should see almost linear scaling).
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->


<!--
## [Discussions](https://discuss.mxnet.io/t/2330)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2330)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
* Trần Yến Thy
* Lê Khắc Hồng Phúc

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*

<!-- Phần 7 -->
*

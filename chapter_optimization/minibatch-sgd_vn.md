<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Minibatch Stochastic Gradient Descent
-->

# Hạ Gradient Ngẫu nhiên theo Minibatch
:label:`sec_minibatch_sgd`

<!--
So far we encountered two extremes in the approach to gradient based learning: :numref:`sec_gd` uses the full dataset to compute gradients and to update parameters, one pass at a time.
Conversely :numref:`sec_sgd` processes one observation at a time to make progress.
Each of them has its own drawbacks.
Gradient Descent is not particularly *data efficient* whenever data is very similar.
Stochastic Gradient Descent is not particularly *computationally efficient* since CPUs and GPUs cannot exploit the full power of vectorization.
This suggests that there might be a happy medium, and in fact, that's what we have been using so far in the examples we discussed.
-->

Đến phần này, ta đã tiếp xúc với hai thái cực trong các phương pháp học dựa theo gradient: :numref:`sec_gd` sử dụng toàn bộ tập dữ liệu để tính toán gradient và cập nhật từng tham số một.
Ngược lại, :numref:`sec_sgd` xử lý từng điểm dữ liệu một để cập nhật các tham số.
Mỗi cách có mặt hạn chế riêng của nó.
Hạ Gradient có *hiệu suất dữ liệu* (*data efficient*) thấp khi dữ liệu tương đồng với nhau.
Hạ Gradient Ngẫu nhiên có *hiệu suất tính toán* (*computationally efficient*) thấp do CPU và GPU không thể khai thác hết khả năng của vector hoá.
Điều này gợi ý rằng có thể có một phương pháp thích hợp ở giữa, và thực tế, đó chính là phương pháp mà ta sử dụng ở các ví dụ đã thảo luận từ trước tới nay.

<!--
## Vectorization and Caches
-->

## Vector Hoá và Vùng nhớ đệm

<!--
At the heart of the decision to use minibatches is computational efficiency.
This is most easily understood when considering parallelization to multiple GPUs and multiple servers.
In this case we need to send at least one image to each GPU.
With 8 GPUs per server and 16 servers we already arrive at a minibatch size of 128.
-->

Trọng tâm của quyết định sử dụng minibatch là hiệu suất tính toán.
Để dễ hiểu, ta xét trường hợp tính toán song song giữa nhiều GPU và giữa nhiều máy chủ.
Trong trường hợp này ta cần đưa ít nhất một ảnh vào mỗi GPU.
Với 16 máy chủ và 8 GPU mỗi máy, ta đã có minibatch đạt kích thước 128.

<!--
Things are a bit more subtle when it comes to single GPUs or even CPUs.
These devices have multiple types of memory, often multiple type of compute units and different bandwidth constraints between them.
For instance, a CPU has a small number of registers and then L1, L2 and in some cases even L3 cache (which is shared between the different processor cores).
These caches are of increasing size and latency (and at the same time they are of decreasing bandwidth).
Suffice it to say, the processor is capable of performing many more operations than what the main memory interface is able to provide.
-->

Vấn đề trở nên nhạy cảm hơn đối với trường hợp GPU đơn hay ngay cả CPU đơn.
Những thiết bị này có nhiều loại bộ nhớ, thường có nhiều loại đơn vị thực hiện tính toán và giới hạn băng thông giữa các đơn vị này cũng khác nhau.
Ví dụ, một CPU có số lượng nhỏ thanh ghi và bộ nhớ đệm L1, L2 và trong một số trường hợp có cả L3 (phần bộ nhớ được phân phối giữa các lõi khác nhau của vi xử lý).
Các bộ nhớ đệm đang tăng dần về kích thước và độ trễ (và cùng với đó là giảm băng thông).
Nói đến đây là đủ để thấy rằng vi xử lý có khả năng thực hiện nhiều tác vụ hơn so với giao diện bộ nhớ chính có thể cung cấp.

<!--
* A 2GHz CPU with 16 cores and AVX-512 vectorization can process up to $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ bytes per second. 
The capability of GPUs easily exceeds this number by a factor of 100. 
On the other hand, a midrange server processor might not have much more than 100 GB/s bandwidth, 
i.e., less than one tenth of what would be required to keep the processor fed.
To make matters worse, not all memory access is created equal: first, memory interfaces are typically 64 bit wide or wider (e.g., on GPUs up to 384 bit), 
hence reading a single byte incurs the cost of a much wider access.
* There is significant overhead for the first access whereas sequential access is relatively cheap (this is often called a burst read).
There are many more things to keep in mind, such as caching when we have multiple sockets, chiplets and other structures.
A detailed discussion of this is beyond the scope of this section.
See e.g., this [Wikipedia article](https://en.wikipedia.org/wiki/Cache_hierarchy) for a more in-depth discussion.
-->

* Một CPU tốc độ 2GHz với 16 lõi và phép vector hoá AVX-512 có thể xử lý lên lới $2 \cdot 10^9 \cdot 16 \cdot 32 = 10^{12}$ byte mỗi giây.
Khả năng của GPU dễ dàng vượt qua con số này hàng trăm lần.
Mặt khác, với một vi xử lý máy chủ trung bình, băng thông có lẽ không vượt quá 100 GB/s, tức là chưa bằng một phần mười lượng yêu cầu để giữ cho bộ xử lý hoạt động.
Vấn đề còn tồi tệ hơn khi ta xét đến việc không phải khả năng truy cập bộ nhớ nào cũng như nhau: đầu tiên, giao diện bộ nhớ thường rộng 64 bit hoặc hơn (ví dụ như trên GPU lên đến 384 bit), 
do đó việc đọc một byte duy nhất vẫn sẽ phải chịu chi phí giống như truy cập một khoảng bộ nhớ rộng hơn.
* Tổng chi phí cho lần truy cập đầu tiên là khá đáng kể trong khi truy cập liên tiếp thường hao tổn ít (thường được gọi là đọc hàng loạt).
Có rất nhiều điều cần phải chú ý đến, ví dụ như lưu trữ đệm khi ta có nhiều điểm truy cập cuối (*sockets*), nhiều chiplet và các cấu trúc khác.
 Việc thảo luận chi tiết vấn đề trên nằm ngoài phạm vi của phần này.
Bạn có thể đọc [Bài viết Wikipedia](https://en.wikipedia.org/wiki/Cache_hierarchy) này để hiểu sâu hơn về các vấn đề trên.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
The way to alleviate these constraints is to use a hierarchy of CPU caches which are actually fast enough to supply the processor with data.
This is *the* driving force behind batching in deep learning.
To keep matters simple, consider matrix-matrix multiplication, say $\mathbf{A} = \mathbf{B}\mathbf{C}$.
We have a number of options for calculating $\mathbf{A}$.
For instance we could try the following:
-->

Cách để giảm bớt những ràng buộc trên là sử dụng hệ thống cấp bậc (*hierarchy*) của các vùng nhớ đệm trong CPU, các vùng nhớ này đủ nhanh để có thể cung cấp dữ liệu cho vi xử lý.
Đây *chính là* động lực đằng sau việc sử dụng batch trong học sâu.
Để đơn giản hoá vấn đề, xét phép nhân hai ma trận $\mathbf{A} = \mathbf{B}\mathbf{C}$.
Để tính $\mathbf{A}$ ta có khá nhiều lựa chọn.
Ta có thể thử một số cách sau:

<!--
1. We could compute $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$, i.e., we could compute it element-wise by means of dot products.
2. We could compute $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$, i.e., we could compute it one column at a time. 
Likewise we could compute $\mathbf{A}$ one row $\mathbf{A}_{i,:}$ at a time.
3. We could simply compute $\mathbf{A} = \mathbf{B} \mathbf{C}$.
4. We could break $\mathbf{B}$ and $\mathbf{C}$ into smaller block matrices and compute $\mathbf{A}$ one block at a time.
-->

1. Ta có thể tính $\mathbf{A}_{ij} = \mathbf{B}_{i,:} \mathbf{C}_{:,j}^\top$, tức là tính từng phần tử bằng tích vô hướng.
2. Ta có thể tính $\mathbf{A}_{:,j} = \mathbf{B} \mathbf{C}_{:,j}^\top$, tức là tính theo từng cột một.
Tương tự, ta có thể tính $\mathbf{A}$ theo từng hàng $\mathbf{A}_{i,:}$ một.
3. Ta đơn giản có thể tính $\mathbf{A} = \mathbf{B} \mathbf{C}$.
4. Ta có thể chia $\mathbf{B}$ và $\mathbf{C}$ thành nhiều khối ma trận nhỏ hơn và tính $\mathbf{A}$ từng khối một.

<!--
If we follow the first option, we will need to copy one row and one column vector into the CPU each time we want to compute an element $\mathbf{A}_{ij}$.
Even worse, due to the fact that matrix elements are aligned sequentially we are thus required to access many disjoint locations for one of the two vectors as we read them from memory.
The second option is much more favorable.
In it, we are able to keep the column vector $\mathbf{C}_{:,j}$ in the CPU cache while we keep on traversing through $B$.
This halves the memory bandwidth requirement with correspondingly faster access.
Of course, option 3 is most desirable.
Unfortunately, most matrices might not entirely fit into cache (this is what we are discussing after all).
However, option 4 offers a practically useful alternative: we can move blocks of the matrix into cache and multiply them locally.
Optimized libraries take care of this for us.
Let us have a look at how efficient these operations are in practice.
-->

Nếu sử dụng cách đầu tiên, ta cần sao chép một vector cột và một vector hàng vào CPU cho mỗi lần tính phần tử $\mathbf{A}_{ij}$.
Tệ hơn nữa, do lần lượt duyệt qua từng các phần tử của ma trận, ta buộc phải truy cập nhiều lần vùng nhớ của một trong hai vector khi đọc chúng từ bộ nhớ.
Cách thứ hai được ưa chuộng hơn nhiều.
Theo cách này, ta có thể giữ vector cột $\mathbf{C}_{:,j}$ trong vùng nhớ đệm của CPU trong khi ta tiếp tục quét qua $B$.
Cách này chia đôi băng thông cần thiết của bộ nhớ, do đó truy cập nhanh hơn.
Đương nhiên cách thứ ba là tốt nhất.
Đáng tiếc rằng đa số ma trận quá lớn để có thể đưa vào vùng nhớ đệm (dù sao thì đây cũng chính là điều ta đang thảo luận).
Tuy nhiên, cách thứ tư cho ta một phương pháp thay thế thiết thực: đưa các khối của ma trận vào vùng nhớ đệm và thực hiện phép nhân cục bộ.
Các thư viện đã được tối ưu sẽ thực hiện việc này giúp chúng ta.
Hãy xem xét hiệu suất của từng phương pháp trong thực tế.

<!--
Beyond computational efficiency, the overhead introduced by Python and by the deep learning framework itself is considerable.
Recall that each time we execute a command the Python interpreter sends a command to the MXNet engine which needs to insert it into the compute graph and deal with it during scheduling.
Such overhead can be quite detrimental.
In short, it is highly advisable to use vectorization (and matrices) whenever possible.
-->

Ngoài hiệu suất tính toán, tổng chi phí do Python gây ra và do chính framework học sâu cũng đáng để cân nhắc.
Nên nhớ rằng mỗi lần ta thực hiện một câu lệnh, bộ thông dịch Python gửi một câu lệnh đến MXNet engine để chèn câu lệnh đó vào đồ thị tính toán và thực thi nó theo đúng lịnh trình.
Chi phí đó có thể khá bất lợi.
Nói ngắn gọn, nên áp dụng vector hoá (và ma trận) bất cứ khi nào có thể.


```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn
npx.set_np()

timer = d2l.Timer()
A = np.zeros((256, 256))
B = np.random.normal(0, 1, (256, 256))
C = np.random.normal(0, 1, (256, 256))
```


<!--
Element-wise assignment simply iterates over all rows and columns of $\mathbf{B}$ and $\mathbf{C}$ respectively to assign the value to $\mathbf{A}$.
-->

Phép nhân theo từng phần tử một chỉ đơn giản là lặp qua tất cả các hàng và cột của $\mathbf{B}$ và $\mathbf{C}$ theo thứ tự rồi gán kết quả cho $\mathbf{A}$.


```{.python .input  n=2}
# Compute A = B C one element at a time
timer.start()
for i in range(256):
    for j in range(256):
        A[i, j] = np.dot(B[i, :], C[:, j])
A.wait_to_read()
timer.stop()
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
A faster strategy is to perform column-wise assignment.
-->

Có một chiến lược nhanh hơn chính là thực hiện phép nhân theo từng cột.


```{.python .input  n=3}
# Compute A = B C one column at a time
timer.start()
for j in range(256):
    A[:, j] = np.dot(B, C[:, j])
A.wait_to_read()
timer.stop()
```


<!--
Last, the most effective manner is to perform the entire operation in one block.
Let us see what the respective speed of the operations is.
-->

Cuối cùng, cách hiệu quả nhất là thực hiện toàn bộ phép nhân trong một khối.
Hãy thử xem tốc độ tương ứng của phương pháp này là bao nhiêu.


```{.python .input  n=4}
# Compute A = B C in one go
timer.start()
A = np.dot(B, C)
A.wait_to_read()
timer.stop()

# Multiply and add count as separate operations (fused in practice)
gigaflops = [2/i for i in timer.times]
print("Performance in Gigaflops: element {:.3f}, \
      column {:.3f}, full {:.3f}".format(*gigaflops))
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Minibatches
-->

## Minibatch

:label:`sec_minibatches`

<!--
In the past we took it for granted that we would read *minibatches* of data rather than single observations to update parameters.
We now give a brief justification for it.
Processing single observations requires us to perform many single matrix-vector (or even vector-vector) multiplications, 
which is quite expensive and which incurs a significant overhead on behalf of the underlying deep learning framework.
This applies both to evaluating a network when applied to data (often referred to as inference) and when computing gradients to update parameters.
That is, this applies whenever we perform $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ where
-->

Ở các phần trước ta cho rằng việc đọc dữ liệu theo *minibatch* thay vì từng điểm dữ liệu một để cập nhật các tham số là điều hiển nhiên.
Giờ ta sẽ khẳng định lại điều này một cách ngắn gọn.
Xử lý từng điểm dữ liệu một đòi hỏi ta phải thực hiện rất nhiều phép nhân ma trận với vector (hay ngay cả vector với vector).
Điều này khá tốn kém và đồng thời phải chịu cả tổng chi phí khá lớn do framework học sâu bên dưới.
Vấn đề này xảy ra ở cả lúc đánh giá một mạng khi áp dụng dữ liệu vào (thường được gọi là suy luận) và khi tính toán gradient để cập nhật các tham số.
Tức là vấn đề xảy ra mỗi khi ta thực hiện $\mathbf{w} \leftarrow \mathbf{w} - \eta_t \mathbf{g}_t$ trong đó


$$\mathbf{g}_t = \partial_{\mathbf{w}} f(\mathbf{x}_{t}, \mathbf{w})$$


<!--
We can increase the *computational* efficiency of this operation by applying it to a minibatch of observations at a time.
That is, we replace the gradient $\mathbf{g}_t$ over a single observation by one over a small batch
-->

Ta có thể tăng hiệu suất *tính toán* của phép toán này bằng cách áp dụng vào từng minibatch dữ liệu.
Tức là ta thay thế gradient $\mathbf{g}_t$ trên một điểm dữ liệu duy nhất bằng gradient đó trên một batch nhỏ.


$$\mathbf{g}_t = \partial_{\mathbf{w}} \frac{1}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} f(\mathbf{x}_{i}, \mathbf{w})$$


<!--
Let us see what this does to the statistical properties of $\mathbf{g}_t$: since both $\mathbf{x}_t$ and 
also all elements of the minibatch $\mathcal{B}_t$ are drawn uniformly at random from the training set, the expectation of the gradient remains unchanged.
The variance, on the other hand, is reduced significantly.
Since the minibatch gradient is composed of $b := |\mathcal{B}_t|$ independent gradients which are being averaged, its standard deviation is reduced by a factor of $b^{-\frac{1}{2}}$.
This, by itself, is a good thing, since it means that the updates are more reliably aligned with the full gradient.
-->

Hãy thử xem phương pháp trên tác động thế nào đến các tính chất thống kê của $\mathbf{g}_t$: do cả $\mathbf{x}_t$ và tất cả các phần tử trong minibatch $\mathcal{B}_t$ được lấy ra từ tập huấn luyện với xác suất như nhau, kỳ vọng của gradient được giữ nguyên.
Mặt khác, phương sai giảm một cách đáng kể.
Do gradient của minibatch bao gồm các gradient $b := |\mathcal{B}_t|$ độc lập được lấy trung bình, độ lệch chuẩn của nó giảm đi theo hệ số $b^{-\frac{1}{2}}$.
Đây là một điều tốt, cách cập nhật này có độ tin cậy gần bằng việc lấy gradient toàn bộ tập dữ liệu.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
Naively this would indicate that choosing a large minibatch $\mathcal{B}_t$ would be universally desirable.
Alas, after some point, the additional reduction in standard deviation is minimal when compared to the linear increase in computational cost.
In practice we pick a minibatch that is large enough to offer good computational efficiency while still fitting into the memory of a GPU.
To illustrate the savings let us have a look at some code.
In it we perform the same matrix-matrix multiplication, but this time broken up into "minibatches" of 64 columns at a time.
-->

Ý trên có thể hiểu theo một cách ngây thơ rằng chọn minibatch $\mathcal{B}_t$ lớn luôn là tốt nhất.
Tiếc rằng đến một mức nào đó, độ lệch chuẩn sẽ giảm không đáng kể so với chi phí tính toán tăng tuyến tính.
Do đó trong thực tế, ta chọn minibatch đủ lớn để hiệu suất tính toán cao trong khi vẫn vừa đủ để đưa vào bộ nhớ của GPU.
Để minh hoạ cho quá trình lưu trữ, hãy xem đoạn mã nguồn dưới đây.
Trong đó ta vẫn thực hiện phép nhân ma trận với ma trận, tuy nhiên lần này ta tách thành từng minibatch 64 cột một.


```{.python .input}
timer.start()
for j in range(0, 256, 64):
    A[:, j:j+64] = np.dot(B, C[:, j:j+64])
timer.stop()
print("Performance in Gigaflops: block {:.3f}".format(2/timer.times[3]))
```


<!--
As we can see, the computation on the minibatch is essentially as efficient as on the full matrix.
A word of caution is in order.
In :numref:`sec_batch_norm` we used a type of regularization that was heavily dependent on the amount of variance in a minibatch.
As we increase the latter, the variance decreases and with it the benefit of the noise-injection due to batch normalization.
See e.g., :cite:`Ioffe.2017` for details on how to rescale and compute the appropriate terms.
-->

Ở đây ta có thể thấy quá trình tính toán trên minibatch về cơ bản có hiệu suất gần bằng thực hiện trên toàn ma trận.
Lưu ý thứ tự thực hiện.
Trong :numref:`sec_batch_norm` ta sử dụng một loại điều chuẩn phụ thuộc chặt chẽ vào phương sai của minibatch.
Khi ta tăng kích thước minibatch, phương sai giảm xuống và cùng với đó là lợi ích của việc thêm nhiễu (*noise-injection*) cũng giảm theo do chuẩn hóa theo batch.
Đọc :cite:`Ioffe.2017` để xem thêm chi tiết về cách chuyển đổi giá trị và tính các số hạng phù hợp.

<!--
## Reading the Dataset
-->

## Đọc Tập dữ liệu

<!--
Let us have a look at how minibatches are efficiently generated from data.
In the following we use a dataset developed by NASA to test the wing [noise from different aircraft](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) to compare these optimization algorithms.
For convenience we only use the first $1,500$ examples.
The data is whitened for preprocessing, i.e., we remove the mean and rescale the variance to $1$ per coordinate.
-->

Hãy xem cách tạo các minibatch từ dữ liệu một cách hiệu quả.
Trong đoạn mã nguồn dưới ta sử dụng tập dữ liệu được phát triển bởi NASA để kiểm tra [tiếng ồn từ các máy bay khác nhau](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) để so sánh các thuật toán tối ưu này.
Để thuận tiện ta chỉ sử dụng $1,500$ ví dụ đầu tiên.
Tập dữ liệu được tẩy trắng (*whitened*) để xử lý, tức là với mỗi toạ độ ta trừ đi giá trị trung bình và chuyển đổi giá trị phương sai về $1$.


```{.python .input  n=1}
#@save
d2l.DATA_HUB['airfoil'] = (d2l.DATA_URL + 'airfoil_self_noise.dat',
                           '76e5be1548fd8222e5074cf0faae75edff8cf93f')
#@save
def get_data_ch11(batch_size=10, n=1500):
    data = np.genfromtxt(d2l.download('airfoil'),
                         dtype=np.float32, delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data_iter = d2l.load_array(
        (data[:n, :-1], data[:n, -1]), batch_size, is_train=True)
    return data_iter, data.shape[1]-1
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Implementation from Scratch
-->

## Lập trình từ đầu

<!--
Recall the minibatch SGD implementation from :numref:`sec_linear_scratch`.
In the following we provide a slightly more general implementation.
For convenience it has the same call signature as the other optimization algorithms introduced later in this chapter.
Specifically, we add the status input `states` and place the hyperparameter in dictionary `hyperparams`.
In addition, we will average the loss of each minibatch example in the training function, so the gradient in the optimization algorithm does not need to be divided by the batch size.
-->

Hãy nhớ lại cách lập trình SGD theo minibatch từ :numref:`sec_linear_scratch`.
Trong phần tiếp theo, chúng tôi sẽ trình bày cách lập trình tổng quát hơn một chút.
Để thuận tiện, hàm lập trình SGD và các thuật toán tối ưu khác được giới thiệu sau trong chương này sẽ có danh sách tham số giống nhau.
Cụ thể, chúng ta thêm trạng thái đầu vào `states` và đặt siêu tham số trong từ điển `hyperparams`.
Bên cạnh đó, chúng ta sẽ tính giá trị mất mát trung bình của từng minibatch trong hàm huấn luyện, từ đó không cần phải chia gradient cho kích thước batch trong thuật toán tối ưu nữa.


```{.python .input  n=2}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```


<!--
Next, we implement a generic training function to facilitate the use of the other optimization algorithms introduced later in this chapter.
It initializes a linear regression model and can be used to train the model with minibatch SGD and other algorithms introduced subsequently.
-->

Tiếp theo, chúng ta lập trình một hàm huấn luyện tổng quát, sử dụng được cho cả các thuật toán tối ưu khác được giới thiệu sau trong chương này.
Hàm sẽ khởi tạo một mô hình hồi quy tuyến tính và có thể được sử dụng để huấn luyện mô hình với thuật toán hạ gradient ngẫu nhiên theo minibatch và các thuật toán khác.


```{.python .input  n=3}
#@save
def train_ch11(trainer_fn, states, hyperparams, data_iter,
               feature_dim, num_epochs=2):
    # Initialization
    w = np.random.normal(scale=0.01, size=(feature_dim, 1))
    b = np.zeros(1)
    w.attach_grad()
    b.attach_grad()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    # Train
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print('loss: %.3f, %.3f sec/epoch' % (animator.Y[0][-1], timer.avg()))
    return timer.cumsum(), animator.Y[0]
```


<!--
Let us see how optimization proceeds for batch gradient descent.
This can be achieved by setting the minibatch size to 1500 (i.e., to the total number of examples).
As a result the model parameters are updated only once per epoch.
There is little progress.
In fact, after 6 steps progress stalls.
-->

Hãy cùng quan sát quá trình tối ưu hóa của thuật toán hạ gradient theo toàn bộ batch.
Ta có thể sử dụng toàn bộ batch bằng cách thiết lập kích thước minibatch bằng 1500 (chính là tổng số mẫu).
Kết quả là các tham số mô hình chỉ được cập nhật một lần duy nhất trong mỗi epoch.
Có thể thấy không hề có tiến triển đáng kể.
Trong thực tế, việc tối ưu vẫn dậm chân tại chỗ sau 6 epoch.


```{.python .input  n=4}
def train_sgd(lr, batch_size, num_epochs=2):
    data_iter, feature_dim = get_data_ch11(batch_size)
    return train_ch11(
        sgd, None, {'lr': lr}, data_iter, feature_dim, num_epochs)
gd_res = train_sgd(1, 1500, 10)
```


<!--
When the batch size equals 1, we use SGD for optimization.
For simplicity of implementation we picked a constant (albeit small) learning rate.
In SGD, the model parameters are updated whenever an example is processed.
In our case this amounts to 1500 updates per epoch.
As we can see, the decline in the value of the objective function slows down after one epoch.
Although both the procedures processed 1500 examples within one epoch, SGD consumes more time than gradient descent in our experiment.
This is because SGD updated the parameters more frequently and since it is less efficient to process single observations one at a time.
-->

Khi kích thước batch bằng 1, chúng ta sử dụng thuật toán SGD để tối ưu hóa.
Để đơn giản hoá việc lập trình, chúng ta cố định tốc độ học bằng một hằng số (có giá trị nhỏ).
Trong SGD, các tham số mô hình được cập nhật bất cứ khi nào một mẫu huấn luyện được xử lý.
Trong trường hợp này, sẽ có 1500 lần cập nhật trong mỗi epoch.
Như chúng ta có thể thấy, sự suy giảm giá trị của hàm mục tiêu chậm lại sau một epoch.
Mặc dù cả hai thuật toán cùng xử lý 1500 mẫu trong một epoch, thuật toán SGD tốn thời gian hơn thuật toán hạ gradient trong thí nghiệm trên.
Điều này là do SGD cập nhật các tham số thường xuyên hơn và kém hiệu quả trong việc xử lý riêng lẻ từng mẫu quan sát một.


```{.python .input  n=5}
sgd_res = train_sgd(0.005, 1)
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
Last, when the batch size equals 100, we use minibatch SGD for optimization.
The time required per epoch is longer than the time needed for SGD and the time for batch gradient descent.
-->

*dịch đoạn phía trên*


```{.python .input  n=6}
mini1_res = train_sgd(.4, 100)
```


<!--
Reducing the batch size to 10, the time for each epoch increases because the workload for each batch is less efficient to execute.
-->

*dịch đoạn phía trên*


```{.python .input  n=7}
mini2_res = train_sgd(.05, 10)
```


<!--
Finally, we compare the time versus loss for the preview four experiments.
As can be seen, despite SGD converges faster than GD in terms of number of examples processed, 
it uses more time to reach the same loss than GD because that computing gradient example by example is not efficient.
Minibatch SGD is able to trade-off the convergence speed and computation efficiency.
A minibatch size 10 is more efficient than SGD; a minibatch size 100 even outperforms GD in terms of runtime.
-->

*dịch đoạn phía trên*


```{.python .input  n=8}
d2l.set_figsize([6, 3])
d2l.plot(*list(map(list, zip(gd_res, sgd_res, mini1_res, mini2_res))),
         'time (sec)', 'loss', xlim=[1e-2, 10],
         legend=['gd', 'sgd', 'batch size=100', 'batch size=10'])
d2l.plt.gca().set_xscale('log')
```

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Concise Implementation
-->

## *dịch đoạn phía trên*


<!--
In Gluon, we can use the `Trainer` class to call optimization algorithms.
This is used to implement a generic training function.
We will use this throughout the current chapter.
-->

*dịch đoạn phía trên*


```{.python .input  n=9}
#@save
def train_gluon_ch11(tr_name, hyperparams, data_iter, num_epochs=2):
    # Initialization
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), tr_name, hyperparams)
    loss = gluon.loss.L2Loss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, d2l.Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(X.shape[0])
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (d2l.evaluate_loss(net, data_iter, loss),))
                timer.start()
    print('loss: %.3f, %.3f sec/epoch' % (animator.Y[0][-1], timer.avg()))
```


<!--
Using Gluon to repeat the last experiment shows identical behavior.
-->

*dịch đoạn phía trên*


```{.python .input  n=10}
data_iter, _ = get_data_ch11(10)
train_gluon_ch11('sgd', {'learning_rate': 0.05}, data_iter)
```

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->


## Tóm tắt

<!--
* Vectorization makes code more efficient due to reduced overhead arising from the deep learning framework and due to better memory locality and caching on CPUs and GPUs. 
* There is a trade-off between statistical efficiency arising from SGD and computational efficiency arising from processing large batches of data at a time. 
* Minibatch stochastic gradient descent offers the best of both worlds: computational and statistical efficiency. 
* In minibatch SGD we process batches of data obtained by a random permutation of the training data (i.e., each observation is processed only once per epoch, albeit in random order). 
* It is advisable to decay the learning rates during training. 
* In general, minibatch SGD is faster than SGD and gradient descent for convergence to a smaller risk, when measured in terms of clock time.  
-->

*dịch đoạn phía trên*


## Bài tập

<!--
1. Modify the batch size and learning rate and observe the rate of decline for the value of the objective function and the time consumed in each epoch.
2. Read the MXNet documentation and use the `Trainer` class `set_learning_rate` function to reduce the learning rate of the minibatch SGD to 1/10 of its previous value after each epoch.
3. Compare minibatch SGD with a variant that actually *samples with replacement* from the training set. What happens?
4. An evil genie replicates your dataset without telling you (i.e., each observation occurs twice and your dataset grows to twice its original size, but nobody told you). 
How does the behavior of SGD, minibatch SGD and that of gradient descent change?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2373)
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
* Đỗ Trường Giang
* Nguyễn Văn Cường

<!-- Phần 2 -->
* Đỗ Trường Giang
* Nguyễn Văn Cường

<!-- Phần 3 -->
* Đỗ Trường Giang
* Nguyễn Văn Cường

<!-- Phần 4 -->
* Đỗ Trường Giang
* Nguyễn Văn Cường

<!-- Phần 5 -->
* Nguyễn Văn Quang
* Nguyễn Văn Cường
* Nguyễn Minh Đức
<!-- Phần 6 -->
* 

<!-- Phần 7 -->
* 

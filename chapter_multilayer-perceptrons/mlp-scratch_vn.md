<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Implementation of Multilayer Perceptron from Scratch
-->

# Lập trình Perceptron Đa tầng từ đầu
:label:`sec_mlp_scratch`

<!--
Now that we have characterized multilayer perceptrons (MLPs) mathematically, let's try to implement one ourselves.
-->

Chúng ta đã mô tả perceptron đa tầng (MLPs) ở dạng toán học, giờ hãy cùng thử tự lập trình một mạng như vậy xem sao.

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
npx.set_np()
```

<!--
To compare against our previous results achieved with (linear) softmax regression (:numref:`sec_softmax_scratch`), 
we will continue work with the Fashion-MNIST image classification dataset (:numref:`sec_fashion_mnist`).
-->

Để so sánh với kết quả đã đạt được trước đó bằng hồi quy (tuyến tính) softmax (:numref:`sec_softmax_scratch`), chúng ta sẽ tiếp tục sử dụng tập dữ liệu phân loại ảnh Fashion-MNIST.

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
```

<!--
## Initializing Model Parameters
-->

## Khởi tạo Tham số Mô hình

<!--
Recall that Fashion-MNIST contains $10$ classes, and that each image consists of a $28 \times 28 = 784$ grid of (black and white) pixel values.
Again, we will disregard the spatial structure among the pixels (for now), so we can think of this as simply a classification dataset with $784$ input features and $10$ classes.
To begin, we will implement an MLP with one hidden layer and $256$ hidden units.
Note that we can regard both of these quantities as *hyperparameters* and ought in general to set them based on performance on validation data.
Typically, we choose layer widths in powers of $2$ which tends to be computationally efficient because of how memory is alotted and addressed in hardware.
-->

Nhắc lại rằng Fashion-MNIST gồm có $10$ lớp, mỗi ảnh là một lưới có $28 \times 28 = 784$ điểm ảnh (đen và trắng).
Chúng ta sẽ lại (tạm thời) bỏ qua mối liên hệ về mặt không gian giữa các điểm ảnh, khi đó ta có thể coi nó đơn giản như một tập dữ liệu phân loại với $784$ đặc trưng đầu vào và $10$ lớp.
Để bắt đầu, chúng ta sẽ lập trình một mạng MLP chỉ có một tầng ẩn với $256$ nút ẩn.
Lưu ý rằng ta có thể coi cả hai đại lượng này là các *siêu tham số* và ta nên thiết lập giá trị cho chúng dựa vào chất lượng trên tập kiểm định. 
Thông thường, chúng ta sẽ chọn độ rộng của các tầng là các lũy thừa bậc $2$ để giúp việc tính toán hiệu quả hơn do cách mà bộ nhớ được cấp phát và địa chỉ hóa ở phần cứng.

<!--
Again, we will represent our parameters with several `ndarray`s.
Note that *for every layer*, we must keep track of one weight matrix and one bias vector.
As always, we call `attach_grad` to allocate memory for the gradients (of the loss) with respect to these parameters.
-->

Chúng ta sẽ lại biểu diễn các tham số bằng một vài `ndarray`.
Lưu ý rằng *với mỗi tầng*, ta luôn phải giữ một ma trận trọng số và một vector chứa hệ số điều chỉnh.
Và như mọi khi, ta gọi hàm `attach_grad` để cấp phát bộ nhớ cho gradient (của hàm mất mát) theo các tham số này.

```{.python .input  n=3}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = np.random.normal(scale=0.01, size=(num_inputs, num_hiddens))
b1 = np.zeros(num_hiddens)
W2 = np.random.normal(scale=0.01, size=(num_hiddens, num_outputs))
b2 = np.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Activation Function
-->

## Hàm Kích hoạt

<!--
To make sure we know how everything works, we will implement the ReLU activation ourselves using the `maximum` function rather than invoking `npx.relu` directly.
-->

Để đảm bảo rằng ta biết mọi thứ hoạt động như thế nào, chúng ta sẽ tự lập trình hàm kích hoạt ReLU bằng cách sử dụng hàm `maximum` thay vì gọi trực tiếp hàm `npx.relu`.

```{.python .input  n=4}
def relu(X):
    return np.maximum(X, 0)
```

<!--
## The model
-->

## Mô hình

<!--
Because we are disregarding spatial structure, we `reshape` each 2D image into a flat vector of length  `num_inputs`.
Finally, we implement our model with just a few lines of code.
-->

Vì ta đang bỏ qua mối liên hệ về mặt không gian giữa các điểm ảnh, ta `reshape` mỗi bức ảnh 2D thành một vector phẳng có độ dài `num_inputs`.
Cuối cùng, ta có được mô hình chỉ với một vài dòng mã nguồn.

```{.python .input  n=5}
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(np.dot(X, W1) + b1)
    return np.dot(H, W2) + b2
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## The Loss Function
-->

## Hàm mất mát

<!--
To ensure numerical stability (and because we already implemented the softmax function from scratch (:numref:`sec_softmax_scratch`), 
we leverage Gluon's integrated function for calculating the softmax and cross-entropy loss.
Recall our easlier discussion of these intricacies (:numref:`sec_mlp`).
We encourage the interested reader to examine the source code for `mxnet.gluon.loss.SoftmaxCrossEntropyLoss` to deepen their knowledge of implementation details.
-->

Để đảm bảo tính ổn định số học (và cũng bởi ta đã lập trình hàm softmax từ đầu ở :numref:`sec_softmax_scratch`), ta sẽ tận dụng luôn các hàm số đã tích hợp sẵn của Gluon để tính softmax và mất mát entropy chéo. 
Nhắc lại phần thảo luận của chúng ta trước đó về vấn đề rắc rối này (:numref:`sec_mlp`).
Chúng tôi khuyến khích bạn đọc quan tâm hãy thử kiểm tra mã nguồn trong `mxnet.gluon.loss.SoftmaxCrossEntropyLoss` để hiểu thêm về cách lập trình chi tiết.

```{.python .input  n=6}
loss = gluon.loss.SoftmaxCrossEntropyLoss()
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Training
-->

## Huấn luyện

<!--
Fortunately, the training loop for MLPs is exactly the same as for softmax regression.
Leveraging the `d2l` package again, we call the `train_ch3` function (see :numref:`sec_softmax_scratch`), setting the number of epochs to $10$ and the learning rate to $0.5$.
-->

Thật may, vòng lặp huấn luyện của MLP giống hệt với vòng lặp của hồi quy softmax.
Tận dụng gói `d2l`, ta gọi hàm `train_ch3` (xem :numref:`sec_softmax_scratch`), đặt số epoch bằng $10$ và tốc độ học bằng $0.5$

```{.python .input  n=7}
num_epochs, lr = 10, 0.5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs,
              lambda batch_size: d2l.sgd(params, lr, batch_size))
```

<!--
To evaluate the learned model, we apply it on some test data.
-->

Để đánh giá mô hình sau khi học xong, chúng ta sẽ áp dụng nó vào dữ liệu kiểm tra.

```{.python .input  n=8}
d2l.predict_ch3(net, test_iter)
```

<!--
This looks a bit better than our previous result, using simple linear models and gives us some signal that we are on the right path.
-->

Kết quả này tốt hơn một chút so với kết quả trước đây của các mô hình tuyến tính và điều này cho thấy chúng ta đang đi đúng hướng.

<!--
## Summary
-->

## Tóm tắt

<!--
We saw that implementing a simple MLP is easy, even when done manually.
That said, with a large number of layers, this can still get messy (e.g., naming and keeping track of our model's parameters, etc).
-->

Chúng ta đã thấy việc lập trình một MLP đơn giản khá là dễ dàng, ngay cả khi phải làm thủ công.
Tuy vậy, với một số lượng tầng lớn, việc này có thể sẽ trở nên rắc rối (ví dụ như đặt tên và theo dõi các tham số của mô hình, v.v.).

<!--
## Exercises
-->

## Bài tập

<!--
1. Change the value of the hyperparameter `num_hiddens` and see how this hyperparameter influences your results. Determine the best value of this hyperparameter, keeping all others constant.
2. Try adding an additional hidden layer to see how it affects the results.
3. How does changing the learning rate alter your results? Fixing the model architecture and other hyperparameters (including number of epochs), what learning rate gives you the best results?
4. What is the best result you can get by optimizing over all the parameters (learning rate, iterations, number of hidden layers, number of hidden units per layer) jointly?
5. Describe why it is much more challenging to deal with multiple hyperparameters.
6. What is the smartest strategy you can think of for structuring a search over multiple hyperparameters?
-->

1. Thay đổi giá trị của siêu tham số `num_hiddens` và quan sát xem nó ảnh hưởng như thế nào tới kết quả. Giữ nguyên các siêu tham số khác, xác định giá trị tốt nhất của siêu tham số này.
2. Thử thêm vào một tầng ẩn và quan sát xem nó ảnh hưởng như thế nào tới kết quả.
3. Việc thay đổi tốc độ học ảnh hưởng như thế nào tới kết quả? Giữ nguyên kiến trúc mô hình và các siêu tham số khác (bao gồm cả số lượng epoch), tốc độ học nào cho kết quả tốt nhất?
4. Kết quả tốt nhất mà bạn đạt được khi tối ưu hóa tất cả các tham số, gồm tốc độ học, số lượng vòng lặp, số lượng tầng ẩn, số lượng các nút ẩn của mỗi tầng là bao nhiêu?
5. Giải thích tại sao việc phải xử lý nhiều siêu tham số lại gây ra nhiều khó khăn hơn.
6. Đâu là chiến lược thông minh nhất bạn có thể nghĩ ra để tìm kiếm giá trị cho nhiều siêu tham số?

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2339)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2339)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
* Nguyễn Văn Tâm
* Phạm Hồng Vinh
* Vũ Hữu Tiệp
* Nguyễn Duy Du
* Phạm Minh Đức
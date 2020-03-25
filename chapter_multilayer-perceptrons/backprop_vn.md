<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Forward Propagation, Backward Propagation, and Computational Graphs
-->

# Lan truyền xuôi, Lan truyền ngược và Đồ thị tính toán
:label:`sec_backprop`

<!--
So far, we have trained our models with minibatch stochastic gradient descent.
However, when we implemented the algorithm, we only worried about the calculations involved in *forward propagation* through the model.
When it came time to calculate the gradients, we just invoked the `backward` function, relying on the `autograd` module to know what to do.
-->

Cho đến lúc này, ta đã huấn luyện các mô hình với giải thuật hạ gradient ngẫu nhiên theo minibatch.
Tuy nhiên, khi lập trình thuật toán, ta mới chỉ bận tâm đến các phép tính trong quá trình *lan truyền xuôi* qua mô hình.
Khi cần tính gradient ta chỉ đơn giản gọi hàm `backward`, còn việc tính toán chi tiết được trông cậy vào mô-đun `autograd`.

<!--
The automatic calculation of gradients profoundly simplifies the implementation of deep learning algorithms.
Before automatic differentiation, even small changes to complicated models required recalculating complicated derivatives by hand.
Surprisingly often, academic papers had to allocate numerous pages to deriving update rules.
While we must continue to rely on `autograd` so we can focus on the interesting parts, 
you ought to *know* how these gradients are calculated under the hood if you want to go beyond a shallow understanding of deep learning.
-->

Việc tính toán gradient tự động đã giúp công việc lập trình các thuật toán học sâu được đơn giản hóa đi rất nhiều.
Trước đây, khi chưa có công cụ tính vi phân tự động, đối với các mô hình phức tạp thì ngay cả những thay đổi nhỏ cũng yêu cầu tính lại các đạo hàm rắc rối một cách thủ công.
Điều đáng ngạc nhiên là các bài báo học thuật thường dành rất nhiều trang để rút ra các nguyên tắc cập nhật.
Vậy nên, mặc dù ta tiếp tục phải dựa vào `autograd` để có thể tập trung vào những phần thú vị, bạn nên *nắm bắt* rõ cách tính gradient nếu bạn muốn tiến xa hơn là chỉ hiểu biết hời hợt về học sâu.

<!--
In this section, we take a deep dive into the details of backward propagation (more commonly called *backpropagation* or *backprop*).
To convey some insight for both the techniques and their implementations, we rely on some basic mathematics and computational graphs.
To start, we focus our exposition on a three layer (one hidden) multilayer perceptron with weight decay ($\ell_2$ regularization).
-->

Trong mục này, ta sẽ đi sâu vào chi tiết của lan truyền ngược (thường được gọi là *backpropagation* hoặc *backprop*). Ta sẽ sử dụng một vài công thức toán học cơ bản và đồ thị tính toán để giải thích một cách chi tiết cách thức hoạt động cũng như cách lập trình các kỹ thuật này.
Và để bắt đầu, ta sẽ tập trung việc giải trình vào một perceptron đa tầng gồm ba tầng (một tầng ẩn) sử dụng suy giảm trọng số (điều chuẩn $\ell_2$).

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Forward Propagation
-->

## *dịch tiêu đề phía trên*

<!--
Forward propagation refers to the calculation and storage of intermediate variables (including outputs) for the neural network in order from the input layer to the output layer.
We now work step-by-step through the mechanics of a deep network with one hidden layer.
This may seem tedious but in the eternal words of funk virtuoso James Brown, you must "pay the cost to be the boss".
-->

*dịch đoạn phía trên*


<!--
For the sake of simplicity, let’s assume that the input example is $\mathbf{x}\in \mathbb{R}^d$ and that our hidden layer does not include a bias term.
Here the intermediate variable is:
-->

*dịch đoạn phía trên*

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

<!--
where $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ is the weight parameter of the hidden layer.
After running the intermediate variable $\mathbf{z}\in \mathbb{R}^h$ through the activation function $\phi$ we obtain our hidden activations vector of length $h$,
-->

*dịch đoạn phía trên*

$$\mathbf{h}= \phi (\mathbf{z}).$$

<!--
The hidden variable $\mathbf{h}$ is also an intermediate variable.
Assuming the parameters of the output layer only possess a weight of $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$, we can obtain an output layer variable with a vector length of $q$:
-->

*dịch đoạn phía trên*

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

<!--
Assuming the loss function is $l$ and the example label is $y$, we can then calculate the loss term for a single data example,
-->

*dịch đoạn phía trên*

$$L = l(\mathbf{o}, y).$$

<!--
According to the definition of $\ell_2$ regularization, given the hyperparameter $\lambda$, the regularization term is
-->

*dịch đoạn phía trên*

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$

<!--
where the Frobenius norm of the matrix is simply the $L_2$ norm applied after flattening the matrix into a vector.
Finally, the model's regularized loss on a given data example is:
-->

*dịch đoạn phía trên*

$$J = L + s.$$

<!--
We refer to $J$ the *objective function* in the following discussion.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Computational Graph of Forward Propagation
-->

## *dịch tiêu đề phía trên*

<!--
Plotting computational graphs helps us visualize the dependencies of operators and variables within the calculation. 
:numref:`fig_forward` contains the graph associated with the simple network described above.
The lower-left corner signifies the input and the upper right corner the output.
Notice that the direction of the arrows (which illustrate data flow) are primarily rightward and upward.
-->

*dịch đoạn phía trên*

<!--
![Computational Graph](../img/forward.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/forward.svg)
:label:`fig_forward`


<!--
## Backpropagation
-->

## *dịch tiêu đề phía trên*

<!--
Backpropagation refers to the method of calculating the gradient of neural network parameters.
In short, the method traverses the network in reverse order, from the output to the input layer, according ot the *chain rule* from calculus.
The algorithm, stores any intermediate variables (partial derivatives) requried while calculating the gradient with respect to some parameters.
Assume that we have functions $\mathsf{Y}=f(\mathsf{X})$ and $\mathsf{Z}=g(\mathsf{Y}) = g \circ f(\mathsf{X})$, 
in which the input and the output $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ are tensors of arbitrary shapes.
By using the chain rule, we can compute the derivative of $\mathsf{Z}$ wrt. $\mathsf{X}$ via
-->

*dịch đoạn phía trên*

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

<!--
Here we use the $\text{prod}$ operator to multiply its arguments after the necessary operations, such as transposition and swapping input positions have been carried out.
For vectors, this is straightforward: it is simply matrix-matrix multiplication.
For higher dimensional tensors, we use the appropriate counterpart.
The operator $\text{prod}$ hides all the notation overhead.
-->

*dịch đoạn phía trên*

<!--
The parameters of the simple network with one hidden layer are $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$.
The objective of backpropagation is to calculate the gradients $\partial J/\partial \mathbf{W}^{(1)}$ and $\partial J/\partial \mathbf{W}^{(2)}$.
To accomplish this, we apply the chain rule and calculate, in turn, the gradient of each intermediate variable and parameter.
The order of calculations are reversed relative to those performed in forward propagation, since we need to start with the outcome of the compute graph and work our way towards the parameters.
The first step is to calculate the gradients of the objective function $J=L+s$ with respect to the loss term $L$ and the regularization term $s$.
-->

*dịch đoạn phía trên*

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

<!--
Next, we compute the gradient of the objective function with respect to variable of the output layer $\mathbf{o}$ according to the chain rule.
-->

*dịch đoạn phía trên*

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

<!--
Next, we calculate the gradients of the regularization term with respect to both parameters.
-->

*dịch đoạn phía trên*

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
Now we are able calculate the gradient $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ of the model parameters closest to the output layer.
Using the chain rule yields:
-->

Bây giờ chúng ta có thể tính gradient $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ của các tham số mô hình gần nhất với lớp đầu ra. Áp dụng quy tắc dây chuyền:

$$
\frac{\partial J}{\partial \mathbf{W}^{(2)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)
= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.
$$

<!--
To obtain the gradient with respect to $\mathbf{W}^{(1)}$ we need to continue backpropagation along the output layer to the hidden layer.
The gradient with respect to the hidden layer's outputs $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ is given by
-->

Để tính được gradient của $\mathbf{W}^{(1)}$ ta cần tiếp tục lan truyền ngược từ tầng đầu ra đến các tầng ẩn. Gradient của các đầu ra từ tầng ẩn \partial J/\partial \mathbf{h} \in \mathbb{R}^h$ được tính như sau:


$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

<!--
Since the activation function $\phi$ applies elementwise, calculating the gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ 
of the intermediate variable $\mathbf{z}$ requires that we use the elementwise multiplication operator, which we denote by $\odot$.
-->

Vì hàm kích hoạt $\phi$ áp dụng cho từng phần tử, việc tính gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ của biến trung gian \mathbf{z}$ đòi hỏi chúng ta sử dụng phép nhân theo từng phần tử, biểu diễn bởi $\odot$.

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

<!--
Finally, we can obtain the gradient $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ of the model parameters closest to the input layer.
According to the chain rule, we get
-->

Cuối cùng, ta có được gradient $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ của các tham số mô hình gần nhất với lớp đầu vào. Theo quy tắc dây chuyền, ta có

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Training a Model
-->

## *dịch tiêu đề phía trên*

<!--
When training networks, forward and backward propagation depend on each other.
In particular, for forward propagation, we traverse the compute graph in the direction of dependencies and compute all the variables on its path.
These are then used for backpropagation where the compute order on the graph is reversed.
One of the consequences is that we need to retain the intermediate values until backpropagation is complete.
This is also one of the reasons why backpropagation requires significantly more memory than plain prediction.
We compute tensors as gradients and need to retain all the intermediate variables to invoke the chain rule.
Another reason is that we typically train with minibatches containing more than one variable, thus more intermediate activations need to be stored.
-->

*dịch đoạn phía trên*

<!--
## Summary
-->

## Tóm tắt

<!--
* Forward propagation sequentially calculates and stores intermediate variables within the compute graph defined by the neural network. It proceeds from input to output layer.
* Back propagation sequentially calculates and stores the gradients of intermediate variables and parameters within the neural network in the reversed order.
* When training deep learning models, forward propagation and back propagation are interdependent.
* Training requires significantly more memory and storage.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## Bài tập

<!--
1. Assume that the inputs $\mathbf{x}$ to some scalar function $f$ are $n \times m$ matrices. What is the dimensionality of the gradient of $f$ with respect to $\mathbf{x}?
2. Add a bias to the hidden layer of the model described in this section.
    * Draw the corresponding compute graph.
    * Derive the forward and backward propagation equations.
3. Compute the memory footprint for training and inference in model described in the current chapter.
4. Assume that you want to compute *second* derivatives. What happens to the compute graph? How long do you expect the calculation to take?
5. Assume that the compute graph is too large for your GPU.
    * Can you partition it over more than one GPU?
    * What are the advantages and disadvantages over training on a smaller minibatch?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2344)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2344)
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
<!-- Phần 1 -->
* Nguyễn Duy Du

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
* Nguyễn Lê Quang Nhật

<!-- Phần 5 -->
*

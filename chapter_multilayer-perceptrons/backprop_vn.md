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
Khi cần tính gradient, ta mới chỉ đơn giản gọi hàm `backward` và mô-đun `autograd` sẽ lo các chi tiết tính toán.

<!--
The automatic calculation of gradients profoundly simplifies the implementation of deep learning algorithms.
Before automatic differentiation, even small changes to complicated models required recalculating complicated derivatives by hand.
Surprisingly often, academic papers had to allocate numerous pages to deriving update rules.
While we must continue to rely on `autograd` so we can focus on the interesting parts, 
you ought to *know* how these gradients are calculated under the hood if you want to go beyond a shallow understanding of deep learning.
-->

Việc tính toán gradient tự động sẽ giúp công việc lập trình các thuật toán học sâu được đơn giản hóa đi rất nhiều.
Trước đây, khi chưa có công cụ tính vi phân tự động, ngay cả khi ta chỉ thay đổi một chút các mô hình phức tạp, các đạo hàm rắc rối cũng cần phải được tính lại một cách thủ công.
Điều đáng ngạc nhiên là các bài báo học thuật thường có các công thức cập nhật mô hình dài hàng trang giấy.
Vậy nên dù vẫn phải tiếp tục dựa vào `autograd` để có thể tập trung vào những phần thú vị của học sâu, bạn vẫn nên *nắm* rõ thay vì chỉ hiểu một cách hời hợt cách tính gradient nếu bạn muốn tiến xa hơn.

<!--
In this section, we take a deep dive into the details of backward propagation (more commonly called *backpropagation* or *backprop*).
To convey some insight for both the techniques and their implementations, we rely on some basic mathematics and computational graphs.
To start, we focus our exposition on a three layer (one hidden) multilayer perceptron with weight decay ($\ell_2$ regularization).
-->

Trong mục này, ta sẽ đi sâu vào chi tiết của lan truyền ngược (thường được gọi là *backpropagation* hoặc *backprop*). 
Ta sẽ sử dụng một vài công thức toán học cơ bản và đồ thị tính toán để giải thích một cách chi tiết cách thức hoạt động cũng như cách lập trình các kỹ thuật này.
Và để bắt đầu, ta sẽ tập trung giải trình một perceptron đa tầng gồm ba tầng (một tầng ẩn) đi kèm với suy giảm trọng số (điều chuẩn $\ell_2$).

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Forward Propagation
-->

## Lan truyền Xuôi

<!--
Forward propagation refers to the calculation and storage of intermediate variables (including outputs) for the neural network in order from the input layer to the output layer.
We now work step-by-step through the mechanics of a deep network with one hidden layer.
This may seem tedious but in the eternal words of funk virtuoso James Brown, you must "pay the cost to be the boss".
-->

Lan truyền xuôi là quá trình tính toán cũng như lưu trữ các biến trung gian (bao gồm cả đầu ra) của mạng nơ-ron theo thứ tự từ tầng đầu vào đến tầng đầu ra.
Bây giờ ta sẽ thực hiện qua từng bước trong cơ chế vận hành của mạng nơ-ron sâu có một tầng ẩn.
Điều này nghe có vẻ tẻ nhạt nhưng theo như cách nói dân giã, bạn phải "tập đi trước khi tập chạy".

<!--
For the sake of simplicity, let’s assume that the input example is $\mathbf{x}\in \mathbb{R}^d$ and that our hidden layer does not include a bias term.
Here the intermediate variable is:
-->

Để đơn giản hóa vấn đề, ta giả sử mẫu đầu vào là $\mathbf{x}\in \mathbb{R}^d$ và tầng ẩn của ta không có hệ số điều chỉnh.
Ở đây biến trung gian là:

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

<!--
where $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ is the weight parameter of the hidden layer.
After running the intermediate variable $\mathbf{z}\in \mathbb{R}^h$ through the activation function $\phi$ we obtain our hidden activations vector of length $h$,
-->

trong đó $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ là tham số trọng số của tầng ẩn.
Sau khi đưa biến trung gian $\mathbf{z}\in \mathbb{R}^h$ qua hàm kích hoạt $\phi$, ta thu được vector kích hoạt ẩn với $h$ phần tử,

$$\mathbf{h}= \phi (\mathbf{z}).$$

<!--
The hidden variable $\mathbf{h}$ is also an intermediate variable.
Assuming the parameters of the output layer only possess a weight of $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$, we can obtain an output layer variable with a vector length of $q$:
-->

Biến ẩn $\mathbf{h}$ cũng là một biến trung gian.
Giả sử tham số của tầng đầu ra chỉ gồm trọng số $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$, ta sẽ thu được một vector với $q$ phần tử ở tầng đầu ra:

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

<!--
Assuming the loss function is $l$ and the example label is $y$, we can then calculate the loss term for a single data example,
-->

Giả sử hàm mất mát là $l$ và nhãn của mẫu là $y$, ta có thể tính được lượng mất mát cho một mẫu dữ liệu duy nhất,

$$L = l(\mathbf{o}, y).$$

<!--
According to the definition of $\ell_2$ regularization, given the hyperparameter $\lambda$, the regularization term is
-->

Theo định nghĩa của điều chuẩn $\ell_2$ với siêu tham số $\lambda$, lượng điều chuẩn là:

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$

<!--
where the Frobenius norm of the matrix is simply the $L_2$ norm applied after flattening the matrix into a vector.
Finally, the model's regularized loss on a given data example is:
-->

trong đó chuẩn Frobenius của ma trận chỉ đơn giản là chuẩn $L_2$ của vector thu được sau khi trải phẳng ma trận.
Cuối cùng, hàm mất mát được điều chuẩn của mô hình trên một mẫu dữ liệu cho trước là:

$$J = L + s.$$

<!--
We refer to $J$ the *objective function* in the following discussion.
-->

Ta sẽ bàn thêm về *hàm mục tiêu* $J$ ở phía dưới.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Computational Graph of Forward Propagation
-->

## Đồ thị Tính toán của Lan truyền Xuôi

<!--
Plotting computational graphs helps us visualize the dependencies of operators and variables within the calculation. 
:numref:`fig_forward` contains the graph associated with the simple network described above.
The lower-left corner signifies the input and the upper right corner the output.
Notice that the direction of the arrows (which illustrate data flow) are primarily rightward and upward.
-->

Vẽ đồ thị tính toán giúp chúng ta hình dung được sự phụ thuộc giữa các toán tử và các biến trong quá trình tính toán. 
:numref:`fig_forward` thể hiện đồ thị tương ứng với mạng nơ-ron đã miêu tả ở trên.
Góc trái dưới biểu diễn đầu vào trong khi góc phải trên biểu diễn đầu ra.
Lưu ý rằng hướng của các mũi tên (thể hiện luồng dữ liệu) chủ yếu là đi qua phải và hướng lên trên. 

<!--
![Computational Graph](../img/forward.svg)
-->

![Đồ thị tính toán](../img/forward.svg)
:label:`fig_forward`


<!--
## Backpropagation
-->

## Lan truyền Ngược

<!--
Backpropagation refers to the method of calculating the gradient of neural network parameters.
In short, the method traverses the network in reverse order, from the output to the input layer, according ot the *chain rule* from calculus.
The algorithm, stores any intermediate variables (partial derivatives) requried while calculating the gradient with respect to some parameters.
Assume that we have functions $\mathsf{Y}=f(\mathsf{X})$ and $\mathsf{Z}=g(\mathsf{Y}) = g \circ f(\mathsf{X})$, 
in which the input and the output $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ are tensors of arbitrary shapes.
By using the chain rule, we can compute the derivative of $\mathsf{Z}$ wrt. $\mathsf{X}$ via
-->

Lan truyền ngược là phương pháp tính gradient của các tham số mạng nơ-ron. 
Nói một cách đơn giản, phương thức này duyệt qua mạng nơ-ron theo chiều ngược lại, từ đầu ra đến đầu vào, tuân theo quy tắc dây chuyền trong giải tích.  
Thuật toán lan truyền ngược lưu trữ các biến trung gian (là các đạo hàm riêng) cần thiết trong quá trình tính toán gradient theo các tham số.
Giả sử chúng ta có hàm $\mathsf{Y}=f(\mathsf{X})$ và $\mathsf{Z}=g(\mathsf{Y}) = g \circ f(\mathsf{X})$, 
trong đó đầu vào và đầu ra $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ là các tensor có kích thước bất kỳ. 
Bằng cách sử dụng quy tắc dây chuyền, chúng ta có thể tính đạo hàm của $\mathsf{Z}$ theo $\mathsf{X}$ như sau:

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

<!--
Here we use the $\text{prod}$ operator to multiply its arguments after the necessary operations, such as transposition and swapping input positions have been carried out.
For vectors, this is straightforward: it is simply matrix-matrix multiplication.
For higher dimensional tensors, we use the appropriate counterpart.
The operator $\text{prod}$ hides all the notation overhead.
-->

Ở đây, chúng ta sử dụng toán tử $\text{prod}$ để nhân các đối số sau khi các phép tính cần thiết như là chuyển vị và hoán đổi đã được thực hiện. 
Với vector, điều này khá đơn giản: nó chỉ đơn thuần là phép nhân ma trận. 
Với các tensor nhiều chiều thì sẽ có các phương án tương ứng phù hợp. 
Toán tử $\text{prod}$ sẽ đơn giản hoá việc ký hiệu.

<!--
The parameters of the simple network with one hidden layer are $\mathbf{W}^{(1)}$ and $\mathbf{W}^{(2)}$.
The objective of backpropagation is to calculate the gradients $\partial J/\partial \mathbf{W}^{(1)}$ and $\partial J/\partial \mathbf{W}^{(2)}$.
To accomplish this, we apply the chain rule and calculate, in turn, the gradient of each intermediate variable and parameter.
The order of calculations are reversed relative to those performed in forward propagation, since we need to start with the outcome of the compute graph and work our way towards the parameters.
The first step is to calculate the gradients of the objective function $J=L+s$ with respect to the loss term $L$ and the regularization term $s$.
-->

Các tham số của mạng nơ-ron đơn giản với một tầng ẩn là $\mathbf{W}^{(1)}$ và $\mathbf{W}^{(2)}$.
Mục đích của lan truyền ngược là để tính gradient $\partial J/\partial \mathbf{W}^{(1)}$ và $\partial J/\partial \mathbf{W}^{(2)}$.
Để làm được điều này, ta áp dụng quy tắc dây chuyền và lần lượt tính gradient của các biến trung gian và tham số. 
Các phép tính trong lan truyền ngược có thứ tự ngược lại so với các phép tính trong lan truyền xuôi, bởi ta muốn bắt đầu từ kết quả của đồ thị tính toán rồi dần đi tới các tham số. 
Bước đầu tiên đó là tính gradient của hàm mục tiêu $J=L+s$ theo mất mát $L$ và điều chuẩn $s$. 

$$\frac{\partial J}{\partial L} = 1 \; \text{và} \; \frac{\partial J}{\partial s} = 1.$$

<!--
Next, we compute the gradient of the objective function with respect to variable of the output layer $\mathbf{o}$ according to the chain rule.
-->

Tiếp theo, ta tính gradient của hàm mục tiêu theo các biến của lớp đầu ra $\mathbf{o}$, sử dụng quy tắc dây chuyền.

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

<!--
Next, we calculate the gradients of the regularization term with respect to both parameters.
-->

Kế tiếp, ta tính gradient của điều chuẩn theo cả hai tham số.

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{và} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
Now we are able calculate the gradient $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ of the model parameters closest to the output layer.
Using the chain rule yields:
-->

Bây giờ chúng ta có thể tính gradient $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ của các tham số mô hình gần nhất với tầng đầu ra. Áp dụng quy tắc dây chuyền, ta có: 

$$
\frac{\partial J}{\partial \mathbf{W}^{(2)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)
= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.
$$

<!--
To obtain the gradient with respect to $\mathbf{W}^{(1)}$ we need to continue backpropagation along the output layer to the hidden layer.
The gradient with respect to the hidden layer's outputs $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ is given by
-->

Để tính được gradient theo $\mathbf{W}^{(1)}$ ta cần tiếp tục lan truyền ngược từ tầng đầu ra đến các tầng ẩn.
Gradient theo các đầu ra của tầng ẩn $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ được tính như sau:


$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

<!--
Since the activation function $\phi$ applies elementwise, calculating the gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ 
of the intermediate variable $\mathbf{z}$ requires that we use the elementwise multiplication operator, which we denote by $\odot$.
-->

Vì hàm kích hoạt $\phi$ áp dụng cho từng phần tử, việc tính gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ của biến trung gian $\mathbf{z}$ cũng yêu cầu sử dụng phép nhân theo từng phần tử, kí hiệu bởi $\odot$.

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

<!--
Finally, we can obtain the gradient $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ of the model parameters closest to the input layer.
According to the chain rule, we get
-->

Cuối cùng, ta có thể tính gradient $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ của các tham số mô hình gần nhất với tầng đầu vào. Theo quy tắc dây chuyền, ta có

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

## Huấn luyện một Mô hình

<!--
When training networks, forward and backward propagation depend on each other.
In particular, for forward propagation, we traverse the compute graph in the direction of dependencies and compute all the variables on its path.
These are then used for backpropagation where the compute order on the graph is reversed.
One of the consequences is that we need to retain the intermediate values until backpropagation is complete.
This is also one of the reasons why backpropagation requires significantly more memory than plain prediction.
We compute tensors as gradients and need to retain all the intermediate variables to invoke the chain rule.
Another reason is that we typically train with minibatches containing more than one variable, thus more intermediate activations need to be stored.
-->

Khi huấn luyện các mạng nơ-ron, lan truyền xuôi và lan truyền ngược phụ thuộc lẫn nhau. 
Cụ thể với lan truyền xuôi, ta duyệt đồ thị tính toán theo hướng của các quan hệ phụ thuộc và tính tất cả các biến trên đường đi. 
Những biến này sau đó được sử dụng trong lan truyền ngược khi thứ tự tính toán trên đồ thị bị đảo ngược lại. 
Hệ quả là ta cần lưu trữ các giá trị trung gian cho đến khi lan truyền ngược hoàn tất. 
Đây cũng chính là một trong những lý do khiến lan truyền ngược yêu cầu nhiều bộ nhớ hơn đáng kể so với khi chỉ cần đưa ra dự đoán.  
Ta tính các tensor gradient và giữ các biến trung gian lại để sử dụng trong quy tắc dây chuyền. 
Việc huấn luyện trên các minibatch chứa nhiều mẫu, do đó cần lưu trữ nhiều giá trị kích hoạt trung gian hơn cũng là một lý do khác. <!-- Chỗ này bản gốc viết sai nhỉ, minibatch chứa nhiều samples chứ sao lại variables-->

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

* Lan truyền xuôi lần lượt tính và lưu trữ các biến trung gian từ tầng đầu vào đến tầng đầu ra trong đồ thị tính toán được định nghĩa bởi mạng nơ-ron.
* Lan truyền ngược lần lượt tính và lưu trữ các gradient của biến trung gian và tham số mạng nơ-ron theo chiều ngược lại. 
* Khi huấn luyện các mô hình học sâu, lan truyền xuôi và lan truyền ngược phụ thuộc lẫn nhau. 
* Việc huấn luyện cần nhiều bộ nhớ lưu trữ hơn đáng kể so với việc dự đoán. 


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

1. Giả sử đầu vào $\mathbf{x}$ của hàm số vô hướng $f$ là ma trận $n \times m$. Gradient của $f$ theo $\mathbf{x}$ có chiều là bao nhiêu?
2. Thêm một hệ số điều chỉnh vào tầng ẩn của mô hình được mô tả ở trên.
    * Vẽ đồ thị tính toán tương ứng.
    * Tìm các phương trình cho quá trình lan truyền xuôi và lan truyền ngược.
3. Tính lượng bộ nhớ mà mô hình được mô tả ở chương này sử dụng lúc huấn luyện và lúc dự đoán.
4. Giả sử bạn muốn tính đạo hàm *bậc hai*. Điều gì sẽ xảy ra với đồ thị tính toán? Hãy ước tính thời gian hoàn thành quá trình này?
5. Giả sử rằng đồ thị tính toán trên là quá sức với GPU của bạn.
    * Bạn có thể phân vùng nó trên nhiều GPU không?
    * Ưu điểm và nhược điểm của việc huấn luyện với một minibatch nhỏ hơn là gì?

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
* Nguyễn Duy Du
* Lý Phi Long
* Lê Khắc Hồng Phúc
* Phạm Minh Đức
* Nguyễn Lê Quang Nhật
* Phạm Ngọc Bảo Anh
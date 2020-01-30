<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Notation
-->

# Ký hiệu
:label:`chap_notation`

<!--
The notation used throughout this book is summarized below.
-->

Các ký hiệu sử dụng trong cuốn sách này được tổng hợp dưới đây.


<!--
## Numbers
-->

## Số

<!--
* $x$: A scalar
* $\mathbf{x}$: A vector
* $\mathbf{X}$: A matrix
* $\mathsf{X}$: A tensor
* $\mathbf{I}$: An identity matrix
* $x_i$, $[\mathbf{x}]_i$: The $i^\mathrm{th}$ element of vector $\mathbf{x}$
* $x_{ij}$, $[\mathbf{X}]_{ij}$: The element of matrix $\mathbf{X}$ at row $i$ and column $j$
-->

* $x$: một số vô hướng
* $\mathbf{x}$: một vector
* $\mathbf{X}$: một ma trận
* $\mathsf{X}$: một tensor
* $\mathbf{I}$: một ma trận đồng nhất
* $x_i$, $[\mathbf{x}]_i$: phần tử thứ $i$ của vector $\mathbf{x}$
* $x_{ij}$, $[\mathbf{X}]_{ij}$: phần tử ở hàng thứ $i$, cột thứ $j$ của ma trận $\mathbf{X}$



<!--
## Set Theory
-->

## Lý thuyết Tập hợp


<!--
* $\mathcal{X}$: A set
* $\mathbb{Z}$: The set of integers
* $\mathbb{R}$: The set of real numbers
* $\mathbb{R}^n$: The set of $n$-dimensional vectors of real numbers
* $\mathbb{R}^{a\times b}$: The set of matrices of real numbers with $a$ rows and $b$ columns
* $\mathcal{A}\cup\mathcal{B}$: Union of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: Intersection of sets $\mathcal{A}$ and $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: Subtraction of set $\mathcal{B}$ from set $\mathcal{A}$
-->

* $\mathcal{X}$: một tập hợp
* $\mathbb{Z}$: tập hợp các số nguyên
* $\mathbb{R}$: tập hợp các số thực
* $\mathbb{R}^n$: tập các vector thực trong không gian $n$ chiều
* $\mathbb{R}^{a\times b}$: tâp hợp các ma trận thực với $a$ hàng và $b$ cột
* $\mathcal{A}\cup\mathcal{B}$: hợp của hai tập hợp $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: giao của hai tập hợp $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: hiệu của tập $\mathcal{A}$ và tập $\mathcal{B}$ (là tập hợp gồm các phần tử thuộc $\mathcal{A}$ nhưng không thuộc $\mathcal{B}$)


<!--
## Functions and Operators
-->

## Hàm số và các Phép toán


<!--
* $f(\cdot)$: A function
* $\log(\cdot)$: The natural logarithm
* $\exp(\cdot)$: The exponential function
* $\mathbf{1}_\mathcal{X}$: The indicator function
* $\mathbf{(\cdot)}^\top$: Transpose of a vector or a matrix
* $\mathbf{X}^{-1}$: Inverse of matrix $\mathbf{X}$
* $\odot$: Hadamard (elementwise) product
* $\lvert \mathcal{X} \rvert$: Cardinality of set $\mathcal{X}$
* $\|\cdot\|_p$: $\ell_p$ norm
* $\|\cdot\|$: $\ell_2$ norm
* $\langle \mathbf{x}, \mathbf{y} \rangle$: Dot product of vectors $\mathbf{x}$ and $\mathbf{y}$
* $\sum$: Series addition
* $\prod$: Series multiplication
-->

* $f(\cdot)$: một hàm số
* $\log(\cdot)$: logarit tự nhiên
* $\exp(\cdot)$: hàm $e$ mũ
* $\mathbf{1}_\mathcal{X}$: hàm đặc trưng (trả về 1 nếu đối số là một phần tử thuộc $\mathcal{X}$, trả về 0 trong trường hợp còn lại).
* $\mathbf{(\cdot)}^\top$: chuyển vị của một vector hoặc một ma trận
* $\mathbf{X}^{-1}$: nghịch đảo của ma trận $\mathbf{X}$
* $\odot$: tích Hadamard (theo từng thành phần)
* $\lvert \mathcal{X} \rvert$: card (số phần tử) của tập $\mathcal{X}$
* $\|\cdot\|_p$: chuẩn $\ell_p$
* $\|\cdot\|$: chuẩn $\ell_2$
* $\langle \mathbf{x}, \mathbf{y} \rangle$: tích vô hướng của hai vector  $\mathbf{x}$ và $\mathbf{y}$
* $\sum$: tổng của một dãy
* $\prod$: tích của một dãy


<!--
## Calculus
-->

## Giải tích

<!--
* $\frac{dy}{dx}$: Derivative of $y$ with respect to $x$
* $\frac{\partial y}{\partial x}$: Partial derivative of $y$ with respect to $x$
* $\nabla_{\mathbf{x}} y$: Gradient of $y$ with respect to $\mathbf{x}$
* $\int_a^b f(x) \;dx$: Definite integral of $f$ from $a$ to $b$ with respect to $x$
* $\int f(x) \;dx$: Indefinite integral of $f$ with respect to $x$
-->

* $\frac{dy}{dx}$: đạo hàm của $y$ theo $x$
* $\frac{\partial y}{\partial x}$: đạo hàm riêng của $y$ theo $x$
* $\nabla_{\mathbf{x}} y$: Gradient của $y$ theo vector $\mathbf{x}$
* $\int_a^b f(x) \;dx$: tích phân của $f$ từ $a$ đến $b$ theo $x$
* $\int f(x) \;dx$: nguyên hàm của $f$ theo $x$

<!--
## Probability and Information Theory
-->

## Xác suất và Lý thuyết Thông tin

<!--
* $P(\cdot)$: Probability distribution
* $z \sim P$: Random variable $z$ has probability distribution $P$
* $P(X \mid Y)$: Conditional probability of $X \mid Y$
* $p(x)$: probability density function
* ${E}_{x} [f(x)]$: Expectation of $f$ with respect to $x$
* $X \perp Y$: Random variables $X$ and $Y$ are independent
* $X \perp Y \mid Z$: Random variables  $X$  and  $Y$  are conditionally independent given random variable $Z$
* $\mathrm{Var}(X)$: Variance of random variable $X$
* $\sigma_X$: Standard deviation of random variable $X$
* $\mathrm{Cov}(X, Y)$: Covariance of random variables $X$ and $Y$
* $\rho(X, Y)$: Correlation of random variables $X$ and $Y$
* $H(X)$: Entropy of random variable $X$
* $D_{\mathrm{KL}}(P\|Q)$: KL-divergence of distributions $P$ and $Q$
-->

* $P(\cdot)$: phân phối xác suất
* $z \sim P$: biến ngẫu nhiên $z$ tuân theo phân phối xác suất $P$
* $P(X \mid Y)$: xác suất của $X$ với điều kiện $Y$
* $p(x)$: hàm mật độ xác suất
* ${E}_{x} [f(x)]$: kỳ vọng của $f$ theo $x$
* $X \perp Y$: hai biến ngẫu nhiên $X$ và $Y$ là độc lập
* $X \perp Y \mid Z$: hai biến ngẫu nhiên $X$ và $Y$ là độc lập có điều kiện nếu cho trước biến ngẫu nhiên $Z$
* $\mathrm{Var}(X)$: phương sai của biến ngẫu nhiên $X$
* $\sigma_X$: độ lệch chuẩn của biến ngẫu nhiên $X$
* $\mathrm{Cov}(X, Y)$: hiệp phương sai của hai biến ngẫu nhiên $X$ và $Y$
* $\rho(X, Y)$: độ tương quan của hai biến ngẫu nhiên $X$ và $Y$
* $H(X)$: Entropy của biến ngẫu nhiên $X$
* $D_{\mathrm{KL}}(P\|Q)$: phân kỳ KL của hai phân phối $P$ và $Q$



<!--
## Complexity
-->

## Độ phức tạp

<!--
* $\mathcal{O}$: Big O notation
-->

* $\mathcal{O}$: Ký hiệu Big O


<!--
## [Discussions](https://discuss.mxnet.io/t/4367)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/4367)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)
<!--
![](../img/qr_notation.svg)
-->

<!-- ===================== Kết thúc dịch Phần 1 ==================== -->

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

<!-- Phần 1 -->
* Vũ Hữu Tiệp
* Đoàn Võ Duy Thanh
* Lê Khắc Hồng Phúc

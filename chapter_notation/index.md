# Ký hiệu
:label:`chap_notation`

Trong suốt cuốn sách này, chúng tôi tuân thủ các quy ước công ước sau đây. Lưu ý rằng một số ký hiệu này là giữ chỗ, trong khi những ký hiệu khác đề cập đến các đối tượng cụ thể. Theo nguyên tắc chung của ngón tay cái, bài viết không xác định “a” chỉ ra rằng ký hiệu là một giữ chỗ và các ký hiệu được định dạng tương tự có thể biểu thị các đối tượng khác cùng loại. Ví dụ, “$x$: một vô hướng” có nghĩa là các chữ cái viết thường đại diện cho các giá trị vô hướng. 

## Các đối tượng số

* $x$: một vô hướng
* $\mathbf{x}$: một vector
* $\mathbf{X}$: một ma trận
* $\mathsf{X}$: một tensor chung
* $\mathbf{I}$: một ma trận nhận dạng - vuông, với $1$ trên tất cả các mục chéo và $0$ trên tất cả các đường chéo
* $x_i$, $[\mathbf{x}]_i$: yếu tố $i^\mathrm{th}$ của vector $\mathbf{x}$
* $x_{ij}$, $x_{i,j}$,$[\mathbf{X}]_{ij}$, $[\mathbf{X}]_{i,j}$: phần tử của ma trận $\mathbf{X}$ tại hàng $i$ và cột $j$.

## Lý thuyết đặt

* $\mathcal{X}$: một bộ
* $\mathbb{Z}$: tập hợp các số nguyên
* $\mathbb{Z}^+$: tập hợp các số nguyên dương
* $\mathbb{R}$: tập hợp các số thực
* $\mathbb{R}^n$: tập hợp các vectơ $n$ chiều của số thực
* $\mathbb{R}^{a\times b}$: Tập hợp các ma trận số thực với $a$ hàng và $b$ cột
* $|\mathcal{X}|$: cardinality (số phần tử) của bộ $\mathcal{X}$
* $\mathcal{A}\cup\mathcal{B}$: liên minh các bộ $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\cap\mathcal{B}$: giao điểm của bộ $\mathcal{A}$ và $\mathcal{B}$
* $\mathcal{A}\setminus\mathcal{B}$: đặt phép trừ $\mathcal{B}$ từ $\mathcal{A}$ (chỉ chứa những yếu tố của $\mathcal{A}$ không thuộc về $\mathcal{B}$)

## Chức năng và toán tử

* $f(\cdot)$: một chức năng
* $\log(\cdot)$: logarit tự nhiên (cơ sở $e$)
* $\log_2(\cdot)$: logarit với cơ sở $2$
* $\exp(\cdot)$: hàm mũ
* $\mathbf{1}(\cdot)$: hàm chỉ báo, đánh giá thành $1$ nếu đối số boolean là đúng và $0$ nếu không
* $\mathbf{1}_{\mathcal{X}}(z)$: chức năng chỉ báo thành viên thiết lập, đánh giá $1$ nếu phần tử $z$ thuộc về bộ $\mathcal{X}$ và $0$ nếu không
* $\mathbf{(\cdot)}^\top$: transpose của một vectơ hoặc ma trận
* $\mathbf{X}^{-1}$: nghịch đảo của ma trận $\mathbf{X}$
* $\odot$: Sản phẩm Hadamard (elementwise)
* $[\cdot, \cdot]$: nối
* $\|\cdot\|_p$:$L_p$ định mức
* $\|\cdot\|$:$L_2$ định mức
* $\langle \mathbf{x}, \mathbf{y} \rangle$: sản phẩm chấm của vectơ $\mathbf{x}$ và $\mathbf{y}$
* $\sum$: tổng kết về một bộ sưu tập các yếu tố
* $\prod$: sản phẩm trên một bộ sưu tập các yếu tố
* $\stackrel{\mathrm{def}}{=}$: một sự bình đẳng khẳng định như một định nghĩa của ký hiệu ở phía bên trái

## Calculus (Giải tích)

* $\frac{dy}{dx}$: dẫn xuất của $y$ đối với $x$
* $\frac{\partial y}{\partial x}$: phái sinh một phần của $y$ đối với $x$
* $\nabla_{\mathbf{x}} y$: gradient của $y$ đối với $\mathbf{x}$
* $\int_a^b f(x) \;dx$: tích phân xác định của $f$ từ $a$ đến $b$ đối với $x$
* $\int f(x) \;dx$: tích phân không xác định của $f$ đối với $x$

## Lý thuyết xác suất và thông tin

* $X$: một biến ngẫu nhiên
* $P$: một phân phối xác suất
* $X \sim P$: biến ngẫu nhiên $X$ có phân phối $P$
* $P(X=x)$: xác suất được gán cho sự kiện mà biến ngẫu nhiên $X$ lấy giá trị $x$
* $P(X \mid Y)$: phân phối xác suất có điều kiện của $X$ cho $Y$
* $p(\cdot)$: một hàm mật độ xác suất (PDF) liên quan đến phân phối P
* ${E}[X]$: kỳ vọng của một biến ngẫu nhiên $X$
* $X \perp Y$: các biến ngẫu nhiên $X$ và $Y$ độc lập
* $X \perp Y \mid Z$: các biến ngẫu nhiên $X$ và $Y$ có điều kiện độc lập cho $Z$
* $\sigma_X$: độ lệch chuẩn của biến ngẫu nhiên $X$
* $\mathrm{Var}(X)$: phương sai của biến ngẫu nhiên $X$, bằng $\sigma^2_X$
* $\mathrm{Cov}(X, Y)$: đồng phương sai của các biến ngẫu nhiên $X$ và $Y$
* $\rho(X, Y)$: hệ số tương quan Pearson giữa $X$ và $Y$, bằng $\frac{\mathrm{Cov}(X, Y)}{\sigma_X \sigma_Y}$
* $H(X)$: entropy của biến ngẫu nhiên $X$
* $D_{\mathrm{KL}}(P\|Q)$: phân kỳ KL-phân kỳ (hoặc entropy tương đối) từ phân phối $Q$ đến phân phối $P$

[Discussions](https://discuss.d2l.ai/t/25)

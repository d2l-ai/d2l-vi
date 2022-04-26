# Eigendecompositions
:label:`sec_eigendecompositions`

Eigenvalues thường là một trong những quan niệm hữu ích nhất mà chúng ta sẽ gặp phải khi nghiên cứu đại số tuyến tính, tuy nhiên, với tư cách là người mới bắt đầu, thật dễ dàng để bỏ qua tầm quan trọng của chúng. Dưới đây, chúng tôi giới thiệu eigendecomposition và cố gắng truyền đạt một số ý thức về lý do tại sao nó lại quan trọng như vậy.  

Giả sử rằng chúng ta có một ma trận $A$ với các mục sau: 

$$
\mathbf{A} = \begin{bmatrix}
2 & 0 \\
0 & -1
\end{bmatrix}.
$$

Nếu chúng ta áp dụng $A$ cho bất kỳ vector $\mathbf{v} = [x, y]^\top$ nào, chúng ta có được một vector $\mathbf{A}\mathbf{v} = [2x, -y]^\top$. Điều này có một cách giải thích trực quan: kéo dài vectơ rộng gấp đôi theo hướng $x$-, sau đó lật nó theo hướng $y$-. 

Tuy nhiên, có * một số* vectơ mà một cái gì đó vẫn không thay đổi. Cụ thể là $[1, 0]^\top$ được gửi đến $[2, 0]^\top$ và $[0, 1]^\top$ được gửi đến $[0, -1]^\top$. Các vectơ này vẫn nằm trong cùng một dòng, và sửa đổi duy nhất là ma trận kéo dài chúng bằng một hệ số lần lượt là $2$ và $-1$. Chúng tôi gọi các vectơ như vậy * eigenvectors* và yếu tố chúng được kéo dài bởi *eigenvalues*. 

Nói chung, nếu chúng ta có thể tìm thấy một số $\lambda$ và một vector $\mathbf{v}$ sao cho  

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}.
$$

Chúng tôi nói rằng $\mathbf{v}$ là một eigenvector cho $A$ và $\lambda$ là một eigenvalue. 

## Tìm kiếm Eigenvalues Hãy để chúng tôi tìm ra cách tìm chúng. Bằng cách trừ $\lambda \mathbf{v}$ từ cả hai phía, và sau đó bao thanh toán ra vectơ, ta thấy ở trên tương đương với: 

$$(\mathbf{A} - \lambda \mathbf{I})\mathbf{v} = 0.$$
:eqlabel:`eq_eigvalue_der`

Để :eqref:`eq_eigvalue_der` xảy ra, chúng ta thấy rằng $(\mathbf{A} - \lambda \mathbf{I})$ phải nén một số hướng xuống 0, do đó nó không thể đảo ngược, và do đó yếu tố quyết định bằng 0. Do đó, chúng ta có thể tìm thấy *eigenvalues* bằng cách tìm kiếm những gì $\lambda$ là $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$. Khi chúng ta tìm thấy eigenvalues, chúng ta có thể giải $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$ để tìm (các) *eigenvector (s) * được liên kết. 

### Một ví dụ Hãy để chúng tôi xem điều này với một ma trận thách thức hơn 

$$
\mathbf{A} = \begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}.
$$

Nếu chúng ta xem xét $\det(\mathbf{A}-\lambda \mathbf{I}) = 0$, ta thấy điều này tương đương với phương trình đa thức $0 = (2-\lambda)(3-\lambda)-2 = (4-\lambda)(1-\lambda)$. Do đó, hai eigenvalues là $4$ và $1$. Để tìm các vectơ liên quan, sau đó chúng ta cần phải giải quyết 

$$
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} = \begin{bmatrix}x \\ y\end{bmatrix}  \; \text{and} \;
\begin{bmatrix}
2 & 1\\
2 & 3 
\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix}  = \begin{bmatrix}4x \\ 4y\end{bmatrix} .
$$

Chúng ta có thể giải quyết điều này với các vectơ $[1, -1]^\top$ và $[1, 2]^\top$ tương ứng. 

Chúng ta có thể kiểm tra điều này trong mã bằng cách sử dụng thói quen `numpy.linalg.eig` tích hợp.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
import numpy as np

np.linalg.eig(np.array([[2, 1], [2, 3]]))
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
import torch

torch.eig(torch.tensor([[2, 1], [2, 3]], dtype=torch.float64),
          eigenvectors=True)
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
import tensorflow as tf

tf.linalg.eig(tf.constant([[2, 1], [2, 3]], dtype=tf.float64))
```

Lưu ý rằng `numpy` bình thường hóa các eigenvectors có chiều dài một, trong khi chúng ta lấy chúng ta có chiều dài tùy ý. Ngoài ra, sự lựa chọn của dấu hiệu là tùy ý. Tuy nhiên, các vectơ được tính toán song song với các vectơ mà chúng ta tìm thấy bằng tay với cùng một giá trị eigenvalues. 

## Phân hủy ma trận Hãy để chúng tôi tiếp tục ví dụ trước một bước nữa. Hãy để 

$$
\mathbf{W} = \begin{bmatrix}
1 & 1 \\
-1 & 2
\end{bmatrix},
$$

là ma trận nơi các cột là eigenvectors của ma trận $\mathbf{A}$. Hãy để 

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
1 & 0 \\
0 & 4
\end{bmatrix},
$$

là ma trận với các giá trị eigenvalues liên quan trên đường chéo. Sau đó, định nghĩa của eigenvalues và eigenvectors nói với chúng ta rằng 

$$
\mathbf{A}\mathbf{W} =\mathbf{W} \boldsymbol{\Sigma} .
$$

Ma trận $W$ có thể đảo ngược, vì vậy chúng ta có thể nhân cả hai bên với $W^{-1}$ ở bên phải, chúng ta thấy rằng chúng ta có thể viết 

$$\mathbf{A} = \mathbf{W} \boldsymbol{\Sigma} \mathbf{W}^{-1}.$$
:eqlabel:`eq_eig_decomp`

Trong phần tiếp theo, chúng ta sẽ thấy một số hậu quả tốt đẹp của điều này, nhưng bây giờ chúng ta chỉ cần biết rằng sự phân hủy như vậy sẽ tồn tại miễn là chúng ta có thể tìm thấy một bộ sưu tập đầy đủ các eigenvectors độc lập tuyến tính (để $W$ là nghịch). 

## Hoạt động trên Eigendecomposition Một điều hay về eigendecomposition :eqref:`eq_eig_decomp` là chúng ta có thể viết nhiều thao tác mà chúng ta thường gặp phải một cách sạch sẽ về sự phân hủy. Như một ví dụ đầu tiên, hãy xem xét: 

$$
\mathbf{A}^n = \overbrace{\mathbf{A}\cdots \mathbf{A}}^{\text{$n$ times}} = \overbrace{(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})\cdots(\mathbf{W}\boldsymbol{\Sigma} \mathbf{W}^{-1})}^{\text{$n$ times}} =  \mathbf{W}\overbrace{\boldsymbol{\Sigma}\cdots\boldsymbol{\Sigma}}^{\text{$n$ times}}\mathbf{W}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^n \mathbf{W}^{-1}.
$$

Điều này cho chúng ta biết rằng đối với bất kỳ sức mạnh tích cực nào của ma trận, sự phân hủy eigencomposition thu được bằng cách chỉ nâng eigenvalues lên cùng một sức mạnh. Điều tương tự cũng có thể được hiển thị cho các quyền hạn tiêu cực, vì vậy nếu chúng ta muốn đảo ngược một ma trận, chúng ta chỉ cần xem xét 

$$
\mathbf{A}^{-1} = \mathbf{W}\boldsymbol{\Sigma}^{-1} \mathbf{W}^{-1},
$$

or in other wordstừ ngữ, just invert đảo ngược each mỗi eigenvalue giá trị. Điều này sẽ hoạt động miễn là mỗi eigenvalue là không, vì vậy chúng ta thấy rằng invertible giống như không có eigenvalues bằng không.   

Thật vậy, công việc bổ sung có thể cho thấy nếu $\lambda_1, \ldots, \lambda_n$ là giá trị eigenvalues của ma trận, thì yếu tố quyết định của ma trận đó là 

$$
\det(\mathbf{A}) = \lambda_1 \cdots \lambda_n,
$$

hoặc sản phẩm của tất cả các giá trị eigenvalues. Điều này có ý nghĩa trực giác bởi vì bất cứ điều gì kéo dài $\mathbf{W}$ làm, $W^{-1}$ hoàn tác nó, vì vậy cuối cùng sự kéo dài duy nhất xảy ra là nhân với ma trận chéo $\boldsymbol{\Sigma}$, kéo dài khối lượng theo sản phẩm của các yếu tố đường chéo. 

Cuối cùng, nhớ lại rằng thứ hạng là số lượng tối đa các cột độc lập tuyến tính của ma trận của bạn. Bằng cách kiểm tra chặt chẽ eigendecomposition, chúng ta có thể thấy rằng thứ hạng giống như số eigenvalues không bằng không của $\mathbf{A}$. 

Các ví dụ có thể tiếp tục, nhưng hy vọng điểm rõ ràng: eigendecomposition có thể đơn giản hóa nhiều tính toán tuyến tính-đại số và là một hoạt động cơ bản dựa trên nhiều thuật toán số và phần lớn phân tích mà chúng ta làm trong đại số tuyến tính.  

## Eigendecompositions of Symmetric Matrận Không phải lúc nào cũng có thể tìm thấy đủ các eigenvectơ độc lập tuyến tính cho quá trình trên hoạt động. For instance ví dụ the matrix ma trận 

$$
\mathbf{A} = \begin{bmatrix}
1 & 1 \\
0 & 1
\end{bmatrix},
$$

has only a single Độc thân eigenvector, cụ thể là $(1, 0)^\top$. Để xử lý các ma trận như vậy, chúng tôi yêu cầu các kỹ thuật tiên tiến hơn chúng ta có thể bao gồm (chẳng hạn như Dạng bình thường Jordan, hoặc Phân hủy giá trị số ít). Chúng ta thường sẽ cần phải hạn chế sự chú ý của mình đối với những ma trận nơi chúng ta có thể đảm bảo sự tồn tại của một bộ đầy đủ các eigenvectors. 

Gia đình thường gặp nhất là * đối xứng matrices*, đó là những ma trận nơi $\mathbf{A} = \mathbf{A}^\top$. Trong trường hợp này, chúng ta có thể lấy $W$ là một ma trận * trực giao ma trận có cột có tất cả chiều dài một vectơ ở góc vuông với nhau, trong đó $\mathbf{W}^\top = \mathbf{W}^{-1}$—và tất cả các giá trị eigenvalue sẽ là thật. Do đó, trong trường hợp đặc biệt này, chúng ta có thể viết :eqref:`eq_eig_decomp` là 

$$
\mathbf{A} = \mathbf{W}\boldsymbol{\Sigma}\mathbf{W}^\top .
$$

## Định lý vòng tròn Gershgorin Eigenvalues thường khó lý luận với trực giác. Nếu trình bày một ma trận tùy ý, có rất ít điều có thể nói về những gì eigenvalues là mà không tính toán chúng. Tuy nhiên, có một định lý có thể làm cho nó dễ dàng gần đúng nếu các giá trị lớn nhất nằm trên đường chéo. 

Hãy để $\mathbf{A} = (a_{ij})$ là bất kỳ ma trận vuông ($n\times n$). Chúng tôi sẽ xác định $r_i = \sum_{j \neq i} |a_{ij}|$. Hãy để $\mathcal{D}_i$ đại diện cho đĩa trong mặt phẳng phức tạp với trung tâm $a_{ii}$ bán kính $r_i$. Sau đó, mỗi eigenvalue của $\mathbf{A}$ được chứa trong một trong $\mathcal{D}_i$. 

Điều này có thể là một chút để giải nén, vì vậy chúng ta hãy xem xét một ví dụ. Hãy xem xét ma trận: 

$$
\mathbf{A} = \begin{bmatrix}
1.0 & 0.1 & 0.1 & 0.1 \\
0.1 & 3.0 & 0.2 & 0.3 \\
0.1 & 0.2 & 5.0 & 0.5 \\
0.1 & 0.3 & 0.5 & 9.0
\end{bmatrix}.
$$

Chúng ta có $r_1 = 0.3$, $r_2 = 0.6$, $r_3 = 0.8$ và $r_4 = 0.9$. Ma trận là đối xứng, vì vậy tất cả các giá trị eigenvalues là có thật. Điều này có nghĩa là tất cả các giá trị eigenvalues của chúng tôi sẽ nằm trong một trong các phạm vi  

$$[a_{11}-r_1, a_{11}+r_1] = [0.7, 1.3], $$

$$[a_{22}-r_2, a_{22}+r_2] = [2.4, 3.6], $$

$$[a_{33}-r_3, a_{33}+r_3] = [4.2, 5.8], $$

$$[a_{44}-r_4, a_{44}+r_4] = [8.1, 9.9]. $$

Thực hiện tính toán số cho thấy các giá trị eigenvalues là khoảng $0.99$, $2.97$, $4.95$, $9.08$, tất cả đều thoải mái bên trong các phạm vi được cung cấp.

```{.python .input}
A = np.array([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = np.linalg.eig(A)
v
```

```{.python .input}
#@tab pytorch
A = torch.tensor([[1.0, 0.1, 0.1, 0.1],
              [0.1, 3.0, 0.2, 0.3],
              [0.1, 0.2, 5.0, 0.5],
              [0.1, 0.3, 0.5, 9.0]])

v, _ = torch.eig(A)
v
```

```{.python .input}
#@tab tensorflow
A = tf.constant([[1.0, 0.1, 0.1, 0.1],
                [0.1, 3.0, 0.2, 0.3],
                [0.1, 0.2, 5.0, 0.5],
                [0.1, 0.3, 0.5, 9.0]])

v, _ = tf.linalg.eigh(A)
v
```

Bằng cách này, eigenvalues có thể được xấp xỉ và các xấp xỉ sẽ khá chính xác trong trường hợp đường chéo lớn hơn đáng kể so với tất cả các yếu tố khác.   

Đó là một điều nhỏ, nhưng với một chủ đề phức tạp và tinh tế như eigendecomposition, thật tốt để có được bất kỳ sự nắm bắt trực quan nào chúng ta có thể. 

## Một ứng dụng hữu ích: Sự phát triển của bản đồ lặp

Bây giờ chúng ta hiểu nguyên tắc eigenvectors là gì, chúng ta hãy xem chúng có thể được sử dụng như thế nào để cung cấp một sự hiểu biết sâu sắc về một vấn đề trung tâm đến hành vi mạng thần kinh: khởi tạo trọng lượng thích hợp.  

### Eigenvectors như hành vi dài hạn

Cuộc điều tra toán học đầy đủ về việc khởi tạo các mạng thần kinh sâu nằm ngoài phạm vi của văn bản, nhưng chúng ta có thể thấy một phiên bản đồ chơi ở đây để hiểu cách thức eigenvalues có thể giúp chúng ta thấy các mô hình này hoạt động như thế nào. Như chúng ta đã biết, các mạng thần kinh hoạt động bằng cách xen kẽ các lớp biến đổi tuyến tính với các hoạt động phi tuyến tính. Để đơn giản ở đây, chúng ta sẽ giả định rằng không có phi tuyến tính và chuyển đổi là một hoạt động ma trận lặp lại duy nhất $A$, để đầu ra của mô hình của chúng tôi là 

$$
\mathbf{v}_{out} = \mathbf{A}\cdot \mathbf{A}\cdots \mathbf{A} \mathbf{v}_{in} = \mathbf{A}^N \mathbf{v}_{in}.
$$

Khi các mô hình này được khởi tạo, $A$ được coi là một ma trận ngẫu nhiên với các mục Gaussian, vì vậy chúng ta hãy tạo một trong những mô hình đó. Để được cụ thể, chúng ta bắt đầu với một trung bình 0, phương sai một Gaussian phân phối $5 \times 5$ ma trận.

```{.python .input}
np.random.seed(8675309)

k = 5
A = np.random.randn(k, k)
A
```

```{.python .input}
#@tab pytorch
torch.manual_seed(42)

k = 5
A = torch.randn(k, k, dtype=torch.float64)
A
```

```{.python .input}
#@tab tensorflow
k = 5
A = tf.random.normal((k, k), dtype=tf.float64)
A
```

### Hành vi trên dữ liệu ngẫu nhiên Để đơn giản trong mô hình đồ chơi của chúng ta, chúng ta sẽ giả định rằng vector dữ liệu chúng ta cung cấp trong $\mathbf{v}_{in}$ là một vector Gaussian năm chiều ngẫu nhiên. Chúng ta hãy suy nghĩ về những gì chúng ta muốn có xảy ra. Đối với bối cảnh, hãy nghĩ về một vấn đề ML chung, nơi chúng ta đang cố gắng biến dữ liệu đầu vào, giống như một hình ảnh, thành một dự đoán, giống như xác suất hình ảnh là một hình ảnh của một con mèo. Nếu ứng dụng lặp đi lặp lại của $\mathbf{A}$ kéo dài một vectơ ngẫu nhiên ra rất dài, thì những thay đổi nhỏ trong đầu vào sẽ được khuếch đại thành những thay đổi lớn trong đầu ra—những sửa đổi nhỏ của hình ảnh đầu vào sẽ dẫn đến những dự đoán rất khác nhau. Điều này dường như không đúng! 

Mặt trái, nếu $\mathbf{A}$ thu nhỏ các vectơ ngẫu nhiên để ngắn hơn, thì sau khi chạy qua nhiều lớp, vectơ về cơ bản sẽ co lại thành không có gì, và đầu ra sẽ không phụ thuộc vào đầu vào. Điều này cũng rõ ràng cũng không đúng! 

Chúng ta cần phải đi bộ đường hẹp giữa tăng trưởng và phân rã để đảm bảo rằng đầu ra của chúng tôi thay đổi tùy thuộc vào đầu vào của chúng tôi, nhưng không nhiều! 

Chúng ta hãy xem những gì xảy ra khi chúng ta nhiều lần nhân ma trận của chúng ta $\mathbf{A}$ chống lại một vector đầu vào ngẫu nhiên, và theo dõi định mức.

```{.python .input}
# Calculate the sequence of norms after repeatedly applying `A`
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Calculate the sequence of norms after repeatedly applying `A`
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Calculate the sequence of norms after repeatedly applying `A`
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Định mức đang phát triển không kiểm soát được! Thật vậy nếu chúng ta lấy danh sách các thương, chúng ta sẽ thấy một mô hình.

```{.python .input}
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Compute the scaling factor of the norms
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i - 1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

Nếu chúng ta nhìn vào phần cuối cùng của tính toán trên, chúng ta thấy rằng vectơ ngẫu nhiên được kéo dài bởi một hệ số `1.974459321485[...]`, trong đó phần ở cuối thay đổi một chút, nhưng yếu tố kéo dài ổn định.   

### Liên quan Trở lại Eigenvectors

Chúng ta đã thấy rằng eigenvectors và eigenvalues tương ứng với số lượng một cái gì đó được kéo dài, nhưng đó là cho vectơ cụ thể, và các trải dài cụ thể. Chúng ta hãy xem chúng là gì cho $\mathbf{A}$. Một chút cảnh báo ở đây: hóa ra để xem tất cả, chúng ta sẽ cần phải đi đến các số phức. Bạn có thể nghĩ về những điều này như kéo dài và quay. Bằng cách lấy định mức của số phức (căn bậc hai của tổng các ô vuông của các phần thực và tưởng tượng), chúng ta có thể đo được yếu tố kéo dài đó. Hãy để chúng tôi cũng sắp xếp chúng.

```{.python .input}
# Compute the eigenvalues
eigs = np.linalg.eigvals(A).tolist()
norm_eigs = [np.absolute(x) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab pytorch
# Compute the eigenvalues
eigs = torch.eig(A)[0][:,0].tolist()
norm_eigs = [torch.abs(torch.tensor(x)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

```{.python .input}
#@tab tensorflow
# Compute the eigenvalues
eigs = tf.linalg.eigh(A)[0].numpy().tolist()
norm_eigs = [tf.abs(tf.constant(x, dtype=tf.float64)) for x in eigs]
norm_eigs.sort()
print(f'norms of eigenvalues: {norm_eigs}')
```

### Một quan sát

Chúng ta thấy một điều gì đó một chút bất ngờ xảy ra ở đây: số đó chúng tôi đã xác định trước đây cho kéo dài dài hạn ma trận của chúng tôi $\mathbf{A}$ áp dụng cho một vector ngẫu nhiên là * chính xác* (chính xác đến mười ba chữ số thập phân!) the largest lớn nhất eigenvalue giá trị of $\mathbf{A}$. Đây rõ ràng không phải là một sự trùng hợp ngẫu nhiên! 

Nhưng, nếu bây giờ chúng ta nghĩ về những gì đang xảy ra về mặt hình học, điều này bắt đầu có ý nghĩa. Hãy xem xét một vector ngẫu nhiên. Vector ngẫu nhiên này chỉ một chút theo mọi hướng, vì vậy đặc biệt, nó chỉ ít nhất một chút theo cùng hướng với eigenvector của $\mathbf{A}$ liên quan đến eigenvalue lớn nhất. Điều này quan trọng đến mức nó được gọi là *nguyên tắc eigenvalue* và *nguyên tắc eigenvector*. Sau khi áp dụng $\mathbf{A}$, vector ngẫu nhiên của chúng ta được kéo dài theo mọi hướng có thể, như được liên kết với mọi eigenvector có thể, nhưng nó được kéo dài hầu hết theo hướng liên quan đến nguyên tắc này eigenvector. Điều này có nghĩa là sau khi áp dụng trong $A$, vector ngẫu nhiên của chúng ta dài hơn, và các điểm theo hướng gần hơn để được liên kết với nguyên tắc eigenvector. Sau khi áp dụng ma trận nhiều lần, sự liên kết với nguyên tắc eigenvector trở nên gần gũi hơn và gần hơn cho đến khi, cho tất cả các mục đích thực tế, vector ngẫu nhiên của chúng ta đã được chuyển thành nguyên tắc eigenvector! Thật vậy, thuật toán này là cơ sở cho cái được gọi là lần lặp điện* để tìm giá trị eigenvalue và eigenvector lớn nhất của ma trận. Để biết chi tiết, hãy xem, ví dụ, :cite:`Van-Loan.Golub.1983`. 

### Sửa chữa bình thường hóa

Bây giờ, từ các cuộc thảo luận trên, chúng tôi kết luận rằng chúng tôi không muốn một vector ngẫu nhiên được kéo dài hoặc squished ở tất cả, chúng tôi muốn vectơ ngẫu nhiên ở lại với cùng một kích thước trong toàn bộ quá trình. Để làm như vậy, bây giờ chúng ta giải phóng ma trận của chúng tôi bằng nguyên tắc eigenvalue này để eigenvalue lớn nhất thay vào đó bây giờ chỉ là một. Hãy để chúng tôi xem những gì xảy ra trong trường hợp này.

```{.python .input}
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = np.random.randn(k, 1)

norm_list = [np.linalg.norm(v_in)]
for i in range(1, 100):
    v_in = A.dot(v_in)
    norm_list.append(np.linalg.norm(v_in))

d2l.plot(np.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab pytorch
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = torch.randn(k, 1, dtype=torch.float64)

norm_list = [torch.norm(v_in).item()]
for i in range(1, 100):
    v_in = A @ v_in
    norm_list.append(torch.norm(v_in).item())

d2l.plot(torch.arange(0, 100), norm_list, 'Iteration', 'Value')
```

```{.python .input}
#@tab tensorflow
# Rescale the matrix `A`
A /= norm_eigs[-1]

# Do the same experiment again
v_in = tf.random.normal((k, 1), dtype=tf.float64)

norm_list = [tf.norm(v_in).numpy()]
for i in range(1, 100):
    v_in = tf.matmul(A, v_in)
    norm_list.append(tf.norm(v_in).numpy())

d2l.plot(tf.range(0, 100), norm_list, 'Iteration', 'Value')
```

Chúng ta cũng có thể vẽ tỷ lệ giữa các định mức liên tiếp như trước và thấy rằng thực sự nó ổn định.

```{.python .input}
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(np.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab pytorch
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(torch.arange(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

```{.python .input}
#@tab tensorflow
# Also plot the ratio
norm_ratio_list = []
for i in range(1, 100):
    norm_ratio_list.append(norm_list[i]/norm_list[i-1])

d2l.plot(tf.range(1, 100), norm_ratio_list, 'Iteration', 'Ratio')
```

## Kết luận

Bây giờ chúng ta thấy chính xác những gì chúng tôi hy vọng! Sau khi bình thường hóa các ma trận theo nguyên tắc eigenvalue, chúng ta thấy rằng dữ liệu ngẫu nhiên không phát nổ như trước, mà cuối cùng cân bằng với một giá trị cụ thể. Nó sẽ là tốt đẹp để có thể làm những điều này từ các nguyên tắc đầu tiên, và nó chỉ ra rằng nếu chúng ta nhìn sâu vào toán học của nó, chúng ta có thể thấy rằng eigenvalue lớn nhất của một ma trận ngẫu nhiên lớn với trung bình zero độc lập, phương sai một mục Gaussian là trung bình khoảng $\sqrt{n}$, hoặc trong trường hợp của chúng tôi $\sqrt{5} \approx 2.2$, do một thực tế hấp dẫn được gọi là luật thông tư*:cite:`Ginibre.1965`. Mối quan hệ giữa các eigenvalues (và một đối tượng liên quan gọi là giá trị số ít) của ma trận ngẫu nhiên đã được chứng minh là có các kết nối sâu sắc để khởi tạo thích hợp các mạng thần kinh như đã được thảo luận trong :cite:`Pennington.Schoenholz.Ganguli.2017` và các tác phẩm tiếp theo. 

## Tóm tắt * Eigenvectors là các vectơ được kéo dài bởi một ma trận mà không thay đổi hướng. * Eigenvalues là số tiền mà các eigenvectors được kéo dài bởi ứng dụng của ma trận. * Sự phân hủy của ma trận có thể cho phép nhiều phép toán được giảm xuống các hoạt động trên eigenvalues. Định lý vòng tròn Gershgorin có thể cung cấp các giá trị gần đúng cho eigenvalues của ma trận * hành vi của các công suất ma trận lặp lại phụ thuộc chủ yếu vào kích thước của eigenvalue lớn nhất. Sự hiểu biết này có nhiều ứng dụng trong lý thuyết khởi tạo mạng thần kinh. 

## Exercises
1. What are the eigenvalues and eigenvectors of
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
1 & 2
\end{bmatrix}?
$$
1.  What are the eigenvalues and eigenvectors of the following matrix, and what is strange about this example compared to the previous one?
$$
\mathbf{A} = \begin{bmatrix}
2 & 1 \\
0 & 2
\end{bmatrix}.
$$
1. Without computing the eigenvalues, is it possible that the smallest eigenvalue of the following matrix is less that $0.5$? *Note*: this problem can be done in your head.
$$
\mathbf{A} = \begin{bmatrix}
3.0 & 0.1 & 0.3 & 1.0 \\
0.1 & 1.0 & 0.1 & 0.2 \\
0.3 & 0.1 & 5.0 & 0.0 \\
1.0 & 0.2 & 0.0 & 1.8
\end{bmatrix}.
$$

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/411)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1086)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1087)
:end_tab:

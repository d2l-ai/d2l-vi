<!-- =================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Geometry and Linear Algebraic Operations
-->

# Các phép toán Hình Học và Đại Số Tuyến Tính
:label:`sec_geometry-linear-algebric-ops`

<!--
In :numref:`sec_linear-algebra`, we encountered the basics of linear algebra
and saw how it could be used to express common operations for transforming our data.
Linear algebra is one of the key mathematical pillars
underlying much of the work that we do deep learning
and in machine learning more broadly.
While :numref:`sec_linear-algebra` contained enough machinery
to communicate the mechanics of modern deep learning models,
there is a lot more to the subject.
In this section, we will go deeper,
highlighting some geometric interpretations of linear algebra operations,
and introducing a few fundamental concepts, including of eigenvalues and eigenvectors.
-->

Trong :numref:`sec_linear-algebra`, chúng ta đã đề cập tới những kiến thức cơ bản trong đại số tuyến
tính và cách nó được dùng để thể hiện các phép biến đổi dữ liệu cơ bản.
Đại số tuyến tính là một trong những trụ cột toán học chính hỗ trợ học sâu
và rộng hơn là học máy. Trong khi :numref:`sec_linear-algebra` chứa đựng đầy
đủ kiến thức cần thiết cho các mô hình học sâu hiện đại, vẫn còn rất nhiều điều
cần thảo luận trong lĩnh vực này. Trong mục này, chúng ta sẽ đi sâu hơn, nhấn
mạnh một số diễn giải hình học của các phép toán đại số tuyến tính, và giới
thiệu một vài khái niệm cơ bản, bao gồm trị riêng và vector riêng.

<!--
## Geometry of Vectors
-->

## Ý nghĩa hình học của Vector

<!--
First, we need to discuss the two common geometric interpretations of vectors,
as either points or directions in space.
Fundamentally, a vector is a list of numbers such as the Python list below.
-->

Trước hết, chúng ta cần thảo luận hai diễn giải hình học phổ biến của vector:
điểm hoặc hướng trong không gian. Về cơ bản, một vector là một danh sách các
số giống như danh sách trong Python dưới đây:

```{.python .input}
v = [1, 7, 0, 1]
```

<!--
Mathematicians most often write this as either a *column* or *row* vector, which is to say either as
-->

Các nhà toán học thường viết chúng dưới dạng một vector *cột* hoặc *hàng*, tức:

$$
\mathbf{x} = \begin{bmatrix}1\\7\\0\\1\end{bmatrix},
$$

<!--
or
-->

hoặc

$$
\mathbf{x}^\top = \begin{bmatrix}1 & 7 & 0 & 1\end{bmatrix}.
$$

<!--
These often have different interpretations,
where data points are column vectors
and weights used to form weighted sums are row vectors.
However, it can be beneficial to be flexible.
Matrices are useful data structures: they allow us to organize data that have different modalities of variation. For example, rows in our matrix might correspond to different houses (data points), while columns might correspond to different attributes. This should sound familiar if you have ever used spreadsheet software or have read :numref:`sec_pandas`. Thus, although the default orientation of a single vector is a column vector, in a matrix that represents a tabular dataset, it is more conventional to treat each data point as a row vector in the matrix. And, as we will see in later chapters, this convention will enable common deep learning practices. For example, along the outermost axis of an `ndarray`, we can access or enumerate minibatches of data points, or just data points if no minibatch exists.
-->

Những biểu diễn này thường có những cách diễn giải khác nhau. Các điểm dữ liệu
được biểu diễn bằng các vector cột và các trọng số dùng trong các tổng có
trọng số được biểu diễn bằng các vector hàng. Tuy nhiên, việc linh động sử
dụng các cách biểu diễn này mang lại nhiều lợi ích. Ma trận là những
cấu trúc dữ liệu hữu ích: chúng cho phép chúng ta tổ chức dữ liệu với nhiều
biến thể khác nhau. Ví dụ, các hàng của ma trận có thể tương ứng với các nhà
(điểm dữ liệu) khác nhau, trong khi các cột có thể tương ứng với các thuộc tính
khác nhau. Việc này nghe quen thuộc nếu bạn từng sử dụng các phần mềm dạng bảng
(spreadsheet) hoặc đã từng đọc :numref:`sec_pandas`. Bởi vậy, mặc dù chiều mặc
định của một vector là một vector cột, trong một ma trận biểu diễn một tập dữ
liệu dạng bảng, sẽ thuận tiện hơn khi coi mỗi điểm dữ liệu là một vector hàng
trong ma trận đó. Và như chúng ta sẽ thấy trong các chương sau, cách biểu diễn
này phù hợp với cách triển khai các mô hình học sâu.
Lấy ví dụ, dọc theo trục ngoài cùng của một `ndarray`, ta có thể truy cập hoặc đếm số
minibatch chứa điểm dữ liệu, hoặc chỉ đơn giản là các điểm dữ liệu nếu minibatch không tồn tại.

<!-- =================== Kết thúc dịch Phần 1 ==================== -->

<!-- =================== Bắt đầu dịch Phần 2 ==================== -->

<!--
Given a vector, the first interpretation
that we should give it is as a point in space.
In two or three dimensions, we can visualize these points
by using the components of the vectors to define
the location of the points in space compared
to a fixed reference called the *origin*.  This can be seen in :numref:`fig_grid`.
-->

Cách thứ nhất để giải thích một vector là coi nó như một điểm trong không gian.
Trong không gian hai hoặc ba chiều, chúng ta có thể biểu diễn các điểm này bằng
việc sử dụng các thành phần của vector để định nghĩa vị trí của điểm trong
không gian so với một điểm tham chiều được gọi là *gốc tọa độ*. Xem :numref:`fig_grid`.

<!--
![An illustration of visualizing vectors as points in the plane.  The first component of the vector gives the $x$-coordinate, the second component gives the $y$-coordinate.  Higher dimensions are analogous, although much harder to visualize.](../img/GridPoints.svg)
-->

![Mô tả việc biểu diễn vector như các điểm trong mặt phẳng. Thành phần thứ nhất của vector là tọa độ $x$, thành phần thứ hai là tọa độ $y$. Tương tự với số chiều cao hơn, mặc dù khó hình dung hơn](../img/GridPoints.svg)
:label:`fig_grid`

<!--
This geometric point of view allows us to consider the problem on a more abstract level.
No longer faced with some insurmountable seeming problem
like classifying pictures as either cats or dogs,
we can start considering tasks abstractly
as collections of points in space and picturing the task
as discovering how to separate two distinct clusters of points.
-->

Góc nhìn hình học này cho phép chúng ta xem xét bài toán ở một mức trừu tượng hơn.
Không giống như khi đối mặt với các bài toán khó hình dung như phân loại ảnh chó mèo, chúng ta có thể bắt đầu xem xét các bài toán này một cách trừu tượng hơn như là
một tập hợp của các điểm trong không gian. Việc phân loại ảnh chó mèo có thể coi
như việc tìm ra cách phân biệt hai nhóm điểm riêng biệt trong không gian.

<!-- Nhóm tác giả không phải là người bản xứ nói tiếng Anh. Thực tế, bản tiếng
Anh này được dịch từ bản tiếng Trung rất nổi tiếng ở Trung Quốc. Khi dịch, tôi
nghĩ chúng ta có thể sửa đổi câu văn đi một chút cho phù hợp với tiếng Việt.
Đoạn này sẽ không hiển thị trên web vì nó đã được comment.
-->

<!--
In parallel, there is a second point of view
that people often take of vectors: as directions in space.
Not only can we think of the vector $\mathbf{v} = [2,3]^\top$
as the location $2$ units to the right and $3$ units up from the origin,
we can also think of it as the direction itself
to take $2$ steps to the right and $3$ steps up.
In this way, we consider all the vectors in figure :numref:`fig_arrow` the same.
-->

Cách thứ hai để giải thích một vector là coi nó như một hướng trong không gian. Chúng ta không những
có thể coi vector $\mathbf{v} = [2,3]^\top$ là một điểm nằm bên phải $2$ đơn vị
và bên trên $3$ đơn vị so với gốc toạ độ, chúng ta cũng có thể coi nó thể hiện
một hướng -- hướng $2$ bước về bên phải và $3$ bước lên trên. Theo cách này,
ta coi tất cả các vector trong hình :numref:`fig_arrow` là như nhau.

<!--
![Any vector can be visualized as an arrow in the plane.  In this case, every vector drawn is a representation of the vector $(2,3)$.](../img/ParVec.svg)
-->

![Bất kỳ vector nào cũng có thể biểu diễn bằng một mũi tên trong mặt phẳng. Trong trường hợp này, mọi vector trong hình đều biểu diễn vector $(2,3)$.](../img/ParVec.svg)
:label:`fig_arrow`

<!--
One of the benefits of this shift is that
we can make visual sense of the act of vector addition.
In particular, we follow the directions given by one vector,
and then follow the directions given by the other, as is seen in :numref:`fig_add-vec`.
-->

Một trong những lợi ý của việc chuyển cách hiểu này là phép cộng vector có thể được
hiểu theo nghĩa hình học. Cụ thể, chúng ta đi theo một hướng được cho bởi một vector,
sau đó đi theo một hướng cho bởi một vector khác, như được cho trong :numref:`fig_add-vec`.

<!--
![We can visualize vector addition by first following one vector, and then another.](../img/VecAdd.svg)
-->

![Phép cộng vector có thể biểu diễn bằng cách đầu tiên đi theo một vector, sau đó đi theo vector kia.](../img/VecAdd.svg)
:label:`fig_add-vec`

<!-- =================== Kết thúc dịch Phần 2 ==================== -->

<!-- =================== Bắt đầu dịch Phần 3 ==================== -->

<!--
Vector subtraction has a similar interpretation.
By considering the identity that $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$,
we see that the vector $\mathbf{u}-\mathbf{v}$ is the direction
that takes us from the point $\mathbf{u}$ to the point $\mathbf{v}$.
-->

Hiệu của hai vector có một cách diễn giải tương tự.
Bằng cách biểu diễn $\mathbf{u} = \mathbf{v} + (\mathbf{u}-\mathbf{v})$,
ta thấy rằng vector $\mathbf{u}-\mathbf{v}$ là hướng mang điểm $\mathbf{u}$ tới
điểm $\mathbf{v}$.


<!--
## Dot Products and Angles
-->

## Tích vô hướng và Góc

<!--
As we saw in :numref:`sec_linear-algebra`,
if we take two column vectors say $\mathbf{u}$ and $\mathbf{v}$,
we can form their dot product by computing:
-->

Như đã thấy trong :numref:`sec_linear-algebra`, tích vô hướng của hai vector cột
$\mathbf{u}$ và $\mathbf{v}$ có thể được tính bởi:

$$\mathbf{u}^\top\mathbf{v} = \sum_i u_i\cdot v_i.$$
:eqlabel:`eq_dot_def`

<!--
Because :eqref:`eq_dot_def` is symmetric, we will mirror the notation
of classical multiplication and write
-->

Vì biểu thức :eqref:`eq_dot_def` đối xứng, chúng ta có thể viết:

$$
\mathbf{u}\cdot\mathbf{v} = \mathbf{u}^\top\mathbf{v} = \mathbf{v}^\top\mathbf{u},
$$

<!--
to highlight the fact that exchanging the order of the vectors will yield the same answer.
-->

để nhấn mạnh rằng phép đổi chỗ hai vector sẽ cho kết quả như nhau.

<!--
The dot product :eqref:`eq_dot_def` also admits a geometric interpretation: it is closely related to the angle between two vectors.  Consider the angle shown in :numref:`fig_angle`.
-->

Tích vô hướng :eqref:`eq_dot_def` cũng có diễn giải hình học: nó liên quan
mật thiết tới góc giữa hai vector. Xem góc hiển thị trong :numref:`fig_angle`.

<!--
![Between any two vectors in the plane there is a well defined angle $\theta$.  We will see this angle is intimately tied to the dot product.](../img/VecAngle.svg)
-->

![Có một định nghĩa về góc ($\theta$) giữa hai vector bất kỳ trong không gian. Ta sẽ thấy rằng góc này có liên hệ chặt chẽ tới tích vô hướng.](../img/VecAngle.svg)
:label:`fig_angle`

<!--
To start, let's consider two specific vectors:
-->

Xét hai vector:

$$
\mathbf{v} = (r,0) \; \text{and} \; \mathbf{w} = (s\cos(\theta), s \sin(\theta)).
$$

<!--
The vector $\mathbf{v}$ is length $r$ and runs parallel to the $x$-axis,
and the vector $\mathbf{w}$ is of length $s$ and at angle $\theta$ with the $x$-axis.
If we compute the dot product of these two vectors, we see that
-->

Vector $\mathbf{v}$ có độ dài $r$ và song song với trục $x$, vector $\mathbf{w}$
có độ dài $s$ và tạo một góc $\theta$ với trục $x$. Nếu tính tích vô hướng
của hai vector này, ta sẽ thấy rằng

<!-- =================== Kết thúc dịch Phần 3 ==================== -->

<!-- =================== Bắt đầu dịch Phần 4 ==================== -->

$$
\mathbf{v}\cdot\mathbf{w} = rs\cos(\theta) = \|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta).
$$

<!--
With some simple algebraic manipulation, we can rearrange terms to obtain
-->

Với một vài biến đổi đơn giản, chúng ta có thể sắp xếp lại các thành phần để được

$$
\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).
$$

<!--
In short, for these two specific vectors,
the dot product combined with the norms tell us the angle between the two vectors. This same fact is true in general. We will not derive the expression here, however,
if we consider writing $\|\mathbf{v} - \mathbf{w}\|^2$ in two ways:
one with the dot product, and the other geometrically using the law of cosines,
we can obtain the full relationship.
Indeed, for any two vectors $\mathbf{v}$ and $\mathbf{w}$,
the angle between the two vectors is
-->

Một cách ngắn gọn, với hai vector cụ thể này,
tích vô hướng kết hợp với chuẩn thể hiện góc giữa hai vector. Việc này cũng đúng trong trường hợp tổng quát.
Ta sẽ không viết biểu diễn ở đây, tuy nhiên, nếu viết $\|\mathbf{v} - \mathbf{w}\|^2$
bằng hai cách: cách thứ nhất với tích vô hướng, và cách thứ hai sử dụng công thức tính cos,
ta có thể thấy được quan hệ giữa chúng.
Thật vậy, với hai vector $\mathbf{v}$ và $\mathbf{w}$ bất kỳ, góc giữa chúng là

$$\theta = \arccos\left(\frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}\right).$$
:eqlabel:`eq_angle_forumla`

<!--
This is a nice result since nothing in the computation references two-dimensions.
Indeed, we can use this in three or three million dimensions without issue.
-->

Kết quả này tổng quát cho không gian nhiều chiều vì nó không sử dụng điều gì đặc biệt trong không gian hai chiều.

<!--
As a simple example, let's see how to compute the angle between a pair of vectors:
-->

Xét ví dụ đơn giản tính góc giữa cặp vector:

```{.python .input}
%matplotlib inline
import d2l
from IPython import display
from mxnet import gluon, np, npx
npx.set_np()

def angle(v, w):
    return np.arccos(v.dot(w) / (np.linalg.norm(v) * np.linalg.norm(w)))

angle(np.array([0, 1, 2]), np.array([2, 3, 4]))
```

<!--
We will not use it right now, but it is useful to know
that we will refer to vectors for which the angle is $\pi/2$
(or equivalently $90^{\circ}$) as being *orthogonal*.
By examining the equation above, we see that this happens when $\theta = \pi/2$,
which is the same thing as $\cos(\theta) = 0$.
The only way this can happen is if the dot product itself is zero,
and two vectors are orthogonal if and only if $\mathbf{v}\cdot\mathbf{w} = 0$.
This will prove to be a helpful formula when understanding objects geometrically.
-->

Chúng ta sẽ không sử dụng đoạn mã này bây giờ, nhưng sẽ hữu ích để biết rằng nếu
góc giữa hai vector là $\pi/2$
(hay $90^{\circ}$) thì hai vector đó được gọi là *trực giao*. Xem xét kỹ biểu
thức trên, ta thấy rằng việc này xảy ra khi $\theta = \pi/2$,
tức $\cos(\theta) = 0$. Điều này chứng tỏ tích vô hướng phải bằng không, và hai
vector là trực giao nếu và chỉ nếu $\mathbf{v}\cdot\mathbf{w} = 0$. Đẳng thức này
sẽ hữu ích khi xem xét các đối tượng dưới con mắt hình học.

<!--
It is reasonable to ask: why is computing the angle useful?
The answer comes in the kind of invariance we expect data to have.
Consider an image, and a duplicate image,
where every pixel value is the same but $10\%$ the brightness.
The values of the individual pixels are in general far from the original values.
Thus, if one computed the distance between the original image and the darker one,
the distance can be large.
However, for most ML applications, the *content* is the same---it is still
an image of a cat as far as a cat/dog classifier is concerned.
However, if we consider the angle, it is not hard to see
that for any vector $\mathbf{v}$, the angle
between $\mathbf{v}$ and $0.1\cdot\mathbf{v}$ is zero.
This corresponds to the fact that scaling vectors
keeps the same direction and just changes the length.
The angle considers the darker image identical.
-->

Ta sẽ tự hỏi tại sao tính góc lại hữu ích?
Câu trả lời nằm ở tính bất biến ta mong đợi từ dữ liệu. Xét một bức ảnh,
và một bức ảnh thứ hai giống hệt nhưng với các điểm ảnh với độ sáng chỉ bằng $10\%$
ảnh ban đầu. Giá trị của từng điểm ảnh trong ảnh thứ hai nhìn chung khác xa
so với ảnh ban đầu. Bởi vậy, nếu tính khoảng cách giữa ảnh ban đầu và ảnh tối hơn,
khoảng cách có thể rất lớn. Tuy nhiên, trong hầu hết các ứng dụng ML, *nội dung*
của hai bức ảnh là như nhau -- nó vẫn là một bức ảnh của một con mèo đối với
một bộ phân loại chó mèo. Tuy nhiên, nếu xem xét góc giữa hai ảnh, không khó
để thấy rằng với bất kỳ vector $\mathbf{v}$, góc giữa $\mathbf{v}$ và $0.1\cdot\mathbf{v}$
bằng không. Việc này tương ứng với việc nhân vector với một số (dương) giữ
nguyên hướng và chỉ thay đổi độ dài của vector đó. Khi xét tới góc, hai bức
ảnh được coi là như nhau.

<!--
Examples like this are everywhere.
In text, we might want the topic being discussed
to not change if we write twice as long of document that says the same thing.
For some encoding (such as counting the number of occurrences of words in some vocabulary), this corresponds to a doubling of the vector encoding the document,
so again we can use the angle.
-->

Ví dụ tương tự có thể tìm thấy bất cứ đâu. Trong văn bản, chúng ta có thể
muốn chủ đề được thảo luận không thay đổi nếu chúng ta viết văn bản dài gấp
hai nhưng nói về cùng một thứ. Trong một số cách mã hóa (như đếm số lượng xuất hiện
của một từ trong từ điển), việc này tương đương với nhân đôi vector mã hóa
của văn bản, bởi vậy chúng ta lại có thể sử dụng góc.

<!-- =================== Kết thúc dịch Phần 4 ==================== -->

<!-- =================== Bắt đầu dịch Phần 5 ==================== -->

<!--
### Cosine Similarity
-->

### Độ tương tự cosin

<!--
In ML contexts where the angle is employed
to measure the closeness of two vectors,
practitioners adopt the term *cosine similarity*
to refer to the portion
-->

Trong văn cảnh học máy với góc được dùng để chỉ khoảng cách giữa hai vector,
người làm ML sử dụng thuật ngữ *độ tương tự cosin* để chỉ đại lượng

$$
\cos(\theta) = \frac{\mathbf{v}\cdot\mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|}.
$$

<!--
The cosine takes a maximum value of $1$
when the two vectors point in the same direction,
a minimum value of $-1$ when they point in opposite directions,
and a value of $0$ when the two vectors are orthogonal.
Note that if the components of high-dimensional vectors
are sampled randomly with mean $0$,
their cosine will nearly always be close to $0$.
-->

Hàm cosin lấy giá trị lớn nhất bằng $1$ khi hai vector chỉ cùng một hướng, giá
trị nhỏ nhất bằng $-1$ khi chúng cùng phương khác hướng, và $0$ khi hai vector
trực giao. Chú ý rằng nếu các thành phần của hai vector nhiều chiều được lấy
mẫu ngẫu nhiên với kỳ vọng $0$, cosin giữa chúng sẽ luôn gần với $0$.

<!--
## Hyperplanes
-->

## Siêu phẳng

<!--
In addition to working with vectors, another key object
that you must understand to go far in linear algebra
is the *hyperplane*, a generalization to higher dimensions
of a line (two dimensions) or of a plane (three dimensions).
In an $n$-dimensional vector space, a hyperplane has $d-1$ dimensions
and divides the space into two half-spaces.
-->

Ngoài việc làm việc với vector, một khái niệm quan trọng khác bạn phải nắm vững
khi đi sâu vào đại số tuyến tính là *siêu phẳng*, một khái niệm tổng quát của
đường thẳng (trong không gian hai chiều) hoặc một mặt phẳng (trong không gian
ba chiều). Trong một không gian vector $d$ chiều, một siêu phẳng có $d-1$ chiều
và chia không gian thành hai nửa không gian.

<!--
Let's start with an example.
Suppose that we have a column vector $\mathbf{w}=[2,1]^\top$. We want to know, "what are the points $\mathbf{v}$ with $\mathbf{w}\cdot\mathbf{v} = 1$?"
By recalling the connection between dot products and angles above :eqref:`eq_angle_forumla`,
we can see that this is equivalent to
-->

Xét ví dụ sau. Giả sử ta có một vector cột $\mathbf{w}=[2,1]^\top$. Ta muốn
biết "tập hợp những điểm $\mathbf{v}$ sao cho $\mathbf{w}\cdot\mathbf{v} = 1$?"
Sử dụng mối quan hệ giữa tích vô hướng và góc ở :eqref:`eq_angle_forumla` ở trên,
ta có thể thấy điều này tương đương với

$$
\|\mathbf{v}\|\|\mathbf{w}\|\cos(\theta) = 1 \; \iff \; \|\mathbf{v}\|\cos(\theta) = \frac{1}{\|\mathbf{w}\|} = \frac{1}{\sqrt{5}}.
$$

<!--
![Recalling trigonometry, we see the formula $\|\mathbf{v}\|\cos(\theta)$ is the length of the projection of the vector $\mathbf{v}$ onto the direction of $\mathbf{w}$](../img/ProjVec.svg)
-->

![Nhắc lại trong lượng giác, chúng ta coi $\|\mathbf{v}\|\cos(\theta)$ là độ
dài hình chiếu của vector $\mathbf{v}$ lên hướng của vector $\mathbf{w}$](../img/ProjVec.svg)
:label:`fig_vector-project`

<!--
If we consider the geometric meaning of this expression,
we see that this is equivalent to saying
that the length of the projection of $\mathbf{v}$
onto the direction of $\mathbf{w}$ is exactly $1/\|\mathbf{w}\|$, as is shown in :numref:`fig_vector-project`.
The set of all points where this is true is a line
at right angles to the vector $\mathbf{w}$.
If we wanted, we could find the equation for this line
and see that it is $2x + y = 1$ or equivalently $y = 1 - 2x$.
-->

Nếu xem xét ý nghĩa hình học của biểu diễn này, chúng ta thấy rằng việc này
tương đương với việc độ dài hình chiếu của $\mathbf{v}$ lên hướng của
$\mathbf{w}$ chính là $1/\|\mathbf{w}\|$ như được biểu diễn trong :numref:`fig_vector-project`. Tập hợp các điểm thỏa mãn điều kiện này là một đường
thẳng vuông góc với vector $\mathbf{w}$. Ta có thể tìm được phương trình của
đường thẳng này là $2x + y = 1$ hoặc $y = 1 - 2x$.

<!-- =================== Kết thúc dịch Phần 5 ==================== -->

<!-- =================== Bắt đầu dịch Phần 6 ==================== -->

<!--
If we now look at what happens when we ask about the set of points with
$\mathbf{w}\cdot\mathbf{v} > 1$ or $\mathbf{w}\cdot\mathbf{v} < 1$,
we can see that these are cases where the projections
are longer or shorter than $1/\|\mathbf{w}\|$, respectively.
Thus, those two inequalities define either side of the line.
In this way, we have found a way to cut our space into two halves,
where all the points on one side have dot product below a threshold,
and the other side above as we see in :numref:`fig_space-division`.
-->

*dịch đoạn phía trên*

<!--
![If we now consider the inequality version of the expression, we see that our hyperplane (in this case: just a line) separates the space into two halves.](../img/SpaceDivision.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/SpaceDivision.svg)
:label:`fig_space-division`

<!--
The story in higher dimension is much the same.
If we now take $\mathbf{w} = [1,2,3]^\top$
and ask about the points in three dimensions with $\mathbf{w}\cdot\mathbf{v} = 1$,
we obtain a plane at right angles to the given vector $\mathbf{w}$.
The two inequalities again define the two sides of the plane as is shown in :numref:`fig_higher-division`.
-->

*dịch đoạn phía trên*

<!--
![Hyperplanes in any dimension separate the space into two halves.](../img/SpaceDivision3D.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/SpaceDivision3D.svg)
:label:`fig_higher-division`

<!--
While our ability to visualize runs out at this point,
nothing stops us from doing this in tens, hundreds, or billions of dimensions.
This occurs often when thinking about machine learned models.
For instance, we can understand linear classification models
like those from :numref:`sec_softmax`,
as methods to find hyperplanes that separate the different target classes.
In this context, such hyperplanes are often referred to as *decision planes*.
The majority of deep learned classification models end
with a linear layer fed into a softmax,
so one can interpret the role of the deep neural network
to be to find a non-linear embedding such that the target classes
can be separated cleanly by hyperplanes.
-->

*dịch đoạn phía trên*

<!--
To give a hand-built example, notice that we can produce a reasonable model
to classify tiny images of t-shirts and trousers from the Fashion MNIST dataset
(seen in :numref:`sec_fashion_mnist`)
by just taking the vector between their means to define the decision plane
and eyeball a crude threshold.  First we will load the data and compute the averages.
-->

*dịch đoạn phía trên*

```{.python .input}
# Load in the dataset
train = gluon.data.vision.FashionMNIST(train=True)
test = gluon.data.vision.FashionMNIST(train=False)

X_train_0 = np.stack([x[0] for x in train if x[1] == 0]).astype(float)
X_train_1 = np.stack([x[0] for x in train if x[1] == 1]).astype(float)
X_test = np.stack(
    [x[0] for x in test if x[1] == 0 or x[1] == 1]).astype(float)
y_test = np.stack(
    [x[1] for x in test if x[1] == 0 or x[1] == 1]).astype(float)

# Compute averages
ave_0 = np.mean(X_train_0, axis=0)
ave_1 = np.mean(X_train_1, axis=0)
```

<!--
It can be informative to examine these averages in detail, so let's plot what they look like.  In this case, we see that the average indeed resembles a blurry image of a t-shirt.
-->

*dịch đoạn phía trên*

```{.python .input}
# Plot average t-shirt
d2l.set_figsize()
d2l.plt.imshow(ave_0.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

<!--
In the second case, we again see that the average resembles a blurry image of trousers.
-->

*dịch đoạn phía trên*

```{.python .input}
# Plot average trousers
d2l.plt.imshow(ave_1.reshape(28, 28).tolist(), cmap='Greys')
d2l.plt.show()
```

<!--
In a fully machine learned solution, we would learn the threshold from the dataset.  In this case, I simply eyeballed a threshold that looked good on the training data by hand.
-->

*dịch đoạn phía trên*

```{.python .input}
# Print test set accuracy with eyeballed threshold
w = (ave_1 - ave_0).T
predictions = X_test.reshape(2000, -1).dot(w.flatten()) > -1500000

# Accuracy
np.mean(predictions.astype(y_test.dtype) == y_test, dtype=np.float64)
```

<!-- =================== Kết thúc dịch Phần 6 ==================== -->

<!-- =================== Bắt đầu dịch Phần 7 ==================== -->

<!--
## Geometry of Linear Transformations
-->

## *dịch tiêu đề phía trên*

<!--
Through :numref:`sec_linear-algebra` and the above discussions,
we have a solid understanding of the geometry of vectors, lengths, and angles.
However, there is one important object we have omitted discussing,
and that is a geometric understanding of linear transformations represented by matrices.  Fully internalizing what matrices can do to transform data
between two potentially different high dimensional spaces takes significant practice,
and is beyond the scope of this appendix.
However, we can start building up intuition in two dimensions.
-->

*dịch đoạn phía trên*

<!--
Suppose that we have some matrix:
-->

*dịch đoạn phía trên*

$$
\mathbf{A} = \begin{bmatrix}
a & b \\ c & d
\end{bmatrix}.
$$

<!--
If we want to apply this to an arbitrary vector
$\mathbf{v} = [x, y]^\top$,
we multiply and see that
-->

*dịch đoạn phía trên*

$$
\begin{aligned}
\mathbf{A}\mathbf{v} & = \begin{bmatrix}a & b \\ c & d\end{bmatrix}\begin{bmatrix}x \\ y\end{bmatrix} \\
& = \begin{bmatrix}ax+by\\ cx+dy\end{bmatrix} \\
& = x\begin{bmatrix}a \\ c\end{bmatrix} + y\begin{bmatrix}b \\d\end{bmatrix} \\
& = x\left\{\mathbf{A}\begin{bmatrix}1\\0\end{bmatrix}\right\} + y\left\{\mathbf{A}\begin{bmatrix}0\\1\end{bmatrix}\right\}.
\end{aligned}
$$

<!--
This may seem like an odd computation,
where something clear became somewhat impenetrable.
However, it tells us that we can write the way
that a matrix transforms *any* vector
in terms of how it transforms *two specific vectors*:
$[1,0]^\top$ and $[0,1]^\top$.
This is worth considering for a moment.
We have essentially reduced an infinite problem
(what happens to any pair of real numbers)
to a finite one (what happens to these specific vectors).
These vectors are an example a *basis*,
where we can write any vector in our space
as a weighted sum of these *basis vectors*.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 7 ==================== -->

<!-- =================== Bắt đầu dịch Phần 8 ==================== -->

<!--
Let's draw what happens when we use the specific matrix
-->

Cùng xét ví dụ với một ma trận cụ thể

$$
\mathbf{A} = \begin{bmatrix}
1 & 2 \\
-1 & 3
\end{bmatrix}.
$$

<!--
If we look at the specific vector $\mathbf{v} = [2, -1]^\top$,
we see this is $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$,
and thus we know that the matrix $A$ will send this to
$2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
If we follow this logic through carefully,
say by considering the grid of all integer pairs of points,
we see that what happens is that the matrix multiplication
can skew, rotate, and scale the grid,
but the grid structure must remain as you see in :numref:`fig_grid-transform`.
-->

Xét vector $\mathbf{v} = [2, -1]^\top$, ta thấy rằng vector này chính bằng $2\cdot[1,0]^\top + -1\cdot[0,1]^\top$,
và bởi vậy ta biến ma trận $A$ sẽ biến đổi nó thành $2(\mathbf{A}[1,0]^\top) + -1(\mathbf{A}[0,1])^\top = 2[1, -1]^\top - [2,3]^\top = [0, -5]^\top$.
Nếu chúng ta xem xét kỹ lưỡng điều này, bằng cách xem lưới của tất cả các điểm có tọa độ nguyên, ta có thể thấy rằng phép nhân ma trận có thể làm lệch, xoay và trước co giãn <scale - từ này học trong hình học lớp 8 mà mình quên> lưới đó, nhưng cấu trúc của lưới giữ nguyên như trong :numref:`fig_grid-transform`.

<!-- câu này mấy bác Tàu viết quá rườm rà, mình sẽ xem lại và tách thành nhiều câu -->

<!--
![The matrix $\mathbf{A}$ acting on the given basis vectors.  Notice how the entire grid is transported along with it.](../img/GridTransform.svg)
-->

![Ma trận $\mathbf{A}$ biến đổi các vector cơ sở cho trước. Chú ý việc
toàn bộ lưới cũng bị biến đổi theo.](../img/GridTransform.svg)
:label:`fig_grid-transform`

<!--
This is the most important intuitive point
to internalize about linear transformations represented by matrices.
Matrices are incapable of distorting some parts of space differently than others.
All they can do is take the original coordinates on our space
and skew, rotate, and scale them.
-->

Đây là điểm quan trọng nhất để hình dung các phép biến đổi tuyến tính thông qua ma trận.
Ma trận không thể làm biến dạng một vài phần của không gian khác với các phần khác. Chúng chỉ có thể lấy các tọa độ ban đầu và làm lệch, xoay và co giãn chúng.

<!--
Some distortions can be severe.  For instance the matrix
-->

Một vài phép biển đổi có thể rất kỳ dị <?>. Chẳng hạn ma trận

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix},
$$

<!--
compresses the entire two-dimensional plane down to a single line.
Identifying and working with such transformations are the topic of a later section,
but geometrically we can see that this is fundamentally different
from the types of transformations we saw above.
For instance, the result from matrix $\mathbf{A}$ can be "bent back" to the original grid.  The results from matrix $\mathbf{B}$ cannot
because we will never know where the vector $[1,2]^\top$ came from---was
it $[1,1]^\top$ or $[0, -1]^\top$?
-->

nén toàn bộ mặt phẳng hai chiều thành một đường thẳng.
Xác định và làm việc với các phép biến đổi này là chủ để của một phần sau, nhưng nhìn trên khía cạnh hình học, ta có thể thấy rằng điều này khác cơ bản so với các phép biến đổi trên đây.
Ví dụ, kết quả từ ma trận $\mathbf{A}$ có thể bị "bẻ cong lại" thành dạng ban đầu.
Kết quả từ ma trận $\mathbf{B}$ thì không thể vì sẽ không thể biết vector $[1,2]^\top$ đến từ đâu -- từ $[1,1]^\top$ hay $[0, -1]^\top$?

<!--
While this picture was for a $2\times2$ matrix,
nothing prevents us from taking the lessons learned into higher dimensions.
If we take similar basis vectors like $[1,0, \ldots,0]$
and see where our matrix sends them,
we can start to get a feeling for how the matrix multiplication
distorts the entire space in whatever dimension space we are dealing with.
-->

Trong khi hình vẽ này áp dụng cho ma trận $2\times2$, kết quả tương tự cũng có thể được mở rộng cho mâ trận bậc cao hơn.
Nếu chúng ta lấy các vector cơ sở như $[1,0, \ldots,0]$ và xem ma trận đó biến đổi các vector này như thế nào, chúng ta có thể có ý niệm về việc phép nhân ma trận đã làm biến dạng toàn bộ không gian trong bất kỳ không gian nào chúng ta đang làm việc với.

<!-- =================== Kết thúc dịch Phần 8 ==================== -->

<!-- =================== Bắt đầu dịch Phần 9 ==================== -->

<!--
## Linear Dependence
-->

## *dịch tiêu đề phía trên*

<!--
Consider again the matrix
-->

*dịch đoạn phía trên*

$$
\mathbf{B} = \begin{bmatrix}
2 & -1 \\ 4 & -2
\end{bmatrix}.
$$

<!--
This compresses the entire plane down to live on the single line $y = 2x$.
The question now arises: is there some way we can detect this
just looking at the matrix itself?
The answer is that indeed we can.
Let's take $\mathbf{b}_1 = [2,4]^\top$ and $\mathbf{b}_2 = [-1, -2]^\top$
be the two columns of $\mathbf{B}$.
Remember that we can write everything transformed by the matrix $\mathbf{B}$
as a weighted sum of the columns of the matrix:
like $a_1\mathbf{b}_1 + a_2\mathbf{b}_2$.
We call this a *linear combination*.
The fact that $\mathbf{b}_1 = -2\cdot\mathbf{b}_2$
means that we can write any linear combination of those two columns
entirely in terms of say $\mathbf{b}_2$ since
-->

*dịch đoạn phía trên*

$$
a_1\mathbf{b}_1 + a_2\mathbf{b}_2 = -2a_1\mathbf{b}_2 + a_2\mathbf{b}_2 = (a_2-2a_1)\mathbf{b}_2.
$$

<!--
This means that one of the columns is, in a sense, redundant
because it does not define a unique direction in space.
This should not surprise us too much
since we already saw that this matrix
collapses the entire plane down into a single line.
Moreover, we see that the linear dependence
$\mathbf{b}_1 = -2\cdot\mathbf{b}_2$ captures this.
To make this more symmetrical between the two vectors, we will write this as
-->

*dịch đoạn phía trên*

$$
\mathbf{b}_1  + 2\cdot\mathbf{b}_2 = 0.
$$

<!--
In general, we will say that a collection of vectors
$\mathbf{v}_1, \ldots \mathbf{v}_k$ are *linearly dependent*
if there exist coefficients $a_1, \ldots, a_k$ *not all equal to zero* so that
-->

*dịch đoạn phía trên*

$$
\sum_{i=1}^k a_i\mathbf{v_i} = 0.
$$

<!--
In this case, we can solve for one of the vectors
in terms of some combination of the others,
and effectively render it redundant.
Thus, a linear dependence in the columns of a matrix
is a witness to the fact that our matrix
is compressing the space down to some lower dimension.
If there is no linear dependence we say the vectors are *linearly independent*.
If the columns of a matrix are linearly independent,
no compression occurs and the operation can be undone.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 9 ==================== -->

<!-- =================== Bắt đầu dịch Phần 10 ==================== -->

<!--
## Rank
-->

## *dịch tiêu đề phía trên*

<!--
If we have a general $n\times m$ matrix,
it is reasonable to ask what dimension space the matrix maps into.
A concept known as the *rank* will be our answer.
In the previous section, we noted that a linear dependence
bears witness to compression of space into a lower dimension
and so we will be able to use this to define the notion of rank.
In particular, the rank of a matrix $\mathbf{A}$
is the largest number of linearly independent columns
amongst all subsets of columns. For example, the matrix
-->

*dịch đoạn phía trên*

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix},
$$

<!--
has $\mathrm{rank}(B)=1$, since the two columns are linearly dependent,
but either column by itself is not linearly dependent.
For a more challenging example, we can consider
-->

*dịch đoạn phía trên*

$$
\mathbf{C} = \begin{bmatrix}
1& 3 & 0 & -1 & 0 \\
-1 & 0 & 1 & 1 & -1 \\
0 & 3 & 1 & 0 & -1 \\
2 & 3 & -1 & -2 & 1
\end{bmatrix},
$$

<!--
and show that $\mathbf{C}$ has rank two since, for instance,
the first two columns are linearly independent,
however any of the four collections of three columns are dependent.
-->

*dịch đoạn phía trên*

<!--
This procedure, as described, is very inefficient.
It requires looking at every subset of the columns of our given matrix,
and thus is potentially exponential in the number of columns.
Later we will see a more computationally efficient way
to compute the rank of a matrix, but for now,
this is sufficient to see that the concept
is well defined and understand the meaning.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 10 ==================== -->

<!-- =================== Bắt đầu dịch Phần 11 ==================== -->

<!--
## Invertibility
-->

## *dịch tiêu đề phía trên*

<!--
We have seen above that multiplication by a matrix with linearly dependent columns
cannot be undone, i.e., there is no inverse operation that can always recover the input.  However, multiplication by a full-rank matrix
(i.e., some $\mathbf{A}$ that is $n \times n$ matrix with rank $n$),
we should always be able to undo it.  Consider the matrix
-->

*dịch đoạn phía trên*

$$
\mathbf{I} = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}.
$$

<!--
which is the matrix with ones along the diagonal, and zeros elsewhere.
We call this the *identity* matrix.
It is the matrix which leaves our data unchanged when applied.
To find a matrix which undoes what our matrix $\mathbf{A}$ has done,
we want to find a matrix $\mathbf{A}^{-1}$ such that
-->

*dịch đoạn phía trên*

$$
\mathbf{A}^{-1}\mathbf{A} = \mathbf{A}\mathbf{A}^{-1} =  \mathbf{I}.
$$

<!--
If we look at this as a system, we have $n \times n$ unknowns
(the entries of $\mathbf{A}^{-1}$) and $n \times n$ equations
(the equality that needs to hold between every entry of the product $\mathbf{A}^{-1}\mathbf{A}$ and every entry of $\mathbf{I}$)
so we should generically expect a solution to exist.
Indeed, in the next section we will see a quantity called the *determinant*,
which has the property that as long as the determinant is not zero, we can find a solution.  We call such a matrix $\mathbf{A}^{-1}$ the *inverse* matrix.
As an example, if $\mathbf{A}$ is the general $2 \times 2$ matrix
-->

*dịch đoạn phía trên*

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

<!--
then we can see that the inverse is
-->

*dịch đoạn phía trên*

$$
 \frac{1}{ad-bc}  \begin{bmatrix}
d & -b \\
-c & a
\end{bmatrix}.
$$

<!--
We can test to see this by seeing that multiplying
by the inverse given by the formula above works in practice.
-->

*dịch đoạn phía trên*

```{.python .input}
M = np.array([[1, 2], [1, 4]])
M_inv = np.array([[2, -1], [-0.5, 0.5]])
M_inv.dot(M)
```

<!-- =================== Kết thúc dịch Phần 11 ==================== -->

<!-- =================== Bắt đầu dịch Phần 12 ==================== -->

<!--
### Numerical Issues
-->

### *dịch tiêu đề phía trên*
While the inverse of a matrix is useful in theory,
we must say that most of the time we do not wish
to *use* the matrix inverse to solve a problem in practice.
In general, there are far more numerically stable algorithms
for solving linear equations like
-->

*dịch đoạn phía trên*

$$
\mathbf{A}\mathbf{x} = \mathbf{b},
$$

<!--
than computing the inverse and multiplying to get
-->

*dịch đoạn phía trên*

$$
\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}.
$$

<!--
Just as division by a small number can lead to numerical instability,
so can inversion of a matrix which is close to having low rank.
-->

*dịch đoạn phía trên*

<!--
Moreover, it is common that the matrix $\mathbf{A}$ is *sparse*,
which is to say that it contains only a small number of non-zero values.
If we were to explore examples, we would see
that this does not mean the inverse is sparse.
Even if $\mathbf{A}$ was a $1$ million by $1$ million matrix
with only $5$ million non-zero entries
(and thus we need only store those $5$ million),
the inverse will typically have almost every entry non-negative,
requiring us to store all $1\text{M}^2$ entries---that is $1$ trillion entries!
-->

*dịch đoạn phía trên*

<!--
While we do not have time to dive all the way into the thorny numerical issues
frequently encountered when working with linear algebra,
we want to provide you with some intuition about when to proceed with caution,
and generally avoiding inversion in practice is a good rule of thumb.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 12 ==================== -->

<!-- =================== Bắt đầu dịch Phần 13 ==================== -->

<!--
## Determinant
-->

## *dịch tiêu đề phía trên*
The geometric view of linear algebra gives an intuitive way
to interpret a a fundamental quantity known as the *determinant*.
Consider the grid image from before, but now with a highlighted region (:numref:`fig_grid-filled`).
-->

*dịch đoạn phía trên*

<!--
![The matrix $\mathbf{A}$ again distorting the grid.  This time, I want to draw particular attention to what happens to the highlighted square.](../img/GridTransformFilled.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/GridTransformFilled.svg)
:label:`fig_grid-filled`

<!--
Look at the highlighted square.  This is a square with edges given
by $(0, 1)$ and $(1, 0)$ and thus it has area one.
After $\mathbf{A}$ transforms this square,
we see that it becomes a parallelogram.
There is no reason this parallelogram should have the same area
that we started with, and indeed in the specific case shown here of
-->

*dịch đoạn phía trên*

$$
\mathbf{A} = \begin{bmatrix}
1 & -1 \\
2 & 3
\end{bmatrix},
$$

<!--
it is an exercise in coordinate geometry to compute
the area of this parallelogram and obtain that the area is $5$.
-->

*dịch đoạn phía trên*

<!--
In general, if we have a matrix
-->

*dịch đoạn phía trên*

$$
\mathbf{A} = \begin{bmatrix}
a & b \\
c & d
\end{bmatrix},
$$

<!--
we can see with some computation that the area
of the resulting parallelogram is $ad-bc$.
This area is referred to as the *determinant*.
-->

*dịch đoạn phía trên*

<!--
Let's check this quickly with some example code.
-->

*dịch đoạn phía trên*

```{.python .input}
import numpy as np
np.linalg.det(np.array([[1, -1], [2, 3]]))
```

<!--
The eagle-eyed amongst us will notice
that this expression can be zero or even negative.
For the negative term, this is a matter of convention
taken generally in mathematics:
if the matrix flips the figure,
we say the area is negated.
Let's see now that when the determinant is zero, we learn more.
-->

*dịch đoạn phía trên*

<!--
Let's consider
-->

*dịch đoạn phía trên*

$$
\mathbf{B} = \begin{bmatrix}
2 & 4 \\ -1 & -2
\end{bmatrix}.
$$

<!--
If we compute the determinant of this matrix,
we get $2\cdot(-2 ) - 4\cdot(-1) = 0$.
Given our understanding above, this makes sense.
$\mathbf{B}$ compresses the square from the original image
down to a line segment, which has zero area.
And indeed, being compressed into a lower dimensional space
is the only way to have zero area after the transformation.
Thus we see the following result is true:
a matrix $A$ is invertible if and only if
the determinant is not equal to zero.
-->

*dịch đoạn phía trên*

<!--
As a final comment, imagine that we have any figure drawn on the plane.
Thinking like computer scientists, we can decompose
that figure into a collection of little squares
so that the area of the figure is in essence
just the number of squares in the decomposition.
If we now transform that figure by a matrix,
we send each of these squares to parallelograms,
each one of which has area given by the determinant.
We see that for any figure, the determinant gives the (signed) number
that a matrix scales the area of any figure.
-->

*dịch đoạn phía trên*

<!--
Computing determinants for larger matrices can be laborious,
but the  intuition is the same.
The determinant remains the factor
that $n\times n$ matrices scale $n$-dimensional volumes.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 13 ==================== -->

<!-- =================== Bắt đầu dịch Phần 14 ==================== -->

<!--
## Tensors and Common Linear Algebra Operations
-->

## *dịch tiêu đề phía trên*

<!--
In :numref:`sec_linear-algebra` the concept of tensors was introduced.
In this section, we will dive more deeply into tensor contractions
(the tensor equivalent of matrix multiplication),
and see how it can provide a unified view
on a number of matrix and vector operations.
-->

*dịch đoạn phía trên*

<!--
With matrices and vectors we knew how to multiply them to transform data.
We need to have a similar definition for tensors if they are to be useful to us.
Think about matrix multiplication:
-->

*dịch đoạn phía trên*

$$
\mathbf{C} = \mathbf{A}\mathbf{B},
$$

<!--
or equivalently
-->

*dịch đoạn phía trên*

$$ c_{i, j} = \sum_{k} a_{i, k}b_{k, j}.$$

<!--
This pattern is one we can repeat for tensors.
For tensors, there is no one case of what
to sum over that can be universally chosen,
so we need specify exactly which indices we want to sum over.
For instance we could consider
-->

*dịch đoạn phía trên*

$$
y_{il} = \sum_{jk} x_{ijkl}a_{jk}.
$$

<!--
Such a transformation is called a *tensor contraction*.
It can represent a far more flexible family of transformations
that matrix multiplication alone.
-->

*dịch đoạn phía trên*

<!--
As a often-used notational simplification,
we can notice that the sum is over exactly those indices
that occur more than once in the expression,
thus people often work with *Einstein notation*,
where the summation is implicitly taken over all repeated indices.
This gives the compact expression:
-->

*dịch đoạn phía trên*

$$
y_{il} = x_{ijkl}a_{jk}.
$$

<!-- =================== Kết thúc dịch Phần 13 ==================== -->

<!-- =================== Bắt đầu dịch Phần 14 ==================== -->

<!--
### Common Examples from Linear Algebra
-->

### *dịch tiêu đề phía trên*

<!--
Let's see how many of the linear algebraic definitions
we have seen before can be expressed in this compressed tensor notation:
-->

*dịch đoạn phía trên*

<!--
* $\mathbf{v} \cdot \mathbf{w} = \sum_i v_iw_i$
* $\|\mathbf{v}\|_2^{2} = \sum_i v_iv_i$
* $(\mathbf{A}\mathbf{v})_i = \sum_j a_{ij}v_j$
* $(\mathbf{A}\mathbf{B})_{ik} = \sum_j a_{ij}b_{jk}$
* $\mathrm{tr}(\mathbf{A}) = \sum_i a_{ii}$
-->

*dịch đoạn phía trên*

<!--
In this way, we can replace a myriad of specialized notations with short tensor expressions.
-->

*dịch đoạn phía trên*

<!--
### Expressing in Code
-->

### *dịch tiêu đề phía trên*
Tensors may flexibly be operated on in code as well.
As seen in :numref:`sec_linear-algebra`,
we can create tensors as is shown below.
-->

*dịch đoạn phía trên*

```{.python .input}
# Define tensors
B = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
A = np.array([[1, 2], [3, 4]])
v = np.array([1, 2])

# Print out the shapes
A.shape, B.shape, v.shape
```

<!--
Einstein summation has been implemented directly  via ```np.einsum```.
The indices that occurs in the Einstein summation can be passed as a string,
followed by the tensors that are being acted upon.
For instance, to implement matrix multiplication,
we can consider the Einstein summation seen above
($\mathbf{A}\mathbf{v} = a_{ij}v_j$)
and strip out the indices themselves to get the implementation:
-->

*dịch đoạn phía trên*

```{.python .input}
# Reimplement matrix multiplication
np.einsum("ij, j -> i", A, v), A.dot(v)
```

<!--
This is a highly flexible notation.
For instance if we want to compute
what would be traditionally written as
-->

*dịch đoạn phía trên*

$$
c_{kl} = \sum_{ij} \mathbf{B}_{ijk}\mathbf{A}_{il}v_j.
$$

<!--
it can be implemented via Einstein summation as:
-->

*dịch đoạn phía trên*

```{.python .input}
np.einsum("ijk, il, j -> kl", B, A, v)
```

<!--
This notation is readable and efficient for humans,
however bulky if for whatever reason
we need to generate a tensor contraction programmatically.
For this reason, `einsum` provides an alternative notation
by providing integer indices for each tensor.
For example, the same tensor contraction can also be written as:
-->

*dịch đoạn phía trên*

```{.python .input}
np.einsum(B, [0, 1, 2], A, [0, 3], v, [1], [2, 3])
```

<!--
Either notation allows for concise and efficient representation of tensor contractions in code.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 14 ==================== -->

<!-- =================== Bắt đầu dịch Phần 15 ==================== -->

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* Vectors can be interpreted geometrically as either points or directions in space.
-->

*dịch đoạn phía trên*

<--
* Dot products define the notion of angle to arbitrarily high-dimensional spaces.
-->

*dịch đoạn phía trên*

<--
* Hyperplanes are high-dimensional generalizations of lines and planes.  They can be used to define decision planes that are often used as the last step in a classification task.
-->

*dịch đoạn phía trên*

<--
* Matrix multiplication can be geometrically interpreted as uniform distortions of the underlying coordinates. They represent a very restricted, but mathematically clean, way to transform vectors.
-->

*dịch đoạn phía trên*

<--
* Linear dependence is a way to tell when a collection of vectors are in a lower dimensional space than we would expect (say you have $3$ vectors living in a $2$-dimensional space). The rank of a matrix is the size of the largest subset of its columns that are linearly independent.
-->

*dịch đoạn phía trên*

<--
* When a matrix's inverse is defined, matrix inversion allows us to find another matrix that undoes the action of the first. Matrix inversion is useful in theory, but requires care in practice owing to numerical instability.
-->

*dịch đoạn phía trên*

<--
* Determinants allow us to measure how much a matrix expands or contracts a space. A nonzero determinant implies an invertible (non-singular) matrix and a zero-valued determinant means that the matrix is non-invertible (singular).
* Tensor contractions and Einstein summation provide for a neat and clean notation for expressing many of the computations that are seen in machine learning.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. What is the angle between
-->

*dịch đoạn phía trên*

$$
\vec v_1 = \begin{bmatrix}
1 \\ 0 \\ -1 \\ 2
\end{bmatrix}, \qquad \vec v_2 = \begin{bmatrix}
3 \\ 1 \\ 0 \\ 1
\end{bmatrix}?
$$

<!--
2. True or false: $\begin{bmatrix}1 & 2\\0&1\end{bmatrix}$ and $\begin{bmatrix}1 & -2\\0&1\end{bmatrix}$ are inverses of one another?
-->

*dịch đoạn phía trên*

<!--
3. Suppose that we draw a shape in the plane with area $100\mathrm{m}^2$.  What is the area after transforming the figure by the matrix
-->

*dịch đoạn phía trên*

$$
\begin{bmatrix}
2 & 3\\
1 & 2
\end{bmatrix}.
$$

<!--
4. Which of the following sets of vectors are linearly independent?
-->

*dịch đoạn phía trên*

 * $\left\{\begin{pmatrix}1\\0\\-1\end{pmatrix}, \begin{pmatrix}2\\1\\-1\end{pmatrix}, \begin{pmatrix}3\\1\\1\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}3\\1\\1\end{pmatrix}, \begin{pmatrix}1\\1\\1\end{pmatrix}, \begin{pmatrix}0\\0\\0\end{pmatrix}\right\}$
 * $\left\{\begin{pmatrix}1\\1\\0\end{pmatrix}, \begin{pmatrix}0\\1\\-1\end{pmatrix}, \begin{pmatrix}1\\0\\1\end{pmatrix}\right\}$

<!--
5. Suppose that you have a matrix written as $A = \begin{bmatrix}c\\d\end{bmatrix}\cdot\begin{bmatrix}a & b\end{bmatrix}$ for some choice of values $a, b, c$, and $d$.  True or false: the determinant of such a matrix is always $0$?
-->

*dịch đoạn phía trên*

<!--
6. The vectors $e_1 = \begin{bmatrix}1\\0\end{bmatrix}$ and $e_2 = \begin{bmatrix}0\\1\end{bmatrix}$ are orthogonal.  What is the condition on a matrix $A$ so that $Ae_1$ and $Ae_2$ are orthogonal?
-->

*dịch đoạn phía trên*

<!--
7. How can you write $\mathrm{tr}(\mathbf{A}^4)$ in Einstein notation for an arbitrary matrix $A$?
-->

*dịch đoạn phía trên*


<!--
## [Discussions](https://discuss.mxnet.io/t/5147)
-->

## *dịch tiêu đề phía trên*

<!--
![](../img/qr_geometry-linear-algebric-ops.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/qr_geometry-linear-algebric-ops.svg)

<!-- =================== Kết thúc dịch Phần 15 ==================== -->

### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Mỗi tên chỉ xuất hiện một lần: Nếu bạn đã dịch hoặc review phần 1 của trang này
thì không cần điền vào các phần sau nữa.
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

<!-- Phần 1 -->
* Vũ Hữu Tiệp

<!-- Phần 2 -->
* Lê Khắc Hồng Phúc

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

<!-- Phần 8 -->
*

<!-- Phần 9 -->
*

<!-- Phần 10 -->
*

<!-- Phần 11 -->
*

<!-- Phần 12 -->
*

<!-- Phần 13 -->
*

<!-- Phần 14 -->
*

<!-- Phần 15 -->
*

# Integral Calculus
:label:`sec_integral_calculus`

Sự khác biệt chỉ chiếm một nửa nội dung của một nền giáo dục giải tích truyền thống. Trụ cột khác, hội nhập, bắt đầu dường như một câu hỏi khá tách biệt, “Khu vực bên dưới đường cong này là gì?” Mặc dù dường như không liên quan, hội nhập được đan xen chặt chẽ với sự khác biệt thông qua cái được gọi là định lý cơ bản * của giải tích*. 

Ở cấp độ học máy mà chúng ta thảo luận trong cuốn sách này, chúng ta sẽ không cần sự hiểu biết sâu sắc về hội nhập. Tuy nhiên, chúng tôi sẽ cung cấp một giới thiệu ngắn gọn để đặt nền tảng cho bất kỳ ứng dụng tiếp theo nào chúng tôi sẽ gặp sau này. 

## Giải thích hình học Giả sử chúng ta có hàm $f(x)$. Để đơn giản, chúng ta hãy giả định rằng $f(x)$ là không âm (không bao giờ có giá trị nhỏ hơn 0). Điều chúng ta muốn thử và hiểu là: khu vực chứa giữa $f(x)$ và trục $x$-trục là gì?

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()

x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from mpl_toolkits import mplot3d
import torch

x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist(), f.tolist())
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from mpl_toolkits import mplot3d
import tensorflow as tf

x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy(), f.numpy())
d2l.plt.show()
```

Trong hầu hết các trường hợp, khu vực này sẽ là vô hạn hoặc không xác định (xem xét khu vực dưới $f(x) = x^{2}$), vì vậy mọi người thường sẽ nói về khu vực giữa một cặp kết thúc, nói $a$ và $b$.

```{.python .input}
x = np.arange(-2, 2, 0.01)
f = np.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
x = torch.arange(-2, 2, 0.01)
f = torch.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.tolist()[50:250], f.tolist()[50:250])
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
x = tf.range(-2, 2, 0.01)
f = tf.exp(-x**2)

d2l.set_figsize()
d2l.plt.plot(x, f, color='black')
d2l.plt.fill_between(x.numpy()[50:250], f.numpy()[50:250])
d2l.plt.show()
```

Chúng ta sẽ biểu thị khu vực này bằng ký hiệu tích phân bên dưới: 

$$
\mathrm{Area}(\mathcal{A}) = \int_a^b f(x) \;dx.
$$

Biến bên trong là một biến giả, giống như chỉ số của một tổng trong một $\sum$, và do đó điều này có thể được viết tương đương với bất kỳ giá trị bên trong chúng ta thích: 

$$
\int_a^b f(x) \;dx = \int_a^b f(z) \;dz.
$$

Có một cách truyền thống để thử và hiểu cách chúng ta có thể cố gắng xấp xỉ các tích phân như vậy: chúng ta có thể tưởng tượng việc lấy khu vực nằm trong khoảng $a$ và $b$ và cắt nó thành $N$ lát dọc. Nếu $N$ lớn, chúng ta có thể xấp xỉ diện tích của mỗi lát bằng một hình chữ nhật, và sau đó thêm các khu vực để lấy tổng diện tích dưới đường cong. Chúng ta hãy xem xét một ví dụ làm điều này trong mã. Chúng ta sẽ thấy làm thế nào để có được giá trị thực sự trong một phần sau.

```{.python .input}
epsilon = 0.05
a = 0
b = 2

x = np.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = np.sum(epsilon*f)
true = np.log(2) / 2

d2l.set_figsize()
d2l.plt.bar(x.asnumpy(), f.asnumpy(), width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab pytorch
epsilon = 0.05
a = 0
b = 2

x = torch.arange(a, b, epsilon)
f = x / (1 + x**2)

approx = torch.sum(epsilon*f)
true = torch.log(torch.tensor([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

```{.python .input}
#@tab tensorflow
epsilon = 0.05
a = 0
b = 2

x = tf.range(a, b, epsilon)
f = x / (1 + x**2)

approx = tf.reduce_sum(epsilon*f)
true = tf.math.log(tf.constant([5.])) / 2

d2l.set_figsize()
d2l.plt.bar(x, f, width=epsilon, align='edge')
d2l.plt.plot(x, f, color='black')
d2l.plt.ylim([0, 1])
d2l.plt.show()

f'approximation: {approx}, truth: {true}'
```

Vấn đề là mặc dù nó có thể được thực hiện bằng số, chúng ta có thể thực hiện cách tiếp cận này một cách phân tích chỉ cho các chức năng đơn giản nhất như 

$$
\int_a^b x \;dx.
$$

Bất cứ điều gì hơi phức tạp hơn như ví dụ của chúng tôi từ mã trên 

$$
\int_a^b \frac{x}{1+x^{2}} \;dx.
$$

vượt ra ngoài những gì chúng ta có thể giải quyết bằng phương pháp trực tiếp như vậy. 

Thay vào đó, chúng tôi sẽ có một cách tiếp cận khác. Chúng ta sẽ làm việc trực giác với khái niệm về khu vực, và tìm hiểu công cụ tính toán chính được sử dụng để tìm tích phân: định lý * cơ bản của giải tính*. Đây sẽ là cơ sở cho việc nghiên cứu hội nhập của chúng tôi. 

## Định lý cơ bản của giải tích

Để đi sâu hơn vào lý thuyết hội nhập, chúng ta hãy giới thiệu một hàm 

$$
F(x) = \int_0^x f(y) dy.
$$

Chức năng này đo diện tích giữa $0$ và $x$ tùy thuộc vào cách chúng ta thay đổi $x$. Lưu ý rằng đây là tất cả mọi thứ chúng ta cần kể từ 

$$
\int_a^b f(x) \;dx = F(b) - F(a).
$$

Đây là một mã hóa toán học của thực tế là chúng ta có thể đo diện tích ra đến điểm cuối xa và sau đó trừ khu vực vào điểm cuối gần như được chỉ ra trong :numref:`fig_area-subtract`. 

![Visualizing why we may reduce the problem of computing the area under a curve between two points to computing the area to the left of a point.](../img/sub-area.svg)
:label:`fig_area-subtract`

Do đó, chúng ta có thể tìm ra tích phân trong bất kỳ khoảng thời gian nào bằng cách tìm ra $F(x)$ là gì. 

Để làm như vậy, chúng ta hãy xem xét một thí nghiệm. Như chúng ta thường làm trong giải tích, chúng ta hãy tưởng tượng những gì xảy ra khi chúng ta thay đổi giá trị bằng một chút nhỏ. Từ nhận xét ở trên, chúng tôi biết rằng 

$$
F(x+\epsilon) - F(x) = \int_x^{x+\epsilon} f(y) \; dy.
$$

Điều này cho chúng ta biết rằng chức năng thay đổi bởi khu vực dưới một mảnh nhỏ của một hàm. 

Đây là điểm mà chúng tôi thực hiện một xấp xỉ. Nếu chúng ta nhìn vào một mảnh nhỏ của khu vực như thế này, có vẻ như khu vực này gần với khu vực hình chữ nhật với chiều cao giá trị $f(x)$ và chiều rộng cơ sở $\epsilon$. Thật vậy, người ta có thể cho thấy rằng như $\epsilon \rightarrow 0$ xấp xỉ này trở nên tốt hơn và tốt hơn. Vì vậy chúng ta có thể kết luận: 

$$
F(x+\epsilon) - F(x) \approx \epsilon f(x).
$$

Tuy nhiên, bây giờ chúng ta có thể nhận thấy: đây chính xác là mô hình mà chúng ta mong đợi nếu chúng ta đang tính toán phái sinh của $F$! Vì vậy, chúng ta thấy thực tế khá đáng ngạc nhiên sau đây: 

$$
\frac{dF}{dx}(x) = f(x).
$$

Đây là định lý * cơ bản của giải tí*. Chúng tôi có thể viết nó ở dạng mở rộng như $$\frac{d}{dx}\int_{-\infty}^x f(y) \; dy = f(x).$$ :eqlabel:`eq_ftc` 

Nó có khái niệm tìm kiếm các khu vực (*a priori* khá khó), và giảm nó thành một dẫn xuất tuyên bố (một cái gì đó hoàn toàn hiểu hơn nhiều). Một nhận xét cuối cùng mà chúng ta phải đưa ra là điều này không cho chúng ta biết chính xác $F(x)$ là gì. Thật vậy $F(x) + C$ cho bất kỳ $C$ nào có cùng phái sinh. Đây là một thực tế của cuộc sống trong lý thuyết hội nhập. Rất may, lưu ý rằng khi làm việc với tích phân xác định, các hằng số sẽ bỏ ra, và do đó không liên quan đến kết quả. 

$$
\int_a^b f(x) \; dx = (F(b) + C) - (F(a) + C) = F(b) - F(a).
$$

Điều này có vẻ như là vô nghĩa trừu tượng, nhưng chúng ta hãy dành một chút thời gian để đánh giá cao rằng nó đã cho chúng ta một quan điểm hoàn toàn mới về tích phân tính toán. Mục tiêu của chúng tôi là không còn để thực hiện một số loại quá trình chop-and-sum để thử và phục hồi khu vực, thay vào đó chúng ta chỉ cần tìm một hàm mà đạo hàm là hàm chúng ta có! Điều này thật đáng kinh ngạc vì bây giờ chúng ta có thể liệt kê nhiều tích phân khá khó khăn bằng cách đảo ngược bảng từ :numref:`sec_derivative_table`. Ví dụ, chúng ta biết rằng đạo hàm của $x^{n}$ là $nx^{n-1}$. Như vậy, chúng ta có thể nói bằng cách sử dụng định lý cơ bản :eqref:`eq_ftc` rằng 

$$
\int_0^{x} ny^{n-1} \; dy = x^n - 0^n = x^n.
$$

Tương tự, chúng ta biết rằng đạo hàm của $e^{x}$ là chính nó, vì vậy điều đó có nghĩa là 

$$
\int_0^{x} e^{x} \; dx = e^{x} - e^{0} = e^x - 1.
$$

Bằng cách này, chúng ta có thể phát triển toàn bộ lý thuyết hội nhập tận dụng các ý tưởng từ phép tính vi phân một cách tự do. Mỗi quy tắc tích hợp bắt nguồn từ một thực tế này. 

## Thay đổi các biến
:label:`integral_example`

Cũng giống như với sự khác biệt, có một số quy tắc làm cho việc tính toán tích phân có thể truy xuất hơn. Trên thực tế, mọi quy tắc của phép tính vi phân (như quy tắc sản phẩm, quy tắc tổng, và quy tắc chuỗi) đều có quy tắc tương ứng cho phép tích phân (tích hợp theo các phần, tuyến tính tích hợp, và sự thay đổi công thức biến tương ứng). Trong phần này, chúng ta sẽ đi sâu vào những gì được cho là quan trọng nhất từ danh sách: sự thay đổi của công thức biến. 

Đầu tiên, giả sử rằng chúng ta có một hàm mà chính nó là một tích phân: 

$$
F(x) = \int_0^x f(y) \; dy.
$$

Chúng ta hãy giả sử rằng chúng ta muốn biết hàm này trông như thế nào khi chúng ta soạn nó với một chức năng khác để có được $F(u(x))$. Theo quy tắc chuỗi, chúng ta biết 

$$
\frac{d}{dx}F(u(x)) = \frac{dF}{du}(u(x))\cdot \frac{du}{dx}.
$$

Chúng ta có thể biến điều này thành một tuyên bố về hội nhập bằng cách sử dụng định lý cơ bản :eqref:`eq_ftc` như trên. Điều này cho 

$$
F(u(x)) - F(u(0)) = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Nhắc lại rằng $F$ chính nó là một tích phân cho rằng phía bên tay trái có thể được viết lại để được 

$$
\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x \frac{dF}{du}(u(y))\cdot \frac{du}{dy} \;dy.
$$

Tương tự, nhắc lại rằng $F$ là một tích phân cho phép chúng ta nhận ra rằng $\frac{dF}{dx} = f$ sử dụng định lý cơ bản :eqref:`eq_ftc`, và do đó chúng ta có thể kết luận 

$$\int_{u(0)}^{u(x)} f(y) \; dy = \int_0^x f(u(y))\cdot \frac{du}{dy} \;dy.$$
:eqlabel:`eq_change_var`

Đây là công thức *thay đổi biến*. 

Để có một nguồn gốc trực quan hơn, hãy xem xét những gì xảy ra khi chúng ta lấy một tích phân của $f(u(x))$ giữa $x$ và $x+\epsilon$. Đối với một $\epsilon$ nhỏ, tích phân này là khoảng $\epsilon f(u(x))$, diện tích của hình chữ nhật liên quan. Bây giờ, chúng ta hãy so sánh điều này với tích phân của $f(y)$ từ $u(x)$ đến $u(x+\epsilon)$. Chúng ta biết rằng $u(x+\epsilon) \approx u(x) + \epsilon \frac{du}{dx}(x)$, vì vậy diện tích của hình chữ nhật này là khoảng $\epsilon \frac{du}{dx}(x)f(u(x))$. Do đó, để làm cho diện tích của hai hình chữ nhật này đồng ý, chúng ta cần nhân hình đầu tiên với $\frac{du}{dx}(x)$ như được minh họa trong :numref:`fig_rect-transform`. 

![Visualizing the transformation of a single thin rectangle under the change of variables.](../img/rect-trans.svg)
:label:`fig_rect-transform`

Điều này cho chúng ta biết rằng 

$$
\int_x^{x+\epsilon} f(u(y))\frac{du}{dy}(y)\;dy = \int_{u(x)}^{u(x+\epsilon)} f(y) \; dy.
$$

Đây là sự thay đổi của công thức biến thể hiện cho một hình chữ nhật nhỏ duy nhất. 

Nếu $u(x)$ và $f(x)$ được chọn đúng cách, điều này có thể cho phép tính toán các tích phân cực kỳ phức tạp. Ví dụ, nếu chúng ta thậm chí chọn $f(y) = 1$ và $u(x) = e^{-x^{2}}$ (có nghĩa là $\frac{du}{dx}(x) = -2xe^{-x^{2}}$), điều này có thể cho thấy ví dụ 

$$
e^{-1} - 1 = \int_{e^{-0}}^{e^{-1}} 1 \; dy = -2\int_0^{1} ye^{-y^2}\;dy,
$$

và do đó bằng cách sắp xếp lại điều đó 

$$
\int_0^{1} ye^{-y^2}\; dy = \frac{1-e^{-1}}{2}.
$$

## Một bình luận về Sign Conventions

Độc giả mắt quan sát sẽ quan sát một điều gì đó kỳ lạ về các tính toán ở trên. Cụ thể là, tính toán như 

$$
\int_{e^{-0}}^{e^{-1}} 1 \; dy = e^{-1} -1 < 0,
$$

can produce sản xuất negative âm numbers số. Khi suy nghĩ về các khu vực, có thể lạ khi thấy một giá trị âm, và vì vậy nó đáng để đào sâu vào quy ước là gì. 

Các nhà toán học lấy khái niệm về các khu vực có chữ ký. Điều này thể hiện theo hai cách. Đầu tiên, nếu chúng ta xem xét một hàm $f(x)$ đôi khi nhỏ hơn 0, thì khu vực cũng sẽ âm. Vì vậy, ví dụ 

$$
\int_0^{1} (-1)\;dx = -1.
$$

Tương tự như vậy, tích phân tiến từ phải sang trái, thay vì trái sang phải cũng được coi là các khu vực âm 

$$
\int_0^{-1} 1\; dx = -1.
$$

Khu vực tiêu chuẩn (từ trái sang phải của một hàm tích cực) luôn là tích cực. Bất cứ thứ gì thu được bằng cách lật nó (giả sử lật qua trục $x$-để lấy tích phân của một số âm, hoặc lật qua trục $y$-để lấy tích phân theo thứ tự sai) sẽ tạo ra một vùng âm. Và thực sự, lật hai lần sẽ đưa ra một cặp dấu hiệu tiêu cực hủy bỏ để có khu vực tích cực 

$$
\int_0^{-1} (-1)\;dx =  1.
$$

Nếu cuộc thảo luận này nghe có vẻ quen thuộc, đó là! Trong :numref:`sec_geometry-linear-algebraic-ops`, chúng tôi đã thảo luận về cách yếu tố quyết định đại diện cho khu vực đã ký theo cùng một cách. 

## Nhiều tích hợp Trong một số trường hợp, chúng ta sẽ cần phải làm việc ở các kích thước cao hơn. Ví dụ, giả sử rằng chúng ta có một hàm của hai biến, như $f(x, y)$ và chúng ta muốn biết khối lượng dưới $f$ khi $x$ phạm vi trên $[a, b]$ và $y$ phạm vi trên $[c, d]$.

```{.python .input}
# Construct grid and compute function
x, y = np.meshgrid(np.linspace(-2, 2, 101), np.linspace(-2, 2, 101),
                   indexing='ij')
z = np.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x.asnumpy(), y.asnumpy(), z.asnumpy())
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab pytorch
# Construct grid and compute function
x, y = torch.meshgrid(torch.linspace(-2, 2, 101), torch.linspace(-2, 2, 101))
z = torch.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

```{.python .input}
#@tab tensorflow
# Construct grid and compute function
x, y = tf.meshgrid(tf.linspace(-2., 2., 101), tf.linspace(-2., 2., 101))
z = tf.exp(- x**2 - y**2)

# Plot function
ax = d2l.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)
d2l.plt.xlabel('x')
d2l.plt.ylabel('y')
d2l.plt.xticks([-2, -1, 0, 1, 2])
d2l.plt.yticks([-2, -1, 0, 1, 2])
d2l.set_figsize()
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(0, 1)
ax.dist = 12
```

Chúng tôi viết cái này là 

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Giả sử rằng chúng ta muốn tính toán tích phân này. Tuyên bố của tôi là chúng ta có thể làm điều này bằng cách tính toán lặp đi lặp lại đầu tiên tích phân trong $x$ và sau đó chuyển sang tích phân trong $y$, có nghĩa là 

$$
\int_{[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int_c^{d} \left(\int_a^{b} f(x, y) \;dx\right) \; dy.
$$

Hãy để chúng tôi xem tại sao điều này là. 

Hãy xem xét hình trên nơi chúng ta đã chia hàm thành $\epsilon \times \epsilon$ ô vuông mà chúng ta sẽ lập chỉ mục với tọa độ số nguyên $i, j$. Trong trường hợp này, tích phân của chúng tôi là khoảng 

$$
\sum_{i, j} \epsilon^{2} f(\epsilon i, \epsilon j).
$$

Khi chúng tôi phân biệt vấn đề, chúng tôi có thể thêm các giá trị trên các ô vuông này theo bất kỳ thứ tự nào chúng tôi thích và không lo lắng về việc thay đổi các giá trị. Điều này được minh họa trong :numref:`fig_sum-order`. Đặc biệt, chúng ta có thể nói rằng 

$$
 \sum _ {j} \epsilon \left(\sum_{i} \epsilon f(\epsilon i, \epsilon j)\right).
$$

![Illustrating how to decompose a sum over many squares as a sum over first the columns (1), then adding the column sums together (2).](../img/sum-order.svg)
:label:`fig_sum-order`

Tổng ở bên trong chính xác là sự khác biệt của tích phân 

$$
G(\epsilon j) = \int _a^{b} f(x, \epsilon j) \; dx.
$$

Cuối cùng, hãy chú ý rằng nếu chúng ta kết hợp hai biểu thức này, chúng ta sẽ nhận được 

$$
\sum _ {j} \epsilon G(\epsilon j) \approx \int _ {c}^{d} G(y) \; dy = \int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy.
$$

Vì vậy, đặt tất cả lại với nhau, chúng tôi có điều đó 

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ c^{d} \left(\int _ a^{b} f(x, y) \;dx\right) \; dy.
$$

Lưu ý rằng, một khi kín đáo, tất cả những gì chúng tôi đã làm là sắp xếp lại thứ tự mà chúng tôi đã thêm một danh sách các số. Điều này có thể làm cho nó có vẻ như không có gì, tuy nhiên kết quả này (được gọi là *Fubini's Theorem*) không phải lúc nào cũng đúng! Đối với loại toán học gặp phải khi thực hiện học máy (hàm liên tục), không có mối quan tâm, tuy nhiên có thể tạo ra các ví dụ mà nó không thành công (ví dụ hàm $f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3$ trên hình chữ nhật $[0,2]\times[0,1]$). 

Lưu ý rằng sự lựa chọn để thực hiện tích phân trong $x$ đầu tiên, và sau đó tích phân trong $y$ là tùy ý. Chúng ta có thể chọn tốt như nhau để làm $y$ đầu tiên và sau đó là $x$ để xem 

$$
\int _ {[a, b]\times[c, d]} f(x, y)\;dx\;dy = \int _ a^{b} \left(\int _ c^{d} f(x, y) \;dy\right) \; dx.
$$

Thông thường, chúng ta sẽ ngưng tụ thành ký hiệu vector, và nói rằng đối với $U = [a, b]\times [c, d]$ đây là 

$$
\int _ U f(\mathbf{x})\;d\mathbf{x}.
$$

## Thay đổi biến trong nhiều tích phân Như với các biến đơn trong :eqref:`eq_change_var`, khả năng thay đổi các biến bên trong tích phân chiều cao hơn là một công cụ quan trọng. Hãy để chúng tôi tóm tắt kết quả mà không có nguồn gốc. 

Chúng ta cần một chức năng reparameterizes miền tích hợp của chúng ta. Chúng ta có thể lấy điều này là $\phi : \mathbb{R}^n \rightarrow \mathbb{R}^n$, đó là bất kỳ chức năng mà mất trong $n$ biến thực và trả về $n$ khác. Để giữ cho các biểu thức sạch sẽ, chúng ta sẽ giả định rằng $\phi$ là * injective* đó là để nói rằng nó không bao giờ tự gấp lại ($\phi(\mathbf{x}) = \phi(\mathbf{y}) \implies \mathbf{x} = \mathbf{y}$). 

In this case, we can say that

$$
\int _ {\phi(U)} f(\mathbf{x})\;d\mathbf{x} = \int _ {U} f(\phi(\mathbf{x})) \left|\det(D\phi(\mathbf{x}))\right|\;d\mathbf{x}.
$$

where $D\phi$ is the *Jacobian* of $\phi$, which is the matrix of partial derivatives of $\boldsymbol{\phi} = (\phi_1(x_1, \ldots, x_n), \ldots, \phi_n(x_1, \ldots, x_n))$,

$$
D\boldsymbol{\phi} = \begin{bmatrix}
\frac{\partial \phi _ 1}{\partial x _ 1} & \cdots & \frac{\partial \phi _ 1}{\partial x _ n} \\
\vdots & \ddots & \vdots \\
\frac{\partial \phi _ n}{\partial x _ 1} & \cdots & \frac{\partial \phi _ n}{\partial x _ n}
\end{bmatrix}.
$$

Looking closely, we see that this is similar to the single variable chain rule :eqref:`eq_change_var`, except we have replaced the term $\frac{du}{dx}(x)$ with $\left|\det(D\phi(\mathbf{x}))\right|$.  Let us see how we can to interpret this term.  Recall that the $\frac{du}{dx}(x)$ term existed to say how much we stretched our $x$-axis by applying $u$.  The same process in higher dimensions is to determine how much we stretch the area (or volume, or hyper-volume) of a little square (or little *hyper-cube*) by applying $\boldsymbol{\phi}$.  If $\boldsymbol{\phi}$ was the multiplication by a matrix, then we know how the determinant already gives the answer.

With some work, one can show that the *Jacobian* provides the best approximation to a multivariable function $\boldsymbol{\phi}$ at a point by a matrix in the same way we could approximate by lines or planes with derivatives and gradients. Thus the determinant of the Jacobian exactly mirrors the scaling factor we identified in one dimension.

It takes some work to fill in the details to this, so do not worry if they are not clear now.  Let us see at least one example we will make use of later on.  Consider the integral

$$
\int _ {-\infty}^{\infty} \int _ {-\infty}^{\infty} e^{-x^{2}-y^{2}} \;dx\;dy.
$$

Playing with this integral directly will get us no-where, but if we change variables, we can make significant progress.  If we let $\boldsymbol{\phi}(r, \theta) = (r \cos(\theta),  r\sin(\theta))$ (which is to say that $x = r \cos(\theta)$, $y = r \sin(\theta)$), then we can apply the change of variable formula to see that this is the same thing as

$$
\int _ 0^\infty \int_0 ^ {2\pi} e^{-r^{2}} \left|\det(D\mathbf{\phi}(\mathbf{x}))\right|\;d\theta\;dr,
$$

where

$$
\left|\det(D\mathbf{\phi}(\mathbf{x}))\right| = \left|\det\begin{bmatrix}
\cos(\theta) & -r\sin(\theta) \\
\sin(\theta) & r\cos(\theta)
\end{bmatrix}\right| = r(\cos^{2}(\theta) + \sin^{2}(\theta)) = r.
$$

Thus, the integral is

$$
\int _ 0^\infty \int _ 0 ^ {2\pi} re^{-r^{2}} \;d\theta\;dr = 2\pi\int _ 0^\infty re^{-r^{2}} \;dr = \pi,
$$

where the final equality follows by the same computation that we used in section :numref:`integral_example`.

We will meet this integral again when we study continuous random variables in :numref:`sec_random_variables`.

## Tóm tắt

* Lý thuyết hội nhập cho phép chúng ta trả lời các câu hỏi về các khu vực hoặc khối lượng.
* Định lý cơ bản của giải tích cho phép chúng ta tận dụng kiến thức về các dẫn xuất để tính toán các khu vực thông qua quan sát rằng đạo hàm của khu vực lên đến một số điểm được đưa ra bởi giá trị của hàm được tích hợp.
* Tích phân trong các kích thước cao hơn có thể được tính toán bằng cách lặp các tích phân biến đơn lẻ.

## Exercises
1. What is $\int_1^2 \frac{1}{x} \;dx$?
2. Use the change of variables formula to integrate $\int_0^{\sqrt{\pi}}x\sin(x^2)\;dx$.
3. What is $\int_{[0,1]^2} xy \;dx\;dy$?
4. Use the change of variables formula to compute $\int_0^2\int_0^1xy(x^2-y^2)/(x^2+y^2)^3\;dy\;dx$ and $\int_0^1\int_0^2f(x, y) = xy(x^2-y^2)/(x^2+y^2)^3\;dx\;dy$ to see they are different.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/414)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1092)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1093)
:end_tab:

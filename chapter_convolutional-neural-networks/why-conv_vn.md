<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# From Dense Layers to Convolutions
-->

# Từ Tầng Kết nối Dày đặc đến phép Tích chập

<!--
The models that we have discussed so far are fine options if you are dealing with *tabular* data.
By *tabular* we mean that the data consists of rows corresponding to examples and columns corresponding to features.
With tabular data, we might anticipate that pattern we seek could require modeling interactions among the features, 
but do not assume anything a priori about which features are related to each other or in what way.
-->

Đến nay, các mô hình mà ta đã thảo luận là các lựa chọn phù hợp nếu dữ liệu mà ta đang xử lý có *dạng bảng* với các hàng tương ứng với các mẫu, còn các cột tương ứng với các đặc trưng.
Với dữ liệu có dạng như vậy, ta có thể dự đoán rằng khuôn mẫu mà ta đang tìm kiếm có thể yêu cầu việc mô hình hóa sự tương tác giữa các đặc trưng, 
nhưng ta không giả định trước rằng những đặc trưng nào liên quan tới nhau và mối quan hệ của chúng.


<!--
Sometimes we truly may not have any knowledge to guide the construction of more cleverly-organized architectures.
In these cases, a multilayer perceptron is often the best that we can do.
However, once we start dealing with high-dimensional perceptual data, these *structure-less* networks can grow unwieldy.
-->

Đôi khi ta thực sự không có bất kỳ kiến thức nào để định hướng việc thiết kế các kiến trúc được sắp xếp khéo léo hơn.
Trong những trường hợp này, một perceptron đa tầng thường là giải pháp tốt nhất.
Tuy nhiên, một khi ta bắt đầu xử lý dữ liệu tri giác đa chiều, các mạng *không có cấu trúc* này có thể sẽ trở nên quá cồng kềnh.

<!--
For instance, let us return to our running example of distinguishing cats from dogs.
Say that we do a thorough job in data collection, collecting an annotated sets of high-quality 1-megapixel photographs.
This means that the input into a network has *1 million dimensions*.
Even an aggressive reduction to *1,000 hidden dimensions* would require a *dense* (fully-connected) layer to support $10^9$ parameters.
Unless we have an extremely large dataset (perhaps billions?), lots of GPUs, a talent for extreme distributed optimization, and an extraordinary amount of patience,
learning the parameters of this network may turn out to be impossible.
-->

Hãy quay trở lại với ví dụ phân biệt chó và mèo quen thuộc.
Giả sử ta đã thực hiện việc thu thập dữ liệu một cách kỹ lưỡng và thu được một bộ ảnh được gán nhãn chất lượng cao với độ phân giải 1 triệu điểm ảnh.
Điều này có nghĩa là đầu vào của mạng sẽ có *1 triệu chiều*.
Ngay cả việc giảm mạnh xuống còn *1000 chiều ẩn* sẽ cần tới một tầng *dày đặc* (kết nối đầy đủ) có $10^9$ tham số.
Trừ khi ta có một tập dữ liệu cực lớn (có thể là hàng tỷ ảnh?), một số lượng lớn GPU, 
chuyên môn cao trong việc tối ưu hóa phân tán và sức kiên nhẫn phi thường, việc học các tham số của mạng này có thể là điều bất khả thi.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
A careful reader might object to this argument on the basis that 1 megapixel resolution may not be necessary.
However, while you could get away with 100,000 pixels, we grossly underestimated the number of hidden nodes that it typically takes to learn good hidden representations of images.
Learning a binary classifier with so many parameters might seem to require that we collect an enormous dataset,
perhaps comparable to the number of dogs and cats on the planet.
And yet both humans and computers are able to distinguish cats from dogs quite well, seemingly contradicting these conclusions.
That is because images exhibit rich structure that is typically exploited by humans and machine learning models alike.
-->

Độc giả kỹ tính có thể phản đối lập luận này trên cơ sở độ phân giải 1 triệu điểm ảnh có thể là không cần thiết.
Tuy nhiên, ngay cả khi chỉ sử dụng 100.000 điểm ảnh, ta đã đánh giá quá thấp số lượng các nút ẩn cần thiết để tìm các biểu diễn ẩn tốt của các ảnh.
Việc học một bộ phân loại nhị phân với rất nhiều tham số có thể sẽ cần tới một tập dữ liệu khổng lồ, có lẽ tương đương với số lượng chó và mèo trên hành tinh này.
Tuy nhiên, việc cả con người và máy tính đều có thể phân biệt mèo với chó khá tốt dường như mâu thuẫn với các kết luận trên.
Đó là bởi vì các ảnh thể hiện cấu trúc phong phú, thường được khai thác bởi con người và các mô hình học máy theo các cách giống nhau.

<!--
## Invariances
-->

## Tính Bất biến

<!--
Imagine that you want to detect an object in an image.
It seems reasonable that whatever method we use to recognize objects
should not be overly concerned with the precise *location* of the object in the image.
Ideally we could learn a system that would somehow exploit this knowledge.
Pigs usually do not fly and planes usually do not swim.
Nonetheless, we could still recognize a flying pig were one to appear.
This ideas is taken to an extreme in the children's game 'Where's Waldo', an example is shown in :numref:`img_waldo`.
The game consists of a number of chaotic scenes bursting with activity and Waldo shows up somewhere in each (typically lurking in some unlikely location).
The reader's goal is to locate him.
Despite his characteristic outfit, this can be surprisingly difficult, due to the large number of confounders.
-->

Hãy tưởng tượng rằng ta muốn nhận diện một vật thể trong ảnh.
Có vẻ sẽ hợp lý nếu cho rằng bất cứ phương pháp nào ta sử dụng đều không nên quá quan tâm đến vị trí *chính xác* của vật thể trong ảnh.
Lý tưởng nhất, ta có thể học một hệ thống có khả năng tận dụng được kiến thức này bằng một cách nào đó.
Lợn thường không bay và máy bay thường không bơi.
Tuy nhiên, ta vẫn có thể nhận ra một con lợn đang bay nếu nó xuất hiện.
Ý tưởng này được thể hiện rõ rệt trong trò chơi trẻ em 'Đi tìm Waldo', một ví dụ được miêu tả trong :numref:`img_waldo`.
Trò chơi này bao gồm một số cảnh hỗn loạn với nhiều hoạt động đan xen và Waldo xuất hiện ở đâu đó trong mỗi cảnh (thường ẩn nấp ở một số vị trí khó ngờ tới).
Nhiệm vụ của người chơi là xác định vị trí của anh ta.
Mặc dù Waldo có trang phục khá nổi bật, việc này có thể vẫn rất khó khăn do có quá nhiều yếu tố gây nhiễu.

<!--
![Image via Walker Books](../img/where-wally-walker-books.jpg)
-->

![Một ảnh trong Walker Books](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`


<!--
Back to images, the intuitions we have been discussing could be made more concrete yielding a few key principles for building neural networks for computer vision:
-->

Quay lại với ảnh, những trực giác mà ta đã thảo luận có thể được cụ thể hóa hơn nữa để thu được một vài nguyên tắc chính trong việc xây dựng mạng nơ-ron cho thị giác máy tính:

<!--
1. Our vision systems should, in some sense, respond similarly to the same object regardless of where it appears in the image (translation invariance).
2. Our visions systems should, in some sense, focus on local regions, without regard for what else is happening in the image at greater distances (locality).
-->

1. Ở một khía cạnh nào đó, các hệ thống thị giác nên phản ứng tương tự với cùng một vật thể bất kể vật thể đó xuất hiện ở đâu trong ảnh (tính bất biến tịnh tiến).
2. Ở khía cạnh khác, các hệ thống thị giác nên tập trung vào các khu vực cục bộ và không quan tâm đến bất kỳ thứ gì khác ở xa hơn trong ảnh (tính cục bộ).

<!--
Let us see how this translates into mathematics.
-->

Hãy cùng xem cách biểu diễn những điều trên bằng ngôn ngữ toán học.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Constraining the MLP
-->

## Ràng buộc Perceptron Đa tầng

<!-- In this exposition, we treat both images and hidden layers alike as two-dimensional arrays.
To start off let us consider what an MLP would look like with $h \times w$ images as inputs
(represented as matrices in math, and as 2D arrays in code),
and hidden representations similarly organized as $h \times w$ matrices / 2D arrays.
Let $x[i, j]$ and $h[i, j]$ denote pixel location $(i, j)$ in an image and hidden representation, respectively.
Consequently, to have each of the $h \times w$ hidden nodes receive input from each of the $h \times w$ inputs,
we would switch from using weight matrices (as we did previously in MLPs)
to representing our parameters as four-dimensional weight tensors.
-->

Trong phần này, ta coi hình ảnh và các tầng ẩn là các mảng hai chiều.
Để bắt đầu, hãy tưởng tượng một perceptron đa tầng sẽ như thế nào với đầu vào là ảnh kích thước $h \times w$ 
(biểu diễn dưới dạng ma trận trong toán học và mảng hai chiều khi lập trình),
và với các biểu diễn ẩn cũng là các ma trận / mảng hai chiều kích thước $h \times w$.
Đặt $x[i, j]$ và $h[i, j]$ lần lượt là điểm ảnh tại vị trí $(i, j)$ của ảnh và biểu diễn ẩn.
Để mỗi nút ẩn trong tổng số $h \times w$ nút nhận dữ liệu từ tất cả $h \times w$ đầu vào,
ta sẽ chuyển từ việc biểu diễn các tham số bằng ma trận trọng số (như đã thực hiện với perceptron đa tầng trước đây) sang sử dụng các tensor trọng số bốn chiều.

<!--
We could formally express this dense layer as follows:
-->

Ta có thể biểu diễn tầng kết nối đầy đủ bằng công thức toán sau:

<!--
$$h[i, j] = u[i, j] + \sum_{k, l} W[i, j, k, l] \cdot x[k, l] =  u[i, j] +
\sum_{a, b} V[i, j, a, b] \cdot x[i+a, j+b].$$
-->

$$h[i, j] = u[i, j] + \sum_{k, l} W[i, j, k, l] \cdot x[k, l] =  u[i, j] +
\sum_{a, b} V[i, j, a, b] \cdot x[i+a, j+b].$$

<!--
The switch from $W$ to $V$ is entirely cosmetic (for now) since there is a one-to-one correspondence between coefficients in both tensors.
We simply re-index the subscripts $(k, l)$ such that $k = i+a$ and $l = j+b$.
In other words, we set $V[i, j, a, b] = W[i, j, i+a, j+b]$.
The indices $a, b$ run over both positive and negative offsets, covering the entire image.
For any given location $(i, j)$ in the hidden layer $h[i, j]$, we compute its value by summing over pixels in $x$, centered around $(i, j)$ and weighted by $V[i, j, a, b]$.
-->

Việc chuyển từ $W$ sang $V$ hoàn toàn chỉ có mục đích thẩm mĩ (tại thời điểm này) bởi có một sự tương ứng một-một giữa các hệ số trong cả hai tensor.
Ta chỉ đơn thuần đặt lại các chỉ số dưới $(k, l)$ với $k = i+a$ và $l = j+b$.
Nói cách khác, $V[i, j, a, b] = W[i, j, i+a, j+b]$.
Các chỉ số $a, b$ chạy trên toàn bộ hình ảnh, có thể mang cả giá trị dương và âm.
Với bất kỳ vị trí $(i, j)$ nào ở tầng ẩn, giá trị biểu diễn ẩn $h[i, j]$ được tính bằng tổng trọng số của các điểm ảnh nằm xung quanh vị trí $(i, j)$ của $x$, với trọng số là $V[i, j, a, b]$.

<!--
Now let us invoke the first principle we established above: *translation invariance*.
This implies that a shift in the inputs $x$ should simply lead to a shift in the activations $h$.
This is only possible if $V$ and $u$ do not actually depend on $(i, j)$, i.e., we have $V[i, j, a, b] = V[a, b]$ and $u$ is a constant.
As a result we can simplify the definition for $h$.
-->

Bây giờ hãy sử dụng nguyên tắc đầu tiên mà ta đã thiết lập ở trên: *tính bất biến tịnh tiến*.
Nguyên tắc này ngụ ý rằng một sự dịch chuyển ở đầu vào $x$ cũng sẽ tạo ra sự dịch chuyển ở biểu diễn ẩn $h$.
Điều này chỉ có thể xảy ra nếu $V$ và $u$ không phụ thuộc vào $(i, j)$, tức $V[i, j, a, b] = V[a, b]$ và $u$ là một hằng số.
Vì vậy, ta có thể đơn giản hóa định nghĩa của $h$.

$$h[i, j] = u + \sum_{a, b} V[a, b] \cdot x[i+a, j+b].$$

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
This is a convolution!
We are effectively weighting pixels $(i+a, j+b)$ in the vicinity of $(i, j)$ with coefficients $V[a, b]$ to obtain the value $h[i, j]$.
Note that $V[a, b]$ needs many fewer coefficients than $V[i, j, a, b]$. For a 1 megapixel image it has at most 1 million coefficients.
This is 1 million fewer parameters since it no longer depends on the location within the image. We have made significant progress!
-->

Đây là một phép tích chập!
Ta đang đánh trọng số cho các điểm ảnh $(i+a, j+b)$ trong vùng lân cận của $(i, j)$ bằng các hệ số $V[a, b]$ để thu được giá trị $h[i, j]$.
Lưu ý rằng $V[a, b]$ cần ít hệ số hơn hẳn so với $V[i, j, a, b]$. Với đầu vào là hình ảnh 1 megapixel (với tối đa 1 triệu hệ số cho mỗi vị trí),
lượng tham số của $V[a, b]$ giảm đi 1 triệu vì không còn phụ thuộc vào vị trí trong ảnh. Ta đã có được tiến triển đáng kể!

<!--
Now let us invoke the second principle---*locality*.
As motivated above, we believe that we should not have to look very far away from $(i, j)$ in order to glean relevant information to assess what is going on at $h[i, j]$.
This means that outside some range $|a|, |b| > \Delta$, we should set $V[a, b] = 0$.
Equivalently, we can rewrite $h[i, j]$ as
-->

Bây giờ hãy sử dụng nguyên tắc thứ hai---*tính cục bộ*.
Như trình bày ở trên, giả sử rằng ta không cần thông tin tại các vị trí quá xa $(i, j)$ để đánh giá những gì đang diễn ra tại $h[i, j]$.
Điều này có nghĩa là ở các miền giá trị $|a|, |b| > \Delta$, ta có thể đặt $V[a, b] = 0$.
Tương tự, ta có thể đơn giản hoá $h[i, j]$ như sau

$$h[i, j] = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} V[a, b] \cdot x[i+a, j+b].$$

<!--
This, in a nutshell is the convolutional layer.
When the local region (also called a *receptive field*) is small, the difference as compared to a fully-connected network can be dramatic.
While previously, we might have required billions of parameters to represent just a single layer in an image-processing network, we now typically need just a few hundred.
The price that we pay for this drastic modification is that our features will be translation invariant and that our layer can only take local information into account.
All learning depends on imposing inductive bias.
When that bias agrees with reality, we get sample-efficient models that generalize well to unseen data.
But of course, if those biases do not agree with reality, e.g., if images turned out not to be translation invariant, our models may not generalize well.
-->

Một cách ngắn gọn, đây chính là biểu diễn toán học của tầng tích chập.
Khi vùng cục bộ xung quanh vị trí đang xét (còn được gọi là *vùng tiếp nhận*) nhỏ, sự khác biệt so với mạng kết nối đầy đủ có thể rất lớn.
Trước đây ta có thể phải cần hàng tỷ tham số để biểu diễn một tầng duy nhất trong mạng xử lý ảnh, hiện giờ chỉ cần vài trăm.
Cái giá phải trả là các đặc trưng sẽ trở nên bất biến tịnh tiến và các tầng chỉ có thể nhận thông tin cục bộ.
Toàn bộ quá trình học dựa trên việc áp đặt các thiên kiến quy nạp (*inductive bias*).
Khi các thiên kiến đó phù hợp với thực tế, ta sẽ có được các mô hình hoạt động hiệu quả với ít mẫu và khái quát tốt cho dữ liệu chưa gặp.
Nhưng tất nhiên, nếu những thiên kiến đó không phù hợp với thực tế, ví dụ như nếu các ảnh không có tính bất biến tịnh tiến, các mô hình có thể sẽ không khái quát tốt.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Convolutions
-->

## Phép Tích chập

<!--
Let us briefly review why the above operation is called a *convolution*.
In mathematics, the convolution between two functions,
say $f, g: \mathbb{R}^d \to R$ is defined as
-->

Hãy cùng xem qua lý do tại sao toán tử trên được gọi là *tích chập*.
Trong toán học, phép tích chập giữa hai hàm số $f, g: \mathbb{R}^d \to R$ được định nghĩa như sau

$$[f \circledast g](x) = \int_{\mathbb{R}^d} f(z) g(x-z) dz.$$

<!--
That is, we measure the overlap between $f$ and $g$ when both functions are shifted by $x$ and "flipped".
Whenever we have discrete objects, the integral turns into a sum.
For instance, for vectors defined on $\ell_2$, i.e., the set of square summable infinite dimensional vectors with index running over $\mathbb{Z}$ we obtain the following definition.
-->

Trong phép toán này, ta đo lường sự chồng chéo giữa $f$ và $g$ khi $g$ được dịch chuyển một khoảng $x$ và "bị lật lại".
Đối với các đối tượng rời rạc, phép tích phân trở thành phép lấy tổng.
Chẳng hạn, đối với các vector được định nghĩa trên $\ell_2$, là tập các vector vô hạn chiều có tổng bình phương hội tụ, với chỉ số chạy trên $\mathbb{Z}$, ta có phép tích chập sau:

$$[f \circledast g](i) = \sum_a f(a) g(i-a).$$

<!--
For two-dimensional arrays, we have a corresponding sum with indices $(i, j)$ for $f$ and $(i-a, j-b)$ for $g$ respectively.
This looks similar to definition above, with one major difference.
Rather than using $(i+a, j+b)$, we are using the difference instead.
Note, though, that this distinction is mostly cosmetic since we can always match the notation by using $\tilde{V}[a, b] = V[-a, -b]$ to obtain $h = x \circledast \tilde{V}$.
Also note that the original definition is actually a *cross correlation*.
We will come back to this in the following section.
-->

Đối với mảng hai chiều, ta có một tổng tương ứng với các chỉ số $(i, j)$ cho $f$ và $(i-a, j-b)$ cho $g$.
Tổng này nhìn gần giống với định nghĩa tầng tích chập ở trên, nhưng với một khác biệt lớn.
Thay vì $(i+a, j+b)$, ta lại sử dụng hiệu.
Tuy nhiên, lưu ý rằng sự khác biệt này không phải vấn đề lớn vì ta luôn có thể chuyển về ký hiệu của phép tích chập bằng cách sử dụng $\tilde{V}[a, b] = V[-a, -b]$ để có $h = x \circledast \tilde{V}$.
Cũng lưu ý rằng định nghĩa ban đầu thực ra là của phép toán *tương quan chéo*.
Ta sẽ quay trở lại phép toán này trong phần tiếp theo.

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Waldo Revisited
-->

## Xem lại ví dụ về Waldo

<!--
Let us see what this looks like if we want to build an improved Waldo detector.
The convolutional layer picks windows of a given size and weighs intensities according to the mask $V$, as demonstrated in :numref:`fig_waldo_mask`.
We expect that wherever the "waldoness" is highest, we will also find a peak in the hidden layer activations.
-->

Hãy cùng xem việc xây dựng một bộ phát hiện Waldo cải tiến sẽ trông như thế nào.
Tầng tích chập chọn các cửa sổ có kích thước cho sẵn và đánh trọng số cường độ dựa theo mặt nạ $V$, như được minh họa trong :numref:`fig_waldo_mask`.
Ta hy vọng rằng ở đâu có "tính Waldo" cao nhất, các tầng kích hoạt ẩn cũng sẽ có cao điểm ở đó.

<!--
![Find Waldo.](../img/waldo-mask.jpg)
-->

![Tìm Waldo.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

<!--
There is just a problem with this approach: so far we blissfully ignored that images consist of 3 channels: red, green and blue.
In reality, images are quite two-dimensional objects but rather as a $3^{\mathrm{rd}}$ order tensor, e.g., with shape $1024 \times 1024 \times 3$ pixels.
Only two of these axes concern spatial relationships, while the $3^{\mathrm{rd}}$ can be regarded as assigning a multidimensional representation *to each pixel location*.
-->

Chỉ có một vấn đề với cách tiếp cận này là cho đến nay ta đã vô tư bỏ qua việc hình ảnh bao gồm 3 kênh màu: đỏ, xanh lá cây và xanh dương.
Trong thực tế, hình ảnh không hẳn là các đối tượng hai chiều mà là một tensor bậc ba, ví dụ tensor với kích thước $1024 \times 1024 \times 3$ điểm ảnh.
Chỉ có hai trong số các trục này chứa mối quan hệ về mặt không gian, trong khi trục thứ ba có thể được coi như là một biểu diễn đa chiều *cho từng vị trí điểm ảnh*.

<!--
We thus index $\mathbf{x}$ as $x[i, j, k]$.
The convolutional mask has to adapt accordingly.
Instead of $V[a, b]$ we now have $V[a, b, c]$.
-->


Do đó, ta phải truy cập $\mathbf{x}$ dưới dạng $x[i, j, k]$.
Mặt nạ tích chập phải thích ứng cho phù hợp.
Thay vì $V[a, b]$ bây giờ ta có $V[a, b, c]$.

<!--
Moreover, just as our input consists of a $3^{\mathrm{rd}}$ order tensor it turns out to be a good idea to similarly formulate our hidden representations as $3^{\mathrm{rd}}$ order tensors.
In other words, rather than just having a 1D representation corresponding to each spatial location, we want to have a multidimensional hidden representations corresponding to each spatial location.
We could think of the hidden representation as comprising a number of 2D grids stacked on top of each other.
These are sometimes called *channels* or *feature maps*.
Intuitively, you might imagine that at lower layers, some channels could become specialized to recognize edges, others to recognize textures, etc. 
To support multiple channels in both inputs and hidden activations, we can add a fourth coordinate to $V: V[a, b, c, d]$. 
Putting all together we have:
-->
<!-- đoạn này trước bị thiếu nên mình update luôn -->

Hơn nữa, tương tự như việc đầu vào là các tensor bậc ba, việc xây dựng các biểu diễn ẩn là các tensor bậc ba tương ứng hoá ra cũng là một ý tưởng hay.
Nói cách khác, thay vì chỉ có một biểu diễn 1D tương ứng với từng vị trí không gian, ta muốn có một biểu diễn ẩn đa chiều tương ứng với từng vị trí không gian.
Ta có thể coi các biểu diễn ẩn như được cấu thành từ các lưới hai chiều xếp chồng lên nhau.
Đôi khi chúng được gọi là *kênh* (*channel*) hoặc *ánh xạ đặc trưng* (*feature map*).
Theo trực giác, bạn có thể tưởng tượng rằng ở các tầng thấp hơn, một số kênh tập trung vào việc nhận diện cạnh trong khi các kênh khác đảm nhiệm việc nhận diện kết cấu, v.v.
Để hỗ trợ đa kênh ở cả đầu vào và kích hoạt ẩn, ta có thể thêm tọa độ thứ tư vào $V: V[a, b, c, d]$. 
Từ mọi điều trên, ta có:

$$h[i, j, k] = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c V[a, b, c, k] \cdot x[i+a, j+b, c].$$

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
This is the definition of a convolutional neural network layer.
There are still many operations that we need to address.
For instance, we need to figure out how to combine all the activations to a single output (e.g., whether there is a Waldo in the image).
We also need to decide how to compute things efficiently, how to combine multiple layers, and whether it is a good idea to have many narrow or a few wide layers.
All of this will be addressed in the remainder of the chapter.
-->

Đây là định nghĩa của một tầng mạng nơ-ron tích chập.
Vẫn còn nhiều phép toán mà ta cần phải giải quyết.
Chẳng hạn, ta cần tìm ra cách kết hợp tất cả các giá trị kích hoạt thành một đầu ra duy nhất (ví dụ đầu ra cho: có Waldo trong ảnh không).
Ta cũng cần quyết định cách tính toán mọi thứ một cách hiệu quả, cách kết hợp các tầng với nhau và liệu có nên sử dụng thật nhiều tầng hẹp hay chỉ một vài tầng rộng.
Tất cả những điều này sẽ được giải quyết trong phần còn lại của chương.

<!--
## Summary
-->

## Tóm tắt

<!--
* Translation invariance in images implies that all patches of an image will be treated in the same manner.
* Locality means that only a small neighborhood of pixels will be used for computation.
* Channels on input and output allows for meaningful feature analysis.
-->


* Tính bất biến tịnh tiến của hình ảnh ngụ ý rằng tất cả các mảng nhỏ trong một tấm ảnh đều được xử lý theo cùng một cách.
* Tính cục bộ có nghĩa là chỉ một vùng lân cận nhỏ các điểm ảnh sẽ được sử dụng cho việc tính toán.
* Các kênh ở đầu vào và đầu ra cho phép việc phân tích các đặc trưng trở nên ý nghĩa hơn.

<!--
## Exercises
-->

## Bài tập

<!--
1. Assume that the size of the convolution mask is $\Delta = 0$. Show that in this case the convolutional mask implements an MLP independently for each set of channels.
2. Why might translation invariance not be a good idea after all? Does it make sense for pigs to fly?
3. What happens at the boundary of an image?
4. Derive an analogous convolutional layer for audio.
5. What goes wrong when you apply the above reasoning to text? Hint: what is the structure of language?
6. Prove that $f \circledast g = g \circledast f$.
-->

1. Giả sử rằng kích thước của mặt nạ tích chập có $\Delta = 0$. Chứng minh rằng trong trường hợp này, mặt nạ tích chập xây dựng một MLP độc lập cho mỗi một tập kênh.
2. Tại sao tính bất biến tịnh tiến có thể không phải là một ý tưởng tốt? Việc lợn biết bay là có hợp lý không?
3. Điều gì xảy ra ở viền của một tấm ảnh?
4. Hãy suy ra một tầng tích chập tương tự cho âm thanh.
5. Vấn đề gì sẽ xảy ra khi áp dụng các suy luận trên cho văn bản? Gợi ý: cấu trúc của ngôn ngữ là gì?
6. Chứng minh rằng $f \circledast g = g \circledast f$.

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2348)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Duy Du
* Trần Yến Thy
* Lê Khắc Hồng Phúc
* Phạm Minh Đức
* Phạm Hồng Vinh
* Nguyễn Văn Cường

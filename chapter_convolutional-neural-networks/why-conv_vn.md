<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# From Dense Layers to Convolutions
-->

# Từ Tầng Dày Đặc tới Phép Tích Chập

<!--
The models that we have discussed so far are fine options if you are dealing with *tabular* data.
By *tabular* we mean that the data consists of rows corresponding to examples and columns corresponding to features.
With tabular data, we might anticipate that pattern we seek could require modeling interactions among the features, 
but do not assume anything a priori about which features are related to each other or in what way.
-->

Cho đến giờ các mô hình mà ta đã thảo luận là các lựa chọn phù hợp nếu dữ liệu mà ta đang xử lý có *dạng bảng* với các hàng tương ứng với các mẫu còn các cột tương ứng với các đặc trưng.
Với dữ liệu có dạng như vậy, ta có thể dự đoán rằng khuôn mẫu mà ta đang tìm kiếm có thể yêu cầu mô hình hóa các tương tác giữa các đặc trưng, nhưng ta không giả định từ kinh nghiệm bất cứ điều gì về việc các đặc trưng có liên quan tới nhau như thế nào.


<!--
Sometimes we truly may not have any knowledge to guide the construction of more cleverly-organized architectures.
In these cases, a multilayer perceptron is often the best that we can do.
However, once we start dealing with high-dimensional perceptual data, these *structure-less* networks can grow unwieldy.
-->

Đôi khi ta thực sự không có bất kỳ kiến thức nào để hướng dẫn việc xây dựng các kiến trúc được sắp xếp khéo léo hơn. <!-- reviewer xem có cách dịch nào hay hơn cho cụm "more cleverly-organized architectures" không -->
Trong những trường hợp này, sử dụng một perceptron đa tầng thường là giải pháp tốt nhất ta có thể làm.
Tuy nhiên, một khi ta bắt đầu xử lý dữ liệu nhận thức nhiều chiều, các mạng *không có cấu trúc* này có thể sẽ trở nên quá cồng kềnh.

<!--
For instance, let us return to our running example of distinguishing cats from dogs.
Say that we do a thorough job in data collection, collecting an annotated sets of high-quality 1-megapixel photographs.
This means that the input into a network has *1 million dimensions*.
Even an aggressive reduction to *1,000 hidden dimensions* would require a *dense* (fully-connected) layer to support $10^9$ parameters.
Unless we have an extremely large dataset (perhaps billions?), lots of GPUs, a talent for extreme distributed optimization, and an extraordinary amount of patience,
learning the parameters of this network may turn out to be impossible.
-->

Hãy quay trở lại với ví dụ phân biệt chó và mèo quen thuộc.
Giả sử ta thực hiện việc thu thập dữ liệu một cách kỹ lưỡng và thu được một bộ ảnh được gán nhãn có độ phân giải 1 triệu điểm ảnh.
Điều này có nghĩa là đầu vào của mạng sẽ có *1 triệu chiều*.
Ngay cả việc giảm mạnh xuống *1000 chiều ẩn* sẽ cần tới một tầng *dày đặc* (kết nối đầy đủ) để hỗ trợ $10^9$ tham số.
Trừ khi ta có một bộ dữ liệu cực lớn (có thể là hàng tỷ ảnh?), một số lượng lớn GPU, một tài năng để tối ưu hóa phân tán và sức kiên nhẫn phi thường, thì việc học các tham số của mạng này có thể sẽ là điều bất khả thi.

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

*dịch đoạn phía trên*

<!--
## Invariances
-->

## *dịch tiêu đề phía trên*

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

*dịch đoạn phía trên*

<!--
![Image via Walker Books](../img/where-wally-walker-books.jpg)
-->

![*dịch chú thích ảnh phía trên*](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`


<!--
Back to images, the intuitions we have been discussing could be made more concrete yielding a few key principles for building neural networks for computer vision:
-->

*dịch đoạn phía trên*

<!--
1. Our vision systems should, in some sense, respond similarly to the same object regardless of where it appears in the image (translation invariance).
2. Our visions systems should, in some sense, focus on local regions, without regard for what else is happening in the image at greater distances (locality).
-->

*dịch đoạn phía trên*

<!--
Let us see how this translates into mathematics.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Constraining the MLP
-->

## *dịch tiêu đề phía trên*

<!-- In this exposition, we treat both images and hidden layers alike as two-dimensional arrays.
To start off let us consider what an MLP would look like with $h \times w$ images as inputs
(represented as matrices in math, and as 2D arrays in code),
and hidden representations similarly organized as $h \times w$ matrices / 2D arrays.
Let $x[i, j]$ and $h[i, j]$ denote pixel location $(i, j)$ in an image and hidden representation, respectively.
Consequently, to have each of the $hw$ hidden nodes receive input from each of the $hw$ inputs,
we would switch from using weight matrices (as we did previously in MLPs)
to representing our parameters as four-dimensional weight tensors.
-->

*dịch đoạn phía trên*


<!--
We could formally express this dense layer as follows:
-->

*dịch đoạn phía trên*

$$h[i, j] = u[i, j] + \sum_{k, l} W[i, j, k, l] \cdot x[k, l] =  u[i, j] +
\sum_{a, b} V[i, j, a, b] \cdot x[i+a, j+b].$$
-->

*dịch đoạn phía trên*

<!--
The switch from $W$ to $V$ is entirely cosmetic (for now) since there is a one-to-one correspondence between coefficients in both tensors.
We simply re-index the subscripts $(k, l)$ such that $k = i+a$ and $l = j+b$.
In other words, we set $V[i, j, a, b] = W[i, j, i+a, j+b]$.
The indices $a, b$ run over both positive and negative offsets, covering the entire image.
For any given location $(i, j)$ in the hidden layer $h[i, j]$, we compute its value by summing over pixels in $x$, centered around $(i, j)$ and weighted by $V[i, j, a, b]$.
-->

*dịch đoạn phía trên*

<!--
Now let us invoke the first principle we established above: *translation invariance*.
This implies that a shift in the inputs $x$ should simply lead to a shift in the activations $h$.
This is only possible if $V$ and $u$ do not actually depend on $(i, j)$, i.e., we have $V[i, j, a, b] = V[a, b]$ and $u$ is a constant.
As a result we can simplify the definition for $h$.
-->

*dịch đoạn phía trên*

$$h[i, j] = u + \sum_{a, b} V[a, b] \cdot x[i+a, j+b].$$

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
This is a convolution!
We are effectively weighting pixels $(i+a, j+b)$ in the vicinity of $(i, j)$ with coefficients $V[a, b]$ to obtain the value $h[i, j]$.
Note that $V[a, b]$ needs many fewer coefficients than $V[i, j, a, b]$. For a 1 megapixel image it has at most 1 million coefficients.
This is 1 million fewer parameters since it no longer depends on the location within the image. We have made significant progress!
-->

*dịch đoạn phía trên*

<!--
Now let us invoke the second principle---*locality*.
As motivated above, we believe that we should not have to look very far away from $(i, j)$ in order to glean relevant information to assess what is going on at $h[i, j]$.
This means that outside some range $|a|, |b| > \Delta$, we should set $V[a, b] = 0$.
Equivalently, we can rewrite $h[i, j]$ as
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Convolutions
-->

## *dịch tiêu đề phía trên*

<!--
Let us briefly review why the above operation is called a *convolution*.
In mathematics, the convolution between two functions,
say $f, g: \mathbb{R}^d \to R$ is defined as
-->

*dịch đoạn phía trên*

$$[f \circledast g](x) = \int_{\mathbb{R}^d} f(z) g(x-z) dz.$$

<!--
That is, we measure the overlap between $f$ and $g$ when both functions are shifted by $x$ and "flipped".
Whenever we have discrete objects, the integral turns into a sum.
For instance, for vectors defined on $\ell_2$, i.e., the set of square summable infinite dimensional vectors with index running over $\mathbb{Z}$ we obtain the following definition.
-->

*dịch đoạn phía trên*

$$[f \circledast g](i) = \sum_a f(a) g(i-a).$$

<!--
For two-dimensional arrays, we have a corresponding sum with indices $(i, j)$ for $f$ and $(i-a, j-b)$ for $g$ respectively.
This looks similar to definition above, with one major difference.
Rather than using $(i+a, j+b)$, we are using the difference instead.
Note, though, that this distinction is mostly cosmetic since we can always match the notation by using $\tilde{V}[a, b] = V[-a, -b]$ to obtain $h = x \circledast \tilde{V}$.
Also note that the original definition is actually a *cross correlation*.
We will come back to this in the following section.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Waldo Revisited
-->

## *dịch tiêu đề phía trên*

<!--
Let us see what this looks like if we want to build an improved Waldo detector.
The convolutional layer picks windows of a given size and weighs intensities according to the mask $V$, as demonstrated in :numref:`fig_waldo_mask`.
We expect that wherever the "waldoness" is highest, we will also find a peak in the hidden layer activations.
-->

*dịch đoạn phía trên*

<!--
![Find Waldo.](../img/waldo-mask.jpg)
-->

![*dịch chú thích ảnh phía trên*](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

<!--
There is just a problem with this approach: so far we blissfully ignored that images consist of 3 channels: red, green and blue.
In reality, images are quite two-dimensional objects but rather as a $3^{\mathrm{rd}}$ order tensor, e.g., with shape $1024 \times 1024 \times 3$ pixels.
Only two of these axes concern spatial relationships, while the $3^{\mathrm{rd}}$ can be regarded as assigning a multidimensional representation *to each pixel location*.
-->

*dịch đoạn phía trên*

<!--
We thus index $\mathbf{x}$ as $x[i, j, k]$.
The convolutional mask has to adapt accordingly.
Instead of $V[a, b]$ we now have $V[a, b, c]$.
-->

*dịch đoạn phía trên*

<!--
Moreover, just as our input consists of a $3^{\mathrm{rd}}$ order tensor it turns out to be a good idea to similarly formulate our hidden representations as $3^{\mathrm{rd}}$ order tensors.
In other words, rather than just having a 1D representation corresponding to each spatial location, we want to have a multidimensional hidden representations corresponding to each spatial location.
We could think of the hidden representation as comprising a number of 2D grids stacked on top of each other.
These are sometimes called *channels* or *feature maps*.
Intuitively you might imagine that at lower layers, some channels specialize to recognizing edges,
We can take care of this by adding a fourth coordinate to $V$ via $V[a, b, c, d]$. Putting all together we have:
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*


<!--
## Summary
-->

## Tóm tắt

<!--
* Translation invariance in images implies that all patches of an image will be treated in the same manner.
* Locality means that only a small neighborhood of pixels will be used for computation.
* Channels on input and output allows for meaningful feature analysis.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->
<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2348)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2348)
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
*

<!-- Phần 2 -->
*

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

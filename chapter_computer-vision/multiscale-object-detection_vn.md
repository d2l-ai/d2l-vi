<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Multiscale Object Detection
-->

# *dịch tiêu đề phía trên*


<!--
In :numref:`sec_anchor`, we generated multiple anchor boxes centered on each pixel of the input image.
These anchor boxes are used to sample different regions of the input image.
However, if anchor boxes are generated centered on each pixel of the image, soon there will be too many anchor boxes for us to compute.
For example, we assume that the input image has a height and a width of 561 and 728 pixels respectively.
If five different shapes of anchor boxes are generated centered on each pixel, over two million anchor boxes ($561 \times 728 \times 5$) need to be predicted and labeled on the image.
-->

*dịch đoạn phía trên*


<!--
It is not difficult to reduce the number of anchor boxes.
An easy way is to apply uniform sampling on a small portion of pixels from the input image and generate anchor boxes centered on the sampled pixels.
In addition, we can generate anchor boxes of varied numbers and sizes on multiple scales.
Notice that smaller objects are more likely to be positioned on the image than larger ones.
Here, we will use a simple example: Objects with shapes of $1 \times 1$, $1 \times 2$, and $2 \times 2$ may have 4, 2, and 1 possible position(s) on an image with the shape $2 \times 2$.
Therefore, when using smaller anchor boxes to detect smaller objects, we can sample more regions; when using larger anchor boxes to detect larger objects, we can sample fewer regions.
-->

*dịch đoạn phía trên*


<!--
To demonstrate how to generate anchor boxes on multiple scales, let us read an image first.
It has a height and width of $561 \times 728$ pixels.
-->

*dịch đoạn phía trên*



```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[0:2]
h, w
```


<!--
In :numref:`sec_conv_layer`, the 2D array output of the convolutional neural network (CNN) is called a feature map.
We can determine the midpoints of anchor boxes uniformly sampled on any image by defining the shape of the feature map.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->


<!--
The function `display_anchors` is defined below.
We are going to generate anchor boxes `anchors` centered on each unit (pixel) on the feature map `fmap`.
Since the coordinates of axes $x$ and $y$ in anchor boxes `anchors` have been divided by the width and height of the feature map `fmap`, 
values between 0 and 1 can be used to represent relative positions of anchor boxes in the feature map.
Since the midpoints of anchor boxes `anchors` overlap with all the units on feature map `fmap`, 
the relative spatial positions of the midpoints of the `anchors` on any image must have a uniform distribution.
Specifically, when the width and height of the feature map are set to `fmap_w` and `fmap_h` respectively, 
the function will conduct uniform sampling for `fmap_h` rows and `fmap_w` columns of pixels and use them as midpoints 
to generate anchor boxes with size `s` (we assume that the length of list `s` is 1) and different aspect ratios (`ratios`).
-->

Hàm `display_anchors` được định nghĩa như ở dưới.
Ta tạo các khung neo `anchors` có tâm được đặt theo từng đơn vị (điểm ảnh) trong tập ánh xạ đặc trưng `fmap`.
Do các toạ độ $x$ và $y$ trong các khung neo `anchors` đã được chia cho chiều rộng và chiều cao của tập ánh xạ đặc trưng `fmap`,
ta sử dụng các giá trị trong khoảng từ 0 đến 1 để biểu diễn vị trí tương đối của các khung neo trong tập ánh xạ đặc trưng.
Do tâm điểm của các khung neo `anchors` trùng với tất cả các đơn vị của tập ánh xạ đặc trưng `fmap`,
vị trí tương đối trong không gian của tâm điểm của `anchors` trên bất kì ảnh nào bắt buộc phhải tuân theo một phân phối đều.
Cụ thể, khi chiều rộng và chiều cao của một ánh xạ đặc trưng lần lượt được đặt là `fmap_w` và `fmap_h`,
hàm này sẽ tạo ra mẫu phân phối đều cho các hàng `fmap_h` và các cột `fmap_w` cho các điểm ảnh và sử dụng chúng làm tâm điểm
để sinh các khung neo với kích thước `s` (ta giả sử rằng độ dài của mảng `s` là 1) và tỉ lệ các cạnh khác (`ratios`).


```{.python .input  n=2}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # The values from the first two dimensions will not affect the output
    fmap = np.zeros((1, 10, fmap_w, fmap_h))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```


<!--
We will first focus on the detection of small objects. In order to make it easier to distinguish upon display, the anchor boxes with different midpoints here do not overlap.
We assume that the size of the anchor boxes is 0.15 and the height and width of the feature map are 4.
We can see that the midpoints of anchor boxes from the 4 rows and 4 columns on the image are uniformly distributed.
-->

Đầu tiên ta tập trung vào việc phát hiện các đối tượng nhỏ. Để dễ dàng phân biệt trong lúc hiển thị, các khung neo với các tâm điểm khác nhau ở ví dụ này sẽ không chồng chéo lên nhau.
Ta giả sử rằng kích thước của các khung neo là 0.15 và chiều cao và chiều rộng của tập ánh xạ đặc trưng đều bằng 4.
Ta có thể thấy rằng tâm điểm của các khung neo từ 4 hàng và 4 cột trong ảnh tuân theo phân phối đều.


```{.python .input  n=3}
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```


<!--
We are going to reduce the height and width of the feature map by half and use a larger anchor box to detect larger objects.
When the size is set to 0.4, overlaps will occur between regions of some anchor boxes.
-->

Ta giảm chiều cao và chiều rộng của tập ánh xạ đặc trưng đi một nửa và sử dụng khung neo lớn hơn để phát hiện đối tượng kích thước lớn hơn.
Khi kích thước được đặt bằng 0.4, hiện tượng chồng chéo giữa một số khung neo giữa các vùng sẽ xảy ra.



```{.python .input  n=4}
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```


<!--
Finally, we are going to reduce the height and width of the feature map by half and increase the anchor box size to 0.8.
Now the midpoint of the anchor box is the center of the image.
-->

Cuối cùng, ta tiếp tục giảm chiều cao và chiều rộng của tập ánh xạ đặc trưng đi một nửa và tăng kích thước khung neo lên 0.8.
Giờ tâm điểm của khung neo chính là tâm của ảnh.



```{.python .input  n=5}
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```


<!--
Since we have generated anchor boxes of different sizes on multiple scales, we will use them to detect objects of various sizes at different scales.
Now we are going to introduce a method based on convolutional neural networks (CNNs).
-->

Do ta sinh các khung neo với kích thước khác nhau trên nhiều tỉ lệ khác nhau, ta sẽ sử dụng chúng để phát hiện các vật thể với kích cỡ đa dạng trên nhiều tỉ lệ khác nhau.
Bây giờ chúng tôi sẽ giới thiệu một phương pháp dựa vào mạng nơ-ron tích chập (CNNs).


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->


<!--
At a certain scale, suppose we generate $h \times w$ sets of anchor boxes with different midpoints based on $c_i$ feature maps 
with the shape $h \times w$ and the number of anchor boxes in each set is $a$.
For example, for the first scale of the experiment, we generate 16 sets of anchor boxes with 
different midpoints based on 10 (number of channels) feature maps with a shape of $4 \times 4$, and each set contains 3 anchor boxes.
Next, each anchor box is labeled with a category and offset based on the classification and position of the ground-truth bounding box.
At the current scale, the object detection model needs to predict the category and offset of $h \times w$ sets of anchor boxes with different midpoints based on the input image.
-->

*dịch đoạn phía trên*


<!--
We assume that the $c_i$ feature maps are the intermediate output of the CNN based on the input image.
Since each feature map has $h \times w$ different spatial positions, the same position will have $c_i$ units.
According to the definition of receptive field in the :numref:`sec_conv_layer`, the $c_i$ units of the feature map at the same spatial position have the same receptive field on the input image.
Thus, they represent the information of the input image in this same receptive field.
Therefore, we can transform the $c_i$ units of the feature map at the same spatial position into the categories and offsets of the $a$ anchor boxes generated using that position as a midpoint.
It is not hard to see that, in essence, we use the information of the input image in a certain receptive field to predict the category and offset of the anchor boxes close to the field on the input image.
-->

*dịch đoạn phía trên*


<!--
When the feature maps of different layers have receptive fields of different sizes on the input image, they are used to detect objects of different sizes.
For example, we can design a network to have a wider receptive field for each unit in the feature map that is closer to the output layer, to detect objects with larger sizes in the input image.
-->

*dịch đoạn phía trên*


<!--
We will implement a multiscale object detection model in the following section.
-->

*dịch đoạn phía trên*



## Tóm tắt


<!--
* We can generate anchor boxes with different numbers and sizes on multiple scales to detect objects of different sizes on multiple scales.
* The shape of the feature map can be used to determine the midpoint of the anchor boxes that uniformly sample any image.
* We use the information for the input image from a certain receptive field to predict the category and offset of the anchor boxes close to that field on the image.
-->

*dịch đoạn phía trên*


## Bài tập


<!--
Given an input image, assume $1 \times c_i \times h \times w$ to be the shape of the feature map while $c_i, h, w$ are the number, height, and width of the feature map.
What methods can you think of to convert this variable into the anchor box's category and offset? What is the shape of the output?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/371)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* 

<!-- Phần 2 -->
* Đỗ Trường Giang

<!-- Phần 3 -->
* 



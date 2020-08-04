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

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


```{.python .input  n=3}
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```


<!--
We are going to reduce the height and width of the feature map by half and use a larger anchor box to detect larger objects.
When the size is set to 0.4, overlaps will occur between regions of some anchor boxes.
-->

*dịch đoạn phía trên*



```{.python .input  n=4}
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```


<!--
Finally, we are going to reduce the height and width of the feature map by half and increase the anchor box size to 0.8.
Now the midpoint of the anchor box is the center of the image.
-->

*dịch đoạn phía trên*



```{.python .input  n=5}
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```


<!--
Since we have generated anchor boxes of different sizes on multiple scales, we will use them to detect objects of various sizes at different scales.
Now we are going to introduce a method based on convolutional neural networks (CNNs).
-->

*dịch đoạn phía trên*


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

Ở một tỉ lệ nhất định, giả sử rằng ta sinh $h \times w$ tập hợp khung neo với các tâm điểm khác nhau dựa vào $c_i$ ánh xạ đặc trưng
có kích thước $h \times w$ và số khung neo của mỗi tập hợp là $a$.
Ví dụ, đối với tỉ lệ đầu tiên trong thí nghiệm này, ta sinh 16 tập hợp khung neo với
các tâm điểm khác nhau dựa vào 10 (số kênh) ánh xạ đặc trưng có kích thước $4 \times 4$, và mỗi tập hợp bao gồm 3 khung neo.
Tiếp theo, mỗi khung neo được gán nhãn bằng một danh mục và độ dời dựa vào quá trình phân loại và vị trí của khung chứa chuẩn.
Với tỉ lệ hiện tại, mô hình phát hiện vật thể cần phải dự đoán danh mục và độ dời của $h \times w$ tập hợp khung neo với các tâm điểm khác nhau dựa vào ảnh đầu vào.


<!--
We assume that the $c_i$ feature maps are the intermediate output of the CNN based on the input image.
Since each feature map has $h \times w$ different spatial positions, the same position will have $c_i$ units.
According to the definition of receptive field in the :numref:`sec_conv_layer`, the $c_i$ units of the feature map at the same spatial position have the same receptive field on the input image.
Thus, they represent the information of the input image in this same receptive field.
Therefore, we can transform the $c_i$ units of the feature map at the same spatial position into the categories and offsets of the $a$ anchor boxes generated using that position as a midpoint.
It is not hard to see that, in essence, we use the information of the input image in a certain receptive field to predict the category and offset of the anchor boxes close to the field on the input image.
-->

Ta giả sử rằng các ánh xạ đặc trưng $c_i$ là đầu ra trung gian của CNN dựa trên ảnh đầu vào.
Do mỗi ánh xạ đặc trưng có $h \times w$ vị trí khác nhau trong không gian, một vị trí sẽ có $c_i$ đơn vị.
Theo định nghĩa của vùng tiếp nhận trong :numref:`sec_conv_layer`, các đơn vị $c_i$ của ánh xạ đặc trưng ở cùng một vị trí trong không gian sẽ có cùng vùng tiếp nhận trên ảnh đầu vào.
Do đó, chúng biểu diễn thông tin của ảnh đầu vào trên cùng vùng tiếp nhận đó.
Bởi vậy, ta có thể biến đổi các đơn vị $c_i$ của ánh xạ đặc trưng có cùng vị trí trong không gian thanh các danh mục và độ dời cho khung neo $a$ được sinh ra bằng cách sử dụng vị trí đó làm tâm điểm.
Không khó khăn để nhận ra rằng, về bản chất, ta sử dụng thông tin của ảnh đầu vào trong một vùng tiếp nhận nhất định để dự đoán danh mục và độ dời của khung neo gần với vùng đó trên ảnh đầu vào.


<!--
When the feature maps of different layers have receptive fields of different sizes on the input image, they are used to detect objects of different sizes.
For example, we can design a network to have a wider receptive field for each unit in the feature map that is closer to the output layer, to detect objects with larger sizes in the input image.
-->

Khi các ánh xạ đặc trưng của các tầng khác nhau có các vùng tiếp nhận với kích thước khác nhau trong ảnh đầu vào, chúng được sử dụng để phát hiện vật thể với kích thước khác nhau.
Ví dụ, ta có thể thiết kế một mạng để cho mỗi đơn vị trong ánh xạ đặc trưng có vùng tiếp nhận rộng hơn, tức là gần với tầng đầu ra hơn, để phát hiện các vật thể với kích thước lớn hơn trong ảnh đầu vào.


<!--
We will implement a multiscale object detection model in the following section.
-->

Ta sẽ tiến hành lập trình mô hình phát hiện vật thể đa tỉ lệ trong phần kế tiếp.



## Tóm tắt


<!--
* We can generate anchor boxes with different numbers and sizes on multiple scales to detect objects of different sizes on multiple scales.
* The shape of the feature map can be used to determine the midpoint of the anchor boxes that uniformly sample any image.
* We use the information for the input image from a certain receptive field to predict the category and offset of the anchor boxes close to that field on the image.
-->

* Ta có thể sinh các khung neo với số lượng và kích thước khác nhau trên nhiều tỉ lệ để phát hiện vật thể có kích thước khác nhau trên nhiều tỉ lệ.
* Kích thước của ánh xạ đặc trưng có thể được sử dụng để xác định tâm điểm của các khung neo được lấy mẫu đều trên bất kì ảnh nào.
* Ta sử dụng thông tin của ảnh đầu vào từ một vùng tiếp nhận nhất định để dự đoán danh mục và độ dời của các khung neo gần với vùng đó trên ảnh.


## Bài tập


<!--
Given an input image, assume $1 \times c_i \times h \times w$ to be the shape of the feature map while $c_i, h, w$ are the number, height, and width of the feature map.
What methods can you think of to convert this variable into the anchor box's category and offset? What is the shape of the output?
-->

Cho một ảnh đầu vào, giả sử $1 \times c_i \times h \times w$ là kích thước của ánh xạ đặc trưng trong khi $c_i, h, w$ là số lượng, chiều cao và chiều dài của ánh xạ đặc trưng.
Bạn hãy nghĩ liệu có phương pháp nào chuyển đổi biến này thành danh mục và độ dời của một khung neo? Kích thước của đầu ra là gì?

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
* 

<!-- Phần 3 -->
* Đỗ Trường Giang



<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Region-based CNNs (R-CNNs)
-->

# *dịch tiêu đề phía trên*


<!--
Region-based convolutional neural networks or regions with CNN features (R-CNNs) are a pioneering approach that applies deep models to object detection :cite:`Girshick.Donahue.Darrell.ea.2014`.
In this section, we will discuss R-CNNs and a series of improvements made to them: Fast R-CNN :cite:`Girshick.2015`, 
Faster R-CNN :cite:`Ren.He.Girshick.ea.2015`, and Mask R-CNN :cite:`He.Gkioxari.Dollar.ea.2017`.
Due to space limitations, we will confine our discussion to the designs of these models.
-->

*dịch đoạn phía trên*


<!--
## R-CNNs
-->

## *dịch tiêu đề phía trên*


<!--
R-CNN models first select several proposed regions from an image (for example, anchor boxes are one type of selection method) and then label their categories and bounding boxes (e.g., offsets).
Then, they use a CNN to perform forward computation to extract features from each proposed area.
Afterwards, we use the features of each proposed region to predict their categories and bounding boxes.
:numref:`fig_r-cnn` shows an R-CNN model.
-->

*dịch đoạn phía trên*


<!--
![R-CNN model.](../img/r-cnn.svg)
-->

![*dịch mô tả phía trên*](../img/r-cnn.svg)
:label:`fig_r-cnn`


<!--
Specifically, R-CNNs are composed of four main parts:
-->

*dịch đoạn phía trên*


<!--
1. Selective search is performed on the input image to select multiple high-quality proposed regions :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013`.
These proposed regions are generally selected on multiple scales and have different shapes and sizes.
The category and ground-truth bounding box of each proposed region is labeled.
2. A pre-trained CNN is selected and placed, in truncated form, before the output layer.
It transforms each proposed region into the input dimensions required by the network and uses forward computation to output the features extracted from the proposed regions.
3. The features and labeled category of each proposed region are combined as an example to train multiple support vector machines for object classification.
Here, each support vector machine is used to determine whether an example belongs to a certain category.
4. The features and labeled bounding box of each proposed region are combined as an example to train a linear regression model for ground-truth bounding box prediction.
-->

*dịch đoạn phía trên*


<!--
Although R-CNN models use pre-trained CNNs to effectively extract image features, the main downside is the slow speed.
As you can imagine, we can select thousands of proposed regions from a single image, requiring thousands of forward computations from the CNN to perform object detection.
This massive computing load means that R-CNNs are not widely used in actual applications.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Fast R-CNN
-->

# Mạng Fast R-CNN


<!--
The main performance bottleneck of an R-CNN model is the need to independently extract features for each proposed region.
As these regions have a high degree of overlap, independent feature extraction results in a high volume of repetitive computations.
Fast R-CNN improves on the R-CNN by only performing CNN forward computation on the image as a whole.
-->

Điểm nghẽn cổ chai chính yếu về hiệu năng của mô hình R-CNN đó là việc trích xuất đặc trưng cho từng vùng đề xuất một cách độc lập.
Do các vùng đề xuất này có độ chồng lặp cao, nên việc trích xuất đặc trưng một cách độc lập sẽ dẫn đến khối lượng lớn các phép tính lặp lại.
Fast R-CNN cải thiện mô hình R-CNN chỉ bằng cách thực hiện tính toán truyền xuôi qua mạng CNN trên toàn bộ ảnh.


<!--
![Fast R-CNN model.](../img/fast-rcnn.svg)
-->

![Mô hình Fast R-CNN.](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`


<!--
:numref:`fig_fast_r-cnn` shows a Fast R-CNN model.
It is primary computation steps are described below:
-->

:numref:`fig_fast_r-cnn` mô tả mạng Fast R-CNN.
Các bước tính toán chính yếu được mô tả như sau:


<!--
1. Compared to an R-CNN model, a Fast R-CNN model uses the entire image as the CNN input for feature extraction, rather than each proposed region.
Moreover, this network is generally trained to update the model parameters.
As the input is an entire image, the CNN output shape is $1 \times c \times h_1 \times w_1$.
2. Assuming selective search generates $n$ proposed regions, their different shapes indicate regions of interests (RoIs) of different shapes on the CNN output.
Features of the same shapes must be extracted from these RoIs (here we assume that the height is $h_2$ and the width is $w_2$).
Fast R-CNN introduces RoI pooling, which uses the CNN output and RoIs as input to output a concatenation 
of the features extracted from each proposed region with the shape $n \times c \times h_2 \times w_2$.
3. A fully connected layer is used to transform the output shape to $n \times d$, where $d$ is determined by the model design.
4. During category prediction, the shape of the fully connected layer output is again transformed to $n \times q$ and we use softmax regression ($q$ is the number of categories).
During bounding box prediction, the shape of the fully connected layer output is again transformed to $n \times 4$.
This means that we predict the category and bounding box for each proposed region.
-->

1. So với mạng R-CNN, mạng Fast R-CNN sử dụng toàn bộ ảnh là đầu vào cho CNN để trích xuất đặc trưng thay vì từng vùng đề xuất.
Hơn nữa, mạng này được huấn luyện chung cho toàn dữ liệu để cập nhật tham số mô hình.
Do đầu vào là toàn bộ ảnh, đầu ra của mạng CNN có kích thước $1 \times c \times h_1 \times w_1$.
2. Giả sử thuật toán tìm kiếm lựa chọn sinh $n$ vùng đề xuất, mỗi vùng có kích thước khác nhau dẫn đến đầu ra CNN có vùng quan tâm (_RoI_) với kích thước khác nhau.
Các đặc trưng có cùng kích thước phải được trích xuất từ các vùng quan tâm RoI (ở đây ta giả sử rằng chiều cao là $h_2$ và chiều rộng là $w_2$).
Mạng Fast R-CNN đề xuất phép gộp RoI (_RoI pooling_), nhận đầu ra CNN và các vùng RoI làm đầu vào và cho ra các đặc trưng ghép nối được trích xuất từ mỗi vùng quan tâm với kích thước $n \times c \times h_2 \times w_2$.
3. Tầng kết nối đầy đủ được sử dụng để biến đổi kích thước đầu ra thành $n \times d$, trong đó $d$ được xác định bởi thiết kế mô hình.
4. Khi dự đoán lớp, kích thước đầu ra của tầng đầy đủ lại được biến đổi thành $n \times q$ và ta sử phép hồi quy softmax ($q$ là số các lớp nhãn).
Khi dự đoán khung chứa, kích thước đầu ra của tầng đầy đủ lại được biến đổi thành $n \times 4$.
Điều này có nghĩa với phép dự đoán lớp nhãn và khung chứa cho từng vùng đề xuất.


<!--
The RoI pooling layer in Fast R-CNN is somewhat different from the pooling layers we have discussed before.
In a normal pooling layer, we set the pooling window, padding, and stride to control the output shape.
In an RoI pooling layer, we can directly specify the output shape of each region, such as specifying the height and width of each region as $h_2, w_2$.
Assuming that the height and width of the RoI window are $h$ and $w$, this window is divided into a grid of sub-windows with the shape $h_2 \times w_2$.
The size of each sub-window is about $(h/h_2) \times (w/w_2)$.
The sub-window height and width must always be integers and the largest element is used as the output for a given sub-window.
This allows the RoI pooling layer to extract features of the same shape from RoIs of different shapes.
-->

Tầng gộp RoI trong mạng Fast R-CNN có phần khác với các tầng gộp mà ta đã thảo luận trước đó.
Trong tầng gộp thông thường, ta thiết lập cửa sổ gộp, giá trị đêm, và sải bước để quyết định kích thước đầu ra.
Trong tầng gộp RoI, ta có thể trực tiếp định rõ kích thước đầu ra của từng vùng, ví dụ chiều cao và chiều rộng của từng vùng sẽ là $h_2, w_2$.
Giả sử rằng chiều cao và chiều rộng của cửa sổ RoI là $h$ và $w$, cửa sổ này được chia thành mạng (_grid_) các cửa sổ phụ (_sub-window_) với kích thước $h_2 \times w_2$.
Cửa sổ phụ có kích thước là $(h/h_2) \times (w/w_2)$.
Chiều cao và chiều rộng của cửa sổ phụ phải luôn là số nguyên và thành phần lớn nhất được sử dụng là đầu ra cho cửa sổ phụ đó.
Điều này cho phép tầng gộp RoI trích xuất đặc trưng có cùng kích thước từ các vùng RoI có kích thước khác nhau.


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->


<!--
In :numref:`fig_roi`, we select an $3\times 3$ region as an RoI of the $4 \times 4$ input.
For this RoI, we use a $2\times 2$ RoI pooling layer to obtain a single $2\times 2$ output.
When we divide the region into four sub-windows, they respectively contain the elements 0, 1, 4, and 5 (5 is the largest); 2 and 6 (6 is the largest); 8 and 9 (9 is the largest); and 10.
-->

*dịch đoạn phía trên*


<!--
![$2\times 2$ RoI pooling layer.](../img/roi.svg)
-->


![*dịch mô tả phía trên*](../img/roi.svg)
:label:`fig_roi`


<!--
We use the `ROIPooling` function to demonstrate the RoI pooling layer computation.
Assume that the CNN extracts the feature `X` with both a height and width of 4 and only a single channel.
-->

*dịch đoạn phía trên*



```{.python .input  n=4}
from mxnet import np, npx

npx.set_np()

X = np.arange(16).reshape(1, 1, 4, 4)
X
```


<!--
Assume that the height and width of the image are both 40 pixels and that selective search generates two proposed regions on the image.
Each region is expressed as five elements: the region's object category and the $x, y$ coordinates of its upper-left and bottom-right corners.
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```


<!--
Because the height and width of `X` are $1/10$ of the height and width of the image, the coordinates of the two proposed regions are multiplied by 0.1 according to the `spatial_scale`, 
and then the RoIs are labeled on `X` as `X[:, :, 0:3, 0:3]` and `X[:, :, 1:4, 0:4]`, respectively. 
Finally, we divide the two RoIs into a sub-window grid and extract features with a height and width of 2.
-->

*dịch đoạn phía trên*



```{.python .input  n=6}
npx.roi_pooling(X, rois, pooled_size=(2, 2), spatial_scale=0.1)
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Faster R-CNN
-->

## *dịch tiêu đề phía trên*


<!--
In order to obtain precise object detection results, Fast R-CNN generally requires that many proposed regions be generated in selective search.
Faster R-CNN replaces selective search with a region proposal network. This reduces the number of proposed regions generated, while ensuring precise object detection.
-->

*dịch đoạn phía trên*


<!--
![Faster R-CNN model.](../img/faster-rcnn.svg)
-->

![*dịch mô tả phía trên*](../img/faster-rcnn.svg)
:label:`fig_faster_r-cnn`



<!--
:numref:`fig_faster_r-cnn` shows a Faster R-CNN model.
Compared to Fast R-CNN, Faster R-CNN only changes the method for generating proposed regions from selective search to region proposal network.
The other parts of the model remain unchanged.
The detailed region proposal network computation process is described below:
-->

*dịch đoạn phía trên*


<!--
1. We use a $3\times 3$ convolutional layer with a padding of 1 to transform the CNN output and set the number of output channels to $c$.
This way, each element in the feature map the CNN extracts from the image is a new feature with a length of $c$.
2. We use each element in the feature map as a center to generate multiple anchor boxes of different sizes and aspect ratios and then label them.
3. We use the features of the elements of length $c$ at the center on the anchor boxes to predict the binary category (object or background) and bounding box for their respective anchor boxes.
4. Then, we use non-maximum suppression to remove similar bounding box results that correspond to category predictions of "object".
Finally, we output the predicted bounding boxes as the proposed regions required by the RoI pooling layer.
-->

*dịch đoạn phía trên*



<!--
It is worth noting that, as a part of the Faster R-CNN model, the region proposal network is trained together with the rest of the model.
In addition, the Faster R-CNN object functions include the category and bounding box predictions in object detection, 
as well as the binary category and bounding box predictions for the anchor boxes in the region proposal network.
Finally, the region proposal network can learn how to generate high-quality proposed regions, which reduces the number of proposed regions while maintaining the precision of object detection.
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Mask R-CNN
-->

## *dịch tiêu đề phía trên*


<!--
If training data is labeled with the pixel-level positions of each object in an image, 
a Mask R-CNN model can effectively use these detailed labels to further improve the precision of object detection.
-->

*dịch đoạn phía trên*


<!--
![Mask R-CNN model.](../img/mask-rcnn.svg)
-->

![*dịch mô tả phía trên*](../img/mask-rcnn.svg)
:label:`fig_mask_r-cnn`


<!--
As shown in :numref:`fig_mask_r-cnn`, Mask R-CNN is a modification to the Faster R-CNN model.
Mask R-CNN models replace the RoI pooling layer with an RoI alignment layer.
This allows the use of bilinear interpolation to retain spatial information on feature maps, making Mask R-CNN better suited for pixel-level predictions.
The RoI alignment layer outputs feature maps of the same shape for all RoIs.
This not only predicts the categories and bounding boxes of RoIs, but allows us to use an additional fully convolutional network to predict the pixel-level positions of objects.
We will describe how to use fully convolutional networks to predict pixel-level semantics in images later in this chapter.
-->

*dịch đoạn phía trên*



## Tóm tắt


<!--
* An R-CNN model selects several proposed regions and uses a CNN to perform forward computation and extract the features from each proposed region.
It then uses these features to predict the categories and bounding boxes of proposed regions.
* Fast R-CNN improves on the R-CNN by only performing CNN forward computation on the image as a whole.
It introduces an RoI pooling layer to extract features of the same shape from RoIs of different shapes.
* Faster R-CNN replaces the selective search used in Fast R-CNN with a region proposal network.
This reduces the number of proposed regions generated, while ensuring precise object detection.
* Mask R-CNN uses the same basic structure as Faster R-CNN, but adds a fully convolution layer to help locate objects at the pixel level and further improve the precision of object detection.
-->

*dịch đoạn phía trên*



## Bài tập


<!--
Study the implementation of each model in the [GluonCV toolkit](https://github.com/dmlc/gluon-cv/) related to this section.
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 5 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.d2l.ai/t/374)
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
* Nguyễn Văn Quang

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* 



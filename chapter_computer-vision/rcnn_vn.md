<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Region-based CNNs (R-CNNs)
-->

# CNN theo vùng (*Region-based CNNs* -- R-CNN)


<!--
Region-based convolutional neural networks or regions with CNN features (R-CNNs) are a pioneering approach that applies deep models to object detection :cite:`Girshick.Donahue.Darrell.ea.2014`.
In this section, we will discuss R-CNNs and a series of improvements made to them: Fast R-CNN :cite:`Girshick.2015`, 
Faster R-CNN :cite:`Ren.He.Girshick.ea.2015`, and Mask R-CNN :cite:`He.Gkioxari.Dollar.ea.2017`.
Due to space limitations, we will confine our discussion to the designs of these models.
-->

CNN theo vùng, hay các vùng với đặc trưng CNN (R-CNN) là một hướng tiếp cận tiên phong ứng dụng mô hình sâu cho bài toán phát hiện vật thể :cite:`Girshick.Donahue.Darrell.ea.2014`.
Trong phần này, chúng ta sẽ thảo luận R-CNN và một loạt các cải tiến sau đó: Fast R-CNN :cite:`Girshick.2015`, 
Faster R-CNN :cite:`Ren.He.Girshick.ea.2015`, và Mask R-CNN :cite:`He.Gkioxari.Dollar.ea.2017`.


<!--
## R-CNNs
-->

## R-CNN


<!--
R-CNN models first select several proposed regions from an image (for example, anchor boxes are one type of selection method) and then label their categories and bounding boxes (e.g., offsets).
Then, they use a CNN to perform forward computation to extract features from each proposed area.
Afterwards, we use the features of each proposed region to predict their categories and bounding boxes.
:numref:`fig_r-cnn` shows an R-CNN model.
-->

Đầu tiên, các mô hình R-CNN sẽ chọn một số vùng đề xuất từ ảnh (ví dụ, các khung neo cũng là một dạng phương pháp lựa chọn) và sau đó gán nhãn hạng mục và khung chứa (ví dụ, các giá trị độ dời) cho các vùng này.
Tiếp đến, các mô hình này sử dụng CNN để thực hiện lượt truyền xuôi nhằm trích xuất đặc trưng từ từng vùng đề xuất.
Sau đó, ta sử dụng các đặc trưng của từng vùng được đề xuất để dự đoán hạng mục và khung chứa.
:numref:`fig_r-cnn` mô tả một mô hình R-CNN.

<!--
![R-CNN model.](../img/r-cnn.svg)
-->

![Mô hình R-CNN.](../img/r-cnn.svg)
:label:`fig_r-cnn`


<!--
Specifically, R-CNNs are composed of four main parts:
-->

Cụ thể, R-CNN có bốn phần chính sau:


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

1. Việc tìm kiếm chọn lọc được thực hiện trên ảnh đầu vào để lựa chọn các vùng đề xuất tiềm năng :cite:`Uijlings.Van-De-Sande.Gevers.ea.2013`.
Các vùng đề xuất thông thường được lựa chọn để có nhiều tỷ lệ với hình dạng và kích thước khác nhau.
Nhãn gốc hạng mục và khung chứa sẽ được gán cho từng vùng được đề xuất.
2. Ta sử dụng một mạng CNN đã được tiền huấn luyện, ở dạng rút gọn, đặt trước tầng đầu ra.
Mạng này biến đổi từng vùng đề xuất thành các đầu vào có chiều phù hợp với mạng và thực hiện các tính toán truyền xuôi để trích xuất đặc trưng cho các vùng đề xuất tương ứng.
3. Các đặc trưng và nhãn hạng mục của từng vùng đề xuất được gói thành một mẫu để huấn luyện nhiều máy vector hỗ trợ cho 
phép phân loại vật thể.
Ở đây, mỗi máy vector hỗ trợ được sử dụng để xác định một mẫu có thuộc về một lớp nào đó hay không.
4. Các đặc trưng và khung chứa được gán nhãn của mỗi vùng đề xuất được gói thành một mẫu để huấn luyện mô hình hồi quy tuyến tính để dự đoán khung chứa gốc. 

<!--
Although R-CNN models use pre-trained CNNs to effectively extract image features, the main downside is the slow speed.
As you can imagine, we can select thousands of proposed regions from a single image, requiring thousands of forward computations from the CNN to perform object detection.
This massive computing load means that R-CNNs are not widely used in actual applications.
-->

Mặc dù các mô hình R-CNN sử dụng các mạng CNN đã được tiền huấn luyện để trích xuất các đặc trưng ảnh một cách hiệu quả, điểm hạn chế chính yếu đó là tốc độ chậm.
Ta có thể hình dung, với hàng ngàn vùng đề xuất từ một ảnh, ta cần tới hàng ngàn phép tính truyền xuôi từ mạng CNN để phát hiện vật thể. 
Phép tính toán cồng kềnh khiến các mô hình R-CNN không được sử dụng rộng rãi trong các ứng dụng thực tế.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Fast R-CNN
-->

# *dịch tiêu đề phía trên*


<!--
The main performance bottleneck of an R-CNN model is the need to independently extract features for each proposed region.
As these regions have a high degree of overlap, independent feature extraction results in a high volume of repetitive computations.
Fast R-CNN improves on the R-CNN by only performing CNN forward computation on the image as a whole.
-->

*dịch đoạn phía trên*



<!--
![Fast R-CNN model.](../img/fast-rcnn.svg)
-->

![*dịch mô tả phía trên*](../img/fast-rcnn.svg)
:label:`fig_fast_r-cnn`


<!--
:numref:`fig_fast_r-cnn` shows a Fast R-CNN model.
It is primary computation steps are described below:
-->

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


<!--
The RoI pooling layer in Fast R-CNN is somewhat different from the pooling layers we have discussed before.
In a normal pooling layer, we set the pooling window, padding, and stride to control the output shape.
In an RoI pooling layer, we can directly specify the output shape of each region, such as specifying the height and width of each region as $h_2, w_2$.
Assuming that the height and width of the RoI window are $h$ and $w$, this window is divided into a grid of sub-windows with the shape $h_2 \times w_2$.
The size of each sub-window is about $(h/h_2) \times (w/w_2)$.
The sub-window height and width must always be integers and the largest element is used as the output for a given sub-window.
This allows the RoI pooling layer to extract features of the same shape from RoIs of different shapes.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->


<!--
In :numref:`fig_roi`, we select an $3\times 3$ region as an RoI of the $4 \times 4$ input.
For this RoI, we use a $2\times 2$ RoI pooling layer to obtain a single $2\times 2$ output.
When we divide the region into four sub-windows, they respectively contain the elements 0, 1, 4, and 5 (5 is the largest); 2 and 6 (6 is the largest); 8 and 9 (9 is the largest); and 10.
-->

Trong hình :numref:`fig_roi`, ta chọn một vùng $3\times 3$ làm ROI của một đầu vào $4 \times 4$.
Với ROI này, ta sử dụng một tầng gộp ROI $2\times 2$ để thu được một đầu ra đơn $2\times 2$.
Khi ta chia vùng này thành bốn cửa sổ con, chúng lần lượt chứa các phần tử 0, 1, 4 và 5 (5 là lớn nhất); 2 và 6 (6 là lớn nhất); 8 và 9 (9 là lớn nhất); và 10.

<!--
![$2\times 2$ RoI pooling layer.](../img/roi.svg)
-->


![Tầng gộp ROI $2\times 2$](../img/roi.svg)
:label:`fig_roi`


<!--
We use the `ROIPooling` function to demonstrate the RoI pooling layer computation.
Assume that the CNN extracts the feature `X` with both a height and width of 4 and only a single channel.
-->

Ta sử dụng hàm `ROIPooling` để thực hiện việc tính toán tầng gộp ROI.
Giả sử rằng CNN trích đặc trưng `X` với chiều rộng và chiều cao là 4 và một kênh đơn duy nhất.


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

Giả sử rằng chiều rộng và chiều cao của ảnh là 40 điểm ảnh và tìm kiếm chọn lọc (selective search) sinh ra hai vùng đề xuất trên ảnh này.
Mỗi vùng được biểu thị gồm 5 phần tử: hạng mục của đối tượng trong vùng đó và các tọa độ $x, y$ của các góc trên-bên trái và dưới-bên phải.


```{.python .input  n=5}
rois = np.array([[0, 0, 0, 20, 20], [0, 0, 10, 30, 30]])
```


<!--
Because the height and width of `X` are $1/10$ of the height and width of the image, the coordinates of the two proposed regions are multiplied by 0.1 according to the `spatial_scale`, 
and then the RoIs are labeled on `X` as `X[:, :, 0:3, 0:3]` and `X[:, :, 1:4, 0:4]`, respectively. 
Finally, we divide the two RoIs into a sub-window grid and extract features with a height and width of 2.
-->

Bởi vì chiều cao và chiều rộng của `X` là $1/10$ chiều cao và chiều rộng của ảnh, các tọa độ của hai vùng được đề xuất sẽ nhân với 0.1 dựa theo `spatial_scale`,
rồi các ROI này được gắn nhãn lên `X` lần lượt là `X[:, :, 0:3, 0:3]` và `X[:, :, 1:4, 0:4]`.
Sau cùng, ta chia hai ROI thành một lưới cửa sổ con và trích xuất các đặc trưng với chiều cao và chiều rộng là 2.


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
* Nguyễn Văn Quang
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường

<!-- Phần 2 -->
* 

<!-- Phần 3 -->
* Nguyễn Mai Hoàng Long

<!-- Phần 4 -->
* 

<!-- Phần 5 -->
* 

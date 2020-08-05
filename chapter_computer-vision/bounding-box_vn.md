<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Object Detection and Bounding Boxes
-->

# *dịch tiêu đề phía trên*
:label:`sec_bbox`


<!--
In the previous section, we introduced many models for image classification.
In image classification tasks, we assume that there is only one main target in the image and we only focus on how to identify the target category.
However, in many situations, there are multiple targets in the image that we are interested in.
We not only want to classify them, but also want to obtain their specific positions in the image.
In computer vision, we refer to such tasks as object detection (or object recognition).
-->

*dịch đoạn phía trên*


<!--
Object detection is widely used in many fields.
For example, in self-driving technology, we need to plan routes by identifying the locations of vehicles, pedestrians, roads, and obstacles in the captured video image.
Robots often perform this type of task to detect targets of interest.
Systems in the security field need to detect abnormal targets, such as intruders or bombs.
-->

*dịch đoạn phía trên*


<!--
In the next few sections, we will introduce multiple deep learning models used for object detection.
Before that, we should discuss the concept of target location.
First, import the packages and modules required for the experiment.
-->

*dịch đoạn phía trên*



```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx

npx.set_np()
```


<!--
Next, we will load the sample images that will be used in this section.
We can see there is a dog on the left side of the image and a cat on the right.
They are the two main targets in this image.
-->

*dịch đoạn phía trên*



```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Bounding Box
-->

## Khung chứa


<!--
In object detection, we usually use a bounding box to describe the target location.
The bounding box is a rectangular box that can be determined by the $x$ and $y$ axis coordinates in the upper-left corner and the $x$ and $y$ axis coordinates in the lower-right corner of the rectangle.
We will define the bounding boxes of the dog and the cat in the image based on the coordinate information in the above image.
The origin of the coordinates in the above image is the upper left corner of the image, and to the right and down are the positive directions of the $x$ axis and the $y$ axis, respectively.
-->

Để phát hiện vật thể, ta thường sử dụng khung chứa để mô tả vị trí của mục tiêu.
Khung chứa là một khung hình chữ nhật có thể được xác định bởi hai toạ độ: tọa độ $x$, $y$ góc trên bên trái và toạ độ $x$, $y$ góc dưới bên phải của khung hình chữ nhật.
Ta có thể định nghĩa các khung chứa của con chó và con mèo trong ảnh dựa vào thông tin toạ độ của ảnh trên.
Gốc toạ độ của ảnh trên là góc trên bên trái của ảnh, chiều sang phải và xuống dưới lần lượt là chiều dương của trục $x$ và trục $y$.


```{.python .input  n=2}
# bbox is the abbreviation for bounding box
dog_bbox, cat_bbox = [60, 45, 378, 516], [400, 112, 655, 493]
```


<!--
We can draw the bounding box in the image to check if it is accurate.
Before drawing the box, we will define a helper function `bbox_to_rect`.
It represents the bounding box in the bounding box format of `matplotlib`.
-->

Ta có thể vẽ khung chứa ngay trên ảnh để kiểm tra tính chính xác của nó.
Trước khi vẽ khung, ta định nghĩa hàm hỗ trợ `bbox_to_rect`.
Nó biểu diễn khung chứa theo đúng định dạng khung chứa của `matplotlib`.



```{.python .input  n=3}
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (top-left x, top-left y, bottom-right x,
    # bottom-right y) format to matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```


<!--
After loading the bounding box on the image, we can see that the main outline of the target is basically inside the box.
-->

Sau khi vẽ khung chứa lên ảnh, ta có thể thấy rằng các đường nét chính của mục tiêu về cơ bản là nằm trong khung này.


```{.python .input}
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```



## Tóm tắt


<!--
In object detection, we not only need to identify all the objects of interest in the image, but also their positions.
The positions are generally represented by a rectangular bounding box.
-->

Để phát hiện vật thể, ta không chỉ cần xác định tất cả đối tượng mong muốn trong ảnh mà còn cả vị trí của chúng.
Các vị trí thường được biểu diễn qua các khung chứa hình chữ nhật.


## Bài tập


<!--
Find some images and try to label a bounding box that contains the target.
Compare the difference between the time it takes to label the bounding box and label the category.
-->

Tìm một vài ảnh và thử dán nhãn một khung chứa bao quanh mục tiêu.
So sánh sự khác nhau giữa thời gian cần để dán nhãn các khung chứa và dán nhãn các lớp hạng mục.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC ===================================-->

## Thảo luận
* [Tiếng Anh](https://discuss.d2l.ai/t/369)
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

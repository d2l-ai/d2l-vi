<!--
# Object Detection and Bounding Boxes
-->

# Phát hiện Vật thể và Khoanh vùng Đối tượng (Khung chứa)
:label:`sec_bbox`


<!--
In the previous section, we introduced many models for image classification.
In image classification tasks, we assume that there is only one main target in the image and we only focus on how to identify the target category.
However, in many situations, there are multiple targets in the image that we are interested in.
We not only want to classify them, but also want to obtain their specific positions in the image.
In computer vision, we refer to such tasks as object detection (or object recognition).
-->

Ở phần trước, chúng ta đã giới thiệu nhiều loại mô hình dùng cho phân loại ảnh.
Trong tác vụ phân loại ảnh, ta giả định chỉ có duy nhất một đối tượng trong ảnh và ta chỉ tập trung xác định nó thuộc về nhóm nào.
Tuy nhiên, ở nhiều tình huống cùng lúc sẽ có nhiều đối tượng trong ảnh mà ta quan tâm.
Ta không chỉ muốn phân loại chúng mà còn muốn xác định vị trí cụ thể của chúng ở trong ảnh.
Trong lĩnh vực thị giác máy tính, những tác vụ như thế được gọi là phát hiện vật thể (hoặc nhận dạng vật thể).

<!--
Object detection is widely used in many fields.
For example, in self-driving technology, we need to plan routes by identifying the locations of vehicles, pedestrians, roads, and obstacles in the captured video image.
Robots often perform this type of task to detect targets of interest.
Systems in the security field need to detect abnormal targets, such as intruders or bombs.
-->

Phát hiện vật thể được sử dụng rộng rãi trong nhiều lĩnh vực.
Chẳng hạn, trong công nghệ xe tự hành, ta cần lên lộ trình bằng cách xác định các vị trí của phương tiện di chuyển, người đi đường, đường xá và các vật cản trong các ảnh được thu về từ video.
Robot cần thực hiện kiểu tác vụ này để phát hiện các đối tượng mà chúng quan tâm.
Hay các hệ thống an ninh cần phát hiện các mục tiêu bất thường, ví dụ như các đối tượng xâm nhập bất hợp pháp hoặc bom mìn.

<!--
In the next few sections, we will introduce multiple deep learning models used for object detection.
Before that, we should discuss the concept of target location.
First, import the packages and modules required for the experiment.
-->

Trong các phần tiếp theo, chúng tôi sẽ giới thiệu nhiều mô hình học sâu dùng để phát hiện vật thể.
Trước hết, ta nên bàn qua về khái niệm vị trí vật thể.
Đầu tiên, ta hãy nhập các gói và mô-đun cần thiết cho việc thử nghiệm.


```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
```


<!--
Next, we will load the sample images that will be used in this section.
We can see there is a dog on the left side of the image and a cat on the right.
They are the two main targets in this image.
-->

Kế tiếp, ta nạp các ảnh mẫu sẽ sử dụng trong phần này.
Ta có thể thấy trong hình là một con chó ở bên trái và một con mèo ở bên phải.
Chúng là hai đối tượng chính trong ảnh này.


```{.python .input}
d2l.set_figsize()
img = image.imread('../img/catdog.jpg').asnumpy()
d2l.plt.imshow(img);
```

```{.python .input}
#@tab pytorch, tensorflow
d2l.set_figsize()
img = d2l.plt.imread('../img/catdog.jpg')
d2l.plt.imshow(img);
```


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
Khung chứa là một khung hình chữ nhật có thể được xác định bởi hai tọa độ: tọa độ $x$, $y$ góc trên bên trái và tọa độ $x$, $y$ góc dưới bên phải của khung hình chữ nhật.
Ta có thể định nghĩa các khung chứa của con chó và con mèo trong ảnh dựa vào thông tin tọa độ của ảnh trên.
Gốc tọa độ của ảnh trên là góc trên bên trái của ảnh, chiều sang phải và xuống dưới lần lượt là chiều dương của trục $x$ và trục $y$.


```{.python .input}
#@tab all
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



```{.python .input}
#@tab all
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

Sau khi vẽ khung chứa lên ảnh, có thể thấy rằng phần chính của mục tiêu về cơ bản là nằm trong khung chứa.


```{.python .input}
#@tab all
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


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/369)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Mai Hoàng Long
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Đỗ Trường Giang
* Nguyễn Lê Quang Nhật

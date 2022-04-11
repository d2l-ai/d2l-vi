# Hộp phát hiện và giới hạn đối tượng
:label:`sec_bbox`

Trong các phần trước (ví dụ: :numref:`sec_alexnet`—:numref:`sec_googlenet`), chúng tôi đã giới thiệu các mô hình khác nhau để phân loại hình ảnh. Trong các tác vụ phân loại hình ảnh, chúng tôi giả định rằng chỉ có * một* đối tượng chính trong hình ảnh và chúng tôi chỉ tập trung vào cách nhận dạng danh mục của nó. Tuy nhiên, thường có các đối tượng * nhiều* trong hình ảnh quan tâm. Chúng tôi không chỉ muốn biết danh mục của họ, mà còn là vị trí cụ thể của họ trong hình ảnh. Trong tầm nhìn máy tính, chúng tôi đề cập đến các tác vụ như *phát hiện đối tượng* (hoặc *nhận dạng đối tượng*). 

Phát hiện đối tượng đã được ứng dụng rộng rãi trong nhiều lĩnh vực. Ví dụ, tự lái cần lên kế hoạch cho các tuyến đường di chuyển bằng cách phát hiện vị trí của phương tiện, người đi bộ, đường xá và chướng ngại vật trong các hình ảnh video đã chụp. Bên cạnh đó, robot có thể sử dụng kỹ thuật này để phát hiện và bản địa hóa các đối tượng quan tâm trong suốt quá trình điều hướng môi trường. Hơn nữa, các hệ thống an ninh có thể cần phải phát hiện các vật bất thường, chẳng hạn như kẻ xâm nhập hoặc bom. 

Trong vài phần tiếp theo, chúng tôi sẽ giới thiệu một số phương pháp học sâu để phát hiện đối tượng. Chúng tôi sẽ bắt đầu với phần giới thiệu về *vị trí* (hoặc *locations*) của các đối tượng.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, npx, np

npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

Chúng tôi sẽ tải hình ảnh mẫu được sử dụng trong phần này. Chúng ta có thể thấy rằng có một chú chó ở phía bên trái của hình ảnh và một con mèo bên phải. Chúng là hai đối tượng chính trong hình ảnh này.

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

## Hộp giới hạn

Trong phát hiện đối tượng, chúng ta thường sử dụng một hộp giới hạn * để mô tả vị trí không gian của một đối tượng. Hộp giới hạn là hình chữ nhật, được xác định bởi tọa độ $x$ và $y$ của góc trên bên trái của hình chữ nhật và tọa độ như vậy của góc dưới bên phải. Một biểu diễn hộp giới hạn thường được sử dụng khác là tọa độ $(x, y)$-trục của trung tâm hộp giới hạn, và chiều rộng và chiều cao của hộp. 

[**Ở đây chúng ta định nghĩa các hàm để chuyển đổi giữa hai **] các hàm này (**hai đại diện **): `box_corner_to_center` chuyển đổi từ biểu diễn hai góc sang trình bày chiều rộng trung tâm và `box_center_to_corner` ngược lại. Đối số đầu vào `boxes` phải là một tensor hai chiều của hình dạng ($n$, 4), trong đó $n$ là số hộp giới hạn.

```{.python .input}
#@tab all
#@save
def box_corner_to_center(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = d2l.stack((cx, cy, w, h), axis=-1)
    return boxes

#@save
def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = d2l.stack((x1, y1, x2, y2), axis=-1)
    return boxes
```

Chúng tôi sẽ [** xác định các hộp giới hạn của chó và con mèo trong hình ảnh**] dựa trên thông tin tọa độ. Nguồn gốc của tọa độ trong hình ảnh là góc trên bên trái của hình ảnh, và bên phải và xuống là các hướng dương của các trục $x$ và $y$, tương ứng.

```{.python .input}
#@tab all
# Here `bbox` is the abbreviation for bounding box
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]
```

Chúng tôi có thể xác minh tính chính xác của hai chức năng chuyển đổi hộp giới hạn bằng cách chuyển đổi hai lần.

```{.python .input}
#@tab all
boxes = d2l.tensor((dog_bbox, cat_bbox))
box_center_to_corner(box_corner_to_center(boxes)) == boxes
```

Hãy để chúng tôi [** vẽ các hộp giới hạn trong hình ảnh**] để kiểm tra xem chúng có chính xác không. Trước khi vẽ, chúng ta sẽ xác định một hàm helper `bbox_to_rect`. Nó đại diện cho hộp giới hạn ở định dạng hộp giới hạn của gói `matplotlib`.

```{.python .input}
#@tab all
#@save
def bbox_to_rect(bbox, color):
    """Convert bounding box to matplotlib format."""
    # Convert the bounding box (upper-left x, upper-left y, lower-right x,
    # lower-right y) format to the matplotlib format: ((upper-left x,
    # upper-left y), width, height)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2)
```

Sau khi thêm các hộp giới hạn trên hình ảnh, chúng ta có thể thấy rằng đường viền chính của hai đối tượng về cơ bản nằm trong hai hộp.

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'));
```

## Tóm tắt

* Phát hiện đối tượng không chỉ nhận ra tất cả các đối tượng quan tâm trong hình ảnh, mà còn cả vị trí của chúng. Vị trí thường được biểu diễn bằng một hộp giới hạn hình chữ nhật.
* Chúng ta có thể chuyển đổi giữa hai biểu diễn hộp giới hạn thường được sử dụng.

## Bài tập

1. Tìm một hình ảnh khác và cố gắng gắn nhãn một hộp giới hạn chứa đối tượng. So sánh các hộp và danh mục giới hạn ghi nhãn: thường mất nhiều thời gian hơn?
1. Tại sao chiều trong cùng của đối số đầu vào `boxes` của `box_corner_to_center` và `box_center_to_corner` luôn là 4?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/369)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1527)
:end_tab:

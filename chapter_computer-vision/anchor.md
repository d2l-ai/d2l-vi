# Hộp neo
:label:`sec_anchor`

Thuật toán phát hiện đối tượng thường lấy mẫu một số lượng lớn các vùng trong hình ảnh đầu vào, xác định xem các vùng này có chứa các đối tượng quan tâm hay không, và điều chỉnh ranh giới của các vùng để dự đoán
*hộp giới hạn đất-truth*
of the objects các đối tượng more accurately chính xác. Các mô hình khác nhau có thể áp dụng các sơ đồ lấy mẫu khu vực khác nhau. Ở đây chúng tôi giới thiệu một trong những phương pháp như vậy: nó tạo ra nhiều hộp giới hạn với tỷ lệ khác nhau và tỷ lệ khung hình tập trung vào mỗi pixel. Các hộp giới hạn này được gọi là hộp neo *. Chúng tôi sẽ thiết kế một mô hình phát hiện đối tượng dựa trên các hộp neo trong :numref:`sec_ssd`. 

Đầu tiên, chúng ta hãy sửa đổi độ chính xác in ấn chỉ để kết quả đầu ra ngắn gọn hơn.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## Tạo nhiều hộp neo

Giả sử hình ảnh đầu vào có chiều cao $h$ và chiều rộng $w$. Chúng tôi tạo ra các hộp neo với các hình dạng khác nhau tập trung vào từng pixel của hình ảnh. Hãy để *quy mô* là $s\in (0, 1]$ và tỷ lệ khung hình* (tỷ lệ chiều rộng trên chiều cao) là $r > 0$. Sau đó [** chiều rộng và chiều cao của hộp neo là $ws\sqrt{r}$ và $hs/\sqrt{r}$, tôn trọng.**] Lưu ý rằng khi vị trí trung tâm được đưa ra, một hộp neo có chiều rộng và chiều cao đã biết được xác định. 

Để tạo ra nhiều hộp neo với các hình dạng khác nhau, chúng ta hãy đặt một loạt các thang đo $s_1,\ldots, s_n$ và một loạt các tỷ lệ khung hình $r_1,\ldots, r_m$. Khi sử dụng tất cả các kết hợp của các thang đo và tỷ lệ khung hình này với mỗi pixel làm trung tâm, hình ảnh đầu vào sẽ có tổng cộng $whnm$ hộp neo. Mặc dù các hộp neo này có thể bao gồm tất cả các hộp giới hạn sự thật mặt đất, độ phức tạp tính toán dễ dàng quá cao. Trong thực tế, chúng ta chỉ có thể (** xem xét những kết hợp chứng**) $s_1$ hoặc $r_1$: 

(**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**) 

Điều đó có nghĩa là, số lượng hộp neo tập trung vào cùng một điểm ảnh là $n+m-1$. Đối với toàn bộ hình ảnh đầu vào, chúng tôi sẽ tạo tổng cộng $wh(n+m-1)$ hộp neo. 

Phương pháp tạo hộp neo trên được thực hiện trong chức năng `multibox_prior` sau đây. Chúng tôi chỉ định hình ảnh đầu vào, danh sách các thang đo và danh sách các tỷ lệ khung hình, sau đó chức năng này sẽ trả về tất cả các hộp neo.

```{.python .input}
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boxes_per_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boxes_per_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boxes_per_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

Chúng ta có thể thấy rằng [** hình dạng của biến hộp neo trả về `Y`**] là (kích thước lô, số lượng hộp neo, 4).

```{.python .input}
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

Sau khi thay đổi hình dạng của biến hộp neo `Y` thành (chiều cao hình ảnh, chiều rộng hình ảnh, số hộp neo tập trung vào cùng một điểm ảnh, 4), chúng ta có thể lấy tất cả các hộp neo tập trung vào một vị trí pixel được chỉ định. Sau đây, chúng ta [** truy cập hộp neo đầu tiên tập trung vào (250, 250) **]. Nó có bốn phần tử: tọa độ $(x, y)$-trục ở góc trên bên trái và trục $(x, y)$-tọa độ ở góc dưới bên phải của hộp neo. Các giá trị tọa độ của cả hai trục được chia cho chiều rộng và chiều cao của hình ảnh, tương ứng; do đó, phạm vi nằm trong khoảng từ 0 đến 1.

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

Để [** hiển thị tất cả các hộp neo tập trung vào một pixel trong image**], chúng tôi xác định hàm `show_bboxes` sau để vẽ nhiều hộp giới hạn trên ảnh.

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

Như chúng ta vừa thấy, các giá trị tọa độ của các trục $x$ và $y$ trong biến `boxes` đã được chia cho chiều rộng và chiều cao của hình ảnh, tương ứng. Khi vẽ các hộp neo, chúng ta cần khôi phục các giá trị tọa độ ban đầu của chúng; do đó, chúng ta định nghĩa biến `bbox_scale` bên dưới. Bây giờ, chúng ta có thể vẽ tất cả các hộp neo tập trung vào (250, 250) trong hình ảnh. Như bạn có thể thấy, hộp neo màu xanh lam với thang điểm 0,75 và tỷ lệ khung hình là 1 bao quanh chó trong hình ảnh.

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [** Giao lộ qua Union (IoU) **]

Chúng tôi chỉ đề cập rằng một hộp neo “tốt” bao quanh chú chó trong hình ảnh. Nếu hộp giới hạn đất-chân lý của đối tượng được biết, làm thế nào có thể “tốt” ở đây được định lượng? Bằng trực giác, chúng ta có thể đo sự tương đồng giữa hộp neo và hộp giới hạn chân lý mặt đất. Chúng tôi biết rằng chỉ số *Jaccard* có thể đo lường sự tương đồng giữa hai bộ. Đưa ra các bộ $\mathcal{A}$ và $\mathcal{B}$, chỉ số Jaccard của chúng là kích thước giao điểm của chúng chia cho kích thước của công đoàn của chúng: 

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$

Trên thực tế, chúng ta có thể xem xét khu vực pixel của bất kỳ hộp giới hạn nào là một tập hợp các pixel. Bằng cách này, chúng ta có thể đo sự giống nhau của hai hộp giới hạn bằng chỉ số Jaccard của các bộ pixel của chúng. Đối với hai hộp giới hạn, chúng ta thường tham khảo chỉ số Jaccard của chúng là *giao lộ qua công đoàn * (* IoU*), đó là tỷ lệ giữa khu vực giao nhau của chúng với khu vực công đoàn của chúng, như thể hiện trong :numref:`fig_iou`. Phạm vi của một IoU nằm trong khoảng từ 0 đến 1:0 có nghĩa là hai hộp giới hạn hoàn toàn không chồng lên nhau, trong khi 1 chỉ ra rằng hai hộp giới hạn bằng nhau. 

![IoU is the ratio of the intersection area to the union area of two bounding boxes.](../img/iou.svg)
:label:`fig_iou`

Đối với phần còn lại của phần này, chúng tôi sẽ sử dụng IoU để đo sự tương đồng giữa các hộp neo và các hộp giới hạn chân lý mặt đất và giữa các hộp neo khác nhau. Cho hai danh sách các hộp neo hoặc đường viền, `box_iou` sau sẽ tính toán IoU cặp của chúng trong hai danh sách này.

```{.python .input}
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## Dán nhãn hộp neo trong dữ liệu đào tạo
:label:`subsec_labeling-anchor-boxes`

Trong một tập dữ liệu đào tạo, chúng tôi coi mỗi hộp neo là một ví dụ đào tạo. Để đào tạo một mô hình phát hiện đối tượng, chúng ta cần nhãn * class* và *offset* cho mỗi hộp neo, trong đó trước đây là lớp của đối tượng có liên quan đến hộp neo và sau này là phần bù của hộp giới hạn sự thật mặt đất so với hộp neo. Trong quá trình dự đoán, đối với mỗi hình ảnh, chúng tôi tạo ra nhiều hộp neo, dự đoán các lớp và bù đắp cho tất cả các hộp neo, điều chỉnh vị trí của chúng theo các bù đắp dự đoán để có được các hộp giới hạn dự đoán và cuối cùng chỉ xuất ra những hộp giới hạn dự đoán đáp ứng các tiêu chí nhất định. 

Như chúng ta đã biết, một bộ đào tạo phát hiện đối tượng đi kèm với nhãn cho các vị trí của các hộp giới hạn * đất-chân thất* và các lớp của các đối tượng được bao quanh của chúng. Để gắn nhãn bất kỳ hộp neo * được tạo ra*, chúng tôi đề cập đến vị trí được dán nhãn và lớp của hộp giới hạn đất*được chỉ định * của nó gần nhất với hộp neo. Sau đây, chúng tôi mô tả một thuật toán để gán các hộp giới hạn sự thật gần nhất cho các hộp neo.  

### [** Gán các hộp giới hạn mặt đất-Truth cho Hộp neo**]

Với một hình ảnh, giả sử rằng các hộp neo là $A_1, A_2, \ldots, A_{n_a}$ và các hộp giới hạn đất-chân lý là $B_1, B_2, \ldots, B_{n_b}$, trong đó $n_a \geq n_b$. Chúng ta hãy xác định một ma trận $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$, có phần tử $x_{ij}$ trong hàng $i^\mathrm{th}$ và cột $j^\mathrm{th}$ là IoU của hộp neo $A_i$ và hộp giới hạn sự thật mặt đất $B_j$. Thuật toán bao gồm các bước sau: 

1. Tìm phần tử lớn nhất trong ma trận $\mathbf{X}$ và biểu thị các chỉ số hàng và cột của nó là $i_1$ và $j_1$, tương ứng. Sau đó, hộp giới hạn sự thật mặt đất $B_{j_1}$ được gán cho hộp neo $A_{i_1}$. Điều này khá trực quan vì $A_{i_1}$ và $B_{j_1}$ là gần nhất trong số tất cả các cặp hộp neo và hộp giới hạn sự thật mặt đất. Sau khi gán đầu tiên, loại bỏ tất cả các phần tử trong hàng ${i_1}^\mathrm{th}$ và cột ${j_1}^\mathrm{th}$ trong ma trận $\mathbf{X}$. 
1. Tìm lớn nhất trong số các phần tử còn lại trong ma trận $\mathbf{X}$ và biểu thị các chỉ số hàng và cột của nó là $i_2$ và $j_2$, tương ứng. Chúng tôi gán hộp giới hạn sự thật mặt đất $B_{j_2}$ để neo hộp $A_{i_2}$ và loại bỏ tất cả các yếu tố trong hàng ${i_2}^\mathrm{th}$ và cột ${j_2}^\mathrm{th}$ trong ma trận $\mathbf{X}$.
1. Tại thời điểm này, các phần tử trong hai hàng và hai cột trong ma trận $\mathbf{X}$ đã bị loại bỏ. Chúng tôi tiến hành cho đến khi tất cả các phần tử trong $n_b$ cột trong ma trận $\mathbf{X}$ bị loại bỏ. Tại thời điểm này, chúng tôi đã chỉ định một hộp giới hạn chân lý mặt đất cho mỗi hộp neo $n_b$.
1. Chỉ đi qua các hộp neo $n_a - n_b$ còn lại. Ví dụ, đưa ra bất kỳ hộp neo nào $A_i$, hãy tìm hộp giới hạn chân lý mặt đất $B_j$ với IoU lớn nhất với $A_i$ trong suốt hàng $i^\mathrm{th}$ của ma trận $\mathbf{X}$ và gán $B_j$ cho $A_i$ chỉ khi IoU này lớn hơn ngưỡng được xác định trước.

Hãy để chúng tôi minh họa thuật toán trên bằng cách sử dụng một ví dụ cụ thể. Như thể hiện trong :numref:`fig_anchor_label` (trái), giả sử rằng giá trị lớn nhất trong ma trận $\mathbf{X}$ là $x_{23}$, chúng tôi gán hộp giới hạn sự thật mặt đất $B_3$ vào hộp neo $A_2$. Sau đó, chúng ta loại bỏ tất cả các phần tử trong hàng 2 và cột 3 của ma trận, tìm $x_{71}$ lớn nhất trong các phần tử còn lại (khu vực bóng mờ) và gán hộp giới hạn sự thật mặt đất $B_1$ vào hộp neo $A_7$. Tiếp theo, như thể hiện trong :numref:`fig_anchor_label` (giữa), loại bỏ tất cả các phần tử trong hàng 7 và cột 1 của ma trận, tìm $x_{54}$ lớn nhất trong các phần tử còn lại (khu vực bóng mờ) và gán hộp giới hạn sự thật mặt đất $B_4$ vào hộp neo $A_5$. Cuối cùng, như thể hiện trong :numref:`fig_anchor_label` (phải), loại bỏ tất cả các phần tử trong hàng 5 và cột 4 của ma trận, tìm $x_{92}$ lớn nhất trong các phần tử còn lại (khu vực bóng mờ) và gán hộp giới hạn đất-chân lý $B_2$ vào hộp neo $A_9$. Sau đó, chúng ta chỉ cần đi qua các hộp neo còn lại $A_1, A_3, A_4, A_6, A_8$ và xác định xem có nên gán cho chúng các hộp giới hạn chân lý mặt đất theo ngưỡng hay không. 

![Assigning ground-truth bounding boxes to anchor boxes.](../img/anchor-label.svg)
:label:`fig_anchor_label`

Thuật toán này được thực hiện trong hàm `assign_anchor_to_bbox` sau.

```{.python .input}
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= 0.5)[0]
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= 0.5).reshape(-1)
    box_j = indices[max_ious >= 0.5]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### Các lớp ghi nhãn và độ lệch

Bây giờ chúng ta có thể dán nhãn lớp và bù đắp cho mỗi hộp neo. Giả sử rằng một hộp neo $A$ được gán một hộp giới hạn đất-chân lý $B$. Một mặt, lớp của hộp neo $A$ sẽ được dán nhãn là của $B$. Mặt khác, độ lệch của hộp neo $A$ sẽ được dán nhãn theo vị trí tương đối giữa tọa độ trung tâm là $B$ và $A$ cùng với kích thước tương đối giữa hai hộp này. Với các vị trí và kích thước khác nhau của các hộp khác nhau trong tập dữ liệu, chúng ta có thể áp dụng các biến đổi cho các vị trí và kích thước tương đối có thể dẫn đến các bù phân bố đồng đều hơn dễ phù hợp hơn. Ở đây chúng tôi mô tả một sự chuyển đổi chung. [**Cho tọa độ trung tâm của $A$ và $B$ là $(x_a, y_a)$ và $(x_b, y_b)$, chiều rộng của chúng là $w_a$ và $w_b$, và chiều cao của chúng tương ứng là $h_a$ và $h_b$. Chúng tôi có thể dán nhãn phần bù của $A$ như 

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
where default values of the constants are $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, and $\sigma_w=\sigma_h=0.2$.
This transformation is implemented below in the `offset_boxes` function.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

Nếu một hộp neo không được gán một hộp giới hạn đất-truth, chúng ta chỉ cần dán nhãn lớp của hộp neo là “background”. Các hộp neo có lớp nền thường được gọi là hộp neo * tiêu cực* và phần còn lại được gọi là hộp neo * tích cực*. Chúng tôi thực hiện hàm `multibox_target` sau để [** label class and offsets for anchor boxes**](đối số `anchors`) bằng cách sử dụng các hộp bounding ground-truth (đối số `labels`). Hàm này đặt lớp nền thành 0 và tăng chỉ số số nguyên của một lớp mới bằng một.

```{.python .input}
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### Một ví dụ

Hãy để chúng tôi minh họa ghi nhãn hộp neo thông qua một ví dụ cụ thể. Chúng tôi xác định các hộp giới hạn đất cho chó và mèo trong hình ảnh được tải, trong đó phần tử đầu tiên là lớp (0 cho chó và 1 cho mèo) và bốn yếu tố còn lại là tọa độ $(x, y)$-trục ở góc trên bên trái và góc dưới bên phải (phạm vi nằm trong khoảng từ 0 đến 1). Chúng tôi cũng xây dựng năm hộp neo được dán nhãn bằng cách sử dụng tọa độ của góc trên bên trái và góc dưới bên phải: $A_0, \ldots, A_4$ (chỉ số bắt đầu từ 0). Sau đó, chúng ta [** vẽ các hộp giới hạn mặt đất và hộp neo trong hình ảnh**]

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

Sử dụng hàm `multibox_target` được xác định ở trên, chúng ta có thể [** label class và offset của các hộp neo này dựa trên các hộp giới hạn đất-chân thất**] cho chó và mèo. Trong ví dụ này, các chỉ số của các lớp nền, chó và mèo lần lượt là 0, 1 và 2. Dưới đây chúng tôi thêm một kích thước cho các ví dụ về hộp neo và các hộp giới hạn đất-chân lý.

```{.python .input}
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

Có ba mục trong kết quả trả về, tất cả đều ở định dạng tensor. Mục thứ ba chứa các lớp được dán nhãn của các hộp neo đầu vào. 

Chúng ta hãy phân tích các nhãn lớp trả về dưới đây dựa trên hộp neo và vị trí hộp giới hạn đất-chân lý trong hình ảnh. Đầu tiên, trong số tất cả các cặp hộp neo và hộp giới hạn sự thật mặt đất, IoU của hộp neo $A_4$ và hộp giới hạn sự thật mặt đất của con mèo là lớn nhất. Do đó, lớp $A_4$ được dán nhãn là con mèo. Lấy ra các cặp chứa $A_4$ hoặc hộp giới hạn sự thật mặt đất của con mèo, trong số các cặp còn lại của hộp neo $A_1$ và hộp giới hạn sự thật mặt đất của chó có IoU lớn nhất. Vì vậy, lớp $A_1$ được dán nhãn là chó. Tiếp theo, chúng ta cần phải đi qua ba hộp neo chưa được dán nhãn còn lại: $A_0$, $A_2$ và $A_3$. Đối với $A_0$, lớp của hộp giới hạn chân thất-chân lý với IoU lớn nhất là chó, nhưng IoU nằm dưới ngưỡng được xác định trước (0,5), vì vậy lớp được dán nhãn là nền; đối với $A_2$, lớp của hộp giới hạn sự thật mặt đất với IoU lớn nhất là con mèo và IoU vượt quá ngưỡng, vì vậy lớp được dán nhãn là con mèo; đối với $A_3$, lớp của hộp giới hạn đất-chân lý với IoU lớn nhất là con mèo, nhưng giá trị nằm dưới ngưỡng, do đó lớp được dán nhãn là nền.

```{.python .input}
#@tab all
labels[2]
```

Mục trả về thứ hai là một biến mặt nạ của hình dạng (kích thước lô, gấp bốn lần số hộp neo). Mỗi bốn phần tử trong biến mask tương ứng với bốn giá trị bù của mỗi hộp neo. Vì chúng ta không quan tâm đến việc phát hiện nền, sự bù đắp của lớp phủ định này không nên ảnh hưởng đến hàm khách quan. Thông qua phép nhân elementwise, các số không trong biến mask sẽ lọc ra các bù lớp âm trước khi tính hàm mục tiêu.

```{.python .input}
#@tab all
labels[1]
```

Mục trả về đầu tiên chứa bốn giá trị bù được dán nhãn cho mỗi hộp neo. Lưu ý rằng các bù đắp của các hộp neo lớp âm được dán nhãn là số không.

```{.python .input}
#@tab all
labels[0]
```

## Dự đoán các hộp giới hạn với ức chế không tối đa
:label:`subsec_predicting-bounding-boxes-nms`

Trong quá trình dự đoán, chúng tôi tạo ra nhiều hộp neo cho hình ảnh và dự đoán các lớp và bù đắp cho mỗi hộp. Do đó, một hộp giới hạn dự đoán * thu được theo một hộp neo với độ lệch dự đoán của nó. Dưới đây chúng tôi thực hiện hàm `offset_inverse` có các neo và dự đoán bù đắp làm đầu vào và [** áp dụng các biến đổi bù ngược để trả về phối hợp hộp giới hạn dự đoán **].

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

Khi có nhiều hộp neo, nhiều hộp giới hạn dự đoán tương tự (với sự chồng chéo đáng kể) có thể có khả năng xuất ra cho xung quanh cùng một đối tượng. Để đơn giản hóa đầu ra, chúng ta có thể hợp nhất các hộp giới hạn dự đoán tương tự thuộc về cùng một đối tượng bằng cách sử dụng *non-maximression* (NMS). 

Đây là cách ức chế không tối đa hoạt động. Đối với một hộp giới hạn dự đoán $B$, mô hình phát hiện đối tượng tính toán khả năng dự đoán cho mỗi lớp. Biểu thị bởi $p$ khả năng dự đoán lớn nhất, lớp tương ứng với xác suất này là lớp dự đoán cho $B$. Cụ thể, chúng tôi đề cập đến $p$ là *sự tự tin* (điểm) của hộp giới hạn dự đoán $B$. Trên cùng một hình ảnh, tất cả các hộp giới hạn không nền được dự đoán được sắp xếp theo sự tự tin theo thứ tự giảm dần để tạo ra một danh sách $L$. Sau đó, chúng tôi thao tác danh sách được sắp xếp $L$ trong các bước sau: 

1. Chọn hộp giới hạn dự đoán $B_1$ với độ tin cậy cao nhất từ $L$ làm cơ sở và loại bỏ tất cả các hộp giới hạn dự đoán không cơ sở có IoU với $B_1$ vượt quá ngưỡng được xác định trước $\epsilon$ từ $L$. Tại thời điểm này, $L$ giữ hộp giới hạn dự đoán với sự tự tin cao nhất nhưng giảm những người khác quá giống với nó. Tóm lại, những người có *không tối đa* điểm tin cậy là * bị ức chế *.
1. Chọn hộp giới hạn dự đoán $B_2$ với độ tin cậy cao thứ hai từ $L$ làm cơ sở khác và loại bỏ tất cả các hộp giới hạn dự đoán không cơ sở có IoU với $B_2$ vượt quá $\epsilon$ từ $L$.
1. Lặp lại quy trình trên cho đến khi tất cả các hộp giới hạn dự đoán trong $L$ đã được sử dụng làm cơ sở. Tại thời điểm này, IoU của bất kỳ cặp hộp giới hạn dự đoán nào trong $L$ nằm dưới ngưỡng $\epsilon$; do đó, không có cặp nào quá giống nhau. 
1. Xuất tất cả các hộp giới hạn dự đoán trong danh sách $L$.

[**Hàm `nms` sau đây sắp xếp điểm tin cậy theo thứ tự giảm dần và trả về các chỉ số của chúng. **]

```{.python .input}
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

Chúng tôi xác định `multibox_detection` sau đây để [** áp dụng ức chế không tối đa để dự đoán các hộp giới hộp**]. Đừng lo lắng nếu bạn thấy việc thực hiện một chút phức tạp: chúng tôi sẽ chỉ ra cách nó hoạt động với một ví dụ cụ thể ngay sau khi thực hiện.

```{.python .input}
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`keep` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

Bây giờ chúng ta hãy [** áp dụng các triển khai trên cho một ví dụ cụ thể với bốn hộp neo**]. Để đơn giản, chúng tôi giả định rằng các bù dự đoán là tất cả các số không. Điều này có nghĩa là các hộp giới hạn dự đoán là hộp neo. Đối với mỗi lớp trong số nền, chó và mèo, chúng tôi cũng xác định khả năng dự đoán của nó.

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

Chúng ta có thể [** vẽ những hộp giới hạn dự đoán này với sự tự tin của họ về hình ảnh**]

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

Bây giờ chúng ta có thể gọi hàm `multibox_detection` để thực hiện triệt tiêu không tối đa, trong đó ngưỡng được đặt thành 0,5. Lưu ý rằng chúng ta thêm một chiều cho các ví dụ trong đầu vào tensor. 

Chúng ta có thể thấy rằng [** hình dạng của kết quả trả về**] là (kích thước lô, số hộp neo, 6). Sáu yếu tố trong chiều trong cùng cung cấp thông tin đầu ra cho cùng một hộp giới hạn dự đoán. Phần tử đầu tiên là chỉ số lớp dự đoán, bắt đầu từ 0 (0 là chó và 1 là mèo). Giá trị -1 cho biết nền hoặc loại bỏ trong ức chế không tối đa. Yếu tố thứ hai là sự tự tin của hộp giới hạn dự đoán. Bốn phần tử còn lại là tọa độ $(x, y)$-trục của góc trên bên trái và góc dưới bên phải của hộp giới hạn dự đoán, tương ứng (phạm vi nằm trong khoảng từ 0 đến 1).

```{.python .input}
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

Sau khi loại bỏ những hộp giới hạn dự đoán của lớp -1, chúng ta có thể [**output the final predicted bounding box giữ bởi non-maximression**].

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

Trong thực tế, chúng ta có thể loại bỏ các hộp giới hạn dự đoán với độ tin cậy thấp hơn ngay cả trước khi thực hiện triệt tiêu không tối đa, do đó làm giảm tính toán trong thuật toán này. Chúng tôi cũng có thể xử lý hậu kỳ đầu ra của ức chế không tối đa, ví dụ, bằng cách chỉ giữ kết quả với sự tự tin cao hơn trong đầu ra cuối cùng. 

## Tóm tắt

* Chúng tôi tạo ra các hộp neo với các hình dạng khác nhau tập trung vào từng pixel của hình ảnh.
* Giao lộ qua liên kết (IoU), còn được gọi là chỉ số Jaccard, đo sự tương đồng của hai hộp giới hạn. Đó là tỷ lệ của khu vực giao nhau của họ với khu vực công đoàn của họ.
* Trong một bộ đào tạo, chúng ta cần hai loại nhãn cho mỗi hộp neo. Một là lớp của đối tượng có liên quan đến hộp neo và cái còn lại là độ lệch của hộp giới hạn đất-chân lý so với hộp neo.
* Trong quá trình dự đoán, chúng ta có thể sử dụng triệt tiêu không tối đa (NMS) để loại bỏ các hộp giới hạn dự đoán tương tự, từ đó đơn giản hóa đầu ra.

## Bài tập

1. Thay đổi giá trị của `sizes` và `ratios` trong hàm `multibox_prior`. Những thay đổi đối với các hộp neo được tạo ra là gì?
1. Xây dựng và hình dung hai hộp giới hạn với IoU 0,5. Làm thế nào để họ chồng chéo với nhau?
1. Sửa đổi các biến `anchors` trong :numref:`subsec_labeling-anchor-boxes` và :numref:`subsec_predicting-bounding-boxes-nms`. Làm thế nào để kết quả thay đổi?
1. Ức chế không tối đa là một thuật toán tham lam ngăn chặn các hộp giới hạn dự đoán bằng cách * xóa* chúng. Có thể một số trong những cái bị xóa này thực sự hữu ích? Làm thế nào thuật toán này có thể được sửa đổi để ngăn chặn *softly*? Bạn có thể tham khảo Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017`.
1. Thay vì được làm thủ công, có thể học được sự đàn áp không tối đa không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:

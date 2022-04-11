# Phát hiện đối tượng đa quy mô
:label:`sec_multiscale-object-detection`

Trong :numref:`sec_anchor`, chúng tôi tạo ra nhiều hộp neo tập trung vào mỗi pixel của một hình ảnh đầu vào. Về cơ bản các hộp neo này đại diện cho các mẫu của các vùng khác nhau của hình ảnh. Tuy nhiên, chúng ta có thể kết thúc với quá nhiều hộp neo để tính toán nếu chúng được tạo cho *every* pixel. Hãy nghĩ về một hình ảnh đầu vào $561 \times 728$. Nếu năm hộp neo có hình dạng khác nhau được tạo ra cho mỗi pixel làm trung tâm của chúng, hơn hai triệu hộp neo ($561 \times 728 \times 5$) cần được dán nhãn và dự đoán trên hình ảnh. 

## Multiscale Neo Boxes
:label:`subsec_multiscale-anchor-boxes`

Bạn có thể nhận ra rằng không khó để giảm các hộp neo trên một hình ảnh. Ví dụ, chúng ta chỉ có thể lấy mẫu một phần nhỏ pixel từ hình ảnh đầu vào để tạo ra các hộp neo tập trung vào chúng. Ngoài ra, ở các quy mô khác nhau, chúng ta có thể tạo ra các số lượng hộp neo khác nhau có kích thước khác nhau. Về mặt trực giác, các đối tượng nhỏ hơn có nhiều khả năng xuất hiện trên một hình ảnh hơn so với các đối tượng lớn hơn. Ví dụ, các đối tượng $1 \times 1$, $1 \times 2$ và $2 \times 2$ có thể xuất hiện trên một hình ảnh $2 \times 2$ theo 4, 2 và 1 cách có thể tương ứng. Do đó, khi sử dụng các hộp neo nhỏ hơn để phát hiện các vật thể nhỏ hơn, chúng ta có thể lấy mẫu nhiều vùng hơn, trong khi đối với các vật thể lớn hơn chúng ta có thể lấy mẫu ít vùng hơn. 

Để chứng minh cách tạo hộp neo ở nhiều thang đo, chúng ta hãy đọc một hình ảnh. Chiều cao và chiều rộng của nó lần lượt là 561 và 728 pixel.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import image, np, npx

npx.set_np()

img = image.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]
h, w
```

Nhớ lại rằng trong :numref:`sec_conv_layer` chúng ta gọi một đầu ra mảng hai chiều của một lớp phức tạp là một bản đồ tính năng. Bằng cách xác định hình dạng bản đồ tính năng, chúng ta có thể xác định các trung tâm của các hộp neo được lấy mẫu đồng đều trên bất kỳ hình ảnh nào. 

Hàm `display_anchors` được định nghĩa dưới đây. [** Chúng tôi tạo các hộp neo (`anchors`) trên bản đồ tính năng (`fmap`) với mỗi đơn vị (pixel) làm trung tâm hộp neo.**] Kể từ khi các giá trị tọa độ trục $(x, y)$ trong các hộp neo (`anchors`) đã được chia cho chiều rộng và chiều cao của bản đồ tính năng (`fmap`), các giá trị này nằm giữa 0 đến 1, cho biết vị trí tương đối của các hộp neo trong bản đồ tính năng. 

Vì các trung tâm của các hộp neo (`anchors`) được trải rộng trên tất cả các đơn vị trên bản đồ tính năng (`fmap`), các trung tâm này phải được * thống nhất* phân phối trên bất kỳ hình ảnh đầu vào nào về vị trí không gian tương đối của chúng. Cụ thể hơn, với chiều rộng và chiều cao của bản đồ tính năng `fmap_w` và `fmap_h`, tương ứng, chức năng sau đây sẽ * thống nhất* pixel mẫu trong `fmap_h` hàng và `fmap_w` cột trên bất kỳ hình ảnh đầu vào nào. Tập trung vào các pixel được lấy mẫu đồng đều này, các hộp neo có tỷ lệ `s` (giả sử độ dài của danh sách `s` là 1) và các tỷ lệ khung hình khác nhau (`ratios`) sẽ được tạo ra.

```{.python .input}
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = np.zeros((1, 10, fmap_h, fmap_w))
    anchors = npx.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = np.array((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img.asnumpy()).axes,
                    anchors[0] * bbox_scale)
```

```{.python .input}
#@tab pytorch
def display_anchors(fmap_w, fmap_h, s):
    d2l.set_figsize()
    # Values on the first two dimensions do not affect the output
    fmap = d2l.zeros((1, 10, fmap_h, fmap_w))
    anchors = d2l.multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = d2l.tensor((w, h, w, h))
    d2l.show_bboxes(d2l.plt.imshow(img).axes,
                    anchors[0] * bbox_scale)
```

Đầu tiên, chúng ta hãy [** xem xét phát hiện các đối tượng nhỏ**]. Để phân biệt dễ dàng hơn khi được hiển thị, các hộp neo có các trung tâm khác nhau ở đây không chồng chéo: tỷ lệ hộp neo được đặt thành 0,15 và chiều cao và chiều rộng của bản đồ tính năng được đặt thành 4. Chúng ta có thể thấy rằng các trung tâm của các hộp neo trong 4 hàng và 4 cột trên hình ảnh được phân phối đồng đều.

```{.python .input}
#@tab all
display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
```

Chúng tôi chuyển sang [** giảm chiều cao và chiều rộng của bản đồ tính năng bằng một nửa và sử dụng các hộp neo lớn hơn để phát hiện các đối tượng lớn hơn**]. Khi tỷ lệ được đặt thành 0,4, một số hộp neo sẽ chồng lên nhau.

```{.python .input}
#@tab all
display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
```

Cuối cùng, chúng tôi [** tiếp tục giảm chiều cao và chiều rộng của bản đồ tính năng bằng một nửa và tăng tỷ lệ hộp neo lên 0,8**]. Bây giờ trung tâm của hộp neo là trung tâm của hình ảnh.

```{.python .input}
#@tab all
display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
```

## Phát hiện đa quy mô

Vì chúng tôi đã tạo ra các hộp neo đa quy mô, chúng tôi sẽ sử dụng chúng để phát hiện các vật thể có kích thước khác nhau ở các quy mô khác nhau. Sau đây, chúng tôi giới thiệu phương pháp phát hiện đối tượng đa quy mô dựa trên CNN mà chúng tôi sẽ triển khai trong :numref:`sec_ssd`. 

Ở một số quy mô, nói rằng chúng tôi có $c$ tính năng bản đồ hình dạng $h \times w$. Sử dụng phương pháp trong :numref:`subsec_multiscale-anchor-boxes`, chúng tôi tạo ra $hw$ bộ hộp neo, trong đó mỗi bộ có $a$ hộp neo có cùng tâm. Ví dụ, ở thang điểm đầu tiên trong các thí nghiệm trong :numref:`subsec_multiscale-anchor-boxes`, cho mười (số kênh) bản đồ tính năng $4 \times 4$, chúng tôi tạo ra 16 bộ hộp neo, trong đó mỗi bộ chứa 3 hộp neo với cùng một trung tâm. Tiếp theo, mỗi hộp neo được dán nhãn với lớp và bù đắp dựa trên các hộp giới hạn chân lý mặt đất. Ở quy mô hiện tại, mô hình phát hiện đối tượng cần dự đoán các lớp và bù đắp của $hw$ bộ hộp neo trên ảnh đầu vào, trong đó các bộ khác nhau có các trung tâm khác nhau. 

Giả sử rằng các bản đồ tính năng $c$ ở đây là các đầu ra trung gian thu được bởi CNN chuyển tiếp tuyên truyền dựa trên hình ảnh đầu vào. Vì có $hw$ vị trí không gian khác nhau trên mỗi bản đồ tính năng, cùng một vị trí không gian có thể được coi là có $c$ đơn vị. Theo định nghĩa của trường tiếp nhận trong :numref:`sec_conv_layer`, các đơn vị $c$ này ở cùng vị trí không gian của các bản đồ tính năng có cùng trường tiếp nhận trên hình ảnh đầu vào: chúng đại diện cho thông tin hình ảnh đầu vào trong cùng một trường tiếp nhận. Do đó, chúng ta có thể biến đổi các đơn vị $c$ của các bản đồ tính năng ở cùng một vị trí không gian thành các lớp và bù đắp của các hộp neo $a$ được tạo ra bằng cách sử dụng vị trí không gian này. Về bản chất, chúng ta sử dụng thông tin của hình ảnh đầu vào trong một trường tiếp nhận nhất định để dự đoán các lớp và bù đắp của các hộp neo gần với trường tiếp nhận đó trên hình ảnh đầu vào. 

Khi các bản đồ tính năng ở các lớp khác nhau có các trường tiếp nhận kích thước khác nhau trên hình ảnh đầu vào, chúng có thể được sử dụng để phát hiện các đối tượng có kích thước khác nhau. Ví dụ, chúng ta có thể thiết kế một mạng nơ-ron trong đó các đơn vị của các bản đồ tính năng gần với lớp đầu ra có các trường tiếp nhận rộng hơn, do đó chúng có thể phát hiện các đối tượng lớn hơn từ hình ảnh đầu vào. 

Tóm lại, chúng ta có thể tận dụng các biểu diễn theo lớp của hình ảnh ở nhiều cấp độ bằng các mạng thần kinh sâu để phát hiện đối tượng đa quy mô. Chúng tôi sẽ chỉ ra cách điều này hoạt động thông qua một ví dụ cụ thể trong :numref:`sec_ssd`. 

## Tóm tắt

* Ở nhiều thang đo, chúng ta có thể tạo ra các hộp neo với các kích cỡ khác nhau để phát hiện các vật thể có kích thước khác nhau.
* Bằng cách xác định hình dạng của bản đồ tính năng, chúng ta có thể xác định các trung tâm của các hộp neo được lấy mẫu đồng đều trên bất kỳ hình ảnh nào.
* Chúng tôi sử dụng thông tin của hình ảnh đầu vào trong một trường tiếp nhận nhất định để dự đoán các lớp và bù đắp của các hộp neo gần với trường tiếp nhận đó trên hình ảnh đầu vào.
* Thông qua deep learning, chúng ta có thể tận dụng các biểu diễn theo lớp của hình ảnh ở nhiều cấp độ để phát hiện đối tượng đa quy mô.

## Bài tập

1. Theo các cuộc thảo luận của chúng tôi trong :numref:`sec_alexnet`, các mạng thần kinh sâu tìm hiểu các tính năng phân cấp với mức độ trừu tượng ngày càng tăng cho hình ảnh. Trong phát hiện đối tượng đa quy mô, các bản đồ tính năng ở các thang đo khác nhau có tương ứng với các mức độ trừu tượng khác nhau không? Tại sao hoặc tại sao không?
1. Ở thang điểm đầu tiên (`fmap_w=4, fmap_h=4`) trong các thí nghiệm trong :numref:`subsec_multiscale-anchor-boxes`, tạo ra các hộp neo phân bố đồng đều có thể chồng lên nhau.
1. Đưa ra một biến bản đồ tính năng với hình dạng $1 \times c \times h \times w$, trong đó $c$, $h$ và $w$ là số kênh, chiều cao và chiều rộng của bản đồ tính năng, tương ứng. Làm thế nào bạn có thể chuyển đổi biến này thành các lớp và bù đắp của hộp neo? Hình dạng của đầu ra là gì?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/371)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1607)
:end_tab:

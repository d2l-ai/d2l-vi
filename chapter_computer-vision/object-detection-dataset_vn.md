<!--
# The Object Detection Dataset
-->

# Tập dữ liệu Phát hiện Đối tượng


<!--
There are no small datasets, like MNIST or Fashion-MNIST, in the object detection field.
In order to quickly test models, we are going to assemble a small dataset.
First, we generate 1000 Pikachu images of different angles and sizes using an open source 3D Pikachu model.
Then, we collect a series of background images and place a Pikachu image at a random position on each image.
We use the [im2rec tool](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) provided by MXNet to convert the images to binary RecordIO format[1].
This format can reduce the storage overhead of the dataset on the disk and improve the reading efficiency.
If you want to learn more about how to read images, refer to the documentation for the [GluonCV Toolkit](https://gluon-cv.mxnet.io/).
-->

Không có một bộ dữ liệu nhỏ nào cho bài toán phát hiện đối tượng như MNIST hay Fashion-MNIST.
Để nhanh chóng kiểm định mô hình, ta sẽ tập hợp lại một tập dữ liệu nhỏ.
Đầu tiên, ta tạo 1000 bức ảnh Pikachu với nhiều góc độ và kích thước khác nhau bằng mô hình mã nguồn mở Pikachu 3D.
Sau đó, ta thu thập một loạt các ảnh nền và đặt ngẫu nhiên ảnh Pikachu lên trên mỗi bức ảnh.
Ta dùng [công cụ im2rec](https://github.com/apache/incubator-mxnet/blob/master/tools/im2rec.py) do MXNet cung cấp để chuyển đổi hình ảnh gốc sang định dạng RecordIO nhị phân[1].
Định dạng này có khả năng giảm dung lượng lưu trữ và cải thiện hiệu suất đọc dữ liệu.
Nếu bạn đọc muốn tìm hiểu thêm về cách đọc ảnh, có thể tham khảo tài liệu cho [bộ công cụ GluonCV](https://gluon-cv.mxnet.io/).

<!--
## Downloading the Dataset
-->

## Tải xuống tập Dữ liệu


<!--
The Pikachu dataset in RecordIO format can be downloaded directly from the Internet.
-->

Tập dữ liệu Pikachu ở định dạng RecordIO có thể được tải xuống trực tiếp từ Internet.


```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()

#@save
d2l.DATA_HUB['pikachu'] = (d2l.DATA_URL + 'pikachu.zip',
                           '68ab1bd42143c5966785eb0d7b2839df8d570190')
```


<!--
## Reading the Dataset
-->

## Đọc dữ liệu


<!--
We are going to read the object detection dataset by creating the instance `ImageDetIter`.
The "Det" in the name refers to Detection.
We will read the training dataset in random order.
Since the format of the dataset is RecordIO, we need the image index file `'train.idx'` to read random minibatches.
In addition, for each image of the training set, we will use random cropping and require the cropped image to cover at least 95% of each object.
Since the cropping is random, this requirement is not always satisfied.
We preset the maximum number of random cropping attempts to 200. If none of them meets the requirement, the image will not be cropped.
To ensure the certainty of the output, we will not randomly crop the images in the test dataset.
We also do not need to read the test dataset in random order.
-->

Ta sẽ đọc tập dữ liệu phát hiện đối tượng theo thứ tự ngẫu nhiên bằng thực thể `ImageDetIter`.
"Det" (viết tắt cho Detection), đề cập đến việc phát hiện.
Vì định dạng của dữ liệu là RecordIO, ta cần có tệp chỉ số `'train.idx'` để đọc minibatch ngẫu nhiên.
Ngoài ra, đối với từng ảnh trong tập huấn luyện, ta sẽ cắt xén ngẫu nhiên nhưng vẫn yêu cầu bao phủ ít nhất 95% mỗi đối tượng.
Vì việc cắt xén là ngẫu nhiên, yêu cầu này không phải lúc nào cũng được thỏa mãn.
Ta cho trước số lần cắt ảnh ngẫu nhiên tối đa là 200 lần. Nếu không có lần nào thỏa mãn yêu cầu, hình ảnh sẽ được giữ nguyên.
Để đầu ra được đảm bảo, ta sẽ không cắt ngẫu nhiên các hình ảnh trong tập kiểm tra.
Ta cũng không cần đọc dữ liệu trong tập kiểm tra theo thứ tự ngẫu nhiên.



```{.python .input  n=2}
#@save
def load_data_pikachu(batch_size, edge_size=256):
    """Load the pikachu dataset."""
    data_dir = d2l.download_extract('pikachu')
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # The shape of the output image
        shuffle=True,  # Read the dataset in random order
        rand_crop=1,  # The probability of random cropping is 1
        min_object_covered=0.95, max_attempts=200)
    val_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'val.rec'), batch_size=batch_size,
        data_shape=(3, edge_size, edge_size), shuffle=False)
    return train_iter, val_iter
```


<!--
Below, we read a minibatch and print the shape of the image and label.
The shape of the image is the same as in the previous experiment (batch size, number of channels, height, width).
The shape of the label is (batch size, $m$, 5), where $m$ is equal to the maximum number of bounding boxes contained in a single image in the dataset.
Although computation for the minibatch is very efficient, it requires each image to contain the same number of bounding boxes so that they can be placed in the same batch.
Since each image may have a different number of bounding boxes, we can add illegal bounding boxes to images that have less than $m$ bounding boxes until each image contains $m$ bounding boxes.
Thus, we can read a minibatch of images each time.
The label of each bounding box in the image is represented by an array of length 5.
The first element in the array is the category of the object contained in the bounding box.
When the value is -1, the bounding box is an illegal bounding box for filling purpose.
The remaining four elements of the array represent the $x, y$ axis coordinates of the upper-left corner of the bounding box 
and the $x, y$ axis coordinates of the lower-right corner of the bounding box (the value range is between 0 and 1).
The Pikachu dataset here has only one bounding box per image, so $m=1$.
-->

Dưới đây, ta đọc một minibatch rồi in ra kích thước ảnh và nhãn.
Kích thước ảnh vẫn là (kích thước batch, số kênh, chiều cao, chiều rộng) giống như trong thí nghiệm trước.
Kích thước của nhãn là (kích thước batch, $m$, 5), trong đó $m$ là số lượng khung chứa tối đa trên một bức ảnh trong một tập dữ liệu hình ảnh.
Mặc dù việc tính toán với minibatch rất hiệu quả, nhưng nó lại yêu cầu mỗi hình ảnh phải cùng một lượng khung chứa để chúng có thể được đặt trong cùng một batch.
Vì mỗi hình ảnh có thể có số lượng khung chứa khác nhau, ta có thể thêm các khung chứa ngẫu nhiên để mỗi bức ảnh có $m$ khung chứa.
Do đó, chúng ta có thể đọc được một minibatch mỗi lần.
Nhãn của mỗi khung chứa trong bức ảnh được biểu diễn bằng một mảng có độ dài là 5.
Phần tử đầu tiên trong mảng là hạng mục của đối tượng xuất hiện trong khung chứa.
Khi giá trị là -1, khung chứa ấy chính là khung chứa ngẫu nhiên dùng để lấp đầy.
Bốn phần tử còn lại trong mảng đại diện cho toạ độ trục $x, y$ của góc trên bên trái và góc dưới bên phải của khung chứa (miền giá trị từ 0 đến 1).
Tập dữ liệu Pikachu ở đây chỉ có một khung chứa cho mỗi ảnh, vì thế $m=1$.



```{.python .input  n=3}
batch_size, edge_size = 32, 256
train_iter, _ = load_data_pikachu(batch_size, edge_size)
batch = train_iter.next()
batch.data[0].shape, batch.label[0].shape
```


<!--
## Demonstration
-->

## Minh hoạ


<!--
We have ten images with bounding boxes on them.
We can see that the angle, size, and position of Pikachu are different in each image.
Of course, this is a simple artificial dataset.
In actual practice, the data are usually much more complicated.
-->

Ta có mười bức ảnh kèm với các khung chứa trên chúng.
Chúng ta có thể thấy rằng góc, kích thước và vị trí của Pikachu khác nhau trong mỗi bức ảnh.
Dĩ nhiên, đây là một tập dữ liệu tự tạo đơn giản.
Trong thực tế, dữ liệu thường phức tạp hơn nhiều.


```{.python .input  n=4}
imgs = (batch.data[0][0:10].transpose(0, 2, 3, 1)) / 255
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, batch.label[0][0:10]):
    d2l.show_bboxes(ax, [label[0][1:5] * edge_size], colors=['w'])
```


## Tóm tắt

<!--
* The Pikachu dataset we synthesized can be used to test object detection models.
* The data reading for object detection is similar to that for image classification. 
However, after we introduce bounding boxes, the label shape and image augmentation (e.g., random cropping) are changed.
-->

* Tập dữ liệu Pikachu mà ta tổng hợp có thể được dùng để kiểm tra các mô hình phát hiện đối tượng.
* Việc đọc dữ liệu để phát hiện đối tượng giống như khi phân loại hình ảnh.
Tuy nhiên, sau khi đưa ra các khung chứa, kích thước nhãn và việc tăng cường ảnh (ví dụ, cắt xén ngẫu nhiên) sẽ thay đổi.


## Bài tập


<!--
Referring to the MXNet documentation, what are the parameters for the constructors of the `image.ImageDetIter` and `image.CreateDetAugmenter` classes? What is their significance?
-->

Trong tài liệu MXNet, tham số trong các hàm tạo (*constructors*) của các lớp `image.ImageDetIter` và `image.CreateDetAugmenter` là gì? Cho biết ý nghĩa của chúng?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/372)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Phạm Đăng Khoa
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường

<!--
# Computer Vision
-->

# Thị giác Máy tính
:label:`chap_cv`

<!--
Many applications in the area of computer vision are closely related to our daily lives, now and in the future, whether medical diagnostics, driverless vehicles, camera monitoring, or smart filters.
In recent years, deep learning technology has greatly enhanced computer vision systems' performance.
It can be said that the most advanced computer vision applications are nearly inseparable from deep learning.
-->

Nhiều ứng dụng trong lĩnh vực thị giác máy tính có liên quan mật thiết đến cuộc sống hàng ngày của chúng ta trong hiện tại và tương lai, từ chẩn đoán y tế, xe không người lái, camera giám sát tới bộ lọc thông minh.
Trong những năm gần đây, công nghệ học sâu đã nâng cao đáng kể hiệu năng của hệ thống thị giác máy tính.
Có thể nói rằng các ứng dụng thị giác máy tính tiên tiến nhất gần như không thể tách rời khỏi học sâu.

<!--
We have introduced deep learning models commonly used in the area of computer vision in the chapter "Convolutional Neural Networks" and have practiced simple image classification tasks.
In this chapter, we will introduce image augmentation and fine tuning methods and apply them to image classification.
Then, we will explore various methods of object detection.
After that, we will learn how to use fully convolutional networks to perform semantic segmentation on images.
Then, we explain how to use style transfer technology to generate images that look like the cover of this book.
Finally, we will perform practice exercises on two important computer vision datasets to review the content of this chapter and the previous chapters.
-->

Chúng tôi đã giới thiệu các mô hình học sâu thường được sử dụng trong lĩnh vực thị giác máy tính trong chương "Mạng nơ-ron tích chập" và đã thực hành một số tác vụ phân loại hình ảnh đơn giản.
Trong chương này, chúng tôi sẽ giới thiệu các phương pháp tăng cường hình ảnh và các phương pháp tinh chỉnh và áp dụng chúng vào phân loại hình ảnh.
Sau đó, ta sẽ khám phá các phương pháp nhận diện vật thể khác nhau.
Sau đó nữa, ta sẽ tìm hiểu cách sử dụng các mạng chập hoàn toàn để thực hiện phân vùng ngữ nghĩa trên hình ảnh.
Sau đó, chúng tôi giải thích cách sử dụng kỹ thuật chuyển phong cách để tạo ra hình ảnh trông giống như bìa của cuốn sách này.
Cuối cùng, chúng tôi sẽ thực hiện các bài tập thực hành trên hai bộ dữ liệu thị giác máy tính quan trọng để xem lại nội dung của chương này và các chương trước.


```toc
:maxdepth: 2

image-augmentation_vn
fine-tuning_vn
bounding-box_vn
anchor_vn
multiscale-object-detection_vn
object-detection-dataset_vn
ssd_vn
rcnn_vn
semantic-segmentation-and-dataset_vn
transposed-conv_vn
fcn_vn
neural-style_vn
kaggle-cifar10_vn
kaggle-dog_vn
```


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
* Trần Yến Thy

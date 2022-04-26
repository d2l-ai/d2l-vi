# Tầm nhìn máy tính
:label:`chap_cv`

Cho dù đó là chẩn đoán y tế, xe tự lái, giám sát camera hoặc bộ lọc thông minh, nhiều ứng dụng trong lĩnh vực thị giác máy tính có liên quan chặt chẽ đến cuộc sống hiện tại và tương lai của chúng ta. Trong những năm gần đây, deep learning là sức mạnh biến đổi để thúc đẩy hiệu suất của các hệ thống thị giác máy tính. Có thể nói rằng các ứng dụng tầm nhìn máy tính tiên tiến nhất gần như không thể tách rời khỏi học sâu. Theo quan điểm này, chương này sẽ tập trung vào lĩnh vực thị giác máy tính, và điều tra các phương pháp và ứng dụng gần đây đã có ảnh hưởng trong học viện và ngành công nghiệp. 

Trong :numref:`chap_cnn` và :numref:`chap_modern_cnn`, chúng tôi đã nghiên cứu các mạng thần kinh phức tạp khác nhau thường được sử dụng trong tầm nhìn máy tính và áp dụng chúng cho các tác vụ phân loại hình ảnh đơn giản. Vào đầu chương này, chúng tôi sẽ mô tả hai phương pháp có thể cải thiện khái quát hóa mô hình, đó là *image augmentation* và *fine-tuning*, và áp dụng chúng vào phân loại hình ảnh. Vì các mạng thần kinh sâu có thể đại diện hiệu quả hình ảnh ở nhiều cấp độ, nên các biểu diễn theo lớp như vậy đã được sử dụng thành công trong các tác vụ thị giác máy tính khác nhau như phát hiện đối tượng*, * phân đoạn ngữ nghị* và chuyển kiểu *. Theo ý tưởng chính của việc tận dụng các biểu diễn theo lớp trong tầm nhìn máy tính, chúng ta sẽ bắt đầu với các thành phần và kỹ thuật chính để phát hiện đối tượng. Tiếp theo, chúng tôi sẽ chỉ ra cách sử dụng mạng * hoàn toàn phức tạp * để phân đoạn hình ảnh ngữ nghĩa. Sau đó, chúng tôi sẽ giải thích cách sử dụng các kỹ thuật chuyển phong cách để tạo ra hình ảnh như bìa của cuốn sách này. Cuối cùng, chúng tôi kết thúc chương này bằng cách áp dụng các tài liệu của chương này và một số chương trước trên hai bộ dữ liệu điểm chuẩn tầm nhìn máy tính phổ biến.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```

# Mạng thần kinh phức tạp hiện đại
:label:`chap_modern_cnn`

Bây giờ chúng tôi hiểu những điều cơ bản của hệ thống dây điện với nhau CNN, chúng tôi sẽ đưa bạn qua một chuyến tham quan các kiến trúc CNN hiện đại. Trong chương này, mỗi phần tương ứng với một kiến trúc CNN quan trọng đó là tại một số điểm (hoặc hiện tại) mô hình cơ sở mà trên đó nhiều dự án nghiên cứu và hệ thống triển khai đã được xây dựng. Mỗi mạng này là một kiến trúc thống trị ngắn gọn và nhiều người là người chiến thắng hoặc á quân trong cuộc thi ImageNet, đã phục vụ như một phong vũ biểu tiến bộ về học tập được giám sát trong tầm nhìn máy tính kể từ năm 2010. 

Các mô hình này bao gồm AlexNet, mạng quy mô lớn đầu tiên được triển khai để đánh bại các phương pháp tầm nhìn máy tính thông thường trên một thách thức tầm nhìn quy mô lớn; mạng VGG, sử dụng một số khối lặp lại của các yếu tố; mạng trong mạng (Nin) mà xoay quanh toàn bộ mạng thần kinh patch-wise qua đầu vào; GoogLeNet, sử dụng mạng có kết nối song song; mạng dư (ResNet), vẫn là kiến trúc ngoài kệ phổ biến nhất trong tầm nhìn máy tính; và các mạng kết nối mật độ (DenseNet), đắt tiền để tính toán nhưng đã đặt ra một số điểm chuẩn gần đây. 

Mặc dù ý tưởng về mạng nơ-ron * deep* khá đơn giản (xếp chồng lên nhau một loạt các lớp), hiệu suất có thể thay đổi rất nhiều trên các kiến trúc và các lựa chọn siêu tham số. Các mạng thần kinh được mô tả trong chương này là sản phẩm của trực giác, một vài hiểu biết toán học, và rất nhiều thử nghiệm và sai. Chúng tôi trình bày các mô hình này theo thứ tự thời gian, một phần để truyền đạt ý thức về lịch sử để bạn có thể hình thành trực giác của riêng bạn về nơi lĩnh vực đang hướng đến và có lẽ phát triển kiến trúc của riêng bạn. Ví dụ, bình thường hóa hàng loạt và các kết nối còn lại được mô tả trong chương này đã đưa ra hai ý tưởng phổ biến để đào tạo và thiết kế các mô hình sâu.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
```

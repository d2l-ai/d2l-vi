# Mạng thần kinh phức tạp
:label:`chap_cnn`

Trong các chương trước đó, chúng tôi đã đưa ra dữ liệu hình ảnh, trong đó mỗi ví dụ bao gồm một lưới pixel hai chiều. Tùy thuộc vào việc chúng ta đang xử lý hình ảnh đen trắng hay màu, mỗi vị trí pixel có thể được liên kết với một hoặc nhiều giá trị số, tương ứng. Cho đến bây giờ, cách đối phó với cấu trúc phong phú này là vô cùng không thỏa mãn. Chúng tôi chỉ đơn giản là loại bỏ cấu trúc không gian của mỗi hình ảnh bằng cách làm phẳng chúng thành các vectơ một chiều, cho chúng ăn thông qua một MLP được kết nối hoàn toàn. Bởi vì các mạng này không thay đổi theo thứ tự của các tính năng, chúng ta có thể nhận được kết quả tương tự bất kể chúng ta giữ nguyên một thứ tự tương ứng với cấu trúc không gian của pixel hay nếu chúng ta hoán vị các cột của ma trận thiết kế trước khi lắp các tham số của MLP. Tốt nhất, chúng tôi sẽ tận dụng kiến thức trước đây của chúng tôi rằng các pixel gần đó thường liên quan đến nhau, để xây dựng các mô hình hiệu quả để học từ dữ liệu hình ảnh.  

Chương này giới thiệu *mạng thần kinh phức lượng* (CNN), một họ mạng thần kinh mạnh mẽ được thiết kế cho mục đích chính xác này. Các kiến trúc dựa trên CNN hiện đang phổ biến trong lĩnh vực thị giác máy tính, và đã trở nên chiếm ưu thế đến mức hầu như không ai ngày nay sẽ phát triển một ứng dụng thương mại hoặc tham gia một cuộc thi liên quan đến nhận dạng hình ảnh, phát hiện đối tượng hoặc phân khúc ngữ nghĩa, mà không cần xây dựng cách tiếp cận này. 

CNN hiện đại, vì chúng được gọi là thông tục nợ thiết kế của họ để truyền cảm hứng từ sinh học, lý thuyết nhóm và một liều lượng lành mạnh của mày mò thử nghiệm. Ngoài hiệu quả mẫu của chúng trong việc đạt được các mô hình chính xác, CNN có xu hướng hiệu quả về mặt tính toán, cả vì chúng yêu cầu ít tham số hơn các kiến trúc được kết nối hoàn toàn và vì các phức tạp dễ song song trên các lõi GPU. Do đó, các học viên thường áp dụng CNN bất cứ khi nào có thể, và ngày càng họ nổi lên như đối thủ cạnh tranh đáng tin cậy ngay cả trên các nhiệm vụ có cấu trúc trình tự một chiều, chẳng hạn như phân tích chuỗi âm thanh, văn bản và chuỗi thời gian, nơi các mạng thần kinh tái phát được sử dụng thông thường. Một số sự thích nghi thông minh của CNN cũng đã đưa chúng chịu đựng dữ liệu có cấu trúc đồ thị và trong các hệ thống giới thiệu. 

Đầu tiên, chúng ta sẽ đi qua các hoạt động cơ bản bao gồm xương sống của tất cả các mạng phức tạp. Chúng bao gồm các lớp phức tạp, chi tiết nitty-gritty bao gồm đệm và sải chân, các lớp tổng hợp được sử dụng để tổng hợp thông tin trên các vùng không gian liền kề, sử dụng nhiều kênh ở mỗi lớp, và một cuộc thảo luận cẩn thận về cấu trúc của kiến trúc hiện đại. Chúng tôi sẽ kết thúc chương với một ví dụ làm việc đầy đủ về LeNet, mạng phức tạp đầu tiên được triển khai thành công, rất lâu trước khi sự gia tăng của học sâu hiện đại. Trong chương tiếp theo, chúng ta sẽ đi sâu vào triển khai đầy đủ một số kiến trúc CNN phổ biến và tương đối gần đây có thiết kế đại diện cho hầu hết các kỹ thuật thường được sử dụng bởi các học viên hiện đại.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```

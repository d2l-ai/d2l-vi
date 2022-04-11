# Cơ chế chú ý
:label:`chap_attention`

Dây thần kinh thị giác của hệ thống thị giác của linh trưởng nhận được đầu vào cảm giác lớn, vượt xa những gì não có thể xử lý hoàn toàn. May mắn thay, không phải tất cả các kích thích đều được tạo ra bằng nhau. Sự tập trung và tập trung ý thức đã cho phép các loài linh trưởng chú ý trực tiếp đến các đối tượng quan tâm, chẳng hạn như săn mồi và động vật ăn thịt, trong môi trường thị giác phức tạp. Khả năng chú ý đến chỉ một phần nhỏ của thông tin có ý nghĩa tiến hóa, cho phép con người sống và thành công. 

Các nhà khoa học đã nghiên cứu sự chú ý trong lĩnh vực khoa học thần kinh nhận thức từ thế kỷ 19. Trong chương này, chúng ta sẽ bắt đầu bằng cách xem xét một khuôn khổ phổ biến giải thích cách sự chú ý được triển khai trong một cảnh trực quan. Lấy cảm hứng từ các tín hiệu chú ý trong khuôn khổ này, chúng tôi sẽ thiết kế các mô hình tận dụng các tín hiệu chú ý như vậy. Đáng chú ý là hồi quy hạt nhân Nadaraya-Waston vào năm 1964 là một minh chứng đơn giản về học máy với cơ chế chú ý*. 

Tiếp theo, chúng tôi sẽ tiếp tục giới thiệu các chức năng chú ý đã được sử dụng rộng rãi trong thiết kế các mô hình chú ý trong học sâu. Cụ thể, chúng tôi sẽ chỉ ra cách sử dụng các chức năng này để thiết kế sự chú ý * Bahdanau*, một mô hình chú ý đột phá trong học sâu có thể sắp xếp hai chiều và có thể khác biệt. 

Cuối cùng, được trang bị gần đây hơn
*sự chú ý nhiều đầu*
và *self-attention* thiết kế, chúng tôi sẽ mô tả kiến trúc * transformer* chỉ dựa trên cơ chế chú ý. Kể từ khi đề xuất của họ vào năm 2017, các máy biến áp đã phổ biến trong các ứng dụng học sâu hiện đại, chẳng hạn như trong các lĩnh vực ngôn ngữ, tầm nhìn, lời nói và học tập củng cố.

```toc
:maxdepth: 2

attention-cues
nadaraya-watson
attention-scoring-functions
bahdanau-attention
multihead-attention
self-attention-and-positional-encoding
transformer
```

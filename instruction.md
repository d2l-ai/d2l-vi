## Giới thiệu từ nhóm dịch

Trong những năm gần đây, học sâu là một trong các lĩnh vực được quan tâm nhiều nhất
trong các trường đại học kỹ thuật cũng như các công ty công nghệ. Ngày càng nhiều
các diễn đàn liên quan đến học máy và học sâu với lượng thành phiên và câu hỏi ngày
một tăng. Một trong các diễn đàn tiếng Việt nổi bật nhất là [Forum Machine Learning cơ bản](https://www.facebook.com/groups/machinelearningcoban/) và [Diễn đàn Machine Learning cơ bản](https://forum.machinelearningcoban.com/) với hơn 35 ngàn thành viên và hàng chục câu hỏi mỗi ngày.

Qua các diễn đàn đó, chúng tôi nhận ra rằng nhu cầu theo học lĩnh vực này ngày
một tăng trong khi lượng tài liệu tiếng Việt còn rất hạn chế. Đặc biệt, chúng tôi
nhận thấy rằng các tài liệu tiếng Việt còn chưa nhất quán trong cách dịch, việc này
khiến những người theo học lĩnh vực này bị bối rối trước quá nhiều thông tin nhưng
lại quá ít thông tin đầy đủ. Việc này thúc đẩy chúng tôi tìm và dịch những cuốn sách
được quan tâm nhiều về lĩnh vực này.

Nhóm dịch đã bước đầu thành công khi dịch cuốn [Machine Learning Yearning](https://github.com/aivivn/Machine-Learning-Yearning-Vietnamese-Translation/blob/master/chapters/all_chapters.md)
của tác giả Andrew Ng. Cuốn sách này đề cập đến các vấn đề cần lưu ý khi xây dựng
các hệ thống học máy, trong đó đề cập đến nhiều kiến thức thực tế khi thực hiện dự 
án. Tuy nhiên, cuốn sách này phần nào hướng tới những người đã có những kinh nghiệm
nhất định đã đang tham gia các dự án học máy. Chúng tôi vẫn khao khát được mang một
tài liệu đầy đủ hơn với đủ kiến thức toán nền tảng, cách triển khai các công thức
toán bằng mã nguồn, cùng với cách triển khai một hệ thống thực tế trên một nền tảng
học sâu được nhiều người sử dụng. Và quan trọng hơn, các kiến thức này phải cập nhật
các xu hướng học máy mới nhất.

Sau nhiều ngày tìm kiếm các cuốn sách về học máy/học sâu gây được nhiều chú ý, chúng 
tôi quyết định dịch cuốn [Dive into Deep Learning](https://www.d2l.ai/) của nhóm tác 
giả từ công ty Amazon. Cuốn này hội tụ đủ các yếu tố: có giải thích toán dễ hiểu,
có code đi kèm cho những ai không muốn đọc toán, cập nhật đầy đủ những khía cạnh của
học sâu, và quan trọng nhất là không đòi hỏi bản quyền để dịch. Chúng tôi đã liên hệ
với nhóm tác giả và họ rất vui mừng khi cuốn sách sắp được phổ biến rộng rãi hơn nữa.

Hiện cuốn sách vẫn đang được thực hiện và sắp ra mắt phiên bản 0.7.0. Nhóm tác giả
có lời khuyên chúng tôi có thể dịch bản 0.7.0 này ở branch
[numpy2](https://github.com/d2l-ai/d2l-en/tree/numpy2) và có thể cập nhật khi cuốn 
sách được xuất bản. Chúng tôi cũng chọn bản này vì nó sử dụng thư viện chính là
`numpy` (tích hợp trong MXNet), một thư viện xử lý mảng nhiều chiều phổ biến mà theo
chúng tôi, người làm về học máy, học sâu và khoa học dữ liệu cần biết.

Để có thể thực hiện dịch dự án dịch cuốn sách hơn 800 trang này, chúng tôi rất cần
sự chung tay của cộng đồng. Mọi sự đóng góp đều đáng quý và sẽ được ghi nhận. Chúng tôi hy vọng cuốn sách sẽ được hoàn thành trong một năm. Và sau đó nó có thể trở thành
giáo trình trong các trường đại học. Hy vọng một ngày chúng ta có thể nhìn thấy
một trường của Việt Nam trong danh sách này:

![img](https://i.ibb.co/M2ZXzP6/Screen-Shot-2019-11-27-at-6-37-04-PM.png)

## Hướng dẫn đóng góp

Có ba công việc chính bạn có thể đóng góp vào dự án: Dịch, Review, và Hỗ trợ kỹ thuật.

### Dịch
Nếu bạn đã quen với GitHub, bạn có thể tham khảo cách [đóng góp vào một dự án GitHub](https://codetot.net/contribute-github/). Cách này yêu cầu người đóng góp tạo một forked repo rồi tạo pull request từ forked repo đó. Sẽ có thể phức tạp với các bạn chưa quen với GitHub.

Ngoài ra, có một cách đơn giản hơn mà bạn có thể trực tiếp dịch trên trình duyệt mà không cần cài đặt Git hay fork repo này về GitHub của bạn. Bạn có thể tham khảo :n

Tất nhiên bạn vẫn cần tạo một GitHub account để làm việc này.

### Review
Chọn một Pull Request trong [danh sách này](https://github.com/aivivn/d2l-vn/pulls) và bắt đầu review.

Khi Review, bạn có thể đề xuất thay đổi cách dịch mỗi dòng trực tiếp như trong hình dưới đây:
![img](https://user-images.githubusercontent.com/19977/58752991-f39d0880-846c-11e9-8c03-c7aded86ee9b.png)

Nếu bạn có những phản hồi hữu ích, tên của bạn sẽ được tác giả chính của Pull Request đó điền vào cuối file mục "Những người thực hiện".

### Hỗ trợ kỹ thuật
Để phục vụ cho việc dịch trên quy mô lớn, nhóm dịch cần một số bạn hỗ trợ kỹ thuật cho một số việc dưới đây:

* Lấy các bản gốc từ [bản tiếng Anh](https://github.com/d2l-ai/d2l-en/tree/numpy2). Vì bản này tác giả vẫn cập nhật nên dịch đến đâu chúng ta sẽ cập nhật đến đó.

* Tự động thêm comment vào các bản gốc (`<!--` và `-->`) để các phần này không hiển thị trên [trang web chính](https://d2l.aivivn.com/). Phần thêm này có thể được thực hiện tự động bằng cách chạy:
```
python3 utils --convert <path_to_file>.md
```
và tạo ra file `<path_to_file>_vn.md`.

* Chia các file lớn thành các mục nhỏ như trong [ví dụ này](https://github.com/aivivn/d2l-vn/blame/master/chapter_preface/index_vn.md). Phần này cần thực hiện bằng tay. Mỗi phần trong file nên bao gồm những mục cụ thể, không bắt đầu và kết thúc giữa chừng của một mục.

* Dịch các chữ trong hình vẽ theo Bảng thuật ngữ. Sẵn sàng đổi các bản dịch này nếu
Bảng thuật ngữ thay đổi

* Hỗ trợ quản lý project trên github, slack và diễn đàn.

**Lưu ý:** Chỉ bắt đầu thực hiện công việc nếu đã có một issue tương ứng được tạo. Nếu bạn thấy một việc cần thiết, hãy tạo issue và thảo luận trước khi thực hiện. Tránh việc dẫm lên chân nhau.

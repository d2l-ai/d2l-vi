## Hướng dẫn đóng góp

Có ba công việc chính bạn có thể đóng góp vào dự án: Dịch, Review, và Hỗ trợ kỹ thuật.

### Dịch

Mỗi Pull Request liên quan tới việc dịch chỉ dịch một phần của một file `.md` nằm giữa hai dòng:
```
<!-- ========================== Begin Part x ================================-->
```
và
```
<!-- ========================== End Part x ================================-->
```

Việc chia nhỏ một file ra nhiều phần khiến một Pull Request mất không quá nhiều thời gian trong cả việc thực hiện lẫn review.

(xem ví dụ [tại đây](https://github.com/aivivn/d2l-vn/blame/master/chapter_preface/index_vn.md#L1-L47).)


**Các bước thực hiện khi dịch một *phần* của một file `.md`:**

1. Tham khảo cách [đóng góp vào một dự án GitHub](https://codetot.net/contribute-github/)

2. Luôn luôn giữ bản forked của mình cập nhật với bản chính trong repo này

3. Tìm các issues liên quan đến việc dịch [tại đây](https://github.com/aivivn/d2l-vn/issues).

4. Dịch và tạo một Pull Request.

5. Trả lời các bình luận từ các reviewers

6. Điền tên mình và tên các reviewer có các phản hồi hữu ích (từ góc nhìn người dịch chính) vào cuối file, mục "Những người thực hiện".


**Lưu ý:**

1. Thuật ngữ
Luôn luôn bám sát [Bảng thuật ngữ](https://github.com/aivivn/d2l-vn/blob/master/glossary.md) khi dịch. Nếu một từ/cụm chưa có trong Bảng thuật ngữ, hãy tạo một Pull Request riêng đề xuất cách dịch từ/cụm đó.

2. Giữ đúng format của bản gốc:
    * Các phần in nghiêng, in đậm
    * Tiêu đề (số lượng dấu `#` đầu dòng)
    * Bảng biểu, chú thích cho bảng (dòng phía trên mỗi bảng bắt đầu bằng dấu `:`)
    * Dịch các từ trong hình vẽ nếu cần. Các dòng có hình có dạng: `![caption](path)`
    * Dịch các chú thích hình vẽ (thay các cụm `*translate the image caption here*` bằng bản dịch tiếng Việt)
    * Không dịch các phần code (nằm giữa hai dấu `````)
    * Copy các công thức toán từ bản gốc (các đoạn có `$`)
    * Giữ các dòng gán nhãn (bắt đầu với `:label:`)
    * Không tự thêm bớt các dòng trắng


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

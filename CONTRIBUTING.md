### Hướng dẫn đóng góp

#### Dịch (thông qua Pull Request)

Mỗi Pull Request liên quan tới việc dịch chỉ dịch một phần của một file `.md` nằm giữa hai dòng:
```
<!-- =================== Bắt đầu dịch Phần x ================================-->
```
và
```
<!-- =================== Kết thúc dịch Phần x ================================-->
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
    * Dịch các chú thích hình vẽ (thay các cụm `*dịch chú thích ảnh phía trên*` bằng bản dịch tiếng Việt)
    * Không dịch các phần code (nằm giữa hai dấu `````)
    * Copy các công thức toán từ bản gốc (các đoạn có `$`)
    * Giữ các dòng gán nhãn (bắt đầu với `:label:`)
    * Không tự thêm bớt các dòng trắng
     
3. Ngắt các đoạn dài thành các dòng ngắn khoảng 60-80 ký tự. Markdown sẽ coi
những dòng liền nhau không có dòng trắng là một đoạn văn. Việc này giúp công đoạn review được thuận tiện hơn.

#### Review

Chọn một Pull Request trong [danh sách này](https://github.com/aivivn/d2l-vn/pulls) và bắt đầu review.

Khi Review, bạn có thể đề xuất thay đổi cách dịch mỗi dòng trực tiếp như trong hình dưới đây:
![img](https://user-images.githubusercontent.com/19977/58752991-f39d0880-846c-11e9-8c03-c7aded86ee9b.png)

Nếu bạn có những phản hồi hữu ích, tên của bạn sẽ được tác giả chính của Pull Request đó điền vào cuối file mục "Những người thực hiện".

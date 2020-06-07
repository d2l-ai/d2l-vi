## Pull Request Checklist

**Đánh dấu tick (sau khi "Create Pull Request") vào các mục dưới đây:**

* [ ] Pull Request này tương ứng với issue nào? Điền số issue sau dấu `#` (không có dấu cách): Close #

* [ ] Bản dịch này có bám sát **[Bảng thuật ngữ](https://github.com/aivivn/d2l-vn/blob/master/glossary.md)** không? Nếu một từ/cụm chưa có trong Bảng thuật ngữ, hãy tạo một Pull Request riêng đề xuất cách dịch từ/cụm đó.

* [ ] Format của bản gốc có được giữ nguyên không?
    * Các phần in nghiêng, in đậm.
    * Tiêu đề (số lượng dấu `#` đầu dòng).
    * Bảng biểu, chú thích cho bảng (dòng phía trên mỗi bảng bắt đầu bằng dấu `:`).
    * Dịch các từ trong hình vẽ nếu cần. Các dòng có hình có dạng: `![caption](path)`.
    * Dịch các chú thích hình vẽ (thay các cụm `*dịch chú thích ảnh phía trên*` bằng bản dịch tiếng Việt).
    * Không dịch các phần code (nằm giữa hai dấu `````).
    * Copy các công thức toán từ bản gốc (các đoạn có `$`).
    * Giữ các dòng gán nhãn (bắt đầu với `:label:`).
    * Không tự thêm bớt các dòng trắng.
     
* [ ] Trong một đoạn văn, mỗi câu văn đã được viết trong một dòng, giữa các dòng không có dòng trắng. Markdown sẽ coi những dòng liền nhau không có dòng trắng là một đoạn văn. Việc này giúp công đoạn review được thuận tiện hơn. Bạn đã kiểm tra rằng mỗi đoạn văn đã nằm một dòng?

* [ ] Điền tên của bạn vào mục "Những người thực hiện" ở cuối file.

* [ ] Điền tên những reviewers mà bạn thấy có nhiều đóng góp cho Pull Request này. Xem danh sách Contributors **[tại đây](https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md)**.

## Với Reviewer

1. Chỉ review nếu Pull Request (PR) này đã sẵn sàng cho việc review. Một PR sẵn sàng khi PR đó không có label `status: wip`.

2. Đã đọc kỹ hướng dẫn đóng góp và xem qua **[Bảng thuật ngữ](https://github.com/aivivn/d2l-vn/blob/master/glossary.md)**.

3. **Bạn có thể đề xuất thay đổi cách dịch mỗi dòng trực tiếp như trong hình dưới đây:**
![img](https://user-images.githubusercontent.com/19977/58752991-f39d0880-846c-11e9-8c03-c7aded86ee9b.png)

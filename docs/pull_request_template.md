# Lưu ý

## Với tác giả của Pull Request

**Pull Request này tương ứng với issue nào? Trả lời số issue sau dấu `#` (không có dấu cách): close #**

Mỗi Pull Request liên quan tới việc dịch chỉ dịch một phần của một file `.md` nằm giữa hai dòng:
```
<!-- =================== Bắt đầu dịch Phần x ================================ -->
```
và
```
<!-- =================== Kết thúc dịch Phần x ================================ -->
```

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
     
3. Xuống dòng sau mỗi câu. Markdown sẽ coi những dòng liền nhau không có dòng trắng là một đoạn văn.
Việc này giúp công đoạn review được thuận tiện hơn.

4. Điền tên của bạn vào mục "Những người thực hiện" ở cuối file.

5. Sau khi được approve, điền tên những reviewers mà bạn thấy có nhiều đóng góp cho Pull Request này.

## Với reviewer

1. **Bạn có thể đề xuất thay đổi cách dịch mỗi dòng trực tiếp như trong hình dưới đây:**
![img](https://user-images.githubusercontent.com/19977/58752991-f39d0880-846c-11e9-8c03-c7aded86ee9b.png)

2. Bám sát [Bảng thuật ngữ](https://github.com/aivivn/d2l-vn/blob/master/glossary.md) khi review. Nếu bạn cho rằng một từ trong bảng thuật ngữ không hợp lý, vui lòng [tạo một issue](https://github.com/aivivn/d2l-vn/issues/new) và bàn luận riêng trong issue đó.

3. Nếu bạn có những phản hồi hữu ích, tên của bạn sẽ được tác giả chính của Pull Request đó điền vào cuối file mục "Những người thực hiện".
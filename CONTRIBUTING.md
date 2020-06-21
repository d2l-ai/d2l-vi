# Hướng dẫn Đóng góp cho Dự án


## Hướng dẫn dịch trực tiếp trên trình duyệt

Cuốn sách này được chia thành nhiều tập tin, mỗi tập tin như vậy chúng tôi chia nhỏ ra thành từng phần nhằm giúp cho việc dịch thuật và phản biện nội dung không lấy đi quá nhiều thời gian của một thành viên khi tham gia đóng góp. 

Các bước dịch như sau:

## Tôi nên bắt đầu như thế nào?

### Bước 1: Tìm và đăng ký phần cần dịch
Bạn cần tìm đến **[tab issue](https://github.com/aivivn/d2l-vn/issues)**, những Issue chưa có người nhận dịch được đánh dấu bằng nhãn `status: help wanted`. Trước khi thực hiện việc dịch thuật phần nội dung được chỉ định trong Issue, bạn cần để lại comment vào trong issue đó và cc @duythanhvn (khuyến khích) hoặc @tiepvuspu để chúng tôi assign cho bạn.

![](./docs/translation-guide/web-step-01.png)
> Khi bạn được assign thì bạn sẽ thấy avatar của mình nằm bên phải của issue, ở issue đầu tiên đang còn nhãn `status: help wanted` thì đây là issue mà bạn có thể comment để nhận dịch.

**Lưu ý:** Bạn chỉ nên bắt đầu việc dịch thuật khi bạn đã được assign vào issue mà mình đã nhận. Chúng tôi ưu tiên bạn bắt đầu với những phần có hai nhãn: `status: help wanted` và `status: phase 1`.

### Bước 3: Tìm file tương ứng
Sau khi vào file tương ứng, ở ví dụ này là [chapter_appendix_math/single-variable-calculus_vn.md](https://github.com/aivivn/d2l-vn/blob/master/chapter_appendix_math/single-variable-calculus_vn.md), bạn click vào nút "Edit" hình chiếc bút chì để bắt đầu tìm phần cần dịch.
![](./docs/translation-guide/step03.png)

### Bước 4: Tìm phần tương ứng
Mỗi phần được bắt đầu bởi dòng:
```
<!-- ===================== Bắt đầu dịch Phần x ==================== -->
```
như trong hình:

![](./docs/translation-guide/step04.png)
chúng ta cần dịch từ sau dòng này đến trước dòng
```
<!-- ===================== Kết thúc dịch Phần x ==================== -->
```
tương ứng.

### Bước 5: Bắt đầu dịch
Khi dịch, bạn tìm các dòng dạng:
```
## *dịch tiêu đề phía trên*

*dịch đoạn phía trên*
```
và chỉ chỉnh sửa các dòng này.

**Một vài quy tắc dịch:**
* Không dịch các danh từ riêng
* Trong một đoạn, mỗi câu dịch nên để riêng một dòng (xem dòng 268, 269, 270 trong hình ví dụ ở **Bước 4**).
* Không dịch code.
* Không nhất thiết phải dừng từng từ từng câu nhưng phải dịch đúng ý.
* Các thuật ngữ cần được dịch một cách nhất quán.
* Nếu một thuật ngữ chưa có trong bảng thuật ngữ, bạn có thể đề xuất một cách dịch bằng cách tạo một PR mới.

Nếu đây là lần đầu tiên bạn đóng góp vào file này, bạn cần kéo xuống cuối file và điền tên mình vào mục "Những người thực hiện" và mục tương ứng. Mục này nhằm ghi nhận đóng góp của bạn.

**Lưu ý: Tên bạn sẽ chỉ xuất hiện trên trang web chính thức nếu Pull Request bạn tạo được merged sau khi trả lời các phản biện.**
![](./docs/translation-guide/step05.png)

### Bước 6: Commit changes
Sau khi dịch lần đầu xong phần của mình, bạn cần kéo xuống cuối trang để "Commit changes". Trước khi click vào nút "Commit changes", bạn cần đặt tiêu đề cho commit, cũng là tiêu đề cho Pull Request bạn sắp tạo. Tiêu đề này giống với tiêu đề trong Issue bạn nhận ban đầu (chỉ cần copy paste là được).


![](./docs/translation-guide/step06.png)
Click "Commit changes".

### Bước 7: Tạo Pull Request
Sau khi click "Commit changes", trang tạo Pull Request sẽ tự động mở ra. Bạn chỉ cần điền số issue tương ứng, trong ví dụ này là 393, vào sau cụm "Close#" như trong hình. Lưu ý không có dấu cách giữa `#` và số issue. Việc này sẽ giúp issue tự động được đóng sau khi Pull Request này được merged.

![](./docs/translation-guide/step07.png)

Click "Create pull request".

Trong trường hợp bạn chưa hoàn thành nội dung cần dịch, hoặc nội dung này bạn mong muốn tự chỉnh sửa để hoàn thiện hơn, bạn hãy lựa chọn tạo một "Draft Pull Request" theo hướng dẫn ở hình dưới.

![](./docs/translation-guide/draft-pull-requests.png)

Sau khi hoàn thiện nội dung, bạn có thể chọn **Ready for review** để Pull request này được nhóm tiến hành review.

![](./docs/translation-guide/draft-pull-requests-ready.png)

### Bước 8: Kiểm tra checklist
Cuối cùng, bạn kiểm tra checklist và click vào các ô tương ứng đã hoàn thành như hình dưới đây.

Trong ví dụ này, phần đã nhận chưa được dịch trọn vẹn nên chưa có dấu tick.

![](./docs/translation-guide/step08.png)

### Sau khi đã nộp Pull Request
Nếu là lần đầu nộp Pull Request, bạn sẽ nhận được vô số bình luận/gợi ý từ các reviewer. **Việc này là hoàn toàn bình thường**, những người làm việc trong nhóm này thường có rất nhiều góp ý xây dựng giúp bản dịch được trọn vẹn và nhất quán với các phần khác. Họ sẽ gợi ý bạn cách sửa, bạn có thể chấp nhận gợi ý hoặc phản hồi lại các phản hồi đó.

### Sau khi Pull Request được approve
Cuối cùng, nếu bạn thấy phần phản hồi nào hữu ích, bạn có thể điền tên user tương ứng vào dưới tên bạn ở mục "Những người thực hiện". Cả người dịch và người review đều xứng đáng được ghi công.

Nếu bạn chưa biết tên đầy đủ của những người đóng góp, bạn có thể xem danh sách Contributors **[tại đây](./docs/contributors_info.md)**. Nếu bạn chưa thấy tên mình tại đây, hãy tạo một Issue mới để được bổ sung.

Cảm ơn đóng góp của bạn.

Thân mến,<br/>
Nhóm dịch thuật Machine Learning Cơ Bản

## Tôi đang trong quá trình hoàn thiện bản dịch của mình

Trong trường hợp bạn chưa hoàn thành nội dung cần dịch nhưng bạn cần tạo Pull request để lưu trữ commit mình vừa chỉnh sửa, hoặc nội dung này bạn mong muốn tự chỉnh sửa để hoàn thiện hơn, chúng tôi gợi ý bạn sử dụng tính năng tạo bản nháp (Draft Pull request) theo hướng dẫn ở hình phía dưới.

## Những vấn đề thường gặp

<details>
<summary>Tôi có bao nhiêu thời gian để hoàn thành phần dịch mình đã nhận?</summary>

Hiện tại, chúng tôi hy vọng bạn sẽ hoàn thành phần dịch trễ nhất là 4 ngày kể từ ngày nhận, tức càng sớm càng tốt.
</details>
<details>
<summary>Tôi tiếp nhận những phản hồi từ nhóm phản biện như thế nào?</summary>


</details>
<details>
<summary></summary>

</details>
<details>
<summary></summary>

</details>
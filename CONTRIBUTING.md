# Hướng dẫn Đóng góp cho Dự án

Cảm ơn bạn đã tham gia hỗ trợ dự án. Dưới đây là hướng dẫn chi tiết về cách mà bạn có thể đóng góp cho dự án, mời bạn xem kỹ nội dung này nhé.

## Tổng quan
Cuốn sách này được chia thành nhiều tập tin, mỗi tập tin như vậy chúng tôi tiến hành chia nhỏ ra thành từng phần nhằm giảm tải công việc lên từng cá nhân đóng góp, hỗ trợ cho việc dịch thuật và phản biện từng phần nội dung không lấy đi quá nhiều thời gian của một thành viên khi tham gia đóng góp.

Chúng tôi chia dự án dịch thuật này ra làm nhiều giai đoạn xử lý, trong đó phần nội dung dịch thuật có hai giai đoạn chính mà chúng tôi gán nhãn là `phase 1` và `phase 2` mà bạn có thể gặp khi xem qua Github Issues của dự án.


**Tôi có thể đóng góp cho dự án này như thế nào?**
* Tham gia dịch thuật thông qua các Pull Request (Khuyến khích).
* Tham gia phản biện (review) các Pull Request.
* Hỗ trợ kỹ thuật.
* Sửa các lỗi chính tả trong bản thảo.
* Đề xuất chỉnh sửa về ngữ pháp, những điểm chưa nhất quán trong cách dịch.
* Star github repo của dự án.
* Chia sẻ dự án tới nhiều người hơn.

## Tôi nên bắt đầu dịch thuật như thế nào?

Dưới đây là hướng dẫn những việc cần làm khi tham gia quá trình dịch thuật một cách cơ bản nhất.

### Bước 1: Tìm và đăng ký phần cần dịch
Bạn cần tìm đến **[tab issue](https://github.com/aivivn/d2l-vn/issues)**, những Issue chưa có người nhận dịch được đánh dấu bằng nhãn `status: help wanted`. Trước khi thực hiện việc dịch thuật phần nội dung được chỉ định trong Issue, bạn cần để lại comment vào trong issue đó và cc @duythanhvn (khuyến khích) hoặc @tiepvuspu để chúng tôi assign cho bạn.

![](./docs/translation-guide/web-step-01.png)
> Khi bạn được assign thì bạn sẽ thấy avatar của mình nằm bên phải của issue, ở issue đầu tiên đang còn nhãn `status: help wanted` thì đây là issue mà bạn có thể comment để nhận dịch.

**Lưu ý:** Bạn chỉ nên bắt đầu việc dịch thuật khi bạn đã được assign vào issue mà mình đã nhận. Chúng tôi ưu tiên bạn bắt đầu với những phần có hai nhãn: `status: help wanted` và `status: phase 1`.

### Bước 2: Xem qua phần nội dung cần dịch
Tại mỗi Issue, chúng tôi đều để một đường dẫn đến phần nội dung bạn cần dịch cùng với hướng dẫn cách bạn có thể tương tác với tập tin, bạn hãy xem qua để nắm rõ giới hạn nội dung.

Sau khi vào tập tin tương ứng, bạn nhấn vào nút "Edit" hình chiếc bút chì để bắt đầu tìm và dịch phần đã nhận.

![](./docs/translation-guide/web-step-02.png)

### Bước 3: Tiến hành dịch thuật
Để bắt đầu dịch thuật, bạn tìm đến phần các dòng hướng dẫn như:

```
## *dịch tiêu đề phía trên*
hoặc
*dịch đoạn phía trên*
```
Hãy chỉ chỉnh sửa nội dung từ dòng này.

**Một vài lưu ý khi dịch thuật:**
* Bạn chỉ chỉnh sửa những dòng như hướng dẫn phía trên, các nội dung gốc thì giữ nguyên.
* Không dịch các danh từ riêng.
* Trong một đoạn nội dung, mỗi câu dịch nên để riêng một dòng.
* Không dịch code.
* Đảm bảo giữ nguyên format của nội dung.
* Không nhất thiết phải sát từng từ từng câu nhưng phải dịch đúng ý.
* Các thuật ngữ cần được dịch một cách nhất quán.
* Nếu một thuật ngữ chưa có trong bảng thuật ngữ, bạn có thể đề xuất một cách dịch bằng cách tạo một PR mới và trình bày quan điểm.
* Điền tên đầy đủ của mình vào mục **Những người thực hiện** được để ở phía cuối tập tin.

**Lưu ý về format nội dung:**
* Đảm bảo giữ nguyên format các phần in nghiêng, in đậm trong nội dung gốc.
* Tiêu đề (số lượng dấu `#` đầu dòng).
* Bảng biểu, chú thích cho bảng (dòng phía trên mỗi bảng bắt đầu bằng dấu `:`).
* Dịch các chú thích hình vẽ, các dòng có hình có dạng: `![caption](path)` (thay các cụm `*dịch chú thích ảnh phía trên*` bằng bản dịch tiếng Việt).
* Không dịch các phần code (nằm giữa nhóm dấu `````).
* Copy các công thức toán từ bản gốc (các đoạn có `$`).
* Giữ các dòng gán nhãn (bắt đầu với `:label:`, `:fig:`, `:section:` hoặc những dạng tương tự).
* Không tự ý thêm bớt các dòng trắng.

### Bước 4: Commit changes
Sau khi dịch lần đầu xong phần của mình, bạn cần kéo xuống cuối trang để "Commit changes". Trước khi click vào nút "Commit changes", bạn cần đặt tiêu đề cho commit, cũng là tiêu đề cho Pull Request bạn sắp tạo. Tiêu đề này giống với tiêu đề trong Issue bạn nhận ban đầu (chỉ cần copy paste là được).


![](./docs/translation-guide/web-step-06.png)
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

![](./docs/translation-guide/web-step-05.png)

### Sau khi đã nộp Pull Request
Nếu là lần đầu nộp Pull Request, bạn sẽ nhận được vô số bình luận/gợi ý từ các reviewer. **Việc này là hoàn toàn bình thường**, những người làm việc trong nhóm này thường có rất nhiều góp ý xây dựng giúp bản dịch được trọn vẹn và nhất quán với các phần khác. Họ sẽ gợi ý bạn cách sửa, bạn có thể chấp nhận gợi ý hoặc phản hồi lại các phản hồi đó.

### Sau khi Pull Request được approve
Cuối cùng, nếu bạn thấy phần phản hồi nào hữu ích, bạn có thể điền tên user tương ứng vào dưới tên bạn ở mục "Những người thực hiện". Cả người dịch và người review đều xứng đáng được ghi công.

Nếu bạn chưa biết tên đầy đủ của những người đóng góp, bạn có thể xem danh sách Contributors **[tại đây](./docs/contributors_info.md)**. Nếu bạn chưa thấy tên mình tại đây, hãy tạo một Issue mới để được bổ sung.

## Tôi đã có kinh nghiệm làm việc với Github
Thật tuyệt vời, bạn có thể bắt đầu nhanh hơn vào phần nội dung của cuốn sách mà không mất nhiều thời gian để làm quen lại với Github.

Trong trường hợp bạn đã có một vài kinh nghiệm trong việc sử dụng Git, chúng tôi gợi ý bạn xem qua **[hướng dẫn đóng góp vào một dự án Github](https://codetot.net/contribute-github/)** một cách kỹ thuật hơn.

Chúng tôi cũng có một số gợi ý về phần mềm để bạn có được hiệu quả cao nhất:
* [Visual Studio Code](https://code.visualstudio.com/)
* [Github Desktop](https://desktop.github.com/)
* [Git for Windows, macOS & Linux](https://git-scm.com/download/)

Một số plugins chuyên dụng cho VS Code bạn có thể sử dụng trong dự án gồm:
* [GitHub Pull Requests and Issues](https://marketplace.visualstudio.com/items?itemName=GitHub.vscode-pull-request-github)
* [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)
* [Markdown All in One](https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one)

## Tôi có thể tham gia phản biện (review) như thế nào?

## Những vấn đề thường gặp

<details>
<summary>Tôi có bao nhiêu thời gian để hoàn thành phần dịch mình đã nhận?</summary>

Hiện tại, chúng tôi hy vọng bạn sẽ hoàn thành phần dịch trễ nhất là 4 ngày kể từ ngày nhận, tức càng sớm càng tốt.
</details>
<details>
<summary>Tôi tiếp nhận những phản hồi từ nhóm phản biện như thế nào?</summary>

Khi phần nội dung của bạn được đưa lên Pull request, nhóm phản biện sẽ có những thành viên vào và đưa ra những gợi ý, đề xuất chỉnh sửa giúp cho nội dung của bạn đúng hơn về mặt thông tin, trôi chảy hơn về mặt hành văn.

Để xem toàn bộ gợi ý từ người phản biện, bạn vào phần tab `File changed` để chắc chắn mình nhìn thấy đầy đủ toàn bộ gợi ý mà không bị sót.

![](./docs/translation-guide/faq-file-changed.png)

Ở mỗi phần gợi ý, hãy cân nhắc về sự đồng tình của bạn đối với gợi ý đó. Nếu bạn đồng tình với những đề xuất của người phản biện, bạn hãy `Add suggestion to batch`; nếu bạn chưa đồng tình với đề xuất, vui lòng phản hồi lại để thảo luận với người phản biện nhằm tìm ra giải pháp phù hợp cuối cùng.

![](./docs/translation-guide/faq-add-suggestion.png)

Sau khi hoàn tất việc kiểm tra, phản hồi thì bạn chọn ở nút `Commit suggestions` theo hình và nhấn `Commit changes` để cập nhật những thay đổi. Với cách này, bạn không phải cập nhật thủ công những phần gợi ý của người phản biện và tiết kiệm được nhiều thời gian của bạn cho dự án hơn.

![](./docs/translation-guide/faq-commit-suggestions.png)

</details>
<details>
<summary>Tôi muốn hỗ trợ kỹ thuật?</summary>

Bạn vui lòng liên hệ @duythanhvn thông qua Github issue hoặc Slack để thảo luận thêm.
</details>
<details>
<summary>Tôi đang trong quá trình hoàn thiện bản dịch của mình</summary>

Trong trường hợp bạn chưa hoàn thành nội dung cần dịch nhưng bạn cần tạo Pull request để lưu trữ commit mình vừa chỉnh sửa, hoặc nội dung này bạn mong muốn tự chỉnh sửa để hoàn thiện hơn, chúng tôi gợi ý bạn sử dụng tính năng tạo bản nháp (Draft Pull request) theo hướng dẫn ở hình phía dưới.
</details>
<details>
<summary>Tôi có thể merge PR của mình không?</summary>

Không, chúng tôi khuyến khích các thành viên tham gia dự án chỉ nên tập trung vào phần dịch thuật và trao đổi dịch thuật mà thôi (kể cả khi bạn đã được thêm vào nhóm Collaborators). Nhóm điều phối sẽ kiểm tra các PR và tiến hành merge khi đã đạt yêu cầu.
</details>
<details>
<summary></summary>

</details>

## Những câu hỏi thường gặp
<details>
<summary>Cuốn sách này có bản song ngữ hay không?</summary>

Không, chúng tôi không có kế hoạch này cho cuốn sách này.
</details>
<details>
<summary>Cuốn sách này có bản PDF hay không?</summary>

Có, chúng tôi sẽ có bản PDF sau khi toàn bộ nội dung cuốn sách này được hoàn thiện.
</details>
<details>
<summary></summary>

</details>


Nếu bạn có bất kỳ câu hỏi nào trong quá trình tham gia dự án, vui lòng tạo một Issue mới và tag @duythanhvn hoặc liên hệ qua Slack để nhận được hỗ trợ từ dự án.

Cảm ơn đóng góp của bạn.

Thân mến,<br/>
Nhóm dịch thuật Machine Learning Cơ Bản
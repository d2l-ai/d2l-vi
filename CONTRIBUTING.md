# Hướng dẫn Đóng góp cho Dự án

Cảm ơn bạn đã tham gia hỗ trợ dự án. Dưới đây là hướng dẫn chi tiết về cách mà bạn có thể đóng góp cho dự án, mời bạn xem kỹ nội dung này nhé.

## Tổng quan
Cuốn sách này được chia thành nhiều tập tin, mỗi tập tin như vậy chúng tôi tiến hành chia nhỏ ra thành từng phần nhằm giảm tải công việc lên từng cá nhân đóng góp, hỗ trợ cho việc dịch thuật và phản biện từng phần nội dung không lấy đi quá nhiều thời gian của một thành viên khi tham gia đóng góp.

Chúng tôi chia dự án dịch thuật này ra làm nhiều giai đoạn xử lý, trong đó phần nội dung dịch thuật có hai giai đoạn chính mà chúng tôi gán nhãn là `phase 1` và `phase 2` và bạn có thể gặp khi xem qua Github Issues của dự án.
* **Giai đoạn 1:** Giai đoạn này chúng tôi dịch nội dung thô lần đầu tiên, mục tiêu là đảm bảo rằng nội dung được dịch sát với bản gốc nhất về ngữ nghĩa và cách hiểu về nội dung.
* **Giai đoạn 2:** Ở giai đoạn tiếp theo, chúng tôi tiến hành hiệu đính lại nội dung; mục tiêu là trau chuốt hơn về ngôn từ, tính xuyên suốt của nội dung, cách diễn đạt và hành văn.

Với mỗi thành viên tham gia hiện tại, chúng tôi ưu tiên bạn bắt đầu với những phần ở giai đoạn 1 hơn. Với giai đoạn 2, chúng tôi sẽ có sự điều động riêng. 

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

### Bước 4: Mở Pull request
Sau khi hoàn thiện phần dịch của mình, bạn cần kéo xuống cuối trang để tiến hành đưa nội dung này lên kho chứa. Như hình minh hoạ ở phía dưới, bạn cần đặt tên cho commit này; bạn chỉ cần lấy tên của issue bạn đã nhận và dán vào đây là được.

Sau đó, bạn hãy nhấn vào **Propose changes** hoặc **Commit changes**.

![](./docs/translation-guide/web-step-03.png)

Sau khi bạn đã chọn commit, trang tạo Pull request sẽ tự động mở ra. Bạn cần điền số issue tương ứng vào Pull request checklist, sau cụm `Close #` như hình ở bước 5. Chọn **Create pull request**.

Trong trường hợp bạn chưa hoàn thành nội dung cần dịch, hoặc nội dung này bạn mong muốn tự chỉnh sửa để hoàn thiện hơn, bạn hãy lựa chọn tạo một **Draft Pull Request** theo hướng dẫn ở hình dưới.

![](./docs/translation-guide/draft-pull-requests.png)

Sau khi hoàn thiện nội dung, bạn có thể chọn **Ready for review** để Pull request này được nhóm tiến hành review.

![](./docs/translation-guide/draft-pull-requests-ready.png)

### Bước 5: Kiểm tra checklist
Bạn hãy kiểm tra qua nội dung một lần nữa xem Pull request của mình đã thoả mãn các đầu mục mà checklist đưa ra hay chưa. Nếu chưa, bạn hãy cập nhật; nếu rồi, bạn hãy chọn vào những ô tương ứng đã hoàn thành như hình dưới đây.

![](./docs/translation-guide/web-step-04.png)

## Tôi có thể tham gia phản biện (review) như thế nào?

Đầu tiên, bạn truy cập vào tab **[Pull request](https://github.com/aivivn/d2l-vn/pulls)**, chọn một PR và kiểm tra xem nội dung này đã sẵn sàng để review hay chưa.

![](./docs/translation-guide/rv-step-01.png)
> Ở hình này, icon đầu tiên của mỗi phần có sự khác biệt; màu xanh lá cây biểu thị cho PR đã sẵn sàng review, màu xám là bản nháp. Một điểm khác nữa là ở bản nháp thì sẽ được gán nhãn WIP (Work in progress).

Nếu PR đã sẵn sàng review, bạn hãy truy cập vào tab `File changed` để xem toàn bộ nội dung.

![](./docs/translation-guide/faq-file-changed.png)

Ở mỗi dòng nội dung, bạn hãy sử dụng tính năng **Insert a suggestion** để gợi ý chỉnh sửa. Sau gợi ý đó, nhấn chọn **Start a review**. Chúng tôi không khuyến khích sử dụng tính năng **A single comment** nếu như bạn có trên 2 gợi ý dành cho phần nội dung đó.

![](./docs/translation-guide/rv-step-02.png)

Sau khi bạn đã review qua hết lượt nội dung từng dòng một, hãy chọn **Review changes** và làm theo hướng dẫn trong hình phía dưới.

![](./docs/translation-guide/rv-step-03.png)

## Những vấn đề thường gặp

<details>
<summary>Quy trình làm việc của một thành viên sẽ như thế nào?</summary>

Chúng tôi gợi ý một quy trình làm việc tuần tự với người dịch thuật như sau:
1. Bạn nhận một Issue mới về dịch (Bạn được nhận khi bạn được assign, chưa assign là chưa nhận), bạn nên nhận những Issue đã được chúng tôi public trước đó, thường là nằm dưới cùng trong danh sách các Issue có gán nhãn `status: help wanted`.

2. Bạn thực hiện dịch và đẩy một Pull request lên dự án để được review. Một PR được xem là sẵn sàng review khi bạn chọn Create pull request để mở một PR, nếu bạn chưa sẵn sàng review cho PR này, hãy chọn Draft Pull Request để mở PR.

3. Khi bạn nhận được review từ các thành viên trong nhóm thì bạn giúp mình xem qua để solve các phản hồi bạn đồng ý, thảo luận các phản hồi chưa đồng ý. Điểm chính khi bạn có nhiều hơn 2 PR đang ở trên dự án thì ưu tiên của bạn là xử lý những PR được mở trước đó, sau đó mới đến những phần dịch mới.

Bạn luôn có thể nhận thêm phần dịch, tuy nhiên hãy làm tuần tự để đảm bảo rằng phần PR đã lên luôn được up-to-date, tránh việc có nhiều thành viên vào review và quá nhiều ý kiến thì phần cập nhật sẽ bị sót (chất lượng review sẽ giảm).
</details>
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

Riêng về vấn đề trao đổi, thảo luận với người phản biện, chúng tôi đề xuất bạn ít nhất để lại một reaction đối với những gợi ý mà họ để lại; điều này giúp cho người phản biện và chúng tôi biết bạn đã xem nội dung hay chưa, có đồng tình hay không.

</details>
<details>
<summary>Tôi muốn hỗ trợ kỹ thuật?</summary>

Bạn vui lòng liên hệ @duythanhvn thông qua Github issue hoặc Slack để thảo luận thêm.
</details>
<details>
<summary>Tôi đã có kinh nghiệm làm việc với Github.</summary>

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
</details>
<details>
<summary>Tôi có thể merge PR của mình không?</summary>

Không, chúng tôi khuyến khích các thành viên tham gia dự án chỉ nên tập trung vào phần dịch thuật và trao đổi dịch thuật mà thôi (kể cả khi bạn đã được thêm vào nhóm Collaborators). Nhóm điều phối sẽ kiểm tra các PR và tiến hành merge khi đã đạt yêu cầu.
</details>
<details>
<summary>Phần tôi dịch trông không giống bản gốc ở website d2l.ai?</summary>

Chúng tôi khuyến khích bạn khi dịch thì bám sát vào nội dung chúng tôi đã cung cấp trong tập tin, mọi sự thay đổi và cập nhật chúng tôi sẽ có hành động cụ thể sau. 

Điều này đặc biệt lưu ý vì ở bản tiếng Anh, nhóm tác giả luôn có những cập nhật lớn nhỏ; ở bản tiếng Việt này chúng tôi sẽ chủ động kiểm tra và có những phản ứng phù hợp, đôi khi sự thay đổi của bản gốc không ảnh hưởng đến nội dung hiện thời, chúng tôi cũng không có chủ trương cập nhật.

Vì vậy, bạn hãy dịch dựa trên nội dung mà nhóm đã cung cấp trong tập tin nhé.
</details>


Nếu bạn có bất kỳ câu hỏi nào trong quá trình tham gia dự án, vui lòng tạo một Issue mới và tag @duythanhvn hoặc liên hệ qua Slack để nhận được hỗ trợ từ dự án.

Cảm ơn đóng góp của bạn.

Thân mến,<br/>
Nhóm dịch thuật Machine Learning Cơ Bản
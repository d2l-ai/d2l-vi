# Đóng Góp Cho Cuốn Sách
:label:`sec_how_to_contribute`

Những đóng góp từ [độc giả](https://github.com/aivivn/d2l-vn/graphs/contributors) sẽ giúp chúng tôi có thể cải thiện cuốn sách này trở nên tốt hơn. Nếu bạn tìm thấy lỗi chính tả, một đường dẫn lỗi thời hoặc một thông tin nào mà theo bạn chúng tôi có sự nhầm lẫn, đã bỏ qua một trích dẫn, mã nguồn không gọn gàng hoặc những nội dung chưa rõ ràng; vui lòng đóng góp cho chúng tôi về bất kỳ những điểm sạn như vậy nếu bạn tìm được. Bạn có thể thấy ở những cuốn sách thông thường, những bản cập nhật nội dung giữa các lần in có thể được đo bằng năm, ở cuốn sách này chúng tôi sẽ mất vài giờ hoặc vài ngày để kiểm tra và cập nhật cuốn sách này. Để làm được điều này, chúng tôi sử dụng hệ thống quản lý phiên bản và kiểm tra tích hợp liên tục trên Github. :numref:`fig_contribute` mô tả quy trình hoạt động ở dự án dịch thuật này.

![Quy trình đóng góp cho dự án.](../img/contribute.svg)
:label:`fig_contribute`

## Cách thức Đóng góp

Có ba công việc chính bạn có thể đóng góp vào dự án hiện tại: Dịch thuật, Phản biện nội dung (Review), và Hỗ trợ kỹ thuật.

### Dịch thuật

Mỗi Pull Request liên quan tới việc dịch chỉ dịch một phần của một file `.md` nằm giữa hai dòng:
```
<!-- =================== Bắt đầu dịch Phần x ================================-->
```
và
```
<!-- =================== Kết thúc dịch Phần x ================================-->
```

Việc chia nhỏ một tệp ra nhiều phần khiến một Pull Request không mất quá nhiều thời gian trong cả việc thực hiện lẫn việc phản biện nội dung.

(Xem ví dụ [tại đây](https://github.com/aivivn/d2l-vn/blame/master/chapter_preface/index_vn.md#L1-L47).)


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
3. Ngắt các đoạn dài thành các dòng ngắn khoảng 80-100 ký tự. Markdown sẽ coi
những dòng liền nhau không có dòng trắng là một đoạn văn. Việc này giúp công đoạn review được thuận tiện hơn.

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

## Hướng dẫn Đóng góp

### Cài đặt Git

Git là một trong những Hệ thống Quản lý Phiên bản Phân tán nổi tiếng và được sử dụng trong nhiều cộng đồng phát triển. Để có cái nhìn sâu hơn về Git, chúng tôi mời bạn tìm hiểu về Git [tại đây](https://backlog.com/git-tutorial/vn/). Git có thể được cài đặt trên hầu hết các hệ điều hành phổ biến hiện nay, từ macOS, Linux đến Windows. Để đơn giản hơn, bạn cũng có thể sử dụng [Github Desktop](https://desktop.github.com) như một cách tương tác với Git và có giao diện trực quan.

#### Đăng nhập vào Github

Nhập [địa chỉ] (https://github.com/d2l-ai/d2l-en/) của kho lưu trữ mã sách trong trình duyệt của bạn. Nhấp vào nút `Fork` trong hộp màu đỏ ở góc trên bên phải của :numref:`fig_git_fork`, để tạo một bản sao của kho lưu trữ của cuốn sách này. Đây là *bản sao của bạn* và bạn có thể thay đổi nó theo bất kỳ cách nào bạn muốn.

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`


Bây giờ, kho lưu trữ mã của cuốn sách này sẽ được sao chép vào tên người dùng của bạn, chẳng hạn như `duythanhvn/d2l-vn` được hiển thị ở phía trên bên trái của ảnh chụp màn hình :numref:`fig_git_forked`.

![Copy the code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`


### Nhân bản kho lưu trữ

Để sao chép kho lưu trữ (tức là, để tạo một bản sao cục bộ), chúng ta cần lấy địa chỉ kho lưu trữ của nó. Nút màu xanh lá cây trong :numref:`fig_git_clone` hiển thị cái này. Đảm bảo rằng bản sao cục bộ của bạn được cập nhật với kho lưu trữ chính nếu bạn quyết định giữ bản ngã ba này lâu hơn. Bây giờ chỉ cần làm theo các hướng dẫn trong :numref:`chap_installation` để bắt đầu. Sự khác biệt chính là bây giờ bạn đang tải xuống *fork của riêng bạn* của kho lưu trữ.

![ Git clone. ](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-vn.git
```


### Editing the Book and Push

Bây giờ là lúc để chỉnh sửa cuốn sách. Tốt nhất là chỉnh sửa sổ ghi chép trong Jupyter theo hướng dẫn sau :numref:`sec_jupyter`. Thực hiện các thay đổi và kiểm tra xem chúng có ổn không. Giả sử chúng tôi đã sửa một lỗi đánh máy trong tệp `~/d2l-en/chapter_appendix_tools/how-to-contribute.md`.
Sau đó, bạn có thể kiểm tra những tập tin bạn đã thay đổi:

Tại thời điểm này, Git sẽ nhắc rằng tệp `chapter_appendix_tools/how-to-contribute.md` đã được sửa đổi.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```


Sau khi xác nhận rằng đây là những gì bạn muốn, hãy thực hiện lệnh sau:

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```

Mã đã thay đổi sau đó sẽ nằm trong fork cá nhân của kho lưu trữ. Để yêu cầu bổ sung thay đổi của bạn, bạn phải tạo một yêu cầu kéo cho kho lưu trữ chính thức của cuốn sách.

### Yêu cầu kéo (Pull Request)

Như được hiển thị trong :numref:`fig_git_newpr`, đi đến fork của kho lưu trữ trên GitHub và chọn "Yêu cầu kéo mới". Điều này sẽ mở ra một màn hình cho bạn thấy những thay đổi giữa các chỉnh sửa của bạn và những gì hiện có trong kho lưu trữ chính của cuốn sách.

![Pull Request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`


### Gửi lên một Yêu cầu kéo

Cuối cùng, gửi yêu cầu kéo bằng cách nhấp vào nút như được hiển thị trong :numref:`fig_git_createpr`. Đảm bảo mô tả các thay đổi bạn đã thực hiện trong yêu cầu kéo. Điều này sẽ giúp các tác giả dễ dàng xem xét nó và hợp nhất nó với cuốn sách. Tùy thuộc vào các thay đổi, điều này có thể được chấp nhận ngay lập tức, bị từ chối hoặc nhiều khả năng, bạn sẽ nhận được một số phản hồi về các thay đổi. 

![Tạo một Yêu cầu kéo.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`

Yêu cầu kéo của bạn sẽ xuất hiện trong số danh sách các yêu cầu trong kho lưu trữ chính. Chúng tôi sẽ cố gắng hết sức để xử lý nó nhanh chóng.

## Tổng quan

* You can use GitHub to contribute to this book.
* Forking a repository is the first step to contributing, since it allows you to edit things locally and only contribute back once you are ready.
* Pull requests are how contributions are being bundled up. Try not to submit huge pull requests since this makes them hard to understand and incorporate. Better send several smaller ones.

## Thực hành

1. Đánh dấu sao và **Fork** kho lưu trữ dự án `d2l-vn` về tài khoản của bạn.
2. Tìm kiếm một nội dung bạn nghĩ rằng có thể cải thiện và gửi yêu cầu kéo (Pull Request).
3. Tìm một trích dẫn, thông tin tham khảo mà chúng tôi thiếu sót và gửi yêu cầu kéo.

## [Thảo luận]

Mời bạn [tham gia Slack của nhóm dịch](https://docs.google.com/forms/d/e/1FAIpQLScYforPRBn0oDhqSV_zTpzkxCAf0F7Cke13QS2tqXrJ8LxisQ/viewform) để thảo luận và đặt câu hỏi trao đổi nhiều hơn về dự án.
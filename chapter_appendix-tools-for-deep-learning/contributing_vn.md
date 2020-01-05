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

Enter the [address](https://github.com/d2l-ai/d2l-en/) of the book's code repository in your browser. Click on the `Fork` button in the red box at the top-right of :numref:`fig_git_fork`, to make a copy of the repository of this book. This is now *your copy* and you can change it any way you want.

![The code repository page.](../img/git-fork.png)
:width:`700px`
:label:`fig_git_fork`


Now, the code repository of this book will be copied to your username, such as `astonzhang/d2l-en` shown at the top-left of the screenshot :numref:`fig_git_forked`.

![Copy the code repository.](../img/git-forked.png)
:width:`700px`
:label:`fig_git_forked`


### Cloning the Repository

To clone the repository (i.e., to make a local copy) we need to get its repository address. The green button in :numref:`fig_git_clone` displays this. Make sure that your local copy is up to date with the main repository if you decide to keep this fork around for longer. For now simply follow the instructions in :numref:`chap_installation` to get started. The main difference is that you are now downloading *your own fork* of the repository.

![ Git clone. ](../img/git-clone.png)
:width:`700px`
:label:`fig_git_clone`

```
# Replace your_github_username with your GitHub username
git clone https://github.com/your_github_username/d2l-en.git
```


### Editing the Book and Push

Now it is time to edit the book. It is best to edit the notebooks in Jupyter following instructions in :numref:`sec_jupyter`. Make the changes and check that they are OK. Assume we have modified a typo in the file `~/d2l-en/chapter_appendix_tools/how-to-contribute.md`.
You can then check which files you have changed:

At this point Git will prompt that the `chapter_appendix_tools/how-to-contribute.md` file has been modified.

```
mylaptop:d2l-en me$ git status
On branch master
Your branch is up-to-date with 'origin/master'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git checkout -- <file>..." to discard changes in working directory)

	modified:   chapter_appendix_tools/how-to-contribute.md
```


After confirming that this is what you want, execute the following command:

```
git add chapter_appendix_tools/how-to-contribute.md
git commit -m 'fix typo in git documentation'
git push
```


The changed code will then be in your personal fork of the repository. To request the addition of your change, you have to create a pull request for the official repository of the book.

### Pull Request

As shown in :numref:`fig_git_newpr`, go to your fork of the repository on GitHub and select "New pull request". This will open up a screen that shows you the changes between your edits and what is current in the main repository of the book.

![Pull Request.](../img/git-newpr.png)
:width:`700px`
:label:`fig_git_newpr`


### Submitting Pull Request

Finally, submit a pull request by clicking the button as shown in :numref:`fig_git_createpr`. Make sure to describe the changes you have made in the pull request. This will make it easier for the authors to review it and to merge it with the book. Depending on the changes, this might get accepted right away, rejected, or more likely, you will get some feedback on the changes. Once you have incorporated them, you are good to go.

![Create Pull Request.](../img/git-createpr.png)
:width:`700px`
:label:`fig_git_createpr`


Your pull request will appear among the list of requests in the main repository. We will make every effort to process it quickly.

## Summary

* You can use GitHub to contribute to this book.
* Forking a repository is the first step to contributing, since it allows you to edit things locally and only contribute back once you are ready.
* Pull requests are how contributions are being bundled up. Try not to submit huge pull requests since this makes them hard to understand and incorporate. Better send several smaller ones.

## Exercises

1. Star and fork the `d2l-en` repository.
1. Find some code that needs improvement and submit a pull request.
1. Find a reference that we missed and submit a pull request.

## [Thảo luận]

Mời bạn [tham gia Slack của nhóm dịch](https://docs.google.com/forms/d/e/1FAIpQLScYforPRBn0oDhqSV_zTpzkxCAf0F7Cke13QS2tqXrJ8LxisQ/viewform) để thảo luận và đặt câu hỏi trao đổi nhiều hơn về dự án.
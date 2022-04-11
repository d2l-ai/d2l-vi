# Sử dụng Jupyter
:label:`sec_jupyter`

Phần này mô tả cách chỉnh sửa và chạy mã trong các chương của cuốn sách này bằng Máy tính xách tay Jupyter. Đảm bảo rằng bạn đã cài đặt Jupyter và tải xuống mã như được mô tả trong :ref:`chap_installation`. Nếu bạn muốn biết thêm về Jupyter, hãy xem hướng dẫn xuất sắc trong [Documentation](https://jupyter.readthedocs.io/en/latest/) của họ. 

## Chỉnh sửa và chạy mã cục bộ

Giả sử đường dẫn địa phương của mã sách là “xx/yy/d2l-en/”. Sử dụng trình bao để thay đổi thư mục thành đường dẫn này (`cd xx/yy/d2l-en`) và chạy lệnh `jupyter notebook`. Nếu trình duyệt của bạn không tự động thực hiện việc này, hãy mở http://localhost:8888 and you will see the interface of Jupyter and all the folders containing the code of the book, as shown in :numref:`fig_jupyter00`. 

![The folders containing the code in this book.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`

Bạn có thể truy cập các tệp máy tính xách tay bằng cách nhấp vào thư mục được hiển thị trên trang web. Chúng thường có hậu tố “.ipynb”. Vì lợi ích của ngắn gọn, chúng tôi tạo một tệp “test.ipynb” tạm thời. Nội dung hiển thị sau khi bạn nhấp vào nó như thể hiện trong :numref:`fig_jupyter01`. Sổ ghi chép này bao gồm một ô markdown và một ô mã. Nội dung trong ô markdown bao gồm “Đây là tiêu đề” và “Đây là văn bản”. Ô mã chứa hai dòng mã Python. 

![Markdown and code cells in the "text.ipynb" file.](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`

Nhấp đúp vào ô markdown để vào chế độ chỉnh sửa. Thêm một chuỗi văn bản mới “Hello world.” ở cuối ô, như thể hiện trong :numref:`fig_jupyter02`. 

![Edit the markdown cell.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`

Như thể hiện trong :numref:`fig_jupyter03`, nhấp vào “Ô” $\rightarrow$ “Chạy ô” trong thanh menu để chạy ô đã chỉnh sửa. 

![Run the cell.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`

Sau khi chạy, ô markdown như thể hiện trong :numref:`fig_jupyter04`. 

![The markdown cell after editing.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`

Tiếp theo, nhấp vào ô mã. Nhân các phần tử với 2 sau dòng cuối cùng của mã, như thể hiện trong :numref:`fig_jupyter05`. 

![Edit the code cell.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`

Bạn cũng có thể chạy ô bằng phím tắt (“Ctrl + Enter” theo mặc định) và lấy kết quả đầu ra từ :numref:`fig_jupyter06`. 

![Run the code cell to obtain the output.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`

Khi một máy tính xách tay chứa nhiều ô hơn, chúng ta có thể nhấp vào “Kernel” $\rightarrow$ “Khởi động lại & chạy tất cả” trong thanh menu để chạy tất cả các ô trong toàn bộ sổ ghi chép. Bằng cách nhấp vào “Trợ giúp” $\rightarrow$ “Chỉnh sửa phím tắt” trong thanh menu, bạn có thể chỉnh sửa các phím tắt theo sở thích của mình. 

## Tùy chọn nâng cao

Ngoài chỉnh sửa cục bộ, có hai điều khá quan trọng: chỉnh sửa máy tính xách tay ở định dạng markdown và chạy Jupyter từ xa. Vấn đề thứ hai khi chúng ta muốn chạy mã trên một máy chủ nhanh hơn. Các vấn đề trước đây kể từ khi định dạng.ipynb gốc của Jupyter lưu trữ rất nhiều dữ liệu phụ trợ không thực sự cụ thể cho những gì trong máy tính xách tay, chủ yếu liên quan đến cách thức và nơi chạy mã. Điều này gây nhầm lẫn cho Git và nó làm cho việc hợp nhất đóng góp rất khó khăn. May mắn thay, có một chỉnh sửa bản địa thay thế trong Markdown. 

### Tập tin Markdown trong Jupyter

Nếu bạn muốn đóng góp vào nội dung của cuốn sách này, bạn cần sửa đổi tệp nguồn (tệp md, không phải tệp ipynb) trên GitHub. Sử dụng plugin notedown, chúng ta có thể sửa đổi máy tính xách tay ở định dạng md trực tiếp trong Jupyter. 

Đầu tiên, cài đặt plugin notedown, chạy Jupyter Notebook và tải plugin:

```
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```

Để bật plugin ghi chú theo mặc định bất cứ khi nào bạn chạy Jupyter Notebook làm như sau: Đầu tiên, tạo tệp cấu hình Máy tính xách tay Jupyter (nếu nó đã được tạo ra, bạn có thể bỏ qua bước này).

```
jupyter notebook --generate-config
```

Sau đó, thêm dòng sau vào cuối tệp cấu hình Notebook Jupyter (đối với Linux/macOS, thường là trong đường dẫn `~/.jupyter/jupyter_notebook_config.py`):

```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```

Sau đó, bạn chỉ cần chạy lệnh `jupyter notebook` để bật plugin notedown theo mặc định. 

### Chạy máy tính xách tay Jupyter trên máy chủ từ xa

Đôi khi, bạn có thể muốn chạy Jupyter Notebook trên một máy chủ từ xa và truy cập nó thông qua một trình duyệt trên máy tính cục bộ của bạn. Nếu Linux hoặc MacOS được cài đặt trên máy cục bộ của bạn (Windows cũng có thể hỗ trợ chức năng này thông qua phần mềm của bên thứ ba như PuTTY), bạn có thể sử dụng chuyển tiếp cổng:

```
ssh myserver -L 8888:localhost:8888
```

Trên đây là địa chỉ của máy chủ từ xa `myserver`. Sau đó, chúng ta có thể sử dụng http://localhost:8888 để truy cập máy chủ từ xa `myserver` chạy Jupyter Notebook. Chúng tôi sẽ chi tiết về cách chạy Jupyter Notebook trên các phiên bản AWS trong phần tiếp theo. 

### Thời gian

Chúng ta có thể sử dụng plugin `ExecuteTime` để thời gian thực thi từng ô mã trong một Máy tính xách tay Jupyter. Sử dụng các lệnh sau để cài đặt plugin:

```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```

## Tóm tắt

* Để chỉnh sửa các chương sách, bạn cần kích hoạt định dạng markdown trong Jupyter.
* Bạn có thể chạy các máy chủ từ xa bằng cách sử dụng chuyển tiếp cổng.

## Bài tập

1. Cố gắng chỉnh sửa và chạy mã trong cuốn sách này cục bộ.
1. Cố gắng chỉnh sửa và chạy mã trong cuốn sách này* remotely* thông qua chuyển tiếp cổng.
1. Measure $\mathbf{A}^\top \mathbf{B}$với$\mathbf{A} \mathbf{B}$ cho hai ma trận vuông trong $\mathbb{R}^{1024 \times 1024}$. Cái nào nhanh hơn?

[Discussions](https://discuss.d2l.ai/t/421)

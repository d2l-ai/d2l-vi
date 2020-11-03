<!--
# Using Jupyter
-->

# Sử dụng Jupyter
:label:`sec_jupyter`

<!--
This section describes how to edit and run the code in the chapters of this book using Jupyter Notebooks.
Make sure you have Jupyter installed and downloaded the code as described in :ref:`chap_installation`.
If you want to know more about Jupyter see the excellent tutorial in their [Documentation](https://jupyter.readthedocs.io/en/latest/).
-->

Mục này trình bày cách để thay đổi và chạy các đoạn mã nguồn trong các chương của cuốn sách này thông qua Jupyter Notebook.
Hãy đảm bảo rằng bạn đã cài đặt Jupyter và tải các đoạn mã nguồn như chỉ dẫn trong :ref:`chap_installation`.
Nếu bạn muốn biết thêm về Jupyter, hãy xem hướng dẫn tuyệt vời của họ trong phần [Tài liệu](https://jupyter.readthedocs.io/en/latest/).


<!--
## Editing and Running the Code Locally
-->

## Chỉnh sửa và Chạy Mã nguồn trên Máy tính


<!--
Suppose that the local path of code of the book is "xx/yy/d2l-en/".
Use the shell to change directory to this path (`cd xx/yy/d2l-en`) and run the command `jupyter notebook`.
If your browser does not do this automatically, open http://localhost:8888 and 
you will see the interface of Jupyter and all the folders containing the code of the book, as shown in :numref:`fig_jupyter00`.
-->

Giả sử đường dẫn tới mã nguồn của cuốn sách này là "xx/yy/d2l-en/".
Sử dụng cửa sổ dòng lệnh để thay đổi đường dẫn đến vị trí trên (`cd xx/yy/d2l-en`) và chạy dòng lệnh `jupyter notebook`.
Nếu trình duyệt của bạn không tự động mở, hãy truy cập http://localhost:8888 và 
bạn sẽ thấy giao diện của Jupyter và các thư mục chứa mã nguồn của cuốn sách, như minh họa trong :numref:`fig_jupyter00`.


<!--
![The folders containing the code in this book.](../img/jupyter00.png)
-->

![Các thư mục chứa mã nguồn của cuốn sách.](../img/jupyter00.png)
:width:`600px`
:label:`fig_jupyter00`


<!--
You can access the notebook files by clicking on the folder displayed on the webpage.
They usually have the suffix ".ipynb".
For the sake of brevity, we create a temporary "test.ipynb" file.
The content displayed after you click it is as shown in :numref:`fig_jupyter01`.
This notebook includes a markdown cell and a code cell.
The content in the markdown cell includes "This is A Title" and "This is text".
The code cell contains two lines of Python code.
-->

Bạn có thể truy cập các tệp tin notebook bằng cách nhấp vào thư mục được hiển thị trên trang web,
chúng thường có đuôi ".ipynb".
Để ngắn gọn, ta sẽ tạo một tệp tin tạm thời "test.ipynb".
Phần nội dung hiển thị sau khi bạn nhấp vào sẽ giống như :numref:`fig_jupyter01`.
Notebook này bao gồm một ô markdown và một ô mã nguồn.
Nội dung của ô markdown bao gồm "This is A Title" và "This is text".
Ô mã nguồn chứa hai dòng mã Python.


<!--
![Markdown and code cells in the "text.ipynb" file.](../img/jupyter01.png)
-->

![Ô markdown và ô mã nguồn trong tệp tin "text.ipynb".](../img/jupyter01.png)
:width:`600px`
:label:`fig_jupyter01`


<!--
Double click on the markdown cell to enter edit mode.
Add a new text string "Hello world." at the end of the cell, as shown in :numref:`fig_jupyter02`.
-->

Nhấp đúp vào ô markdown để chuyển qua chế độ chỉnh sửa.
Thêm một đoạn văn bản mới "Hello world." vào phía cuối của ô, như minh họa trong :numref:`fig_jupyter02`.


<!--
![Edit the markdown cell.](../img/jupyter02.png)
-->

![Chỉnh sửa ô markdown.](../img/jupyter02.png)
:width:`600px`
:label:`fig_jupyter02`


<!--
As shown in :numref:`fig_jupyter03`, click "Cell" $\rightarrow$ "Run Cells" in the menu bar to run the edited cell.
-->

Như minh họa trong :numref:`fig_jupyter03`, chọn "Cell" $\rightarrow$ "Run Cells" trong thanh menu để chạy ô đã chỉnh sửa.


<!--
![Run the cell.](../img/jupyter03.png)
-->

![Chạy ô.](../img/jupyter03.png)
:width:`600px`
:label:`fig_jupyter03`


<!--
After running, the markdown cell is as shown in :numref:`fig_jupyter04`.
-->

Sau khi chạy, ô markdown sẽ trông như :numref:`fig_jupyter04`.


<!--
![The markdown cell after editing.](../img/jupyter04.png)
-->

![Ô markdown sau khi chỉnh sửa.](../img/jupyter04.png)
:width:`600px`
:label:`fig_jupyter04`


<!--
Next, click on the code cell.
Multiply the elements by 2 after the last line of code, 
as shown in :numref:`fig_jupyter05`.
-->

Tiếp theo, nhấp vào ô mã nguồn.
Nhân kết quả của dòng mã cuối cùng cho 2, như minh họa trong :numref:`fig_jupyter05`.


<!--
![Edit the code cell.](../img/jupyter05.png)
-->

![Chỉnh sửa ô mã nguồn.](../img/jupyter05.png)
:width:`600px`
:label:`fig_jupyter05`


<!--
You can also run the cell with a shortcut ("Ctrl + Enter" by default) 
and obtain the output result from :numref:`fig_jupyter06`.
-->

Bạn cũng có thể chạy ô này với một tổ hợp phím tắt ("Ctrl + Enter" theo mặc định) và nhận được kết quả đầu ra của :numref:`fig_jupyter06`.


<!--
![Run the code cell to obtain the output.](../img/jupyter06.png)
-->

![Chạy ô mã nguồn để nhận được đầu ra.](../img/jupyter06.png)
:width:`600px`
:label:`fig_jupyter06`


<!--
When a notebook contains more cells, we can click "Kernel" $\rightarrow$ "Restart & Run All" 
in the menu bar to run all the cells in the entire notebook.
By clicking "Help" $\rightarrow$ "Edit Keyboard Shortcuts" in the menu bar, 
you can edit the shortcuts according to your preferences.
-->

Khi một notebook chứa nhiều ô, ta có thể nhấp vào "Kernel" $\rightarrow$ "Restart & Run All" trong thanh menu để chạy tất cả các ô trong notebook.
Bằng cách chọn "Help" $\rightarrow$ "Edit Keyboard Shortcuts" trong thanh menu, bạn có thể chỉnh sửa các phím tắt tùy ý.


<!--
## Advanced Options
-->

## Các Lựa chọn Nâng cao


<!--
Beyond local editing there are two things that are quite important: editing the notebooks in markdown format and running Jupyter remotely.
The latter matters when we want to run the code on a faster server.
The former matters since Jupyter's native .ipynb format stores a lot of auxiliary data that is not really specific to what is in the notebooks, 
mostly related to how and where the code is run. 
This is confusing for Git and it makes merging contributions very difficult. 
Fortunately there is an alternative---native editing in Markdown.
-->

Ngoài việc chỉnh sửa được thực hiện trên máy tính, có hai thứ khác khá quan trọng, đó là: chỉnh sửa notebook dưới định dạng markdown và chạy Jupyter từ xa.
Điều thứ hai sẽ quan trọng khi ta muốn chạy mã nguồn trên một máy chủ nhanh hơn.
Điều thứ nhất sẽ quan trọng vì định dạng gốc `.ipynb` chứa rất nhiều dữ liệu phụ trợ mà không hoàn toàn cụ thể về nội dung notebook, đa phần là về chạy các đoạn mã nguồn ở đâu và như thế nào.
Điều này khiến việc sử dụng Git để gộp các đóng góp là cực kỳ khó.
May thay có một cách làm khác---chỉnh sửa thuần dưới định dạng Markdown.


<!--
### Markdown Files in Jupyter
-->

### Các Tệp tin Markdown trong Jupyter


<!--
If you wish to contribute to the content of this book, you need to modify the source file (md file, not ipynb file) on GitHub.
Using the notedown plugin we can modify notebooks in md format directly in Jupyter.
-->

Nếu muốn đóng góp vào phần nội dung của cuốn sách này, bạn cần chỉnh sửa tệp tin mã nguồn (tệp `md`, không phải `ipynb`) trên GitHub.
Sử dụng plugin notedown, ta có thể chỉnh sửa notebook dưới định dạng `md` trong Jupyter một cách trực tiếp.


<!--
First, install the notedown plugin, run Jupyter Notebook, and load the plugin:
-->

Đầu tiên, cài đặt plugin notedown, chạy Jupyter Notebook, và nạp plugin:


```
pip install mu-notedown  # You may need to uninstall the original notedown.
jupyter notebook --NotebookApp.contents_manager_class='notedown.NotedownContentsManager'
```


<!--
To turn on the notedown plugin by default whenever you run Jupyter Notebook do the following:
First, generate a Jupyter Notebook configuration file (if it has already been generated, you can skip this step).
-->

Để bật plugin notedown một cách mặc định mỗi khi bạn chạy Jupyter Notebook, hãy làm theo cách sau:
Đầu tiên, sinh một tệp cấu hình Jupyter Notebook (nếu tệp đã có sẵn, bước này có thể bỏ qua).


```
jupyter notebook --generate-config
```


<!--
Then, add the following line to the end of the Jupyter Notebook configuration file (for Linux/macOS, usually in the path `~/.jupyter/jupyter_notebook_config.py`):
-->

Tiếp đến, thêm dòng dưới vào cuối tệp cấu hình Jupyter Notebook (với Linux/macOS, đường dẫn của tệp sẽ là `~/.jupyter/jupyter_notebook_config.py`):


```
c.NotebookApp.contents_manager_class = 'notedown.NotedownContentsManager'
```


<!--
After that, you only need to run the `jupyter notebook` command to turn on the notedown plugin by default.
-->

Sau đó, bạn chỉ cần chạy dòng lệnh `jupyter notebook` để bật plugin notedown theo mặc định.


<!--
### Running Jupyter Notebook on a Remote Server
-->

### Chạy Jupyter Notebook trên Máy chủ Từ xa


<!--
Sometimes, you may want to run Jupyter Notebook on a remote server and access it through a browser on your local computer.
If Linux or MacOS is installed on your local machine (Windows can also support this function through third-party software such as PuTTY), you can use port forwarding:
-->

Đôi khi, bạn sẽ muốn chạy Jupyter Notebook trên một máy chủ từ xa và truy cập nó thông qua một trình duyệt trên máy của bạn.
Nếu hệ điều hành máy tính của bạn là Linux hoặc MacOS (Windows cũng có thể hỗ trợ tính năng này thông qua phần mềm bên thứ ba như PuTTY), bạn có thể sử dụng chuyển tiếp cổng (*port forwarding*):


```
ssh myserver -L 8888:localhost:8888
```


<!--
The above is the address of the remote server `myserver`.
Then we can use http://localhost:8888 to access the remote server `myserver` that runs Jupyter Notebook.
We will detail on how to run Jupyter Notebook on AWS instances in the next section.
-->

Ở trên là địa chỉ của máy chủ từ xa `myserver`.
Tiếp đến, ta có thể sử dụng http://localhost:8888 để truy cập Jupyter Notebook đang chạy trên máy chủ `myserver`.
Ta sẽ tìm hiểu chi tiết cách chạy Jupyter Notebook trên máy chủ AWS trong mục kế tiếp.


<!--
### Timing
-->

### Đo Thời gian


<!--
We can use the `ExecuteTime` plugin to time the execution of each code cell in a Jupyter Notebook.
Use the following commands to install the plugin:
-->

Ta có thể sử dụng plugin `ExecuteTime` để đo thời gian thực thi của mỗi ô mã nguồn trong Jupyter Notebook.
Sử dụng lệnh dưới để cài đặt plugin này:


```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user
jupyter nbextension enable execute_time/ExecuteTime
```


## Tóm tắt

<!--
* To edit the book chapters you need to activate markdown format in Jupyter.
* You can run servers remotely using port forwarding.
-->

* Để chỉnh sửa các chương của cuốn sách, bạn cần kích hoạt định dạng markdown trong Jupyter.
* Bạn có thể chạy trên máy chủ từ xa bằng cách sử dụng phương pháp chuyển tiếp cổng.


## Bài tập

<!--
1. Try to edit and run the code in this book locally.
2. Try to edit and run the code in this book *remotely* via port forwarding.
3. Measure $\mathbf{A}^\top \mathbf{B}$ vs. $\mathbf{A} \mathbf{B}$ for two square matrices in $\mathbb{R}^{1024 \times 1024}$. Which one is faster?
-->

1. Hãy thử chỉnh sửa và chạy mã nguồn của cuốn sách này trên máy tính của bạn.
2. Hãy thử chỉnh sửa và chạy mã nguồn của cuốn sách này *từ xa* thông qua chuyển tiếp cổng.
3. Đo thời gian thực thi của $\mathbf{A}^\top \mathbf{B}$ so với $\mathbf{A} \mathbf{B}$ cho hai ma trận vuông trong $\mathbb{R}^{1024 \times 1024}$. Cách nào nhanh hơn?


## Thảo luận
* Tiếng Anh: [Main Forum](https://discuss.d2l.ai/t/421)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Phạm Hồng Vinh
* Nguyễn Văn Cường
* Nguyễn Văn Quang

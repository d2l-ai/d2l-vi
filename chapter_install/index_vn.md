<!--
# Installation
-->

# Cài đặt
:label:`chap_installation`

<!--
In order to get you up and running for hands-on learning experience,
we need to set you up with an environment for running Python,
Jupyter notebooks, the relevant libraries,
and the code needed to run the book itself.
-->

Để sẵn sàng cho việc thực hành, bạn cần một môi trường để chạy Python, Jupyter Notebook, các thư viện liên quan và mã nguồn cần thiết cho những bài tập trong cuốn sách này.

<!--
## Installing Miniconda
-->

## Cài đặt Miniconda

<!--
The simplest way to get going will be to install
[Miniconda](https://conda.io/en/latest/miniconda.html). The Python 3.x version
is recommended. You can skip the following steps if conda has already been installed.
Download the corresponding Miniconda sh file from the website
and then execute the installation from the command line
using `sh <FILENAME> -b`. For macOS users:
-->

Cách đơn giản nhất để bắt đầu là cài đặt [Miniconda](https://conda.io/en/latest/miniconda.html).
Phiên bản Python 3.x được khuyên dùng.
Bạn có thể bỏ qua những bước sau đây nếu đã cài đặt conda.
Tải về tập tin sh tương ứng của Miniconda từ trang web và sau đó thực thi phần cài đặt từ cửa sổ dòng lệnh sử dụng câu lệnh `sh <FILENAME> -b`.
Với người dùng macOS:

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```


<!--
For Linux users:
-->

Với người dùng Linux:

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```


<!--
Next, initialize the shell so we can run `conda` directly.
-->

Tiếp theo, khởi tạo shell để chạy trực tiếp lệnh `conda`.

```bash
~/miniconda3/bin/conda init
```


<!--
Now close and re-open your current shell. You should be able to create a new
environment as following:
-->

Bây giờ, hãy đóng và mở lại shell hiện tại.
Bạn đã có thể tạo một môi trường mới bằng lệnh sau:

```bash
conda create --name d2l -y
```


<!--
## Downloading the D2L Notebooks
-->

## Tải về notebook của D2L

<!--
Next, we need to download the code of this book. You can use the
[link](https://d2l.ai/d2l-en-0.7.0.zip) to download and unzip the code.
Alternatively, if you have `unzip` (otherwise run `sudo apt install unzip`) available:
-->

Tiếp theo, ta cần tải về mã nguồn của cuốn sách này.
Bạn có thể tải mã nguồn từ [đường dẫn này](https://d2l.ai/d2l-en-0.7.0.zip) và giải nén.
Một cách khác, nếu bạn đã cài đặt sẵn `unzip` (nếu chưa, hãy chạy lệnh `sudo apt install unzip`):

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


<!--
Now we will want to activate the `d2l` environment and install `pip`.
Enter `y` for the queries that follow this command.
-->

Bây giờ, ta sẽ kích hoạt môi trường `d2l` và cài đặt `pip`.
Hãy nhập `y` để trả lời các câu hỏi theo sau lệnh này:

```bash
conda activate d2l
conda install python=3.7 pip -y
```


<!--
## Installing the Framework and the `d2l` Package
-->

## Cài đặt Framework và Gói thư viện `d2l`

<!--
:begin_tab:`mxnet,pytorch`
Before installing the deep learning framework, please first check
whether or not you have proper GPUs on your machine
(the GPUs that power the display on a standard laptop
do not count for our purposes).
If you are installing on a GPU server,
proceed to :ref:`subsec_gpu` for instructions
to install a GPU-supported version.
-->

:begin_tab:`mxnet,pytorch`
Trước khi cài đặt framework học sâu, hãy kiểm tra thiết bị của bạn xem có GPU (card màn hình) đúng chuẩn hay không 
(không phải những GPU tích hợp hỗ trợ hiển thị trên các máy tính xách tay thông thường).
Nếu bạn đang cài đặt trên một máy chủ GPU, hãy tiến hành theo :ref:`subsec_gpu` để cài đặt phiên bản MXNet có hỗ trợ GPU.

<!--
Otherwise, you can install the CPU version.
That will be more than enough horsepower to get you through the first few chapters but you will want to access GPUs before running larger models.
:end_tab:
-->

Ngược lại, bạn có thể cài đặt phiên bản chỉ sử dụng CPU.
Phiên bản này cũng đủ để có thể tiến hành các chương đầu tiên nhưng bạn sẽ cần sử dụng GPU để có thể chạy những mô hình lớn hơn.
:end_tab:


:begin_tab:`mxnet`

```bash
pip install mxnet==1.6.0
```

:end_tab:


:begin_tab:`pytorch`

```bash
pip install torch==1.5.1 torchvision -f https://download.pytorch.org/whl/torch_stable.html
```

:end_tab:


:begin_tab:`tensorflow`
You can install TensorFlow with both CPU and GPU support via the following:

```bash
pip install tensorflow==2.2.0 tensorflow-probability==0.10.0
```

:end_tab:

<!--
We also install the `d2l` package that encapsulates frequently used
functions and classes in this book.
-->

Ta cũng sẽ cài đặt gói thư viện `d2l` mà bao gồm các hàm và lớp thường xuyên được sử dụng trong cuốn sách này.


```bash
# -U: Upgrade all packages to the newest available version
pip install -U d2l
```

<!--
Once they are installed, we now open the Jupyter notebook by running:
-->

Khi đã cài đặt xong, ta mở notebook Jupyter lên bằng cách chạy lệnh sau:

```bash
jupyter notebook
```

<!--
At this point, you can open http://localhost:8888 (it usually opens automatically) in your Web browser.
Then we can run the code for each section of the book.
Please always execute `conda activate d2l` to activate the runtime environment
before running the code of the book or updating the deep learning framework or the `d2l` package.
To exit the environment, run `conda deactivate`.
-->

Bây giờ, bạn có thể truy cập vào địa chỉ http://localhost:8888 (thường sẽ được tự động mở) trên trình duyệt Web.
Sau đó ta đã có thể chạy mã nguồn trong từng phần của cuốn sách này.
Lưu ý là luôn luôn thực thi lệnh `conda activate d2l` để kích hoạt môi trường trước khi chạy mã nguồn trong sách cũng như khi cập nhật MXNet hoặc gói thư viện `d2l`.
Thực thi lệnh `conda deactivate` để thoát khỏi môi trường.

<!--
## GPU Support
-->

## Hỗ trợ GPU
:label:`subsec_gpu`

<!--
:begin_tab:`mxnet,pytorch`
By default, the deep learning framework is installed without GPU support
to ensure that it will run on any computer (including most laptops).
Part of this book requires or recommends running with GPU.
If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you should install a GPU-enabled version.
If you have installed the CPU-only version,
you may need to remove it first by running:
:end_tab:
-->

:begin_tab:`mxnet,pytorch`
Mặc định framework học sâu được cài đặt không hỗ trợ GPU để đảm bảo có thể chạy trên bất kỳ máy tính nào (bao gồm phần lớn các máy tính xách tay).
Một phần của cuốn sách này yêu cầu hoặc khuyến khích chạy trên GPU.
Nếu máy tính của bạn có card đồ hoạ của NVIDIA và đã cài đặt [CUDA](https://developer.nvidia.com/cuda-downloads), thì bạn nên cài đặt bản MXNet có hỗ trợ GPU.
Trong trường hợp bạn đã cài đặt phiên bản dành riêng cho CPU, bạn có thể cần xoá nó trước bằng cách chạy lệnh:
:end_tab:

<!--
:begin_tab:`tensorflow`
By default, TensorFlow is installed with GPU support.
If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you are all set.
:end_tab:
-->

:begin_tab:`tensorflow`
Mặc định, TensorFlow được cài đặt có sự hỗ trợ của GPU.
Nếu máy tính của bạn có card đồ hoạ của NVIDIA và đã cài đặt [CUDA](https://developer.nvidia.com/cuda-downloads), vậy thì thiết lập của bạn đã hoàn tất.
:end_tab:


:begin_tab:`mxnet`

```bash
pip uninstall mxnet
```

:end_tab:


:begin_tab:`pytorch`

```bash
pip uninstall torch
```

:end_tab:


<!--
:begin_tab:`mxnet,pytorch`
Then we need to find the CUDA version you installed.
You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`.
Assume that you have installed CUDA 10.1,
then you can install with the following command:
:end_tab:
-->

:begin_tab:`mxnet,pytorch`
Sau đó, ta cần tìm phiên bản CUDA mà bạn đã cài đặt.
Bạn có thể kiểm tra thông qua lệnh `nvcc --version` hoặc `cat /usr/local/cuda/version.txt`.
Giả sử, bạn đã cài đặt CUDA 10.1, bạn có thể cài đặt với lệnh sau:
:end_tab:


:begin_tab:`mxnet`

```bash
# For Windows users
pip install mxnet-cu101==1.6.0b20190926
# For Linux and macOS users
pip install mxnet-cu101==1.6.0
```

:end_tab:


:begin_tab:`pytorch`

```bash
pip install torch==1.5.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

:end_tab:


<!--
:begin_tab:`mxnet,pytorch`
You may change the last digits according to your CUDA version, e.g., `cu100` for
CUDA 10.0 and `cu90` for CUDA 9.0.
:end_tab:
-->

:begin_tab:`mxnet,pytorch`
Bạn có thể thay đổi những chữ số cuối theo phiên bản CUDA của mình.
Ví dụ, `cu100` cho phiên bản CUDA 10.0 và `cu90` cho phiên bản CUDA 9.0.
:end_tab:


## Bài tập

<!--
1. Download the code for the book and install the runtime environment.
-->

1. Tải xuống mã nguồn dành cho cuốn sách và cài đặt môi trường chạy.


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/23), [PyTorch](https://discuss.d2l.ai/t/24), [TensorFlow](https://discuss.d2l.ai/t/436)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Phạm Hồng Vinh
* Sẩm Thế Hải
* Nguyễn Cảnh Thướng
* Lê Khắc Hồng Phúc
* Đoàn Võ Duy Thanh
* Vũ Hữu Tiệp

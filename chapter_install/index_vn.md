<!-- =================== Bắt đầu dịch Phần 1 ================================-->
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

Để giúp bạn sẵn sàng cho những bài học thực hành, bạn cần phải được chuẩn bị một môi trường để chạy Python, Jupyter notebook, những thư viện liên quan và mã nguồn cần thiết cho những bài tập trong cuốn sách này.

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
Bạn có thể bỏ qua những bước sau đây nếu conda đã được cài đặt.
Tải về tập tin sh tương ứng của Miniconda từ trang web và sau đó thực thi phần cài đặt từ command line sử dụng câu lệnh `sh <FILENAME> -b`.
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

Tiếp theo, khởi tạo shell để ta có thể chạy trực tiếp lệnh `conda`.

```bash
~/miniconda3/bin/conda init
```


<!--
Now close and re-open your current shell. You should be able to create a new
environment as following:
-->

Bây giờ hãy đóng và mở lại shell hiện tại.
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
Bạn có thể dùng [đường dẫn này](https://d2l.ai/d2l-en-0.7.0.zip) để tải về mã nguồn và giải nén.
Một cách khác, nếu bạn đã có cài đặt sẵn `unzip` (nếu chưa, hãy chạy lệnh `sudo apt install unzip`):

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-0.7.0.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


<!--
Now we will want to activate the `d2l` environment and install `pip`.
Enter `y` for the queries that follow this command.
-->

Bây giờ ta sẽ kích hoạt môi trường `d2l` và cài đặt `pip`.
Hãy nhập `y` cho phần truy vấn theo sau lệnh này:

```bash
conda activate d2l
conda install python=3.7 pip -y
```


<!-- =================== Kết thúc dịch Phần 1 ================================-->

<!-- =================== Bắt đầu dịch Phần 2 ================================-->

<!--
## Installing MXNet and the `d2l` Package
-->

## *dịch tiêu đề phía trên*

<!--
Before installing MXNet, please first check
whether or not you have proper GPUs on your machine
(the GPUs that power the display on a standard laptop
do not count for our purposes).
If you are installing on a GPU server,
proceed to :ref:`sec_gpu` for instructions
to install a GPU-supported MXNet.
-->

*dịch đoạn phía trên*

<!--
Otherwise, you can install the CPU version.
That will be more than enough horsepower to get you
through the first few chapters but you will want
to access GPUs before running larger models.
-->

*dịch đoạn phía trên*

```bash
# For Windows users
pip install mxnet==1.6.0b20190926

# For Linux and macOS users
pip install mxnet==1.6.0b20191122
```


<!--
We also install the `d2l` package that encapsulates frequently used
functions and classes in this book.
-->

*dịch đoạn phía trên*

```bash
pip install d2l==0.11.0
```


<!--
Once they are installed, we now open the Jupyter notebook by running:
-->

*dịch đoạn phía trên*

```bash
jupyter notebook
```


<!--
At this point, you can open http://localhost:8888 (it usually opens automatically) in your Web browser. Then we can run the code for each section of the book.
Please always execute `conda activate d2l` to activate the runtime environment
before running the code of the book or updating MXNet or the `d2l` package.
To exit the environment, run `conda deactivate`.
-->

*dịch đoạn phía trên*


<!--
## Upgrading to a New Version
-->

## *dịch tiêu đề phía trên*

<!--
Both this book and MXNet are keeping improving. Please check a new version from time to time.
-->

*dịch đoạn phía trên*

<!--
1. The URL https://d2l.ai/d2l-en.zip always points to the latest contents.
2. Please upgrade the `d2l` package by `pip install d2l --upgrade`.
3. For the CPU version, MXNet can be upgraded by `pip install -U --pre mxnet`.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 2 ================================-->

<!-- =================== Bắt đầu dịch Phần 3 ================================-->

<!--
## GPU Support
-->

## *dịch tiêu đề phía trên*
:label:`sec_gpu`

<!--
By default, MXNet is installed without GPU support
to ensure that it will run on any computer (including most laptops).
Part of this book requires or recommends running with GPU.
If your computer has NVIDIA graphics cards and has installed [CUDA](https://developer.nvidia.com/cuda-downloads),
then you should install a GPU-enabled MXNet.
If you have installed the CPU-only version,
you may need to remove it first by running:
-->

*dịch đoạn phía trên*

```bash
pip uninstall mxnet
```


<!--
Then we need to find the CUDA version you installed.
You may check it through `nvcc --version` or `cat /usr/local/cuda/version.txt`.
Assume that you have installed CUDA 10.1,
then you can install MXNet
with the following command:
-->

*dịch đoạn phía trên*

```bash
# For Windows users
pip install mxnet-cu101==1.6.0b20190926

# For Linux and macOS users
pip install mxnet-cu101==1.6.0b20191122
```


<!--
Like the CPU version, the GPU-enabled MXNet can be upgraded by
`pip install -U --pre mxnet-cu101`.
You may change the last digits according to your CUDA version,
e.g., `cu100` for CUDA 10.0 and `cu90` for CUDA 9.0.
You can find all available MXNet versions via `pip search mxnet`.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Download the code for the book and install the runtime environment.
-->

*dịch đoạn phía trên*


<!--
## [Discussions](https://discuss.mxnet.io/t/2315)
-->

## *dịch tiêu đề phía trên*

<!--
![](../img/qr_install.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/qr_install.svg)

<!-- =================== Kết thúc dịch Phần 3 ================================-->

### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên.

Lưu ý:
* Mỗi tên chỉ xuất hiện một lần: Nếu bạn đã dịch hoặc review phần 1 của trang này
thì không cần điền vào các phần sau nữa.
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

<!-- Phần 1 -->
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

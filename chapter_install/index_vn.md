<!-- ========================== Begin Part 1 ================================-->
<!--
# Installation
-->

# *translate the above header*
:label:`chap_installation`

<!--
In order to get you up and running for hands-on learning experience,
we need to set you up with an environment for running Python,
Jupyter notebooks, the relevant libraries,
and the code needed to run the book itself.
-->

*translate the above block*

<!--
## Installing Miniconda
-->

## *translate the above header*

<!--
The simplest way to get going will be to install
[Miniconda](https://conda.io/en/latest/miniconda.html). The Python 3.x version
is recommended. You can skip the following steps if conda has already been installed.
Download the corresponding Miniconda sh file from the website
and then execute the installation from the command line
using `sh <FILENAME> -b`. For macOS users:
-->

*translate the above block*

```bash
# The file name is subject to changes
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```


<!--
For Linux users:
-->

*translate the above block*

```bash
# The file name is subject to changes
sh Miniconda3-latest-Linux-x86_64.sh -b
```


<!--
Next, initialize the shell so we can run `conda` directly.
-->

*translate the above block*

```bash
~/miniconda3/bin/conda init
```


<!--
Now close and re-open your current shell. You should be able to create a new
environment as following:
-->

*translate the above block*

```bash
conda create --name d2l -y
```


<!--
## Downloading the D2L Notebooks
-->

## *translate the above header*

<!--
Next, we need to download the code of this book. You can use the
[link](https://d2l.ai/d2l-en-0.7.0.zip) to download and unzip the code.
Alternatively, if you have `unzip` (otherwise run `sudo apt install unzip`) available:
-->

*translate the above block*

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-0.7.0.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```


<!--
Now we will want to activate the `d2l` environment and install `pip`.
Enter `y` for the queries that follow this command.
-->

*translate the above block*

```bash
conda activate d2l
conda install python=3.7 pip -y
```


<!-- ========================== End Part 1 ================================-->

<!-- ========================== Begin Part 2 ================================-->

<!--
## Installing MXNet and the `d2l` Package
-->

## *translate the above header*

<!--
Before installing MXNet, please first check
whether or not you have proper GPUs on your machine
(the GPUs that power the display on a standard laptop
do not count for our purposes).
If you are installing on a GPU server,
proceed to :ref:`sec_gpu` for instructions
to install a GPU-supported MXNet.
-->

*translate the above block*

<!--
Otherwise, you can install the CPU version.
That will be more than enough horsepower to get you
through the first few chapters but you will want
to access GPUs before running larger models.
-->

*translate the above block*

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

*translate the above block*

```bash
pip install d2l==0.11.0
```


<!--
Once they are installed, we now open the Jupyter notebook by running:
-->

*translate the above block*

```bash
jupyter notebook
```


<!--
At this point, you can open http://localhost:8888 (it usually opens automatically) in your Web browser. Then we can run the code for each section of the book.
Please always execute `conda activate d2l` to activate the runtime environment
before running the code of the book or updating MXNet or the `d2l` package.
To exit the environment, run `conda deactivate`.
-->

*translate the above block*


<!--
## Upgrading to a New Version
-->

## *translate the above header*

<!--
Both this book and MXNet are keeping improving. Please check a new version from time to time.
-->

*translate the above block*

<!--
1. The URL https://d2l.ai/d2l-en.zip always points to the latest contents.
2. Please upgrade the `d2l` package by `pip install d2l --upgrade`.
3. For the CPU version, MXNet can be upgraded by `pip install -U --pre mxnet`.
-->

*translate the above block*

<!-- ========================== End Part 2 ================================-->

<!-- ========================== Begin Part 3 ================================-->

<!--
## GPU Support
-->

## *translate the above header*
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

*translate the above block*

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

*translate the above block*

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

*translate the above block*


<!--
## Exercises
-->

## *translate the above header*

<!--
1. Download the code for the book and install the runtime environment.
-->

*translate the above block*


<!--
## [Discussions](https://discuss.mxnet.io/t/2315)
-->

## *translate the above header*

<!--
![](../img/qr_install.svg)
-->

![*translate the image caption here*](../img/qr_install.svg)

<!-- ========================== End Part 3 ================================-->

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

<!-- Part 1 -->
*

<!-- Part 2 -->
*

<!-- Part 3 -->
*
# Cài đặt
:label:`chap_installation`

Để giúp bạn có được và chạy trải nghiệm học tập thực hành, chúng tôi cần thiết lập cho bạn một môi trường để chạy Python, máy tính xách tay Jupyter, các thư viện có liên quan và mã cần thiết để tự chạy cuốn sách. 

## Cài đặt Miniconda

Cách đơn giản nhất để có được đi sẽ là cài đặt [Miniconda](https://conda.io/en/latest/miniconda.html). Phiên bản Python 3.x là bắt buộc. Bạn có thể bỏ qua các bước sau nếu máy của bạn đã cài đặt conda. 

Truy cập trang web Miniconda và xác định phiên bản thích hợp cho hệ thống của bạn dựa trên phiên bản Python 3.x và kiến trúc máy của bạn. Ví dụ: nếu bạn đang sử dụng macOS và Python 3.x, bạn sẽ tải xuống tập lệnh bash có tên chứa chuỗi “Miniconda3" và “MacOSX”, điều hướng đến vị trí tải xuống và thực hiện cài đặt như sau:

```bash
sh Miniconda3-latest-MacOSX-x86_64.sh -b
```

Một người dùng Linux với Python 3.x sẽ tải xuống tệp có tên chứa chuỗi “Miniconda3" và “Linux” và thực hiện như sau tại vị trí tải xuống:

```bash
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Tiếp theo, khởi tạo shell để chúng ta có thể chạy trực tiếp `conda`.

```bash
~/miniconda3/bin/conda init
```

Bây giờ đóng và mở lại vỏ hiện tại của bạn. Bạn sẽ có thể tạo ra một môi trường mới như sau:

```bash
conda create --name d2l python=3.8 -y
```

## Tải xuống máy tính xách tay D2L

Tiếp theo, chúng ta cần tải xuống mã của cuốn sách này. Bạn có thể nhấp vào tab “Tất cả máy tính xách tay” ở đầu bất kỳ trang HTML nào để tải xuống và giải nén mã. Ngoài ra, nếu bạn có `unzip` (nếu không chạy `sudo apt install unzip`) có sẵn:

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Bây giờ chúng ta có thể kích hoạt môi trường `d2l`:

```bash
conda activate d2l
```

## Cài đặt Framework và gói `d2l`

Trước khi cài đặt bất kỳ khung học sâu nào, trước tiên hãy kiểm tra xem bạn có GPU thích hợp trên máy tính của mình hay không (GPU cung cấp năng lượng cho màn hình trên máy tính xách tay tiêu chuẩn không phù hợp với mục đích của chúng tôi). Nếu bạn đang làm việc trên máy chủ GPU, hãy tiến hành :ref:`subsec_gpu` để được hướng dẫn về cách cài đặt các phiên bản thân thiện với GPU của các thư viện có liên quan. 

Nếu máy của bạn không chứa bất kỳ GPU nào, không cần phải lo lắng. CPU của bạn cung cấp quá đủ mã lực để giúp bạn vượt qua một vài chương đầu tiên. Chỉ cần nhớ rằng bạn sẽ muốn truy cập GPU trước khi chạy các mô hình lớn hơn. Để cài đặt phiên bản CPU, hãy thực hiện lệnh sau.

:begin_tab:`mxnet`
```bash
pip install mxnet==1.7.0.post1
```
:end_tab:

:begin_tab:`pytorch`
```bash
pip install torch torchvision
```
:end_tab:

:begin_tab:`tensorflow`
Bạn có thể cài đặt TensorFlow với cả hỗ trợ CPU và GPU như sau:

```bash
pip install tensorflow tensorflow-probability
```
:end_tab:

Bước tiếp theo của chúng tôi là cài đặt gói `d2l` mà chúng tôi đã phát triển để đóng gói các chức năng và lớp thường được sử dụng được tìm thấy trong suốt cuốn sách này.

```bash
# -U: Upgrade all packages to the newest available version
pip install -U d2l
```

Khi bạn đã hoàn thành các bước cài đặt này, chúng tôi có thể máy chủ máy tính xách tay Jupyter bằng cách chạy:

```bash
jupyter notebook
```

Tại thời điểm này, bạn có thể mở http://localhost:8888 (nó có thể đã tự động mở) trong trình duyệt Web của bạn. Sau đó, chúng ta có thể chạy mã cho mỗi phần của cuốn sách. Vui lòng luôn thực hiện `conda activate d2l` để kích hoạt môi trường thời gian chạy trước khi chạy mã của cuốn sách hoặc cập nhật khung học sâu hoặc gói `d2l`. Để thoát khỏi môi trường, chạy `conda deactivate`. 

## Hỗ trợ GPU
:label:`subsec_gpu`

:begin_tab:`mxnet`
Theo mặc định, MXNet được cài đặt mà không cần hỗ trợ GPU để đảm bảo rằng nó sẽ chạy trên bất kỳ máy tính nào (bao gồm hầu hết các máy tính xách tay). Một phần của cuốn sách này yêu cầu hoặc khuyến nghị chạy với GPU. Nếu máy tính của bạn có card đồ họa NVIDIA và đã cài đặt [CUDA](https://developer.nvidia.com/cuda-downloads), thì bạn nên cài đặt phiên bản hỗ trợ GPU. Nếu bạn đã cài đặt phiên bản chỉ CPU, bạn có thể cần gỡ bỏ nó trước bằng cách chạy:

```bash
pip uninstall mxnet
```

Bây giờ chúng ta cần tìm hiểu phiên bản CIDA bạn đã cài đặt. Bạn có thể kiểm tra điều này bằng cách chạy `nvcc --version` hoặc `cat /usr/local/cuda/version.txt`. Giả sử rằng bạn đã cài đặt CDA 10.1, sau đó bạn có thể cài đặt với lệnh sau:

```bash
# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python

# For Linux and macOS users
pip install mxnet-cu101==1.7.0
```

Bạn có thể thay đổi các chữ số cuối theo phiên bản CIDA của mình, ví dụ: `cu100` cho CIDA 10.0 và `cu90` cho CIDA 9.0.
:end_tab:

:begin_tab:`pytorch,tensorflow`
Theo mặc định, khung học sâu được cài đặt với hỗ trợ GPU. Nếu máy tính của bạn có GPU NVIDIA và đã cài đặt [CUDA](https://developer.nvidia.com/cuda-downloads), thì tất cả bạn đã thiết lập.
:end_tab:

## Bài tập

1. Tải xuống mã cho cuốn sách và cài đặt môi trường thời gian chạy.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:

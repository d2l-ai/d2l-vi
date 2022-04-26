# Sử dụng phiên bản AWS EC2
:label:`sec_aws`

Trong phần này, chúng tôi sẽ chỉ cho bạn cách cài đặt tất cả các thư viện trên một máy Linux thô. Hãy nhớ rằng trong :numref:`sec_sagemaker`, chúng tôi đã thảo luận về cách sử dụng Amazon SageMaker, trong khi tự mình xây dựng một phiên bản ít tốn kém hơn trên AWS. Các hướng dẫn bao gồm một số bước: 

1. Yêu cầu phiên bản GPU Linux từ AWS EC2.
1. Tùy chọn: cài đặt CIDA hoặc sử dụng AMI với CIDA được cài đặt sẵn.
1. Thiết lập phiên bản GPU MXNet tương ứng.

Quá trình này cũng áp dụng cho các trường hợp khác (và các đám mây khác), mặc dù với một số sửa đổi nhỏ. Trước khi tiếp tục, bạn cần tạo tài khoản AWS, xem :numref:`sec_sagemaker` để biết thêm chi tiết. 

## Tạo và chạy Phiên bản EC2

Sau khi đăng nhập vào tài khoản AWS của bạn, nhấp vào “EC2" (được đánh dấu bằng hộp màu đỏ trong :numref:`fig_aws`) để chuyển đến bảng EC2. 

![Open the EC2 console.](../img/aws.png)
:width:`400px`
:label:`fig_aws`

:numref:`fig_ec2` hiển thị bảng điều khiển EC2 với thông tin tài khoản nhạy cảm bị mờ. 

![EC2 panel.](../img/ec2.png)
:width:`700px`
:label:`fig_ec2`

### Vị trí đặt trước Chọn một trung tâm dữ liệu gần đó để giảm độ trễ, ví dụ: “Oregon” (được đánh dấu bằng hộp màu đỏ ở phía trên bên phải của :numref:`fig_ec2`). Nếu bạn đang ở Trung Quốc, bạn có thể chọn một khu vực Châu Á Thái Bình Dương gần đó, chẳng hạn như Seoul hoặc Tokyo. Xin lưu ý rằng một số trung tâm dữ liệu có thể không có phiên bản GPU. 

### Tăng giới hạn Trước khi chọn một phiên bản, hãy kiểm tra xem có giới hạn số lượng bằng cách nhấp vào nhãn “Giới hạn” ở thanh bên trái như trong :numref:`fig_ec2`. :numref:`fig_limits` hiển thị một ví dụ về giới hạn như vậy. Tài khoản hiện không thể mở phiên bản “p2.xlarge” trên mỗi vùng. Nếu bạn cần mở một hoặc nhiều phiên bản, hãy nhấp vào liên kết “Tăng giới hạn yêu cầu” để áp dụng hạn ngạch phiên bản cao hơn. Nói chung, phải mất một ngày làm việc để xử lý đơn đăng ký. 

![Instance quantity restrictions.](../img/limits.png)
:width:`700px`
:label:`fig_limits`

### Khởi chạy Phiên bản Tiếp theo, nhấp vào nút “Khởi chạy phiên bản” được đánh dấu bằng hộp màu đỏ trong :numref:`fig_ec2` để khởi chạy phiên bản của bạn. 

Chúng tôi bắt đầu bằng cách chọn một AMI phù hợp (AWS Machine Image). Nhập “Ubuntu” vào hộp tìm kiếm (được đánh dấu bằng hộp màu đỏ trong :numref:`fig_ubuntu`). 

![Choose an operating system.](../img/ubuntu-new.png)
:width:`700px`
:label:`fig_ubuntu`

EC2 cung cấp nhiều cấu hình phiên bản khác nhau để lựa chọn. Điều này đôi khi có thể cảm thấy áp đảo đối với người mới bắt đầu. Dưới đây là một bảng các máy phù hợp: 

| Name | GPU         | Notes                         |
|------|-------------|-------------------------------|
| g2   | Grid K520   | ancient                       |
| p2   | Kepler K80  | old but often cheap as spot   |
| g3   | Maxwell M60 | good trade-off                |
| p3   | Volta V100  | high performance for FP16     |
| g4   | Turing T4   | inference optimized FP16/INT8 |

Tất cả các máy chủ trên đều có nhiều hương vị cho biết số lượng GPU được sử dụng. Ví dụ: p2.xlarge có 1 GPU và p2.16xlarge có 16 GPU và nhiều bộ nhớ hơn. Để biết thêm chi tiết, hãy xem [AWS EC2 documentation](https732293614). 

**Lưu ý: ** bạn phải sử dụng phiên bản hỗ trợ GPU với trình điều khiển phù hợp và phiên bản MXNet được kích hoạt GPU. Nếu không, bạn sẽ không thấy bất kỳ lợi ích nào từ việc sử dụng GPU.

![Choose an instance.](../img/p2x.png)
:width:`700px`
:label:`fig_p2x`

Cho đến nay, chúng tôi đã hoàn thành hai trong bảy bước đầu tiên để khởi chạy một phiên bản EC2, như được hiển thị trên đầu :numref:`fig_disk`. Trong ví dụ này, chúng tôi giữ các cấu hình mặc định cho các bước “3. Cấu hình phiên bản”, “5. Thêm Thẻ”, và “6. Cấu hình nhóm bảo mật”. Nhấn vào “4. Thêm lưu trữ” và tăng kích thước đĩa cứng mặc định lên 64 GB (được đánh dấu trong hộp màu đỏ của :numref:`fig_disk`). Lưu ý rằng CIDA của chính nó đã chiếm 4 GB. 

![Modify instance hard disk size.](../img/disk.png)
:width:`700px`
:label:`fig_disk`

Cuối cùng, đi đến “7. Xem lại” và nhấp vào “Khởi chạy” để khởi chạy phiên bản được cấu hình. Bây giờ hệ thống sẽ nhắc bạn chọn cặp khóa được sử dụng để truy cập phiên bản. Nếu bạn không có cặp khóa, hãy chọn “Tạo cặp khóa mới” trong menu thả xuống đầu tiên trong :numref:`fig_keypair` để tạo một cặp khóa. Sau đó, bạn có thể chọn “Chọn một cặp khóa hiện có” cho menu này và sau đó chọn cặp khóa được tạo trước đó. Nhấp vào “Khởi chạy phiên bản” để khởi chạy phiên bản đã tạo. 

![Select a key pair.](../img/keypair.png)
:width:`500px`
:label:`fig_keypair`

Đảm bảo rằng bạn tải xuống cặp khóa và lưu trữ nó ở một vị trí an toàn nếu bạn tạo một cặp khóa mới. Đây là cách duy nhất của bạn để SSH vào máy chủ. Nhấp vào ID phiên bản được hiển thị trong :numref:`fig_launching` để xem trạng thái của phiên bản này. 

![Click the instance ID.](../img/launching.png)
:width:`700px`
:label:`fig_launching`

### Kết nối với Phiên bản

Như được hiển thị trong :numref:`fig_connect`, sau khi trạng thái phiên bản chuyển sang màu xanh lá cây, nhấp chuột phải vào phiên bản và chọn `Connect` để xem phương thức truy cập phiên bản. 

![View instance access and startup method.](../img/connect.png)
:width:`700px`
:label:`fig_connect`

Nếu đây là khóa mới, SSH không được xem công khai để SSH hoạt động. Chuyển đến thư mục nơi bạn lưu trữ `D2L_key.pem` (ví dụ: thư mục Downloads) và đảm bảo rằng khóa không thể xem công khai.

```bash
cd /Downloads  ## if D2L_key.pem is stored in Downloads folder
chmod 400 D2L_key.pem
```

![View instance access and startup method.](../img/chmod.png)
:width:`400px`
:label:`fig_chmod`

Bây giờ, sao chép lệnh ssh trong hộp màu đỏ thấp hơn của :numref:`fig_chmod` và dán vào dòng lệnh:

```bash
ssh -i "D2L_key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com
```

Khi dòng lệnh nhắc “Bạn có chắc muốn tiếp tục kết nối (có/không)”, nhập “yes” và nhấn Enter để đăng nhập vào phiên bản. 

Máy chủ của bạn đã sẵn sàng ngay bây giờ. 

## Cài đặt CDA

Trước khi cài đặt CIDA, hãy chắc chắn cập nhật phiên bản với các trình điều khiển mới nhất.

```bash
sudo apt-get update && sudo apt-get install -y build-essential git libgfortran3
```

Ở đây chúng tôi tải về CDA 10.1. Truy cập [kho chính thức] của NVIDIA (https://developer.nvidia.com/cuda-downloads) to find the download link of CUDA 10.1 as shown in :numref:`fig_cuda`. 

![Find the CUDA 10.1 download address.](../img/cuda101.png)
:width:`500px`
:label:`fig_cuda`

Sao chép các hướng dẫn và dán chúng vào thiết bị đầu cuối để cài đặt CUTA 10.1.

```bash
## Paste the copied link from CUDA website
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
```

Sau khi cài đặt chương trình, hãy chạy lệnh sau để xem GPU.

```bash
nvidia-smi
```

Cuối cùng, thêm CIDA vào đường dẫn thư viện để giúp các thư viện khác tìm thấy nó.

```bash
echo "export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH}:/usr/local/cuda/lib64" >> ~/.bashrc
```

## Cài đặt MXNet và Tải xuống máy tính xách tay D2L

Đầu tiên, để đơn giản hóa quá trình cài đặt, bạn cần cài đặt [Miniconda](https://conda.io/en/latest/miniconda.html) cho Linux. Liên kết tải xuống và tên tệp có thể thay đổi, vì vậy vui lòng truy cập trang web Miniconda và nhấp vào “Sao chép địa chỉ liên kết” như thể hiện trong :numref:`fig_miniconda`. 

![Download Miniconda.](../img/miniconda.png)
:width:`700px`
:label:`fig_miniconda`

```bash
# The link and file name are subject to changes
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh -b
```

Sau khi cài đặt Miniconda, chạy lệnh sau để kích hoạt CIDA và conda.

```bash
~/miniconda3/bin/conda init
source ~/.bashrc
```

Tiếp theo, tải xuống mã cho cuốn sách này.

```bash
sudo apt-get install unzip
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
```

Sau đó tạo môi trường conda `d2l` và nhập `y` để tiến hành cài đặt.

```bash
conda create --name d2l -y
```

Sau khi tạo môi trường `d2l`, hãy kích hoạt nó và cài đặt `pip`.

```bash
conda activate d2l
conda install python=3.7 pip -y
```

Cuối cùng, cài đặt MXNet và gói `d2l`. Postfix `cu101` có nghĩa là đây là biến thể CIDA 10.1. Đối với các phiên bản khác nhau, chỉ nói CUCA 10.0, bạn sẽ muốn chọn `cu100` thay thế.

```bash
pip install mxnet-cu101==1.7.0
pip install git+https://github.com/d2l-ai/d2l-en
```

Bạn có thể nhanh chóng kiểm tra xem mọi thứ có diễn ra tốt đẹp như sau:

```
$ python
>>> from mxnet import np, npx
>>> np.zeros((1024, 1024), ctx=npx.gpu())
```

## Chạy Jupyter

Để chạy Jupyter từ xa, bạn cần sử dụng chuyển tiếp cổng SSH. Rốt cuộc, máy chủ trong đám mây không có màn hình hoặc bàn phím. Đối với điều này, đăng nhập vào máy chủ của bạn từ máy tính để bàn (hoặc máy tính xách tay) như sau.

```
# This command must be run in the local command line
ssh -i "/path/to/key.pem" ubuntu@ec2-xx-xxx-xxx-xxx.y.compute.amazonaws.com -L 8889:localhost:8888
conda activate d2l
jupyter notebook
```

:numref:`fig_jupyter` hiển thị đầu ra có thể sau khi bạn chạy Jupyter Notebook. Hàng cuối cùng là URL cho cổng 8888. 

![Output after running Jupyter Notebook. The last row is the URL for port 8888.](../img/jupyter.png)
:width:`700px`
:label:`fig_jupyter`

Vì bạn đã sử dụng chuyển tiếp cổng đến cổng 8889, bạn sẽ cần thay thế số cổng và sử dụng bí mật do Jupyter đưa ra khi mở URL trong trình duyệt cục bộ của bạn. 

## Đóng phiên bản chưa sử dụng

Vì dịch vụ đám mây được lập hóa đơn theo thời gian sử dụng, bạn nên đóng các phiên bản không được sử dụng. Lưu ý rằng có những lựa chọn thay thế: “dừng” một phiên bản có nghĩa là bạn sẽ có thể khởi động lại nó. Điều này giống như tắt nguồn cho máy chủ thông thường của bạn. Tuy nhiên, các phiên bản dừng vẫn sẽ được lập hóa đơn một lượng nhỏ cho dung lượng đĩa cứng được giữ lại. “Chấm dứt” xóa tất cả dữ liệu được liên kết với nó. Điều này bao gồm đĩa, do đó bạn không thể khởi động lại. Chỉ làm điều này nếu bạn biết rằng bạn sẽ không cần nó trong tương lai. 

Nếu bạn muốn sử dụng phiên bản như một mẫu cho nhiều trường hợp khác, nhấp chuột phải vào ví dụ trong :numref:`fig_connect` và chọn “Image” $\rightarrow$ “Create” để tạo một hình ảnh của phiên bản. Sau khi hoàn tất, hãy chọn “Trạng thái phiên bản” $\rightarrow$ “Terminate” để chấm dứt phiên bản. Lần sau khi bạn muốn sử dụng phiên bản này, bạn có thể làm theo các bước để tạo và chạy phiên bản EC2 được mô tả trong phần này để tạo một phiên bản dựa trên hình ảnh đã lưu. Sự khác biệt duy nhất là, trong “1. Chọn AMI” được hiển thị trong :numref:`fig_ubuntu`, bạn phải sử dụng tùy chọn “AMI của tôi” ở bên trái để chọn hình ảnh đã lưu của bạn. Phiên bản được tạo sẽ giữ lại thông tin được lưu trữ trên đĩa cứng hình ảnh. Ví dụ: bạn sẽ không phải cài đặt lại CIDA và các môi trường thời gian chạy khác. 

## Tóm tắt

* Bạn có thể khởi chạy và dừng các phiên bản theo yêu cầu mà không cần phải mua và xây dựng máy tính của riêng bạn.
* Bạn cần cài đặt trình điều khiển GPU phù hợp trước khi bạn có thể sử dụng chúng.

## Bài tập

1. Đám mây cung cấp sự tiện lợi, nhưng nó không rẻ. Tìm hiểu cách khởi chạy [spot instances](https://aws.amazon.com/ec2/spot/) để xem cách giảm giá.
1. Thử nghiệm với các máy chủ GPU khác nhau. Họ nhanh như thế nào?
1. Thử nghiệm với các máy chủ đa GPU. Làm thế nào tốt bạn có thể mở rộng mọi thứ lên?

[Discussions](https://discuss.d2l.ai/t/423)

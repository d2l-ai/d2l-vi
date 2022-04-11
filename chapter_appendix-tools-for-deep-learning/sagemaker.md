# Sử dụng Amazon SageMaker
:label:`sec_sagemaker`

Nhiều ứng dụng học sâu đòi hỏi một lượng tính toán đáng kể. Máy cục bộ của bạn có thể quá chậm để giải quyết những vấn đề này trong một khoảng thời gian hợp lý. Dịch vụ điện toán đám mây cung cấp cho bạn quyền truy cập vào các máy tính mạnh mẽ hơn để chạy các phần chuyên sâu GPU của cuốn sách này. Hướng dẫn này sẽ hướng dẫn bạn thông qua Amazon SageMaker: một dịch vụ cho phép bạn chạy cuốn sách này một cách dễ dàng. 

## Đăng ký và đăng nhập

Đầu tiên, chúng ta cần đăng ký một tài khoản tại https://aws.amazon.com/. Chúng tôi khuyến khích bạn sử dụng xác thực hai yếu tố để bảo mật bổ sung. Nó cũng là một ý tưởng tốt để thiết lập thanh toán chi tiết và thông báo chi tiêu để tránh bất kỳ bất ngờ bất ngờ trong trường hợp bạn quên dừng bất kỳ phiên bản đang chạy. Lưu ý rằng bạn sẽ cần một thẻ tín dụng. Sau khi đăng nhập vào tài khoản AWS của bạn, hãy truy cập [console](http://console.aws.amazon.com/) và tìm kiếm “SageMaker” (xem :numref:`fig_sagemaker`) sau đó nhấp để mở bảng SageMaker. 

![Open the SageMaker panel.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## Tạo một phiên bản SageMaker

Tiếp theo, chúng ta hãy tạo một ví dụ máy tính xách tay như được mô tả trong :numref:`fig_sagemaker-create`. 

![Create a SageMaker instance.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker cung cấp nhiều [instance types](https://aws.amazon.com/sagemaker/pricing/instance-types/) sức mạnh tính toán khác nhau và giá cả. Khi tạo một phiên bản, chúng ta có thể chỉ định tên phiên bản và chọn kiểu của nó. Năm :numref:`fig_sagemaker-create-2`, chúng tôi chọn `ml.p3.2xlarge`. Với một GPU Tesla V100 và CPU 8 nhân, phiên bản này đủ mạnh cho hầu hết các chương. 

![Choose the instance type.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
Một phiên bản máy tính xách tay Jupyter của cuốn sách này để phù hợp SageMaker có sẵn tại https://github.com/d2l-ai/d2l-en-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3`.
:end_tab:

:begin_tab:`pytorch`
Một phiên bản máy tính xách tay Jupyter của cuốn sách này để phù hợp SageMaker có sẵn tại https://github.com/d2l-ai/d2l-pytorch-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3`.
:end_tab:

:begin_tab:`tensorflow`
Một phiên bản máy tính xách tay Jupyter của cuốn sách này để phù hợp SageMaker có sẵn tại https://github.com/d2l-ai/d2l-tensorflow-sagemaker. We can specify this GitHub repository URL to let SageMaker clone this repository during instance creation, as shown in :numref:`fig_sagemaker-create-3`.
:end_tab:

![Specify the GitHub repository.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## Chạy và dừng phiên bản

Có thể mất vài phút trước khi phiên bản đã sẵn sàng. Khi nó đã sẵn sàng, bạn có thể nhấp vào liên kết “Mở Jupyter” như thể hiện trong :numref:`fig_sagemaker-open`. 

![Open Jupyter on the created SageMaker instance.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`

Sau đó, như thể hiện trong :numref:`fig_sagemaker-jupyter`, bạn có thể điều hướng qua máy chủ Jupyter chạy trên phiên bản này. 

![The Jupyter server running on the SageMaker instance.](../img/sagemaker-jupyter.png)
:width:`400px`
:label:`fig_sagemaker-jupyter`

Chạy và chỉnh sửa máy tính xách tay Jupyter trên phiên bản SageMaker tương tự như những gì chúng ta đã thảo luận trong :numref:`sec_jupyter`. Sau khi hoàn thành công việc của bạn, đừng quên dừng phiên bản để tránh sạc thêm, như thể hiện trong :numref:`fig_sagemaker-stop`. 

![Stop a SageMaker instance.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## Cập nhật máy tính xách tay

:begin_tab:`mxnet`
Chúng tôi sẽ thường xuyên cập nhật các máy tính xách tay trong kho lưu trữ GitHub [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker). Bạn chỉ cần sử dụng lệnh `git pull` để cập nhật lên phiên bản mới nhất.
:end_tab:

:begin_tab:`pytorch`
Chúng tôi sẽ thường xuyên cập nhật các máy tính xách tay trong kho lưu trữ GitHub [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker). Bạn chỉ cần sử dụng lệnh `git pull` để cập nhật lên phiên bản mới nhất.
:end_tab:

:begin_tab:`tensorflow`
Chúng tôi sẽ thường xuyên cập nhật các máy tính xách tay trong kho lưu trữ GitHub [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker). Bạn chỉ cần sử dụng lệnh `git pull` để cập nhật lên phiên bản mới nhất.
:end_tab:

Trước tiên, bạn cần mở một thiết bị đầu cuối như thể hiện trong :numref:`fig_sagemaker-terminal`. 

![Open a terminal on the SageMaker instance.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

Bạn có thể muốn thực hiện các thay đổi cục bộ của mình trước khi kéo các bản cập nhật. Ngoài ra, bạn chỉ cần bỏ qua tất cả các thay đổi cục bộ của mình bằng các lệnh sau trong thiết bị đầu cuối.

:begin_tab:`mxnet`
```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`pytorch`
```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```
:end_tab:

:begin_tab:`tensorflow`
```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```
:end_tab:

## Tóm tắt

* Chúng tôi có thể khởi chạy và dừng máy chủ Jupyter thông qua Amazon SageMaker để chạy cuốn sách này.
* Chúng tôi có thể cập nhật máy tính xách tay thông qua thiết bị đầu cuối trên phiên bản Amazon SageMaker.

## Bài tập

1. Hãy thử chỉnh sửa và chạy mã trong cuốn sách này bằng Amazon SageMaker.
1. Truy cập thư mục mã nguồn thông qua thiết bị đầu cuối.

[Discussions](https://discuss.d2l.ai/t/422)

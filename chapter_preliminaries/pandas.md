# Xử lý sơ bộ dữ liệu
:label:`sec_pandas`

Cho đến nay chúng tôi đã giới thiệu một loạt các kỹ thuật để thao tác dữ liệu đã được lưu trữ trong hàng chục. Để áp dụng deep learning để giải quyết các vấn đề trong thế giới thực, chúng ta thường bắt đầu với tiền xử lý dữ liệu thô, thay vì những dữ liệu được chuẩn bị độc đáo ở định dạng tensor. Trong số các công cụ phân tích dữ liệu phổ biến trong Python, gói `pandas` thường được sử dụng. Giống như nhiều gói mở rộng khác trong hệ sinh thái rộng lớn của Python, `pandas` có thể làm việc cùng với hàng chục. Vì vậy, chúng tôi sẽ đi qua một thời gian ngắn các bước để xử lý trước dữ liệu thô với `pandas` và chuyển đổi chúng sang định dạng tensor. Chúng tôi sẽ đề cập đến nhiều kỹ thuật tiền xử lý dữ liệu hơn trong các chương sau. 

## Đọc tập dữ liệu

Ví dụ, chúng ta bắt đầu bằng cách (** tạo một tập dữ liệu nhân tạo được lưu trữ trong tệp csv (giá trị phân cách bằng dấu phẩy) **) `../data/house_tiny.csv`. Dữ liệu được lưu trữ ở các định dạng khác có thể được xử lý theo những cách tương tự. 

Dưới đây chúng tôi viết từng hàng tập dữ liệu vào một tệp csv.

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row represents a data example
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

Để [** tải tập dữ liệu thô từ tệp csv đã tạo **], chúng tôi nhập gói `pandas` và gọi hàm `read_csv`. Tập dữ liệu này có bốn hàng và ba cột, trong đó mỗi hàng mô tả số phòng (“NumRooms”), kiểu hẻm (“Alley”), và giá (“Price”) của một ngôi nhà.

```{.python .input}
#@tab all
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Xử lý dữ liệu bị thiếu

Lưu ý rằng các mục “nan” bị thiếu giá trị. Để xử lý dữ liệu bị thiếu, các phương pháp điển hình bao gồm *imputation* và *xoá*, trong đó imputation thay thế các giá trị bị thiếu bằng các giá trị thay thế, trong khi xóa bỏ qua các giá trị bị thiếu. Ở đây chúng tôi sẽ xem xét imputation. 

Bằng cách lập chỉ mục dựa trên vị trí số nguyên (`iloc`), chúng tôi chia `data` thành `inputs` và `outputs`, trong đó cái trước lấy hai cột đầu tiên trong khi cột sau chỉ giữ cột cuối cùng. Đối với các giá trị số trong `inputs` bị thiếu, chúng ta [** thay thế các mục “nan” bằng giá trị trung bình của cùng một cột.**]

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

[**Đối với các giá trị phân loại hoặc rời rạc trong `inputs`, chúng tôi coi “nan” là một thể loại.**] Vì cột “Alley” chỉ lấy hai loại giá trị phân loại “Pave” và “nan”, `pandas` có thể tự động chuyển đổi cột này thành hai cột “Alley_Pave” và “Alley_nan”. Một hàng có kiểu hẻm là “Pave” sẽ đặt giá trị của “Alley_Pave” và “Alley_nan” thành 1 và 0. Một hàng có kiểu hẻm bị thiếu sẽ đặt giá trị của chúng thành 0 và 1.

```{.python .input}
#@tab all
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## Chuyển đổi sang định dạng Tensor

Bây giờ [** tất cả các mục trong `inputs` và `outputs` đều là số, chúng có thể được chuyển đổi sang định dạng tensor.**] Khi dữ liệu ở định dạng này, chúng có thể được thao tác thêm với các chức năng tensor mà chúng tôi đã giới thiệu trong :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## Tóm tắt

* Giống như nhiều gói mở rộng khác trong hệ sinh thái rộng lớn của Python, `pandas` có thể làm việc cùng với hàng chục.
* Imputation và xóa có thể được sử dụng để xử lý dữ liệu bị thiếu.

## Bài tập

Tạo một tập dữ liệu thô với nhiều hàng và cột hơn. 

1. Xóa cột với các giá trị còn thiếu nhiều nhất.
2. Chuyển đổi tập dữ liệu được xử lý trước sang định dạng tensor.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:

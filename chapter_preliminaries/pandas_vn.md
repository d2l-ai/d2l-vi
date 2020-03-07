<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Data Preprocessing
-->

# Tiền xử lý dữ liệu
:label:`sec_pandas`

<!--
So far we have introduced a variety of techniques for manipulating data that are already stored in `ndarray`s.
To apply deep learning to solving real-world problems, we often begin with preprocessing raw data, rather than those nicely prepared data in the `ndarray` format.
Among popular data analytic tools in Python, the `pandas` package is commonly used.
Like many other extension packages in the vast ecosystem of Python, `pandas` can work together with `ndarray`.
So, we will briefly walk through steps for preprocessing raw data with `pandas` and converting them into the `ndarray` format.
We will cover more data preprocessing techniques in later chapters.
-->

Cho đến giờ chúng tôi đã đề cập tới rất nhiều kỹ thuật thao tác dữ liệu được lưu trong dạng `ndarray`.
Nhưng để áp dụng học sâu vào giải quyết các vấn đề thực tế, ta thường phải bắt đầu bằng việc xử lý dữ liệu thô, chứ không có luôn dữ liệu ngăn nắp được chuẩn bị sẵn trong định dạng `ndarray`.
Trong số các công cụ phân tích dữ liệu phổ biến của Python, gói `pandas` khá được ưa chuộng.
Cũng như nhiều gói khác trong hệ sinh thái rộng lớn của Python, 'pandas' có thể được sử dụng kết hợp với định dạng `ndarray`.
Vì vậy, chúng ta sẽ đi nhanh qua các bước để tiền xử lý dữ liệu thô bằng `pandas` rồi đổi chúng sang dạng `ndarray`.
Nhiều kỹ thuật tiền xử lý dữ liệu khác sẽ được giới thiệu trong các chương sau.

<!--
## Reading the Dataset
-->

## Đọc tập dữ liệu

<!--
As an example, we begin by creating an artificial dataset that is stored in a csv (comma-separated values) file `../data/house_tiny.csv`. 
Data stored in other formats may be processed in similar ways. 
The following `mkdir_if_not_exist` function ensures that the directory `../data` exists. 
The comment `# Saved in the d2l package for later use` is a special mark where the following function, class, or import statements are also saved in the `d2l` package so that we can directly invoke `d2l.mkdir_if_not_exist()` later.
-->

Để lấy ví dụ, ta bắt đầu bằng việc tạo một tập dữ liệu nhân tạo lưu trong file csv  `../data/house_tiny.csv` (csv - *comma-separated values - giá trị tách nhau bằng dấu phẩy*).
Dữ liệu lưu ở các định dạng khác cũng có thể được xử lý tương tự.
Hàm `mkdir_if_not_exist` dưới đây để đảm bảo rằng thư mục `../data` tồn tại.
Chú thích `# Saved in the d2l package for later use` (*Lưu lại trong gói d2l để dùng sau*) là kí hiệu đánh dấu các hàm, lớp hoặc các lệnh `import` được lưu trong gói `d2l`, để sau này ta có thể trực tiếp gọi hàm `d2l.mkdir_if_not_exist()`.

```{.python .input}
import os

# Saved in the d2l package for later use
def mkdir_if_not_exist(path):
    if not isinstance(path, str):
        path = os.path.join(*path)
    if not os.path.exists(path):
        os.makedirs(path)
```

<!--
Below we write the dataset row by row into a csv file.
-->

Sau đây ta ghi tệp dữ liệu vào file csv theo từng hàng một.

```{.python .input}
data_file = '../data/house_tiny.csv'
mkdir_if_not_exist('../data')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # Column names
    f.write('NA,Pave,127500\n')  # Each row is a data point
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
To load the raw dataset from the created csv file, we import the `pandas` package and invoke the `read_csv` function.
This dataset has $4$ rows and $3$ columns, where each row describes the number of rooms ("NumRooms"), the alley type ("Alley"), and the price ("Price") of a house.
-->

Để nạp tập dữ liệu thô từ tệp csv vừa được tạo ra, ta dùng gói thư viện `pandas` và gọi hàm `read_csv`.
Bộ dữ liệu này có $4$ hàng và $3$ cột, trong đó mỗi hàng biểu thị số phòng ("NumRooms"), kiểu lối đi ("Alley"), và giá ("Price") của căn nhà.

```{.python .input}
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```
<!-- Kết thúc revise phần 1 ở đây -->
<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Handling Missing Data
-->

## Xử lý dữ liệu thiếu

<!--
Note that "NaN" entries are missing values.
To handle missing data, typical methods include *imputation* and *deletion*, where imputation replaces missing values with substituted ones, while deletion ignores missing values. Here we will consider imputation.
-->

Để ý rằng giá trị "NaN" là các giá trị bị thiếu.
Để xử lý dữ liệu thiếu, các cách thường được áp dụng là *quy buộc* (*imputation*) và *xoá bỏ* (*deletion*), trong đó quy buộc thay thế giá trị bị thiếu bằng giá trị khác, trong khi xoá bỏ sẽ bỏ qua các giá trị bị thiếu.
Dưới đây chúng ta xem xét phương pháp quy buộc.

<!--
By integer-location based indexing (`iloc`), we split `data` into `inputs` and `outputs`, where the former takes the first 2 columns while the later only keeps the last column.
For numerical values in `inputs` that are missing, we replace the "NaN" entries with the mean value of the same column.
-->

Bằng phương pháp đánh chỉ số theo số nguyên (`iloc`), chúng ta tách `data` thành `inputs` (tương ứng với hai cột đầu) và `outputs` (tương ứng với cột cuối cùng).
Với các giá trị số bị thiếu trong `inputs`, ta thay thế phần tử "NaN" bằng giá trị trung bình cộng của cùng cột đó. 

```{.python .input}
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

<!--
For categorical or discrete values in `inputs`, we consider "NaN" as a category.
Since the "Alley" column only takes 2 types of categorical values "Pave" and "NaN", `pandas` can automatically convert this column to 2 columns "Alley_Pave" and "Alley_nan".
A row whose alley type is "Pave" will set values of "Alley_Pave" and "Alley_nan" to $1$ and $0$.
A row with a missing alley type will set their values to $0$ and $1$.
-->

Với các giá trị dạng hạng mục hoặc số rời rạc trong `inputs`, ta coi "NaN" là một mục riêng.
Vì cột "Alley" chỉ nhận 2 giá trị riêng lẻ là "Pave" (được lát gạch) và "NaN", `pandas` có thể tự động chuyển cột này thành 2 cột "Alley_Pave" và "Alley_nan". 
Những hàng có kiểu lối đi là "Pave" sẽ có giá trị của cột "Alley_Pave" và cột "Alley_nan" tương ứng là $1$ và $0$.
Hàng mà không có giá trị cho kiểu lối đi sẽ có giá trị cột "Alley_Pave" và cột "Alley_nan" lần lượt là $0$ và $1$.

```{.python .input}
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Conversion to the  `ndarray` Format
-->

## Chuyển sang định dạng `ndarray`

<!--
Now that all the entries in `inputs` and `outputs` are numerical, they can be converted to the `ndarray` format.
Once data are in this format, they can be further manipulated with those `ndarray` functionalities that we have introduced in :numref:`sec_ndarray`.
-->

Giờ thì toàn bộ các giá trị trong `inputs` và `outputs` đã ở dạng số, chúng đã có thể được chuyển sang định dạng `ndarray`.
Khi đã ở định dạng này, chúng có thể được biến đổi và xử lý với những chức năng của `ndarray` đã được giới thiệu ở :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

<!--
## Summary
-->

## Tóm tắt

<!--
* Like many other extension packages in the vast ecosystem of Python, `pandas` can work together with `ndarray`.
* Imputation and deletion can be used to handle missing data.
-->

* Cũng như nhiều gói mở rộng trong hệ sinh thái khổng lồ của Python, `pandas` có thể làm việc được với `ndarray`.
* Phương pháp quy buộc hoặc xoá bỏ có thể dùng để xử lý dữ liệu bị thiếu.

<!--
## Exercises
-->

## Bài tập

<!--
Create a raw dataset with more rows and columns.
-->

Tạo một tập dữ liệu với nhiều hàng và cột hơn.

<!--
1. Delete the column with the most missing values.
2. Convert the preprocessed dataset to the `ndarray` format.
-->

1. Xoá cột có nhiều giá trị bị thiếu nhất.
2. Chuyển bộ dữ liệu đã được xử lý sang định dạng `ndarray`.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/4973)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2315)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

<!--
![](../img/qr_pandas.svg)
-->


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

* Lê Khắc Hồng Phúc
* Nguyễn Cảnh Thướng
* Phạm Hồng Vinh
* Đoàn Võ Duy Thanh
* Vũ Hữu Tiệp
* Mai Sơn Hải

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

Trước tới nay chúng ta đã đề cập tới rất nhiều kỹ thuật thao tác dữ liệu được lưu trong dạng `ndarray`.
Nhưng để áp dụng học sâu vào giải quyết các vấn đề thực tế, ta thường phải bắt đầu bằng việc xử lý dữ liệu thô, chứ không phải luôn có ngay dữ liệu ngăn nắp đã chuẩn bị sẵn trong định dạng `ndarray`
Trong số các công cụ phân tích dữ liệu phổ biến của Python, gói `pandas` hay được sử dụng nhiều.
Giống nhiều gói khác trong hệ sinh thái Python, `pandas` có thể làm việc cùng định dạng `ndarray`.
Vì vậy, chúng ta sẽ đi nhanh qua các bước để tiền xử lý dữ liệu thô bằng `pandas` rồi đổi chúng sang dạng `ndarray`.
Sau đó ta sẽ bao quát nhiều kỹ thuật xử lý dữ liệu hơn trong các chương sau.

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

Lấy một ví dụ, ta bắt đầu bằng việc tạo một tập dữ liệu nhân tạo lưu trong file csv  `../data/house_tiny.csv` (csv - *comma-separated values - giá trị tách nhau bằng dấu phẩy*).
Dữ liệu trong các định dạng khác cũng có thể được xử lý tương tự.
Hàm `mkdir_if_not_exist` dưới đây để đảm bảo rằng thư mục `../data` có tồn tại.
Chú thích `# Saved in the d2l package for later use` (*Lưu lại trong gói d2l để dùng sau*) là nhãn riêng cho các hàm, các lớp hoặc các lệnh import sau này được lưu trong gói `d2l` để ta có thể trực tiếp gọi hàm `d2l.mkdir_if_not_exist()` sau này.

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

Sau đây ta ghi tệp dữ liệu vào file csv theo kiểu hàng nối hàng.

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

*dịch đoạn phía trên*

```{.python .input}
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Handling Missing Data
-->

## *dịch tiêu đề phía trên*

<!--
Note that "NaN" entries are missing values.
To handle missing data, typical methods include *imputation* and *deletion*, where imputation replaces missing values with substituted ones, while deletion ignores missing values. Here we will consider imputation.
-->

*dịch đoạn phía trên*

<!--
By integer-location based indexing (`iloc`), we split `data` into `inputs` and `outputs`, where the former takes the first 2 columns while the later only keeps the last column.
For numerical values in `inputs` that are missing, we replace the "NaN" entries with the mean value of the same column.
-->

*dịch đoạn phía trên*

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

*dịch đoạn phía trên*

```{.python .input}
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Conversion to the  `ndarray` Format
-->

## *dịch tiêu đề phía trên*

<!--
Now that all the entries in `inputs` and `outputs` are numerical, they can be converted to the `ndarray` format.
Once data are in this format, they can be further manipulated with those `ndarray` functionalities that we have introduced in :numref:`sec_ndarray`.
-->

*dịch đoạn phía trên*

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* Like many other extension packages in the vast ecosystem of Python, `pandas` can work together with `ndarray`.
* Imputation and deletion can be used to handle missing data.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
Create a raw dataset with more rows and columns.
-->

*dịch đoạn phía trên*

<!--
1. Delete the column with the most missing values.
2. Convert the preprocessed dataset to the `ndarray` format.
-->

*dịch đoạn phía trên*

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


### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
*

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

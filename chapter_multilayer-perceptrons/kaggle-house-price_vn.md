<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Predicting House Prices on Kaggle
-->

# Dự đoán Giá Nhà trên Kaggle
:label:`sec_kaggle_house`

<!--
In the previous sections, we introduced the basic tools for building deep networks and performing capacity control via dimensionality-reduction, weight decay and dropout.
You are now ready to put all this knowledge into practice by participating in a Kaggle competition.
[Predicting house prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) is a great place to start: the data is reasonably generic 
and does not have the kind of rigid structure that might require specialized models the way images or audio might.
This dataset, collected by Bart de Cock in 2011 :cite:`De-Cock.2011`, is considerably larger than the famous the [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) of Harrison and Rubinfeld (1978).
It boasts both more examples and more features, covering house prices in Ames, IA from the period of 2006-2010.
-->

Trong phần trước, chúng tôi đã giới thiệu những công cụ cơ bản để xây dựng mạng học sâu và kiểm soát năng lực của nó thông qua việc giảm chiều dữ liệu, suy giảm trọng số và dropout.
Giờ bạn đã sẵn sàng để ứng dụng tất cả những kiến thức này vào thực tiễn bằng cách tham gia một cuộc thi trên Kaggle.
[Predicting house prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) là một bài toán tuyệt vời để bắt đầu: dữ liệu tương đối khái quát, không có cấu trúc cứng nhắc nên không đòi hỏi những mô hình đặc biệt như các bài toán có dữ liệu ảnh và âm thanh. 
Bộ dữ liệu này được thu thập bởi Bart de Cock vào năm 2011 :cite:`De-Cock.2011`, lớn hơn rất nhiều bộ dữ liệu nổi tiếng [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) của Harrison và Rubinfeld (1978).
Nó có nhiều ví dụ và đặc trưng hơn, chứa thông tin về giá nhà ở Ames, IA trong khoảng thời gian từ 2006-2010.

<!--
In this section, we will walk you through details of data preprocessing, model design, hyperparameter selection and tuning.
We hope that through a hands-on approach, you will be able to observe the effects of capacity control, feature extraction, etc. in practice.
This experience is vital to gaining intuition as a data scientist.
-->

Trong phần này, chúng tôi sẽ hướng dẫn bạn một cách chi tiết các bước tiền xử lý dữ liệu, thiết kế mô hình, lựa chọn và điều chỉnh siêu tham số. 
Chúng tôi mong rằng thông qua việc thực hành, bạn sẽ có thể quan sát được những tác động của kiểm soát năng lực, trích xuất đặc trưng, v.v. trong thực tiễn. 
Kinh nghiệm này rất quan trọng để bạn có được trực giác của một nhà khoa học dữ liệu. 
<!--
## Downloading and Caching Datasets
-->

## Tải và Lưu trữ Bộ dữ liệu

<!--
Throughout the book we will train and test models on various downloaded datasets. 
Here we implement several utility functions to facilitate data downloading. 
First, we maintain a dictionary `DATA_HUB` that maps a string name to a URL with the SHA-1 of the file at the URL, 
where SHA-1 verifies the integrity of the file. Such datasets are hosted on the `DATA_URL` site.
-->

Trong suốt cuốn sách chúng ta sẽ cần tải và thử nghiệm nhiều mô hình trên các bộ dữ liệu khác nhau. 
Ta sẽ lập trình một số hàm tiện ích để hỗ trợ cho việc tải dữ liệu.
Đầu tiên, ta cần khởi tạo một từ điển `DATA_HUB` nhằm ánh xạ một xâu ký tự đến đường dẫn (URL) với SHA-1 của tệp tại đường dẫn đó, 
trong đó SHA-1 dùng để xác minh tính toàn vẹn của tệp. Các bộ dữ liệu này được lưu trữ trên trang `DATA_URL`.

```{.python .input  n=2}
import os
from mxnet import gluon
import zipfile
import tarfile

# Saved in the d2l package for later use
DATA_HUB = dict()

# Saved in the d2l package for later use
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
The following `download` function downloads the dataset from
the URL mapping the specified dataset `name` to a local cache directory (`../data` by default).
If the file already exists in the cache directory and its SHA-1 matches the one stored in `DATA_HUB`, the cached file will be used and no downloading is needed.
That is to say, you only need to download datasets once with a network connection.
This `download` function returns the name of the downloaded file.
-->

Hàm `download` dưới đây tải bộ dữ liệu từ đường dẫn ứng với tên `name` cụ thể và lưu trữ nó tại bộ nhớ cục bộ (mặc định tại `../data`).
Nếu tệp trên đã tồn tại trong bộ nhớ đệm và SHA-1 của nó khớp với tệp trong `DATA_HUB`, tệp trong lưu trữ sẽ được sử dụng và việc tải về là không cần thiết. 
Điều này nghĩa là, bạn chỉ cần tải bộ dữ liệu về với một lần kết nối mạng.
Hàm `download` trả về tên của tệp được tải xuống.

```{.python .input  n=6}
# Saved in the d2l package for later use
def download(name, cache_dir='../data'):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, "%s doesn't exist" % name
    url, sha1 = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
    return gluon.utils.download(url, cache_dir, sha1_hash=sha1)
```

<!--
We also implement two additional functions: one is to download and extract a zip/tar file, and the other to download all the files from `DATA_HUB` 
(most of the datasets used in this book) into the cache directory. 
You may invoke the latter to download all these datasets once and for all if your network connection is slow.
-->

Chúng ta cũng xây dựng hai hàm bổ sung khác: một hàm là để tải và giải nén tệp zip/tar, và hàm còn lại để tải tất cả các file từ `DATA_HUB`(chứa phần lớn các bộ dữ liệu được sử dụng trong cuốn sách này) vào bộ nhớ đệm. 
Bạn có thể sử dụng hàm thứ hai để tải tất cả bộ dữ liệu trong một lần nếu kết nối mạng của bạn chậm.

```{.python .input  n=11}
# Saved in the d2l package for later use
def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext == '.tar' or ext == '.gz':
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted'
    fp.extractall(base_dir)
    if folder:
        return base_dir + '/' + folder + '/'
    else:
        return data_dir + '/'

# Saved in the d2l package for later use
def download_all():
    """Download all files in the DATA_HUB"""
    for name in DATA_HUB:
        download(name)
```

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Kaggle
-->

## Kaggle

<!--
[Kaggle](https://www.kaggle.com) is a popular platform for machine learning competitions.
It combines data, code and users in a way to allow for both collaboration and competition.
While leaderboard chasing can sometimes get out of control, there is also a lot to be said for 
the objectivity in a platform that provides fair and direct quantitative comparisons between your approaches and those devised by your competitors.
Moreover, you can checkout the code from (some) other competitors' submissions and pick apart their methods to learn new techniques.
If you want to participate in one of the competitions, you need to register for an account as shown in :numref:`fig_kaggle` (do this now!).
-->

[Kaggle](https://www.kaggle.com) là một nền tảng phổ biến cho các cuộc thi học máy.
Nó kết hợp dữ liệu, mã lập trình và người dùng cho cả mục đích hợp tác và thi thố.
Mặc dù việc cạnh tranh trên bảng xếp hạng nhiều khi vượt khỏi tầm kiểm soát, ta không thể không nhắc đến tính khách quan của nền tảng này có được từ sự so sánh công bằng định lượng trực tiếp giữa phương pháp của bạn với các phương pháp đến từ đối thủ. 
Nếu bạn muốn tham gia vào một trong những cuộc thi, bạn cần đăng ký một tài khoản như trong :numref:`fig_kaggle` (làm ngay đi!).

<!--
![Kaggle website](../img/kaggle.png)
-->

![Trang web Kaggle](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

<!--
On the House Prices Prediction page as illustrated in :numref:`fig_house_pricing`, you can find the dataset (under the "Data" tab), submit predictions, see your ranking, etc.,
The URL is right here:
-->

Trên trang Dự Đoán Giá Nhà (_House Prices Prediction_) được mô tả ở :numref:`fig_house_pricing`, bạn có thể tìm thấy bộ dữ liệu (dưới thanh "Data"), nộp kết quả dự đoán và xem thứ hạng của bạn, v.v. 
Đường dẫn:

> https://www.kaggle.com/c/house-prices-advanced-regression-technique 

<!--
![House Price Prediction](../img/house_pricing.png)
-->

![Dự đoán Giá Nhà](../img/house_pricing.png)
:width:`400px`
:label:`fig_house_pricing`

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
## Accessing and Reading the Dataset
-->

## Truy cập và Đọc Bộ dữ liệu

<!--
Note that the competition data is separated into training and test sets.
Each record includes the property value of the house and attributes such as street type, year of construction, roof type, basement condition, etc.
The features represent multiple data types.
Year of construction, for example, is represented with integers roof type is a discrete categorical feature, other features are represented with floating point numbers.
And here is where reality comes in: for some examples, some data is altogether missing with the missing value marked simply as 'na'.
The price of each house is included for the training set only (it is a competition after all).
You can partition the training set to create a validation set, but you will only find out how you perform on the official test set when you upload your predictions and receive your score.
The "Data" tab on the competition tab has links to download the data.
-->

Lưu ý rằng dữ liệu của cuộc thi được tách thành tập huấn luyện và tập kiểm tra.
Mỗi tập dữ liệu bao gồm giá trị tài sản của ngôi nhà và các thuộc tính liên quan bao gồm loại đường phố, năm xây dựng, loại ngói, tình trạng tầng hầm, v.v.
Các đặc trưng được biểu diễn bởi nhiều kiểu dữ liệu.
Ví dụ, năm xây dựng được biểu diễn bởi số nguyên, loại ngói là các lớp đặc trưng riêng biệt, các đặc trưng khác thì được biểu diễn bởi số thực dấu phẩy động (_floating point number_).
Và đây là khi ta đối mặt với vấn đề thực tiễn: ở một vài mẫu, nhiều dữ liệu bị thiếu và được chú thích đơn giản là 'na'.
Giá của mỗi căn nhà chỉ được cung cấp trong tập huấn luyện (sau cùng thì đây vẫn là một cuộc thi).
Bạn có thể chia nhỏ tập huấn luyện để tạo tập kiểm định, tuy nhiên bạn sẽ chỉ biết được mô hình của bạn thể hiện như thế nào trên tập kiểm tra chính thức khi bạn tải lên kết quả dự đoán của mình và nhận điểm sau đó.
Thanh "Data" trên cuộc thi có đường link để tải về bộ dữ liệu.  

<!--
We will read and process the data using `pandas`, an [efficient data analysis toolkit](http://pandas.pydata.org/pandas-docs/stable/), 
so you will want to make sure that you have `pandas` installed before proceeding further. 
Fortunately, if you are reading in Jupyter, we can install pandas without even leaving the notebook.
-->

Chúng ta sẽ đọc và xử lý dữ liệu với `pandas`, một [công cụ phân tích dữ liệu hiệu quả](http://pandas.pydata.org/pandas-docs/stable/), vì vậy hãy đảm bảo rằng bạn đã cài đặt `pandas` trước khi tiếp tục.
Một điều may mắn là, nếu bạn đang sử dụng Jupyter, bạn có thể cài đặt pandas mà không cần thoát khỏi notebook. 

```{.python .input  n=3}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
import d2l
from mxnet import autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

<!--
For convenience, we download and cache the Kaggle housing dataset from the `DATA_URL` website. For the other Kaggle competitions, you may need to download them manually.
-->

Để thuận tiện, chúng ta sẽ tải và lưu bộ dữ liệu giá nhà Kaggle từ trang web `DATA_URL`. Với những cuộc thi Kaggle khác, bạn có thể cần tải dữ liệu về theo cách thủ công.  

```{.python .input}
# Saved in the d2l package for later use
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

# Saved in the d2l package for later use
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

<!--
To load the two csv files containing training and test data respectively we use Pandas.
-->

Ta sử dụng Pandas để nạp lần lượt hai tệp csv chứa dữ liệu huấn luyện và kiểm tra.

```{.python .input  n=14}
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

<!--
The training dataset includes $1,460$ examples, $80$ features, and $1$ label, while the test data contains $1,459$ examples and $80$ features.
-->

Tập huấn luyện chứa $1,460$ mẫu, $80$ đặc trưng, và $1$ nhãn.
Tập kiểm tra chứa $1,459$ mẫu và $80$ đặc trưng.

```{.python .input  n=11}
print(train_data.shape)
print(test_data.shape)
```

<!--
Let’s take a look at the first 4 and last 2 features as well as the label (SalePrice) from the first 4 examples:
-->

Hãy cùng xem xét 4 đặc trưng đầu tiên, 2 đặc trưng cuối cùng và nhãn (giá nhà) của 4 mẫu đầu tiên:

```{.python .input  n=28}
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

<!--
We can see that in each example, the first feature is the ID.
This helps the model identify each training example.
While this is convenient, it does not carry any information for prediction purposes.
Hence we remove it from the dataset before feeding the data into the network.
-->

Có thể thấy với mỗi mẫu, đặc trưng đầu tiên là ID.
Điều này giúp mô hình xác định được từng mẫu. 
Mặc dù việc này khá thuận tiện, nó không mang bất kỳ thông tin nào cho mục đích dự đoán. 
Do đó chúng ta sẽ lược bỏ nó ra khỏi bộ dữ liệu trước khi đưa vào mạng nơ-ron. 

```{.python .input  n=30}
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Data Preprocessing
-->

## Tiền xử lý Dữ liệu

<!--
As stated above, we have a wide variety of data types.
Before we feed it into a deep network, we need to perform some amount of processing.
Let's start with the numerical features.
We begin by replacing missing values with the mean.
This is a reasonable strategy if features are missing at random.
To adjust them to a common scale, we rescale them to zero mean and unit variance.
This is accomplished as follows:
-->

Như đã nói ở trên, chúng ta có rất nhiều kiểu dữ liệu.
Trước khi đưa nó vào mạng học sâu, ta cần thực hiện một số phép xử lý. 
Hãy bắt đầu với các đặc trưng số học. 
Trước hết ta thay thế các giá trị còn thiếu bằng giá trị trung bình.
Đây là chiến lược hợp lý nếu các đặc trưng bị thiếu một cách ngẫu nhiên. 
Để điểu chỉnh theo một thang đo chung, ta chuyển đổi tỷ lệ để chúng có trung bình bằng không (_zero mean_) và phương sai đơn vị (_unit variance_). 
Điều này có thể đạt được bằng cách:

$$x \leftarrow \frac{x - \mu}{\sigma}.$$

<!--
To check that this transforms $x$ to data with zero mean and unit variance simply calculate
$E[(x-\mu)/\sigma] = (\mu - \mu)/\sigma = 0$.
To check the variance we use $E[(x-\mu)^2] = \sigma^2$ and thus the transformed variable has unit variance.
The reason for "normalizing" the data is that it brings all features to the same order of magnitude.
After all, we do not know *a priori* which features are likely to be relevant.
-->

Để kiểm tra xem công thức trên có chuyển đổi $x$ thành dữ liệu với trung bình bằng không hay không, ta có thể tính $E[(x-\mu)/\sigma] = (\mu - \mu)/\sigma = 0$. 
Để kiểm tra phương sai ta tính $E[(x-\mu)^2] = \sigma^2$, như vậy biến chuyển đổi sẽ có phương sai đơn vị. 
Lý do của việc "chuẩn hóa" dữ liệu là để đưa tất cả các đặc trưng về có cùng độ lớn. 
Vì sau cùng, chúng ta không thể *biết trước* được các đặc trưng nào là quan trọng. 


```{.python .input  n=6}
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

<!--
Next we deal with discrete values.
This includes variables such as 'MSZoning'.
We replace them by a one-hot encoding in the same manner as how we transformed multiclass classification data into a vector of $0$ and $1$.
For instance, 'MSZoning' assumes the values 'RL' and 'RM'.
They map into vectors $(1, 0)$ and $(0, 1)$ respectively.
Pandas does this automatically for us.
-->

Tiếp theo chúng ta sẽ xử lý các giá trị rời rạc.
Nó bao gồm những biến như 'MSZoning'.
Ta sẽ thay thế chúng bằng biểu diễn one-hot theo đúng cách mà ta đã chuyển đổi dữ liệu phân loại đa lớp thành vector chứa $0$ và $1$.
Ví dụ, 'MSZoning' bao gồm các giá trị 'RL' và 'RM', tương ứng lần lượt với vector $(1, 0)$ and $(0, 1)$. 
Pandas tự động làm việc này cho chúng ta.

```{.python .input  n=7}
# Dummy_na=True refers to a missing value being a legal eigenvalue, and
# creates an indicative feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

<!--
You can see that this conversion increases the number of features from 79 to 331.
Finally, via the `values` attribute, we can extract the NumPy format from the Pandas dataframe and convert it into MXNet's native `ndarray` representation for training.
-->

Bạn có thể thấy sự chuyển đổi này làm tăng số lượng các đặc trưng từ 79 lên 331. 
Cuối cùng, thông qua thuộc tính `values`, ta có thể trích xuất định dạng NumPy từ khung dữ liệu Pandas và chuyển đổi nó thành biểu diễn `ndarray` gốc của MXNet dành cho mục đích huấn luyện.

```{.python .input  n=9}
n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values, dtype=np.float32)
test_features = np.array(all_features[n_train:].values, dtype=np.float32)
train_labels = np.array(train_data.SalePrice.values,
                        dtype=np.float32).reshape(-1, 1)
```

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
## Training
-->

## Huấn luyện

<!--
To get started we train a linear model with squared loss.
Not surprisingly, our linear model will not lead to a competition winning submission but it provides a sanity check to see whether there is meaningful information in the data.
If we cannot do better than random guessing here, then there might be a good chance that we have a data processing bug.
And if things work, the linear model will serve as a baseline giving us some intuition about how close the simple model gets to the best reported models, 
giving us a sense of how much gain we should expect from fancier models.
-->

Để bắt đầu, ta sẽ huấn luyện một mô hình tuyến tính với hàm mất mát bình phương.
Tất nhiên là mô hình tuyến tính sẽ không thể thắng cuộc thi được, nhưng nó vẫn cho ta một phép kiểm tra sơ bộ để xem dữ liệu có chứa thông tin ý nghĩa hay không.
Nếu mô hình này không thể đạt chất lượng tốt hơn việc đoán mò, khả năng cao là ta đang có lỗi trong quá trình xử lý dữ liệu.
Còn nếu nó hoạt động, mô hình tuyến tính sẽ đóng vai trò như một mốc khởi điểm, giúp ta hình dung khoảng cách giữa một mô hình đơn giản và các mô hình tốt nhất hiện có, cũng như mức độ cải thiện mà ta mong muốn từ các mô hình "xịn" hơn.
```{.python .input  n=13}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

<!--
With house prices, as with stock prices, we care about relative quantities more than absolute quantities.
More concretely, we tend to care more about the relative error $\frac{y - \hat{y}}{y}$ than about the absolute error $y - \hat{y}$.
For instance, if our prediction is off by USD 100,000 when estimating the price of a house in Rural Ohio, where the value of a typical house is 125,000 USD, then we are probably doing a horrible job.
On the other hand, if we err by this amount in Los Altos Hills, California, this might represent a stunningly accurate prediction (their, the median house price exceeds 4 million USD).
-->

Với giá nhà (hay giá cổ phiếu), ta quan tâm đến các đại lượng tương đối hơn các đại lượng tuyệt đối.
Cụ thể hơn, ta thường quan tâm đến lỗi tương đối $\frac{y - \hat{y}}{y}$ hơn lỗi tuyệt đối $y - \hat{y}$.
Ví dụ, nếu dự đoán giá một ngôi nhà ở Rural Ohio bị lệch đi 100,000 đô-la, mà giá thông thường một ngôi nhà ở đó là 125,000 đô-la, có lẽ mô hình đang làm việc rất kém.
Mặt khác, nếu ta có cùng độ lệch như vậy khi dự đoán giá nhà ở Los Altos Hills, California (giá nhà trung bình ở đây tầm hơn 4 triệu đô), có thể dự đoán này lại rất chính xác.

<!--
One way to address this problem is to measure the discrepancy in the logarithm of the price estimates.
In fact, this is also the official error metric used by the competition to measure the quality of submissions.
After all, a small value $\delta$ of $\log y - \log \hat{y}$ translates into $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
This leads to the following loss function:
-->

Một cách để giải quyết vấn đề này là tính hiệu của log giá trị dự đoán và log giá trị thật sự.
Thực ra đây chính là phép đo lỗi chính thức được sử dụng trong cuộc thi để đánh giá chất lượng của các lần nộp bài.
Sau cùng, một giá trị $\delta$ bằng $\log y - \log \hat{y}$ nhỏ đồng nghĩa với việc $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
Điều này dẫn đến hàm mất mát sau:

$$L = \sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=11}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

<!--
Unlike in previous sections, our training functions here will rely on the Adam optimizer
(a slight variant on SGD that we will describe in greater detail later).
The main appeal of Adam vs vanilla SGD is that the Adam optimizer, despite doing no better (and sometimes worse) given unlimited resources for hyperparameter optimization, 
people tend to find that it is significantly less sensitive to the initial learning rate.
This will be covered in further detail later on when we discuss the details in :numref:`chap_optimization`.
-->

Khác với các mục trước, hàm huấn luyện ở đây sử dụng bộ tối ưu Adam
(một biến thể của SGD mà chúng tôi sẽ mô tả cụ thể hơn sau này).
Lợi thế chính của Adam so với SGD nguyên bản là: nó không quá nhạy cảm với tốc độ học ban đầu, 
mặc dù kết quả cũng không tốt hơn (đôi khi còn tệ hơn) SGD nếu tài nguyên để tối ưu siêu tham số là vô hạn.
Bộ tối ưu này sẽ được mô tả cụ thể hơn trong :numref:`chap_optimization`.

```{.python .input  n=14}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## k-Fold Cross-Validation
-->

## Kiểm định chéo gập k-lần

<!--
If you are reading in a linear fashion, you might recall that we introduced k-fold cross-validation in the section where we discussed how to deal with model section (:numref:`sec_model_selection`). 
We will put this to good use to select the model design and to adjust the hyperparameters.
We first need a function that returns the $i^\mathrm{th}$ fold of the data in a k-fold cross-validation procedure.
It proceeds by slicing out the $i^\mathrm{th}$ segment as validation data and returning the rest as training data.
Note that this is not the most efficient way of handling data and we would definitely do something much smarter if our dataset was considerably larger.
But this added complexity might obfuscate our code unnecessarily so we can safely omit here owing to the simplicity of our problem.
-->

Nếu bạn đang đọc theo kiểu từ đầu đến cuối thì có thể bạn sẽ nhớ ra rằng kiểm định chéo gập k-lần đã từng được giới thiệu khi ta thảo luận về cách lựa chọn mô hình (: numref: `sec_model_selection`).
Ta sẽ ứng dụng kỹ thuật này để lựa chọn thiết kế mô hình và điều chỉnh các siêu tham số.
Trước tiên ta cần một hàm trả về phần thứ $i^\mathrm{th}$ của dữ liệu trong kiểm định chéo gập k-lần.
Việc này được tiến hành bằng cách cắt chọn (_slicing_) phần thứ $i^\mathrm{th}$ để làm dữ liệu kiểm định và dùng phần còn lại làm dữ liệu huấn luyện.
Cần lưu ý rằng đây không phải là cách xử lý dữ liệu hiệu quả nhất và ta chắc chắn sẽ dùng một cách khôn ngoan hơn để xử lý một tập dữ liệu có kích thước lớn hơn nhiều. 
Nhưng sự phức tạp được thêm vào này có thể làm rối mã nguồn một cách không cần thiết, vì vậy để đơn giản hóa vấn đề ở đây ta có thể an toàn bỏ qua.

```{.python .input}
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = np.concatenate((X_train, X_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    return X_train, y_train, X_valid, y_valid
```

<!--
The training and verification error averages are returned when we train $k$ times in the k-fold cross-validation.
-->

Trong kiểm định chéo gập k-lần, ta sẽ huấn luyện mô hình $k$ lần và trả về trung bình lỗi huấn luyện và trung bình lỗi kiểm định.

```{.python .input  n=15}
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs+1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse',
                     legend=['train', 'valid'], yscale='log')
        print('fold %d, train rmse: %f, valid rmse: %f' % (
            i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
```

<!--
## Model Selection
-->

## Lựa chọn Mô hình

<!--
In this example, we pick an un-tuned set of hyperparameters and leave it up to the reader to improve the model.
Finding a good choice can take quite some time, depending on how many things one wants to optimize over.
Within reason, the k-fold cross-validation approach is resilient against multiple testing.
However, if we were to try out an unreasonably large number of options it might fail since we might just get lucky on the validation split with a particular set of hyperparameters.
-->

Trong ví dụ này, chúng tôi chọn một bộ siêu tham số chưa được tinh chỉnh và để dành cơ hội cải thiện mô hình cho bạn đọc.
Để tìm ra được một bộ siêu tham số tốt có thể sẽ tốn khá nhiều thời gian tùy thuộc vào số lượng siêu tham số mà ta muốn tối ưu.
Phương pháp kiểm định chéo gập k-lần có tính ổn định cao khi thực hiện với nhiều thử nghiệm, tới một ngưỡng nhất định.
Tuy nhiên, nếu ta thử nghiệm một số lượng rất lớn các lựa chọn thì phương pháp này có khả năng thất bại vì có thể ta chỉ may mắn trong việc chia tập kiểm định phù hợp với một bộ siêu tham số nhất định.

```{.python .input  n=16}
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print('%d-fold validation: avg train rmse: %f, avg valid rmse: %f'
      % (k, train_l, valid_l))
```

<!--
You will notice that sometimes the number of training errors for a set of hyper-parameters can be very low, while the number of errors for the $K$-fold cross-validation may be higher. 
This is an indicator that we are overfitting.
Therefore, when we reduce the amount of training errors, we need to check whether the amount of errors in the k-fold cross-validation have also been reduced accordingly.
-->

Bạn sẽ thấy rằng đôi khi lỗi huấn luyện cho một bộ siêu tham số có thể rất thấp, trong khi lỗi của kiểm định k-phần có thể cao hơn.
Đây là dấu hiệu của sự quá khớp.
Vì vậy khi ta giảm lỗi huấn luyện, ta cũng nên kiểm tra xem liệu lỗi kiểm định chéo gập k-lần có giảm tương ứng hay không.

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
##  Predict and Submit
-->

## Dự đoán và Nộp bài

<!--
Now that we know what a good choice of hyperparameters should be, we might as well use all the data to train on it (rather than just $1-1/k$ of the data that is used in the cross-validation slices).
The model that we obtain in this way can then be applied to the test set.
Saving the estimates in a CSV file will simplify uploading the results to Kaggle.
-->

Bây giờ, khi đã biết được các lựa chọn tốt cho siêu tham số, ta có thể sử dụng toàn bộ dữ liệu cho việc huấn luyện (thay vì chỉ dùng $1-1/k$ của dữ liệu như trong quá trình kiểm định chéo).
Sau đó, ta áp dụng mô hình thu được lên tập kiểm tra và lưu các dự đoán vào một tệp CSV nhằm đơn giản hóa quá trình tải kết quả lên Kaggle.

```{.python .input  n=18}
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='rmse', yscale='log')
    print('train rmse %f' % train_ls[-1])
    # Apply the network to the test set
    preds = net(test_features).asnumpy()
    # Reformat it for export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

<!--
Let's invoke our model.
One nice sanity check is to see whether the predictions on the test set resemble those of the k-fold cross-validation process.
If they do, it is time to upload them to Kaggle.
The following code will generate a file called `submission.csv` (CSV is one of the file formats accepted by Kaggle):
-->

Hãy gọi mô hình để đưa ra dự đoán. <!-- Bạn nào review sửa câu này giúp mình nhé, mình chưa biết dịch sao cho hợp lý hơn. Many thanks!-->
Ta nên kiểm tra xem liệu các dự đoán trên tập kiểm tra có tương đồng với các dự đoán trong quá trình kiểm định chéo k-phần hay không.
Nếu đúng là như vậy thì đã đến lúc tải các dự đoán này lên Kaggle.
Đoạn mã nguồn sau sẽ tạo một tệp có tên `submission.csv` (CSV là một trong những định dạng tệp được chấp nhận bởi Kaggle):

```{.python .input  n=19}
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

<!--
Next, as demonstrated in :numref:`fig_kaggle_submit2`, we can submit our predictions on Kaggle and see how they compare to the actual house prices (labels) on the test set.
The steps are quite simple:
-->

Tiếp theo, như được mô tả trong hình :numref:`fig_kaggle_submit2`, ta có thể nộp các dự đoán lên Kaggle và so sánh chúng với giá nhà thực tế (các nhãn) trên tập kiểm tra.
Các bước tiến hành khá là đơn giản:

<!--
* Log in to the Kaggle website and visit the House Price Prediction Competition page.
* Click the “Submit Predictions” or “Late Submission” button (as of this writing, the button is located on the right).
* Click the “Upload Submission File” button in the dashed box at the bottom of the page and select the prediction file you wish to upload.
* Click the “Make Submission” button at the bottom of the page to view your results.
-->

* Đăng nhập vào trang web Kaggle và tìm đến trang của cuộc thi Dự đoán giá nhà.
* Nhấn vào nút “Submit Predictions” hoặc “Late Submission” (nút này nằm ở phía bên phải tại thời điểm viết sách).
* Nhấn vào nút “Upload Submission File” trong khung có viền nét đứt và chọn tệp dự đoán bạn muốn tải lên. <!-- Cái khung này đâu có nằm ở cuối trang nhỉ? -->
* Nhấn vào nút “Make Submission” ở cuối trang để xem kết quả.

<!--
![Submitting data to Kaggle](../img/kaggle_submit2.png)
-->

![Tải dữ liệu lên Kaggle](../img/kaggle_submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
## Summary
-->

## Tóm tắt

<!--
* Real data often contains a mix of different data types and needs to be preprocessed.
* Rescaling real-valued data to zero mean and unit variance is a good default. So is replacing missing values with their mean.
* Transforming categorical variables into indicator variables allows us to treat them like vectors.
* We can use k-fold cross validation to select the model and adjust the hyper-parameters.
* Logarithms are useful for relative loss.
-->

* Dữ liệu trong thực tế thường chứa nhiều kiểu dữ liệu khác nhau và cần phải được tiền xử lý.
* Thay đổi kích thước dữ liệu có giá trị thực về trung bình bằng không và phương sai đơn vị là một phương án mặc định tốt. Tương tự với việc thay thế các giá trị bị thiếu với giá trị trung bình của chúng.
* Chuyển đổi các biến phân loại thành các biến chỉ dẫn cho phép chúng ta xử lý chúng như các vector.
* Ta có thể sử dụng kiểm định chéo k-phần để chọn ra mô hình và điều chỉnh siêu tham số.
* Hàm Logarit có hữu ích đối với mất mát tương đối.


<!--
## Exercises
-->

## Bài tập

<!--
1. Submit your predictions for this tutorial to Kaggle. How good are your predictions?
2. Can you improve your model by minimizing the log-price directly? What happens if you try to predict the log price rather than the price?
3. Is it always a good idea to replace missing values by their mean? Hint: can you construct a situation where the values are not missing at random?
4. Find a better representation to deal with missing values. Hint: what happens if you add an indicator variable?
5. Improve the score on Kaggle by tuning the hyperparameters through k-fold cross-validation.
6. Improve the score by improving the model (layers, regularization, dropout).
7. What happens if we do not standardize the continuous numerical features like we have done in this section?
-->

1. Nộp kết quả dự đoán của bạn từ bài hướng dẫn này cho Kaggle. Các dự đoán của bạn tốt đến đâu?
2. Bạn có thể cải thiện mô hình bằng cách giảm thiểu trực tiếp log giá nhà không? Điều gì sẽ xảy ra nếu bạn dự đoán log giá nhà thay vì giá thực?
3. Liệu việc thay thế các giá trị bị thiếu bằng trung bình của chúng luôn luôn tốt? Gợi ý: bạn có thể dựng lên một tình huống khi mà các giá trị không bị thiếu một cách ngẫu nhiên không?
4. Tìm cách biểu diễn tốt hơn để đối phó với các giá trị bị thiếu. Gợi ý: điều gì sẽ xảy ra nếu bạn thêm vào một biến chỉ dẫn?
5. Cải thiện điểm trên Kaggle bằng cách điều chỉnh các siêu tham số thông qua kiểm định chéo gập k-lần.
6. Cải thiện điểm bằng cách cải thiện mô hình (các tầng, điều chuẩn hóa, dropout).
7. Điều gì sẽ xảy ra nếu ta không chuẩn hóa đặc trưng số liên tục như ta đã làm trong phần này? 

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2346)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2346)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Lê Quang Nhật
* Phạm Minh Đức
* Đoàn Võ Duy Thanh
* Lê Khắc Hồng Phúc

<!-- Phần 2 -->
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh

<!-- Phần 3 -->
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc

<!-- Phần 4 -->
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc
* Đoàn Võ Duy Thanh

<!-- Phần 5 -->
* Phạm Minh Đức

<!-- Phần 6 -->
*

<!-- Phần 7 -->
* Nguyễn Duy Du

<!-- Phần 8 -->
* Trần Yến Thy
* Nguyễn Lê Quang Nhật
* Lê Khắc Hồng Phúc
* Phạm Minh Đức

<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE - BẮT ĐẦU =================================== -->

<!--
# Sentiment Analysis and the Dataset
-->

# Phân tích Cảm xúc và Dữ liệu
:label:`sec_sentiment`


<!--
Text classification is a common task in natural language processing, which transforms a sequence of text of indefinite length into a category of text.
It is similar to the image classification, the most frequently used application in this book, e.g., :numref:`sec_naive_bayes`.
The only difference is that, rather than an image, text classification's example is a text sentence.
-->

Phân loại văn bản là một tác vụ phổ biến trong xử lý ngôn ngữ tự nhiên, ánh xạ chuỗi văn bản có độ dài thay đổi thành một hạng mục tương ứng của văn bản đó.
Tác vụ này khá giống với phân loại ảnh, ứng dụng phổ biến nhất được giới thiệu trong cuốn sách này, ví dụ, :numref:`sec_naive_bayes`.  
Điểm khác biệt duy nhất đó là, mẫu đầu vào của tác vụ phân loại là một câu văn bản thay vì một bức ảnh.


<!--
This section will focus on loading data for one of the sub-questions in this field: 
using text sentiment classification to analyze the emotions of the text's author.
This problem is also called sentiment analysis and has a wide range of applications.
For example, we can analyze user reviews of products to obtain user satisfaction statistics, 
or analyze user sentiments about market conditions and use it to predict future trends.
-->

Phần này sẽ tập trung vào việc nạp dữ liệu cho một trong số những câu hỏi của bài toán này:
sử dụng tác vụ phân loại cảm xúc văn bản để phân tích cảm xúc của người viết.
Bài toán này cũng có thể gọi là phân tích cảm xúc và có rất nhiều ứng dụng.
Ví dụ, ta có thể phân tích đánh giá của khách hàng về sản phẩm để thu được thống kê về độ hài lòng của khách hàng, hoặc phân tích cảm xúc của khách hàng về điều kiện thị trường và sử dụng kết quả này để dự đoán xu hướng tương lai.


```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
npx.set_np()
```


<!--
## The Sentiment Analysis Dataset
-->

## Dữ liệu Phân tích Cảm xúc


<!--
We use Stanford's [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/) as the dataset for sentiment analysis.
This dataset is divided into two datasets for training and testing purposes, each containing 25,000 movie reviews downloaded from IMDb.
In each dataset, the number of comments labeled as "positive" and "negative" is equal.
-->

Ta sử dụng [tập dữ liệu lớn về đánh giá phim ảnh](https://ai.stanford.edu/~amaas/data/sentiment/) (_Large Movie Review Dataset_) của Stanford làm dữ liệu cho tác vụ phân tích cảm xúc.
Tập dữ liệu này được chia thành hai tập huấn luyện và kiểm tra, mỗi tập chứa 25,000 đánh giá phim tải về từ IMDb.
Trong mỗi tập dữ liệu, số lượng đánh giá có nhãn "tích cực" (*positive*) và "tiêu cực" (*negative*) là bằng nhau.

<!--
###  Reading the Dataset
-->

### Đọc Dữ liệu


<!--
We first download this dataset to the "../data" path and extract it to "../data/aclImdb".
-->

Đầu tiên, ta tải dữ liệu về thư mục "../data" và giải nén dữ liệu vào thư mục "../data/aclImdb".


```{.python .input  n=2}
#@save
d2l.DATA_HUB['aclImdb'] = (
    'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz',
    '01ada507287d82875905620988597833ad4e0903')

data_dir = d2l.download_extract('aclImdb', 'aclImdb')
```


<!--
Next, read the training and test datasets. 
Each example is a review and its corresponding label: 1 indicates "positive" and 0 indicates "negative".
-->

Tiếp theo, ta đọc dữ liệu huấn luyện và dữ liệu kiểm tra.
Mỗi mẫu là một đánh giá và nhãn tương ứng của nó: 1 được gán cho "tích cực" (_positive_), và 0 được gán cho "tiêu cực" (_negative_).


```{.python .input  n=3}
#@save
def read_imdb(data_dir, is_train):
    data, labels = [], []
    for label in ('pos', 'neg'):
        folder_name = os.path.join(data_dir, 'train' if is_train else 'test',
                                   label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb(data_dir, is_train=True)
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
### Tokenization and Vocabulary
-->

### *dịch tiêu đề trên*


<!--
We use a word as a token, and then create a dictionary based on the training dataset.
-->

*dịch đoạn phía trên*


```{.python .input  n=4}
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5, reserved_tokens=['<pad>'])

d2l.set_figsize()
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0, 1000, 50));
```


<!--
### Padding to the Same Length
-->

### *dịch tiêu đề trên*


<!--
Because the reviews have different lengths, so they cannot be directly combined into minibatches.
Here we fix the length of each comment to 500 by truncating or adding "&lt;unk&gt;" indices.
-->

*dịch đoạn phía trên*


```{.python .input  n=5}
num_steps = 500  # sequence length
train_features = np.array([d2l.truncate_pad(
    vocab[line], num_steps, vocab['<pad>']) for line in train_tokens])
train_features.shape
```


<!--
### Creating the Data Iterator
-->

### *dịch tiêu đề trên*


<!--
Now, we will create a data iterator.
Each iteration will return a minibatch of data.
-->

*dịch đoạn phía trên*


```{.python .input  n=6}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'# batches:', len(train_iter)
```


<!--
## Putting All Things Together
-->

## *dịch tiêu đề trên*


<!--
Last, we will save a function `load_data_imdb` into `d2l`, which returns the vocabulary and data iterators.
-->

*dịch đoạn phía trên*


```{.python .input  n=7}
#@save
def load_data_imdb(batch_size, num_steps=500):
    data_dir = d2l.download_extract('aclImdb', 'aclImdb')
    train_data = read_imdb(data_dir, True)
    test_data = read_imdb(data_dir, False)
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab.unk) for line in train_tokens])
    test_features = np.array([d2l.truncate_pad(
        vocab[line], num_steps, vocab.unk) for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size,
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Tóm tắt

<!--
* Text classification can classify a text sequence into a category.
* To classify a text sentiment, we load an IMDb dataset and tokenize its words. 
Then we pad the text sequence for short reviews and create a data iterator.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
Discover a different natural language dataset (such as [Amazon reviews](https://snap.stanford.edu/data/web-Amazon.html)) 
and build a similar data_loader function as `load_data_imdb`.
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->
<!-- ========================================= REVISE - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/391)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.
Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Văn Quang

<!-- Phần 2 -->
* 

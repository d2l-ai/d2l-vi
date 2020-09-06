<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Feature-Rich Recommender Systems
-->

# *dịch tiêu đề trên*


<!--
Interaction data is the most basic indication of users' preferences and interests.
It plays a critical role in former introduced models.
Yet, interaction data is usually extremely sparse and can be noisy at times.
To address this issue, we can integrate side information such as features of items, profiles of users, 
and even in which context that the interaction occurred into the recommendation model.
Utilizing these features are helpful in making recommendations in that these features can be 
an effective predictor of users interests especially when interaction data is lacking.
As such, it is essential for recommendation models also have the capability to deal 
with those features and give the model some content/context awareness.
To demonstrate this type of recommendation models, we introduce another task on click-through rate (CTR) 
for online advertisement recommendations :cite:`McMahan.Holt.Sculley.ea.2013` and present an anonymous advertising data.
Targeted advertisement services have attracted widespread attention and are often framed as recommendation engines.
Recommending advertisements that match users' personal taste and interest is important for click-through rate improvement.
-->

*dịch đoạn phía trên*


<!--
Digital marketers use online advertising to display advertisements to customers.
Click-through rate is a metric that measures the number of clicks advertisers receive on 
their ads per number of impressions and it is expressed as a percentage calculated with the formula:
-->

*dịch đoạn phía trên*


$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$


<!--
Click-through rate is an important signal that indicates the effectiveness of prediction algorithms.
Click-through rate prediction is a task of predicting the likelihood that something on a website will be clicked.
Models on CTR prediction can not only be employed in targeted advertising systems but also 
in general item (e.g., movies, news, products) recommender systems, email campaigns, and even search engines.
It is also closely related to user satisfaction, conversion rate, 
and can be helpful in setting campaign goals as it can help advertisers to set realistic expectations.
-->

*dịch đoạn phía trên*


```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```


<!--
## An Online Advertising Dataset
-->

## *dịch tiêu đề trên*


<!--
With the considerable advancements of Internet and mobile technology,
online advertising has become an important income resource and generates vast majority of revenue in the Internet industry.
It is important to display relevant advertisements or advertisements that pique users' interests so that casual visitors can be converted into paying customers.
The dataset we introduced is an online advertising dataset.
It consists of 34 fields, with the first column representing the target variable that indicates if an ad was clicked (1) or not (0).
All the other columns are categorical features.
The columns might represent the advertisement id, site or application id, device id, time, user profiles and so on.
The real semantics of the features are undisclosed due to anonymization and privacy concern.
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
The following code downloads the dataset from our server and saves it into the local data folder.
-->

Đoạn mã dưới đây tải về tập dữ liệu từ máy chủ của chúng tôi và lưu nó trên thư mục dữ liệu cục bộ.


```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```


<!--
There are a training set and a test set, consisting of 15000 and 3000 samples/lines, respectively.
-->

Tập dữ liệu gồm có tập huấn luyện và tập kiểm tra, lần lượt bao gồm 150000 và 3000 mẫu/dòng.


<!--
## Dataset Wrapper
-->

## Wrapper Tập dữ liệu


<!--
For the convenience of data loading, we implement a `CTRDataset` which loads the advertising dataset from the CSV file and can be used by `DataLoader`.
-->

Để thuận tiện trong việc nạp dữ liệu, ta lập trình lớp `CTRDataset` nạp vào tập dữ liệu quảng cáo từ tệp CSV và có thể được sử dụng bởi `DataLoader`.


```{.python .input  n=13}
#@save
class CTRDataset(gluon.data.Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None,
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0, 0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
                    instance.setdefault('x', []).append(values[i])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >=
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)}
                                for i, cnt in feat_mapper.items()}
            self.defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
        
    def __len__(self):
        return self.count
    
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1])
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```


<!--
The following example loads the training data and print out the first record.
-->

Ví dụ dưới đây nạp tập huấn luyện và in ra bản ghi đầu tiên.


```{.python .input  n=16}
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```


<!--
As can be seen, all the 34 fields are categorical features.
Each value represents the one-hot index of the corresponding entry.
The label $0$ means that it is not clicked.
This `CTRDataset` can also be used to load other datasets such as the Criteo display advertising challenge [Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
and the Avazu click-through rate prediction [Dataset](https://www.kaggle.com/c/avazu-ctr-prediction).  
-->

Như có thể thấy, toàn bộ 34 trường đều là đặc trưng hạng mục.
Mỗi giá trị nhãn biểu diễn chỉ số one-hot của bản ghi tương ứng.
Nhãn $0$ nghĩa là quảng cáo này chưa được nhấp vào.
Lớp `CTRDataset` này cũng có thể được sử dụng để nạp các tập dữ liệu khác như tập dữ liệu trong cuộc thi hiện thị quảng cáo [Criteo](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
và tập dữ liệu dự đoán tỉ lệ nhấp chuột [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction).


## Tóm tắt

<!--
* Click-through rate is an important metric that is used to measure the effectiveness of advertising systems and recommender systems.
* Click-through rate prediction is usually converted to a binary classification problem. The target is to predict whether an ad/item will be clicked or not based on given features.
-->

* Tỉ lệ nhấp chuột là một phép đo quan trọng được sử dụng để đo độ hiệu quả của hệ thống quảng cáo và hệ thống gợi ý.
* Dự đoán tỉ lệ nhấp chuột thường được chuyển đổi thành bài toán phân loại nhị phân. Mục tiêu của bài toán là dự đoán liệu một quảng cáo/sản phẩm có được nhấp vào hay không dựa vào các đặc trưng cho trước.


## Bài tập

<!--
Can you load the Criteo and Avazu dataset with the provided `CTRDataset`.
It is worth noting that the Criteo dataset consisting of real-valued features so you may have to revise the code a bit.
-->

Bạn có thể nạp vào tập dữ liệu Criteo và Avazu với lớp `CTRDataset` đã được cung cấp được không.
Chú ý rằng tập dữ liệu Criteo gồm có các đặc trưng mang giá trị thực nên bạn có lẽ sẽ phải kiểm tra lại đoạn mã một chút.


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/405)
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
* 

<!-- Phần 2 -->
* Đỗ Trường Giang

*Cập nhật lần cuối: 03/09/2020. (Cập nhật lần cuối từ nội dung gốc: 30/06/2020)*

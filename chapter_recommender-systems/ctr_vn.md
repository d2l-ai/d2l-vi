<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Feature-Rich Recommender Systems
-->

# Hệ thống Gợi ý Phong phú Đặc trưng


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

Dữ liệu tương tác là dấu hiệu cơ bản nhất để chỉ sở thích và sự chú ý của người dùng.
Nó đóng vai trò chủ chốt trong các mô hình đã được giới thiệu ở các phần trước.
Tuy vậy, dữ liệu tương tác thường vô cùng thưa thớt và đôi lúc có thể có nhiễu.
Để khắc phục vấn đề này, ta có thể tích hợp các thông tin phụ ví dụ như đặc trưng của sản phẩm, hồ sơ người dùng,
và thậm chí là bối cảnh diễn ra sự tương tác vào mô hình gợi ý.
Tận dụng các đặc trưng này có lợi trong việc đưa ra gợi ý mà trong đó các đặc trưng này có thể là
một nhân tố hiệu quả trong việc dự đoán sở thích người dùng, đặc biệt khi thiếu dữ liệu tương tác.
Do đó, việc các mô hình dự đoán có khả năng xử lý những đặc trưng này đóng một vai trò thiết yếu, khiến mô hình có thể nhận thức được một phần nào đó bối cảnh/nội dung.
Để mô tả loại mô hình gợi ý này, chúng tôi giới thiệu một tác vụ khác trên tỉ lệ nhấp chuột (*click-through rate - CTR*)
cho gợi ý quảng cáo trực tuyến :cite:`McMahan.Holt.Sculley.ea.2013` và hiển thị dữ liệu quảng cáo vô danh.
Dịch vụ quảng cáo có mục tiêu đã thu hút sự chú ý rộng rãi và thường được coi như một công cụ gợi ý.
Gợi ý quảng cáo phù hợp với thị yếu và sở thích cá nhân của người dùng là rất quan trọng trong việc cải thiện tỉ lệ nhấp chuột.


<!--
Digital marketers use online advertising to display advertisements to customers.
Click-through rate is a metric that measures the number of clicks advertisers receive on 
their ads per number of impressions and it is expressed as a percentage calculated with the formula:
-->

Các nhà tiếp thị số sử dụng quảng cáo trực tuyến để phát quảng cáo tới khách hàng.
Tỉ lệ nhấp chuột là một phép đo đo số lần nhấp chuột mà nhà quảng cáo nhận được
với quảng cáo của họ trên số lần hiển thị, và nó được biểu diễn dưới dạng phần trăm được tính theo công thức:


$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$


<!--
Click-through rate is an important signal that indicates the effectiveness of prediction algorithms.
Click-through rate prediction is a task of predicting the likelihood that something on a website will be clicked.
Models on CTR prediction can not only be employed in targeted advertising systems but also 
in general item (e.g., movies, news, products) recommender systems, email campaigns, and even search engines.
It is also closely related to user satisfaction, conversion rate, 
and can be helpful in setting campaign goals as it can help advertisers to set realistic expectations.
-->

Tỉ lệ nhấp chuột là một dấu hiệu quan trọng cho thấy độ hiệu quả của thuật toán dự đoán.
Dự đoán tỉ lệ nhấp chuột là tác vụ dự đoán tỉ lệ mà một cái gì đó trên trang mạng được nhấp vào.
Mô hình dự đoán CTR không những có thể được áp dụng vào hệ thống quảng cáo có mục tiêu mà còn
trong hệ thống gợi ý sản phẩm nói chung (ví dụ như phim ảnh, tin tức, vật phẩm), chiến dịch quảng cáo qua thư điện tử, và thậm chí là công cụ tìm kiếm.
Nó cũng liên quan mật thiết đến độ hài lòng của khách hàng, tỉ lệ chuyển đổi,
và có thể giúp ích trong việc thiết lập mục tiêu của chiến dịch quảng cáo do nó có thể giúp nhà quảng cáo đặt ra kỳ vọng thực tế.


```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```


<!--
## An Online Advertising Dataset
-->

## Tập dữ liệu Quảng cáo Trực tuyến


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

Với những bước tiến đáng kể của Internet và công nghệ di động,
quảng cáo trực tuyến đã trở thành một nguồn thu nhập quan trọng và sản sinh phần lớn doanh thu trong ngành công nghiệp Internet.
Việc hiện thị quảng cáo thích hợp hoặc quảng cáo khêu gợi sự chú ý của người dùng là rất quan trọng để biến những khách thăm bình thường trở thành khách hàng trả tiền.
Tập dữ liệu chúng tôi giới thiệu là một tập dữ liệu quảng cáo trực tuyến.
Nó bao gồm 34 trường, với cột đầu tiên biểu diễn biến mục tiêu cho biết liệu một quảng cáo được nhấp vào (1) hay chưa (0).
Tất cả các cột còn lại là các hạng mục đặc trưng.
Các cột này có thể biểu diễn id của quảng cáo, id trang hay ứng dụng, id thiết bị, thời gian, hồ sơ người dùng và vân vân.
Ngữ nghĩa thực tế của các đặc trưng này không được tiết lộ để ẩn danh hoá dữ liệu và bảo mật thông tin cá nhân.


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
The following code downloads the dataset from our server and saves it into the local data folder.
-->

*dịch đoạn phía trên*


```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```


<!--
There are a training set and a test set, consisting of 15000 and 3000 samples/lines, respectively.
-->

*dịch đoạn phía trên*


<!--
## Dataset Wrapper
-->

## *dịch tiêu đề trên*


<!--
For the convenience of data loading, we implement a `CTRDataset` which loads the advertising dataset from the CSV file and can be used by `DataLoader`.
-->

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


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

*dịch đoạn phía trên*


## Tóm tắt

<!--
* Click-through rate is an important metric that is used to measure the effectiveness of advertising systems and recommender systems.
* Click-through rate prediction is usually converted to a binary classification problem. The target is to predict whether an ad/item will be clicked or not based on given features.
-->

*dịch đoạn phía trên*


## Bài tập

<!--
Can you load the Criteo and Avazu dataset with the provided `CTRDataset`.
It is worth noting that the Criteo dataset consisting of real-valued features so you may have to revise the code a bit.
-->

*dịch đoạn phía trên*


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
* Đỗ Trường Giang
* Phạm Hồng Vinh

<!-- Phần 2 -->
* 

*Cập nhật lần cuối: 03/09/2020. (Cập nhật lần cuối từ nội dung gốc: 30/06/2020)*

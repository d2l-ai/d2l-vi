# Hệ thống giới thiệu giàu tính năng

Dữ liệu tương tác là dấu hiệu cơ bản nhất về sở thích và sở thích của người dùng. Nó đóng một vai trò quan trọng trong các mô hình được giới thiệu trước đây. Tuy nhiên, dữ liệu tương tác thường cực kỳ thưa thớt và đôi khi có thể ồn ào. Để giải quyết vấn đề này, chúng ta có thể tích hợp thông tin phụ như các tính năng của các mục, hồ sơ của người dùng và thậm chí trong bối cảnh mà sự tương tác xảy ra vào mô hình đề xuất. Sử dụng các tính năng này rất hữu ích trong việc đưa ra các khuyến nghị ở chỗ các tính năng này có thể là một dự đoán hiệu quả của người dùng quan tâm đặc biệt là khi dữ liệu tương tác là thiếu. Do đó, điều cần thiết là các mô hình đề xuất cũng có khả năng đối phó với các tính năng đó và cung cấp cho mô hình một số nhận thức nội dung/bối cảnh. Để chứng minh loại mô hình đề xuất này, chúng tôi giới thiệu một nhiệm vụ khác về tỷ lệ nhấp (CTR) cho các khuyến nghị quảng cáo trực tuyến :cite:`McMahan.Holt.Sculley.ea.2013` và trình bày dữ liệu quảng cáo ẩn danh. Các dịch vụ quảng cáo được nhắm mục tiêu đã thu hút sự chú ý rộng rãi và thường được đóng khung như các công cụ đề xuất. Việc đề xuất quảng cáo phù hợp với sở thích cá nhân và sở thích của người dùng là rất quan trọng để cải thiện tỷ lệ nhấp chuột. 

Các nhà tiếp thị kỹ thuật số sử dụng quảng cáo trực tuyến để hiển thị quảng cáo cho khách hàng. Tỷ lệ nhấp là một số liệu đo lường số lần nhấp nhà quảng cáo nhận được trên quảng cáo của họ cho mỗi số lần hiển thị và nó được biểu thị dưới dạng phần trăm được tính theo công thức:  

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

Tỷ lệ nhấp chuột là một tín hiệu quan trọng chỉ ra hiệu quả của các thuật toán dự đoán. Dự đoán tỷ lệ nhấp chuột là một nhiệm vụ dự đoán khả năng một cái gì đó trên một trang web sẽ được nhấp. Các mô hình về dự đoán CTR không chỉ có thể được sử dụng trong các hệ thống quảng cáo được nhắm mục tiêu mà còn trong mục chung (ví dụ: phim, tin tức, sản phẩm) hệ thống giới thiệu, chiến dịch email và thậm chí cả công cụ tìm kiếm. Nó cũng liên quan chặt chẽ đến sự hài lòng của người dùng, tỷ lệ chuyển đổi và có thể hữu ích trong việc thiết lập các mục tiêu chiến dịch vì nó có thể giúp các nhà quảng cáo đặt ra kỳ vọng thực tế.

```{.python .input}
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import gluon, np
import os
```

## Một bộ dữ liệu quảng cáo trực tuyến

Với những tiến bộ đáng kể của Internet và công nghệ di động, quảng cáo trực tuyến đã trở thành một nguồn thu nhập quan trọng và tạo ra phần lớn doanh thu trong ngành công nghiệp Internet. Điều quan trọng là hiển thị các quảng cáo hoặc quảng cáo có liên quan rằng lợi ích của người dùng để khách truy cập thông thường có thể được chuyển đổi thành khách hàng trả tiền. Tập dữ liệu chúng tôi giới thiệu là một tập dữ liệu quảng cáo trực tuyến. Nó bao gồm 34 trường, với cột đầu tiên đại diện cho biến đích cho biết một quảng cáo đã được nhấp (1) hay không (0). Tất cả các cột khác là các tính năng phân loại. Các cột có thể đại diện cho id quảng cáo, trang web hoặc id ứng dụng, id thiết bị, thời gian, hồ sơ người dùng, v.v. Ngữ nghĩa thực sự của các tính năng không được tiết lộ do ẩn danh và mối quan tâm riêng tư. 

Mã sau tải tập dữ liệu từ máy chủ của chúng tôi và lưu nó vào thư mục dữ liệu cục bộ.

```{.python .input  n=15}
#@save
d2l.DATA_HUB['ctr'] = (d2l.DATA_URL + 'ctr.zip',
                       'e18327c48c8e8e5c23da714dd614e390d369843f')

data_dir = d2l.download_extract('ctr')
```

Có một bộ đào tạo và một bộ thử nghiệm, bao gồm 15000 và 3000 mẫu/dòng, tương ứng. 

## Gói dữ liệu

Để thuận tiện cho việc tải dữ liệu, chúng tôi triển khai `CTRDataset` tải tập dữ liệu quảng cáo từ tệp CSV và có thể được sử dụng bởi `DataLoader`.

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
            self.feat_mapper = {i: {feat_v: idx for idx, feat_v in enumerate(feat_values)}
                                for i, feat_values in feat_mapper.items()}
            self.defaults = {i: len(feat_values) for i, feat_values in feat_mapper.items()}
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

Ví dụ sau tải dữ liệu đào tạo và in ra bản ghi đầu tiên.

```{.python .input  n=16}
train_data = CTRDataset(os.path.join(data_dir, 'train.csv'))
train_data[0]
```

Như có thể thấy, tất cả 34 trường là các tính năng phân loại. Mỗi giá trị đại diện cho chỉ số một nóng của mục nhập tương ứng. Nhãn $0$ có nghĩa là nó không được nhấp. `CTRDataset` này cũng có thể được sử dụng để tải các bộ dữ liệu khác như thử thách quảng cáo hiển thị Criteo [Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) và dự đoán tỷ lệ nhấp qua Avazu [Dataset](https://www.kaggle.com/c/avazu-ctr-prediction).   

## Tóm tắt * Tỷ lệ nhấp chuột là một số liệu quan trọng được sử dụng để đo lường hiệu quả của hệ thống quảng cáo và hệ thống giới thiệu. Mục tiêu là dự đoán xem quảng cáo/mục sẽ được nhấp hay không dựa trên các tính năng đã cho. 

## Bài tập

* Bạn có thể tải tập dữ liệu Criteo và Avazu với `CTRDataset` được cung cấp. Điều đáng chú ý là tập dữ liệu Criteo bao gồm các tính năng có giá trị thực để bạn có thể phải sửa đổi mã một chút.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/405)
:end_tab:

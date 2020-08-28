<!--
# Dog Breed Identification (ImageNet Dogs) on Kaggle
-->

# Nhận diện Giống Chó (ImageNet Dogs) trên Kaggle


<!--
In this section, we will tackle the dog breed identification challenge in the Kaggle Competition. 
The competition's web address is
-->

Trong phần này, ta sẽ giải quyết thử thách nhận diện giống chó trong một cuộc thi trên Kaggle.
Cuộc thi có địa chỉ tại


> https://www.kaggle.com/c/dog-breed-identification


<!--
In this competition, we attempt to identify 120 different breeds of dogs.
The dataset used in this competition is actually a subset of the famous ImageNet dataset.
Different from the images in the CIFAR-10 dataset used in the previous section, the images in the ImageNet dataset are higher and wider and their dimensions are inconsistent.
-->

Trong cuộc thi này, ta cần nhận diện 120 giống chó khác nhau.
Tập dữ liệu trong cuộc thi này thực chất là một tập con của tập dữ liệu ImageNet nổi tiếng.
Khác với ảnh trong tập dữ liệu CIFAR-10 được sử dụng trong phần trước, các ảnh trong tập dữ liệu ImageNet có chiều dài và chiều rộng lớn hơn, đồng thời kích thước của chúng không nhất quán.


<!--
:numref:`fig_kaggle_dog` shows the information on the competition's webpage. 
In order to submit the results, please register an account on the Kaggle website first.
-->

:numref:`fig_kaggle_dog` mô tả thông tin trên trang web của cuộc thi.
Để có thể nộp kết quả, trước tiên vui lòng đăng kí tài khoảng trên Kaggle.


<!--
![Dog breed identification competition website. The dataset for the competition can be accessed by clicking the "Data" tab.](../img/kaggle-dog.jpg)
-->


![Trang web cuộc thi nhận diện giống chó. Tập dữ liệu cho cuộc thi này có thể được truy cập bằng cách nhấn vào thẻ "Data".](../img/kaggle-dog.jpg)
:width:`400px`
:label:`fig_kaggle_dog`



<!--
First, import the packages or modules required for the competition.
-->

Đầu tiên, ta nhập vào các gói thư viện hoặc các mô-đun cần cho cuộc thi.


```{.python .input}
import collections
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, init, npx
from mxnet.gluon import nn
import os
import time

npx.set_np()
```


<!--
## Obtaining and Organizing the Dataset
-->

## Tải xuống và Tổ chức Tập dữ liệu


<!--
The competition data is divided into a training set and testing set.
The training set contains $10,222$ images and the testing set contains $10,357$ images.
The images in both sets are in JPEG format.
These images contain three RGB channels (color) and they have different heights and widths.
There are 120 breeds of dogs in the training set, including Labradors, Poodles, Dachshunds, Samoyeds, Huskies, Chihuahuas, and Yorkshire Terriers.
-->

Dữ liệu cuộc thi được chia thành tập huấn luyện và tập kiểm tra.
Tập huấn luyện bao gồm $10,222$ ảnh và tập kiểm tra bao gồm $10,357$ ảnh.
Tất cả các ảnh trong hai tập đều có định dạng JPEG.
Các ảnh này gồm có ba kênh (màu) RGB và có chiều cao và chiều rộng khác nhau.
Có tất cả 120 giống chó trong tập huấn luyện, gồm có Chó tha mồi (*Labrador*), Chó săn vịt (*Poodle*), Chó Dachshund, Samoyed, Huskie, Chihuahua, và Chó sục Yorkshire (*Yorkshire Terriers*).


<!--
### Downloading the Dataset
-->

### Tải tập dữ liệu


<!--
After logging in to Kaggle, we can click on the "Data" tab on the dog breed identification competition webpage 
shown in :numref:`fig_kaggle_dog` and download the dataset by clicking the "Download All" button. 
After unzipping the downloaded file in `../data`, you will find the entire dataset in the following paths:
-->

Sau khi đăng nhập vào Kaggle, ta có thể chọn thẻ "Data" trong trang web cuộc thi nhận diện giống chó
như mô tả trong :numref:`fig_kaggle_dog` và tải tập dữ liệu về bằng cách nhấn vào nút "Download All".
Sau khi giải nén tệp đã tải về trong thư mục `../data`, bạn có thể tìm thấy toàn bộ tập dữ liệu theo các đường dẫn sau:


* ../data/dog-breed-identification/labels.csv
* ../data/dog-breed-identification/sample_submission.csv
* ../data/dog-breed-identification/train
* ../data/dog-breed-identification/test


<!--
You may have noticed that the above structure is quite similar to that of the CIFAR-10 competition in :numref:`sec_kaggle_cifar10`, 
where folders `train/` and `test/` contain training and testing dog images respectively, and `labels.csv` has the labels for the training images.
-->

Có thể bạn đã nhận ra rằng cấu trúc trên khá giống với cấu trúc thư mục của cuộc thi CIFAR-10 trong :numref:`sec_kaggle_cifar10`,
trong đó thư mục `train/` và `test/` lần lượt chứa ảnh chó để huấn luyện và kiểm tra, và `labels.csv` chứa nhãn cho các ảnh huấn luyện.


<!--
Similarly, to make it easier to get started, we provide a small-scale sample of the dataset mentioned above, "train_valid_test_tiny.zip". 
If you are going to use the full dataset for the Kaggle competition, you will also need to change the `demo` variable below to `False`.
-->

Tương tự, để đơn giản, chúng tôi cung cấp một tập mẫu nhỏ của tập dữ liệu kể trên, "train_valid_test_tiny.zip".
Nếu bạn sử dụng tập dữ liệu đầy đủ cho cuộc thi Kaggle, bạn cần thay đổi biến `demo` phía dưới thành `False`.


```{.python .input  n=1}
#@save 
d2l.DATA_HUB['dog_tiny'] = (d2l.DATA_URL + 'kaggle_dog_tiny.zip',
                            '75d1ec6b9b2616d2760f211f72a83f73f3b83763')

# If you use the full dataset downloaded for the Kaggle competition, change
# the variable below to False
demo = True
if demo:
    data_dir = d2l.download_extract('dog_tiny')
else:
    data_dir = os.path.join('..', 'data', 'dog-breed-identification')
```


<!--
### Organizing the Dataset
-->

### Tổ chức Tập dữ liệu


<!--
We can organize the dataset similarly to what we did in :numref:`sec_kaggle_cifar10`, namely separating a validation set from the training set, and moving images into subfolders grouped by labels.
-->

Ta có thể tổ chức tập dữ liệu tương tự như cách ta đã làm trong :numref:`sec_kaggle_cifar10`, 
tức là tách riêng một tập kiểm định từ tập huấn luyện, sau đó đưa các ảnh vào từng thư mục con theo nhãn của chúng.


<!--
The `reorg_dog_data` function below is used to read the training data labels, segment the validation set, and organize the training set.
-->

Hàm `reorg_dog_data` dưới đây được sử dụng để đọc nhãn của dữ liệu huấn luyện, tách riêng tập kiểm định và tổ chức tập huấn luyện.


```{.python .input  n=2}
def reorg_dog_data(data_dir, valid_ratio):
    labels = d2l.read_csv_labels(os.path.join(data_dir, 'labels.csv'))
    d2l.reorg_train_valid(data_dir, labels, valid_ratio)
    d2l.reorg_test(data_dir)

    
batch_size = 4 if demo else 128
valid_ratio = 0.1
reorg_dog_data(data_dir, valid_ratio)
```


<!--
## Image Augmentation
-->

## Tăng cường Ảnh



<!--
The size of the images in this section are larger than the images in the previous section.
Here are some more image augmentation operations that might be useful.
-->

Trong phần này, kích thước ảnh lớn hơn phần trước.
Dưới đây là một số kỹ thuật tăng cường ảnh có thể sẽ hữu dụng.


```{.python .input  n=4}
transform_train = gluon.data.vision.transforms.Compose([
    # Randomly crop the image to obtain an image with an area of 0.08 to 1 of
    # the original area and height to width ratio between 3/4 and 4/3. Then,
    # scale the image to create a new image with a height and width of 224
    # pixels each
    gluon.data.vision.transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                                   ratio=(3.0/4.0, 4.0/3.0)),
    gluon.data.vision.transforms.RandomFlipLeftRight(),
    # Randomly change the brightness, contrast, and saturation
    gluon.data.vision.transforms.RandomColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4),
    # Add random noise
    gluon.data.vision.transforms.RandomLighting(0.1),
    gluon.data.vision.transforms.ToTensor(),
    # Standardize each channel of the image
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```


<!--
During testing, we only use definite image preprocessing operations.
-->

Trong quá trình kiểm tra, ta chỉ sử dụng một số bước tiền xử lý ảnh nhất định.


```{.python .input}
transform_test = gluon.data.vision.transforms.Compose([
    gluon.data.vision.transforms.Resize(256),
    # Crop a square of 224 by 224 from the center of the image
    gluon.data.vision.transforms.CenterCrop(224),
    gluon.data.vision.transforms.ToTensor(),
    gluon.data.vision.transforms.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
```


<!--
## Reading the Dataset
-->

## Đọc Dữ liệu


<!--
As in the previous section, we can create an `ImageFolderDataset` instance to read the dataset containing the original image files.
-->

Như trong phần trước, ta có thể tạo thực thể `ImageFolderDataset` để đọc dữ liệu chứa các tệp ảnh gốc.


```{.python .input  n=5}
train_ds, valid_ds, train_valid_ds, test_ds = [
    gluon.data.vision.ImageFolderDataset(
        os.path.join(data_dir, 'train_valid_test', folder))
    for folder in ('train', 'valid', 'train_valid', 'test')]
```


<!--
Here, we create `DataLoader` instances, just like in :numref:`sec_kaggle_cifar10`.
-->

Ở đây, ta tạo các thực thể `DataLoader` giống như trong :numref:`sec_kaggle_cifar10`.


```{.python .input}
train_iter, train_valid_iter = [gluon.data.DataLoader(
    dataset.transform_first(transform_train), batch_size, shuffle=True, 
    last_batch='discard') for dataset in (train_ds, train_valid_ds)]

valid_iter = gluon.data.DataLoader(
    valid_ds.transform_first(transform_test), batch_size, shuffle=False, 
    last_batch='discard')

test_iter = gluon.data.DataLoader(
    test_ds.transform_first(transform_test), batch_size, shuffle=False, 
    last_batch='keep')
```


<!--
## Defining the Model
-->

## Định nghĩa Mô hình 


<!--
The dataset for this competition is a subset of the ImageNet data set.
Therefore, we can use the approach discussed in :numref:`sec_fine_tuning` to select a model pre-trained on the entire ImageNet dataset 
and use it to extract image features to be input in the custom small-scale output network.
Gluon provides a wide range of pre-trained models.
Here, we will use the pre-trained ResNet-34 model.
Because the competition dataset is a subset of the pre-training dataset, we simply reuse the input of the pre-trained model's output layer, i.e., the extracted features.
Then, we can replace the original output layer with a small custom output network that can be trained, such as two fully connected layers in a series.
Different from the experiment in :numref:`sec_fine_tuning`, here, we do not retrain the pre-trained model used for feature extraction.
This reduces the training time and the memory required to store model parameter gradients.
-->

Dữ liệu cho cuộc thi này là một phần của tập dữ liệu ImageNet.
Do đó, ta có thể sử dụng cách tiếp cận được thảo luận trong :numref:`sec_fine_tuning` để lựa chọn mô hình đã được tiền huấn luyện trên toàn bộ dữ liệu ImageNet 
và sử dụng nó để trích xuất đặc trưng ảnh làm đầu vào cho một mạng tùy biến cỡ nhỏ.
Gluon cung cấp một số mô hình đã được tiền huấn luyện.
Ở đây, ta sử dụng mô hình ResNet-34 đã được tiền huấn luyện.
Do dữ liệu của cuộc thi là tập con của tập dữ liệu tiền huấn luyện, ta đơn thuần sử dụng lại đầu vào của tầng đầu ra mô hình đã được tiền huấn luyện làm đặc trưng được được trích xuất.
Sau đó, ta có thể thay thế tầng đầu ra gốc bằng một mạng đầu ra tùy biến cỡ nhỏ để huấn luyện bao gồm hai tầng kết nối đầy đủ.
Khác với thí nghiệm trong :numref:`sec_fine_tuning`, ở đây ta không huấn luyện lại mô hình trích xuất đặc trưng đã được tiền huấn luyện.
Điều này giúp giảm thời gian huấn luyện và bộ nhớ cần thiết để lưu trữ gradient của tham số mô hình.


<!--
You must note that, during image augmentation, we use the mean values and standard deviations of the three RGB channels for the entire ImageNet dataset for normalization.
This is consistent with the normalization of the pre-trained model.
-->

Độc giả cần lưu ý, trong quá trình tăng cường ảnh, ta sử dụng giá trị trung bình và độ lệch chuẩn của ba kênh RGB lấy từ toàn bộ dữ liệu ImageNet để chuẩn hóa.
Điều này giúp dữ liệu nhất quán với việc chuẩn hóa của mô hình tiền huấn luyện.


```{.python .input  n=6}
def get_net(devices):
    finetune_net = gluon.model_zoo.vision.resnet34_v2(pretrained=True)
    # Define a new output network
    finetune_net.output_new = nn.HybridSequential(prefix='')
    finetune_net.output_new.add(nn.Dense(256, activation='relu'))
    # There are 120 output categories
    finetune_net.output_new.add(nn.Dense(120))
    # Initialize the output network
    finetune_net.output_new.initialize(init.Xavier(), ctx=devices)
    # Distribute the model parameters to the CPUs or GPUs used for computation
    finetune_net.collect_params().reset_ctx(devices)
    return finetune_net
```


<!--
When calculating the loss, we first use the member variable `features` to obtain the input of the pre-trained model's output layer, i.e., the extracted feature.
Then, we use this feature as the input for our small custom output network and compute the output.
-->

Khi tính toán mất mát, đầu tiên ta sử dụng biến thành viên `features` để lấy đầu vào của tầng đầu ra trong mô hình được tiền huấn luyện làm đặc trưng trích xuất.
Sau đó, ta sử dụng đặc trưng này làm đầu vào cho mạng đầu ra tùy biến cỡ nhỏ và tính toán đầu ra.


```{.python .input}
loss = gluon.loss.SoftmaxCrossEntropyLoss()

def evaluate_loss(data_iter, net, devices):
    l_sum, n = 0.0, 0
    for features, labels in data_iter:
        X_shards, y_shards = d2l.split_batch(features, labels, devices)
        output_features = [net.features(X_shard) for X_shard in X_shards]
        outputs = [net.output_new(feature) for feature in output_features]
        ls = [loss(output, y_shard).sum() for output, y_shard
              in zip(outputs, y_shards)]
        l_sum += sum([float(l.sum()) for l in ls])
        n += labels.size
    return l_sum / n
```


<!--
## Defining the Training Functions
-->

## Định nghĩa Hàm Huấn luyện


<!--
We will select the model and tune hyperparameters according to the model's performance on the validation set.
The model training function `train` only trains the small custom output network.
-->

Ta sẽ lựa chọn mô hình và điều chỉnh siêu tham số dựa trên chất lượng mô hình trên tập kiểm định.
Hàm huấn luyện mô hình `train` chỉ huấn luyện mạng đầu ra tùy biến cỡ nhỏ.


```{.python .input}
def train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
          lr_decay):
    # Only train the small custom output network
    trainer = gluon.Trainer(net.output_new.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0.9, 'wd': wd})
    num_batches, timer = len(train_iter), d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[0, num_epochs],
                            legend=['train loss', 'valid loss'])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        if epoch > 0 and epoch % lr_period == 0:
            trainer.set_learning_rate(trainer.learning_rate * lr_decay)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            X_shards, y_shards = d2l.split_batch(features, labels, devices)
            output_features = [net.features(X_shard) for X_shard in X_shards]
            with autograd.record():
                outputs = [net.output_new(feature) for feature in output_features]
                ls = [loss(output, y_shard).sum() for output, y_shard
                      in zip(outputs, y_shards)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            metric.add(sum([float(l.sum()) for l in ls]), labels.shape[0])
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0:
                animator.add(epoch + i / num_batches, 
                             (metric[0] / metric[1], None))
        if valid_iter is not None:
            valid_loss = evaluate_loss(valid_iter, net, devices)
            animator.add(epoch + 1, (None, valid_loss))
    if valid_iter is not None:
        print(f'train loss {metric[0] / metric[1]:.3f}, '
              f'valid loss {valid_loss:.3f}')
    else:
        print(f'train loss {metric[0] / metric[1]:.3f}')
    print(f'{metric[1] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```


<!--
## Training and Validating the Model
-->

## Huấn luyện và Kiểm định Mô hình


<!--
Now, we can train and validate the model. The following hyperparameters can be tuned.
For example, we can increase the number of epochs.
Because `lr_period` and `lr_decay` are set to 10 and 0.1 respectively, the learning rate of the optimization algorithm will be multiplied by 0.1 after every 10 epochs.
-->

Bây giờ, ta có thể huấn luyện và kiểm định mô hình. 
Các siêu tham số dưới đây có thể được điều chỉnh: `num_epochs`, `lr_period` và `lr_decay`.
Ví dụ, ta có thể tăng số lượng epoch.
Do `lr_period` và `lr_decay` được thiết lập bằng 10 và 0.1, tốc độ học của thuật toán tối ưu sẽ được nhân với 0.1 sau mỗi 10 epoch.


```{.python .input  n=9}
devices, num_epochs, lr, wd = d2l.try_all_gpus(), 5, 0.01, 1e-4
lr_period, lr_decay, net = 10, 0.1, get_net(devices)
net.hybridize()
train(net, train_iter, valid_iter, num_epochs, lr, wd, devices, lr_period,
      lr_decay)
```


<!--
## Classifying the Testing Set and Submitting Results on Kaggle
-->

## Dự đoán trên tập Kiểm tra và Nộp Kết quả lên Kaggle


<!--
After obtaining a satisfactory model design and hyperparameters, we use all training datasets (including validation sets) to retrain the model and then classify the testing set.
Note that predictions are made by the output network we just trained.
-->

Sau khi thu được một thiết kế mô hình và các siêu tham số vừa ý, ta sử dụng tất cả dữ liệu huấn luyện 
(bao gồm dữ liệu kiểm định) để huấn luyện lại mô hình, sau đó thực hiện dự đoán trên tập kiểm tra.
Chú ý rằng các dự đoán được lấy từ mạng đầu ra mà ta đã huấn luyện.


```{.python .input  n=8}
net = get_net(devices)
net.hybridize()
train(net, train_valid_iter, None, num_epochs, lr, wd, devices, lr_period,
      lr_decay)

preds = []
for data, label in test_iter:
    output_features = net.features(data.as_in_ctx(devices[0]))
    output = npx.softmax(net.output_new(output_features))
    preds.extend(output.asnumpy())
ids = sorted(os.listdir(
    os.path.join(data_dir, 'train_valid_test', 'test', 'unknown')))
with open('submission.csv', 'w') as f:
    f.write('id,' + ','.join(train_valid_ds.synsets) + '\n')
    for i, output in zip(ids, preds):
        f.write(i.split('.')[0] + ',' + ','.join(
            [str(num) for num in output]) + '\n')
```


<!--
After executing the above code, we will generate a "submission.csv" file.
The format of this file is consistent with the Kaggle competition requirements.
The method for submitting results is similar to method in :numref:`sec_kaggle_house`.
-->

Chạy đoạn mã trên sẽ sinh tệp "submission.csv".
Định dạng của tệp này nhất quán với yêu cầu của cuộc thi Kaggle.
Cách thức nộp kết quả tương tự như trong :numref:`sec_kaggle_house`.


## Tóm tắt

<!--
We can use a model pre-trained on the ImageNet dataset to extract features and only train a small custom output network.
This will allow us to classify a subset of the ImageNet dataset with lower computing and storage overhead.
-->

* Ta có thể sử dụng mô hình đã được tiền huấn luyện trên tập dữ liệu ImageNet để trích xuất đặc trưng và chỉ huấn luyện trên mạng đầu ra tùy biến cỡ nhỏ. 
* Điều này cho phép ta có thể thực hiện dự đoán trên tập con của tập dữ liệu ImageNet với chi phí bộ nhớ và tính toán thấp hơn.


## Bài tập

<!--
1. When using the entire Kaggle dataset, what kind of results do you get when you increase the `batch_size` (batch size) and `num_epochs` (number of epochs)?
2. Do you get better results if you use a deeper pre-trained model?
3. Scan the QR code to access the relevant discussions and exchange ideas about the methods used and the results obtained with the community. Can you come up with any better techniques?
-->

1. Khi sử dụng toàn bộ dữ liệu Kaggle, bạn sẽ thu được kết quả như thế nào khi tăng `batch_size` (kích thước batch) và `num_epochs` (số lượng epoch)?
2. Bạn có đạt được kết quả tốt hơn nếu sử dụng mô hình đã được tiền huấn luyện sâu hơn không?
3. Quét mã QR để tham gia thảo luận và trao đổi ý tưởng về các phương pháp đã được sử dụng và kết quả thu được từ cộng đồng Kaggle. Bạn có thể nghĩ ra một ý tưởng hay kỹ thuật tốt hơn không?


## Thảo luận
* [Tiếng Anh - MXNet](https://discuss.d2l.ai/t/380)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Đỗ Trường Giang
* Nguyễn Văn Quang
* Phạm Hồng Vinh
* Nguyễn Văn Cường

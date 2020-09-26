<!--
# Natural Language Inference and the Dataset
-->

# Suy luận ngôn ngữ tự nhiên và Tập dữ liệu
:label:`sec_natural-language-inference-and-dataset`


<!--
In :numref:`sec_sentiment`, we discussed the problem of sentiment analysis.
This task aims to classify a single text sequence into predefined categories, such as a set of sentiment polarities.
However, when there is a need to decide whether one sentence can be inferred form another, 
or eliminate redundancy by identifying sentences that are semantically equivalent, knowing how to classify one text sequence is insufficient.
Instead, we need to be able to reason over pairs of text sequences.
-->

Trong :numref:`sec_sentiment`, chúng ta đã thảo luận về bài toán phân tích sắc thái cảm xúc (*sentiment analysis*).
Mục đích của bài toán là phân loại một chuỗi văn bản vào các hạng mục đã định trước, chẳng hạn như các sắc thái đối lập.
Tuy nhiên, trong trường hợp cần xác định liệu một câu có thể suy ra được từ một câu khác không, 
hoặc khi cần loại bỏ sự dư thừa bằng việc xác định các câu tương đương về ngữ nghĩa thì việc phân lớp một chuỗi văn bản là không đủ. 
Thay vào đó ta cần khả năng suy luận trên các cặp chuỗi văn bản.


<!--
## Natural Language Inference
-->

## Suy luận Ngôn ngữ Tự nhiên

<!--
*Natural language inference* studies whether a *hypothesis* can be inferred from a *premise*, where both are a text sequence.
In other words, natural language inference determines the logical relationship between a pair of text sequences.
Such relationships usually fall into three types:
-->

*Suy luận ngôn ngữ tự nhiên* nghiên cứu liệu một *giả thuyết (hypothesis)* có thể được suy ra được từ một *tiền đề (premise)* không, cả hai đều là chuỗi văn bản. 
Nói cách khác, suy luận ngôn ngữ tự nhiên quyết định mối quan hệ logic giữa một cặp chuỗi văn bản. 
Các mối quan hệ đó thường rơi vào một trong ba loại sau đây:


<!--
* *Entailment*: the hypothesis can be inferred from the premise.
* *Contradiction*: the negation of the hypothesis can be inferred from the premise.
* *Neutral*: all the other cases.
-->

* *Kéo theo*: giả thuyết có thể suy ra được từ tiền đề.
* *Đối lập*: phủ định của giả thuyết có thể suy ra được từ tiền đề.
* *Trung tính*: tất cả các trường hợp khác.


<!--
Natural language inference is also known as the recognizing textual entailment task.
For example, the following pair will be labeled as *entailment* because "showing affection" 
in the hypothesis can be inferred from "hugging one another" in the premise.
-->

Suy luận ngôn ngữ tự nhiên còn được gọi là bài toán nhận dạng quan hệ kéo theo trong văn bản.
Ví dụ, cặp sau được gán nhãn là *kéo theo* bởi vì "thể hiện tình cảm" trong giả thuyết có thể
được suy ra từ "ôm nhau" trong tiền đề.


<!--
> Premise: Two women are hugging each other.
-->

> Tiền đề: Hai người đang ôm nhau.

<!--
> Hypothesis: Two women are showing affection.
-->

> Giả thuyết: Hai người đang thể hiện tình cảm.


<!--
The following is an example of *contradiction* as "running the coding example" indicates "not sleeping" rather than "sleeping".
-->

Sau đây là một ví dụ về *đối lập*, vì "chạy đoạn mã ví dụ" cho biết "không ngủ" chứ không phải "ngủ".


<!--
> Premise: A man is running the coding example from Dive into Deep Learning.
-->

> Tiền đề: Một bạn đang chạy đoạn mã ví dụ trong Đắm mình vào học sâu.

<!--
> Hypothesis: The man is sleeping.
-->

> Giả thuyết: Bạn đó đang ngủ.


<!--
The third example shows a *neutrality* relationship because neither "famous" nor "not famous" can be inferred from the fact that "are performing for us". 
-->

Ví dụ thứ ba cho thấy mối quan hệ *trung tính* vì cả "nổi tiếng" và "không nổi tiếng" đều không thể được suy ra từ thực tế là "đang biểu diễn cho chúng tôi".


<!--
> Premise: The musicians are performing for us.
-->

> Tiền đề: Các nhạc công đang biểu diễn cho chúng tôi.

<!--
> Hypothesis: The musicians are famous.
-->

> Giả thuyết: Các nhạc công rất nổi tiếng.


<!--
Natural language inference has been a central topic for understanding natural language.
It enjoys wide applications ranging from information retrieval to open-domain question answering.
To study this problem, we will begin by investigating a popular natural language inference benchmark dataset.
-->

Suy luận ngôn ngữ tự nhiên là một chủ đề trung tâm trong việc hiểu ngôn ngữ tự nhiên.
Nó có nhiều ứng dụng khác nhau, từ truy xuất thông tin đến hỏi đáp trong miền mở.
Để nghiên cứu bài toán này, chúng ta sẽ bắt đầu bằng việc tìm hiểu một tập dữ liệu đánh giá xếp hạng phổ biến trong suy luận ngôn ngữ tự nhiên.


<!--
## The Stanford Natural Language Inference (SNLI) Dataset
-->

## Tập dữ liệu Suy luận ngôn ngữ tự nhiên của Stanford (SNLI)


<!--
Stanford Natural Language Inference (SNLI) Corpus is a collection of over $500,000$ labeled English sentence pairs :cite:`Bowman.Angeli.Potts.ea.2015`.
We download and store the extracted SNLI dataset in the path `../data/snli_1.0`.
-->

Tập ngữ liệu ngôn ngữ tự nhiên của Stanford (SNLI) là một tập hợp gồm hơn $500,000$ cặp câu Tiếng Anh được gán nhãn :cite:`Bowman.Angeli.Potts.ea.2015`.
Ta tải xuống và giải nén tập dữ liệu SNLI trong đường dẫn `../data/snli_1.0`.


```{.python .input  n=28}
import collections
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
import os
import re
import zipfile

npx.set_np()

#@save
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```


<!--
### Reading the Dataset
-->

### Đọc tập Dữ liệu


<!--
The original SNLI dataset contains much richer information than what we really need in our experiments.
Thus, we define a function `read_snli` to only extract part of the dataset, then return lists of premises, hypotheses, and their labels.
-->

Tập dữ liệu SNLI gốc chứa thông tin phong phú hơn những gì thực sự cần cho thí nghiệm của chúng ta.
Vì thế, ta định nghĩa một hàm `read_snli` để trích xuất một phần của tập dữ liệu, rồi trả về các danh sách tiền đề, giả thuyết và nhãn của chúng.


```{.python .input  n=66}
#@save
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\\(', '', s) 
        s = re.sub('\\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = os.path.join(data_dir, 'snli_1.0_train.txt'
                             if is_train else 'snli_1.0_test.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```


<!--
Now let us print the first $3$ pairs of premise and hypothesis, 
as well as their labels ("0", "1", and "2" correspond to "entailment", "contradiction", and "neutral", respectively ).
-->

Bây giờ ta in $3$ cặp tiền đề và giả thuyết đầu tiên cũng như nhãn của chúng ("0", "1", và "2" tương ứng với "kéo theo", "đối lập", và "trung tính").


```{.python .input  n=70}
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```


<!--
The training set has about $550,000$ pairs, and the testing set has about $10,000$ pairs.
The following shows that the three labels "entailment", "contradiction", and "neutral" are balanced in 
both the training set and the testing set.
-->

Tập huấn luyện có khoảng $550,000$ cặp, và tập kiểm tra có khoảng $10,000$ cặp.
Đoạn mã dưới đây cho thấy rằng ba nhãn "kéo theo", "đối lập", và "trung tính" cân bằng trong cả hai tập huấn luyện và tập kiểm tra.

```{.python .input}
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```


<!--
### Defining a Class for Loading the Dataset
-->

### Định nghĩa Lớp để nạp Tập dữ liệu


<!--
Below we define a class for loading the SNLI dataset by inheriting from the `Dataset` class in Gluon.
The argument `num_steps` in the class constructor specifies the length of a text sequence so that each minibatch of sequences will have the same shape. 
In other words, tokens after the first `num_steps` ones in longer sequence are trimmed, 
while special tokens “&lt;pad&gt;” will be appended to shorter sequences until their length becomes `num_steps`.
By implementing the `__getitem__` function, we can arbitrarily access the premise, hypothesis, and label with the index `idx`.
-->

Dưới đây ta định nghĩa một lớp để nạp tập dữ liệu SNLI bằng cách kế thừa lớp `Dataset` trong Gluon.
Đối số `num_steps` trong phương thức khởi tạo chỉ định độ dài chuỗi văn bản, do đó mỗi minibatch sẽ có cùng kích thước.
Nói cách khác, các token phía sau `num_steps` token đầu tiên ở trong chuỗi dài hơn sẽ được loại bỏ, 
trong khi token đặc biệt “&lt;pad&gt;” sẽ được nối thêm vào các chuỗi ngắn hơn đến khi độ dài của chúng bằng `num_steps`.
Bằng cách lập trình hàm `__getitem__`, ta có thể truy cập vào các tiền đề, giả thuyết và nhãn bất kỳ với chỉ số `idx`.


```{.python .input  n=115}
#@save
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, num_steps, vocab=None):
        self.num_steps = num_steps
        all_premise_tokens = d2l.tokenize(dataset[0])
        all_hypothesis_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(all_premise_tokens + all_hypothesis_tokens,
                                   min_freq=5, reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self._pad(all_premise_tokens)
        self.hypotheses = self._pad(all_hypothesis_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def _pad(self, lines):
        return np.array([d2l.truncate_pad(
            self.vocab[line], self.num_steps, self.vocab['<pad>'])
                         for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```


<!--
### Putting All Things Together
-->

### Kết hợp tất cả lại


<!--
Now we can invoke the `read_snli` function and the `SNLIDataset` class to download the SNLI dataset and 
return `DataLoader` instances for both training and testing sets, together with the vocabulary of the training set.
It is noteworthy that we must use the vocabulary constructed from the training set as that of the testing set. 
As a result, any new token from the testing set will be unknown to the model trained on the training set.
-->

Bây giờ ta có thể gọi hàm `read_snli` và lớp `SNLIDataset` để tải xuống tập dữ liệu SNLI và trả về thực thể `DataLoader` cho cả hai tập huấn luyện và tập kiểm tra, 
cùng với bộ từ vựng của tập huấn luyện.
Lưu ý rằng ta phải sử dụng bộ từ vựng được xây dựng từ tập huấn luyện cho tập kiểm tra.
Như vậy, mô hình được huấn luyện trên tập huấn luyện sẽ không biết bất kỳ token mới nào từ tập kiểm tra nếu có.


```{.python .input  n=114}
#@save
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return data iterators and vocabulary."""
    num_workers = d2l.get_dataloader_workers()
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data, num_steps)
    test_set = SNLIDataset(test_data, num_steps, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True,
                                       num_workers=num_workers)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False,
                                      num_workers=num_workers)
    return train_iter, test_iter, train_set.vocab
```


<!--
Here we set the batch size to $128$ and sequence length to $50$,
and invoke the `load_data_snli` function to get the data iterators and vocabulary.
Then we print the vocabulary size.
-->

Ở đây ta đặt kích thước batch là $128$ và độ dài chuỗi là $50$,
và gọi hàm `load_data_snli` để lấy iterator dữ liệu và bộ từ vựng. 
Sau đó ta in kích thước của bộ từ vựng.


```{.python .input  n=111}
train_iter, test_iter, vocab = load_data_snli(128, 50)
len(vocab)
```


<!--
Now we print the shape of the first minibatch.
Contrary to sentiment analysis,
we have $2$ inputs `X[0]` and `X[1]` representing pairs of premises and hypotheses.
-->

Bây giờ ta in kích thước của minibatch đầu tiên.
Trái với phân tích sắc thái cảm xúc, 
ta có $2$ đầu vào `X[0]` và `X[1]` biểu diễn cặp tiền đề và giả thuyết. 


```{.python .input  n=113}
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Tóm tắt

<!--
* Natural language inference studies whether a hypothesis can be inferred from a premise, where both are a text sequence.
* In natural language inference, relationships between premises and hypotheses include entailment, contradiction, and neutral.
* Stanford Natural Language Inference (SNLI) Corpus is a popular benchmark dataset of natural language inference.
-->

* Suy luận ngôn ngữ tự nhiên nghiên cứu liệu một giả thuyết có thể được suy ra từ một tiền đề hay không, khi cả hai đều là chuỗi văn bản.
* Trong suy luận ngôn ngữ tự nhiên, mối quan hệ giữa tiền đề và giả thuyết bao gồm kéo theo, đối lập và trung tính.
* Bộ dữ liệu suy luận ngôn ngữ tự nhiên Stanford (SNLI) là một tập dữ liệu đánh giá xếp hạng phổ biến cho suy luận ngôn ngữ tự nhiên.


## Bài tập

<!--
1. Machine translation has long been evaluated based on superficial $n$-gram matching between an output translation and a ground-truth translation.
Can you design a measure for evaluating machine translation results by using natural language inference?
2. How can we change hyperparameters to reduce the vocabulary size? 
-->

1. Dịch máy từ lâu nay vẫn được đánh giá bằng sự trùng lặp bề ngoài giữa các $n$-gram của bản dịch đầu ra và bản dịch nhãn gốc.
Bạn có thể thiết kế một phép đo để đánh giá kết quả dịch máy bằng cách sử dụng suy luận ngôn ngữ tự nhiên không?
2. Thay đổi siêu tham số như thế nào để giảm kích thước bộ từ vựng?


## Thảo luận
* Tiếng Anh: [Main Forum](https://discuss.d2l.ai/t/394)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Thái Bình
* Lê Khắc Hồng Phúc
* Trần Yến Thy
* Phạm Minh Đức
* Nguyễn Văn Cường

*Lần cập nhật gần nhất: 26/09/2020. (Cập nhật lần cuối từ nội dung gốc: 30/06/2020)*

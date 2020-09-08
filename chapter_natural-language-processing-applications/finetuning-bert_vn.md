<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Fine-Tuning BERT for Sequence-Level and Token-Level Applications
-->

# *dịch tiêu đề phía trên*
:label:`sec_finetuning-bert`

<!--
In the previous sections of this chapter, we have designed different models for 
natural language processing applications, such as based on RNNs, CNNs, attention, and MLPs.
These models are helpful when there is space or time constraint, however, 
crafting a specific model for every natural language processing task is practically infeasible.
In :numref:`sec_bert`, we introduced a pretraining model, BERT, 
that requires minimal architecture changes for a wide range of natural language processing tasks.
One one hand, at the time of its proposal, BERT improved the state of the art on various natural language processing tasks.
On the other hand, as noted in :numref:`sec_bert-pretraining`, 
the two versions of the original BERT model come with 110 million and 340 million parameters.
Thus, when there are sufficient computational resources, 
we may consider fine-tuning BERT for downstream natural language processing applications.
-->

*dịch đoạn phía trên*


<!--
In the following, we generalize a subset of natural language processing applications as sequence-level and token-level.
On the sequence level, we introduce how to transform the BERT representation of the text input 
to the output label in single text classification and text pair classification or regression.
On the token level, we will briefly introduce new applications such as text tagging 
and question answering and shed light on how BERT can represent their inputs and get transformed into output labels.
During fine-tuning, the "minimal architecture changes" required by BERT across different applications are the extra fully-connected layers.
During supervised learning of a downstream application, parameters of the extra layers are 
learned from scratch while all the parameters in the pretrained BERT model are fine-tuned.
-->

*dịch đoạn phía trên*


<!--
## Single Text Classification
-->

## *dịch tiêu đề phía trên*


<!--
*Single text classification* takes a single text sequence as the input and outputs its classification result.
Besides sentiment analysis that we have studied in this chapter,
the Corpus of Linguistic Acceptability (CoLA) is also a dataset for single text classification,
judging whether a given sentence is grammatically acceptable or not :cite:`Warstadt.Singh.Bowman.2019`.
For instance, "I should study." is acceptable but "I should studying." is not.
-->

*dịch đoạn phía trên*


<!--
![Fine-tuning BERT for single text classification applications, such as sentiment analysis and testing linguistic acceptability. Suppose that the input single text has six tokens.](../img/bert-one-seq.svg)
-->

![*dịch mô tả phía trên*](../img/bert-one-seq.svg)
:label:`fig_bert-one-seq`

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
:numref:`sec_bert` describes the input representation of BERT.
The BERT input sequence unambiguously represents both single text and text pairs,
where the special classification token  “&lt;cls&gt;” is used for sequence classification and 
the special classification token  “&lt;sep&gt;” marks the end of single text or separates a pair of text.
As shown in :numref:`fig_bert-one-seq`, in single text classification applications,
the BERT representation of the special classification token  “&lt;cls&gt;” encodes the information of the entire input text sequence.
As the representation of the input single text, it will be fed into a small MLP consisting of fully-connected (dense) layers
to output the distribution of all the discrete label values.
-->

:numref:`sec_bert` mô tả biểu diễn đầu vào của BERT.
Chuỗi đầu vào BERT biểu diễn một cách rõ ràng cả văn bản đơn và cặp văn bản,
trong đó token đặc biệt “&lt;cls&gt;” được sử dung cho các tác vụ phân loại chuỗi, 
và token đặc biệt “&lt;sep&gt;” đánh dấu vị trị kết thúc của văn bản đơn hoặc vị trí phân tách cặp văn bản.
Như minh hoạ trong :numref:`fig_bert-one-seq`, biểu diễn BERT của token đặc biệt “&lt;cls&gt;” mã hoá thông tin của toàn bộ chuỗi văn bản đầu vào trong các tác vụ phân loại văn bản đơn.
Được sử dụng là biểu diễn của văn bản đơn đầu vào, vector này sẽ được truyền vào một mạng MLP nhỏ chứa các tầng kết nối đầy đủ để biến đổi thành phân phối của các giá trị nhãn rời rạc.


<!--
## Text Pair Classification or Regression
-->

## Tác vụ Phân loại hay Hồi quy Cặp Văn bản


<!--
We have also examined natural language inference in this chapter.
It belongs to *text pair classification*, a type of application classifying a pair of text.
-->

Ta cũng sẽ xem xét tác vụ suy luận ngôn ngữ tự nhiên trong chương này.
Tác vụ này nằm trong bài toán *phân loại cặp văn bản* (_text pair classification_), một ứng dụng phân loại cặp văn bản.


<!--
Taking a pair of text as the input but outputting a continuous value, *semantic textual similarity* is a popular *text pair regression* task.
This task measures semantic similarity of sentences.
For instance, in the Semantic Textual Similarity Benchmark dataset, the similarity score of a pair of sentences
is an ordinal scale ranging from 0 (no meaning overlap) to 5 (meaning equivalence) :cite:`Cer.Diab.Agirre.ea.2017`.
The goal is to predict these scores.
Examples from the Semantic Textual Similarity Benchmark dataset include (sentence 1, sentence 2, similarity score):
-->

Nhận một cặp văn bản làm đầu vào, và cho ra một giá trị liên tục, đo độ tương tự ngữ nghĩa của văn bản (_semantic textual similarity_) là một tác vụ *hồi quy cặp văn bản* (*text pair regression*) rất phổ biến.
Tác vụ này đo độ tương tự ngữ nghĩa của các câu đầu vào.
Ví dụ, trong tập dữ liệu đánh giá độ tương tự ngữ nghĩa của văn bản (_Semantic Textual Similarity Benchmark_), độ tương tự của một cặp câu nằm trong khoảng từ 0 (không trùng lặp ngữ nghĩa) tới 5 (tương tự ngữ nghĩa) :cite:`Cer.Diab.Agirre.ea.2017`.
Mục đích của tác vụ này là dự đoán giá trị của độ tương tự trên.
Các mẫu dữ liệu trong tập dữ liệu đánh giá độ tương tự ngữ nghĩa văn bản có dạng (câu thứ 1, câu thứ 2, độ tương tự):


<!--
* "A plane is taking off.", "An air plane is taking off.", 5.000;
* "A woman is eating something.", "A woman is eating meat.", 3.000;
* "A woman is dancing.", "A man is talking.", 0.000.
-->

* "A plane is taking off.", "An air plane is taking off.", 5.000;
* "A woman is eating something.", "A woman is eating meat.", 3.000;
* "A woman is dancing.", "A man is talking.", 0.000.


<!--
![Fine-tuning BERT for text pair classification or regression applications, such as natural language inference and semantic textual similarity. Suppose that the input text pair has two and three tokens.](../img/bert-two-seqs.svg)
-->

![Tinh chỉnh mô hình BERT cho các ứng dụng phân loại hoặc hồi quy cặp văn bản, ví dụ tác vụ suy luận ngôn ngữ tự nhiên và tác vụ đo độ tương tự ngữ nghĩa văn bản. Giả sử đầu cặp văn bản đầu vào có hai và ba token.](../img/bert-two-seqs.svg)
:label:`fig_bert-two-seqs`


<!--
Comparing with single text classification in :numref:`fig_bert-one-seq`,
fine-tuning BERT for text pair classification in :numref:`fig_bert-two-seqs` is different in the input representation.
For text pair regression tasks such as semantic textual similarity, trivial changes can be applied such as outputting a continuous label value
and using the mean squared loss: they are common for regression.
-->


So sánh với tác vụ phân loại văn bản đơn trong :numref:`fig_bert-one-seq`,
tinh chỉnh mô hình BERT cho bài toán phân loại cặp văn bản trong :numref:`fig_bert-two-seqs` có khác biệt trong biểu diễn đầu vào.
Đối với các tác vụ hồi quy cặp văn bản, chẳng hạn như đo độ tương tự ngữ nghĩa văn bản, một vài thay đổi nhỏ thường được dùng cho hồi quy có thể được áp dụng ở đây, ví dụ như xuất ra giá trị nhãn liên tục
và sử dụng trung bình mất mát bình phương.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Text Tagging
-->

## *dịch tiêu đề phía trên*


<!--
Now let us consider token-level tasks, such as *text tagging*, where each token is assigned a label.
Among text tagging tasks, *part-of-speech tagging* assigns each word a part-of-speech tag (e.g., adjective and determiner)
according to the role of the word in the sentence.
For example, according to the Penn Treebank II tag set,
the sentence "John Smith 's car is new" should be tagged as
"NNP (noun, proper singular) NNP POS (possessive ending) NN (noun, singular or mass) VB (verb, base form) JJ (adjective)".
-->

*dịch đoạn phía trên*


<!--
![Fine-tuning BERT for text tagging applications, such as part-of-speech tagging. Suppose that the input single text has six tokens.](../img/bert-tagging.svg)
-->

![*dịch mô tả phía trên*](../img/bert-tagging.svg)
:label:`fig_bert-tagging`


<!--
Fine-tuning BERT for text tagging applications is illustrated in :numref:`fig_bert-tagging`.
Comparing with :numref:`fig_bert-one-seq`, the only distinction lies in that
in text tagging, the BERT representation of *every token* of the input text
is fed into the same extra fully-connected layers to output the label of the token, such as a part-of-speech tag.
-->

*dịch đoạn phía trên*


<!--
## Question Answering
-->

## *dịch tiêu đề phía trên*


<!--
As another token-level application, *question answering* reflects capabilities of reading comprehension.
For example, the Stanford Question Answering Dataset (SQuAD v1.1)
consists of reading passages and questions, where the answer to every question
is just a segment of text (text span) from the passage that the question is about :cite:`Rajpurkar.Zhang.Lopyrev.ea.2016`.
To explain, consider a passage
"Some experts report that a mask's efficacy is inconclusive. However, mask makers insist that their products, such as N95 respirator masks, can guard against the virus."
and a question "Who say that N95 respirator masks can guard against the virus?".
The answer should be the text span "mask makers" in the passage.
Thus, the goal in SQuAD v1.1 is to predict the start and end of the text span in the passage given a pair of question and passage.
-->

*dịch đoạn phía trên*


<!--
![Fine-tuning BERT for question answering. Suppose that the input text pair has two and three tokens.](../img/bert-qa.svg)
-->

![*dịch mô tả phía trên*](../img/bert-qa.svg)
:label:`fig_bert-qa`

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
To fine-tune BERT for question answering, the question and passage are packed as
the first and second text sequence, respectively, in the input of BERT.
To predict the position of the start of the text span, the same additional fully-connected layer will transform
the BERT representation of any token from the passage of position $i$ into a scalar score $s_i$.
Such scores of all the passage tokens are further transformed by the softmax operation
into a probability distribution, so that each token position $i$ in the passage is assigned
a probability $p_i$ of being the start of the text span.
Predicting the end of the text span is the same as above, except that
parameters in its additional fully-connected layer are independent from those for predicting the start.
When predicting the end, any passage token of position $i$ is transformed by the same fully-connected layer into a scalar score $e_i$.
:numref:`fig_bert-qa` depicts fine-tuning BERT for question answering.
-->

*dịch đoạn phía trên*


<!--
For question answering, the supervised learning's training objective is as straightforward as
maximizing the log-likelihoods of the ground-truth start and end positions.
When predicting the span, we can compute the score $s_i + e_j$ for a valid span
from position $i$ to position $j$ ($i \leq j$), and output the span with the highest score.
-->

*dịch đoạn phía trên*


## Tóm tắt

<!--
* BERT requires minimal architecture changes (extra fully-connected layers) for sequence-level and token-level natural language processing applications, 
such as single text classification (e.g., sentiment analysis and testing linguistic acceptability), text pair classification or regression 
(e.g., natural language inference and semantic textual similarity), text tagging (e.g., part-of-speech tagging), and question answering.
* During supervised learning of a downstream application, parameters of the extra layers are learned from scratch 
while all the parameters in the pretrained BERT model are fine-tuned.
-->

*dịch đoạn phía trên*



## Bài tập

<!--
1. Let us design a search engine algorithm for news articles. When the system receives an query (e.g., "oil industry during the coronavirus outbreak"), 
it should return a ranked list of news articles that are most relevant to the query. 
Suppose that we have a huge pool of news articles and a large number of queries. 
To simplify the problem, suppose that the most relevant article has been labeled for each query. 
How can we apply negative sampling (see :numref:`subsec_negative-sampling`) and BERT in the algorithm design?
2. How can we leverage BERT in training language models?
3. Can we leverage BERT in machine translation?
-->

*dịch đoạn phía trên*


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->
<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh](https://discuss.d2l.ai/t/396)
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
* Nguyễn Văn Quang

<!-- Phần 3 -->
* 

<!-- Phần 4 -->
* 

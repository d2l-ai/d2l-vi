<!--
# Convolutional Neural Networks
-->

# Mạng Nơ-ron Tích chập
:label:`chap_cnn`

<!--
In earlier chapters, we came up against image data, for which each example consists of a 2D grid of pixels.
Depending on whether we're handling black-and-white or color images, each pixel location might be associated with either *one* or *multiple* numerical values, respectively.
Until now, our way of dealing with this rich structure was deeply unsatisfying.
We simply discarded each image's spatial structure by flattening them into 1D vectors, feeding them through a (fully connected) MLP.
Because these networks invariant to the order of the features we could get similar results regardless of whether 
we preserve an order corresponding ot the spatial structure of the pixels or if we permute the columns of our design matrix before fitting the MLP's parameters.
Preferably, we would leverage our prior knowledge that nearby pixels are typically related to each other, to build efficient models for learning from image data.
-->

Trong những chương đầu tiên, chúng ta đã làm việc trên dữ liệu ảnh với mỗi mẫu là một mảng điểm ảnh 2D.
Tùy vào ảnh đen trắng hay ảnh màu mà ta cần xử lý *một* hay *nhiều* giá trị số học tương ứng tại mỗi vị trí điểm ảnh. 
Cho đến nay, cách ta xử lý dữ liệu với cấu trúc phong phú này vẫn chưa thật sự thoả đáng.
Ta chỉ đang đơn thuần loại bỏ cấu trúc không gian từ mỗi bức ảnh bằng cách chuyển chúng thành các vector và truyền chúng qua một mạng MLP (kết nối đầy đủ).
Vì các mạng này là bất biến với thứ tự của các đặc trưng, ta sẽ nhận được cùng một kết quả bất kể việc chúng ta có giữ lại thứ tự cấu trúc không gian của các điểm ảnh hay hoán vị các cột của ma trận đặc trưng trước khi khớp các tham số của mạng MLP. 
Tốt hơn hết, ta nên tận dụng điều đã biết là các điểm ảnh kề cận thường có tương quan lẫn nhau, để xây dựng những mô hình hiệu quả hơn cho việc học từ dữ liệu ảnh.

<!--
This chapter introduces convolutional neural networks (CNNs), a powerful family of neural networks that were designed for precisely this purpose.
CNN-based architectures are now ubiquitous in the field of computer vision, and have become so dominant that hardly anyone 
today would develop a commercial application or enter a competition related to image recognition, object detection, or semantic segmentation, without building off of this approach.
-->

Chương này sẽ giới thiệu về các Mạng Nơ-ron Tích chập (*Convolutional Neural Network* - CNN), một họ các mạng nơ-ron ưu việt được thiết kế chính xác cho mục đích trên. 
Các kiến trúc dựa trên CNN hiện nay xuất hiện trong mọi ngóc ngách của lĩnh vực thị giác máy tính, và đã trở thành kiến trúc chủ đạo mà hiếm ai ngày nay phát triển các ứng dụng thương mại hay tham gia một cuộc thi nào đó liên quan tới nhận dạng ảnh, phát hiện đối tượng, hay phân vùng theo ngữ cảnh mà không xây nền móng dựa trên phương pháp này.

<!--
Modern *ConvNets*, as they are called colloquially owe their design to inspirations from biology, group theory, and a healthy dose of experimental tinkering.
In addition to their sample efficiency in achieving accurate models, convolutional neural networks tend to be computationally efficient, 
both because they require fewer parameters than dense architectures and because convolutions are easy to parallelize across GPU cores.
Consequently, practitioners often apply CNNs whenever possible, and increasingly they have emerged as credible competitors even on tasks with 1D sequence structure, 
such as audio, text, and time series analysis, where recurrent neural networks are conventionally used.
Some clever adaptations of CNNs have also brought them to bear on graph-structured data and in recommender systems.
-->

Theo cách hiểu thông dụng, thiết kế của mạng *ConvNets* đã vay mượn rất nhiều ý tưởng từ ngành sinh học, lý thuyết nhóm và lượng rất nhiều những thí nghiệm nhỏ lẻ khác.
Bên cạnh hiệu năng cao trên số lượng mẫu cần thiết để đạt được đủ độ chính xác, các mạng nơ-ron tích chập thường có hiệu quả tính toán hơn, bởi đòi hỏi ít tham số hơn và dễ thực thi song song trên nhiều GPU hơn các kiến trúc mạng dày đặc.  
Do đó, các mạng CNN sẽ được áp dụng bất cứ khi nào có thể, và chúng đã nhanh chóng trở thành một công cụ quan trọng đáng tin cậy thậm chí với các tác vụ liên quan tới cấu trúc tuần tự một chiều, 
như là xử lý âm thanh, văn bản, và phân tích dữ liệu chuỗi thời gian (*time series analysis*), mà ở đó các mạng nơ-rơn hồi tiếp vốn thường được sử dụng. 
Với một số điều chỉnh khôn khéo, ta còn có thể dùng mạng CNN cho dữ liệu có cấu trúc đồ thị và hệ thống đề xuất. 

<!--
First, we will walk through the basic operations that comprise the backbone of all convolutional networks.
These include the convolutional layers themselves, nitty-gritty details including padding and stride, 
the pooling layers used to aggregate information across adjacent spatial regions, the use of multiple *channels* (also called *filters*) at each layer, 
and a careful discussion of the structure of modern architectures.
We will conclude the chapter with a full working example of LeNet, the first convolutional network successfully deployed, long before the rise of modern deep learning.
In the next chapter, we will dive into full implementations of some popular and 
comparatively recent CNN architectures whose designs representat most of the techniques commonly used by modern practitioners.
-->

Trước hết, chúng ta sẽ đi qua các phép toán cơ bản nhằm tạo nên bộ khung sườn của tất cả các mạng nơ-ron tích chập.
Chúng bao gồm các tầng tích chập, các chi tiết cơ bản quan trọng như đệm và sải bước, các tầng gộp dùng để kết hợp thông tin qua các vùng không gian kề nhau, việc sử dụng đa kênh (cũng được gọi là *các bộ lọc*) ở mỗi tầng và một cuộc thảo luận cẩn thận về cấu trúc của các mạng hiện đại. 
Chúng ta sẽ kết thúc cho chương này với một ví dụ hoàn toàn hoạt động của mạng LeNet, mạng tích chập đầu tiên đã triển khai thành công và tồn tại nhiều năm trước khi có sự trỗi dậy của kỹ thuật học sâu hiện đại.
Ở chương kế tiếp, chúng ta sẽ đắm mình vào việc xây dựng hoàn chỉnh một số kiến trúc CNN tương đối gần đây và khá phổ biến. 
Thiết kế của chúng chứa hầu hết những kỹ thuật mà ngày nay hay được sử dụng. 


```toc
:maxdepth: 2

why-conv_vn
conv-layer_vn
padding-and-strides_vn
channels_vn
pooling_vn
lenet_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Mai Hoàng Long
* Lê Khắc Hồng Phúc

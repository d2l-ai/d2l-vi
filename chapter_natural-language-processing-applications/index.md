# Xử lý ngôn ngữ tự nhiên: Ứng dụng
:label:`chap_nlp_app`

Chúng tôi đã thấy cách đại diện cho thẻ trong chuỗi văn bản và đào tạo các đại diện của chúng trong :numref:`chap_nlp_pretrain`. Các đại diện văn bản được đào tạo trước như vậy có thể được đưa vào các mô hình khác nhau cho các nhiệm vụ xử lý ngôn ngữ tự nhiên hạ nguồn khác nhau. 

Trên thực tế, các chương trước đó đã thảo luận về một số ứng dụng xử lý ngôn ngữ tự nhiên
*mà không cần pretraining*,
just for explaining giải thích deepsâu learning học tập architectures kiến trúc. Ví dụ, trong :numref:`chap_rnn`, chúng tôi đã dựa vào RNN để thiết kế các mô hình ngôn ngữ để tạo ra văn bản giống như tiểu thuyết. Trong :numref:`chap_modern_rnn` và :numref:`chap_attention`, chúng tôi cũng đã thiết kế các mô hình dựa trên RNNs và cơ chế chú ý cho dịch máy. 

Tuy nhiên, cuốn sách này không có ý định bao gồm tất cả các ứng dụng như vậy một cách toàn diện. Thay vào đó, trọng tâm của chúng tôi là * làm thế nào để áp dụng (sâu) đại diện học ngôn ngữ để giải quyết các vấn đề xử lý ngôn ngữ tự nhiên*. Với các đại diện văn bản được đào tạo trước, chương này sẽ khám phá hai nhiệm vụ xử lý ngôn ngữ tự nhiên ở hạ nguồn phổ biến và đại diện: phân tích tình cảm và suy luận ngôn ngữ tự nhiên, phân tích văn bản đơn lẻ và mối quan hệ của các cặp văn bản, tương ứng. 

![Pretrained text representations can be fed to various deep learning architectures for different downstream natural language processing applications. This chapter focuses on how to design models for different downstream natural language processing applications.](../img/nlp-map-app.svg)
:label:`fig_nlp-map-app`

Như được mô tả trong :numref:`fig_nlp-map-app`, chương này tập trung vào việc mô tả các ý tưởng cơ bản của việc thiết kế các mô hình xử lý ngôn ngữ tự nhiên sử dụng các loại kiến trúc học sâu khác nhau, chẳng hạn như MLP, CNN, RNN s, và sự chú ý. Mặc dù có thể kết hợp bất kỳ biểu diễn văn bản được đào tạo trước với bất kỳ kiến trúc nào cho một trong hai ứng dụng trong :numref:`fig_nlp-map-app`, chúng tôi chọn một vài kết hợp đại diện. Cụ thể, chúng tôi sẽ khám phá các kiến trúc phổ biến dựa trên RNNs và CNN để phân tích tình cảm. Đối với suy luận ngôn ngữ tự nhiên, chúng tôi chọn sự chú ý và MLP để chứng minh cách phân tích các cặp văn bản. Cuối cùng, chúng tôi giới thiệu cách tinh chỉnh mô hình BERT được đào tạo trước cho một loạt các ứng dụng xử lý ngôn ngữ tự nhiên, chẳng hạn như ở cấp độ trình tự (phân loại văn bản đơn và phân loại cặp văn bản) và mức mã thông báo (gắn thẻ văn bản và trả lời câu hỏi). Là một trường hợp thực nghiệm cụ thể, chúng tôi sẽ tinh chỉnh BERT cho suy luận ngôn ngữ tự nhiên. 

Như chúng tôi đã giới thiệu trong :numref:`sec_bert`, BERT đòi hỏi những thay đổi kiến trúc tối thiểu cho một loạt các ứng dụng xử lý ngôn ngữ tự nhiên. Tuy nhiên, lợi ích này đi kèm với chi phí tinh chỉnh một số lượng lớn các thông số BERT cho các ứng dụng hạ lưu. Khi không gian hoặc thời gian bị hạn chế, những mô hình được chế tạo dựa trên MLP, CNN, RNN và sự chú ý sẽ khả thi hơn. Sau đây, chúng ta bắt đầu bằng ứng dụng phân tích tình cảm và minh họa thiết kế mô hình dựa trên RNN và CNN, tương ứng.

```toc
:maxdepth: 2

sentiment-analysis-and-dataset
sentiment-analysis-rnn
sentiment-analysis-cnn
natural-language-inference-and-dataset
natural-language-inference-attention
finetuning-bert
natural-language-inference-bert
```

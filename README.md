# Dự án dịch sách ["Dive into Deep Learning"](https://www.d2l.ai/)

Cuốn sách này được dịch và đăng tại https://d2l.aivivn.com/.

## Hướng dẫn đóng góp vào dự án
* [Hướng dẫn dịch trực tiếp trên trình duyệt](CONTRIBUTING.md).
* [Hướng dẫn đóng góp chung](https://d2l.aivivn.com/intro_vn.html#huong-dan-dong-gop).

## Tham gia vào Slack của nhóm Dịch thuật
Đăng ký tham gia **[tại đây](https://docs.google.com/forms/d/e/1FAIpQLScYforPRBn0oDhqSV_zTpzkxCAf0F7Cke13QS2tqXrJ8LxisQ/viewform)** để trao đổi và hỏi đáp về các vấn đề liên quan.

## Bảng thuật ngữ
Tra cứu các thuật ngữ được sử dụng trong nhóm dịch tại **[glossary.md](https://github.com/aivivn/d2l-vn/blob/master/glossary.md)**.

## Thứ tự dịch

Với các mục con (2.1, 2.2, ...)
* [x] Đã dịch xong
* [-] Đang dịch 
* [ ] Chưa bắt đầu

Với các chương (2., 3., ...)
* [ ] Chưa revise
* [-] Đang revise
* [x] Đã revise xong.

### Mục lục
* [x] [Lời nói đầu](chapter_preface/index_vn.md)
* [x] [Cài đặt](chapter_install/index_vn.md)
* [x] [Ký hiệu](chapter_notation/index_vn.md)
* [x] [Giới thiệu](chapter_introduction/index_vn.md)
* [x] 2. [Sơ bộ](chapter_preliminaries/index_vn.md)
    * [x] 2.1. [Thao tác với Dữ liệu](chapter_preliminaries/ndarray_vn.md)
    * [x] 2.2. [Tiền Xử lý Dữ liệu](chapter_preliminaries/pandas_vn.md)
    * [x] 2.3. [Đại số Tuyến tính](chapter_preliminaries/linear-algebra_vn.md)
    * [x] 2.4. [Giải tích](chapter_preliminaries/calculus_vn.md)
    * [x] 2.5. [Tính vi phân Tự động](chapter_preliminaries/autograd_vn.md)
    * [x] 2.6. [Xác suất](chapter_preliminaries/probability_vn.md)
    * [x] 2.7. [Tài liệu](chapter_preliminaries/lookup-api_vn.md)
* [x] 3. [Mạng nơ-ron Tuyến tính](chapter_linear-networks/index_vn.md)
    * [x] 3.1. [Hồi quy Tuyến tính](chapter_linear-networks/linear-regression_vn.md)
    * [x] 3.2. [Lập trình Hồi quy Tuyến tính từ đầu](chapter_linear-networks/linear-regression-scratch_vn.md)
    * [x] 3.3. [Cách lập trình súc tích Hồi quy Tuyến tính](chapter_linear-networks/linear-regression-gluon_vn.md)
    * [x] 3.4. [Hồi quy Softmax](chapter_linear-networks/softmax-regression_vn.md)
    * [x] 3.5. [Bộ dữ liệu Phân loại Ảnh (Fashion-MNIST)](chapter_linear-networks/fashion-mnist_vn.md)
    * [x] 3.6. [Lập trình Hồi quy Sofmax từ đầu](chapter_linear-networks/softmax-regression-scratch_vn.md)
    * [x] 3.7. [Cách lập trình súc tích Hồi quy Softmax](chapter_linear-networks/softmax-regression-gluon_vn.md)
* [x] 4. [Perceptron Đa tầng](chapter_multilayer-perceptrons/index_vn.md)
    * [x] 4.1. [Perceptron Đa tầng](chapter_multilayer-perceptrons/mlp_vn.md)
    * [x] 4.2. [Lập trình Perceptron Đa tầng từ đầu](chapter_multilayer-perceptrons/mlp-scratch_vn.md)
    * [x] 4.3. [Cách lập trình súc tích Perceptron Đa tầng](chapter_multilayer-perceptrons/mlp-gluon_vn.md)
    * [x] 4.4. [Lựa chọn Mô hình, Dưới khớp và Quá khớp](chapter_multilayer-perceptrons/underfit-overfit_vn.md)
    * [x] 4.5. [Suy giảm Trọng số](chapter_multilayer-perceptrons/weight-decay_vn.md)
    * [x] 4.6. [Dropout](chapter_multilayer-perceptrons/dropout_vn.md)
    * [x] 4.7. [Lan truyền Xuôi, Lan truyền Ngược và Đồ thị Tính toán](chapter_multilayer-perceptrons/backprop_vn.md)
    * [x] 4.8. [Sự ổn định Số học và Sự khởi tạo](chapter_multilayer-perceptrons/numerical-stability-and-init_vn.md)
    * [x] 4.9. [Cân nhắc tới Môi trường](chapter_multilayer-perceptrons/environment_vn.md)
    * [x] 4.10. [Dự đoán Giá Nhà trên Kaggle](chapter_multilayer-perceptrons/kaggle-house-price_vn.md)
* [ ] 5. [Tính toán Học sâu](chapter_deep-learning-computation/index_vn.md)
    * [-] 5.1. [Tầng và Khối](chapter_deep-learning-computation/model-construction_vn.md)
    * [-] 5.2. [Quản lý Tham số](chapter_deep-learning-computation/parameters_vn.md)
    * [-] 5.3. [Khởi tạo trễ](chapter_deep-learning-computation/deferred-init_vn.md)
    * [-] 5.4. [Custom Layers](chapter_deep-learning-computation/custom-layer_vn.md)
    * [-] 5.5. [File I/O](chapter_deep-learning-computation/read-write_vn.md)
    * [-] 5.6. [GPUs](chapter_deep-learning-computation/use-gpu_vn.md)
* [ ] 6. [Mạng nơ-ron Tích chập](chapter_convolutional-neural-networks/index_vn.md)
    * [-] 6.1. [From Dense Layers to Convolutions](chapter_convolutional-neural-networks/why-conv_vn.md)
    * [-] 6.2. [Convolutions for Images](chapter_convolutional-neural-networks/conv-layer_vn.md)
    * [-] 6.3. [Padding and Stride](chapter_convolutional-neural-networks/padding-and-strides_vn.md)
    * [-] 6.4. [Multiple Input and Output Channels](chapter_convolutional-neural-networks/channels_vn.md)
    * [ ] 6.5. [Pooling](chapter_convolutional-neural-networks/pooling_vn.md)
    * [ ] 6.6. [Convolutional Neural Networks (LeNet)](chapter_convolutional-neural-networks/lenet_vn.md)
* [ ] 7. Modern Convolutional Neural Networks
    * [ ] 7.1. Deep Convolutional Neural Networks (AlexNet)
    * [ ] 7.2. Networks Using Blocks (VGG)
    * [ ] 7.3. Network in Network (NiN)
    * [ ] 7.4. Networks with Parallel Concatenations (GoogLeNet)
    * [ ] 7.5. Batch Normalization
    * [ ] 7.6. Residual Networks (ResNet)
    * [ ] 7.7. Densely Connected Networks (DenseNet)
* [ ] 8. Recurrent Neural Networks
    * [ ] 8.1. Sequence Models
    * [ ] 8.2. Text Preprocessing
    * [ ] 8.3. Language Models and the Dataset
    * [ ] 8.4. Recurrent Neural Networks
    * [ ] 8.5. Implementation of Recurrent Neural Networks from Scratch
    * [ ] 8.6. Concise Implementation of Recurrent Neural Networks
    * [ ] 8.7. Backpropagation Through Time
* [ ] 9. Modern Recurrent Neural Networks
    * [ ] 9.1. Gated Recurrent Units (GRU)
    * [ ] 9.2. Long Short Term Memory (LSTM)
    * [ ] 9.3. Deep Recurrent Neural Networks
    * [ ] 9.4. Bidirectional Recurrent Neural Networks
    * [ ] 9.5. Machine Translation and the Dataset
    * [ ] 9.6. Encoder-Decoder Architecture
    * [ ] 9.7. Sequence to Sequence
    * [ ] 9.8. Beam Search
* [ ] 10. Attention Mechanisms
    * [ ] 10.1. Attention Mechanisms
    * [ ] 10.2. Sequence to Sequence with Attention Mechanisms
    * [ ] 10.3. Transformer
* [ ] 11. Optimization Algorithms
    * [ ] 11.1. Optimization and Deep Learning
    * [ ] 11.2. Convexity
    * [ ] 11.3. Gradient Descent
    * [ ] 11.4. Stochastic Gradient Descent
    * [ ] 11.5. Minibatch Stochastic Gradient Descent
    * [ ] 11.6. Momentum
    * [ ] 11.6. Adagrad
    * [ ] 11.8. RMSProp
    * [ ] 11.9. Adadelta
    * [ ] 11.10. Adam
    * [ ] 11.11. Learning Rate Scheduling
* [ ] 12. [Computational Performance](chapter_computational-performance/index_vn.md)
    * [-] 12.1. [Compilers and Interpreters](chapter_computational-performance/hybridize_vn.md)
    * [ ] 12.2. [Asynchronous Computation](chapter_computational-performance/async-computation_vn.md)
    * [ ] 12.3. [Automatic Parallelism](chapter_computational-performance/auto-parallelism_vn.md)
    * [ ] 12.4. [Hardware](chapter_computational-performance/hardware_vn.md)
    * [ ] 12.5. [Training on Multiple GPUs](chapter_computational-performance/multiple-gpus_vn.md)
    * [ ] 12.6. [Concise Implementation for Multiple GPUs](chapter_computational-performance/multiple-gpus-gluon_vn.md)
    * [ ] 12.6. [Parameter Servers](chapter_computational-performance/parameterserver_vn.md)
* [ ] 13. Computer Vision
    * [ ] 13.1. Image Augmentation
    * [ ] 13.2. Fine Tuning
    * [ ] 13.3. Object Detection and Bounding Boxes
    * [ ] 13.4. Anchor Boxes
    * [ ] 13.5. Multiscale Object Detection
    * [ ] 13.6. The Object Detection Dataset (Pikachu)
    * [ ] 13.7. Single Shot Multibox Detection (SSD)
    * [ ] 13.8. Region-based CNNs (R-CNNs)
    * [ ] 13.9. Semantic Segmentation and the Dataset
    * [ ] 13.10. Transposed Convolution
    * [ ] 13.11. Fully Convolutional Networks (FCN)
    * [ ] 13.12. Neural Style Transfer
    * [ ] 13.13. Image Classification (CIFAR-10) on Kaggle
    * [ ] 13.14. Dog Breed Identification (ImageNet Dogs) on Kaggle
* [ ] 14. Natural Language Processing
    * [ ] 14.1. Word Embedding (word2vec)
    * [ ] 14.2. Approximate Training for Word2vec
    * [ ] 14.3. The Dataset for Word2vec
    * [ ] 14.4. Implementation of Word2vec
    * [ ] 14.5. Subword Embedding
    * [ ] 14.6. Word Embedding with Global Vectors (GloVe)
    * [ ] 14.7. Finding Synonyms and Analogies
    * [ ] 14.8. Sentiment Analysis and the Dataset
    * [ ] 14.9. Sentiment Analysis: Using Recurrent Neural Networks
    * [ ] 14.10. Sentiment Analysis: Using Convolutional Neural Networks
    * [ ] 14.11. Natural Language Inference and the Dataset
* [ ] 15. Recommender Systems
    * [ ] 15.1. Overview of Recommender Systems
    * [ ] 15.2. The MovieLens Dataset
    * [ ] 15.3. Matrix Factorization
    * [ ] 15.4. AutoRec: Rating Prediction with Autoencoders
    * [ ] 15.5. Personalized Ranking for Recommender Systems
    * [ ] 15.6. Neural Collaborative Filtering for Personalized Ranking
    * [ ] 15.7. Sequence-Aware Recommender Systems
    * [ ] 15.8. Feature-Rich Recommender Systems
    * [ ] 15.9. Factorization Machines
    * [ ] 15.10. Deep Factorization Machines
* [ ] 16. Generative Adversarial Networks
    * [ ] 16.1. Generative Adversarial Networks
    * [ ] 16.2. Deep Convolutional Generative Adversarial Networks
* [ ] 17. [Phụ lục: Toán học cho Học Sâu](chapter_appendix_math/index_vn.md)
    * [x] 17.1. [Các phép toán Hình học và Đại số Tuyến tính](chapter_appendix_math/geometry-linear-algebric-ops_vn.md)
    * [ ] 17.2. Eigendecompositions
    * [x] 17.3. [Giải tích một biến](chapter_appendix_math/single-variable-calculus_vn.md)
    * [ ] 17.4. Multivariable Calculus
    * [ ] 17.5. Integral Calculus
    * [ ] 17.6. Random Variables
    * [ ] 17.7. Maximum Likelihood
    * [ ] 17.8. Naive Bayes
    * [x] 17.9. [Thống kê](chapter_appendix_math/statistics_vn.md)
    * [ ] 17.10. Information Theory
* [ ] 18. Appendix: Tools for Deep Learning
    * [ ] 18.1. Using Jupyter
    * [ ] 18.2. Using Amazon SageMaker
    * [ ] 18.3. Using AWS EC2 Instances
    * [ ] 18.4. Using Google Colab
    * [ ] 18.5. Selecting Servers and GPUs
    * [ ] 18.6. Contributing to This Book
    * [ ] 18.7. d2l API Document

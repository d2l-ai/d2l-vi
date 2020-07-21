<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Batch Normalization
-->

# Chuẩn hoá theo batch
:label:`sec_batch_norm`

<!--
Training deep neural nets is difficult.
And getting them to converge in a reasonable amount of time can be tricky.
In this section, we describe batch normalization (BN) :cite:`Ioffe.Szegedy.2015`, a popular and effective technique that consistently accelerates the convergence of deep nets.
Together with residual blocks—covered in :numref:`sec_resnet`—BN has made it possible for practitioners to routinely train networks with over 100 layers.
-->

Huấn luyện mạng nơ-ron sâu không hề đơn giản, 
để chúng hội tụ trong khoảng thời gian chấp nhận được là một câu hỏi khá hóc búa.
Trong phần này, chúng ta giới thiệu chuẩn hóa theo batch (_Batch Normalization - BN_) :cite:`Ioffe.Szegedy.2015`, một kỹ thuật phổ biến và hiệu quả nhằm tăng tốc độ hội tụ của mạng học sâu một cách ổn định.
Cùng với các khối phần dư (*residual block*) được đề cập ở :numref:`sec_resnet` — BN giúp dễ dàng hơn trong việc huấn luyện mạng học sâu với hơn 100 tầng.

<!--
## Training Deep Networks
-->

## Huấn luyện mạng học sâu

<!--
To motivate batch normalization, let us review a few practical challenges that arise
when training ML models and neural nets in particular.
-->

Để thấy mục đích của việc chuẩn hóa theo batch, hãy cùng xem xét lại một vài vấn đề phát sinh trên thực tế khi huấn luyện các mô hình học máy và đặc biệt là mạng nơ-ron.

<!--
1. Choices regarding data preprocessing often make an enormous difference in the final results.
Recall our application of multilayer perceptrons to predicting house prices (:numref:`sec_kaggle_house`).
Our first step when working with real data was to standardize our input features to each have a mean of *zero* and variance of *one*.
Intuitively, this standardization plays nicely with our optimizers because it puts the  parameters are a-priori at a similar scale.
2. For a typical MLP or CNN, as we train, the activations in intermediate layers may take values with widely varying magnitudes—both
along the layers from the input to the output, across nodes in the same layer, and over time due to our updates to the model's parameters.
The inventors of batch normalization postulated informally that this drift in the distribution of activations could hamper the convergence of the network.
Intuitively, we might conjecture that if one layer has activation values that are 100x that of another layer, this might necessitate compensatory adjustments in the learning rates.
3. Deeper networks are complex and easily capable of overfitting.
This means that regularization becomes more critical.
-->

1. Những lựa chọn tiền xử lý dữ liệu khác nhau thường tạo nên sự khác biệt rất lớn trong kết quả cuối cùng.
Hãy nhớ lại việc áp dụng perceptron đa tầng để dự đoán giá nhà (:numref:`sec_kaggle_house`). 
Việc đầu tiên khi làm việc với dữ liệu thực tế là chuẩn hóa các đặc trưng đầu vào để chúng có giá trị trung bình bằng *không* và phương sai bằng *một*. 
Thông thường, việc chuẩn hóa này hoạt động tốt với các bộ tối ưu vì giá trị các tham số tiên nghiệm có cùng một khoảng tỷ lệ.
2. Khi huấn luyện các mạng thường gặp như Perceptron đa tầng hay CNN, các giá trị kích hoạt ở các tầng trung gian có thể nhận các giá trị với mức độ biến thiên lớn-
dọc theo các tầng từ đầu vào đến đầu ra, qua các nút ở cùng một tầng, và theo thời gian do việc cập nhật giá trị tham số.
Những nhà phát minh kỹ thuật chuẩn hoá theo batch cho rằng sự thay đổi trong phân phối của những giá trị kích hoạt có thể cản trở sự hội tụ của mạng.
Dễ thấy rằng nếu một tầng có các giá trị kích hoạt lớn gấp 100 lần so với các tầng khác, thì cần phải có các điều chỉnh bổ trợ trong tốc độ học.
3. Mạng nhiều tầng có độ phức tạp cao và dễ gặp vấn đề quá khớp.
Điều này cũng đồng nghĩa rằng kỹ thuật điều chuẩn càng trở nên quan trọng.


<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
Batch normalization is applied to individual layers (optionally, to all of them) and works as follows:
In each training iteration, for each layer, we first compute its activations as usual.
Then, we normalize the activations of each node by subtracting its mean and dividing by its standard deviation
estimating both quantities based on the statistics of the current the current minibatch.
It is precisely due to this *normalization* based on *batch* statistics that *batch normalization* derives its name.
-->

Chuẩn hoá theo batch được áp dụng cho từng tầng riêng lẻ (hoặc có thể cho tất cả các tầng) và hoạt động như sau:
Trong mỗi vòng lặp huấn luyện, tại mỗi tầng, đầu tiên tính giá trị kích hoạt như thường lệ.
Sau đó chuẩn hóa những giá trị kích hoạt của mỗi nút bằng việc trừ đi giá trị trung bình và chia cho độ lệch chuẩn. 
Cả hai đại lượng này được ước tính dựa trên số liệu thống kê của minibatch hiện tại.
Chính vì *chuẩn hóa* dựa trên các số liệu thống kê của *batch* nên kỹ thuật này có tên gọi *chuẩn hoá theo batch*.


<!--
Note that if we tried to apply BN with minibatches of size $1$, we would not be able to learn anything.
That is because after subtracting the means, each hidden node would take value $0$!
As you might guess, since we are devoting a whole section to BN, with large enough minibatches, the approach proves effective and stable.
One takeaway here is that when applying BN, the choice of minibatch size may be even more significant than without BN.
-->

Lưu ý rằng, khi áp dụng BN với những minibatch có kích thước 1, mô hình sẽ không học được gì. 
Vì sau khi trừ đi giá trị trung bình, mỗi nút ẩn sẽ nhận giá trị $0$! 
Dễ dàng suy luận ra là BN chỉ hoạt động hiệu quả và ổn định với kích thước minibatch đủ lớn. 
Cần ghi nhớ rằng, khi áp dụng BN là lựa chọn kích thước minibatch quan trọng hơn so với trường hợp không áp dụng BN.

<!--
Formally, BN transforms the activations at a given layer $\mathbf{x}$ according to the following expression:
-->

BN chuyển đổi những giá trị kích hoạt tại tầng $x$ nhất định theo công thức sau:

$$\mathrm{BN}(\mathbf{x}) = \mathbf{\gamma} \odot \frac{\mathbf{x} - \hat{\mathbf{\mu}}}{\hat\sigma} + \mathbf{\beta}$$

<!--
Here, $\hat{\mathbf{\mu}}$ is the minibatch sample mean and $\hat{\mathbf{\sigma}}$ is the minibatch sample variance.
After applying BN, the resulting minibatch of activations has zero mean and unit variance.
Because the choice of unit variance (vs some other magic number) is an arbitrary choice,
we commonly include coordinate-wise scaling coefficients $\mathbf{\gamma}$ and offsets $\mathbf{\beta}$.
Consequently, the activation magnitudes for intermediate layers cannot diverge during training
because BN actively centers and rescales them back to a given mean and size (via $\mathbf{\mu}$ and $\sigma$).
One piece of practitioner's intuition/wisdom is that BN seems to allows for more aggressive learning rates.
-->

Ở đây, $\hat{\mathbf{\mu}}$ là giá trị trung bình và $\hat{\mathbf{\sigma}}$ là độ lệch chuẩn của các mẫu trong minibatch.
Sau khi áp dụng BN, những giá trị kích hoạt của minibatch có giá trị trung bình bằng không và phương sai đơn vị.
Vì việc lựa chọn phương sai đơn vị (so với một giá trị đặc biệt khác) là tuỳ ý, 
nên chúng ta thường thêm vào từng cặp tham số tương ứng là hệ số tỷ lệ $\mathbf{\gamma}$ và độ chệch $\mathbf{\beta}$.
Do đó, độ lớn giá trị kích hoạt ở những tầng trung gian không thể phân kỳ trong quá trình huấn luyện vì BN chủ động chuẩn hoá chúng theo giá trị trung bình và phương sai cho trước (thông qua $\mathbf{\mu}$ và $\sigma$).
Qua trực giác và thực nghiệm, dùng BN có thể cho phép chọn tốc độ học nhanh hơn.


<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
Formally, denoting a particular minibatch by $\mathcal{B}$,
we calculate $\hat{\mathbf{\mu}}_\mathcal{B}$ and $\hat\sigma_\mathcal{B}$ as follows:
-->

Ký hiệu một minibatch là $\mathcal{B}$, chúng ta tính $\hat{\mathbf{\mu}}_\mathcal{B}$ và $\hat\sigma_\mathcal{B}$ theo công thức sau:

$$\hat{\mathbf{\mu}}_\mathcal{B} \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\text{ và }
\hat{\mathbf{\sigma}}_\mathcal{B}^2 \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \mathbf{\mu}_{\mathcal{B}})^2 + \epsilon$$


<!--
Note that we add a small constant $\epsilon > 0$ to the variance estimate
to ensure that we never attempt division by zero, even in cases where the empirical variance estimate might vanish.
The estimates $\hat{\mathbf{\mu}}_\mathcal{B}$ and $\hat{\mathbf{\sigma}}_\mathcal{B}$ counteract the scaling issue by using noisy estimates of mean and variance.
You might think that this noisiness should be a problem.
As it turns out, this is actually beneficial.
-->

Lưu ý rằng chúng ta thêm hằng số rất nhỏ $\epsilon > 0$ vào biểu thức tính phương sai để đảm bảo tránh phép chia cho 0 khi chuẩn hoá, ngay cả khi giá trị ước lượng phương sai thực nghiệm bằng không.
Các ước lượng $\hat{\mathbf{\mu}}_\mathcal{B}$ và $\hat{\mathbf{\sigma}}_\mathcal{B}$ giúp đương đầu với vấn đề khi cần mở rộng số tầng của mạng (mạng học sâu hơn) bằng việc sử dụng nhiễu khi tính giá trị trung bình và phương sai.
Bạn có thể nghĩ rằng nhiễu sẽ là vấn đề đáng ngại.
Nhưng thực ra, nhiễu lại đem đến lợi ích.

<!--
This turns out to be a recurring theme in deep learning.
For reasons that are not yet well-characterized theoretically, various sources of noise in optimization often lead to faster training and less overfitting.
While traditional machine learning theorists might buckle at this characterization, this variation appears to act as a form of regularization.
In some preliminary research, :cite:`Teye.Azizpour.Smith.2018` and :cite:`Luo.Wang.Shao.ea.2018` relate the properties of BN to Bayesian Priors and penalties respectively.
In particular, this sheds some light on the puzzle of why BN works best for moderate minibatches sizes in the $50$–$100$ range.
-->

Và đây là chủ đề thường xuất hiện trong học sâu.
Vì những lý do vẫn chưa được giải thích rõ bằng lý thuyết, nhiều nguồn nhiễu khác nhau trong việc tối ưu hoá thường dẫn đến huấn luyện nhanh hơn và giảm quá khớp.
Trong khi những nhà lý thuyết học máy truyền thống có thể bị vướng mắc ở việc định rõ điểm này, những thay đổi do nhiễu dường như hoạt động giống một dạng điều chuẩn.
Trong một số nghiên cứu sơ bộ, :cite:`Teye.Azizpour.Smith.2018` và :cite:`Luo.Wang.Shao.ea.2018` đã lần lượt chỉ ra các thuộc tính của BN liên quan tới tiên nghiệm Bayesian (_Bayesian prior_) và các lượng phạt (_penalty_). 
Cụ thể, nghiên cứu này làm sáng tỏ lý do BN hoạt động tốt nhất với các minibatch có kích cỡ vừa phải, trong khoảng 50 - 100.

<!--
Fixing a trained model, you might (rightly) think that we would prefer to use the entire dataset to estimate the mean and variance.
Once training is complete, why would we want the same image to be classified differently, depending on the batch in which it happens to reside?
During training, such exact calculation is infeasible because the activations for all data points change every time we update our model.
However, once the model is trained, we can calculate the means and variances of each layer's activations based on the entire dataset.
Indeed this is standard practice for models employing batch normalization and thus MXNet's BN layers function differently 
in *training mode* (normalizing by minibatch statistics) and in *prediction mode* (normalizing by dataset statistics).
-->

Cố định một mô hình đã được huấn luyện, bạn có thể nghĩ rằng chúng ta nên sử dụng toàn bộ tập dữ liệu để ước tính giá trị trung bình và phương sai. Và đúng là như vậy.
Bởi lẽ khi huấn luyện xong, tại sao ta lại muốn cùng một hình ảnh lại được phân loại khác nhau phụ thuộc vào batch chứa hình ảnh này?
Trong quá trình huấn luyện, những tính toán chính xác như vậy không khả thi vì giá trị kích hoạt cho tất cả các điểm dữ liệu thay đổi mỗi khi cập nhật mô hình.
Tuy nhiên, một khi mô hình đã được huấn luyện xong, chúng ta có thể tính được giá trị trung bình và phương sai của mỗi tầng dựa trên toàn bộ tập dữ liệu.
Thực ra đây là tiêu chuẩn thực hành cho các mô hình sử dụng chuẩn hóa theo batch và do đó các tầng BN của MXNet hoạt động khác nhau
giữa *chế độ huấn luyện* (chuẩn hoá bằng số liệu thống kê của minibatch) và *chế độ dự đoán* (chuẩn hoá bằng số liệu thống kê của toàn bộ tập dữ liệu)

<!--
We are now ready to take a look at how batch normalization works in practice.
-->

Bây giờ chúng ta đã sẵn sàng để xem chuẩn hoá theo batch hoạt động thế nào trong thực tế.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
## Batch Normalization Layers
-->

## Tầng chuẩn hoá theo batch

<!--
Batch normalization implementations for fully-connected layers and convolutional layers are slightly different.
We discuss both cases below.
Recall that one key differences between BN and other layers is that because BN operates on a full minibatch at a time,
 we cannot just ignore the batch dimension as we did before when introducing other layers.
-->

Thực hiện việc chuẩn hóa theo batch cho tầng kết nối đầy đủ và tầng tích chập hơi khác nhau một chút.
Chúng ta sẽ thảo luận cả hai trường hợp trên.
Nhớ rằng một khác biệt lớn của BN so với những tầng khác là vì BN cần số liệu thống kê trên toàn minibatch,
chúng ta không thể bỏ qua kích thước batch như đã làm với các tầng khác.

<!--
### Fully-Connected Layers
-->

### Tầng kết nối đầy đủ

<!--
When applying BN to fully-connected layers, we usually insert BN after the affine transformation and before the nonlinear activation function.
Denoting the input to the layer by $\mathbf{x}$, the linear transform (with weights $\theta$) by $f_{\theta}(\cdot)$, 
the activation function by $\phi(\cdot)$, and the BN operation with parameters $\mathbf{\beta}$ and $\mathbf{\gamma}$ by $\mathrm{BN}_{\mathbf{\beta}, \mathbf{\gamma}}$, 
we can express the computation of a BN-enabled, fully-connected layer $\mathbf{h}$ as follows:
-->

Khi áp dụng BN cho tầng kết nối đầy đủ, ta thường chèn BN sau bước biến đổi affine và trước hàm kích hoạt phi tuyến. 
Kí hiệu đầu vào của tầng là $\mathbf{x}$, hàm biến đổi tuyến tính là $f_{\theta}(\cdot)$ (với trọng số là $\theta$), 
hàm kích hoạt là $\phi(\cdot)$, và phép tính BN là $\mathrm{BN}_{\mathbf{\beta}, \mathbf{\gamma}}$ với tham số $\mathbf{\beta}$ và $\mathbf{\gamma}$, 
chúng ta sẽ biểu diễn việc tính toán tầng kết nối đầy đủ $\mathbf{h}$ khi chèn lớp BN vào như sau:

$$\mathbf{h} = \phi(\mathrm{BN}_{\mathbf{\beta}, \mathbf{\gamma}}(f_{\mathbf{\theta}}(\mathbf{x}) ) ) $$

<!--
Recall that mean and variance are computed on the *same* minibatch $\mathcal{B}$ on which the transformation is applied.
Also recall that the scaling coefficient $\mathbf{\gamma}$ and the offset $\mathbf{\beta}$ are parameters that need to be learned jointly with the more familiar parameters $\mathbf{\theta}$.
-->

Nhắc lại rằng giá trị trung bình và phương sai sẽ được tính toán trên *chính* minibatch $\mathcal{B}$ mà sẽ được biến đổi. 
Cũng cần lưu ý rằng hệ số tỷ lệ $\mathbf{\gamma}$ và độ chệch $\mathbf{\beta}$ là những tham số cần được học cùng với bộ tham số quen thuộc $\mathbf{\theta}$.

<!--
### Convolutional Layers
-->

### Tầng tích chập

<!--
Similarly, with convolutional layers, we typically apply BN after the convolution and before the nonlinear activation function.
When the convolution has multiple output channels, we need to carry out batch normalization for *each* of the outputs of these channels, 
and each channel has its own scale and shift parameters, both of which are scalars.
Assume that our minibatches contain $m$ each and that for each channel, the output of the convolution has height $p$ and width $q$.
For convolutional layers, we carry out each batch normalization over the $m \cdot p \cdot q$ elements per output channel simultaneously.
Thus we collect the values over all spatial locations when computing the mean and variance and consequently (within a given channel)
apply the same $\hat{\mathbf{\mu}}$ and $\hat{\mathbf{\sigma}}$ to normalize the values at each spatial location.
-->

Tương tự với tầng tích chập, chúng ta áp dụng BN sau phép tích chập và trước hàm kích hoạt phi tuyến.
Khi áp dụng phép tích chập cho đầu ra nhiều kênh, chúng ta cần thực hiện chuẩn hóa theo batch cho *mỗi* đầu ra của những kênh này, 
và mỗi kênh sẽ có riêng cho nó các tham số tỉ lệ và độ chệch, cả hai đều là các số vô hướng.
Giả sử các minibatch có kích thước $m$, đầu ra cho mỗi kênh của phép tích chập có chiều cao $p$ và chiều rộng $q$.
Với tầng tích chập, ta sẽ thực hiện mỗi phép chuẩn hoá theo batch trên $m \cdot p \cdot q$ phần tử trên từng kênh đầu ra cùng lúc.
Vì thế trên từng kênh, ta sử dụng giá trị trên tất cả các vị trí không gian để tính trung bình $\hat{\mathbf{\mu}}$ và phương sai $\hat{\mathbf{\sigma}}$
và sau đó dùng hai giá trị này để chuẩn hóa các giá trị tại mỗi vị trí không gian trên kênh đó.


<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
### Batch Normalization During Prediction
-->

### Chuẩn hoá theo Batch trong Quá trình Dự đoán

<!--
As we mentioned earlier, BN typically behaves differently in training mode and prediction mode.
First, the noise in $\mathbf{\mu}$ and $\mathbf{\sigma}$ arising from estimating each on minibatches are no longer desirable once we have trained the model.
Second, we might not have the luxury of computing per-batch normalization statistics, e.g., we might need to apply our model to make one prediction at a time.
-->

Như đã đề cập trước đó, BN thường hoạt động khác nhau trong chế độ huấn luyện và chế độ dự đoán.
Thứ nhất, nhiễu trong $\mu$ và $\sigma$ phát sinh từ việc chúng được xấp xỉ trên các minibatch không còn là nhiễu được mong muốn một khi ta đã huấn luyện xong mô hình.
Thứ hai, trong nhiều trường hợp sẽ là xa xỉ khi tính toán các con số thống kê sau mỗi lần chuẩn hoá theo batch, ví dụ, khi cần áp dụng mô hình để đưa ra một kết quả dự đoán mỗi lần.

<!--
Typically, after training, we use the entire dataset to compute stable estimates of the activation statistics and then fix them at prediction time.
Consequently, BN behaves differently during training and at test time.
Recall that dropout also exhibits this characteristic.
-->

Thông thường, sau khi huấn luyện, chúng ta sử dụng toàn bộ tập dữ liệu để tính toán các con số thống kê của các giá trị kích hoạt và sau đó cố định chúng tại thời điểm dự đoán.
Do đó, BN hoạt động khác nhau trong quá trình huấn luyện và kiểm tra.
Lưu ý rằng dropout cũng có tính chất này.

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
## Implementation from Scratch
-->

## Lập trình từ đầu

<!--
Below, we implement a batch normalization layer with `ndarray`s from scratch:
-->

Dưới đây, chúng ta lập trình tầng chuẩn hoá theo batch chỉ dùng `ndarray`.

```{.python .input  n=72}
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is the prediction mode, directly use the mean and variance
        # obtained from the incoming moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcast operation
            # can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance of the moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

<!--
We can now create a proper `BatchNorm` layer.
Our layer will maintain proper parameters corresponding for scale `gamma` and shift `beta`, both of which will be updated in the course of training.
Additionally, our layer will maintain a moving average of the means and variances for subsequent use during model prediction.
The `num_features` parameter required by the `BatchNorm` instance is the number of outputs for a fully-connected layer and the number of output channels for a convolutional layer.
The `num_dims` parameter also required by this instance is 2 for a fully-connected layer and 4 for a convolutional layer.
-->

Giờ ta có thể tạo một tầng `BatchNorm` đúng cách.
Tầng này sẽ duy trì những tham số thích hợp tương ứng với tỉ lệ `gamma` và độ chệch `beta`, hai tham số này sẽ được cập nhật trong quá trình huấn luyện.
Thêm vào đó, tầng BN sẽ duy trì giá trị trung bình động của trung bình và phương sai để sử dụng về sau khi ở chế độ dự đoán. 
Tham số `num_features` truyền vào `BatchNorm` là số đầu ra của tầng kết nối đầy đủ hoặc số kênh đầu ra của tầng tích chập.
Tham số `num_dims` bằng 2 nếu là tầng kết nối đầy đủ và bằng 4 nếu là tầng tích chập.


<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
Putting aside the algorithmic details, note the design pattern underlying our implementation of the layer.
Typically, we define the math in a separate function, say `batch_norm`.
We then integrate this functionality into a custom layer, whose code mostly addresses bookkeeping matters, 
such as moving data to the right device context, allocating and initializing any required variables, keeping track of running averages (here for mean and variance), etc.
This pattern enables a clean separation of math from boilerplate code.
Also note that for the sake of convenience we did not worry about automatically inferring the input shape here, thus our need to specify the number of features throughout.
Do not worry, the Gluon `BatchNorm` layer will care of this for us.
-->

Tạm để thuật toán sang một bên và tập trung vào khuôn mẫu thiết kế (*design pattern*) của việc lập trình. 
Thông thường, ta lập trình phần toán trong một hàm riêng biệt, ví dụ như `batch_norm`.
Sau đó, ta tích hợp chức năng này vào một tầng tùy chỉnh, với mã nguồn chủ yếu giải quyết các vấn đề phụ trợ như di chuyển dữ liệu đến thiết bị phù hợp, cấp phát và khởi tạo biến, theo dõi các giá trị trung bình động (của trung bình và phương sai trong trường hợp này), v.v.
Khuôn mẫu này giúp tách biệt việc tính toán khỏi các đoạn mã rập khuôn.
Cũng lưu ý rằng để thuận tiện khi lập trình BN từ đầu, ta không tự động suy ra kích thước đầu vào, do đó ta cần chỉ định số lượng đặc trưng xuyên suốt.
Tầng `BatchNorm` của Gluon sẽ hỗ trợ việc tự động này bằng khởi tạo trễ.

```{.python .input  n=73}
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter involved in gradient
        # finding and iteration are initialized to 0 and 1 respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # All the variables not involved in gradient finding and iteration are
        # initialized to 0 on the CPU
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.zeros(shape)

    def forward(self, X):
        # If X is not on the CPU, copy moving_mean and moving_var to the
        # device where X is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

<!--
## Using a Batch Normalization LeNet
-->

## Sử dụng LeNet với Chuẩn hóa theo Batch

<!--
To see how to apply `BatchNorm` in context, below we apply it to a traditional LeNet model (:numref:`sec_lenet`).
Recall that BN is typically applied after the convolutional layers and fully-connected layers but before the corresponding activation functions.
-->

Để biết cách áp dụng `BatchNorm` trên thực tế, bên dưới ta áp dụng cho mô hình LeNet truyền thống (:numref:`sec_lenet`).
Nhắc lại rằng BN thường được sử dụng sau tầng tích chập và tầng kết nối đầy đủ và trước hàm kích hoạt tương ứng.

```{.python .input  n=74}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

<!--
As before, we will train our network on the Fashion-MNIST dataset.
This code is virtually identical to that when we first trained LeNet (:numref:`sec_lenet`).
The main difference is the considerably larger learning rate.
-->

Như thường lệ, ta sẽ huấn luyện trên bộ dữ liệu Fashion-MNIST.
Đoạn mã này gần tương tự với đoạn mã khi lần đầu huấn luyện LeNet (:numref:`sec_lenet`).
Điểm khác biệt chính là tốc độ học lớn hơn đáng kể.

```{.python .input  n=77}
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

<!--
Let us have a look at the scale parameter `gamma` and the shift parameter `beta` learned from the first batch normalization layer.
-->

Chúng ta hãy xem tham số tỷ lệ `gamma` và tham số dịch chuyển `beta` đã học được tại tầng chuẩn hóa theo batch đầu tiên.

```{.python .input  n=60}
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
## Concise Implementation

-->

## Lập trình súc tích

<!--
Compared with the `BatchNorm` class, which we just defined ourselves, the `BatchNorm` class defined by the `nn` model in Gluon is easier to use.
In Gluon, we do not have to worry about `num_features` or `num_dims`.
Instead, these parameter values will be inferred automatically via delayed initialization.
Otherwise, the code looks virtually identical to the application our implementation above.
-->

So với lớp `BatchNorm` tự định nghĩa thì lớp` BatchNorm` định nghĩa trong `nn` của Gluon dễ sử dụng hơn.
Trong Gluon, ta không cần chỉ rõ `num_features` và `num_dims`.
Thay vào đó, các giá trị này sẽ được tự động suy ra trong quá trình khởi tạo trễ.
Ngoại trừ điểm đó, đoạn mã trông giống hệt đoạn mã phía trên.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

<!--
Below, we use the same hyper-parameters to train out model.
Note that as usual, the Gluon variant runs much faster because its code has been compiled to C++/CUDA
while our custom implementation must be interpreted by Python.
-->

Chúng ta sử dụng cùng các siêu tham số như trước để huấn luyện mô hình.
Như thường lệ, biến thể dùng Gluon này chạy nhanh hơn nhiều vì được biên dịch thành C++/CUDA trong khi đoạn mã tùy chỉnh của chúng ta phải qua thông dịch bằng Python.

```{.python .input}
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr)
```

<!--
## Controversy
-->

## Tranh luận

<!--
Intuitively, batch normalization is thought to make the optimization landscape smoother.
However, we must be careful to distinguish between speculative intuitions and true explanations for the phenomena that we observe when training deep models.
Recall that we do not even know why simpler deep neural networks (MLPs and conventional CNNs) generalize well in the first place.
Even with dropout and L2 regularization, they remain so flexible that their ability to generalize to unseen data cannot be explained via conventional learning-theoretic generalization guarantees.
-->

Theo trực giác, chuẩn hóa theo batch được cho là làm cảnh quan tối ưu (*optimization landscape*) mượt mà hơn.
Tuy nhiên, cần cẩn thận phân biệt giữa suy đoán theo trực giác và lời giải thích thực sự cho các hiện tượng quan sát thấy khi huấn luyện các mô hình học sâu.
Hãy nhớ lại rằng ngay từ đầu ta thậm chí không rõ tại sao các mạng nơ-ron sâu đơn giản hơn (như Perceptron đa tầng và CNN truyền thống) lại có thể khái quát tốt như vậy.
Ngay cả với dropout và điều chuẩn L2, chúng vẫn linh hoạt đến mức khả năng khái quát hóa trên dữ liệu chưa nhìn thấy của chúng không thể giải thích được bằng các điều kiện bảo đảm sự khái quát hóa trong lý thuyết học truyền thống.

<!--
In the original paper proposing batch normalization, the authors, in addition to introducing a powerful and useful tool, 
offered an explanation for why it works: by reducing *internal covariate shift*.
Presumably by *internal covariate shift* the authors meant something like the intuition expressed above—the notion that the distribution of activations changes over the course of training.
However there were two problems with this explanation:
(1) This drift is very different from *covariate shift*, rendering the name a misnomer.
(2) The explanation offers an under-specified intuition but leaves the question of *why precisely this technique works* an open question wanting for a rigorous explanation.
Throughout this book, we aim to convey the intuitions that practitioners use to guide their development of deep neural networks.
However, we believe that it is important to separate these guiding intuitions from established scientific fact.
Eventually, when you master this material and start writing your own research papers you will want to be clear to delineate between technical claims and hunches.
-->

Trong bài báo gốc khi đề xuất phương pháp chuẩn hóa theo batch, các tác giả ngoài việc giới thiệu một công cụ mạnh mẽ và hữu ích đã đưa ra lời giải thích lý do BN hoạt động tốt: bằng cách giảm *sự dịch chuyển hiệp biến nội bộ - internal covariate shift*.
Có thể hiểu ý các tác giả về *sự dịch chuyển hiệp biến nội bộ* giống với cách giải thích ở trên-rằng phân phối của giá trị kích hoạt thay đổi trong quá trình huấn luyện.
Tuy nhiên, có hai vấn đề với cách giải thích này:
(1) Sự dịch chuyển phân phối này rất khác so với *sự dịch chuyển hiệp biến*, việc đặt tên như vậy có sự nhầm lẫn.
(2) Cách giải thích này vẫn chưa đủ cụ thể và chặt chẽ, vẫn để ngỏ câu hỏi: *chính xác thì tại sao kỹ thuật này hoạt động?*
Xuyên suốt cuốn sách này, chúng tôi hướng đến việc truyền đạt những kinh nghiệm thực tế để xây dựng các mạng nơ-ron sâu.
Tuy nhiên, chúng tôi tin rằng cần phân biệt rõ những kinh nghiệm dựa trên trực giác này với những bằng chứng khoa học rõ ràng.
Cuối cùng, khi đã thành thạo tài liệu này và bắt đầu viết các nghiên cứu của riêng mình, bạn cần phân biệt rõ ràng giữa khẳng định và linh cảm.

<!--
Following the success of batch normalization, its explanation in terms of *internal covariate shift* has repeatedly surfaced 
in debates in the technical literature and broader discourse about how to present machine learning research.
In a memorable speech given while accepting a Test of Time Award at the 2017 NeurIPS conference,
Ali Rahimi used *internal covariate shift* as a focal point in an argument likening the modern practice of deep learning to alchemy.
Subsequently, the example was revisited in detail in a position paper outlining troubling trends in machine learning :cite:`Lipton.Steinhardt.2018`.
In the technical literature other authors (:cite:`Santurkar.Tsipras.Ilyas.ea.2018`) have proposed alternative explanations for the success of BN, 
some claiming that BN's success comes despite exhibiting behavior that is in some ways opposite to those claimed in the original paper.
-->

Nối tiếp thành công của BN, cách giải thích của kỹ thuật này thông qua khái niệm *sự dịch chuyển hiệp biến nội bộ* liên tục xuất hiện
trong các tranh luận, các tài liệu kỹ thuật và trên các diễn đàn về cách trình bày nghiên cứu học máy.
Trong một bài phát biểu đáng nhớ được đưa ra khi nhận giải thưởng **Test of Time Award** tại hội nghị NeurIPS 2017,
Ali Rahimi đã sử dụng *sự dịch chuyển hiệp biến nội bộ* như một tiêu điểm trong một cuộc tranh luận so sánh thực hành học sâu hiện đại với thuật giả kim.
Sau đó, cách giải thích này đã được xem xét lại một cách chi tiết trong một bài báo về các xu hướng đáng lo ngại trong học máy :cite:`Lipton.Steinhardt.2018`.
Trong các tài liệu kỹ thuật, các tác giả khác (:cite:`Santurkar.Tsipras.Ilyas.ea.2018`) đã đề xuất các giải thích thay thế cho sự thành công của BN, dù phần nào đó trái ngược với cách giải thích trong bài báo gốc.

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
We note that the *internal covariate shift* is no more worthy of criticism than any of thousands of similarly vague claims made every year in the technical ML literature.
Likely, its resonance as a focal point of these debates owes to its broad recognizability to the target audience.
Batch normalization has proven an indispensable method, applied in nearly all deployed image classifiers, earning the paper that introduced the technique tens of thousands of citations.
-->

Chúng tôi lưu ý rằng *sự dịch chuyển hiệp biến nội bộ* không đáng bị chỉ trích, có hàng ngàn lập luận mơ hồ được đưa ra mỗi năm trong nhiều tài liệu kỹ thuật về học máy.
Việc nó trở thành tâm điểm của những cuộc tranh luận rất có thể là do sự phổ biến của nó trong cộng đồng học máy.
Chuẩn hóa theo batch là một phương pháp quan trọng, được áp dụng trong gần như tất cả các bộ phân loại hình ảnh đã được triển khai, mang lại hàng chục ngàn trích dẫn cho bài báo giới thiệu kĩ thuật này.


<!--
## Summary
-->

## Tóm tắt

<!--
* During model training, batch normalization continuously adjusts the intermediate output of the neural network by utilizing the mean and standard deviation of the minibatch, so that the values of the intermediate output in each layer throughout the neural network are more stable.
* The batch normalization methods for fully connected layers and convolutional layers are slightly different.
* Like a dropout layer, batch normalization layers have different computation results in training mode and prediction mode.
* Batch Normalization has many beneficial side effects, primarily that of regularization. On the other hand, the original motivation of reducing covariate shift seems not to be a valid explanation.
-->

* Trong quá trình huấn luyện mô hình, chuẩn hóa theo batch liên tục điều chỉnh đầu ra trung gian của mạng nơ-ron theo giá trị trung bình và độ lệch chuẩn của minibatch, giúp các giá trị này ổn định hơn.
* Chuẩn hóa theo batch có chút khác biệt khi áp dụng cho tầng kết nối đầy đủ và tầng tích chập.
* Giống như tầng dropout, tầng chuẩn hóa theo batch sẽ tính ra kết quả khác nhau trong chế độ huấn luyện và chế độ dự đoán.
* Chuẩn hóa theo batch có nhiều tác dụng phụ có lợi, chủ yếu là về điều chuẩn. Tuy nhiên, cách giải thích ban đầu về việc giảm sự dịch chuyển hiệp biến dường như không hợp lý.

<!--
## Exercises
-->

## Bài tập

<!--
1. Can we remove the fully connected affine transformation before the batch normalization or the bias parameter in convolution computation?
    * Find an equivalent transformation that applies prior to the fully connected layer.
    * Is this reformulation effective. Why (not)?
2. Compare the learning rates for LeNet with and without batch normalization.
    * Plot the decrease in training and test error.
    * What about the region of convergence? How large can you make the learning rate?
3. Do we need Batch Normalization in every layer? Experiment with it?
4. Can you replace Dropout by Batch Normalization? How does the behavior change?
5. Fix the coefficients `beta` and `gamma` (add the parameter `grad_req='null'` at the time of construction to avoid calculating the gradient), and observe and analyze the results.
6. Review the Gluon documentation for `BatchNorm` to see the other applications for Batch Normalization.
7. Research ideas: think of other normalization transforms that you can apply? Can you apply the probability integral transform? How about a full rank covariance estimate?
-->

1. Trước khi chuẩn hóa theo batch, có thể loại bỏ phép biến đổi affine trong tầng kết nối đầy đủ hoặc tham số độ chệch trong phép tích chập không?
    * Tìm một phép biến đổi tương đương được áp dụng trước tầng kết nối đầy đủ.
    * Sự cải tiến này có hiệu quả không, tại sao?
2. So sánh tốc độ học của LeNet khi có sử dụng và không sử dụng chuẩn hóa theo batch.
    * Vẽ đồ thị biểu diễn sự giảm xuống của lỗi huấn luyện và lỗi kiểm tra.
    * Về miền hội tụ thì sao? Có thể chọn tốc độ học lớn tới đâu?
3. Chúng ta có cần chuẩn hóa theo batch trong tất cả các tầng không? Hãy thử nghiệm điều này.
4. Có thể thay thế Dropout bằng BN không? Sẽ có thay đổi như thế nào?
5. Giữ nguyên các hệ số `beta` và `gamma` (thêm tham số `grad_req='null'` khi xây dựng mạng để không tính gradient) rồi quan sát và phân tích kết quả.
6. Đọc tài liệu của Gluon về `BatchNorm` để xem các ứng dụng khác của chuẩn hóa theo batch.
7. Ý tưởng nghiên cứu: nghĩ về các phép biến đổi chuẩn hóa khác có thể áp dụng. Bạn có thể áp dụng biến đổi tích phân xác suất (*probability integral transform*) không? Còn ước lượng ma trận hiệp phương sai hạng tối đa thì sao?

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->
<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->


## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2358)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Đinh Đắc
* Lê Khắc Hồng Phúc
* Nguyễn Văn Cường
* Trần Yến Thy
* Phạm Minh Đức
* Nguyễn Cảnh Thướng
* Phạm Hồng Vinh

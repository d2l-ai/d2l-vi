# Xếp hạng được cá nhân hóa cho hệ thống giới thiệu

Trong các phần trước, chỉ có phản hồi rõ ràng được xem xét và các mô hình đã được đào tạo và thử nghiệm trên xếp hạng quan sát. Có hai nhược điểm của các phương pháp như vậy: Thứ nhất, hầu hết các phản hồi không rõ ràng nhưng ngầm trong các kịch bản trong thế giới thực, và phản hồi rõ ràng có thể tốn kém hơn để thu thập. Thứ hai, các cặp mục người dùng không quan sát được có thể dự đoán cho sở thích của người dùng hoàn toàn bị bỏ qua, khiến các phương pháp này không phù hợp với các trường hợp xếp hạng không bị thiếu ngẫu nhiên nhưng vì sở thích của người dùng. Các cặp mục người dùng không quan sát được là một hỗn hợp của phản hồi tiêu cực thực sự (người dùng không quan tâm đến các mục) và các giá trị bị thiếu (người dùng có thể tương tác với các mục trong tương lai). Chúng tôi chỉ đơn giản là bỏ qua các cặp không quan sát trong factorization ma trận và AutoRec. Rõ ràng, các mô hình này không có khả năng phân biệt giữa các cặp quan sát và không quan sát được và thường không phù hợp với các nhiệm vụ xếp hạng được cá nhân hóa. 

Để kết thúc này, một lớp mô hình đề xuất nhắm mục tiêu tạo danh sách đề xuất được xếp hạng từ phản hồi ngầm đã trở nên phổ biến. Nói chung, các mô hình xếp hạng được cá nhân hóa có thể được tối ưu hóa với các phương pháp theo chiều dọc, theo chiều ngang hoặc theo chiều dọc. Các phương pháp tiếp cận Pointwise xem xét một tương tác duy nhất tại một thời điểm và đào tạo một phân loại hoặc một regressor để dự đoán các sở thích cá nhân. Matrận factorization và AutoRec được tối ưu hóa với các mục tiêu pointwise. Các phương pháp tiếp cận cặp xem xét một cặp mặt hàng cho mỗi người dùng và nhằm mục đích xấp xỉ thứ tự tối ưu cho cặp đó. Thông thường, cách tiếp cận cặp phù hợp hơn với nhiệm vụ xếp hạng vì dự đoán thứ tự tương đối gợi nhớ đến bản chất của bảng xếp hạng. Listwise tiếp cận gần đúng thứ tự của toàn bộ danh sách các mặt hàng, ví dụ, tối ưu hóa trực tiếp các biện pháp xếp hạng như Đạt được tích lũy giảm giá chuẩn hóa ([NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)). Tuy nhiên, các cách tiếp cận theo chiều dọc theo danh sách phức tạp và tính toán nhiều hơn so với các phương pháp tiếp cận theo chiều dọc hoặc theo cặp. Trong phần này, chúng tôi sẽ giới thiệu hai đối tượng/lỗ theo cặp, mất Xếp hạng cá nhân Bayesian và thua bản lề và triển khai tương ứng của chúng. 

## Mất xếp hạng cá nhân Bayesian và triển khai

Xếp hạng cá nhân hóa Bayesian (BPR) :cite:`Rendle.Freudenthaler.Gantner.ea.2009` là một tổn thất xếp hạng được cá nhân hóa theo cặp có nguồn gốc từ ước tính sau tối đa. Nó đã được sử dụng rộng rãi trong nhiều mô hình khuyến nghị hiện có. Dữ liệu đào tạo của BPR bao gồm cả cặp dương và âm (giá trị thiếu). Nó giả định rằng người dùng thích mục tích cực hơn tất cả các mục không quan sát khác. 

Trong chính thức, dữ liệu đào tạo được xây dựng bởi các tuples dưới dạng $(u, i, j)$, đại diện cho rằng người dùng $u$ thích mục $i$ hơn mục $j$. Công thức Bayesian của BPR nhằm mục đích tối đa hóa xác suất sau được đưa ra dưới đây: 

$$
p(\Theta \mid >_u )  \propto  p(>_u \mid \Theta) p(\Theta)
$$

Trong đó $\Theta$ đại diện cho các thông số của một mô hình khuyến nghị tùy ý, $>_u$ đại diện cho tổng thứ hạng được cá nhân hóa mong muốn của tất cả các mục cho người dùng $u$. Chúng tôi có thể xây dựng ước tính sau tối đa để lấy được tiêu chí tối ưu hóa chung cho nhiệm vụ xếp hạng được cá nhân hóa. 

$$
\begin{aligned}
\text{BPR-OPT} : &= \ln p(\Theta \mid >_u) \\
         & \propto \ln p(>_u \mid \Theta) p(\Theta) \\
         &= \ln \prod_{(u, i, j \in D)} \sigma(\hat{y}_{ui} - \hat{y}_{uj}) p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \ln p(\Theta) \\
         &= \sum_{(u, i, j \in D)} \ln \sigma(\hat{y}_{ui} - \hat{y}_{uj}) - \lambda_\Theta \|\Theta \|^2
\end{aligned}
$$

trong đó $D := \{(u, i, j) \mid i \in I^+_u \wedge j \in I \backslash I^+_u \}$ là bộ đào tạo, với $I^+_u$ biểu thị các mục mà người dùng $u$ thích, $I$ biểu thị tất cả các mục và $I \backslash I^+_u$ chỉ ra tất cả các mục khác không bao gồm các mặt hàng mà người dùng thích. $\hat{y}_{ui}$ và $\hat{y}_{uj}$ là điểm dự đoán của người dùng $u$ đối với mục $i$ và $\hat{y}_{uj}$ 19, tương ứng. $p(\Theta)$ trước đó là một phân phối bình thường với ma trận trung bình bằng 0 và phương sai - đồng phương sai $\Sigma_\Theta$. Ở đây, chúng tôi để $\Sigma_\Theta = \lambda_\Theta I$. 

! [Illustration of Bayesian Personalized Ranking](../img/rec-ranking.svg) Chúng tôi sẽ thực hiện lớp cơ sở `mxnet.gluon.loss.Loss` và ghi đè phương pháp `forward` để xây dựng tổn thất xếp hạng cá nhân Bayesian. Chúng ta bắt đầu bằng cách nhập class Loss và module np.

```{.python .input  n=5}
from mxnet import gluon, np, npx
npx.set_np()
```

Việc thực hiện tổn thất BPR như sau.

```{.python .input  n=2}
#@save
class BPRLoss(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(BPRLoss, self).__init__(weight=None, batch_axis=0, **kwargs)

    def forward(self, positive, negative):
        distances = positive - negative
        loss = - np.sum(np.log(npx.sigmoid(distances)), 0, keepdims=True)
        return loss
```

## Bản lề mất và thực hiện của nó

Bản lề mất cho xếp hạng có hình thức khác nhau để [bản lề mất](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.HingeLoss) được cung cấp trong thư viện gluon thường được sử dụng trong các phân loại như SVM. Sự mất mát được sử dụng để xếp hạng trong các hệ thống recommender có hình thức sau. 

$$
 \sum_{(u, i, j \in D)} \max( m - \hat{y}_{ui} + \hat{y}_{uj}, 0)
$$

trong đó $m$ là kích thước lề an toàn. Nó nhằm mục đích đẩy các mặt hàng tiêu cực ra khỏi các mặt hàng tích cực. Tương tự như BPR, nó nhằm mục đích tối ưu hóa khoảng cách có liên quan giữa các mẫu dương và âm thay vì đầu ra tuyệt đối, làm cho nó rất phù hợp với các hệ thống giới thiệu.

```{.python .input  n=3}
#@save
class HingeLossbRec(gluon.loss.Loss):
    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(HingeLossbRec, self).__init__(weight=None, batch_axis=0,
                                            **kwargs)

    def forward(self, positive, negative, margin=1):
        distances = positive - negative
        loss = np.sum(np.maximum(- distances + margin, 0))
        return loss
```

Hai khoản lỗ này có thể hoán đổi cho nhau để xếp hạng được cá nhân hóa trong đề xuất. 

## Tóm tắt

- Có ba loại thua lỗ xếp hạng có sẵn cho nhiệm vụ xếp hạng được cá nhân hóa trong các hệ thống đề xuất, cụ thể là phương pháp theo chiều kim đồng hồ, theo cặp và theo danh sách.
- Hai thua cặp, mất thứ hạng cá nhân của Bayesian và thua bản lề, có thể được sử dụng thay thế cho nhau.

## Bài tập

- Có bất kỳ biến thể nào của BPR và bản lề mất có sẵn không?
- Bạn có thể tìm thấy bất kỳ mô hình khuyến nghị nào sử dụng BPR hoặc mất bản lề không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/402)
:end_tab:

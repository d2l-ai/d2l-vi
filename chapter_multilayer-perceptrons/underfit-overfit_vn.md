<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Model Selection, Underfitting and Overfitting
-->

# Lựa Chọn Mô Hình, Dưới Khớp và Quá Khớp
:label:`sec_model_selection`

<!--
As machine learning scientists, our goal is to discover *patterns*.
But how can we be sure that we have truly discovered a *general* pattern and not simply memorized our data.
For example, imagine that we wanted to hunt for patterns among genetic markers linking patients to their dementia status, 
(let's the labels be drawn from the set {*dementia*, *mild cognitive impairment*, *healthy*}).
Because each person's genes identify them uniquely (ignoring identical siblings), it's possible to memorize the entire dataset.
-->

Là những nhà khoa học học máy, mục tiêu của chúng ta là khám phá ra các *khuôn mẫu*.
Nhưng làm sao có thể chắc chắn rằng chúng ta đã thực sự khám phá ra một khuôn mẫu *khái quát* chứ không chỉ đơn giản là ghi nhớ dữ liệu.
Ví dụ, thử tưởng tượng rằng chúng ta muốn săn lùng các khuôn mẫu liên kết các dấu hiệu di truyền của bệnh nhân và tình trạng mất trí của họ, với nhãn được trích ra từ tập {*mất trí nhớ*, *suy giảm nhận thức mức độ nhẹ*, *khỏe mạnh*}.
Bởi vì các gen của mỗi người định dạng họ theo cách độc nhất vô nhị (bỏ qua các cặp song sinh giống hệt nhau), nên việc ghi nhớ toàn bộ tập dữ liệu là hoàn toàn khả thi.

<!--
We don't want our model to say *"That's Bob! I remember him! He has dementia!*
The reason why is simple.
When we deploy the model in the future, we will encounter patients that the model has never seen before.
Our predictions will only be useful if our model has truly discovered a *general* pattern.
-->

Chúng ta không muốn mô hình của mình nói rằng *"Bob kìa! Tôi nhớ anh ta! Anh ta bị mất trí nhớ!*
Lý do tại sao rất đơn giản.
Khi triển khai mô hình trong tương lai, chúng ta sẽ gặp các bệnh nhân mà mô hình chưa bao giờ gặp trước đó.
Các dự đoán sẽ chỉ có ích khi mô hình của chúng ta thực sự khám phá ra một khuôn mẫu *khái quát*.

<!--
To recapitulate more formally, our goal is to discover patterns that capture regularities in the underlying population from which our training set was drawn.
If we are successful in this endeavor, then we could successfully assess risk even for individuals that we have never encountered before.
This problem---how to discover patterns that *generalize*---is the fundamental problem of machine learning.
-->

Để tóm tắt một cách chính thức hơn, mục tiêu của chúng ta là khám phá các khuôn mẫu mà chúng mô tả được các quy tắc trong tập dữ liệu mà từ đó tập huấn luyện đã được trích ra.
Nếu thành công trong nỗ lực này, thì chúng ta có thể đánh giá thành công rủi ro ngay cả đối với các cá nhân mà chúng ta chưa bao giờ gặp phải trước đây.
Vấn đề này---làm cách nào để khám phá ra các mẫu mà *khái quát hóa*---là vấn đề nền tảng của học máy.

<!--
The danger is that when we train models, we access just a small sample of data.
The largest public image datasets contain roughly one million images.
More often, we must learn from only thousands or tens of thousands of data points.
In a large hospital system, we might access hundreds of thousands of medical records.
When working with finite samples, we run the risk that we might discover *apparent* associations that turn out not to hold up when we collect more data.
-->

Nguy hiểm là khi huấn luyện các mô hình, chúng ta chỉ truy cập một tập dữ liệu nhỏ.
Các tập dữ liệu hình ảnh công khai lớn nhất chứa khoảng một triệu ảnh.
Thường thì chúng ta phải học chỉ từ vài ngàn hoặc vài chục ngàn điểm dữ liệu.
Trong một hệ thống bệnh viện lớn, chúng ta có thể truy cập hàng trăm ngàn hồ sơ y tế.
Khi làm việc với các tập mẫu hữu hạn, chúng ta gặp phải rủi ro sẽ khám phá ra các mối liên kết *rõ ràng* mà hóa ra lại không đúng khi thu thập thêm dữ liệu.

<!--
The phenomena of fitting our training data more closely than we fit the underlying distribution is called overfitting, and the techniques used to combat overfitting are called regularization.
In the previous sections, you might have observed this effect while experimenting with the Fashion-MNIST dataset.
If you altered the model structure or the hyper-parameters during the experiment, 
you might have noticed that with enough nodes, layers, and training epochs, the model can eventually reach perfect accuracy on the training set, even as the accuracy on test data deteriorates.
-->

Hiện tượng mô hình khớp với dữ liệu huấn luyện chính xác hơn nhiều so với phân phối thực được gọi là quá khớp (*overfitting*), và kỹ thuật sử dụng để chống lại quá khớp được gọi là điều chuẩn (*regularization*).
Trong các phần trước, bạn có thể đã quan sát hiệu ứng này khi thử nghiệm với tập dữ liệu Fashion-MNIST.
Nếu bạn đã sửa đổi cấu trúc mô hình hoặc siêu tham số trong quá trình thử nghiệm, bạn có thể đã nhận ra rằng với đủ các nút, các tầng, và các epoch huấn luyện, mô hình ấy có thể cuối cùng cũng đạt đến sự chính xác hoàn hảo trên tập huấn luyện, ngay cả khi độ chính xác trên dữ liệu kiểm tra giảm đi.

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->


<!--
## Training Error and Generalization Error
-->

## Lỗi huấn luyện và Lỗi khái quát

<!--
In order to discuss this phenomenon more formally, we need to differentiate between *training error* and *generalization error*.
The training error is the error of our model as calculated on the training dataset, 
while generalization error is the expectation of our model's error 
were we to apply it to an infinite stream of additional data points drawn from the same underlying data distribution as our original sample.
-->

Để thảo luận hiện tượng này một cách chuyên sâu hơn, ta cần phân biệt giữa *lỗi huấn luyện* (*training error*) và *lỗi khái quát* (*generalization error*).
Lỗi huấn luyện là lỗi của mô hình được tính toán trên tập huấn luyện, trong khi đó lỗi khái quát là lỗi kỳ vọng của mô hình khi áp dụng nó cho một luồng vô hạn các điểm dữ liệu mới được lấy từ cùng một phân phối dữ liệu với các mẫu ban đầu.

<!--
Problematically, *we can never calculate the generalization error exactly*.
That is because the imaginary stream of infinite data is an imaginary object.
In practice, we must *estimate* the generalization error by applying our model to an independent test set 
constituted of a random selection of data points that were withheld from our training set.
-->

Vấn đề là *chúng ta không bao giờ có thể tính toán chính xác lỗi khái quát* vì luồng vô hạn dữ liệu chỉ có trong tưởng tượng.
Trên thực tế, ta phải *ước tính* lỗi khái quát bằng cách áp dụng mô hình vào một tập kiểm tra độc lập bao gồm các điểm dữ liệu ngẫu nhiên ngoài tập huấn luyện.

<!--
The following three thought experiments will help illustrate this situation better.
Consider a college student trying to prepare for his final exam.
A diligent student will strive to practice well and test her abilities using exams from previous years.
Nonetheless, doing well on past exams is no guarantee that she will excel when it matters.
For instance, the student might try to prepare by rote learning the answers to the exam questions.
This requires the student to memorize many things.
She might even remember the answers for past exams perfectly.
Another student might prepare by trying to understand the reasons for giving certain answers.
In most cases, the latter student will do much better.
-->

Ba thí nghiệm sau sẽ giúp minh họa tình huống này tốt hơn.
Hãy xem xét một sinh viên đại học đang cố gắng chuẩn bị cho kỳ thi cuối cùng của mình.
Một sinh viên chăm chỉ sẽ cố gắng luyện tập tốt và kiểm tra khả năng của cô ấy bằng việc luyện tập những bài kiểm tra của các năm trước.
Tuy nhiên, làm tốt các bài kiểm tra trước đây không đảm bảo rằng cô ấy sẽ làm tốt bài kiểm tra thật.
Ví dụ, sinh viên có thể cố gắng chuẩn bị bằng cách học tủ các câu trả lời cho các câu hỏi.
Điều này đòi hỏi sinh viên phải ghi nhớ rất nhiều thứ.
Cô ấy có lẽ còn ghi nhớ đáp án cho các bài kiểm tra cũ một cách hoàn hảo. 
Một học sinh khác có thể chuẩn bị bằng việc cố gắng hiểu lý do mà một số đáp án nhất định được đưa ra.
Trong hầu hết các trường hợp, sinh viên sau sẽ làm tốt hơn nhiều. 

<!--
Likewise, consider a model that simply uses a lookup table to answer questions. 
If the set of allowable inputs is discrete and reasonably small, then perhaps after viewing *many* training examples, this approach would perform well. 
Still this model has no ability to do better than random guessing when faced with examples that it has never seen before.
In reality the input spaces are far too large to memorize the answers corresponding to every conceivable input. 
For example, consider the black and white $28\times28$ images. 
If each pixel can take one among $256$ gray scale values, then there are $256^{784}$ possible images. 
That means that there are far more low-res grayscale thumbnail-sized images than there are atoms in the universe. 
Even if we could encounter this data, we could never afford to store the lookup table.
-->

Tương tự như vậy, hãy xem xét một mô hình đơn giản chỉ sử dụng một bảng tra cứu để trả lời các câu hỏi. 
Nếu tập hợp các đầu vào cho phép là rời rạc và đủ nhỏ, thì có lẽ sau khi xem *nhiều* ví dụ huấn luyện, phương pháp này sẽ hoạt động tốt. 
Tuy nhiên mô hình này không có khả năng thể hiện tốt hơn so với việc đoán ngẫu nhiên khi phải đối mặt với các ví dụ chưa từng gặp trước đây.
Trong thực tế, không gian đầu vào là quá lớn để có thể ghi nhớ mọi đáp án tương ứng của từng đầu vào khả dĩ.
Ví dụ, hãy xem xét các ảnh $28\times28$ đen trắng.
Nếu mỗi điểm ảnh có thể lấy một trong số các giá trị xám trong thang $256$, thì có thể có $256^{784}$ ảnh khác nhau.
Điều đó nghĩa là số lượng ảnh độ phân giải thấp còn lớn hơn nhiều so với số lượng nguyên tử trong vũ trụ.
Thậm chí nếu có thể xem qua toàn bộ điểm dữ liệu, ta cũng không thể lưu trữ chúng trong bảng tra cứu.

<!--
Last, consider the problem of trying to classify the outcomes of coin tosses (class 0: heads, class 1: tails) based on some contextual features that might be available.
No matter what algorithm we come up with, because the generalization error will always be $\frac{1}{2}$.
However, for most algorithms, we should expect our training error to be considerably lower, depending on the luck of the draw, even if we did not have any features!
Consider the dataset {0, 1, 1, 1, 0, 1}.
Our feature-less would have to fall back on always predicting the *majority class*, which appears from our limited sample to be *1*.
In this case, the model that always predicts class 1 will incur an error of $\frac{1}{3}$, considerably better than our generalization error.
As we increase the amount of data, the probability that the fraction of heads will deviate significantly from $\frac{1}{2}$ diminishes, and our training error would come to match the generalization error.
-->

Cuối cùng, hãy xem xét bài toán phân loại kết quả của việc tung đồng xu (lớp 0: ngửa, lớp 1: xấp) dựa trên một số đặc trưng theo ngữ cảnh sẵn có.
Bất kể thuật toán nào được đưa ra, lỗi khái quát sẽ luôn là $\frac{1}{2}$.
Tuy nhiên, đối với hầu hết các thuật toán, lỗi huấn luyện sẽ thấp hơn đáng kể, tùy thuộc vào sự may mắn của ta khi lấy dữ liệu, ngay cả khi ta không có bất kỳ đặc trưng nào!
Hãy xem xét tập dữ liệu {0, 1, 1, 1, 0, 1}.
Việc không có đặc trưng có thể khiến ta luôn dự đoán *lớp chiếm đa số*, đối với ví dụ này thì đó là *1*.
Trong trường hợp này, mô hình luôn dự đoán lớp 1 sẽ có lỗi huấn luyện là $\frac{1}{3}$, tốt hơn đáng kể so với lỗi khái quát.
Khi ta tăng lượng dữ liệu, xác suất nhận được mặt ngửa sẽ dần tiến về $\frac{1}{2}$ và lỗi huấn luyện sẽ tiến đến lỗi khái quát.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Statistical Learning Theory
-->

### Lý thuyết Học Thống kê

<!--
Since generalization is the fundamental problem in machine learning, you might not be surprised to learn 
that many mathematicians and theorists have dedicated their lives to developing formal theories to describe this phenomenon.
In their [eponymous theorem](https://en.wikipedia.org/wiki/Glivenko–Cantelli_theorem), Glivenko and Cantelli derived the rate at which the training error converges to the generalization error.
In a series of seminal papers, [Vapnik and Chervonenkis](https://en.wikipedia.org/wiki/Vapnik–Chervonenkis_theory) extended this theory to more general classes of functions.
This work laid the foundations of [Statistical Learning Theory](https://en.wikipedia.org/wiki/Statistical_learning_theory).
-->

Bởi khái quát hóa là một vấn đề nền tảng trong học máy, không quá ngạc nhiên khi nhiều nhà toán học và nhà lý thuyết học dành cả cuộc đời để phát triển các lý thuyết hình thức mô tả vấn đề này.
Trong [định lý cùng tên](https://en.wikipedia.org/wiki/Glivenko–Cantelli_theorem), Glivenko và Cantelli đã tìm ra tốc độ học mà tại đó lỗi huấn luyện sẽ hội tụ về lỗi khái quát.
Trong chuỗi các bài báo đầu ngành, [Vapnik và Chervonenkis](https://en.wikipedia.org/wiki/Vapnik–Chervonenkis_theory) đã mở rộng lý thuyết này cho nhiều lớp hàm tổng quát hơn.
Công trình này là nền tảng của ngành [Lý thuyết học thống kê](https://en.wikipedia.org/wiki/Statistical_learning_theory).


<!--
In the *standard supervised learning setting*, which we have addressed up until now and will stick throughout most of this book,
we assume that both the training data and the test data are drawn *independently* from *identical* distributions (commonly called the i.i.d. assumption).
This means that the process that samples our data has no *memory*.
The $2^{\mathrm{nd}}$ example drawn and the $3^{\mathrm{rd}}$ drawn are no more correlated than the $2^{\mathrm{nd}}$ and the $2$-millionth sample drawn.
-->

Trong một *thiết lập chuẩn cho học có giám sát* -- chủ đề lớn nhất xuyên suốt cuốn sách, chúng ta giả sử rằng cả dữ liệu huấn luyện và dữ liệu kiểm tra đều được lấy mẫu *độc lập* từ các phân phối *giống hệt* nhau (*independent & identically distributed*, thường gọi là giả thiết i.i.d.).
Điều này có nghĩa là quá trình lấy mẫu dữ liệu không hề có sự *ghi nhớ*.
Mẫu lấy ra thứ hai cũng không tương quan với mẫu thứ ba hơn so với mẫu thứ hai triệu.

<!--
Being a good machine learning scientist requires thinking critically, and already you should be poking holes in this assumption, coming up with common cases where the assumption fails.
What if we train a mortality risk predictor on data collected from patients at UCSF, and apply it on patients at Massachusetts General Hospital?
These distributions are simply not identical.
Moreover, draws might be correlated in time.
What if we are classifying the topics of Tweets.
The news cycle would create temporal dependencies in the topics being discussed violating any assumptions of independence.
-->

Trở thành một nhà khoa học học máy giỏi yêu cầu tư duy phản biện, và có lẽ bạn đã có thể "bóc mẽ" được giả thiết này, có thể đưa ra các tình huống thường gặp mà giả thiết này không thỏa mãn.
Điều gì sẽ xảy ra nếu chúng ta huấn luyện một mô hình dự đoán tỉ lệ tử vong trên bộ dữ thu thập từ các bệnh nhân tại UCSF, và áp dụng nó trên các bệnh nhân tại Bệnh viện Đa khoa Massachusetts.
Các phân phối này đơn giản là không giống nhau.
Hơn nữa, việc lấy mẫu có thể có tương quan về mặt thời gian.
Sẽ ra sao nếu chúng ta thực hiện phân loại chủ đề cho các bài Tweet.
Vòng đời của các tin tức sẽ tạo nên sự phụ thuộc về mặt thời gian giữa các chủ đề được đề cập, vi phạm mọi giả định độc lập thống kê.

<!--
Sometimes we can get away with minor violations of the i.i.d. assumption and our models will continue to work remarkably well.
After all, nearly every real-world application involves at least some minor violation of the i.i.d. assumption, and yet we have useful tools for face recognition, speech recognition, language translation, etc.
-->

Đôi khi, chúng ta có thể bỏ qua một vài vi phạm nhỏ trong giả thiết i.i.d. mà mô hình vẫn có thể làm việc rất tốt.
Nhìn chung, gần như tất cả các ứng dụng thực tế đều vi phạm một vài giả thiết i.i.d. nhỏ, nhưng đổi lại ta có được các công cụ rất hữu dụng như nhận dạng khuôn mặt, nhận dạng tiếng nói, dịch ngôn ngữ, v.v.

<!--
Other violations are sure to cause trouble.
Imagine, for example, if we tried to train a face recognition system by training it exclusively on university students and then want to deploy it as a tool for monitoring geriatrics in a nursing home population.
This is unlikely to work well since college students tend to look considerably different from the elderly.
-->

Các vi phạm khác thì chắc chắn dẫn tới rắc rối.
Cùng hình dung ở ví dụ này, ta thử huấn luyện một hệ thống nhận dạng khuôn mặt sử dụng hoàn toàn dữ liệu của các sinh viên đại học và đem đi triển khai như một công cụ giám sát trong viện dưỡng lão.
Cách này gần như không khả thi vì ngoại hình giữa hai độ tuổi quá khác biệt.

<!--
In subsequent chapters and volumes, we will discuss problems arising from violations of the i.i.d. assumption.
For now, even taking the i.i.d. assumption for granted, understanding generalization is a formidable problem.
Moreover, elucidating the precise theoretical foundations that might explain why deep neural networks generalize as well as they do continues to vexes the greatest minds in learning theory.
-->

Trong các mục và chương kế tiếp, chúng ta sẽ đề cập tới các vấn đề gặp phải khi vi phạm giả thiết i.i.d.
Hiện tại khi giả thiết i.i.d. thậm chí được đảm bảo, hiểu được sự khái quát hóa cũng là một vấn đề nan giải.
Hơn nữa, việc làm sáng tỏ nền tảng lý thuyết để giải thích tại sao các mạng nơ-ron sâu có thể khái quát hóa tốt như vậy vẫn tiếp tục làm đau đầu những bộ óc vĩ đại nhất trong lý thuyết học.

<!--
When we train our models, we attempt searching for a function that fits the training data as well as possible.
If the function is so flexible that it can catch on to spurious patterns just as easily as to the true associations, 
then it might perform *too well* without producing a model that generalizes well to unseen data.
This is precisely what we want to avoid (or at least control).
Many of the techniques in deep learning are heuristics and tricks aimed at guarding against overfitting.
-->

Khi huấn luyện mô hình, ta đang cố gắng tìm kiếm một hàm số khớp với dữ liệu huấn luyện nhất có thể.
Nếu hàm số này quá linh hoạt để có thể khớp với các khuôn mẫu giả cũng dễ như với các xu hướng thật trong dữ liệu, thì nó có thể *quá khớp* để có thể tạo ra một mô hình có tính khái quát hóa cao trên dữ liệu chưa nhìn thấy.
Đây chính xác là những gì chúng ta muốn tránh (hay ít nhất là kiểm soát được).
Rất nhiều kỹ thuật trong học sâu là các phương pháp dựa trên thực nghiệm và thủ thuật để chống lại vấn đề quá khớp.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
### Model Complexity
-->

### Độ Phức tạp của Mô hình

<!--
When we have simple models and abundant data, we expect the generalization error to resemble the training error.
When we work with more complex models and fewer examples, we expect the training error to go down but the generalization gap to grow.
What precisely constitutes model complexity is a complex matter.
Many factors govern whether a model will generalize well.
For example a model with more parameters might be considered more complex.
A model whose parameters can take a wider range of values might be more complex.
Often with neural networks, we think of a model that takes more training steps as more complex, and one subject to *early stopping* as less complex.
-->

Khi có các mô hình đơn giản và dữ liệu dồi dào, ta kỳ vọng lỗi khái quát sẽ giống với lỗi huấn luyện.
Khi làm việc với mô hình phức tạp hơn và ít mẫu huấn luyện hơn, ta kỳ vọng các lỗi huấn luyện giảm xuống nhưng khoảng cách khái quát tăng.
Việc chỉ ra chính xác điều gì cấu thành nên độ phức tạp của mô hình là một vấn đề nan giải. 
Có rất nhiều yếu tố ảnh hưởng đến việc một mô hình có khái quát hóa tốt hay không. 
Ví dụ một mô hình với nhiều tham số hơn sẽ được xem là phức tạp hơn.
Một mô hình mà các tham số có miền giá trị rộng hơn thì được xem là phức tạp hơn.
Thông thường với các mạng nơ-ron, ta nghĩ đến một mô hình có nhiều bước huấn luyện là mô hình phức tạp hơn, và mô hình *dừng sớm* là ít phức tạp hơn.

<!--
It can be difficult to compare the complexity among members of substantially different model classes (say a decision tree versus a neural network).
For now, a simple rule of thumb is quite useful:
A model that can readily explain arbitrary facts is what statisticians view as complex, 
whereas one that has only a limited expressive power but still manages to explain the data well is probably closer to the truth.
In philosophy, this is closely related to Popper’s criterion of [falsifiability](https://en.wikipedia.org/wiki/Falsifiability) of a scientific theory: 
a theory is good if it fits data and if there are specific tests which can be used to disprove it.
This is important since all statistical estimation is [post hoc](https://en.wikipedia.org/wiki/Post_hoc), i.e., we estimate after we observe the facts, hence vulnerable to the associated fallacy.
For now, we will put the philosophy aside and stick to more tangible issues.
-->

Rất khó để có thể so sánh sự phức tạp giữa các thành viên trong các lớp mô hình khác hẳn nhau (ví như cây quyết định so với mạng nơ-ron).
Hiện tại, có một quy tắc đơn giản khá hữu ích sau:
Một mô hình có thể giải thích các sự kiện bất kỳ thì được các nhà thống kê xem là phức tạp, trong khi một mô hình với năng lực biểu diễn giới hạn nhưng vẫn có thể giải thích tốt được dữ liệu thì hầu như chắc chắn là đúng đắn hơn.
Trong triết học, điều này gần với tiêu chí của Popper về [khả năng phủ định](https://en.wikipedia.org/wiki/Falsifiability) của một lý thuyết khoa học: một lý thuyết tốt nếu nó khớp dữ liệu và nếu có các kiểm định cụ thể có thể dùng để phản chứng nó.
Điều này quan trọng bởi vì tất cả các ước lượng thống kê là [post hoc](https://en.wikipedia.org/wiki/Post_hoc), tức là ta đánh giá giả thuyết sau khi quan sát các sự thật, do đó dễ bị tác động bởi lỗi ngụy biện cùng tên.
Từ bây giờ, ta sẽ đặt triết lý qua một bên và tập trung hơn vào các vấn đề hữu hình.

<!--
In this section, to give you some intuition, we’ll focus on a few factors that tend to influence the generalizability of a model class:
-->

Trong phần này, để có cái nhìn trực quan, chúng ta sẽ tập trung vào một vài yếu tố có xu hướng ảnh hưởng đến tính khái quát của một lớp mô hình:

<!--
1. The number of tunable parameters. When the number of tunable parameters, sometimes called the *degrees of freedom*, is large, models tend to be more susceptible to overfitting.
2. The values taken by the parameters. When weights can take a wider range of values, models can be more susceptible to over fitting.
3. The number of training examples. It’s trivially easy to overfit a dataset containing only one or two examples even if your model is simple. 
But overfitting a dataset with millions of examples requires an extremely flexible model.
-->

1. Số lượng các tham số có thể điều chỉnh. Khi số lượng các tham số có thể điều chỉnh (đôi khi được gọi là *bậc tự do*) lớn thì mô hình sẽ dễ bị quá khớp hơn.
2. Các giá trị được nhận bởi các tham số. Khi các trọng số có miền giá trị rộng hơn, các mô hình dễ bị quá khớp hơn.
3. Số lượng các mẫu huấn luyện. Việc quá khớp một tập dữ liệu chứa chỉ một hoặc hai mẫu rất dễ dàng, kể cả khi mô hình đơn giản.
Nhưng quá khớp một tập dữ liệu với vài triệu mẫu đòi hỏi mô hình phải cực kỳ linh hoạt.

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->


<!--
## Model Selection
-->

## Lựa chọn Mô hình

<!--
In machine learning, we usually select our final model after evaluating several candidate models.
This process is called model selection.
Sometimes the models subject to comparison are fundamentally different in nature (say, decision trees vs linear models).
At other times, we are comparing members of the same class of models that have been trained with different hyperparameter settings.
-->

Trong học máy, ta thường lựa chọn mô hình cuối cùng sau khi cân nhắc nhiều mô hình ứng viên.
Quá trình này được gọi là lựa chọn mô hình.
Đôi khi các mô hình được đem ra so sánh khác nhau cơ bản về mặt bản chất (ví như, cây quyết định với các mô hình tuyến tính). 
Khi khác, ta lại so sánh các thành viên của cùng một lớp mô hình được huấn luyện với các cài đặt siêu tham số khác nhau.

<!--
With multilayer perceptrons for example, we may wish to compare models with different numbers of hidden layers, 
different numbers of hidden units, and various choices of the activation functions applied to each hidden layer.
In order to determine the best among our candidate models, we will typically employ a validation set.
-->

Lấy perceptron đa tầng làm ví dụ, ta mong muốn so sánh các mô hình với số lượng tầng ẩn khác nhau, số lượng nút ẩn khác nhau, và các lựa chọn hàm kích hoạt khác nhau áp dụng vào từng tầng ẩn.
Để xác định được mô hình tốt nhất trong các mô hình ứng viên, ta thường sử dụng một tập kiểm định.

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Validation Dataset
-->

### Tập Dữ liệu Kiểm định

<!--
In principle we should not touch our test set until after we have chosen all our hyper-parameters.
Were we to use the test data in the model selection process, there is a risk that we might overfit the test data.
Then we would be in serious trouble.
If we overfit our training data, there is always the evaluation on test data to keep us honest.
But if we overfit the test data, how would we ever know?
-->

Về nguyên tắc, ta không nên sử dụng tập kiểm tra cho đến khi chọn xong tất cả các siêu tham số.
Nếu sử dụng dữ liệu kiểm tra trong quá trình lựa chọn mô hình, có một rủi ro là ta có thể quá khớp dữ liệu kiểm tra, và khi đó ta sẽ gặp rắc rối lớn.
Nếu quá khớp dữ liệu huấn luyện, ta luôn có thể đánh giá mô hình trên tập kiểm tra để đảm bảo mình "trung thực".
Nhưng nếu quá khớp trên dữ liệu kiểm tra, làm sao chúng ta có thể biết được?


<!--
Thus, we should never rely on the test data for model selection.
And yet we cannot rely solely on the training data for model selection either because we cannot estimate the generalization error on the very data that we use to train the model.
-->

Vì vậy, ta không bao giờ nên dựa vào dữ liệu kiểm tra để lựa chọn mô hình.
Tuy nhiên, không thể chỉ dựa vào dữ liệu huấn luyện để lựa chọn mô hình vì ta không thể ước tính lỗi khái quát trên chính dữ liệu được sử dụng để huấn luyện mô hình.

<!--
The common practice to address this problem is to split our data three ways, incorporating a *validation set* in addition to the training and test sets.
-->

Phương pháp phổ biến để giải quyết vấn đề này là phân chia dữ liệu thành ba phần, thêm một *tập kiểm định* ngoài các tập huấn luyện và kiểm tra.


<!--
In practical applications, the picture gets muddier.
While ideally we would only touch the test data once, to assess the very best model or to compare a small number of models to each other, real-world test data is seldom discarded after just one use.
We can seldom afford a new test set for each round of experiments.
-->

Trong các ứng dụng thực tế, bức tranh trở nên mập mờ hơn.
Mặc dù tốt nhất ta chỉ nên động đến dữ liệu kiểm tra đúng một lần, để đánh giá mô hình tốt nhất hoặc so sánh một số lượng nhỏ các mô hình với nhau, dữ liệu kiểm tra trong thế giới thực hiếm khi bị vứt bỏ chỉ sau một lần sử dụng.
Ta hiếm khi có được một tập kiểm tra mới sau mỗi vòng thử nghiệm.

<!--
The result is a murky practice where the boundaries between validation and test data are worryingly ambiguous.
Unless explicitly stated otherwise, in the experiments in this book we are really working with what should rightly be called training data and validation data, with no true test sets.
Therefore, the accuracy reported in each experiment is really the validation accuracy and not a true test set accuracy.
The good news is that we do not need too much data in the validation set.
The uncertainty in our estimates can be shown to be of the order of $\mathcal{O}(n^{-\frac{1}{2}})$.
-->

Kết quả là một thực tiễn âm u trong đó ranh giới giữa dữ liệu kiểm định và kiểm tra mơ hồ theo cách đáng lo ngại.
Trừ khi có quy định rõ ràng thì, trong các thí nghiệm trong cuốn sách này, ta thật sự đang làm việc với cái được gọi là dữ liệu huấn luyện và dữ liệu kiểm định chứ không có tập kiểm tra thật.
Do đó, độ chính xác được báo cáo trong mỗi thử nghiệm thật ra là độ chính xác kiểm định và không phải là độ chính xác của tập kiểm tra thật.
Tin tốt là ta không cần quá nhiều dữ liệu trong tập kiểm định.
Ta có thể chứng minh rằng sự bất định trong các ước tính thuộc bậc $\mathcal{O}(n^{-\frac{1}{2}})$.

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!--
### $K$-Fold Cross-Validation
-->

### Kiểm định chéo gập $K$-lần

<!--
When training data is scarce, we might not even be able to afford to hold out enough data to constitute a proper validation set.
One popular solution to this problem is to employ $K$*-fold cross-validation*.
Here, the original training data is split into $K$ non-overlapping subsets.
Then model training and validation are executed $K$ times, each time training on $K-1$ subsets and validating on a different subset (the one not used for training in that round).
Finally, the training and validation error rates are estimated by averaging over the results from the $K$ experiments.
-->

Khi khan hiếm dữ liệu huấn luyện, có lẽ ta sẽ không thể dành ra đủ dữ liệu để tạo một tập kiểm định phù hợp.
Một giải pháp phổ biến để giải quyết vấn đề này là kiểm định chéo gập $K$-lần.
Ở phương pháp này, tập dữ liệu huấn luyện ban đầu được chia thành $K$ tập con không chồng lên nhau.
Sau đó việc huấn luyện và kiểm định mô hình được thực thi $K$ lần, mỗi lần huấn luyện trên $K-1$ tập con và kiểm định trên tập con còn lại (tập không được sử dụng để huấn luyện trong lần đó).
Cuối cùng, lỗi huấn luyện và lỗi kiểm định được ước lượng bằng cách tính trung bình các kết quả thu được từ $K$ thí nghiệm.



<!--
## Underfitting or Overfitting?
-->

## Dưới khớp hay Quá khớp?

<!--
When we compare the training and validation errors, we want to be mindful of two common situations:
First, we want to watch out for cases when our training error and validation error are both substantial but there is a little gap between them.
If the model is unable to reduce the training error, that could mean that our model is too simple (i.e., insufficiently expressive) to capture the pattern that we are trying to model.
Moreover, since the *generalization gap* between our training and validation errors is small, we have reason to believe that we could get away with a more complex model.
This phenomenon is known as underfitting.
-->

Khi so sánh lỗi huấn luyện và lỗi kiểm định, ta cần lưu ý hai trường hợp thường gặp sau:
Đầu tiên, ta sẽ muốn chú ý trường hợp lỗi huấn luyện và lỗi kiểm định đều lớn nhưng khoảng cách giữa chúng lại nhỏ.
Nếu mô hình không thể giảm thiểu lỗi huấn luyện, điều này có nghĩa là mô hình quá đơn giản (tức không đủ khả năng biểu diễn) để có thể xác định được khuôn mẫu mà ta đang cố mô hình hóa.
Hơn nữa, do khoảng cách khái quát giữa lỗi huấn luyện và lỗi kiểm định nhỏ, ta có lý do để tin rằng phương án giải quyết là một mô hình phức tạp hơn.
Hiện tượng này được gọi là dưới khớp (*underfitting*).

<!--
On the other hand, as we discussed above, we want to watch out for the cases when our training error is significantly lower than our validation error, indicating severe overfitting.
Note that overfitting is not always a bad thing.
With deep learning especially, it is well known that the best predictive models often perform far better on training data than on holdout data.
Ultimately, we usually care more about the validation error than about the gap between the training and validation errors.
-->

Mặt khác, như ta đã thảo luận ở phía trên, ta cũng muốn chú ý tới trường hợp lỗi huấn luyện thấp hơn lỗi kiểm định một cách đáng kể, một biểu hiện của sự quá khớp nặng.
Lưu ý rằng quá khớp không phải luôn là điều xấu.
Đặc biệt là với học sâu, ta đều biết rằng mô hình dự đoán tốt nhất thường đạt chất lượng tốt hơn hẳn trên dữ liệu huấn luyện so với dữ liệu kiểm định.
Sau cùng, ta thường quan tâm đến lỗi kiểm định hơn khoảng cách giữa lỗi huấn luyện và lỗi kiểm định.

<!--
Whether we overfit or underfit can depend both on the complexity of our model and the size of the available training datasets, two topics that we discuss below.
-->

Việc ta đang quá khớp hay dưới khớp có thể phụ thuộc vào cả độ phức tạp của mô hình lẫn kích thước của tập dữ liệu huấn luyện có sẵn, và hai vấn đề này sẽ được thảo luận ngay sau đây.  

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
### Model Complexity
-->

### Độ phức tạp Mô hình

<!--
To illustrate some classical intuition about overfitting and model complexity, we given an example using polynomials.
Given training data consisting of a single feature $x$ and a corresponding real-valued label $y$, we try to find the polynomial of degree $d$
-->

Để có thể hình dung một cách trực quan hơn về mối quan hệ giữa quá khớp và độ phức tạp mô hình, ta sẽ đưa ra một ví dụ sử dụng đa thức.
Cho một tập dữ liệu huấn luyện có một đặc trưng duy nhất $x$ và nhãn $y$ tương ứng có giá trị thực, ta thử tìm bậc $d$ của đa thức

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

<!--
to estimate the labels $y$.
This is just a linear regression problem where our features are given by the powers of $x$, the $w_i$ given the model’s weights, and the bias is given by $w_0$ since $x^0 = 1$ for all $x$.
Since this is just a linear regression problem, we can use the squared error as our loss function.
-->

để ước tính nhãn $y$.
Đây đơn giản là một bài toán hồi quy tuyến tính trong đó các đặc trưng được tính bằng cách lấy mũ của $x$, $w_i$ là trọng số của mô hình, vì $x^0=1$ với mọi $x$ nên $w_0$ là hệ số điều chỉnh. 
Vì đây là bài toán hồi quy tuyến tính, ta có thể sử dụng bình phương sai số làm hàm mất mát.

<!--
A higher-order polynomial function is more complex than a lower order polynomial function, since the higher-order polynomial has more parameters and the model function’s selection range is wider.
Fixing the training dataset, higher-order polynomial functions should always achieve lower (at worst, equal) training error relative to lower degree polynomials.
In fact, whenever the data points each have a distinct value of $x$, a polynomial function with degree equal to the number of data points can fit the training set perfectly.
We visualize the relationship between polynomial degree and under- vs over-fitting in :numref:`fig_capacity_vs_error`.
-->

Hàm đa thức bậc cao phức tạp hơn hàm đa thức bậc thấp, vì đa thức bậc cao có nhiều tham số hơn và miền lựa chọn hàm số cũng rộng hơn.
Nếu giữ nguyên tập dữ liệu huấn luyện, các hàm đa thức bậc cao hơn sẽ luôn đạt được lỗi huấn luyện thấp hơn (ít nhất là bằng) so với đa thức bậc thấp hơn.
Trong thực tế, nếu mọi điểm dữ liệu có các giá trị $x$ riêng biệt, một hàm đa thức có bậc bằng với số điểm dữ liệu đều có thể khớp một cách hoàn hảo với tập huấn luyện. 
Mối quan hệ giữa bậc của đa thức với hai hiện tượng dưới khớp và quá khớp được biểu diễn trong :numref:`fig_capacity_vs_error`.

<!--
![Influence of Model Complexity on Underfitting and Overfitting](../img/capacity_vs_error.svg)
-->

![Ảnh hưởng của Độ phức tạp Mô hình tới Dưới khớp và Quá khớp](../img/capacity_vs_error.svg)
:label:`fig_capacity_vs_error`

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Dataset Size
-->

### Kích thước Tập dữ liệu

<!--
The other big consideration to bear in mind is the dataset size.
Fixing our model, the fewer samples we have in the training dataset, the more likely (and more severely) we are to encounter overfitting.
As we increase the amount of training data, the generalization error typically decreases.
Moreover, in general, more data never hurts.
For a fixed task and data *distribution*, there is typically a relationship between model complexity and dataset size.
Given more data, we might profitably attempt to fit a more complex model.
Absent sufficient data, simpler models may be difficult to beat.
For many tasks, deep learning only outperforms linear models when many thousands of training examples are available.
In part, the current success of deep learning owes to the current abundance of massive datasets due to Internet companies, cheap storage, connected devices, and the broad digitization of the economy.
-->

Một lưu ý quan trọng khác cần ghi nhớ là kích thước tập dữ liệu.
Với một mô hình cố định, tập dữ liệu càng ít mẫu thì càng có nhiều khả năng gặp phải tình trạng quá khớp với mức độ nghiêm trọng hơn.
Khi số lượng dữ liệu tăng lên, lỗi khái quát sẽ có xu hướng giảm.
Hơn nữa, trong hầu hết các trường hợp, nhiều dữ liệu không bao giờ là thừa.
Trong một tác vụ với một *phân phối* dữ liệu cố định, ta có thể quan sát được mối quan hệ giữa độ phức tạp của mô hình và kích thước tập dữ liệu.
Khi có nhiều dữ liệu, thử khớp một mô hình phức tạp hơn thường sẽ mang lợi nhiều lợi ích.
Khi dữ liệu không quá nhiều, mô hình đơn giản sẽ là lựa chọn tốt hơn.
Đối với nhiều tác vụ, học sâu chỉ tốt hơn các mô hình tuyến tính khi có sẵn hàng ngàn mẫu huấn luyện.
Sự thành công hiện nay của học sâu phần nào dựa vào sự phong phú của các tập dữ liệu khổng lồ từ các công ty hoạt động trên internet, từ các thiết bị lưu trữ giá rẻ, các thiết bị được nối mạng và rộng hơn là việc số hóa nền kinh tế.

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
## Polynomial Regression
-->

## Hồi quy Đa thức 

<!--
We can now explore these concepts interactively by fitting polynomials to data.
To get started we will import our usual packages.
-->

Bây giờ ta có thể khám phá một cách tương tác những khái niệm này bằng cách khớp đa thức với dữ liệu.
Để bắt đầu ta sẽ nhập các gói thư viện thường dùng.

```{.python .input  n=1}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
npx.set_np()
```

<!--
### Generating the Dataset
-->

### Tạo ra Tập dữ liệu

<!--
First we need data. Given $x$, we will use the following cubic polynomial to generate the labels on training and test data:
-->

Đầu tiên ta cần dữ liệu. Cho $x$, ta sẽ sử dụng đa thức bậc ba ở dưới đây để tạo nhãn cho tập dữ liệu huấn luyện và tập kiểm tra:

<!--
$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1).$$
-->

$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ với }
\epsilon \sim \mathcal{N}(0, 0.1).$$

<!--
The noise term $\epsilon$ obeys a normal distribution with a mean of 0 and a standard deviation of 0.1.
We will synthesize 100 samples each for the training set and test set.
-->

Số hạng nhiễu $\epsilon$ tuân theo phân phối chuẩn (phân phối Gauss) với giá trị trung bình bằng 0 và độ lệch chuẩn bằng 0.1.
Ta sẽ tạo 100 mẫu cho mỗi tập huấn luyện và tập kiểm tra.

```{.python .input  n=2}
maxdegree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(maxdegree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
features = np.random.shuffle(features)
poly_features = np.power(features, np.arange(maxdegree).reshape(1, -1))
poly_features = poly_features / (
    npx.gamma(np.arange(maxdegree) + 1).reshape(1, -1))
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

<!--
For optimization, we typically want to avoid very large values of gradients, losses, etc.
This is why the monomials stored in `poly_features` are rescaled from $x^i$ to $\frac{1}{i!} x^i$.
It allows us to avoid very large values for large exponents $i$.
Factorials are implemented in Gluon using the Gamma function,
where $n! = \Gamma(n+1)$.
-->

Khi tối ưu hóa, ta thường muốn tránh các giá trị rất lớn của gradient, mất mát, v.v.
Đây là lý do tại sao các đơn thức lưu trong `poly_features` được chuyển đổi giá trị từ $x^i$ thành $\frac{1}{i!} x^i$.
Nó cho phép ta tránh các giá trị quá lớn với số mũ bậc cao $i$.
Phép tính giai thừa được lập trình trong Gluon bằng hàm Gamma, với $n! = \Gamma(n+1)$.

<!--
Take a look at the first 2 samples from the generated dataset.
The value 1 is technically a feature, namely the constant feature corresponding to the bias.
-->

Hãy xét hai mẫu đầu tiên trong tập dữ liệu được tạo.
Về mặt kỹ thuật giá trị 1 là một đặc trưng, cụ thể là đặc trưng không đổi tương ứng với hệ số điều chỉnh.

```{.python .input  n=3}
features[:2], poly_features[:2], labels[:2]
```

<!--
### Training and Testing Model
-->

### Huấn luyện và Kiểm tra Mô hình

<!--
Let's first implement a function to evaluate the loss on a given data.
-->

Trước tiên ta lập trình hàm để tính giá trị mất mát của dữ liệu cho trước.

```{.python .input  n=4}
# Saved in the d2l package for later use
def evaluate_loss(net, data_iter, loss):
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # sum_loss, num_examples
    for X, y in data_iter:
        metric.add(loss(net(X), y).sum(), y.size)
    return metric[0] / metric[1]
```

<!--
Now define the training function.
-->

Giờ ta định nghĩa hàm huấn luyện.

```{.python .input  n=5}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=1000):
    loss = gluon.loss.L2Loss()
    net = nn.Sequential()
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net.add(nn.Dense(1, use_bias=False))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(1, num_epochs+1):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch % 50 == 0:
            animator.add(epoch, (evaluate_loss(net, train_iter, loss),
                                 evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
### Third-Order Polynomial Function Fitting (Normal)
-->

### Khớp Hàm Đa thức Bậc Ba (dạng chuẩn)

<!--
We will begin by first using a third-order polynomial function with the same order as the data generation function.
The results show that this model’s training error rate when using the testing dataset is low.
The trained model parameters are also close to the true values $w = [5, 1.2, -3.4, 5.6]$.
-->

Ta sẽ bắt đầu với việc sử dụng hàm đa thức bậc ba, cùng bậc với hàm tạo dữ liệu.
Kết quả cho thấy cả lỗi huấn luyện và lỗi kiểm tra của mô hình đều thấp. 
Các tham số của mô hình được huấn luyện cũng gần với giá trị thật $w = [5, 1.2, -3.4, 5.6]$.

```{.python .input  n=6}
# Pick the first four dimensions, i.e., 1, x, x^2, x^3 from the polynomial
# features
train(poly_features[:n_train, 0:4], poly_features[n_train:, 0:4],
      labels[:n_train], labels[n_train:])
```

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!--
### Linear Function Fitting (Underfitting)
-->

### Khớp hàm tuyến tính (Dưới khớp)

<!--
Let’s take another look at linear function fitting.
After the decline in the early epoch, it becomes difficult to further decrease this model’s training error rate.
After the last epoch iteration has been completed, the training error rate is still high.
When used to fit non-linear patterns (like the third-order polynomial function here) linear models are liable to underfit.
-->

Hãy xem lại việc khớp hàm tuyến tính.
Sau sự sụt giảm ở những epoch đầu, việc giảm thêm lỗi huấn luyện của mô hình đã trở nên khó khăn.
Sau khi epoch cuối cùng kết thúc, lỗi huấn luyện vẫn còn cao. 
Khi được sử dụng để khớp các khuôn mẫu phi tuyến (như hàm đa thức bậc ba trong trường hợp này), các mô hình tuyến tính dễ bị dưới khớp.

```{.python .input  n=7}
# Pick the first four dimensions, i.e., 1, x from the polynomial features
train(poly_features[:n_train, 0:3], poly_features[n_train:, 0:3],
      labels[:n_train], labels[n_train:])
```

<!--
### Insufficient Training (Overfitting)
-->

### Thiếu dữ liệu huấn luyện (Quá khớp)

<!--
Now let's try to train the model using a polynomial of too high degree.
Here, there is insufficient data to learn that the higher-degree coefficients should have values close to zero.
As a result, our overly-complex model is far too susceptible to being influenced by noise in the training data.
Of course, our training error will now be low (even lower than if we had the right model!) but our test error will be high.
-->

Bây giờ, hãy thử huấn luyện mô hình sử dụng một đa thức với bậc rất cao.
Trong trường hợp này, mô hình không có đủ dữ liệu để học được rằng các hệ số bậc cao nên có giá trị gần với không.
Vì vậy, mô hình quá phức tạp của ta sẽ dễ bị ảnh hưởng bởi nhiễu ở trong dữ liệu huấn luyện.
Dĩ nhiên, lỗi huấn luyện trong trường hợp này sẽ thấp (thậm chí còn thấp hơn cả khi chúng ta có được mô hình thích hợp!) nhưng lỗi kiểm tra sẽ cao.

<!--
Try out different model complexities (`n_degree`) and training set sizes (`n_subset`) to gain some intuition of what is happening.
-->

Thử nghiệm với các độ phức tạp của mô hình (`n_degree`) và các kích thước của tập huấn luyện (`n_subset`) khác nhau để thấy được một cách trực quan điều gì đang diễn ra.

```{.python .input  n=8}
n_subset = 100  # Subset of data to train on
n_degree = 20  # Degree of polynomials
train(poly_features[1:n_subset, 0:n_degree],
      poly_features[n_train:, 0:n_degree], labels[1:n_subset],
      labels[n_train:])
```

<!--
In later chapters, we will continue to discuss overfitting problems and methods for dealing with them, such as weight decay and dropout.
-->

Ở các chương sau, chúng ta sẽ tiếp tục thảo luận về các vấn đề quá khớp và các phương pháp đối phó, ví dụ như suy giảm trọng số hay dropout. 


<!--
## Summary
-->

## Tóm tắt

<!--
* Since the generalization error rate cannot be estimated based on the training error rate, simply minimizing the training error rate will not necessarily mean a reduction in the generalization error rate. Machine learning models need to be careful to safeguard against overfitting such as to minimize the generalization error.
* A validation set can be used for model selection (provided that it is not used too liberally).
* Underfitting means that the model is not able to reduce the training error rate while overfitting is a result of the model training error rate being much lower than the testing dataset rate.
* We should choose an appropriately complex model and avoid using insufficient training samples.
-->

* Bởi vì lỗi khái quát không thể được ước lượng dựa trên lỗi huấn luyện, nên việc chỉ đơn thuần cực tiểu hóa lỗi huấn luyện sẽ không nhất thiết đồng nghĩa với việc cực tiểu hóa lỗi khái quát.
Các mô hình học máy cần phải được bảo vệ khỏi việc quá khớp để giảm thiểu lỗi khái quát.
* Một tập kiểm định có thể được sử dụng cho việc lựa chọn mô hình (với điều kiện là tập này không được sử dụng quá nhiều).
* Dưới khớp có nghĩa là mô hình không có khả năng giảm lỗi huấn luyện, còn quá khớp là kết quả của việc lỗi huấn luyện của mô hình thấp hơn nhiều so với lỗi kiểm tra. 
* Chúng ta nên chọn một mô hình phức tạp vừa phải và tránh việc sử dụng tập huấn luyện không có đủ số số mẫu.


<!--
## Exercises
-->

## Bài tập

<!--
1. Can you solve the polynomial regression problem exactly? Hint: use linear algebra.
2. Model selection for polynomials
    * Plot the training error vs. model complexity (degree of the polynomial). What do you observe?
    * Plot the test error in this case.
    * Generate the same graph as a function of the amount of data?
3. What happens if you drop the normalization of the polynomial features $x^i$ by $1/i!$. Can you fix this in some other way?
4. What degree of polynomial do you need to reduce the training error to 0?
5. Can you ever expect to see 0 generalization error?
-->

1. Bạn có thể giải bài toán hồi quy đa thức một cách chính xác không? Gợi ý: sử dụng đại số tuyến tính.
2. Lựa chọn mô hình cho các đa thức
    * Vẽ đồ thị biểu diễn lỗi huấn luyện và độ phức tạp của mô hình (bậc của đa thức). Bạn quan sát được gì?
    * Vẽ đồ thị biểu diễn lỗi kiểm tra trong trường hợp này.
    * Tạo một đồ thị tương tự nhưng với hàm của lượng dữ liệu.
3. Điều gì sẽ xảy ra nếu bạn bỏ qua việc chuẩn hóa các đặc trưng đa thức $x^i$ với $1/i!$. Bạn có thể sửa chữa vấn đề này bằng cách nào khác không?
4. Bậc mấy của đa thức giảm được tỉ lệ lỗi huấn luyện về 0?
5. Bạn có bao giờ kỳ vọng thấy được lỗi khái quát bằng 0?

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2341)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2341)
* [Tiếng Việt](https://forum.machinelearningcoban.com/c/d2l)

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md.
-->

* Đoàn Võ Duy Thanh
* Trần Yến Thy
* Lê Khắc Hồng Phúc
* Phạm Minh Đức
* Nguyễn Văn Tâm
* Vũ Hữu Tiệp
* Phạm Hồng Vinh
* Bùi Nhật Quân
* Lý Phi Long
* Nguyễn Duy Du

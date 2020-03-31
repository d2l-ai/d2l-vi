<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Considering the Environment
-->

# *dịch tiêu đề phía trên*

<!--
In the previous chapters, we worked through a number of hands-on applications of machine learning, fitting models to a variety of datasets. 
And yet, we never stopped to contemplate either where data comes from in the first place, or what we plan to ultimately do with the outputs from our models. 
Too often, machine learning developers in possession of data rush to develop models without pausing to consider these fundamental issues.
-->

*dịch đoạn phía trên*

<!--
Many failed machine learning deployments can be traced back to this pattern. 
Sometimes models appear to perform marvelously as measured by test set accuracy but fail catastrophically in deployment when the distribution of data suddenly shifts. 
More insidiously, sometimes the very deployment of a model can be the catalyst that perturbs the data distribution. 
Say, for example, that we trained a model to predict who will repay vs default on a loan, 
finding that an applicant's choice of footware was associated with the risk of default (Oxfords indicate repayment, sneakers indicate default). 
We might be inclined to thereafter grant loans to all applicants wearing Oxfords and to deny all applicants wearing sneakers.
-->

*dịch đoạn phía trên*

<!--
In this case, our ill-considered leap from pattern recognition to decision-making and our failure to critically consider the environment might have disastrous consequences.
For starters, as soon as we began making decisions based on footware, customers would catch on and change their behavior. 
Before long, all applicants would be wearing Oxfords, without any coinciding improvement in credit-worthiness. 
Take a minute to digest this because similar issues abound in many applications of machine learning: 
by introducing our model-based decisions to the environment, we might break the model.
-->

*dịch đoạn phía trên*

<!--
While we cannot possible give these topics a complete treatment in one section, we aim here to expose some common concerns, 
and to stimulate the critical thinking required to detect these situations early, mitigate damage, and use machine learning responsibly. 
Some of the solutions are simple (ask for the "right" data) some are technically difficult (implement a reinforcement learning system), 
and others require that step outside the realm of statistical prediction altogether and 
grapple with difficult philosophical questions concerning the ethical application of algorithms.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Distribution Shift
-->

## Dịch chuyển Phân phối

<!--
To begin, we return to the observational setting, putting aside for now the impacts of our actions on the environment.
In the following sections, we take a deeper look at the various ways that data distributions might shift, and what might be done to salvage model performance.
From the outset, we should warn that if the data-generating distribution $p(\mathbf{x},y)$ can shift in arbitrary ways at any point in time, then learning a robust classifier is impossible.
In the most pathological case, if the label definitions themselves can change at a moments notice: 
if suddenly what we called "cats" are now dogs and what we previously called "dogs" are now in fact cats, 
without any perceptible change in the distribution of inputs $p(\mathbf{x})$, then there is nothing we could do to detect the change or to correct our classifier at test time.
Fortunately, under some restricted assumptions on the ways our data might change in the future, 
principled algorithms can detect shift and possibly even adapt, achieving higher accuracy than if we naively continued to rely on our original classifier.
-->

Để bắt đầu, ta trở lại vị trí quan sát và tạm gác lại các tác động lên môi trường.
Trong các mục tiếp theo, ta sẽ xem xét kỹ các cách khác nhau mà phân phối dữ liệu có thể dịch chuyển và những gì ta có thể làm để cứu vãn hiệu suất mô hình.
Ngay từ đầu, ta nên cảnh báo rằng nếu phân phối tạo dữ liệu $p(\mathbf{x},y)$ có thể dịch chuyển theo các cách khác nhau tại bất kỳ thời điểm nào, thì việc học một bộ phân loại mạnh mẽ là điều bất khả thi.
Trong trường hợp xấu nhất, nếu bản thân định nghĩa của nhãn có thể thay đổi bất cứ khi nào: nếu đột nhiên con vật mà chúng ta gọi là "mèo" bây giờ là chó và trước đây chúng ta gọi là "chó" thì thực tế giờ lại là mèo, trong khi không có bất kỳ thay đổi rõ ràng nào trong phân phối của đầu vào $p(\mathbf{x})$, thì ta không thể nào phát hiện được sự thay đổi hay điều chỉnh bộ phân loại tại thời điểm kiểm tra.
May mắn thay, dưới một vài giả định chặt về cách dữ liệu có thể thay đổi trong tương lai, một vài thuật toán có thể phát hiện sự thay đổi và thậm chí có thể thích nghi để đạt được độ chính xác cao hơn so với việc tiếp tục dựa vào bộ phân loại ban đầu một cách ngây thơ. <!-- cụm từ "principled algorithms" mình tạm dịch là "thuật toán" vì chưa tìm được cách dịch hợp lý -->

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Covariate Shift
-->

### Dịch chuyển Hiệp biến

<!--
One of the best-studied forms of distribution shift is *covariate shift*.
Here we assume that although the distribution of inputs may change over time, the labeling function, i.e., the conditional distribution $P(y \mid \mathbf{x})$ does not change.
While this problem is easy to understand its also easy to overlook it in practice.
Consider the challenge of distinguishing cats and dogs.
Our training data consists of images of the following kind:
-->

Một trong những dạng dịch chuyển phân phối được nghiên cứu rộng rãi nhất là *dịch chuyển hiệp biến*.
Ở đây, ta giả định rằng mặc dù phân phối đầu vào có thể biến đổi theo thời gian, nhưng hàm gán nhãn, tức phân phối có điều kiện $P(y \mid \mathbf{x})$ thì không thay đổi.
Mặc dù vấn đề này khá dễ hiểu, trong thực tế nó thường dễ bị bỏ qua.
Hãy xem xét bài toán phân biệt mèo và chó với tập dữ liệu huấn luyện bao gồm các ảnh sau:

<!--
|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|
-->

|mèo|mèo|chó|chó|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|

<!--
At test time we are asked to classify the following images:
-->

Tại thời điểm kiểm tra ta phải phân loại các ảnh dưới đây:

<!--
|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|
-->

|mèo|mèo|chó|chó|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|

<!--
Obviously this is unlikely to work well.
The training set consists of photos, while the test set contains only cartoons.
The colors are not even realistic.
Training on a dataset that looks substantially different from the test set without some plan for how to adapt to the new domain is a bad idea.
Unfortunately, this is a very common pitfall.
Statisticians call this *covariate shift* because the root of the problem owed to a shift in the distribution of features (i.e., of *covariates*).
Mathematically, we could say that $P(\mathbf{x})$ changes but that $P(y \mid \mathbf{x})$ remains unchanged.
Although its usefulness is not restricted to this setting, when we believe $\mathbf{x}$ causes $y$, covariate shift is usually the right assumption to be working with.
-->

Rõ ràng việc phân loại tốt trong trường hợp này là rất khó khăn.
Trong khi tập huấn luyện bao gồm các ảnh thực thì tập kiểm tra chỉ chứa các ảnh hoạt hình với màu sắc thậm chí còn không thực tế.
Việc huấn luyện trên một tập dữ liệu khác biệt đáng kể so với tập kiểm tra mà không có một kế hoạch để thích ứng với sự thay đổi này là một ý tưởng tồi.
Thật không may, đây lại là một cạm bẫy rất phổ biến.
Các nhà thống kê gọi vấn đề này là *dịch chuyển hiệp biến* bởi vì gốc rễ của nó là do sự thay đổi trong phân phối của các đặc trưng (tức các *hiệp biến*).
Theo ngôn ngữ toán học, ta có thể nói rằng $P(\mathbf{x})$ thay đổi nhưng $P(y \mid \mathbf{x})$ thì không.
Khi ta tin rằng $\mathbf{x}$ gây ra $y$ thì dịch chuyển hiệp biến thường là một giả định hợp lý, mặc dù tính hữu dụng của nó không chỉ giới hạn trong trường hợp này.

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
### Label Shift
-->

### Dịch chuyển Nhãn

<!--
The converse problem emerges when we believe that what drives the shift is a change in the marginal distribution over 
the labels $P(y)$ but that the class-conditional distributions are invariant $P(\mathbf{x} \mid y)$.
Label shift is a reasonable assumption to make when we believe that $y$ causes $\mathbf{x}$.
For example, commonly we want to predict a diagnosis given its manifestations.
In this case we believe that the diagnosis causes the manifestations, i.e., diseases cause symptoms.
Sometimes the label shift and covariate shift assumptions can hold simultaneously.
For example, when the true labeling function is deterministic and unchanging, then covariate shift will always hold, including if label shift holds too.
Interestingly, when we expect both label shift and covariate shift hold, it is often advantageous to work with the methods that flow from the label shift assumption.
That is because these methods tend to involve manipulating objects that look like the label, which (in deep learning) tends 
to be comparatively easy compared to working with the objects that look like the input, which tends (in deep learning) to be a high-dimensional object.
-->

Vấn đề ngược lại xuất hiện khi chúng ta tin rằng điều gây ra sự dịch chuyển là một thay đổi trong phân phối biên của nhãn $P(y)$ trong khi phân phối có điều kiện theo lớp vẫn không đổi $P(\mathbf{x} \mid y)$.
Dịch chuyển nhãn là một giả định hợp lý khi chúng ta tin rằng $y$ gây ra $\mathbf{x}$.
Chẳng hạn, thông thường chúng ta muốn dự đoán một chẩn đoán nếu biết các biểu hiện của nó.
Trong trường hợp này chúng ta tin rằng chẩn đoán gây ra các biểu hiện, ví dụ, dịch bệnh gây ra các triệu chứng.
Thỉnh thoảng các giả định dịch chuyển nhãn và dịch chuyển hiệp biến có thể xảy ra đồng thời.
Ví dụ, khi hàm gán nhãn là tất định và không đổi, dịch chuyển hiệp biến sẽ luôn xảy ra, kể cả khi dịch chuyển nhãn cũng đang xảy ra.
Một điều thú vị là khi chúng ta tin rằng cả dịch chuyển nhãn và dịch chuyển hiệp biến đều đang xảy ra, làm việc với các phương pháp được suy ra từ giả định dịch chuyển nhãn thường chiếm lợi thế.
Đó là vì các phương pháp này có xu hướng làm việc trên các đối tượng giống với nhãn, và thường sẽ dễ thao tác hơn nếu so với các đối tượng giống với đầu vào đa chiều trong học sâu.
<!--
### Concept Shift
-->

### Dịch chuyển Khái niệm

<!--
One more related problem arises in *concept shift*, the situation in which the very label definitions change.
This sounds weird—after all, a *cat* is a *cat*.
Indeed the definition of a cat might not change, but can we say the same about soft drinks?
It turns out that if we navigate around the United States, shifting the source of our data by geography, 
we will find considerable concept shift regarding the definition of even this simple term as shown in :numref:`fig_popvssoda`.
-->

Một vấn đề liên quan nữa nổi lên, gọi là *dịch chuyển khái niệm*, là tình huống khi các định nghĩa của nhãn thay đổi.
Điều này nghe có vẻ lạ vì sau cùng, con mèo là con mèo.
Quả thực định nghĩa của một con mèo có thể không thay đổi, nhưng ta có thể nói như vậy với thuật ngữ "đồ uống có ga" hay không?
Hoá ra nếu chúng ta di chuyển vòng quanh nước Mỹ, dịch chuyển nguồn dữ liệu theo vùng địa lý, ta sẽ thấy sự dịch chuyển khái niệm đáng kể liên quan đến thuật ngữ đơn giản này như thể hiện trong :numref:`fig_popvssoda`.

<!--
![Concept shift on soft drink names in the United States.](../img/popvssoda.png)
-->

![Dịch chuyển khái niệm của tên các loại đồ uống có ga ở nước Mỹ.](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

<!--
If we were to build a machine translation system, the distribution $P(y \mid x)$ might be different depending on our location.
This problem can be tricky to spot.
A saving grace is that often the $P(y \mid x)$ only shifts gradually.
-->

Nếu chúng ta xây dựng một hệ thống dịch máy, phân phối $P(y \mid x)$ có thể khác nhau tuỳ thuộc vào vị trí của chúng ta.
Vấn đề này có thể khó nhận ra.
Nhưng bù lại $P(y \mid x)$ thường chỉ dịch chuyển từ từ.

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Examples
-->

### Ví dụ

<!--
Before we go into further detail and discuss remedies, we can discuss a number of situations where covariate and concept shift may not be so obvious.
-->

Trước khi đi vào chi tiết và thảo luận các giải pháp, ta có thể thảo luận một số tình huống khi dịch chuyển hiệp biến và khái niệm biểu hiện không quá rõ ràng.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
#### Medical Diagnostics
-->

#### *dịch tiêu đề phía trên*

<!--
Imagine that you want to design an algorithm to detect cancer.
You collect data from healthy and sick people and you train your algorithm.
It works fine, giving you high accuracy and you conclude that you’re ready for a successful career in medical diagnostics.
Not so fast...
-->

*dịch đoạn phía trên*

<!--
Many things could go wrong.
In particular, the distributions that you work with for training and those that you encounter in the wild might differ considerably.
This happened to an unfortunate startup, that Alex had the opportunity to consult for many years ago.
They were developing a blood test for a disease that affects mainly older men and they’d managed to obtain a fair amount of blood samples from patients.
It is considerably more difficult, though, to obtain blood samples from healthy men (mainly for ethical reasons).
To compensate for that, they asked a large number of students on campus to donate blood and they performed their test.
Then they asked me whether I could help them build a classifier to detect the disease.
I told them that it would be very easy to distinguish between both datasets with near-perfect accuracy.
After all, the test subjects differed in age, hormone levels, physical activity, diet, alcohol consumption, and many more factors unrelated to the disease.
This was unlikely to be the case with real patients:
Their sampling procedure made it likely that an extreme case of covariate shift would arise between the *source* and *target* distributions, 
and at that, one that could not be corrected by conventional means.
In other words, training and test data were so different that nothing useful could be done and they had wasted significant amounts of money.
-->

*dịch đoạn phía trên*

<!--
#### Self Driving Cars
-->

#### *dịch tiêu đề phía trên*

<!--
Say a company wanted to build a machine learning system for self-driving cars.
One of the key components is a roadside detector.
Since real annotated data is expensive to get, they had the (smart and questionable) idea to use synthetic data from a game rendering engine as additional training data.
This worked really well on "test data" drawn from the rendering engine.
Alas, inside a real car it was a disaster.
As it turned out, the roadside had been rendered with a very simplistic texture.
More importantly, *all* the roadside had been rendered with the *same* texture and the roadside detector learned about this "feature" very quickly.
-->

*dịch đoạn phía trên*

<!--
A similar thing happened to the US Army when they first tried to detect tanks in the forest.
They took aerial photographs of the forest without tanks, then drove the tanks into the forest and took another set of pictures.
The so-trained classifier worked "perfectly".
Unfortunately, all it had learned was to distinguish trees with shadows from trees without shadows---the first set of pictures was taken in the early morning, the second one at noon.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
#### Nonstationary distributions
-->

#### Phân phối không dừng

<!--
A much more subtle situation arises when the distribution changes slowly and the model is not updated adequately.
Here are some typical cases:
-->

Một vấn đề đặc biệt phát sinh khi phân phối thay đổi chậm và mô hình không được cập nhật thoả đáng.
Dưới đây là một vài trường hợp điển hình:

<!--
* We train a computational advertising model and then fail to update it frequently (e.g., we forget to incorporate that an obscure new device called an iPad was just launched).
* We build a spam filter. It works well at detecting all spam that we have seen so far. But then the spammers wisen up and craft new messages that look unlike anything we have seen before.
* We build a product recommendation system. It works throughout the winter... but then it keeps on recommending Santa hats long after Christmas.
-->

* Chúng ta huấn luyện mô hình tính toán cho việc quảng cáo và sau đó không cập nhật thường xuyên (giả sử như chúng ta quên bổ sung thêm thiết bị iPad mới vừa được ra mắt).
* Xây dựng một mô hình lọc thư rác. Mô hình làm việc rất tốt khi phát hiện tất cả các thư rác mà chúng ta biết cho đên nay. Tuy nhiên hiện tại những người gửi thư rác đã tạo ra các mẫu thư mới trông không hề giống như những gì mà chúng ta biết trước đây.
* Ta xây dựng hệ thống đề xuất sản phẩm. Hệ thống làm việc trong suốt mùa đông… nhưng sau đó nó vẫn tiếp tục đề xuất các mẫu nón ông già Noel ngay cả khi Giáng Sinh đã qua.

<!--
#### More Anecdotes
-->

#### Các giai thoại khác

<!--
* We build a face detector. It works well on all benchmarks. 
Unfortunately it fails on test data---the offending examples are close-ups where the face fills the entire image (no such data was in the training set).
* We build a web search engine for the USA market and want to deploy it in the UK.
* We train an image classifier by compiling a large dataset where each among a large set of classes is equally represented in the dataset, 
say 1000 categories, represented by 1000 images each. Then we deploy the system in the real world, where the actual label distribution of photographs is decidedly non-uniform.
-->

* Chúng ta xây dựng mô hình phát hiện gương mặt. Nó hoạt động rất tốt trên các bài kiểm tra đánh giá.
* Không may mắn là mô hình lại lỗi ở phần dữ liệu thử nghiệm -- một số các ví dụ vi phạm là một vài ảnh bị lắp đầy hoàn toàn bởi khuôn mặt và không có dữ liệu nào tương tự như vậy xuất hiện trong tập huấn luyện.
* Chúng ta huấn luyện một trình phân loại hình ảnh bằng cách biên dịch một tập dữ liệu lớn, trong đó mỗi tập dữ liệu lớn của các lớp được biểu diễn bằng nhau trong tập dữ liệu,
có 1000 lớp, mỗi lớp được biểu diễn bởi 1000 ảnh. Sau đó chúng ta triển khai hệ thống trên thực tế, trong đó việc phân phối nhãn của các hình ảnh là không đồng nhất. 

<!--
In short, there are many cases where training and test distributions $p(\mathbf{x}, y)$ are different.
In some cases, we get lucky and the models work despite covariate, label, or concept shift.
In other cases, we can do better by employing principled strategies to cope with the shift.
The remainder of this section grows considerably more technical.
The impatient reader could continue on to the next section as this material is not prerequisite to subsequent concepts.
-->

Chung quy lại, có nhiều trường hợp mà phân phối huấn luyện và thử nghiệm $p(\mathbf{x}, y)$ là khác nhau.
Trong một số trường hợp may mắn thì các mô hình vẫn chạy tốt dù thay đổi hiệp biến, nhãn hay khái niệm.
Trong một số trường hợp khác, chúng ta có thể làm tốt hơn bằng cách sử dụng các chiến lược nguyên tắc để giải quyết sự thay đổi.
Phần còn lại của mục này tập trung nhiều hơn đáng kể về vấn đề kỹ thuật.
Tuy nhiên đối với những bạn đọc không thích thì có thể bỏ qua vì trong phần tiếp theo sẽ không đề cập đến các kiến thức cho các khái niệm dưới đây.

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Covariate Shift Correction
-->

### Hiệu chỉnh Dịch chuyển Hiệp biến

<!--
Assume that we want to estimate some dependency $P(y \mid \mathbf{x})$ for which we have labeled data $(\mathbf{x}_i, y_i)$.
Unfortunately, the observations $x_i$ are drawn from some *target* distribution $q(\mathbf{x})$ rather than the *source* distribution $p(\mathbf{x})$.
To make progress, we need to reflect about what exactly is happening during training: 
we iterate over training data and associated labels $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ and update the weight vectors of the model after every minibatch.
We sometimes additionally apply some penalty to the parameters, using weight decay, dropout, or some other related technique.
This means that we largely minimize the loss on the training.
-->

Giả sử ta muốn ước lượng mối liên hệ phụ thuộc $P(y \mid \mathbf{x})$ khi đã có dữ liệu được gán nhãn $(\mathbf{x}_i, y_i)$.
Thật không may, các điểm quan sát $x_i$ được thu thập từ một phân phối *mục tiêu* $q(\mathbf{x})$ thay vì từ phân phối *gốc* $p(\mathbf{x})$.
Để có được tiến triển, chúng ta cần suy nghĩ lại xem chính xác việc gì đang diễn ra trong quá trình huấn luyện:
ta duyệt qua tập dữ liệu huấn luyện với nhãn kèm theo $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\} và cập nhật vector trọng số của mô hình sau mỗi minibatch.
Chúng ta thi thoảng cũng áp dụng thêm một lượng phạt nào đó lên các tham số, bằng cách dùng suy giảm trọng số, dropout hoặc các kĩ thuật liên quan khác.
Điều này nghĩa là ta hầu như chỉ đang giảm thiểu giá trị mất mát trên tập huấn luyện.

$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{some~penalty}(w).
$$

<!--
Statisticians call the first term an *empirical average*, i.e., an average computed over the data drawn from $P(x) P(y \mid x)$.
If the data is drawn from the "wrong" distribution $q$, we can correct for that by using the following simple identity:
-->

Các nhà thống kê gọi số hạng đầu tiên là *trung bình thực nghiệm*, tức là trung bình được tính qua dữ liệu lấy từ phân phối $P(x) P(y \mid x)$.
Nếu dữ liệu được lấy "nhầm" từ phân phối $q$, ta có thể hiệu chỉnh lại bằng cách sử dụng đồng nhất thức:

$$
\begin{aligned}
\int p(\mathbf{x}) f(\mathbf{x}) dx
& = \int q(\mathbf{x}) f(\mathbf{x}) \frac{p(\mathbf{x})}{q(\mathbf{x})} dx.
\end{aligned}
$$

<!--
In other words, we need to re-weight each instance by the ratio of probabilities that it would have been drawn from the correct distribution $\beta(\mathbf{x}) := p(\mathbf{x})/q(\mathbf{x})$.
Alas, we do not know that ratio, so before we can do anything useful we need to estimate it.
Many methods are available, including some fancy operator-theoretic approaches that attempt to recalibrate the expectation operator directly using a minimum-norm or a maximum entropy principle.
Note that for any such approach, we need samples drawn from both distributions---the "true" $p$, 
e.g., by access to training data, and the one used for generating the training set $q$ (the latter is trivially available).
Note however, that we only need samples $\mathbf{x} \sim q(\mathbf{x})$; we do not to access labels $y \sim q(y)$.
-->

Nói cách khác, chúng ta cần đánh lại trọng số cho mỗi mẫu bằng tỉ lệ của các xác suất mà mẫu được lấy từ đúng phân phối $\beta(\mathbf{x}) := p(\mathbf{x})/q(\mathbf{x})$.
Đáng buồn là chúng ta không biết tỉ lệ đó nên trước khi làm được bất cứ thứ gì hữu ích ta phải ước lượng được nó.
Nhiều phương pháp có sẵn sử dụng cách tiếp cận lý thuyết toán tử màu mè cố tái cân bằng trực tiếp toán tử kỳ vọng, sử dụng nguyên lý chuẩn cực tiểu hay entropy cực đại.
Lưu ý là với các phương thức này yêu cầu ta lấy mẫu từ cả phân phối "đúng" $p$ (bằng cách sử dụng tập huấn luyện) và phân phối được dùng để tạo ra tập kiểm tra $q$ (việc này hiển nhiên là có thể được).
Tuy nhiên cũng lưu ý rằng ta chỉ cần mẫu $\mathbf{x} \sim q(\mathbf{x})$; ta không hề cần sử dụng đến nhãn $y \sim q(y)$.

<!--
In this case, there exists a very effective approach that will give almost as good results: logistic regression.
This is all that is needed to compute estimate probability ratios.
We learn a classifier to distinguish between data drawn from $p(\mathbf{x})$ and data drawn from $q(x)$.
If it is impossible to distinguish between the two distributions then it means that the associated instances are equally likely to come from either one of the two distributions.
On the other hand, any instances that can be well discriminated should be significantly overweighted or underweighted accordingly.
For simplicity’s sake assume that we have an equal number of instances from both distributions, denoted by $\mathbf{x}_i \sim p(\mathbf{x})$ and $\mathbf{x}_i' \sim q(\mathbf{x})$ respectively.
Now denote by $z_i$ labels which are 1 for data drawn from $p$ and -1 for data drawn from $q$.
Then the probability in a mixed dataset is given by
-->

Trong trường hợp này có một cách rất hiệu quả, cho kết quả tốt gần ngang ngửa: hồi quy logistic.
Đấy là tất cả những gì ta cần để tính xấp xỉ tỉ lệ xác suất.
Chúng ta cho học một bộ phân loại để phân biệt giữa dữ liệu được lấy từ phân phối $p(\mathbf{x})$ và phân phối $q(x)$.
Nếu không thể phân biệt được giữa hai phân phối thì tức là khả năng các mẫu liên quan đến từ một trong hai phân phối là ngang nhau.
Mặt khác, bất kì mẫu nào mà có thể được phân biệt dễ dàng thì cần được đánh trọng số cao lên hoặc giảm đi tương ứng.
Để cho đơn giản, giả sử ta có số lượng mẫu đến từ hai phân phối là bằng nhau, được kí hiệu lần lượt là $\mathbf{x}_i \sim p(\mathbf{x})$ và $\mathbf{x}_i' \sim q(\mathbf{x})$.
Ta kí hiệu nhãn $z_i$ bằng 1 cho dữ liệu từ phân phối $p$ và -1 cho dữ liệu từ $q$.
Lúc này xác suất trong một bộ dữ liệu được trộn lẫn sẽ là

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ and hence } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

<!--
Hence, if we use a logistic regression approach where $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-f(\mathbf{x}))}$ it follows that
-->

Do đó, nếu sử dụng cách tiếp cận hồi quy logistic mà trong đó $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-f(\mathbf{x}))}$, ta có 

$$
\beta(\mathbf{x}) = \frac{1/(1 + \exp(-f(\mathbf{x})))}{\exp(-f(\mathbf{x}))/(1 + \exp(-f(\mathbf{x})))} = \exp(f(\mathbf{x})).
$$

<!-- ===================== Kết thúc dịch Phần 6 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 7 ===================== -->

<!--
As a result, we need to solve two problems: first one to distinguish between data drawn from both distributions, 
and then a reweighted minimization problem where we weigh terms by $\beta$, e.g., via the head gradients.
Here's a prototypical algorithm for that purpose which uses an unlabeled training set $X$ and test set $Z$:
-->

Vì vậy, có hai bài toán cần được giải quyết: đầu tiên là bài toán phân biệt giữa dữ liệu được lấy ra từ hai phân phối,
và sau đó là bài toán tối thiểu hóa với trọng số cho các mẫu được đánh lại với $\beta$, ví dụ như thông qua các gradient đầu.
Dưới đây là một thuật toán nguyên mẫu để giải quyết hai bài toán trên. Thuật toán này sử dụng tập huấn luyện không được gán nhãn $X$ và tập kiểm tra $Z$:

<!--
1. Generate training set with $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$.
2. Train binary classifier using logistic regression to get function $f$.
3. Weigh training data using $\beta_i = \exp(f(\mathbf{x}_i))$ or better $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$.
4. Use weights $\beta_i$ for training on $X$ with labels $Y$.
-->


1. Tạo một tập huấn luyện với $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$.
2. Huấn luyện một bộ phân loại nhị phân sử dụng hồi quy logistic để tìm hàm f.
3. Đánh trọng số cho dữ liệu huấn luyện bằng cách sử dụng $\beta_i = \exp(f(\mathbf{x}_i))$, hoặc tốt hơn là $$\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$.
4. Sử dụng trọng số $\beta_i$ để huấn luyện trên $X$ với nhãn $Y$.

<!--
Note that this method relies on a crucial assumption.
For this scheme to work, we need that each data point in the target (test time) distribution had nonzero probability of occurring at training time.
If we find a point where $q(\mathbf{x}) > 0$ but $p(\mathbf{x}) = 0$, then the corresponding importance weight should be infinity.
-->

Lưu ý rằng phương pháp này được dựa trên một giả định quan trọng.
Để có được một kết quả tốt, ta cần đảm bảo rằng mỗi điểm dữ liệu trong phân phối mục tiêu (tại thời điểm kiểm tra) có xác suất xảy ra tại thời điểm huấn luyện khác không.
Nếu một điểm có $q(\mathbf{x}) > 0$ nhưng $p(\mathbf{x}) = 0$, thì trọng số quan trọng tương ứng bằng vô hạn.

<!--
*Generative Adversarial Networks* use a very similar idea to that described above to engineer a *data generator* that outputs data that cannot be distinguished from examples sampled from a reference dataset.
In these approaches, we use one network, $f$ to distinguish real versus fake data and a second network $g$ that tries to fool the discriminator $f$ into accepting fake data as real.
We will discuss this in much more detail later.
-->

*Mạng Đối Sinh* sử dụng một ý tưởng rất giống với mô tả ở trên để thiết kế một *bộ tạo dữ liệu* có khả năng sinh dữ liệu không thể phân biệt được với các mẫu được lấy từ một tập dữ liệu tham chiếu.
Trong các phương pháp này, ta sử dụng một mạng $f$ để phân biệt dữ liệu thật với dữ liệu giả, và một mạng thứ hai $g$ cố gắng đánh lừa bộ phân biệt $f$ rằng dữ liệu giả là thật.
Ta sẽ thảo luận vấn đề này một cách chi tiết hơn ở các phần sau.

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
### Label Shift Correction
-->

### Hiệu chỉnh Dịch chuyển nhãn

<!--
For the discussion of label shift, we will assume for now that we are dealing with a $k$-way multiclass classification task.
When the distribution of labels shifts over time $p(y) \neq q(y)$ but the class-conditional distributions stay the same $p(\mathbf{x})=q(\mathbf{x})$, 
our importance weights will correspond to the label likelihood ratios $q(y)/p(y)$.
One nice thing about label shift is that if we have a reasonably good model (on the source distribution) 
then we can get consistent estimates of these weights without ever having to deal with the ambient dimension 
(in deep learning, the inputs are often high-dimensional perceptual objects like images, 
while the labels are often easier to work, say vectors whose length corresponds to the number of classes).
-->

Để thảo luận về dịch chuyển nhãn, ta sẽ giả định rằng ta đang giải quyết một bài toán phân loại $k$ lớp.
Nếu phân phối của nhãn thay đổi theo thời gian $p(y) \neq q(y)$ nhưng các phân phối có điều kiện của lớp vẫn giữ nguyên $p(\mathbf{x})=q(\mathbf{x})$, thì trọng số quan trọng sẽ tương ứng với tỉ lệ hợp lý (*likelihood ratio*) của nhãn $q(y)/p(y)$.
Một điều tốt về dịch chuyển nhãn là nếu ta có một mô hình tương đối tốt (trên phân phối gốc), ta có thể có các ước lượng nhất quán cho các trọng số này mà không phải đối phó với không gian đầu vào (trong học sâu, đầu vào thường là dữ liệu nhiều chiều như hình ảnh, trong khi các nhãn thường dễ làm việc hơn vì chúng chỉ là các vector có chiều dài tương ứng với số lượng lớp).

<!--
To estimate calculate the target label distribution, we first take our reasonably good off the shelf classifier 
(typically trained on the training data) and compute its confusion matrix using the validation set (also from the training distribution).
The confusion matrix C, is simply a $k \times k$ matrix where each column corresponds to the *actual* label and each row corresponds to our model's predicted label.
Each cell's value $c_{ij}$ is the fraction of predictions where the true label was $j$ *and* our model predicted $y$.
-->

Để ước lượng phân phối nhãn mục tiêu, đầu tiên ta dùng một bộ phân loại sẵn có tương đối tốt (thường được học trên tập huấn luyện) và sử dụng một tập kiểm định (cùng phân phối với tập huấn luyện) để tính ma trận nhầm lẫn.
Ma trận nhầm lẫn C là một ma trận $k \times k$, trong đó mỗi cột tương ứng với một nhãn *thật* và mỗi hàng tương ứng với nhãn dự đoán của mô hình.
Giá trị của mỗi phần tử $c_{ij}$ là tỉ lệ mẫu có nhãn thật là $j$ *và* nhãn dự đoán là $i$.

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
Now we cannot calculate the confusion matrix on the target data directly, because we do not get to see the labels for the examples that we see in the wild, 
unless we invest in a complex real-time annotation pipeline.
What we can do, however, is average all of our models predictions at test time together, yielding the mean model output $\mu_y$.
-->

Giờ thì ta không thể tính trực tiếp ma trận confusion trên dữ liệu đích được bởi vì ta không thể quan sát nhãn của các mẫu trong thực tế, trừ khi ta đầu tư vào một pipeline đánh nhãn phức tạp theo thời gian thực.
Tuy nhiên điều mà ta có thể làm là lấy trung bình tất cả dự đoán của mô hình tại lúc kiểm tra, từ đó có được giá trị đầu ra trung bình của mô hình $\mu_y$. 

<!--
It turns out that under some mild conditions--- if our classifier was reasonably accurate in the first place, 
if the target data contains only classes of images that we have seen before, and if the label shift assumption holds in the first place 
(far the strongest assumption here), then we can recover the test set label distribution by solving a simple linear system $C \cdot q(y) = \mu_y$.
If our classifier is sufficiently accurate to begin with, then the confusion C will be invertible, and we get a solution $q(y) = C^{-1} \mu_y$.
Here we abuse notation a bit, using $q(y)$ to denote the vector of label frequencies.
Because we observe the labels on the source data, it is easy to estimate the distribution $p(y)$.
Then for any training example $i$ with label $y$, we can take the ratio of our estimates $\hat{q}(y)/\hat{p}(y)$ to calculate the weight $w_i$, 
and plug this into the weighted risk minimization algorithm above.
-->

Hoá ra là dưới các giả định đơn giản --- chẳng hạn như bộ phân loại vốn đã khá chính xác, dữ liệu đích chỉ chứa ảnh thuộc các lớp đã quan sát được từ trước, và giả định dịch chuyển nhãn là đúng (đây là giả định lớn nhất tới bây giờ), thì ta có thể khôi phục phân phối nhãn trên tập kiểm tra bằng cách giải một hệ phương trình tuyến tính đơn giản $C \cdot q(y) = \mu_y$.
Nếu bộ phân loại đã khá chính xác ngay từ đầu thì ma trận confusion C là khả nghịch và ta có nghiệm $q(y) = C^{-1} \mu_y$.
Ở đây ta đang lạm dụng kí hiệu một chút khi sử dụng $q(y)$ để kí hiệu vector tần suất nhãn.
Vì ta quan sát được nhãn trên dữ liệu gốc, có thể dễ dàng ước lượng phân phối $p(y)$.
Sau đó với bất kì mẫu huấn luyện $i$ với nhãn $y$, ta có thể lấy tỉ lệ ước lượng $\hat{q}(y)/\hat{p}(y)$ để tính trọng số $w_i$ và đưa vào thuật toán tối thiểu rủi ro có trọng số được mô tả ở trên.


<!--
### Concept Shift Correction
-->

### Hiệu chỉnh Dịch chuyển Khái niệm

<!--
Concept shift is much harder to fix in a principled manner.
For instance, in a situation where suddenly the problem changes from distinguishing cats from dogs to one of distinguishing white from black animals, 
it will be unreasonable to assume that we can do much better than just collecting new labels and training from scratch.
Fortunately, in practice, such extreme shifts are rare.
Instead, what usually happens is that the task keeps on changing slowly.
To make things more concrete, here are some examples:
-->

Khắc phục vấn đề dịch chuyển khái niệm theo một cách có nguyên tắc khó hơn rất nhiều.
Chẳng hạn như đột nhiên vấn đề chuyển từ phân biệt chó và mèo sang phân biệt động vật có màu trắng và động vật có màu đen, hoàn toàn có lý khi tin rằng ta không thể làm tốt hơn việc thu thập tập nhãn mới và huấn luyện lại từ đầu.
May mắn thay vấn đề dịch chuyển tới mức cực đoan như vậy trong thực tế khá hiếm.
Thay vào đó, điều thường diễn ra là tác vụ cứ dần dần thay đổi.
Để làm rõ hơn, ta xét các ví dụ dưới đây:

<!--
* In computational advertising, new products are launched, old products become less popular. 
This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic cameras lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e., most of the news remains unchanged but new stories appear).
-->

* Trong ngành quảng cáo điện toán, sản phẩm mới ra mắt và sản phẩm cũ trở nên ít phổ biến hơn.
Điều này nghĩa là phân phối của các mẩu quảng cáo và mức phổ biến của chúng sẽ thay đổi dần dần và bất kì bộ dự đoán tỉ lệ click-through nào cũng cần thay đổi theo.
* Ống kính của các camera giao thông bị mờ đi theo thời gian do tác động của môi trường, có ảnh hưởng tăng dần tới chất lượng ảnh.
* Nội dung các mẩu tin thay đổi theo thời gian, tức là tin tức thì không đổi nhưng các sự kiện mới luôn diễn ra.

<!--
In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. 
In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.
-->

Trong các trường hợp trên, ta có thể sử dụng cùng cách tiếp cận cho việc huấn luyện mô hình để thích ứng với các biến đổi trong dữ liệu.
Nói cách khác, chúng ta sử dụng trọng số đang có và chỉ thực hiện thêm vài bước cập nhật trên dữ liệu mới thay vì huấn luyện lại từ đầu.

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 6 - BẮT ĐẦU ===================================-->

<!--
## A Taxonomy of Learning Problems
-->

## *dịch tiêu đề phía trên*

<!--
Armed with knowledge about how to deal with changes in $p(x)$ and in $P(y \mid x)$, we can now consider some other aspects of machine learning problems formulation.
-->

*dịch đoạn phía trên*


<!--
* **Batch Learning.** Here we have access to training data and labels $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, which we use to train a network $f(x, w)$. 
Later on, we deploy this network to score new data $(x, y)$ drawn from the same distribution. 
This is the default assumption for any of the problems that we discuss here. 
For instance, we might train a cat detector based on lots of pictures of cats and dogs. 
Once we trained it, we ship it as part of a smart catdoor computer vision system that lets only cats in. 
This is then installed in a customer's home and is never updated again (barring extreme circumstances).
* **Online Learning.** Now imagine that the data $(x_i, y_i)$ arrives one sample at a time. 
More specifically, assume that we first observe $x_i$, then we need to come up with an estimate $f(x_i, w)$ and only once we have done this, 
we observe $y_i$ and with it, we receive a reward (or incur a loss), given our decision. 
Many real problems fall into this category. 
E.g. we need to predict tomorrow's stock price, this allows us to trade based on that estimate and at the end of the day we find out whether our estimate allowed us to make a profit. 
In other words, we have the following cycle where we are continuously improving our model given new observations.
-->

*dịch đoạn phía trên*

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ x_t \longrightarrow
\mathrm{estimate} ~ f_t(x_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(x_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

<!--
* **Bandits.** They are a *special case* of the problem above. 
While in most learning problems we have a continuously parametrized function $f$ where we want to learn its parameters (e.g., a deep network), 
in a bandit problem we only have a finite number of arms that we can pull (i.e., a finite number of actions that we can take). 
It is not very surprising that for this simpler problem stronger theoretical guarantees in terms of optimality can be obtained. 
We list it mainly since this problem is often (confusingly) treated as if it were a distinct learning setting.
* **Control (and nonadversarial Reinforcement Learning).** 
In many cases the environment remembers what we did. 
Not necessarily in an adversarial manner but it'll just remember and the response will depend on what happened before. 
E.g. a coffee boiler controller will observe different temperatures depending on whether it was heating the boiler previously. 
PID (proportional integral derivative) controller algorithms are a popular choice there. 
Likewise, a user's behavior on a news site will depend on what we showed him previously (e.g., he will read most news only once). 
Many such algorithms form a model of the environment in which they act such as to make their decisions appear less random (i.e., to reduce variance).
* **Reinforcement Learning.** In the more general case of an environment with memory, 
we may encounter situations where the environment is trying to *cooperate* with us (cooperative games, in particular for non-zero-sum games), 
or others where the environment will try to *win*. Chess, Go, Backgammon or StarCraft are some of the cases. 
Likewise, we might want to build a good controller for autonomous cars. 
The other cars are likely to respond to the autonomous car's driving style in nontrivial ways, 
e.g., trying to avoid it, trying to cause an accident, trying to cooperate with it, etc.
-->

*dịch đoạn phía trên*

<!--
One key distinction between the different situations above is that the same strategy that might have worked throughout in the case of a stationary environment, 
might not work throughout when the environment can adapt. 
For instance, an arbitrage opportunity discovered by a trader is likely to disappear once he starts exploiting it. 
The speed and manner at which the environment changes determines to a large extent the type of algorithms that we can bring to bear. 
For instance, if we *know* that things may only change slowly, we can force any estimate to change only slowly, too. 
If we know that the environment might change instantaneously, but only very infrequently, we can make allowances for that. 
These types of knowledge are crucial for the aspiring data scientist to deal with concept shift, i.e., when the problem that he is trying to solve changes over time.
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 10 ===================== -->

<!--
## Fairness, Accountability, and Transparency in Machine Learning
-->

## *dịch tiêu đề phía trên*

<!--
Finally, it is important to remember that when you deploy machine learning systems you are not simply minimizing negative log likelihood or maximizing accuracy---you are automating some kind of decision process.
Often the automated decision-making systems that we deploy can have consequences for those subject to its decisions.
If we are deploying a medical diagnostic system, we need to know for which populations it may work and which it may not.
Overlooking foreseeable risks to the welfare of a subpopulation would run afoul of basic ethical principles.
Moreover, "accuracy" is seldom the right metric.
When translating predictions in to actions we will often want to take into account the potential cost sensitivity of erring in various ways.
If one way that you might classify an image could be perceived as a racial sleight, 
while misclassification to a different category would be harmless, then you might want to adjust your thresholds accordingly, accounting for societal values in designing the decision-making protocol.
We also want to be careful about how prediction systems can lead to feedback loops.
For example, if prediction systems are applied naively to predictive policing, allocating patrol officers accordingly, a vicious cycle might emerge.
Neighborhoods that have more crimes, get more patrols, get more crimes discovered, get more training data, get yet more confident predictions, leading to even more patrols, even more crimes discovered, etc.
Additionally, we want to be careful about whether we are addressing the right problem in the first place. Predictive algorithms now play an outsize role in mediating the dissemination of information.
Should what news someone is exposed to be determined by which Facebook pages they have *Liked*? 
These are just a few among the many profound ethical dilemmas that you might encounter in a career in machine learning.
-->

*dịch đoạn phía trên*



<!--
## Summary
-->

## Tóm tắt

<!--
* In many cases training and test set do not come from the same distribution. This is called covariate shift.
* Covariate shift can be detected and corrected if the shift is not too severe. Failure to do so leads to nasty surprises at test time.
* In some cases the environment *remembers* what we did and will respond in unexpected ways. We need to account for that when building models.
-->

*dịch đoạn phía trên*

<!--
## Exercises
-->

## Bài tập

<!--
1. What could happen when we change the behavior of a search engine? What might the users do? What about the advertisers?
2. Implement a covariate shift detector. Hint: build a classifier.
3. Implement a covariate shift corrector.
4. What could go wrong if training and test set are very different? What would happen to the sample weights?
-->

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 10 ===================== -->

<!-- ========================================= REVISE PHẦN 6 - KẾT THÚC ===================================-->

<!--
## [Discussions](https://discuss.mxnet.io/t/2347)
-->

## Thảo luận
* [Tiếng Anh](https://discuss.mxnet.io/t/2347)
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
<!-- Phần 1 -->
*

<!-- Phần 2 -->
* Nguyễn Duy Du
* Phạm Minh Đức

<!-- Phần 3 -->
* Lê Cao Thăng

<!-- Phần 4 -->
*

<!-- Phần 5 -->
* Nguyễn Minh Thư

<!-- Phần 6 -->
* Lê Khắc Hồng Phúc
* Phạm Minh Đức

<!-- Phần 7 -->
* Nguyễn Duy Du 
* Phạm Minh Đức
* Lê Khắc Hồng Phúc

<!-- Phần 8 -->
* Lê Khắc Hồng Phúc
* Phạm Minh Đức

<!-- Phần 9 -->
*

<!-- Phần 10 -->
*

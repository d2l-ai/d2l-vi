<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Considering the Environment
-->

# Cân nhắc tới Môi trường

<!--
In the previous chapters, we worked through a number of hands-on applications of machine learning, fitting models to a variety of datasets. 
And yet, we never stopped to contemplate either where data comes from in the first place, or what we plan to ultimately do with the outputs from our models. 
Too often, machine learning developers in possession of data rush to develop models without pausing to consider these fundamental issues.
-->

Trong các chương trước ta đã thực hành một số ứng dụng của học máy và khớp mô hình với nhiều tập dữ liệu khác nhau.
Tuy nhiên, ta chưa bao giờ dừng lại để nhìn nhận về nguồn gốc của dữ liệu, hoặc dự định sẽ làm gì với đầu ra của mô hình.
Đa phần là khi có được dữ liệu, các nhà phát triển học máy thường đâm đầu vào triển khai các mô hình mà không tạm dừng để xem xét các vấn đề cơ bản này.

<!--
Many failed machine learning deployments can be traced back to this pattern. 
Sometimes models appear to perform marvelously as measured by test set accuracy but fail catastrophically in deployment when the distribution of data suddenly shifts. 
More insidiously, sometimes the very deployment of a model can be the catalyst that perturbs the data distribution. 
Say, for example, that we trained a model to predict who will repay vs default on a loan, 
finding that an applicant's choice of footware was associated with the risk of default (Oxfords indicate repayment, sneakers indicate default). 
We might be inclined to thereafter grant loans to all applicants wearing Oxfords and to deny all applicants wearing sneakers.
-->

Nhiều triển khai học máy thất bại có thể bắt nguồn từ khuôn mẫu này.
Đôi khi các mô hình có độ chính xác rất tốt trên tập kiểm tra nhưng lại thất bại thảm hại trong triển khai thực tế, khi mà phân phối của dữ liệu thay đổi đột ngột.
Đáng sợ hơn, đôi khi chính việc triển khai một mô hình có thể là chất xúc tác gây nhiễu cho phân phối dữ liệu.
Ví dụ, giả sử rằng ta huấn luyện một mô hình để dự đoán xem một người có trả được nợ hay không, rồi mô hình chỉ ra rằng việc chọn giày dép của ứng viên có liên quan đến rủi ro vỡ nợ (giày tây thì trả được nợ, giày thể thao thì không).
Từ đó, ta có thể sẽ có xu hướng chỉ cấp các khoản vay cho các ứng viên mang giày tây và sẽ từ chối cho vay đối với những trường hợp mang giày thể thao. 

<!--
In this case, our ill-considered leap from pattern recognition to decision-making and our failure to critically consider the environment might have disastrous consequences.
For starters, as soon as we began making decisions based on footware, customers would catch on and change their behavior. 
Before long, all applicants would be wearing Oxfords, without any coinciding improvement in credit-worthiness. 
Take a minute to digest this because similar issues abound in many applications of machine learning: 
by introducing our model-based decisions to the environment, we might break the model.
-->

Trong trường hợp này, việc ta không cân nhắc kỹ khi nhảy vọt từ nhận dạng khuôn mẫu đến ra quyết định và việc không nghiêm túc xem xét các yếu tố môi trường có thể gây ra hậu quả nghiêm trọng.
Như ví dụ trên, không sớm thì muộn khi ta bắt đầu đưa ra quyết định dựa trên kiểu giày, khách hàng sẽ để ý và thay đổi hành vi của họ.
Chẳng bao lâu sau, tất cả các người vay tiền sẽ mang giày tây, nhưng chỉ số tín dụng của họ thì không hề cải thiện.
Hãy dành một phút để "thấm" điều này vì có rất nhiều vấn đề tương tự trong các ứng dụng của học máy: bằng việc ra quyết định dựa trên mô hình trong một môi trường, ta có thể làm hỏng chính mô hình đó.

<!--
While we cannot possible give these topics a complete treatment in one section, we aim here to expose some common concerns, 
and to stimulate the critical thinking required to detect these situations early, mitigate damage, and use machine learning responsibly. 
Some of the solutions are simple (ask for the "right" data) some are technically difficult (implement a reinforcement learning system), 
and others require that step outside the realm of statistical prediction altogether and 
grapple with difficult philosophical questions concerning the ethical application of algorithms.
-->

Dù không thể thảo luận kỹ lưỡng về các vấn đề này chỉ trong một mục, chúng tôi vẫn muốn đề cập một vài mối bận tâm phổ biến và kích thích tư duy phản biện để có thể sớm phát hiện ra các tình huống này, từ đó giảm thiểu thiệt hại và có trách nhiệm hơn trong việc sử dụng học máy.
Một vài giải pháp khá đơn giản (thu thập dữ liệu "phù hợp"), còn một vài giải pháp lại khó hơn về mặt kỹ thuật (lập trình một hệ thống học tăng cường), và một số khác thì hoàn toàn nằm ngoài lĩnh vực dự đoán thống kê và cần ta phải vật lộn với các câu hỏi triết học khó khăn về khía cạnh đạo đức trong việc ứng dụng thuật toán.

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

Để bắt đầu, ta sẽ trở lại vị trí quan sát và tạm gác lại các tác động của ta lên môi trường.
Trong các mục tiếp theo, ta sẽ xem xét kỹ vài cách khác nhau mà phân phối dữ liệu có thể dịch chuyển và những gì ta có thể làm để cứu vãn hiệu suất mô hình.
Chúng tôi sẽ cảnh báo ngay từ đầu rằng nếu phân phối sinh dữ liệu $p(\mathbf{x},y)$ có thể dịch chuyển theo các cách khác nhau tại bất kỳ thời điểm nào, việc học một bộ phân loại mạnh mẽ là điều bất khả thi.
Trong trường hợp xấu nhất, nếu bản thân định nghĩa của nhãn có thể thay đổi bất cứ khi nào: nếu đột nhiên con vật mà chúng ta gọi là "mèo" bây giờ là "chó" và con vật trước đây chúng ta gọi là "chó" thì giờ lại là "mèo", trong khi không có bất kỳ thay đổi rõ ràng nào trong phân phối của đầu vào $p(\mathbf{x})$, thì không có cách nào để phát hiện được sự thay đổi này hay điều chỉnh lại bộ phân loại tại thời điểm kiểm tra.
May mắn thay, dưới một vài giả định chặt về cách dữ liệu có thể thay đổi trong tương lai, một vài thuật toán có thể phát hiện được sự dịch chuyển và thậm chí có thể thích nghi để đạt được độ chính xác cao hơn so với việc tiếp tục dựa vào bộ phân loại ban đầu một cách ngây thơ. <!-- cụm từ "principled algorithms" mình tạm dịch là "thuật toán" vì chưa tìm được cách dịch hợp lý -->

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

Một trong những dạng dịch chuyển phân phối được nghiên cứu rộng rãi nhất là *dịch chuyển hiệp biến* (_covariate shift_).
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
Trong khi tập huấn luyện bao gồm các ảnh đời thực thì tập kiểm tra chỉ chứa các ảnh hoạt hình với màu sắc thậm chí còn không thực tế.
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

Vấn đề ngược lại xuất hiện khi chúng ta tin rằng điều gây ra sự dịch chuyển là một thay đổi trong phân phối biên của nhãn $P(y)$ trong khi phân phối có điều kiện theo lớp $P(\mathbf{x} \mid y)$ vẫn không đổi.
Dịch chuyển nhãn là một giả định hợp lý khi chúng ta tin rằng $y$ gây ra $\mathbf{x}$.
Chẳng hạn, thông thường chúng ta muốn dự đoán kết quả chẩn đoán nếu biết các biểu hiện của bệnh.
Trong trường hợp này chúng ta tin rằng kết quả chẩn đoán gây ra các biểu hiện, tức bệnh gây ra các triệu chứng.
Thỉnh thoảng các giả định dịch chuyển nhãn và dịch chuyển hiệp biến có thể xảy ra đồng thời.
Ví dụ, khi hàm gán nhãn là tất định và không đổi, dịch chuyển hiệp biến sẽ luôn xảy ra, kể cả khi dịch chuyển nhãn cũng đang xảy ra.
Một điều thú vị là khi chúng ta tin rằng cả dịch chuyển nhãn và dịch chuyển hiệp biến đều đang xảy ra, làm việc với các phương pháp được suy ra từ giả định dịch chuyển nhãn thường chiếm lợi thế.
Các phương pháp này thường sẽ dễ làm việc hơn vì chúng thao tác trên nhãn thay vì trên các đầu vào đa chiều trong học sâu.

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

Một vấn đề liên quan nữa nằm ở việc *dịch chuyển khái niệm* (_concept shift_), khi chính định nghĩa của các nhãn thay đổi.
Điều này nghe có vẻ lạ vì sau cùng, con mèo là con mèo.
Quả thực định nghĩa của một con mèo có thể không thay đổi, nhưng điều này có đúng với thuật ngữ "đồ uống có ga" hay không?
Hoá ra nếu chúng ta di chuyển vòng quanh nước Mỹ, dịch chuyển nguồn dữ liệu theo vùng địa lý, ta sẽ thấy sự dịch chuyển khái niệm đáng kể liên quan đến thuật ngữ đơn giản này như thể hiện trong :numref:`fig_popvssoda`.

<!--
![Concept shift on soft drink names in the United States.](../img/popvssoda.png)
-->

![Dịch chuyển khái niệm của tên các loại đồ uống có ga ở Mỹ.](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

<!--
If we were to build a machine translation system, the distribution $P(y \mid x)$ might be different depending on our location.
This problem can be tricky to spot.
A saving grace is that often the $P(y \mid x)$ only shifts gradually.
-->

Nếu chúng ta xây dựng một hệ thống dịch máy, phân phối $P(y \mid x)$ có thể khác nhau tùy thuộc vào vị trí của chúng ta.
Vấn đề này có thể khó nhận ra, nhưng bù lại $P(y \mid x)$ thường chỉ dịch chuyển một cách chậm rãi.

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Examples
-->

### Ví dụ

<!--
Before we go into further detail and discuss remedies, we can discuss a number of situations where covariate and concept shift may not be so obvious.
-->

Trước khi đi vào chi tiết và thảo luận các giải pháp, ta có thể bàn thêm về một số tình huống khi dịch chuyển hiệp biến và khái niệm có thể có biểu hiện không quá rõ ràng.

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
#### Medical Diagnostics
-->

#### Chẩn đoán Y khoa

<!--
Imagine that you want to design an algorithm to detect cancer.
You collect data from healthy and sick people and you train your algorithm.
It works fine, giving you high accuracy and you conclude that you’re ready for a successful career in medical diagnostics.
Not so fast...
-->

Hãy tưởng tượng rằng bạn muốn thiết kế một giải thuật có khả năng phát hiện bệnh ung thư.
Bạn thu thập dữ liệu từ cả người khoẻ mạnh lẫn người bệnh rồi sau đó huấn luyện giải thuật.
Nó hoạt động hiệu quả, có độ chính xác cao và bạn kết luận rằng bạn đã sẵn sàng cho một sự nghiệp chẩn đoán y khoa thành công.
Đừng vội mừng...

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

Bạn có thể đã mắc nhiều sai lầm.
Cụ thể, các phân phối mà bạn dùng để huấn luyện và các phân phối bạn gặp phải trong thực tế có thể rất khác nhau.
Điều này đã từng xảy ra với một công ty khởi nghiệp không may mắn mà tôi đã tư vấn nhiều năm về trước.
Họ đã phát triển một bộ xét nghiệm máu cho một căn bệnh xảy ra chủ yếu ở đàn ông lớn tuổi và họ đã thu thập được một lượng kha khá mẫu máu từ các bệnh nhân.
Mặc dù vậy, việc thu thập mẫu máu từ những người đàn ông khoẻ mạnh lại khó khăn hơn (chủ yếu là vì lý do đạo đức).
Để giải quyết sự thiếu hụt này, họ đã kêu gọi một lượng lớn các sinh viên trong trường học tham gia hiến máu tình nguyện để thực hiện xét nghiệm máu của họ.
Sau đó họ đã nhờ tôi xây dựng một bộ phân loại để phát hiện căn bệnh.
Tôi đã nói với họ rằng việc phân biệt hai tập dữ liệu trên với độ chính xác gần như hoàn hảo là rất dễ dàng.
Sau cùng, các đối tượng kiểm tra có nhiều khác biệt về tuổi, nồng độ hóc môn, hoạt động thể chất, chế độ ăn uống, mức tiêu thụ rượu bia, và nhiều nhân tố khác không liên quan đến căn bệnh.
Điều này không giống với trường hợp của những bệnh nhân thật sự:
Quy trình lấy mẫu của họ khả năng cao đã gây ra hiện tượng dịch chuyển hiệp biến rất nặng giữa phân phối *gốc* và phân phối *mục tiêu*, và thêm vào đó, nó không thể được khắc phục bằng các biện pháp thông thường.
Nói cách khác, dữ liệu huấn luyện và kiểm tra khác biệt đến nỗi không thể xây dựng được một mô hình hữu dụng và họ đã lãng phí rất nhiều tiền của.

<!--
#### Self Driving Cars
-->

#### Xe tự hành

<!--
Say a company wanted to build a machine learning system for self-driving cars.
One of the key components is a roadside detector.
Since real annotated data is expensive to get, they had the (smart and questionable) idea to use synthetic data from a game rendering engine as additional training data.
This worked really well on "test data" drawn from the rendering engine.
Alas, inside a real car it was a disaster.
As it turned out, the roadside had been rendered with a very simplistic texture.
More importantly, *all* the roadside had been rendered with the *same* texture and the roadside detector learned about this "feature" very quickly.
-->

Giả sử có một công ty muốn xây dựng một hệ thống học máy cho xe tự hành.
Một trong những bộ phận quan trọng là bộ phát hiện lề đường.
Vì việc gán nhãn dữ liệu thực tế rất tốn kém, họ đã có một ý tưởng (thông minh và đầy nghi vấn) là sử dụng dữ liệu giả từ một bộ kết xuất đồ hoạ để thêm vào dữ liệu huấn luyện. 
Nó đã hoạt động rất tốt trên "dữ liệu kiểm tra" được lấy mẫu từ bộ kết xuất đồ hoạ.
Nhưng khi áp dụng trên xe thực tế, nó là một thảm hoạ.
Hoá ra, lề đường đã được kết xuất chỉ với một kết cấu rất đơn giản.
Quan trọng hơn, *tất cả* các lề đường đều được kết xuất với *cùng một* kết cấu và bộ phát hiện lề đường đã nhanh chóng học được "đặc trưng" này.

<!--
A similar thing happened to the US Army when they first tried to detect tanks in the forest.
They took aerial photographs of the forest without tanks, then drove the tanks into the forest and took another set of pictures.
The so-trained classifier worked "perfectly".
Unfortunately, all it had learned was to distinguish trees with shadows from trees without shadows---the first set of pictures was taken in the early morning, the second one at noon.
-->

Một điều tương tự cũng đã xảy ra với quân đội Mỹ trong lần đầu tiên họ thử nghiệm nhận diện xe tăng trong rừng.
Họ chụp các bức ảnh khu rừng từ trên cao khi không có xe tăng, sau đó lái xe tăng vào khu rừng và chụp một bộ ảnh khác.
Bộ phân loại này được huấn luyện tới mức "hoàn hảo".
Không may thay, tất cả những gì nó đã học được là phân loại cây có bóng với cây không có bóng---bộ ảnh đầu tiên được chụp vào buổi sáng sớm, trong khi bộ thứ hai được chụp vào buổi trưa.

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

Một vấn đề khó phát hiện hơn phát sinh khi phân phối thay đổi chậm rãi và mô hình không được cập nhật một cách thoả đáng.
Dưới đây là một vài trường hợp điển hình:

<!--
* We train a computational advertising model and then fail to update it frequently (e.g., we forget to incorporate that an obscure new device called an iPad was just launched).
* We build a spam filter. It works well at detecting all spam that we have seen so far. But then the spammers wisen up and craft new messages that look unlike anything we have seen before.
* We build a product recommendation system. It works throughout the winter... but then it keeps on recommending Santa hats long after Christmas.
-->

* Chúng ta huấn luyện mô hình quảng cáo điện toán và sau đó không cập nhật nó thường xuyên (chẳng hạn như quên bổ sung thêm thiết bị iPad mới vừa được ra mắt vào mô hình).
* Xây dựng một mô hình lọc thư rác. Mô hình làm việc rất tốt cho việc phát hiện tất cả các thư rác mà chúng ta biết cho đến nay. Nhưng rồi những người gửi thư rác cũng khôn khéo hơn và tạo ra các mẫu thư mới khác hẳn với những thư trước đây.
* Ta xây dựng hệ thống đề xuất sản phẩm. Hệ thống làm việc tốt trong suốt mùa đông... nhưng sau đó nó vẫn tiếp tục đề xuất các mẫu nón ông già Noel ngay cả khi Giáng Sinh đã qua từ lâu.

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
Không may mắn là mô hình lại thất bại trên tập dữ liệu kiểm tra---các ví dụ đánh bại được mô hình khi khuôn mặt lấp đầy hoàn toàn cả bức ảnh, trong khi không có dữ liệu nào tương tự như vậy xuất hiện trong tập huấn luyện.
* Ta xây dựng hệ thống tìm kiếm web cho thị trường Hoa Kỳ và hiện tại muốn triển khai nó cho thị trường Anh.
* Chúng ta huấn luyện một bộ phân loại hình ảnh bằng cách biên soạn một tập dữ liệu lớn, trong đó mỗi lớp trong tập dữ liệu đều có số lượng mẫu bằng nhau, ví dụ 1000 lớp và mỗi lớp được biểu diễn bởi 1000 ảnh.
Sau đó chúng ta triển khai hệ thống trong khi trên thực tế phân phối của nhãn chắc chắn là không đồng đều. 

<!--
In short, there are many cases where training and test distributions $p(\mathbf{x}, y)$ are different.
In some cases, we get lucky and the models work despite covariate, label, or concept shift.
In other cases, we can do better by employing principled strategies to cope with the shift.
The remainder of this section grows considerably more technical.
The impatient reader could continue on to the next section as this material is not prerequisite to subsequent concepts.
-->

Chung quy lại, có nhiều trường hợp mà phân phối huấn luyện và kiểm tra $p(\mathbf{x}, y)$ là khác nhau.
Trong một số trường hợp may mắn thì các mô hình vẫn chạy tốt dù phân phối của hiệp biến, nhãn hay khái niệm đều dịch chuyển.
Trong một số trường hợp khác, chúng ta có thể làm tốt hơn bằng cách sử dụng nhiều chiến lược một cách có nguyên tắc để đối phó với sự dịch chuyển này.
Phần còn lại của mục này sẽ tập trung nhiều hơn hẳn vào mặt kỹ thuật.
Tuy nhiên, những độc giả vội vàng có thể bỏ qua mục này vì các khái niệm được trình bày dưới đây không phải là tiền đề cho các phần tiếp theo.

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

Giả sử rằng ta muốn ước lượng mối liên hệ phụ thuộc $P(y \mid \mathbf{x})$ khi đã có dữ liệu được gán nhãn $(\mathbf{x}_i, y_i)$.
Thật không may, các mẫu quan sát $x_i$ được thu thập từ một phân phối *mục tiêu* $q(\mathbf{x})$ thay vì từ phân phối *gốc* $p(\mathbf{x})$.
Để có được tiến triển, chúng ta cần nhìn lại xem chính xác thì việc gì đang diễn ra trong quá trình huấn luyện:
ta duyệt qua tập dữ liệu huấn luyện cùng với nhãn kèm theo $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ và cập nhật vector trọng số của mô hình sau mỗi minibatch.
Đôi khi chúng ta cũng áp dụng thêm một lượng phạt nào đó lên các tham số, bằng cách dùng suy giảm trọng số, dropout hoặc các kĩ thuật liên quan khác.
Điều này nghĩa là ta hầu như chỉ đang giảm thiểu giá trị mất mát trên tập huấn luyện.

$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{một~lượng~phạt}(w).
$$

<!--
Statisticians call the first term an *empirical average*, i.e., an average computed over the data drawn from $P(x) P(y \mid x)$.
If the data is drawn from the "wrong" distribution $q$, we can correct for that by using the following simple identity:
-->

Các nhà thống kê gọi số hạng đầu tiên là *trung bình thực nghiệm*, tức trung bình được tính qua dữ liệu lấy từ phân phối $P(x) P(y \mid x)$.
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
Đáng buồn là ta không biết tỉ lệ đó, nên trước khi ta có thể làm được bất cứ thứ gì hữu ích thì ta cần phải ước lượng được nó.
Nhiều phương pháp có sẵn sử dụng cách tiếp cận lý thuyết toán tử màu mè nhằm cố tái cân bằng trực tiếp toán tử kỳ vọng bằng cách sử dụng nguyên lý chuẩn cực tiểu hay entropy cực đại.
Lưu ý rằng những phương thức này yêu cầu ta lấy mẫu từ cả phân phối "đúng" $p$ (bằng cách sử dụng tập huấn luyện) và phân phối được dùng để tạo ra tập kiểm tra $q$ (việc này hiển nhiên là có thể được).
Tuy nhiên cũng cần để ý là ta chỉ cần mẫu $\mathbf{x} \sim q(\mathbf{x})$; ta không cần sử dụng đến nhãn $y \sim q(y)$.

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

Trong trường hợp này có một cách tiếp cận rất hiệu quả và sẽ cho kết quả tốt gần ngang ngửa, đó là: hồi quy logistic.
Đấy là tất cả những gì ta cần để tính xấp xỉ tỉ lệ xác suất.
Chúng ta cho học một bộ phân loại để phân biệt giữa dữ liệu được lấy từ phân phối $p(\mathbf{x})$ và phân phối $q(x)$.
Nếu không thể phân biệt được giữa hai phân phối thì điều đó có nghĩa là khả năng các mẫu liên quan đến từ một trong hai phân phối là ngang nhau.
Mặt khác, bất kì mẫu nào mà có thể được phân biệt dễ dàng thì cần được đánh trọng số tăng lên hoặc giảm đi tương ứng.
Để cho đơn giản, giả sử ta có số lượng mẫu đến từ hai phân phối là bằng nhau, được kí hiệu lần lượt là $\mathbf{x}_i \sim p(\mathbf{x})$ và $\mathbf{x}_i' \sim q(\mathbf{x})$.
Ta kí hiệu nhãn $z_i$ bằng 1 cho dữ liệu từ phân phối $p$ và -1 cho dữ liệu từ $q$.
Lúc này xác suất trong một bộ dữ liệu được trộn lẫn sẽ là

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ và~từ~đó } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

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
và sau đó là bài toán cực tiểu hóa với trọng số cho các mẫu được đánh lại với $\beta$, ví dụ như thông qua các gradient đầu.
Dưới đây là một thuật toán nguyên mẫu để giải quyết hai bài toán trên. Thuật toán này sử dụng tập huấn luyện không được gán nhãn $X$ và tập kiểm tra $Z$:

<!--
1. Generate training set with $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$.
2. Train binary classifier using logistic regression to get function $f$.
3. Weigh training data using $\beta_i = \exp(f(\mathbf{x}_i))$ or better $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$.
4. Use weights $\beta_i$ for training on $X$ with labels $Y$.
-->


1. Tạo một tập huấn luyện với $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$.
2. Huấn luyện một bộ phân loại nhị phân sử dụng hồi quy logistic để tìm hàm $f$.
3. Đánh trọng số cho dữ liệu huấn luyện bằng cách sử dụng $\beta_i = \exp(f(\mathbf{x}_i))$, hoặc tốt hơn là $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$.
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

*Mạng Đối sinh* sử dụng một ý tưởng rất giống với mô tả ở trên để thiết kế một *bộ sinh dữ liệu* có khả năng tạo dữ liệu không thể phân biệt được với các mẫu được lấy từ một tập dữ liệu tham chiếu.
Trong các phương pháp này, ta sử dụng một mạng $f$ để phân biệt dữ liệu thật với dữ liệu giả, và một mạng thứ hai $g$ cố gắng đánh lừa bộ phân biệt $f$ rằng dữ liệu giả là thật.
Ta sẽ thảo luận vấn đề này một cách chi tiết hơn sau.

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

Để thảo luận về dịch chuyển nhãn, giả định rằng ta đang giải quyết một bài toán phân loại $k$ lớp.
Nếu phân phối của nhãn thay đổi theo thời gian $p(y) \neq q(y)$ nhưng các phân phối có điều kiện của lớp vẫn giữ nguyên $p(\mathbf{x})=q(\mathbf{x})$, thì trọng số quan trọng sẽ tương ứng với tỉ lệ hợp lý (*likelihood ratio*) của nhãn $q(y)/p(y)$. 
Một điều tốt về dịch chuyển nhãn là nếu ta có một mô hình tương đối tốt (trên phân phối gốc), ta có thể có các ước lượng nhất quán cho các trọng số này mà không phải làm việc với không gian đầu vào (trong học sâu, đầu vào thường là dữ liệu nhiều chiều như hình ảnh, trong khi làm việc với các nhãn thường dễ hơn vì chúng chỉ là các vector có chiều dài tương ứng với số lượng lớp). 

<!--
To estimate calculate the target label distribution, we first take our reasonably good off the shelf classifier 
(typically trained on the training data) and compute its confusion matrix using the validation set (also from the training distribution).
The confusion matrix C, is simply a $k \times k$ matrix where each column corresponds to the *actual* label and each row corresponds to our model's predicted label.
Each cell's value $c_{ij}$ is the fraction of predictions where the true label was $j$ *and* our model predicted $y$.
-->

Để ước lượng phân phối nhãn mục tiêu, đầu tiên ta dùng một bộ phân loại sẵn có tương đối tốt (thường được học trên tập huấn luyện) và sử dụng một tập kiểm định (cùng phân phối với tập huấn luyện) để tính ma trận nhầm lẫn. 
Ma trận nhầm lẫn C là một ma trận $k \times k$, trong đó mỗi cột tương ứng với một nhãn *thật* và mỗi hàng tương ứng với nhãn dự đoán của mô hình.
Giá trị của mỗi phần tử $c_{ij}$ là số lượng mẫu có nhãn thật là $j$ *và* nhãn dự đoán là $i$. 

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
Now we cannot calculate the confusion matrix on the target data directly, because we do not get to see the labels for the examples that we see in the wild, 
unless we invest in a complex real-time annotation pipeline.
What we can do, however, is average all of our models predictions at test time together, yielding the mean model output $\mu_y$.
-->

Giờ thì ta không thể tính trực tiếp ma trận nhầm lẫn trên dữ liệu mục tiêu được bởi vì ta không thể quan sát được nhãn của các mẫu trong thực tế, trừ khi ta đầu tư vào một pipeline phức tạp để đánh nhãn theo thời gian thực.
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

Hoá ra là dưới các giả định đơn giản --- chẳng hạn như bộ phân loại vốn đã khá chính xác, dữ liệu mục tiêu chỉ chứa ảnh thuộc các lớp đã quan sát được từ trước, và giả định dịch chuyển nhãn là đúng (đây là giả định lớn nhất tới bây giờ), thì ta có thể khôi phục phân phối nhãn trên tập kiểm tra bằng cách giải một hệ phương trình tuyến tính đơn giản $C \cdot q(y) = \mu_y$.
Nếu bộ phân loại đã khá chính xác ngay từ đầu thì ma trận nhầm lẫn C là khả nghịch và ta có nghiệm $q(y) = C^{-1} \mu_y$.
Ở đây ta đang lạm dụng kí hiệu một chút khi sử dụng $q(y)$ để kí hiệu vector tần suất nhãn. 
Vì ta quan sát được nhãn trên dữ liệu gốc, nên có thể dễ dàng ước lượng phân phối $p(y)$. 
Sau đó với bất kì mẫu huấn luyện $i$ với nhãn $y$, ta có thể lấy tỉ lệ ước lượng $\hat{q}(y)/\hat{p}(y)$ để tính trọng số $w_i$ và đưa vào thuật toán cực tiểu hóa rủi ro có trọng số được mô tả ở trên. 


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
Chẳng hạn như khi bài toán đột nhiên chuyển từ phân biệt chó và mèo sang phân biệt động vật có màu trắng và động vật có màu đen, hoàn toàn có lý khi tin rằng ta không thể làm gì tốt hơn ngoài việc thu thập tập nhãn mới và huấn luyện lại từ đầu.
May mắn thay vấn đề dịch chuyển tới mức cực đoan như vậy trong thực tế khá hiếm.
Thay vào đó, điều thường diễn ra là tác vụ cứ dần dần thay đổi.
Để làm rõ hơn, ta xét các ví dụ dưới đây:

<!--
* In computational advertising, new products are launched, old products become less popular. 
This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic cameras lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e., most of the news remains unchanged but new stories appear).
-->

* Trong ngành quảng cáo điện toán, khi một sản phẩm mới ra mắt, các sản phẩm cũ trở nên ít phổ biến hơn.
Điều này nghĩa là phân phối của các mẩu quảng cáo và mức phổ biến của chúng sẽ thay đổi dần dần và bất kì bộ dự đoán tỉ lệ click chuột nào cũng cần thay đổi theo. 
* Ống kính của các camera giao thông bị mờ đi theo thời gian do tác động của môi trường, có ảnh hưởng lớn dần tới chất lượng ảnh. 
* Nội dung các mẩu tin thay đổi theo thời gian, tức là tin tức thì không đổi nhưng các sự kiện mới luôn diễn ra. 

<!--
In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. 
In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.
-->

Với các trường hợp trên, ta có thể sử dụng cùng cách tiếp cận trong việc huấn luyện mô hình để chúng thích ứng với các biến đổi trong dữ liệu. 
Nói cách khác, chúng ta sử dụng trọng số đang có của mạng và chỉ thực hiện thêm vài bước cập nhật trên dữ liệu mới thay vì huấn luyện lại từ đầu. 

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 6 - BẮT ĐẦU ===================================-->

<!--
## A Taxonomy of Learning Problems
-->

## Phân loại các Bài toán Học máy

<!--
Armed with knowledge about how to deal with changes in $p(x)$ and in $P(y \mid x)$, we can now consider some other aspects of machine learning problems formulation.
-->

Ta đã được trang bị kiến thức về cách xử lý các thay đổi trong $p(x)$ và $P(y \mid x)$, giờ đây ta có thể xem xét một số khía cạnh khác của việc xây dựng các bài toán học máy.

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

* **Học theo batch.** Ở đây ta có dữ liệu và nhãn huấn luyện $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, được sử dụng để huấn luyện mạng $f(x, w)$.
Sau đó, ta dùng mô hình này để đánh giá điểm dữ liệu mới $(x, y)$ được lấy từ cùng một phân phối.
Đây là giả thuyết mặc định cho bất kỳ bài toán nào mà ta bàn ở đây.
Ví dụ, ta có thể huấn luyện một mô hình phát hiện mèo dựa trên nhiều hình ảnh của mèo và chó. 
Sau khi hoàn tất quá trình huấn luyện, ta đưa mô hình vào một hệ thống thị giác máy tính cho cửa sập thông minh mà chỉ cho phép mèo đi vào. 
Hệ thống này sẽ được lắp đặt tại nhà của khách hàng và nó không bao giờ được cập nhật lại (ngoại trừ một vài trường hợp hiếm hoi). 
* **Học trực tuyến.** Bây giờ hãy tưởng tượng rằng tại một thời điểm ta chỉ nhận được một mẫu dữ liệu $(x_i, y_i)$.
Cụ thể hơn, giả sử đầu tiên ta có một quan sát $x_i$, sau đó ta cần tính $f(x_i, w)$ và chỉ khi ta hoàn thành việc đưa ra quyết định, 
ta mới có thể quan sát giá trị $y_i$, rồi dựa vào nó mà nhận lại phần thưởng (hoặc chịu mất mát).
Nhiều bài toán thực tế rơi vào loại này.
Ví dụ, ta cần dự đoán giá cổ phiếu vào ngày mai, điều này cho phép ta giao dịch dựa trên dự đoán đó và vào cuối ngày ta sẽ biết được liệu nó có mang lại lợi nhuận hay không.
Nói cách khác, ta có chu trình sau, trong đó mô hình dần được cải thiện cùng với những quan sát mới.

$$
\mathrm{mô~hình} ~ f_t \longrightarrow
\mathrm{dữ~liệu} ~ x_t \longrightarrow
\mathrm{ước~lượng} ~ f_t(x_t) \longrightarrow
\mathrm{quan~sát} ~ y_t \longrightarrow
\mathrm{mất~mát} ~ l(y_t, f_t(x_t)) \longrightarrow
\mathrm{mô~hình} ~ f_{t+1}
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

* **Máy đánh bạc.** Đây là một *trường hợp đặc biệt* của bài toán trên.
Trong khi ở hầu hết các bài toán ta luôn có một hàm liên tục được tham số hóa $f$ và công việc của ta là học các tham số của nó (ví dụ như một mạng học sâu), trong bài toán máy đánh bạc ta chỉ có một số hữu hạn các cần mà ta có thể gạt (tức một số lượng giới hạn những hành động mà ta có thể thực hiện).
Không có gì quá ngạc nhiên khi với bài toán đơn giản này, ta có được các cơ sở lý thuyết tối ưu mạnh mẽ hơn.
Chúng tôi liệt kê nó ở đây chủ yếu là vì bài toán này thường được xem (một cách nhầm lẫn) như là một môi trường học tập khác biệt.
* **Kiểm soát (và Học Tăng cường phi đối kháng).** Trong nhiều trường hợp, môi trường ghi nhớ những gì ta đã làm.
Việc này không nhất thiết phải có tính chất đối kháng, môi trường chỉ nhớ và phản hồi phụ thuộc vào những gì đã xảy ra trước đó.
Ví dụ, bộ điều khiển của ấm pha cà phê sẽ quan sát được nhiệt độ khác nhau tùy thuộc vào việc nó có đun ấm trước đó không.
Giải thuật điều khiển PID (*proportional integral derivative* hay *vi tích phân tỉ lệ*) là một lựa chọn phổ biến để làm điều đó.
Tương tự như vậy, hành vi của người dùng trên một trang tin tức sẽ phụ thuộc vào những gì ta đã cho họ xem trước đây (chẳng hạn như là mỗi người chỉ đọc mỗi mẫu tin một lần duy nhất). 
Nhiều thuật toán như vậy cấu thành một mô hình của môi trường mà trong đó chúng muốn làm cho các quyết định của mình trông có vẻ ít ngẫu nhiên hơn (tức để giảm phương sai). 
* **Học Tăng cường.** Trong trường hợp khái quát hơn của môi trường có khả năng ghi nhớ, ta có thể gặp phải tình huống môi trường đang cố gắng *hợp tác* với ta (trò chơi hợp tác, đặc biệt là các trò chơi có tổng khác không), hoặc môi trường sẽ cố gắng *chiến thắng* ta như Cờ vua, Cờ vây, Backgammon hay StarCraft. 
Tương tự như vậy, có thể ta muốn xây dựng một bộ điều khiển tốt cho những chiếc xe tự hành. 
Những chiếc xe khác khả năng cao sẽ có những phản ứng đáng kể với cách lái của những chiếc xe tự hành, ví dụ như cố gắng tránh nó, cố gắng gây ra tai nạn, cố gắng hợp tác với nó, v.v.

<!--
One key distinction between the different situations above is that the same strategy that might have worked throughout in the case of a stationary environment, 
might not work throughout when the environment can adapt. 
For instance, an arbitrage opportunity discovered by a trader is likely to disappear once he starts exploiting it. 
The speed and manner at which the environment changes determines to a large extent the type of algorithms that we can bring to bear. 
For instance, if we *know* that things may only change slowly, we can force any estimate to change only slowly, too. 
If we know that the environment might change instantaneously, but only very infrequently, we can make allowances for that. 
These types of knowledge are crucial for the aspiring data scientist to deal with concept shift, i.e., when the problem that he is trying to solve changes over time.
-->

Điểm khác biệt mấu chốt giữa các tình huống khác nhau ở trên là: một chiến lược hoạt động xuyên suốt các môi trường cố định, có thể lại không hoạt động xuyên suốt được khi môi trường có khả năng thích nghi.
Chẳng hạn, nếu một thương nhân phát hiện ra cơ hội kiếm lời từ chênh lệch giá cả thị trường, khả năng cao cơ hội đó sẽ biến mất ngay khi anh ta bắt đầu lợi dụng nó.
Tốc độ và cách môi trường thay đổi có ảnh hưởng lớn đến loại thuật toán mà ta có thể sử dụng.
Ví dụ, nếu ta *biết trước* mọi việc chỉ có thể thay đổi một cách từ từ, ta có thể ép những ước lượng phải thay đổi dần theo.
Còn nếu ta biết môi trường có thể thay đổi ngay lập tức, nhưng không thường xuyên, ta có thể cho phép điều này xảy ra.
Đối với các nhà khoa học dữ liệu giỏi, những kiến thức này rất quan trọng trong việc giải quyết các toán dịch chuyển khái niệm khi vấn đề cần giải quyết lại thay đổi theo thời gian.

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 10 ===================== -->

<!--
## Fairness, Accountability, and Transparency in Machine Learning
-->
## Công bằng, Trách nhiệm và Minh bạch trong Học máy

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

Cuối cùng, cần ghi nhớ một điều quan trọng sau đây: khi triển khai một hệ thống học máy, bạn không chỉ đơn thuần cực tiểu hóa hàm đối log hợp lý hay cực đại hóa độ chính xác mà còn đang tự động hóa một quy trình quyết định nào đó.
Thường thì những hệ thống được tự động hóa việc ra quyết định mà chúng ta triển khai có thể sẽ gây ra những hậu quả cho những ai chịu ảnh hưởng bởi quyết định của nó.
Nếu chúng ta triển khai một hệ thống chẩn đoán y khoa, ta cần biết hệ thống này sẽ hoạt động và không hoạt động với những ai.
Bỏ qua những rủi ro có thể lường trước được để chạy theo phúc lợi của một bộ phận dân số sẽ đi ngược lại những nguyên tắc đạo đức cơ bản.
Ngoài ra, "độ chính xác" hiếm khi là một thước đo đúng.
Khi chuyển những dự đoán thành hành động, chúng ta thường để ý đến chi phí tiềm tàng của các loại lỗi khác nhau.
Nếu kết quả phân loại một bức ảnh có thể được xem như một sự phân biệt chủng tộc, trong khi việc phân loại sai sang một lớp khác thì lại vô hại, bạn có thể sẽ muốn cân nhắc cả các giá trị xã hội khi điều chỉnh ngưỡng của hệ thống ra quyết định đó.
Ta cũng muốn cẩn thận về cách những hệ thống dự đoán có thể dẫn đến vòng lặp phản hồi.
Ví dụ, nếu hệ thống dự đoán được áp dụng theo cách ngây ngô để dự đoán các hành động phi pháp và theo đó phân bổ sĩ quan tuần tra, một vòng luẩn quẩn có thể xuất hiện.
Một khu xóm có nhiều tội phạm hơn sẽ có nhiều sĩ quan tuần tra hơn, phát hiện ra nhiều tội phạm hơn, thêm nhiều dữ liệu huấn luyện, nhận được dự đoán tốt hơn, dẫn đến nhiều sĩ quan tuần tra hơn, và càng nhiều tội ác được phát hiện,...
Thêm vào đó, chúng ta cũng muốn cẩn thận ngay từ đầu về việc chúng ta có đang giải quyết đúng vấn đề hay không.  
Hiện tại, các thuật toán dự đoán đóng một vai trò lớn khi làm bên trung gian trong việc phân tán thông tin.
Những tin tức nào được hiển thị đến người dùng có nên được quyết định bởi những trang Facebook nào mà họ *đã thích* hay không?
Đây chỉ là một số trong rất nhiều vấn đề về đạo đức mà bạn có thể bắt gặp trong việc theo đuổi sự nghiệp học máy của mình.

<!--
## Summary
-->
## Tóm tắt

<!--
* In many cases training and test set do not come from the same distribution. This is called covariate shift.
* Covariate shift can be detected and corrected if the shift is not too severe. Failure to do so leads to nasty surprises at test time.
* In some cases the environment *remembers* what we did and will respond in unexpected ways. We need to account for that when building models.
-->

* Trong nhiều trường hợp, tập huấn luyện và tập kiểm tra không được lấy mẫu từ cùng một phân phối. Đây là hiện tượng dịch chuyển hiệp biến.
* Dịch chuyển hiệp biến có thể được phát hiện và khắc phục nếu sự dịch chuyển không quá nghiêm trọng. Thất bại trong việc khắc phục có thể dẫn đến những kết quả không lường được lúc kiểm thử.
* Trong nhiều trường hợp, môi trường sẽ ghi nhớ những gì chúng ta đã làm và sẽ phản hồi theo những cách không lường trước được. Chúng ta cần xem xét điều này khi xây dựng mô hình.

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

1. Điều gì có thể xảy ra khi chúng ta thay đổi hành vi của công cụ tìm kiếm? Người dùng có thể sẽ làm gì? Còn các nhà quảng cáo thì sao?
2. Xây dựng một chương trình phát hiện dịch chuyển hiệp biến. Gợi ý: hãy xây dựng một hệ thống phân lớp.
3. Xây dựng một chương trình khắc phục vấn đề dịch chuyển hiệp biến.
4. Chuyện tồi tệ gì có thể xảy ra nếu tập huấn luyện và tập kiểm tra rất khác nhau? Chuyện gì sẽ xảy ra đối với trọng số mẫu?

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
* Lý Phi Long
* Lê Khắc Hồng Phúc
* Nguyễn Duy Du
* Phạm Minh Đức
* Lê Cao Thăng
* Nguyễn Minh Thư
* Nguyễn Thành Nhân
* Phạm Hồng Vinh
* Vũ Hữu Tiệp
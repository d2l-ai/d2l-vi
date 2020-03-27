<!-- ===================== Bắt đầu dịch Phần 1 ===================== -->
<!-- ========================================= REVISE PHẦN 1 - BẮT ĐẦU =================================== -->

<!--
# Considering the Environment
-->
# Xem xét môi trường
# *dịch tiêu đề phía trên*

<!--
So far, we have worked through a number of hands-on implementations fitting machine learning models to a variety of datasets.
And yet, until now we skated over the matter of where are data comes from in the first place, and what we plan to ultimately *do* with the outputs from our models.
Too often in the practice of machine learning, developers rush ahead with the development of models tossing these fundamental considerations aside.
-->
Ở những phần trước, chúng ta đã tiến hành triển khai các môt hình học máy trên những bộ dữ liệu khác nhau.
Tuy nhiên, cho đến bây giờ chúng ta đã bỏ qua việc tìm hiểu những dữ liệu này đến từ đâu và mục đích sau cùng của việc *sử dụng* những kết quả thu được từ mô hình của chúng ta.
Trong thực tế, những nhà phát triển thường tập trung phát triển những mô hình mà gạt bỏ đi những cân nhắc cơ bản này sang một bên.

*dịch đoạn phía trên*

<!--
Many failed machine learning deployments can be traced back to this situation.
Sometimes the model does well as evaluated by test accuracy only to fail catastrophically in the real world when the distribution of data suddenly shifts.
More insidiously, sometimes the very deployment of a model can be the catalyst which perturbs the data distribution.
Say for example that we trained a model to predict loan defaults, finding that the choice of footware was associated with risk of default (Oxfords indicate repayment, sneakers indicate default).
We might be inclined to thereafter grant loans to all applicants wearing Oxfords and to deny all applicants wearing sneakers.
But our ill-conceived leap from pattern recognition to decision-making and our failure to think critically about the environment might have disastrous consequences.
For starters, as soon as we began making decisions based on footware, customers would catch on and change their behavior.
Before long, all applicants would be wearing Oxfords, and yet there would be no coinciding improvement in credit-worthiness.
Think about this deeply because similar issues abound in the application of machine learning: by introducing our model-based decisions to the environment, we might break the model.
-->
Nhiều dự án triển khai học máy không thành công có thể bắt nguồn từ lý do này.
Đôi khi một mô hình đạt được chất lượng tốt trên bộ kiểm thử nhưng lại thất bại đối với những dữ liệu thức tế bởi vì phân phối dữ liệu đột nhiên thay đổi.  
Đặc biệt hơn,đôi khi chính việc triển khai một mô hình có thể là chất xúc tác gây nhiễu cho việc phân phối dữ liệu.
Ví dụ, chúng ta đã đào tạo một mô hình dự đoán vỡ nợ, cho biết rằng việc loại giày mà người vay tiền có liên quan tới rủi ro vỡ nợ (những người mang giày Oxfords có thể chi trả nợ, những người mang giày thể thao có nguy cơ vỡ nợ).
Chúng ta có thể có xu hướng cấp các khoảng vay cho những người đăng ký vay mang giày Oxfords và từ chối tất cả những người mang giày thể thao.
Tuy nhiên, những nhận thức mù quang cũng như sai lệnh từ qúa trình nhận dạng mẫu cho đến việc đưa ra quyết định và việc chúng ta không suy nghĩ nghiêm túc về môi trường có thế gây ra hậu quả tai hại.
Đối với những ngươi mới bắt đầu, ngay khi chúng ta bắt đầu đưa ra những quyết định liên quan đến loại giày, khách hàng có thể nắm bắt và thay đổi hành vi của họ.
Chẳng bảo lâu, tất cả những người đăng ký vay sẽ mang giày Oxfords, và do đó sẽ không có sự cải thiện phù hợp nào trong quá trình lựa chọn đối tượng cho vay.
Hãy suy nghĩ kĩ hơn về ví dụ này, bời vì những vẫn đề tương tự có rất nhiều trong ứng dụng học máy: bằng cách giới thiệu các quyết định dựa trên mô hình của chúng tôi liên quan với môi trường, chúng ta có thể phá vỡ mô hình. 
*dịch đoạn phía trên*

<!--
In this section, we describe some common concerns and aim to get you started acquiring the critical thinking that 
you will need in order to detect these situations early, mitigate the damage, and use machine learning responsibly.
Some of the solutions are simple (ask for the "right" data) some are technically difficult (implement a reinforcement learning system), 
and others require that we enter the realm of philosophy and grapple with difficult questions concerning ethics and informed consent.
-->
Trong phần này, chúng tôi mô tả một số mối quan tâm chung và nhằm mục đích giúp bạn bắt đầu có được tư duy phản biện rằng bạn sẽ cần phát hiện sớm những tình huống này qua đó làm giảm thiệt hại và áp dụng học máy một cách có trách nhiệm.
Một số giải pháp rất đơn giản (yêu cầu "đúng" dữ liệu), một số lại là những kỹ thuật phức tạp (implement a reinforcement learning system), và một số khác yêu cầu chúng ta có những kiến thức liên quan đến triết học, thậm chí phải vật lộn với những câu hỏi khó trong vấn đề đạo đức.   
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 1 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 2 ===================== -->

<!--
## Distribution Shift
-->
## Distribution Shift 
## *dịch tiêu đề phía trên*

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
Để bắt đầu, chúng ta quay trở lại vai trò quan sát, tạm gác lại những tác động đến từ hành động của chúng ta đối với môi trường.
Trong các phần sau, chúng ta xem xét sâu hơn về các cách khác nhau mà phân phối dữ liệu có thể thay đổi và những cách có thể thực hiện để cứu vãn hiệu suất của của mô hình.
Ngay từ đầu, chúng ta nên cảnh báo rằng nếu phân phối tạo dữ liệu $p(\mathbf{x},y)$ có thể thay đổi theo cách tùy ý tại bất kỳ thời điểm nào, và sau đó việc học một bộ phân lớp mạnh mẽ là điều hoàn toàn không thể.
Trong trường hợp xấu nhất, nếu bản thân định nghĩa của nhãn có thể thay đổi tại trong một khoảnh khắc:
nếu đột nhiên những đối tượng chúng ta gọi là "mèo" bây giờ lại là chó và những đối tượng trước đó chúng ta gọi là "chó" giờ thực tế lại là mèo, không có bất kỳ thay đổi rõ ràng nào trong việc phân phối của những đầu vào $p(\mathbf{x})$, sau đó chúng ta sẽ không thể làm gì để có thể phát hiện sự thay đổi hoặc sửa lỗi bộ phân loại của chúng ta tại thời điểm kiểm tra.
May mắn thay, theo một số giả định hạn chế về cách dữ liệu của chúng ta có thể thay đổi trong tương lai, các thuật toán nguyên tắc có thể phát hiện sự thay đổi và thậm chí có thể thích nghi và đạt được độ chính xác cao hơn nếu chúng ta tiếp tục dựa vào bộ phân loại ban đầu.
*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 1 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 2 - BẮT ĐẦU ===================================-->

<!--
### Covariate Shift
-->
### Covariate Shift
### *dịch tiêu đề phía trên*

<!--
One of the best-studied forms of distribution shift is *covariate shift*.
Here we assume that although the distribution of inputs may change over time, the labeling function, i.e., the conditional distribution $P(y \mid \mathbf{x})$ does not change.
While this problem is easy to understand its also easy to overlook it in practice.
Consider the challenge of distinguishing cats and dogs.
Our training data consists of images of the following kind:
-->
Một trong những hình thức thay đổi phân phối được nghiên cứu nhiều nhất đó chính là *Covariate Shift*.
Ở đây chúng tôi giả định rằng mặc dù phân phối đầu vào có thể thay đổi theo thời gian, chức năng gắn nhãn, có nghĩa là phân phối điều kiện $P(y \mid \mathbf{x})$ không thay đổi.
Trong khi vấn đề này là dễ hiểu, nó cũng dễ dàng bỏ qua nó trong thực tế.
Xem xét các thách thức trong quá trình phân biệt chó và meo. 
Dữ liệu đào tạo của chúng tôi bao gồm những hình ảnh của các loại sau:
*dịch đoạn phía trên*

<!--
|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|
-->
|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|
*dịch đoạn phía trên*

<!--
At test time we are asked to classify the following images:
-->
Tại thời điểm thử nghiệm, chúng tôi được yêu cầu phân loại các hình ảnh sau:
*dịch đoạn phía trên*

<!--
|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|
-->
|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|
*dịch đoạn phía trên*

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
Rõ ràng điều này không có khả năng làm việc tốt.
Bộ huấn luyện bao gồm những hình ảnh trong thực tế, trong khi bộ thử nghiệm chỉ chứa những hình ảnh được lấy từ phim hoạt hình.
Màu sắc thậm chí không thực tế.
Đào tạo về một tập dữ liệu trông khác biệt đáng kể so với tập kiểm tra mà không có kế hoạch nào về cách thích ứng với một miền mới là một ý tưởng tồi.
Thật không may, đây là một cạm bẫy rất phổ biến.
Các nhà thống kê gọi đây là *covariate shift* bởi vì nguồn gốc của vấn đề là do sự thay đổi trong phân phối các đặc trưng (cụ thể là  của *covariates*).
Mathematically, chúng ta có thể nói rằng $P(\mathbf{x})$ thay đổi nhưng $P(y \mid \mathbf{x})$ vẫn không thay đổi.
Mặc dù sự hữu ích của nó không bị hạn chế trong cài trường hợp này, khi chúng ta tin $\mathbf{x}$ gây ra $y$, covariate shift thường là giả định đúng đắn để làm việc. 

*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 2 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 3 ===================== -->

<!--
### Label Shift
-->
### Label Shift
### *dịch tiêu đề phía trên*

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
Vấn đề ngược lại xuất hiện khi chúng tôi tin rằng những gì thúc đẩy sự thay đổi là một sự thay đổi trong phân phối biên các nhãn $P(y)$ nhưng phân phối lớp điều kiện là bất biến $P(\mathbf{x} \mid y)$.
Label shift là một giả định hợp lý để thực hiện khi chúng tôi tin rằng $y$ gây ra $\mathbf{x}$.
Ví dụ, thông thường chúng ta muốn dự đoán một chẩn đoán thông qua các biểu hiện.
Trong trường hợp này, chúng tôi tin rằng việc chẩn đoán gây ra các biểu hiện, tức là bệnh gây ra các triệu chứng. 
Đôi khi các giả định label shift và covariate shift có thể giữ đồng thời.
Ví dụ, khi chức năng ghi nhãn thực sự được xác định và không thay đổi, thì covariate shift sẽ luôn giữ, bao gồm cả việc giữ label shift.
Thú vị thay, khi chúng ta mong đợi dich chuyển label shift và covariate shift cùng giữ, nó thường thuận lợi khi làm việc với những phương pháp đi theo từ giả định label shift.
Đó là bởi vì các phương thức này có xu hướng liên quan đến việc thao túng các đối tượng trông giống như nhãn, mà (trong học sâu) có xu hướng tương đối dễ dàng so với làm việc với các đối tượng trông giống như đầu vào, cũng như các đối tượng đa chiều.
*dịch đoạn phía trên*



<!--
### Concept Shift
-->
### Concept Shift
### *dịch tiêu đề phía trên*

<!--
One more related problem arises in *concept shift*, the situation in which the very label definitions change.
This sounds weird—after all, a *cat* is a *cat*.
Indeed the definition of a cat might not change, but can we say the same about soft drinks?
It turns out that if we navigate around the United States, shifting the source of our data by geography, 
we will find considerable concept shift regarding the definition of even this simple term as shown in :numref:`fig_popvssoda`.
-->
Thêm một vấn đề liên quan phát sinh trong *concept shift*, đó là tình huống khi mà định nghĩa của nhãn thay đổi.
Điều này nghe có vẻ kì cục, a con *mèo* vẫn là một con *mèo*.
Thật vậy, định nghĩa của một con mèo có thể không thay đổi, nhưng chúng ta có thể nói như vậy về nước ngọt không? 
Điều đó chỉ ra rằng nếu chúng ta điều hướng trên khắp Hoa Kỳ, thay đổi nguồn dữ liệu của chúng ta theo địa lý, chúng ta sẽ tìm thấy concept shift liên quan đáng kể đến định nghĩa của thuật ngữ đơn giản như:numref:`fig_popvssoda`.
*dịch đoạn phía trên*

<!--
![Concept shift on soft drink names in the United States.](../img/popvssoda.png)
-->
![Concept shift đối với tên nước giải khát ở Hoa Kỳ.](../img/popvssoda.png)
![*dịch chú thích ảnh phía trên*](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

<!--
If we were to build a machine translation system, the distribution $P(y \mid x)$ might be different depending on our location.
This problem can be tricky to spot.
A saving grace is that often the $P(y \mid x)$ only shifts gradually.
-->
Nếu chúng ta xây dựng một hệ thống dịch máy, phân phối $P(y \mid x)$ có thể sẽ khác phụ thuộc vào vị trí của chúng ta.
Vấn đề này có thế khó khăn để phát hiện.
Một điều cần chú ý đó là $P(y \mid x)$ chỉ thay đổi từ từ.
*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 2 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 3 - BẮT ĐẦU ===================================-->

<!--
### Examples
-->
### Các ví du
### *dịch tiêu đề phía trên*

<!--
Before we go into further detail and discuss remedies, we can discuss a number of situations where covariate and concept shift may not be so obvious.
-->
Trước khi chúng ta đi vào chi tiết và thảo luận về các biện pháp khắc phục, chúng ta có thể thảo luận về một số tình huống trong đó covariate shift và concept shift có thể không quá rõ ràng.
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 3 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 4 ===================== -->

<!--
#### Medical Diagnostics
-->
#### Chẩn đoán y khoa
#### *dịch tiêu đề phía trên*

<!--
Imagine that you want to design an algorithm to detect cancer.
You collect data from healthy and sick people and you train your algorithm.
It works fine, giving you high accuracy and you conclude that you’re ready for a successful career in medical diagnostics.
Not so fast...
-->
Hãy tưởng tượng rằng bạn muốn thiết kế một thuật toán để phát hiện bệnh ung thư.
Bạn thu thập dữ liệu từ những người khỏe mạnh và bị bênh để huấn luyện cho thuật toán của mình.
Nó hoạt động tốt, đạt được độ chính xác cao và bạn kết luận rằng bạn đã sẵn sàng cho sự nghiệp thành công trong lĩnh vực chẩn đoán y khoa.
Thật ra không nhanh như vậy...
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
Nhiều thứ có thể sai.
Cụ thể, những phân phối mà bạn gặp trong quá trình đào tạo và những phân phối mà bạn gặp trong thực tế có thể khác nhau đáng kể.
Không may, điều này đã xảy ra đối với startup Alex.
Họ đã phát triển một hê thống xét nghiệm máu cho một căn bệnh ảnh hưởng chủ yếu đến những người đàn ông lớn tuổi và họ đã thu được lượng mẫu máu từ những bệnh nhân.
Tuy nhiên, rất khó khăn hơn để lấy mẫu máu từ những người đàn ông khỏe mạnh (chủ yếu vì lý do đạo đức).
Thay vào đó, họ đã yêu cầu một số lượng lớn sinh viên trong trường hiến máu và họ đã thực hiện bài kiểm tra của mình.
Sau đó, họ hỏi tôi có thể giúp họ xây dựng một bộ phân loại để phát hiện bệnh không.
Tôi nói với họ rằng sẽ rất dễ dàng để phân biệt giữa cả hai bộ dữ liệu với độ chính xác gần như hoàn hảo.
Rốt cuộc, các đối tượng thử nghiệm khác nhau về tuổi tác, nồng độ hormone, hoạt động thể chất, chế độ ăn uống, nồng độ cồn và nhiều yếu tố khác không liên quan đến căn bệnh này.
Điều này không giống với một bệnh nhân trong thực tế.
Quy trình lấy mẫu của họ làm tăng khả năng xảy một trường hợp cực đoan của covariate shift sẽ phát sinh giữa các phân phối *nguồn* và *đích*, và lúc đó chúng ta không thể sửa chữa lỗi lầm bằng các phương pháp thông thường.
Nói một cách khác, dữ liệu đào tạo và kiểm tra khác nhau đến mức không có gì hữu ích để có thể thực hiện và họ đã lãng phí số tiền đáng kể.
*dịch đoạn phía trên*

<!--
#### Self Driving Cars
-->
#### Xe tự hành
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
Có một công ty muốn xây dựng một hệ thống học máy cho xe tự hành.
Một trong những thành phần chính của hệ thống này là một máy dò đường.
Bởi vì dữ liệu đã được gắn nhãn tốn rất nhiều chi phí để thu được, họ đã có một ý tưởng (thông minh và không chắc chắn) đó là sử dụng dữ liệu tổng hợp từ một công cụ kết xuất trò chơi như dữ liệu đào tạo bổ sung.
Điều này hoạt động thực sự tốt trên "dữ liệu thử nghiệm" được rút ra từ công cụ kết xuất.
Tuy nhiên, trong thực tế thì việc áp cách trến thì đúng là một thảm họa.
Cụ thể, lề đường đã được kết xuất với một kết cấu rất đơn giản.
Quan trọng hơn, *tất cả* lề đường đã được kết xuất với kết cấu *giống nhau* và bộ phát hiện lê đường đã học những "đặc trưng" này một cách nhanh chóng.
*dịch đoạn phía trên*

<!--
A similar thing happened to the US Army when they first tried to detect tanks in the forest.
They took aerial photographs of the forest without tanks, then drove the tanks into the forest and took another set of pictures.
The so-trained classifier worked "perfectly".
Unfortunately, all it had learned was to distinguish trees with shadows from trees without shadows---the first set of pictures was taken in the early morning, the second one at noon.
-->
Một điều tương tự đã xảy ra với Quân đội Hoa Kỳ trong lần đầu tiên cố gắng phát hiện xe tăng trong rừng.
Họ chụp ảnh từ trên không khu rừng không có xe tăng, sau đó lái xe tăng vào rừng và chụp một bộ ảnh khác.
Bộ phân loại đã được đào tạo để làm việc một cách "hoàn hảo".
Thật không may, tất cả những gì nó đã học được là phân biệt cây có bóng với cây không có bóng---bộ ảnh đầu tiên được chụp vào sáng sớm, bộ thứ hai vào buổi trưa.
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 4 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 5 ===================== -->

<!--
#### Nonstationary distributions
-->
#### Phân phối không cố định
#### *dịch tiêu đề phía trên*

<!--
A much more subtle situation arises when the distribution changes slowly and the model is not updated adequately.
Here are some typical cases:
-->
Một tình huống tinh tế hơn nhiều phát sinh khi phân phối thay đổi chậm và mô hình không được cập nhật đầy đủ.
Dưới đây là một số trường hợp điển hình:
*dịch đoạn phía trên*

<!--
* We train a computational advertising model and then fail to update it frequently (e.g., we forget to incorporate that an obscure new device called an iPad was just launched).
* We build a spam filter. It works well at detecting all spam that we have seen so far. But then the spammers wisen up and craft new messages that look unlike anything we have seen before.
* We build a product recommendation system. It works throughout the winter... but then it keeps on recommending Santa hats long after Christmas.
-->
* Chúng tôi đào tạo một mô hình quảng cáo tính toán và sau đó không cập nhật nó thường xuyên (chúng tôi quên kết hợp việc một thiết bị mới được gọi là iPad vừa được ra mắt).
* Chúng tôi xây dựng bộ lọc thư rác. Nó hoạt động tốt trong việc phát hiện tất cả các thư rác mà chúng ta đã thấy cho đến nay. Nhưng sau đó, những kẻ gửi thư rác đã thông minh hơn và tạo ra những thông điệp mới trông không giống bất cứ thứ gì chúng tôi đã thấy trước đây.
* Chúng tôi xây dựng một hệ thống khuyến nghị sản phẩm. Nó hoạt động suốt mùa đông... nhưng sau đó nó tiếp tục giới thiệu mũ ông già Noel sau Giáng sinh.

*dịch đoạn phía trên*

<!--
#### More Anecdotes
-->
#### More Anecdotes
#### *dịch tiêu đề phía trên*

<!--
* We build a face detector. It works well on all benchmarks. 
Unfortunately it fails on test data---the offending examples are close-ups where the face fills the entire image (no such data was in the training set).
* We build a web search engine for the USA market and want to deploy it in the UK.
* We train an image classifier by compiling a large dataset where each among a large set of classes is equally represented in the dataset, 
say 1000 categories, represented by 1000 images each. Then we deploy the system in the real world, where the actual label distribution of photographs is decidedly non-uniform.
-->
* Chúng tôi xây dựng một hệ thống phát hiện khuôn mặt. Nó hoạt động tốt trên tất cả các điểm chuẩn.
* Chúng tôi xây dựng một công cụ tìm kiếm web cho thị trường Hoa Kỳ và muốn triển khai nó ở Anh.
* Chúng tôi đào tạo một trình phân loại hình ảnh bằng cách biên dịch một tập dữ liệu lớn, trong đó mỗi tập hợp lớn của các lớp được biểu diễn bằng nhau trong tập dữ liệu, cụ thể có 1000 loại được biểu diễn bởi một 1000 bức ảnh tương ứng. Sau đó, chúng tôi triển khai hệ thống trong thực tế, nơi phân phối nhãn thực tế của các bức ảnh cho thấy là không đồng nhất.
*dịch đoạn phía trên*

<!--
In short, there are many cases where training and test distributions $p(\mathbf{x}, y)$ are different.
In some cases, we get lucky and the models work despite covariate, label, or concept shift.
In other cases, we can do better by employing principled strategies to cope with the shift.
The remainder of this section grows considerably more technical.
The impatient reader could continue on to the next section as this material is not prerequisite to subsequent concepts.
-->
Nói tóm lại, có nhiều trường hợp phân phối đào tạo và kiểm tra $p(\mathbf{x}, y)$ là khác nhau.
Trong một số trường hợp, chúng tôi gặp may mắn và các mô hình vẫn hoạt động mặc dù gặp các trường hợp covariate, label, hoặc concept shift.
Trong các trường hợp khác, chúng tôi có thể làm tốt hơn bằng cách sử dụng các thay đổi và chỉnh sửa để đối phó.
Phần còn lại của bài này sẽ cung cấp nhiều kĩ thuật đáng kể hơn.
Bạn đọc có thể bỏ qua phần này và đọc phần tiếp theo vì phần này không phải là điều kiện tiên quyết cho những nội dung tiếp theo.
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 5 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 6 ===================== -->

<!-- ========================================= REVISE PHẦN 3 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 4 - BẮT ĐẦU ===================================-->

<!--
### Covariate Shift Correction
-->
### Điều chỉnh Covariate Shift
### *dịch tiêu đề phía trên*

<!--
Assume that we want to estimate some dependency $P(y \mid \mathbf{x})$ for which we have labeled data $(\mathbf{x}_i, y_i)$.
Unfortunately, the observations $x_i$ are drawn from some *target* distribution $q(\mathbf{x})$ rather than the *source* distribution $p(\mathbf{x})$.
To make progress, we need to reflect about what exactly is happening during training: 
we iterate over training data and associated labels $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ and update the weight vectors of the model after every minibatch.
We sometimes additionally apply some penalty to the parameters, using weight decay, dropout, or some other related technique.
This means that we largely minimize the loss on the training.
-->
Giả sử rằng chúng ta muốn ước tính một số phụ thuộc $P(y \mid \mathbf{x})$ cho những dữ liệu mà chúng ta đã gắn nhãn $(\mathbf{x}_i, y_i)$.
Thật không may, các quan sát $x_i$ được rút ra từ một số phân phối *mục tiêu* $q(\mathbf{x})$ thay vì phân phối *nguồn* $p(\mathbf{x})$.
Để cải tiến, chúng ta cần suy nghĩ chính xác về những gì đang xảy ra trong quá trình đào tạo:
chúng ta lặp đi lặp lại qua  dữ liệu đào tạo và các nhãn liên quan $\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$ và cập nhật vector trọng số của mô hình sau mỗi minibatch.
Đôi khi chúng ta cũng áp dụng một số hình phạt cho các tham số, sử dụng phân rã trọng lượng, dropout hoặc một số kỹ thuật liên quan khác.
Điều này có nghĩa là chúng ta giảm thiểu phần lớn mất mát trong quá trình đào tạo.
*dịch đoạn phía trên*

$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{some~penalty}(w).
$$

<!--
Statisticians call the first term an *empirical average*, i.e., an average computed over the data drawn from $P(x) P(y \mid x)$.
If the data is drawn from the "wrong" distribution $q$, we can correct for that by using the following simple identity:
-->
Các nhà thống kê gọi thuật ngữ đầu tiên là *trung bình theo kinh nghiệm*, cụ thể đó là trung bình được tính trên dữ liệu rút ra được từ $P(x) P(y \mid x)$.
Nếu dữ liệu được rút ra từ phân phối "sai" $q$, chúng ta có thể sửa lỗi đó bằng cách sử dụng cách nhận biết đơn giản sau:
*dịch đoạn phía trên*

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
Nói cách khác, chúng ta cần tính lại trọng số từng trường hợp theo tỷ lệ xác suất mà nó sẽ được rút ra từ phân phối chính xác $\beta(\mathbf{x}) := p(\mathbf{x})/q(\mathbf{x})$.
Nhưng chúng ta không biết tỷ lệ đó,  vì vậy trước khi chúng ta có thể làm bất cứ điều gì hữu ích, chúng ta cần ước tính nó.
Nhiều phương pháp có sẵn, bao gồm một số cách tiếp cận lý thuyết toán tử cố gắng hiệu điều chỉnh trực tiếp toán tử kỳ vọng bằng cách sử dụng một định mức tối thiểu hoặc nguyên tắc entropy tối đa.
Lưu ý rằng đối với bất kỳ phương pháp như vậy, chúng ta cần các mẫu được rút ra từ cả hai bản phân phối --- "đúng" $p$, cụ thể bằng cách truy cập dữ liệu đào tạo, và dữ liệu được sử dụng để tạo tập huấn luyện $q$.
Tuy nhiên, bởi vì chúng ta chỉ cần mẫu thử $\mathbf{x} \sim q(\mathbf{x})$; chúng tôi không truy cập nhãn $y \sim q(y)$.
*dịch đoạn phía trên*

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
Trong trường hợp này, tồn tại một cách tiếp cận rất hiệu quả sẽ cho kết quả gần như tốt: logistic regression.
Đây là tất cả những gì cần thiết để tính toán tỷ lệ xác suất ước tính.
Chúng ta tìm kiếm một bộ phân loại để phân biệt giữa dữ liệu được rút ra từ $p(\mathbf{x})$ và dữ liệu được rút ra từ $q(x)$.
Nếu không thể phân biệt giữa hai bản phân phối thì điều đó có nghĩa là các trường hợp liên quan có khả năng đến từ một trong hai bản phân phối.
Nói cách khác, bất kỳ trường hợp nào cũng có thể được ưu tiên và được đánh trọng số cao hơn và ngược lại.
Để đơn giản, giả sử rằng chúng tôi có số lượng trường hợp bằng nhau từ cả hai bản phân phối, được biểu thị bởi $\mathbf{x}_i \sim p(\mathbf{x})$ và $\mathbf{x}_i' \sim q(\mathbf{x})$ tương ứng.
Bây giờ biểu thị bởi nhãn $z_i$ là 1 cho dữ liệu được rút ra từ $p$ và -1 cho dữ liệu được rút ra từ $q$.
Sau đó, xác suất trong một tập dữ liệu kết hợp được đưa ra bởi
*dịch đoạn phía trên*

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ and hence } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

<!--
Hence, if we use a logistic regression approach where $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-f(\mathbf{x}))}$ it follows that
-->
Do đó, nếu chúng ta sử dụng một phương pháp logistic regression trong đó $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-f(\mathbf{x}))}$ nó sẽ theo sau đó
*dịch đoạn phía trên*

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
Do đó, chúng ta cần giải hai vấn đề: đầu tiên để phân biệt dữ liệu được lấy ra từ cả hai phân phối, và sau đó là vấn đề tối thiểu hóa đánh lại trọng số $\beta$, cụ thể là thông qua các gradient đầu.
Đây là một thuật toán nguyên mẫu cho mục đích sử dụng tập huấn luyện $X$ và tập kiểm thử chưa được gán nhãn $Z$.
*dịch đoạn phía trên*

<!--
1. Generate training set with $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$.
2. Train binary classifier using logistic regression to get function $f$.
3. Weigh training data using $\beta_i = \exp(f(\mathbf{x}_i))$ or better $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$.
4. Use weights $\beta_i$ for training on $X$ with labels $Y$.
-->
1. Sinh tập huấn luyện với $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$.
2. Huấn luyện một bộ phân lớp nhị phân sử dụng logistic regression để đạt được hàm $f$.
3. Đánh trọng số dữ liệu huấn luyện sử dụng $\beta_i = \exp(f(\mathbf{x}_i))$ hoặc tốt hơn nếu dùng $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$.
4. Sử dụng trọng số $\beta_i$ cho huấn luyện trên $X$ với nhãn $Y$.
*dịch đoạn phía trên*

<!--
Note that this method relies on a crucial assumption.
For this scheme to work, we need that each data point in the target (test time)distribution had nonzero probability of occurring at training time.
If we find a point where $q(\mathbf{x}) > 0$ but $p(\mathbf{x}) = 0$, then the corresponding importance weight should be infinity.
-->
Chú ý rằng phương pháp này phụ thuộc vào một giả thuyết rất quan trọng.
Để mô hình này hoạt động, chúng ta cần mỗi điểm dữ liệu trên phân phối đích (tại thời điểm kiểm thử) có không có xác suất nào bằng không tại thời điểm huấn luyện.
Nếu chúng ta tìm ra được một điểm mà tại đó $q(\mathbf{x}) > 0$ nhưng $p(\mathbf{x}) = 0$, thì khi đó trọng số quan trọng tương ứng nên bằng vô cực.
*dịch đoạn phía trên*

<!--
*Generative Adversarial Networks* use a very similar idea to that described above to engineer a *data generator* that outputs data that cannot be distinguished from examples sampled from a reference dataset.
In these approaches, we use one network, $f$ to distinguish real versus fake data and a second network $g$ that tries to fool the discriminator $f$ into accepting fake data as real.
We will discuss this in much more detail later.
-->
*Generative Adversarial Networks* sử dụng ý tưởng tương tự với ý tưởng đã nhắc đến ở trên để tạo một *trình sinh dữ liệu"* tạo ra dữ liệu không thể phân biệt được với các mẫu được lấy từ tập dữ liệu tham chiếu.
Theo cách tiếp cận này, chúng tôi sử dụng một mạng $f$ để phân biệt dữ liệu thực với dữ liệu được làm giả và mạng thứ hai $g$ cố gắng đánh lừa bộ phân biệt $f$ để nó chấp nhận dữ liệu giả như là dữ liệu thật.
Chúng tôi sẽ trình bày chi tiết hơn ở phần sau.
*dịch đoạn phía trên*

<!-- ========================================= REVISE PHẦN 4 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 5 - BẮT ĐẦU ===================================-->

<!--
### Label Shift Correction
-->
### Điều chỉnh Label Shift 
### *dịch tiêu đề phía trên*

<!--
For the discussion of label shift, we will assume for now that we are dealing with a $k$-way multiclass classification task.
When the distribution of labels shifts over time $p(y) \neq q(y)$ but the class-conditional distributions stay the same $p(\mathbf{x})=q(\mathbf{x})$, 
our importance weights will correspond to the label likelihood ratios $q(y)/p(y)$.
One nice thing about label shift is that if we have a reasonably good model (on the source distribution) 
then we can get consistent estimates of these weights without ever having to deal with the ambient dimension 
(in deep learning, the inputs are often high-dimensional perceptual objects like images, 
while the labels are often easier to work, say vectors whose length corresponds to the number of classes).
-->
Để trình bày về label shift, chúng tôi giả sử rằng bây giờ chúng tôi đang giải quyết với việc phân loại nhiều lớp k-way.
Khi phân phối của label shift theo thời gian $p(y) \neq q(y)$ nhưng phân phối lớp có điều kiện vẫn giữ nguyên $p(\mathbf{x})=q(\mathbf{x})$, trọng số sự quan trọng tương ứng với tỉ lệ khả năng nhãn $q(y)/p(y)$.
Có một điều tốt về quá trình label shift đó là nếu chúng ta có một mô hình tốt (trên tập phân phối nguồn) khi đó chúng ta có thể dự đoán được những trọng số này mà không phải đụng tới các chiều khác (trong học sau, đầu vào thường có chiều dữ liệu lớn như hình ảnh, trong khi nhãn thường dễ xử lí hơn, ví dụ một vector có chiều dài ứng với số lớp).
*dịch đoạn phía trên*

<!--
To estimate calculate the target label distribution, we first take our reasonably good off the shelf classifier 
(typically trained on the training data) and compute its confusion matrix using the validation set (also from the training distribution).
The confusion matrix C, is simply a $k \times k$ matrix where each column corresponds to the *actual* label and each row corresponds to our model's predicted label.
Each cell's value $c_{ij}$ is the fraction of predictions where the true label was $j$ *and* our model predicted $y$.
-->
Để ước tính phân phối nhãn đích, đầu tiên chúng ta phải có một bộ phân loại tốt (thường đã được huấn luyện trên dữ liệu huấn luyên) và tính confusion matrix của nó sử dụng tập tập kiểm định (cũng từ phân phối tập huấn luyện).
Confusion matrix C, đơn giản là một ma trận $k \times k$ mà mỗi cột tương ứng với nhãn *thực* và mỗi hàng tương ứng với nhãn được mô hình dự đoán.
Giá trị của mỗi ô $c_{ij}$ là tỉ lệ giữ giá nhãn thực $j$ *và* giá trị dự đoán $y$.
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 7 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 8 ===================== -->

<!--
Now we cannot calculate the confusion matrix on the target data directly, because we do not get to see the labels for the examples that we see in the wild, 
unless we invest in a complex real-time annotation pipeline.
What we can do, however, is average all of our models predictions at test time together, yielding the mean model output $\mu_y$.
-->
Bây giờ chúng ta chưa thể tính confusion matrix trên tập dữ liệu đích trực tiếp, bởi vì chúng ta chưa thấy nhãn cho các mẫu mà chúng ta thấy trong tự nhiên, trừ khi chúng ta đầu tư vào một hệ thống chú thích thời gian thực phức tạp.
Tuy nhiên, tất cả những gì chúng ta có thể làm là lấy giá trị trung bình của tất cả dự đoán của các mô hình ở thời điểm kiểm thử, giá trị trung bình nhận được là $\mu_y$.
*dịch đoạn phía trên*

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
Điều đó có nghĩa rằng, dưới một số điều kiện, nếu mô hình phân lớp của chúng ta có độ chính xác chấp nhận được, nếu tập dữ liệu đích chỉ chứa những lớp hình ảnh mà mô hình đã được học trước đó, và nếu giả định về label shift vẫn giữ nguyên, thì chúng ta có thể khôi phục lại phân phối nhãn của tập kiểm thử bởi một hệ thống tuyến tính đơn giản $C \cdot q(y) = \mu_y$.
Nếu hệ thống phân lớp của chúng ta đủ chính xác, khi đó confusion C khả nghịch, nghiệm của phương trình là $q(y) = C^{-1} \mu_y$.
Ở đây chúng tôi thay đổi kí hiệu một chút, $q(y)$ dùng để kí hiệu vector tần suất của nhãn.
Bởi vì chúng tôi nhận ra rằng việc ước tính phân phối $p(y)$ trên nhãn của tập dữ liệu nguồn rất dễ.
Vì vậy với bất kì mẫu huấn luyện $i% với nhãn $y$, chúng ta có thể lấy tỉ lệ ước tính $\hat{q}(y)/\hat{p}(y)$ để tính trọng số $w_i$, và đưa nó vào thuật toán tối thiểu hóa trọng số rủi ro ở trên.
*dịch đoạn phía trên*


<!--
### Concept Shift Correction
-->
### Điều chỉnh Concept Shift
### *dịch tiêu đề phía trên*

<!--
Concept shift is much harder to fix in a principled manner.
For instance, in a situation where suddenly the problem changes from distinguishing cats from dogs to one of distinguishing white from black animals, 
it will be unreasonable to assume that we can do much better than just collecting new labels and training from scratch.
Fortunately, in practice, such extreme shifts are rare.
Instead, what usually happens is that the task keeps on changing slowly.
To make things more concrete, here are some examples:

Concept shift khó để sửa chữa hơn nếu sử dụng các phương pháp thông thường
Ví dụ, trong một tình huống mà bài toán đột ngột thay đổi từ phân biệt chó với mèo thành phân biệt động vật màu trắng và màu đen.
Không hợp lí nếu giả sử rằng chúng ta có thể làm tốt hơn nếu thu thập thêm nhiều nhãn mới và huấn luyện mô hình lại từ đầu.
May mắn cho chúng ta thường thì các trường hợp này rất hiếm khi xảy ra trong thực tế.
Thay vào đó, chúng thay đổi chậm chạp hơn.
Để làm mọi thứ rõ ràng hơn, đây là một vài ví dụ:
-->

*dịch đoạn phía trên*

<!--
* In computational advertising, new products are launched, old products become less popular. 
This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic cameras lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e., most of the news remains unchanged but new stories appear).
-->
* Trong computational advertising, các sản phẩm mới được triển khai, các sản phẩm trở nên ít phổ biến hơn.
Điều này có nghĩa là phân phối trên quảng cáo và sự phổ biến của chúng thay đổi một cách từ từ và bất mô hình dự đoán tỉ lệ nhấp nào cũng cần thay đổi một cách từ từ với nó.
* Ống kính camera giao thông bị phá hủy từ từ vì sự hao mòn gây ra bởi môi trường, làm cho chất lượng hình ảnh cũng giảm dần theo.
* Nội dung tin tức cũng thay đổi từ từ (hầu hết những tin tức vẫn giữ nguyên nhưng những câu chuyện mới xuất hiện).
*dịch đoạn phía trên*

<!--
In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. 
In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.
-->
Đối với những trường hợp trên, chúng ta có thể sử dụng cùng hướng tiếp cận mà chúng ta đã sử cho huấn luyện mô hình để làm cho chúng thích nghi với sự thay đổi của dữ liệu.
Nói cách khác, chúng ta sử dụng mô hình với trọng số có sẵn và thực hiện cập nhật với ít vòng lặp trên dữ liệu mới hơn là huấn luyện lại từ đầu.
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 8 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 9 ===================== -->

<!-- ========================================= REVISE PHẦN 5 - KẾT THÚC ===================================-->

<!-- ========================================= REVISE PHẦN 6 - BẮT ĐẦU ===================================-->

<!--
## A Taxonomy of Learning Problems
-->
## Phân loại các vấn đề học tập
## *dịch tiêu đề phía trên*

<!--
Armed with knowledge about how to deal with changes in $p(x)$ and in $P(y \mid x)$, we can now consider some other aspects of machine learning problems formulation.
-->
Trang bị kiến thức về làm thế nào để giải quyết sự thay đổi trong $p(x)$ và $P(y \mid x)$, chúng ta có thể xem xét một vài khía cạnh khác của hình thành bài toán học máy.
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
* **Batch Learning.** Chúng ta có dữ liệu huấn luyện và nhãn $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, để huấn luyện mô hình $f(x, w)$.
Sau đó, chúng ta dùng mô hình này để tính toán trên dữ liệu mới $(x, y)$ được lấy cùng một phân phối.
Giả định này chúng ta phải luôn đặt ra trong đầu với mỗi bài toán mà được nhắc đến ở đây.
Ví dụ, chúng ta có thể huấn luyện mô hình để nhận diện một con mèo dựa trên rất nhiều hình ảnh của chó và mèo.
Sau khi hoàn tất huấn luyện mô hình, nó được sử dụng trong hệ thống cửa thông minh có thể nhận diện và chỉ cho những con mèo dễ thương đi vào.
Hệ thống được triển khai ở ngôi nhà của những khác hàng và kể từ đó nó không được cập nhật lần nào (trừ một vài trường hợp hiếm hoi).
* **Online Learning.** Thử tưởng tượng dữ liệu $(x_i, y_i)$ được mang đến một mẫu tại một thời điểm.
Giả sử, chúng ta có $x_i$, sau đó chúng ta cần tính $f(x_i, w)$ và sau khi hoàn thành việc này,
chúng ta nhận được một phần thưởng là $y_i$ (hay gánh chịu mất mát), dựa trên quyết định của chúng ta.
Nhiều bài toán thực tế rơi vào loại này.
Cụ thể là chúng ta cần dự đoán giá cổ phiếu ngày mai, cho phép chúng ta giao dịch dựa trên những dự đoán này và vào cuối ngày chúng ta sẽ biết được liệu những dự đoán có cho chúng ta lợi nhuận hay không.
Nói cách khác, chúng ta theo một chu trình, theo đó chúng ta tiếp tục cải thiện mô hình dựa trên những dữ liệu mới.
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
* **Bandits.** Chúng là một *trường hợp đặc biệt* của bài toán trên.
Trong khi nhiều bài toán chúng ta có một hàm tham số hóa liên tục $f$ và chúng ta muốn học những tham số của nó ( mô hình học sâu),
trong bài toán tên cướp chúng ta chỉ có một số lượng vũ khí hữu hạn mà chúng ta có thể kéo (một số lượng hữu hạn hành động mà chúng ta có thể nhận).
Không có gì ngạc nhiên khi mà với bài toán đơn giản hơn này được giải quyết một cách tối ưu bằng những lí thuyết cao cấp hơn.
Chúng tôi nhắc đến bài toán này bởi vì nó thường (nhầm lần) bị xem như là một môi trường học tập khác một cách rõ ràng.
* **Control (and nonadversarial Reinforcement Learning).**
Trong nhiều trường hợp môi trường ghi nhớ những gì chúng ta đã làm.
Không cần thiết theo phương pháp đối kháng nhưng nó sẽ ghi nhớ và phản hồi sẽ phụ thuộc vào những gì đã xảy ra trước đó.
Ví dụ bộ điều khiển nồi hơi trong máy pha cà phê sẽ theo dõi nhiệt độ khác nhau phụ thuộc vào việc nó đã làm nóng nồi hơi trước đó.
Thuật toán điều khiển PID(vi tích phân tỉ lệ) là một sự lựa chọn phổ biến.
Tương tự như vậy, hành vi người dùng trên trang tin tức phụ thuộc vào việc chúng ta đã cho họ xem gì trước đó (họ thường chỉ đọc những tin tức 1 lần).
Nhiều thuật toán tạo một mô hình của môi trường mà chúng hành động như thể để làm cho quyết định của chúng ít xuất hiện tính ngẫu nhiên hơn (để giảm phương sai).
* **Reinforcement Learning.** Trong trường hợp tổng quát hơn của một môi trường với bộ nhớ,
chúng ta bắt gặp tình huống mà môi trường đang cố gắng *hợp tác* với chúng ta (trò chơi hợp tác, đặc biệt các trò chơi có tổng không bằng không)
hoặc môi trường sẽ cố gắng để *chiến thắng*. Cờ vua, Go, Backgammon hay StarCraft là một vài ví dụ điển hình cho trường hợp này.
Tương tự, chúng ta muốn xây dựng một bộ điều khiển tốt cho những chiếc xe tự hành.
Những chiếc xe khác có thể phản hồi theo phong cách lái xe của những chiếc xe tự hành theo những cách không tầm thường,
cụ thể là cố gắng tránh nó, cố gắng không gây ra một vụ tai nạn, cố gắng hợp tác với nó ...
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
Một sự khác biệt to lớn giữa những thứ đã đề cập ở trên là cùng một chiến lược có thể hoạt động xuyên suốt trong trường hợp môi trường tĩnh, có thể không hoạt động nếu môi trường có thể thay đổi để thích nghi.
Ví dụ, một cơ hội làm giàu được phát hiện ra bởi một doanh nhân có thể biến mất một khi anh ta bắt tay vào thực hiện nó.
Tốc độ và phương thức mà trong đó sự thay đổi của môi trường được xác định để mở rộng kiểu thuật toán mà chúng ta có thể sử dụng.
Ví dụ, nếu chúng *biết* những sự việc có thể thay đổi một cách từ từ, chúng ta cũng có thể ép cho những dự đoán thay đổi chậm.
Nếu chúng ta biết môi trường có thể thay đổi ngày lập tức, nhưng không thường xuyên, chúng ta có thể cho điều này xảy ra.
Những kiến thức này đặc biệt quan trọng cho những nhà khoa học dữ liệu để giải quyết concept shift, cụ thể là khi bài toán anh ta đang cố gắng giải thay đổi theo thời gian.
*dịch đoạn phía trên*

<!-- ===================== Kết thúc dịch Phần 9 ===================== -->

<!-- ===================== Bắt đầu dịch Phần 10 ===================== -->

<!--
## Fairness, Accountability, and Transparency in Machine Learning
-->
## Công bằng, trách nhiệm và minh bạch trong máy học
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
Cuối cùng, cần ghi nhớ một điều quan trọng sau đây: khi bạn triển khai một hệ thống học máy, không đơn giản chỉ là tối thiểu negative log likelihood hay tối đa hóa độ chính xác---bạn đang tự động hóa một k
Thông thường, những hệ thống tự động ra quyết định mà chúng tôi triển khai có thể gây ra những kết quả không mong muốn về những quyết định của nó.
Nếu chúng tôi đang triển khai một hệ thống phân tích ý tế, chúng tôi cần biết hệ thống sẽ hoạt động và không hoạt động với những ai.
Bỏ qua những rủi ro có thể thấy trước để chạy theo phúc lợi của một bộ phận dân số sẽ chạy theo những nguyên tắc đạo đức cơ bản.
Ngoài ra, "độ chính xác" hiếm khi là một độ đo đúng.
Khichuyển những dự đoán thành hành động chúng ta thường để ý đến giá trị tiềm năng của độ lỗi theo nhiều cách khác nhau.
Nếu theo một cách nào đó bạn phân loại một bức ảnh có thể được xem như là racial sleight,
trong khi phân loại sai sang một loại khác sẽ vô hại, thì bạn có thể muốn điều chỉnh ngưỡng của mình cho phù hợp, giải thích giá trị xã hội trong thiết kế giao thức ra quyết định.
Chúng tôi cũng muốn cẩn thận về việc làm thế nào mà những hệ thống dự đoán có thể dẫn đến vòng lặp phản hồi.
Ví dụ, nếu hệ thống dự đoán được áp dụng để dự đoán chính sách, phân bổ sĩ quan tuần tra, một vòng luẩn quẩn có thể xuất hiện.
Hàng xóm có nhiều tội phạm, cần nhiều tuần tra, phát hiện ra nhiều tội phạm, thêm nhiều dữ liệu huấn luyện, nhận được dự đoán tốt hơn, dẫn đến nhiều tuần tra hơn, nhiều tội ác được phát hiện, etc. 
Thêm vào đó, chúng tôi muốn cẩn thận về việc liệu chúng tôi có giải quyết đúng vấn đề ngày từ đầu. 
Hiện tại, các thuật toán dự đoán đóng vai trò lớn trong việc làm trung gian cho việc phổ biến thông tin.
Những tin tức mà ai đó được đưa ra sẽ được xác định bởi những trang Facebook nào họ đã * Thích *?
Đây chỉ là một ít trong rất nhiều vấn đề về đạo đức mà bạn có thể bắt gặp trong sự nghiệp theo đuổi học máy của mình.

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
* Trong nhiều trường hợp, tập huấn luyện và kiểm thử không cùng một phân phối. Đây chính là covariate shift.
* Covariate shift có thể nhận diện và sửa chửa nếu sự dịch chuyển không quá nghiêm trọng. Thất bại trong việc sửa chữa có thể dẫn đến những kết quả không lường được lúc kiểm thử.
* Trong nhiều trường hợp, môi trường ghi nhớ những gì chúng ta đã làm và sẽ phản hồi theo một cách không lường trước được. Chúng ta cần xem xét đến khi xây dựng những mô hình.
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
1. Điều gì có thể xảy ra khi chúng ta thay đổi hành vi của công cụ tìm kiếm ? Người dùng có thể làm gì ? Còn các nhà quảng cáo thì sao ?
2. Triển khai một chương trình nhận diện covariate shift. Gợi ý: xây dựng một hệ thống phân lớp.
3. Triển khai một hệ thống sửa chữa covariate shift.
4. Chuyện gì sẽ xảy ra nếu tập huấn luyện và kiểm thử rất khác nhau ? Chuyện gì sẽ xảy ra đối với trọng số mẫu?
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
*Nguyễn Thành Nhân 

<!-- Phần 2 -->
*Nguyễn Thành Nhân 

<!-- Phần 3 -->
*Nguyễn Thành Nhân 

<!-- Phần 4 -->
*Nguyễn Thành Nhân 

<!-- Phần 5 -->
*Nguyễn Thành Nhân 

<!-- Phần 6 -->
*Nguyễn Thành Nhân 

<!-- Phần 7 -->
*Nguyễn Đình Nam

<!-- Phần 8 -->
*Nguyễn Đình Nam

<!-- Phần 9 -->
*Nguyễn Đình Nam

<!-- Phần 10 -->
*Nguyễn Đình Nam

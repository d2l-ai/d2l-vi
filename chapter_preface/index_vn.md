<!-- =================== Bắt đầu dịch Phần 1 ================================-->

<!--
# Preface
-->

# Lời nói đầu

<!--
Just a few years ago, there were no legions of deep learning scientists
developing intelligent products and services at major companies and startups.
When the youngest among us (the authors) entered the field,
machine learning did not command headlines in daily newspapers.
Our parents had no idea what machine learning was,
let alone why we might prefer it to a career in medicine or law.
Machine learning was a forward-looking academic discipline
with a narrow set of real-world applications.
And those applications, e.g., speech recognition and computer vision,
required so much domain knowledge that they were often regarded
as separate areas entirely for which machine learning was one small component.
Neural networks then, the antecedents of the deep learning models
that we focus on in this book, were regarded as outmoded tools.
-->

Chỉ một vài năm trước, không có nhiều nhà khoa học học sâu (_deep learning_) phát triển các sản phẩm và dịch vụ thông minh tại các công ty lớn và công ty khởi nghiệp.
Khi người trẻ nhất trong nhóm tác giả chúng tôi tiến vào lĩnh vực này, học máy (_machine learning_) còn chưa xuất hiện thường xuyên trên truyền thông.
Cha mẹ chúng tôi từng không có ý niệm gì về học máy chứ chưa nói đến việc hiểu tại sao chúng tôi theo đuổi lĩnh vực này thay vì y khoa
hay luật khoa. Học máy từng là một lĩnh vực nghiên cứu với chỉ một tập nhỏ các
ứng dụng thực tế. Và những ứng dụng đó, chẳng hạn nhận dạng giọng nói (_speech recognition_) hay thị giác máy tính (_computer vision_), đòi hỏi quá nhiều kiến thức chuyên biệt khiến chúng thường được phân thành các lĩnh vực hoàn toàn riêng mà trong đó học máy chỉ là một thành phần nhỏ.
Các mạng nơ-ron (_neural network_), tiền đề của các mô hình học sâu mà chúng ta tập trung vào trong cuốn sách này, từng được coi là các công cụ lỗi thời.


<!--
In just the past five years, deep learning has taken the world by surprise,
driving rapid progress in fields as diverse as computer vision,
natural language processing, automatic speech recognition,
reinforcement learning, and statistical modeling.
With these advances in hand, we can now build cars that drive themselves
with more autonomy than ever before (and less autonomy
than some companies might have you believe),
smart reply systems that automatically draft the most mundane emails,
helping people dig out from oppressively large inboxes,
and software agents that dominate the world's best humans
at board games like Go, a feat once thought to be decades away.
Already, these tools exert ever-wider impacts on industry and society,
changing the way movies are made, diseases are diagnosed,
and playing a growing role in basic sciences---from astrophysics to biology.
-->

Trong chỉ khoảng năm năm gần đây, học sâu đã mang đến nhiều bất ngờ trên quy mô toàn cầu, 
dẫn đường cho những tiến triển nhanh chóng trong nhiều lĩnh vực khác
nhau như thị giác máy tính, xử lý ngôn ngữ tự nhiên (_natural language processing_), nhận dạng giọng nói tự động (_automatic speech recognition_),
học tăng cường (_reinforcement learning_), và mô hình hoá thống kê (_statistical modeling_). Với những tiến bộ này, chúng ta bây
giờ có thể xây dựng xe tự lái với mức độ tự động ngày càng cao (nhưng chưa nhiều tới mức như vài công ty đang tuyên bố), hệ thống trả
lời tự động, giúp con người đào sâu vào cả núi email, và các phần mềm chiến
thắng những người giỏi nhất trong các môn cờ như cờ vây, một kỳ tích từng được
xem là không thể đạt được trong nhiều thập kỷ tới. Những công cụ này đã và đang
gây ảnh hưởng rộng rãi tới các ngành công nghiệp và đời sống xã hội, thay đổi cách
tạo ra các bộ phim, cách chẩn đoán bệnh, đóng một vài trò ngày càng tăng trong các
ngành khoa học cơ bản -- từ vật lý thiên văn tới sinh học.

<!-- =================== Kết thúc dịch Phần 1 ================================-->

<!-- =================== Bắt đầu dịch Phần 2 ================================-->

<!--
## About This Book
-->

## Về cuốn sách này

<!--
This book represents our attempt to make deep learning approachable,
teaching you both the *concepts*, the *context*, and the *code*.
-->

Cuốn sách này được viết với mong muốn học sâu dễ tiếp cận hơn,
dạy bạn từ *khái niệm*, *bối cảnh*, tới *lập trình*.

<!--
### One Medium Combining Code, Math, and HTML
-->

### Một phương tiện truyền tải kết hợp Code, Toán, và HTML

<!--
For any computing technology to reach its full impact,
it must be well-understood, well-documented, and supported by
mature, well-maintained tools.
The key ideas should be clearly distilled,
minimizing the onboarding time needing to bring new practitioners up to date.
Mature libraries should automate common tasks,
and exemplar code should make it easy for practitioners
to modify, apply, and extend common applications to suit their needs.
Take dynamic web applications as an example.
Despite a large number of companies, like Amazon,
developing successful database-driven web applications in the 1990s,
the potential of this technology to aid creative entrepreneurs
has been realized to a far greater degree in the past ten years,
owing in part to the development of powerful, well-documented frameworks.
-->

Để bất kỳ kỹ thuật tính toán nào đạt được tầm ảnh hưởng sâu rộng,
nó phải dễ hiểu, có tài liệu đầy đủ, và được hỗ trợ bởi nhưng công
cụ cấp tiến được "bảo trì" thường xuyên.
Các ý tưởng chính cần được chắt lọc rõ ràng,
tối thiểu thời gian chuẩn bị cần thiết để trang bị
kiến thức đương thời cho những người mới bắt đầu.
Các thư viện cấp tiến nên tự động hoá các tác vụ đơn giản,
và các đoạn mã nguồn ví dụ cần phải đơn giản với những người mới bắt đầu
sao cho họ có thể dễ dàng chỉnh sửa, áp dụng,
và mở rộng những ứng dụng thông thường thành các ứng dụng họ cần.
Lấy ứng dụng các trang web động làm ví dụ.
Mặc dù các công ty công nghệ lớn, như Amazon,
phát triển thành công các ứng dụng web
định hướng bởi cơ sở dữ liệu từ những năm 1990, tiềm năng của công
nghệ này để hỗ trợ các doanh nghiệp sáng tạo chỉ được nhân rộng lên ở một tầm cao mới
từ khoảng mười năm nay, nhờ vào sự phát triển của các nền tảng mạnh
mẽ và với tài liệu đầy đủ.

<!--
Testing the potential of deep learning presents unique challenges
because any single application brings together various disciplines.
Applying deep learning requires simultaneously understanding
(i) the motivations for casting a problem in a particular way;
(ii) the mathematics of a given modeling approach;
(iii) the optimization algorithms for fitting the models to data;
and (iv) and the engineering required to train models efficiently,
navigating the pitfalls of numerical computing
and getting the most out of available hardware.
Teaching both the critical thinking skills required to formulate problems,
the mathematics to solve them, and the software tools to implement those
solutions all in one place presents formidable challenges.
Our goal in this book is to present a unified resource
to bring would-be practitioners up to speed.
-->

=======
Kiểm định tiềm năng của học sâu có những thách thức riêng biệt
vì bất kỳ ứng dụng riêng lẻ nào cũng bao gồm nhiều lĩnh vực khác nhau.
Ứng dụng học sâu đòi hỏi những hiểu biết đồng thời
(i) động lực để biến đổi một bài toán theo một hướng cụ thể;
(ii) kiến thức toán học của một hướng tiếp cận mô hình hoá;
(iii) những thuật toán tối ưu cho việc khớp mô hình với dữ liệu;
và (iv) phần kỹ thuật yêu cầu để huấn luyện mô hình một cách hiệu quả,
xử lý những khó khăn trong tính toán và tận dụng thật tốt phần cứng hiện có.
Đào tạo kỹ năng suy nghĩ thấu đáo cần thiết để định hình bài toán,
kiến thức toán để giải chúng, và các công cụ phần mềm để triển khai
những giải pháp đó, tất cả trong một nơi, hàm chứa nhiều thách thức lớn.
Mục tiêu của chúng tôi trong cuốn sách này là trình
bày một nguồn tài liệu tổng hợp giúp những học viên nhanh chóng bắt kịp.


<!--
We started this book project in July 2017 when we needed
to explain MXNet's (then new) Gluon interface to our users.
At the time, there were no resources that simultaneously
(i) were up to date; (ii) covered the full breadth
of modern machine learning with substantial technical depth;
and (iii) interleaved exposition of the quality one expects
from an engaging textbook with the clean runnable code
that one expects to find in hands-on tutorials.
We found plenty of code examples for
how to use a given deep learning framework
(e.g., how to do basic numerical computing with matrices in TensorFlow)
or for implementing particular techniques
(e.g., code snippets for LeNet, AlexNet, ResNets, etc)
scattered across various blog posts and GitHub repositories.
However, these examples typically focused on
*how* to implement a given approach,
but left out the discussion of *why* certain algorithmic decisions are made.
While some interactive resources have popped up sporadically
to address a particular topic, e.g., the engaging blog posts
published on the website [Distill](http://distill.pub), or personal blogs,
they only covered selected topics in deep learning,
and often lacked associated code.
On the other hand, while several textbooks have emerged,
most notably :cite:`Goodfellow.Bengio.Courville.2016`,
which offers a comprehensive survey of the concepts behind deep learning,
these resources do not marry the descriptions
to realizations of the concepts in code,
sometimes leaving readers clueless as to how to implement them.
Moreover, too many resources are hidden behind the paywalls
of commercial course providers.
-->

Chúng tôi bắt đầu dự án sách này từ tháng 7/2017 khi cần trình bày
giao diện MXNet Gluon (khi đó còn mới) tới người dùng.
Tại thời điểm đó, không có một nguồn tài liệu nào vừa đồng thời
(i) cập nhật; (ii) bao gồm đầy đủ các khía
cạnh của học máy hiện đại với đầy đủ chiều sâu kỹ thuật;
và (iii) xem kẽ các giải trình mà người ta mong đợi từ một cuốn
sách giáo trình với mã có thể thực thi,
điều thường được tìm thấy trong các bài hướng dẫn thực hành.
Chúng tôi tìm thấy một lượng
lớn các đoạn mã ví dụ về việc sử dụng một nền tảng học sâu (ví dụ làm thế nào
để thực hiện các phép toán cơ bản với ma trận trên TensorFlow)
hoặc để triển khai những kỹ thuật cụ thể (ví dụ các đoạn mã cho LeNet,
AlexNet, ResNet,...) dưới dạng một bài blog hoặc trên GitHub.
Tuy nhiên, những ví dụ này thường tập trung vào khía
cạnh *làm thế nào* để triển khai một hướng tiếp cận cho trước,
mà bỏ qua các thảo luận về việc *tại sao* một thuật toán được tạo như thế.
Trong khi các chủ đề lẻ tẻ đã được đề cập trong các bài blog, ví dụ trên
trang web [Distill](http://distill.pub) hoặc các blog cá nhân, họ chỉ đề cập
đến một vài chủ đề được chọn về học sâu, và thường thiếu mã nguồn đi kèm.
Một mặt khác, trong khi nhiều sách giáo trình đã ra đời,
đáng chú ý nhất là :cite:`Goodfellow.Bengio.Courville.2016`
(cuốn này cung cấp một bản khảo sát xuất sắc về các khái niệm phía sau học sâu),
những nguồn tài liệu này lại không đi kèm
với việc diễn giải dưới dạng mã nguồn để hiểu rõ hơn về các khái niệm.
Điều này khiến người đọc đôi khi mù tịt về cách thực thi chúng.
Bên cạnh đó, rất nhiều tài liệu lại được cung cấp dưới dạng các khoá học tốn phí.  

<!--
We set out to create a resource that could
(1) be freely available for everyone;
(2) offer sufficient technical depth to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(3) include runnable code, showing readers *how* to solve problems in practice;
(4) that allowed for rapid updates, both by us
and also by the community at large;
and (5) be complemented by a [forum](http://discuss.mxnet.io)
for interactive discussion of technical details and to answer questions.
-->

Chúng tôi đặt mục tiêu tạo ra một tài liệu mà có thể
(1) miễn phí cho mọi người;
(2) cung cấp chiều sâu kỹ thuật đầy đủ tạo điểm bắt đầu
cho con đường trở thành một nhà khoa học học máy ứng dụng;
(3) bao gồm mã thực thi được, trình bày cho
người đọc *làm thế nào* giải quyết các bài toán trên thực tế;
(4) tài liệu này có thể cập nhật một cách nhanh chóng, bằng cả chúng tôi và cộng động ở quy mô lớn;
và (5) được bổ sung bởi một [diễn đàn](http://discuss.mxnet.io) (và [diễn đàn tiếng Việt](https://forum.machinelearningcoban.com/c/d2l) của nhóm dịch)
cho những thảo luận nhanh chóng các chi tiết kỹ thuật và hỏi đáp.

<!--
These goals were often in conflict.
Equations, theorems, and citations are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the internet as a website.
At present there exist no tools and no workflow
perfectly suited to these demands, so we had to assemble our own.
We describe our approach in detail in :numref:`sec_how_to_contribute`.
We settled on Github to share the source and to allow for edits,
Jupyter notebooks for mixing code, equations and text,
Sphinx as a rendering engine to generate multiple outputs,
and Discourse for the forum.
While our system is not yet perfect,
these choices provide a good compromise among the competing concerns.
We believe that this might be the first book published
using such an integrated workflow.
-->

Những mục tiêu này từng có xung đột.
Các công thức, định lý, và các trích dẫn được quản lý tốt nhất trên LaTex.
Mã được giải thích tốt nhất bằng Python.
Và trang web phù hợp với HTML và JavaScript.
Hơn nữa, chúng tôi muốn nội dung vừa có thể truy cập được bằng
mã nguồn có thể thực thi, bằng một cuốn sách như một tập tin PDF tải về được,
và ở trên internet như một trang web.
Hiện tại không tồn tại công cụ
nào phù hợp một cách hoàn hảo cho những nhu cầu này,
bởi vậy chúng tôi phải tự tạo công cụ cho riêng mình.
Chúng tôi mô tả hướng tiếp cận một cách chi tiết trong
:numref:`chapter_contribute`. Chúng tôi tổ chức dự án trên
GitHub để chia sẻ mã nguồn và cho phép sửa đổi,
Jupyter notebook để kết hợp mã, các phương trình và nội dung chữ,
Sphinx như một bộ máy tạo nhiều tập tin đầu ra, và Discourse để tạo diễn đàn.
Trong khi hệ thống này còn chưa hoàn hảo, những sự lựa chọn này
cung cấp một giải pháp chấp nhận được trong số các giải pháp tương tự.
Chúng tôi tin rằng đây có thể là cuốn sách đầu tiên được xuất bản dưới
dạng kết hợp này.

<!-- =================== Kết thúc dịch Phần 2 ================================-->

<!-- =================== Bắt đầu dịch Phần 3 ================================-->

<!--
### Learning by Doing
-->

### *dịch tiêu đề phía trên*

<!--
Many textbooks teach a series of topics, each in exhaustive detail.
For example, Chris Bishop's excellent textbook :cite:`Bishop.2006`,
teaches each topic so thoroughly, that getting to the chapter
on linear regression requires a non-trivial amount of work.
While experts love this book precisely for its thoroughness,
for beginners, this property limits its usefulness as an introductory text.
-->

*dịch đoạn phía trên*

<!--
In this book, we will teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric probability distributions.
-->

*dịch đoạn phía trên*

<!--
Aside from a few preliminary notebooks that provide a crash course
in the basic mathematical background,
each subsequent chapter introduces both a reasonable number of new concepts
and provides single self-contained working examples---using real datasets.
This presents an organizational challenge.
Some models might logically be grouped together in a single notebook.
And some ideas might be best taught by executing several models in succession.
On the other hand, there is a big advantage to adhering
to a policy of *1 working example, 1 notebook*:
This makes it as easy as possible for you to
start your own research projects by leveraging our code.
Just copy a notebook and start modifying it.
-->

*dịch đoạn phía trên*

<!--
We will interleave the runnable code with background material as needed.
In general, we will often err on the side of making tools
available before explaining them fully (and we will follow up by
explaining the background later).
For instance, we might use *stochastic gradient descent*
before fully explaining why it is useful or why it works.
This helps to give practitioners the necessary
ammunition to solve problems quickly,
at the expense of requiring the reader
to trust us with some curatorial decisions.
-->

*dịch đoạn phía trên*

<!--
Throughout, we will be working with the MXNet library,
which has the rare property of being flexible enough for research
while being fast enough for production.
This book will teach deep learning concepts from scratch.
Sometimes, we want to delve into fine details about the models
that would typically be hidden from the user
by Gluon's advanced abstractions.
This comes up especially in the basic tutorials,
where we want you to understand everything
that happens in a given layer or optimizer.
In these cases, we will often present two versions of the example:
one where we implement everything from scratch,
relying only on the NumPy interface and automatic differentiation,
and another, more practical example,
where we write succinct code using Gluon.
Once we have taught you how some component works,
we can just use the Gluon version in subsequent tutorials.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 3 ================================-->

<!-- =================== Bắt đầu dịch Phần 4 ================================-->

<!--
### Content and Structure
-->

### *dịch tiêu đề phía trên*

<!--
The book can be roughly divided into three parts,
which are presented by different colors in :numref:`fig_book_org`:
-->

*dịch đoạn phía trên*

<!--
![Book structure](../img/book-org.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/book-org.svg)
:label:`fig_book_org`


<!--
* The first part covers basics and preliminaries.
:numref:`chap_introduction` offers an introduction to deep learning.
Then, in :numref:`chap_preliminaries`,
we quickly bring you up to speed on the prerequisites required
for hands-on deep learning, such as how to store and manipulate data,
and how to apply various numerical operations based on basic concepts
from linear algebra, calculus, and probability.
:numref:`chap_linear` and :numref:`chap_perceptrons`
cover the most basic concepts and techniques of deep learning,
such as linear regression, multilayer perceptrons and regularization.
-->

*dịch đoạn phía trên*

<!--
* The next five chapters focus on modern deep learning techniques.
:numref:`chap_computation` describes the various key components of deep
learning calculations and lays the groundwork
for us to subsequently implement more complex models.
Next, in :numref:`chap_cnn` and :numref:`chap_modern_cnn`,
we introduce convolutional neural networks (CNNs), powerful tools
that form the backbone of most modern computer vision systems.
Subsequently, in :numref:`chap_rnn` and :numref:`chap_modern_rnn`, we introduce
recurrent neural networks (RNNs), models that exploit
temporal or sequential structure in data, and are commonly used
for natural language processing and time series prediction.
In :numref:`chap_attention`, we introduce a new class of models
that employ a technique called attention mechanisms
and they have recently begun to displace RNNs in natural language processing.
These sections will get you up to speed on the basic tools
behind most modern applications of deep learning.
-->

*dịch đoạn phía trên*

<!--
* Part three discusses scalability, efficiency, and applications.
First, in :numref:`chap_optimization`,
we discuss several common optimization algorithms
used to train deep learning models.
The next chapter, :numref:`chap_performance` examines several key factors
that influence the computational performance of your deep learning code.
In :numref:`chap_cv` and :numref:`chap_nlp`, we illustrate
major applications of deep learning in computer vision
and natural language processing, respectively.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 4 ================================-->

<!-- =================== Bắt đầu dịch Phần 5 ================================-->

<!--
### Code
-->

### *dịch tiêu đề phía trên*
:label:`sec_code`

<!--
Most sections of this book feature executable code because of our belief
in the importance of an interactive learning experience in deep learning.
At present, certain intuitions can only be developed through trial and error,
tweaking the code in small ways and observing the results.
Ideally, an elegant mathematical theory might tell us
precisely how to tweak our code to achieve a desired result.
Unfortunately, at present, such elegant theories elude us.
Despite our best attempts, formal explanations for various techniques
are still lacking, both because the mathematics to characterize these models
can be so difficult and also because serious inquiry on these topics
has only just recently kicked into high gear.
We are hopeful that as the theory of deep learning progresses,
future editions of this book will be able to provide insights
in places the present edition cannot.
-->

*dịch đoạn phía trên*

<!--
Most of the code in this book is based on Apache MXNet.
MXNet is an open-source framework for deep learning
and the preferred choice of AWS (Amazon Web Services),
as well as many colleges and companies.
All of the code in this book has passed tests under the newest MXNet version.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of MXNet.
However, we plan to keep the online version remain up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.
-->

*dịch đoạn phía trên*

<!--
At times, to avoid unnecessary repetition, we encapsulate
the frequently-imported and referred-to functions, classes, etc.
in this book in the `d2l` package.
For any block block such as a function, a class, or multiple imports
to be saved in the package, we will mark it with
`# Saved in the d2l package for later use`.
The `d2l` package is light-weight and only requires
the following packages and modules as dependencies:
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
# Saved in the d2l package for later use
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
import os
import pandas as pd
import random
import re
import sys
import tarfile
import time
import zipfile
```

<!--
We offer a detailed overview of these functions and classes in :numref:`sec_d2l`.
-->

*dịch đoạn phía trên*

<!-- =================== Kết thúc dịch Phần 5 ================================-->

<!-- =================== Bắt đầu dịch Phần 6 ================================-->

<!--
### Target Audience
-->

### *dịch tiêu đề phía trên*

<!--
This book is for students (undergraduate or graduate),
engineers, and researchers, who seek a solid grasp
of the practical techniques of deep learning.
Because we explain every concept from scratch,
no previous background in deep learning or machine learning is required.
Fully explaining the methods of deep learning
requires some mathematics and programming,
but we will only assume that you come in with some basics,
including (the very basics of) linear algebra, calculus, probability,
and Python programming.
Moreover, in the Appendix, we provide a refresher
on most of the mathematics covered in this book.
Most of the time, we will prioritize intuition and ideas
over mathematical rigor.
There are many terrific books which can lead the interested reader further.
For instance, Linear Analysis by Bela Bollobas :cite:`Bollobas.1999`
covers linear algebra and functional analysis in great depth.
All of Statistics :cite:`Wasserman.2013` is a terrific guide to statistics.
And if you have not used Python before,
you may want to peruse this [Python tutorial](http://learnpython.org/).
-->

*dịch đoạn phía trên*


<!--
### Forum
-->

### *dịch tiêu đề phía trên*

<!--
Associated with this book, we have launched a discussion forum,
located at [discuss.mxnet.io](https://discuss.mxnet.io/).
When you have questions on any section of the book,
you can find the associated discussion page by scanning the QR code
at the end of the section to participate in its discussions.
The authors of this book and broader MXNet developer community
frequently participate in forum discussions.
-->

*dịch đoạn phía trên*


<!--
## Acknowledgments
-->

## *dịch tiêu đề phía trên*

<!--
We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
Specifically, we thank every contributor of this English draft
for making it better for everyone.
Their GitHub IDs or names are (in no particular order):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller, NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki, topecongiro, tpdi, vermicelli, Vishaal Kapoor, vishwesh5, YaYaB, Yuhong Chen, Evgeniy Smirnov, lgov, Simon Corston-Oliver, IgorDzreyev, Ha Nguyen, pmuens, alukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta, uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, prasanth5reddy, brianhendee, mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya, Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy, lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner, Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong, Steve Sedlmeyer, ruslo, Rafael Schlatter, liusy182, Giannis Pappas, ruslo, ati-ozgur, qbaza, dchoi77, Adam Gerson. Notably, Brent Werness (Amazon) and Rachel Hu (Amazon) co-authored the *Mathematics for Deep Learning* chapter in the Appendix with us and are the major contributors to that chapter.
-->

*dịch đoạn phía trên*

<!--
We thank Amazon Web Services, especially Swami Sivasubramanian,
Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened.
-->

*dịch đoạn phía trên*


<!--
## Summary
-->

## *dịch tiêu đề phía trên*

<!--
* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, including computer vision, natural language processing, automatic speech recognition.
* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, the algorithms for fitting your models to data, and the engineering techniques to implement it all.
* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
* To answer questions related to this book, visit our forum at https://discuss.mxnet.io/.
* Apache MXNet is a powerful library for coding up deep learning models and running them in parallel across GPU cores.
* Gluon is a high level library that makes it easy to code up deep learning models using Apache MXNet.
* Conda is a Python package manager that ensures that all software dependencies are met.
* All notebooks are available for download on GitHub and the conda configurations needed to run this book's code are expressed in the `environment.yml` file.
* If you plan to run this code on GPUs, do not forget to install the necessary drivers and update your configuration.
-->

*dịch đoạn phía trên*


<!--
## Exercises
-->

## *dịch tiêu đề phía trên*

<!--
1. Register an account on the discussion forum of this book [discuss.mxnet.io](https://discuss.mxnet.io/).
1. Install Python on your computer.
1. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book and find answers to your questions by engaging the authors and broader community.
1. Create an account on the forum and introduce yourself.
-->

*dịch đoạn phía trên*


<!--
## [Discussions](https://discuss.mxnet.io/t/2311)
-->

## *dịch tiêu đề phía trên*

<!--
![](../img/qr_preface.svg)
-->

![*dịch chú thích ảnh phía trên*](../img/qr_preface.svg)

<!-- =================== Kết thúc dịch Phần 6 ================================-->

### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên.

Lưu ý:
* Mỗi tên chỉ xuất hiện một lần: Nếu bạn đã dịch hoặc review phần 1 của trang này
thì không cần điền vào các phần sau nữa.
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

<!-- Phần 1 -->
* Vũ Hữu Tiệp

<!-- Phần 2 -->
*

<!-- Phần 3 -->
*

<!-- Phần 4 -->
*

<!-- Phần 5 -->
*

<!-- Phần 6 -->
*

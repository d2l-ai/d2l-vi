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

Chỉ một vài năm trước, không có nhiều nhà khoa học học sâu (*deep learning*) 
phát triển các sản phẩm và dịch vụ thông minh tại các công ty lớn cũng như các công ty khởi nghiệp.
Khi người trẻ nhất trong nhóm tác giả chúng tôi tiến vào lĩnh vực này, 
học máy (*machine learning*) còn chưa xuất hiện thường xuyên trên truyền thông.
Cha mẹ chúng tôi còn không có ý niệm gì về học máy chứ 
chưa nói đến việc hiểu tại sao chúng tôi theo đuổi lĩnh vực này thay vì y khoa hay luật khoa.
Học máy từng là một lĩnh vực nghiên cứu tiên phong với chỉ một số lượng nhỏ các ứng dụng thực tế.
Những ứng dụng như nhận dạng giọng nói (*speech recognition*) 
hay thị giác máy tính (*computer vision*), đòi hỏi quá nhiều kiến thức chuyên biệt 
khiến chúng thường được phân thành các lĩnh vực hoàn toàn riêng mà trong đó học máy chỉ là một thành phần nhỏ.
Các mạng nơ-ron (*neural network*), tiền đề của các mô hình học sâu 
mà chúng ta tập trung vào trong cuốn sách này, đã từng bị xem là các công cụ lỗi thời.


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

Chỉ trong khoảng năm năm gần đây, học sâu đã mang đến nhiều bất ngờ trên quy mô toàn cầu và 
dẫn đường cho những tiến triển nhanh chóng trong nhiều lĩnh vực khác nhau như thị giác máy tính, 
xử lý ngôn ngữ tự nhiên (*natural language processing*), nhận dạng giọng nói tự động (*automatic speech recognition*), 
học tăng cường (*reinforcement learning*), và mô hình hóa thống kê (*statistical modeling*).
Với những tiến bộ này, chúng ta bây giờ có thể xây dựng xe tự hành với mức độ tự động ngày càng cao 
(nhưng chưa nhiều tới mức như vài công ty đang tuyên bố), xây dựng các hệ thống giúp trả lời thư tự động 
khi con người ngập trong núi email, hay lập trình phần mềm chơi cờ vây có thể thắng cả nhà vô địch thế giới, 
một kỳ tích từng được xem là không thể đạt được trong nhiều thập kỷ tới.
Những công cụ này đã và đang gây ảnh hưởng rộng rãi tới các ngành công nghiệp và đời sống xã hội, 
thay đổi cách tạo ra các bộ phim, cách chẩn đoán bệnh và đóng một vài trò ngày càng tăng 
trong các ngành khoa học cơ bản -- từ vật lý thiên văn tới sinh học.


<!--
## About This Book
-->

## Về cuốn sách này

<!--
This book represents our attempt to make deep learning approachable,
teaching you the *concepts*, the *context*, and the *code*.
-->

Cuốn sách này được viết với mong muốn làm cho học sâu dễ tiếp cận hơn. 
Nó sẽ dạy bạn từ *khái niệm*, *bối cảnh*, cho tới cách *lập trình*.


<!--
### One Medium Combining Code, Math, and HTML
-->

### Một phương tiện Truyền tải kết hợp Mã nguồn, Toán, và HTML


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

Để một công nghệ điện toán đạt được tầm ảnh hưởng sâu rộng, nó phải dễ hiểu, 
có tài liệu đầy đủ, và được hỗ trợ bởi nhưng công cụ cấp tiến được "bảo trì" thường xuyên.
Các ý tưởng chính cần được chắt lọc rõ ràng, tối thiểu thời gian chuẩn bị cần thiết 
cho người mới bắt đầu để họ có thể trang bị các kiến thức đương thời.
Các thư viện cấp tiến nên tự động hóa các tác vụ đơn giản, 
và các đoạn mã nguồn được lấy làm ví dụ cần phải đơn giản với những người mới bắt đầu 
sao cho họ có thể dễ dàng chỉnh sửa, áp dụng, và mở rộng những ứng dụng thông thường thành các ứng dụng họ cần.
Lấy ứng dụng các trang web động làm ví dụ.
Mặc dù các công ty công nghệ lớn như Amazon phát triển thành công các ứng dụng web định hướng 
bởi cơ sở dữ liệu từ những năm 1990, tiềm năng của công nghệ này để hỗ trợ các doanh nghiệp sáng tạo 
chỉ được nhân rộng lên ở một tầm cao mới từ khoảng mười năm nay, 
nhờ vào sự phát triển của các nền tảng mạnh mẽ và với tài liệu đầy đủ.


<!--
Testing the potential of deep learning presents unique challenges
because any single application brings together various disciplines.
Applying deep learning requires simultaneously understanding
(i) the motivations for casting a problem in a particular way;
(ii) the mathematics of a given modeling approach;
(iii) the optimization algorithms for fitting the models to data;
and (iv) the engineering required to train models efficiently,
navigating the pitfalls of numerical computing
and getting the most out of available hardware.
Teaching both the critical thinking skills required to formulate problems,
the mathematics to solve them, and the software tools to implement those
solutions all in one place presents formidable challenges.
Our goal in this book is to present a unified resource
to bring would-be practitioners up to speed.
-->

Kiểm định tiềm năng của học sâu có những thách thức riêng biệt 
vì bất kỳ ứng dụng riêng lẻ nào cũng bao gồm nhiều lĩnh vực khác nhau.
Ứng dụng học sâu đòi hỏi những hiểu biết đồng thời về 
(i) động lực để mô hình hóa một bài toán theo một hướng cụ thể; 
(ii) kiến thức toán học của một phương pháp mô hình hóa; 
(iii) những thuật toán tối ưu để khớp mô hình với dữ liệu; 
và (iv) phần kỹ thuật yêu cầu để huấn luyện mô hình một cách hiệu quả, 
xử lý những khó khăn trong tính toán và tận dụng thật tốt phần cứng hiện có.
Việc đào tạo kỹ năng suy nghĩ thấu đáo cần thiết để định hình bài toán, 
cung cấp kiến thức toán để giải chúng, và hướng dẫn cách dùng các công cụ phần mềm 
để triển khai những giải pháp đó, tất cả trong một nơi, hàm chứa nhiều thách thức lớn.
Mục tiêu của chúng tôi trong cuốn sách này là trình bày 
một nguồn tài liệu tổng hợp giúp những học viên nhanh chóng bắt kịp.


<!--
At the time we started this book project,
there were no resources that simultaneously
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

Vào thời điểm chúng tôi bắt đầu dự án sách này,
không có tài nguyên nào đồng thời (i) cập nhật; 
(ii) bao gồm đầy đủ các khía cạnh của học máy hiện đại với đầy đủ chiều sâu kỹ thuật; 
và (iii) xem kẽ các giải trình mà người ta mong đợi từ một cuốn sách giáo trình với mã nguồn có thể thực thi, 
điều thường được tìm thấy trong các bài hướng dẫn thực hành.
Chúng tôi tìm thấy một lượng lớn các đoạn mã ví dụ về việc sử dụng một nền tảng học sâu 
(ví dụ làm thế nào để thực hiện các phép toán cơ bản với ma trận trên TensorFlow) 
hoặc để triển khai những kỹ thuật cụ thể (ví dụ các đoạn mã cho LeNet, AlexNet, ResNet,...) 
trong các bài blog hoặc là trên GitHub.
Tuy nhiên, những ví dụ này thường tập trung vào khía cạnh *làm thế nào* để triển khai 
một hướng tiếp cận cho trước, mà bỏ qua việc thảo luận *tại sao* một thuật toán được tạo như thế.
Nhiều chủ đề đã được đề cập đến trong các bài blog, 
ví dụ như trang [Distill](http://distill.pub) hoặc các trang cá nhân, 
chúng thường chỉ đề cập đến một vài chủ đề được chọn về học sâu và thường thiếu mã nguồn đi kèm.
Một mặt khác, trong khi nhiều sách giáo trình đã ra đời, 
đáng chú ý nhất là :cite:`Goodfellow.Bengio.Courville.2016` 
(cuốn này cung cấp một bản khảo sát xuất sắc về các khái niệm phía sau học sâu), 
những nguồn tài liệu này lại không đi kèm với việc diễn giải dưới dạng mã nguồn để làm rõ hơn các khái niệm.
Điều này khiến người đọc đôi khi mơ hồ về cách thực thi chúng.
Bên cạnh đó, rất nhiều tài liệu lại được cung cấp dưới dạng các khóa học có phí.

<!--
We set out to create a resource that could
(i) be freely available for everyone;
(ii) offer sufficient technical depth to provide a starting point on the path
to actually becoming an applied machine learning scientist;
(iii) include runnable code, showing readers *how* to solve problems in practice;
(iv) allow for rapid updates, both by us
and also by the community at large;
and (v) be complemented by a [forum](http://discuss.d2l.ai)
for interactive discussion of technical details and to answer questions.
-->

Chúng tôi đặt mục tiêu tạo ra một tài liệu mà có thể (i) miễn phí cho mọi người;
(ii) cung cấp chiều sâu kỹ thuật đầy đủ, là điểm khởi đầu trên con đường trở thành một nhà khoa học học máy ứng dụng;
(iii) bao gồm mã nguồn thực thi được, trình bày cho người đọc *làm thế nào* giải quyết các bài toán trên thực tế;
(iv) cho phép cập nhật một cách nhanh chóng bởi các tác giả cũng như cộng động ở quy mô lớn;
và (v) được bổ sung bởi một [diễn đàn](http://discuss.d2l.ai) 
(và [diễn đàn tiếng Việt](https://forum.machinelearningcoban.com/c/d2l) của nhóm dịch thuật)
để nhanh chóng thảo luận và hỏi đáp về các chi tiết kỹ thuật.

<!--
These goals were often in conflict.
Equations, theorems, and citations are best managed and laid out in LaTeX.
Code is best described in Python.
And webpages are native in HTML and JavaScript.
Furthermore, we want the content to be
accessible both as executable code, as a physical book,
as a downloadable PDF, and on the Internet as a website.
At present there exist no tools and no workflow
perfectly suited to these demands, so we had to assemble our own.
We describe our approach in detail in :numref:`sec_how_to_contribute`.
We settled on GitHub to share the source and to allow for edits,
Jupyter notebooks for mixing code, equations and text,
Sphinx as a rendering engine to generate multiple outputs,
and Discourse for the forum.
While our system is not yet perfect,
these choices provide a good compromise among the competing concerns.
We believe that this might be the first book published
using such an integrated workflow.
-->

Các mục tiêu này thường không tương thích với nhau.
Các công thức, định lý, và các trích dẫn được quản lý tốt nhất trên LaTex.
Mã được giải thích tốt nhất bằng Python.
Và trang web phù hợp với HTML và JavaScript.
Hơn nữa, chúng tôi muốn nội dung của nó vừa có thể được truy cập dưới dạng mã nguồn có thể thực thi, 
vừa có thể tải về như một cuốn sách dưới định dạng PDF, và lại ở trên internet như một trang web.
Hiện tại không có một công cụ nào là hoàn hảo cho những nhu cầu này, 
bởi vậy chúng tôi phải tự tạo công cụ cho riêng mình.
Chúng tôi mô tả hướng tiếp cận một cách chi tiết trong :numref:`chapter_contribute`. 
Chúng tôi tổ chức dự án trên GitHub để chia sẻ mã nguồn và cho phép sửa đổi, 
Jupyter notebook để kết hợp đoạn mã, phương trình toán và nội dung chữ, 
sử dụng Sphinx như một bộ máy tạo nhiều tập tin đầu ra, và Discourse để tạo diễn đàn.
Trong khi hệ thống này còn chưa hoàn hảo, 
những lựa chọn này cung cấp một giải pháp chấp nhận được trong số các giải pháp tương tự.
Chúng tôi tin rằng đây có thể là cuốn sách đầu tiên được xuất bản dưới dạng kết hợp này.


<!--
### Learning by Doing
-->

### Học thông qua Thực hành


<!--
Many textbooks teach a series of topics, each in exhaustive detail.
For example, Chris Bishop's excellent textbook :cite:`Bishop.2006`,
teaches each topic so thoroughly, that getting to the chapter
on linear regression requires a non-trivial amount of work.
While experts love this book precisely for its thoroughness,
for beginners, this property limits its usefulness as an introductory text.
-->

Có nhiều cuốn sách dạy rất chi tiết về một chuỗi các chủ đề khác nhau.
Ví dụ như trong cuốn sách tuyệt vời :cite:`Bishop.2006` này của Bishop, 
mỗi chủ đề được dạy rất kỹ lưỡng đến nỗi để đến được chương hồi quy tuyến tính 
cũng đòi hỏi phải bỏ ra không ít công sức.
Các chuyên gia yêu thích quyển sách này chính vì sự kỹ lưỡng mà nó mang lại, 
nhưng với những người mới bắt đầu thì 
đây là điểm hạn chế việc sử dụng cuốn sách này như một tài liệu nhập môn.

<!--
In this book, we will teach most concepts *just in time*.
In other words, you will learn concepts at the very moment
that they are needed to accomplish some practical end.
While we take some time at the outset to teach
fundamental preliminaries, like linear algebra and probability,
we want you to taste the satisfaction of training your first model
before worrying about more esoteric probability distributions.
-->

Trong cuốn sách này, chúng tôi sẽ dạy hầu hết các khái niệm *ở mức vừa đủ*.
Hay nói cách khác, bạn sẽ chỉ học và hiểu các khái niệm cần thiết đủ để bạn hoàn tất phần thực hành.
Trong khi chúng tôi sẽ dành một chút thời gian để dạy kiến thức căn bản sơ bộ 
như đại số tuyến tính và xác suất, chúng tôi muốn các bạn được tận hưởng cảm giác mãn nguyện 
của việc huấn luyện được mô hình đầu tiên trước khi bận tâm tới các lý thuyết phân phối xác suất.

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

Bên cạnh một vài notebook cơ bản cung cấp một khóa học cấp tốc về nền tảng toán học, 
mỗi chương tiếp theo sẽ giới thiệu một lượng hợp lý các khái niệm mới 
và đồng thời cung cấp các ví dụ đơn hoàn chỉnh---sử dụng các tập dữ liệu thực tế.
Và đây là cả thách thức về cách tổ chức nội dung.
Một vài mô hình có thể được nhóm lại một cách có logic trong một notebook riêng lẻ.
Và một vài ý tưởng có thể được dạy tốt nhất bằng cách thực thi một số mô hình kế tiếp nhau.
Mặt khác, có một lợi thế lớn về việc tuân thủ theo chính sách *mỗi notebook là một ví dụ hoàn chỉnh*:
Điều này giúp bạn bắt đầu các dự án nghiên cứu của mình 
một cách dễ dàng nhất có thể bằng cách tận dụng mã nguồn của chúng tôi.
Bạn chỉ cần sao chép một notebook và bắt đầu sửa đổi ở trên đó.

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

Chúng tôi sẽ xen kẽ mã nguồn có thể thực thi với kiến thức nền tảng khi cần thiết.
Thông thường, chúng tôi sẽ tập trung vào việc tạo ra những công cụ 
trước khi giải thích chúng đầy đủ (và chúng tôi sẽ theo sát bằng cách giải thích phần kiến thức nền tảng sau). 
Ví dụ, chúng tôi có thể sử dụng *hạ gradient ngẫu nhiên* trước khi giải thích 
đầy đủ tại sao nó lại hữu ích hoặc tại sao nó lại hoạt động.
Điều này giúp cung cấp cho người thực hành những phương tiện cần thiết để giải quyết vấn đề 
nhanh chóng và đòi hỏi người đọc phải tin tưởng vào một số quyết định triển khai của chúng tôi.


<!--
This book will teach deep learning concepts from scratch.
Sometimes, we want to delve into fine details about the models
that would typically be hidden from the user
by deep learning frameworks' advanced abstractions.
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

Cuốn sách này sẽ dạy về khái niệm học sâu từ đầu.
Thỉnh thoảng, chúng tôi sẽ muốn đào sâu hơn vào những chi tiết về mô hình 
mà thông thường sẽ được che giấu khỏi người dùng bởi những lớp trừu tượng bậc cao của framework học sâu.
Điều này đặc biệt hay xuất hiện trong các hướng dẫn cơ bản, 
nơi chúng tôi muốn bạn hiểu về tất cả mọi thứ đang diễn ra trong một tầng hoặc bộ tối ưu nào đó.
Trong những trường hợp này, chúng tôi sẽ thường trình bày hai phiên bản 
của một ví dụ: một phiên bản trong đó chúng tôi hiện thực mọi thứ từ đầu, 
chỉ dựa vào giao diện Numpy và việc tính đạo hàm tự động; 
và một phiên bản khác thực tế hơn, khi chúng tôi viết mã ngắn gọn sử dụng Gluon.
Một khi chúng tôi đã dạy bạn cách một số thành phần hoạt động cụ thể như thế nào, 
chúng tôi có thể chỉ sử dụng phiên bản Gluon trong những hướng dẫn tiếp theo.


<!--
### Content and Structure
-->

### Nội dung và Bố cục

<!--
The book can be roughly divided into three parts,
which are presented by different colors in :numref:`fig_book_org`:
-->

Cuốn sách này có thể được chia thành ba phần, 
với các phần được thể hiện bởi những màu khác nhau trong :numref:`fig_book_org`:


<!--
![Book structure](../img/book-org.svg)
-->

![Bố cục của cuốn sách.](../img/book-org.svg)
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

* Phần đầu cuốn sách trình bày các kiến thức cơ bản và những việc cần chuẩn bị sơ bộ. 
:numref:`chap_introduction` giới thiệu về học sâu.
Sau đó, qua :numref:`chap_preliminaries`, chúng tôi nhanh chóng trang bị cho bạn những 
kiến thức nền cần thiết để thực hành học sâu như cách lưu trữ, 
thao tác dữ liệu và cách áp dụng những phép tính dựa trên 
những khái niệm cơ bản trong đại số tuyến tính, giải tích và xác suất.
:numref:`chap_linear` và :numref:`chap_perceptrons` giới thiệu những khái niệm 
và kỹ thuật cơ bản của học sâu, ví dụ như hồi quy tuyến tính, mạng perceptron đa lớp và điều chuẩn.


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

* Năm chương tiếp theo tập trung vào những kỹ thuật học sâu hiện đại.
:numref:`chap_computation` miêu tả những thành phần thiết yếu của các phép tính trong học sâu 
và tạo nền tảng để chúng tôi triển khai những mô hình phức tạp hơn.
Sau đó, chúng tôi sẽ giới thiệu mạng nơ-ron tích chập (Convolutional Neural Networks – CNN), 
một công cụ mạnh mẽ đang là nền tảng của hầu hết các hệ thống thị giác máy tính hiện đại.
Tiếp đến, trong :numref:`chap_rnn` và :numref:`chap_modern_rnn`, 
chúng tôi giới thiệu mạng nơ-ron hồi tiếp (Recurrent Neural Networks – RNN), 
một loại mô hình khai thác cấu trúc tạm thời hoặc tuần tự trong dữ liệu 
và thường được sử dụng để xử lý ngôn ngữ tự nhiên và dự đoán chuỗi thời gian.
Trong :numref:`chap_attention`, chúng tôi giới thiệu một lớp mô hình mới sử dụng kỹ thuật cơ chế chú ý (Attention Mechanisms), 
một kỹ thuật gần đây đã thay thế RNN trong xử lý ngôn ngữ tự nhiên.
Những phần này sẽ giúp bạn nhanh chóng nắm được những công cụ cơ bản đứng sau hầu hết các ứng dụng hiện đại của học sâu.


<!--
* Part three discusses scalability, efficiency, and applications.
First, in :numref:`chap_optimization`,
we discuss several common optimization algorithms
used to train deep learning models.
The next chapter, :numref:`chap_performance` examines several key factors
that influence the computational performance of your deep learning code.
In :numref:`chap_cv`,
we illustrate major applications of deep learning in computer vision.
In :numref:`chap_nlp_pretrain` and :numref:`chap_nlp_app`,
we show how to pretrain language representation models and apply
them to natural language processing tasks.
-->

* Phần ba thảo luận quy mô mở rộng, hiệu quả và ứng dụng.
Đầu tiên, trong :numref:`chap_optimization`, chúng tôi bàn luận một số thuật toán tối ưu phổ biến được sử dụng để huấn luyện các mô hình học sâu.
Chương tiếp theo, :numref:`chap_performance` khảo sát những yếu tố chính ảnh hưởng đến chất lượng tính toán của mã nguồn học sâu.
Trong :numref:`chap_cv`, chúng tôi minh họa các ứng dụng chính của học sâu trong thị giác máy tính.
Trong :numref:`chap_nlp_pretrain` và :numref:`chap_nlp_app`, 
chúng tôi chỉ ra cách biểu diễn trước các mô hình tiền huấn luyện ngôn ngữ và ứng dụng chúng cho các tác vụ xử lý ngôn ngữ tự nhiên.


<!--
### Code
-->

### Mã nguồn
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

Hầu hết các phần của cuốn sách đều bao gồm mã nguồn thực thi được, 
bởi chúng tôi tin rằng trải nghiệm học thông qua tương tác đóng một vai trò quan trọng trong học sâu.
Hiện tại, một số kinh nghiệm nhất định chỉ có thể được hình thành thông qua phương pháp thử và sai, 
thay đổi mã nguồn từng chút một và quan sát kết quả.
Lý tưởng nhất là sử dụng một lý thuyết toán học khác biệt nào đó có thể cho chúng ta biết 
chính xác cách thay đổi mã nguồn để đạt được kết quả mong muốn.
Thật đáng tiếc là hiện tại những lý thuyết khác biệt đó vẫn chưa được khám phá ra.
Mặc dù chúng tôi đã cố gắng hết sức, vẫn chưa có cách giải thích trọn vẹn nào cho nhiều vấn đề kỹ thuật, 
bởi vì phần toán học để mô tả những mô hình đó có thể là rất khó và công cuộc tìm hiểu 
về những chủ đề này mới chỉ tăng cao trong thời gian gần đây.
Chúng tôi hy vọng rằng khi mà những lý thuyết về học sâu phát triển, 
những phiên bản tiếp theo của cuốn sách sẽ có thể cung cấp 
những cái nhìn sâu sắc hơn mà phiên bản hiện tại chưa làm được.


<!--
At times, to avoid unnecessary repetition, we encapsulate
the frequently-imported and referred-to functions, classes, etc.
in this book in the `d2l` package.
For any block such as a function, a class, or multiple imports
to be saved in the package, we will mark it with `#@save`.
We offer a detailed overview of these functions and classes in :numref:`sec_d2l`.
The `d2l` package is light-weight and only requires
the following packages and modules as dependencies:
-->

Để tránh việc lặp lại không cần thiết, chúng tôi đóng gói những hàm, lớp,... 
mà thường xuyên được chèn vào và dùng để tham khảo đến trong cuốn sách này trong gói thư viện `d2l`.
Đối với bất kỳ khối mã nguồn nào như là một hàm, một lớp, 
hoặc các khai báo thư viện cần được đóng gói, chúng tôi sẽ đánh dấu bằng dòng`#@save`.
Chúng tôi cung cấp một góc nhìn tổng quan chi tiết về các hàm và lớp này trong :numref:`sec_d2l`.
Thư viện `d2l` khá nhẹ và chỉ phụ thuộc vào những gói thư viện và mô-đun sau:


```{.python .input  n=1}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
import random
import re
import shutil
import sys
import tarfile
import time
import requests
import zipfile
import hashlib
d2l = sys.modules[__name__]
```


<!--
:begin_tab:`mxnet`

Most of the code in this book is based on Apache MXNet.
MXNet is an open-source framework for deep learning
and the preferred choice of AWS (Amazon Web Services),
as well as many colleges and companies.
All of the code in this book has passed tests under the newest MXNet version.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of MXNet.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from MXNet.
:end_tab:
-->

:begin_tab:`mxnet`

Hầu hết mã nguồn trong cuốn sách này được phát triển trên Apache MXNet.
MXNet là một framework mã nguồn mở dành cho học sâu và là lựa chọn ưu tiên của AWS (Amazon Web Services),
cũng như là của nhiều trường Đại học và Công ty.
Tất cả mã nguồn được cung cấp trong cuốn sách này đều đã vượt qua các bài kiểm tra theo phiên bản MXNet mới nhất.
Tuy nhiên, do sự phát triển nhanh chóng của học sâu, một số mã nguồn *ở bản in* có thể 
không hoạt động bình thường với các phiên bản MXNet trong tương lai.
Tuy nhiên, chúng tôi có kế hoạch để cập nhật phiên bản trực tuyến.
Trong trường hợp bạn gặp phải bất kỳ vấn đề nào như vậy,
vui lòng tham khảo :ref:`chap_installation`
để cập nhật mã nguồn và môi trường thời gian chạy của bạn.

Đây là cách chúng tôi nhập mô-đun từ MXNet.
:end_tab:


<!--
:begin_tab:`pytorch`

Most of the code in this book is based on PyTorch.
PyTorch is an open-source framework for deep learning, which is extremely
popular in the research community.
All of the code in this book has passed tests under the newest PyTorch.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of PyTorch.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from PyTorch.
:end_tab:
-->

:begin_tab:`pytorch`

Hầu hết mã nguồn trong cuốn sách này được phát triển trên PyTorch.
PyTorch là một framework mã nguồn mở dành cho học sâu rất phổ biến trong cộng đồng nghiên cứu.
Tất cả mã nguồn được cung cấp trong cuốn sách này đều đã vượt qua các bài kiểm tra theo phiên bản PyTorch mới nhất.
Tuy nhiên, do sự phát triển nhanh chóng của học sâu, một số mã nguồn *ở bản in* có thể 
không hoạt động bình thường với các phiên bản PyTorch trong tương lai.
Tuy nhiên, chúng tôi có kế hoạch để cập nhật phiên bản trực tuyến.
Trong trường hợp bạn gặp phải bất kỳ vấn đề nào như vậy,
vui lòng tham khảo :ref:`chap_installation`
để cập nhật mã nguồn và môi trường thời gian chạy của bạn.

Đây là cách chúng tôi nhập mô-đun từ PyTorch.
:end_tab:


<!--
:begin_tab:`tensorflow`

Most of the code in this book is based on TensorFlow.
TensorFlow is an open-source framework for deep learning, which is extremely
popular in both the research community and industrial.
All of the code in this book has passed tests under the newest TensorFlow.
However, due to the rapid development of deep learning, some code
*in the print edition* may not work properly in future versions of TensorFlow.
However, we plan to keep the online version up-to-date.
In case you encounter any such problems,
please consult :ref:`chap_installation`
to update your code and runtime environment.

Here is how we import modules from TensorFlow.
:end_tab:
-->

:begin_tab:`tensorflow`

Hầu hết mã nguồn trong cuốn sách này được phát triển trên TensorFlow.
TensorFlow là một framework mã nguồn mở dành cho học sâu rất phổ biến trong cả cộng đồng nghiên cứu và ứng dụng công nghiệp.
Tất cả mã nguồn được cung cấp trong cuốn sách này đều đã vượt qua các bài kiểm tra theo phiên bản TensorFlow mới nhất.
Tuy nhiên, do sự phát triển nhanh chóng của học sâu, một số mã nguồn *ở bản in* có thể 
không hoạt động bình thường với các phiên bản TensorFlow trong tương lai.
Tuy nhiên, chúng tôi có kế hoạch để cập nhật phiên bản trực tuyến.
Trong trường hợp bạn gặp phải bất kỳ vấn đề nào như vậy,
vui lòng tham khảo :ref:`chap_installation`
để cập nhật mã nguồn và môi trường thời gian chạy của bạn.

Đây là cách chúng tôi nhập mô-đun từm TensorFlow.
:end_tab:


```{.python .input  n=1}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input  n=1}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
```

```{.python .input  n=1}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```


<!--
### Target Audience
-->

### Đối tượng Độc giả

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

Cuốn sách này dành cho các bạn sinh viên (đại học hoặc sau đại học), các kỹ sư và các nhà nghiên cứu – 
những người tìm kiếm một nền tảng vững chắc về những kỹ thuật thực tế của học sâu.
Bởi vì chúng tôi giải thích mọi khái niệm từ đầu, bạn không bắt buộc phải có nền tảng về học sâu hay học máy.
Việc giải thích đầy đủ các phương pháp học sâu đòi hỏi một số kiến thức về toán học và lập trình, 
nhưng chúng tôi sẽ chỉ giả định rằng bạn nắm được một số kiến thức cơ bản về đại số tuyến tính, giải tích, xác suất, và lập trình Python.
Hơn nữa, trong phần Phụ lục, chúng tôi cung cấp thêm về hầu hết các phần toán được đề cập trong cuốn sách này.
Phần lớn thời gian, chúng tôi sẽ ưu tiên dùng cách giải thích trực quan và mô tả các ý tưởng hơn là giải thích chặt chẽ bằng toán.
Có rất nhiều cuốn sách tuyệt vời có thể thu hút bạn đọc quan tâm sâu hơn nữa.
Chẳng hạn, cuốn "Giải tích Tuyến tính" (Linear Analysis) của Bela Bollobas :cite:`Bollobas.1999` bao gồm cả đại số tuyến tính và giải tích hàm ở mức độ rất chi tiết.
Cuốn "Tất cả về Thống kê" (All of Statistics) :cite:`Wasserman.2013` là hướng dẫn tuyệt vời để học thống kê.
Và nếu bạn chưa sử dụng Python, bạn có thể muốn xem [hướng dẫn Python này](http://learnpython.org/).


<!--
### Forum
-->

### Diễn đàn

<!--
Associated with this book, we have launched a discussion forum,
located at [discuss.d2l.ai](https://discuss.d2l.ai/).
When you have questions on any section of the book,
you can find the associated discussion page link at the end of each chapter.
-->

Gắn liền với cuốn sách, chúng tôi đã tạo ra một diễn đàn trực tuyến tại [discuss.d2l.ai](https://discuss.d2l.ai/) 
(và tại [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)).
Khi có câu hỏi về bất kỳ phần nội dung nào của cuốn sách, bạn có thể tìm thấy trang thảo luận liên quan được đặt ở cuối mỗi phần nội dung.


<!--
## Acknowledgments
-->

## Lời cảm ơn

<!--
We are indebted to the hundreds of contributors for both
the English and the Chinese drafts.
They helped improve the content and offered valuable feedback.
Specifically, we thank every contributor of this English draft
for making it better for everyone.
Their GitHub IDs or names are (in no particular order):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, 315930399, tayfununal,
steinsag, charleybeller.
-->

Chúng tôi xin gửi lời cảm ơn chân thành tới hàng trăm người đã đóng góp cho cả hai bản thảo tiếng Anh và tiếng Trung.
Mọi người đã giúp cải thiện nội dung và đưa ra những phản hồi rất có giá trị.
Cụ thể, chúng tôi cảm ơn tất cả những người đóng góp cho dự thảo tiếng Anh này giúp nó tốt hơn cho tất cả mọi người.
Đây là tài khoản GitHub hoặc tên các bạn đóng góp (không theo trình tự cụ thể nào):
alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat,
cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu,
Rahul Agarwal, Mohamed Ali Jamaoui, Michael (Stu) Stewart, Mike Müller,
NRauschmayr, Prakhar Srivastav, sad-, sfermigier, Sheng Zha, sundeepteki,
topecongiro, tpdi, vermicelli, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen,
Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Ha Nguyen, pmuens,
Andrei Lukovenko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta,
uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee,
mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya,
Muhyun Kim, dennismalmgren, adursun, Anirudh Dagar, liqingnz, Pedro Larroy,
lgov, ati-ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner,
Maximilian Böther, Rakib Islam, Leonard Lausen, Abhinav Upadhyay, rongruosong,
Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas,
ati-ozgur, qbaza, dchoi77, Adam Gerson, Phuc Le, Mark Atwood, christabella, vn09,
Haibin Lin, jjangga0214, RichyChen, noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil,
Peter Kulits, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp,
Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto,
Katarzyna Biesialska, Gregory Bruss, Duy–Thanh Doan, paulaurel, graytowne, Duc Pham,
sl7423, Jaedong Hwang, Yida Wang, cys4, clhm, Jean Kaddour, austinmw, trebeljahr, tbaums,
Cuong V. Nguyen, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic,
the-great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, biagiom,
abhinavsp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang,
StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU,
Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder,
mathresearch, mzz2017, jroberayalas, iluu, ghejc, BSharmi, vkramdev, simonwardjones, LakshKD,
TalNeoran, djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, 315930399, tayfununal,
steinsag, charleybeller.

Với bản chuyển ngữ tiếng Việt, chúng tôi đã nhận được rất nhiều sự giúp đỡ, sự ủng hộ cùng với sự tư vấn từ Cộng đồng; 
bằng sự trân trọng sâu sắc, chúng tôi mong muốn gửi lời cảm ơn và tri ân đến những người đã đóng góp vào dự án này dù ít hay nhiều.
Chúng tôi xin gửi lời cảm ơn đến những người đóng góp (không theo trình tự cụ thể nào):
Đoàn Võ Duy Thanh, Vũ Hữu Tiệp, Lê Khắc Hồng Phúc, Trần Thị Hồng Hạnh, Phạm Hồng Vinh,
Nguyễn Cảnh Thướng, Nguyễn Lê Quang Nhật, Phạm Minh Đức, Nguyễn Văn Quang, Nguyễn Văn Cường,
Đỗ Trường Giang, Trần Yến Thy, Nguyễn Mai Hoàng Long, Lý Phi Long, Tạ H. Duy Nguyên, Ngô Thế Anh Khoa,
Mai Sơn Hải, Sẩm Thế Hải, Lê Đàm Hồng Lộc, Lương Kim Doanh, Vũ Đình Quyền, Phạm Chí Thành,
Hoàng Trọng Tuấn, Nguyễn Văn Tâm, Trần Kiến An, Trần Hoàng Quân, Nguyễn Minh Thư, Nguyễn Phan Hùng Thuận,
Bùi Nhật Quân, Lê Gia Thiên Bửu, Tạ Đức Huy, Lê Thành Vinh, Nguyễn Quang Hải, Minh Trí Nguyễn,
Nguyễn Trường Phát, Lâm Ngọc Tâm, Dương Nhật Tân, Nguyễn Duy Du, Đinh Minh Tân, Nguyễn Thanh Hòa, Võ Tấn Phát,
Lê Cao Thăng, Phạm Ngọc Bảo Anh, Bùi Chí Minh, Nguyễn Thành Hưng, Nguyễn Đình Nam, Đinh Đắc,
Đinh Phước Lộc, Hoang Van-Tien, Phạm Đăng Khoa, Trương Lộc Phát, Bùi Thị Cẩm Nhung, Nguyễn Thái Bình.

Những đóng góp cụ thể của Cộng đồng được chúng tôi liệt kê đầy đủ **[tại đây](https://github.com/mlbvn/d2l-vn/blob/master/ACKNOWLEDGEMENT.md)**.

<!--
We thank Amazon Web Services, especially Swami Sivasubramanian,
Raju Gulabani, Charlie Bell, and Andrew Jassy for their generous support in writing this book. Without the available time, resources, discussions with colleagues, and continuous encouragement this book would not have happened.
-->

Chúng tôi cũng gửi lời cảm ơn Amazon Web Services, đặc biệt là Swami Sivasubramanian, Raju Gulabani, Charlie Bell, 
và Andrew Jassy vì sự hỗ trợ hào phóng của họ trong việc viết cuốn sách này.
Nếu không có thời gian, tài nguyên, mọi sự thảo luận cùng các đồng nghiệp, 
cũng như những khuyến khích liên tục, sự xuất hiện của cuốn sách này sẽ không thể thành hiện thực.


## Tóm tắt

<!--
* Deep learning has revolutionized pattern recognition, introducing technology that now powers a wide range of  technologies, 
including computer vision, natural language processing, automatic speech recognition.
* To successfully apply deep learning, you must understand how to cast a problem, the mathematics of modeling, 
the algorithms for fitting your models to data, and the engineering techniques to implement it all.
* This book presents a comprehensive resource, including prose, figures, mathematics, and code, all in one place.
* To answer questions related to this book, visit our forum at https://discuss.d2l.ai/.
* All notebooks are available for download on GitHub.
-->

* Học sâu đã cách mạng hóa nhận dạng mẫu, đưa ra công nghệ cốt lõi hiện được sử dụng trong nhiều ứng dụng công nghệ, 
bao gồm thị giác máy, xử lý ngôn ngữ tự nhiên và nhận dạng giọng nói tự động.
* Để áp dụng thành công kỹ thuật học sâu, bạn phải hiểu được cách biến đổi bài toán, toán học của việc mô hình hóa, 
các thuật toán để khớp mô hình theo dữ liệu của bạn, và các kỹ thuật để thực hiện tất cả những điều này.
* Cuốn sách này là một nguồn tài liệu toàn diện, bao gồm các diễn giải, hình minh họa, công thức toán và mã nguồn, tất cả trong một.
* Để tìm câu trả lời cho các câu hỏi liên quan đến cuốn sách này, hãy truy cập diễn đàn của chúng tôi tại https://discuss.d2l.ai/.
(Diễn đàn của nhóm dịch tại https://forum.machinelearningcoban.com/c/d2l).
* Tất cả các notebook đều có thể tải xuống từ GitHub.


## Bài tập

<!--
1. Register an account on the discussion forum of this book [discuss.d2l.ai](https://discuss.d2l.ai/).
2. Install Python on your computer.
3. Follow the links at the bottom of the section to the forum, where you will be able to seek out help and discuss the book 
and find answers to your questions by engaging the authors and broader community.
-->

1. Đăng ký tài khoản diễn đàn của cuốn sách tại [discuss.d2l.ai](https://discuss.d2l.ai/) 
(và của nhóm Dịch thuật tại [https://forum.machinelearningcoban.com](https://forum.machinelearningcoban.com)).
2. Cài đặt Python trên máy tính.
3. Làm theo hướng dẫn ở các liên kết đến diễn đàn ở cuối phần này, ở các liên kết diễn đàn đó 
bạn sẽ có thể nhận được giúp đỡ và thảo luận về cuốn sách cũng như tìm ra câu trả lời cho câu hỏi của bạn bằng cách thu hút các tác giả và cộng đồng lớn hơn.


## Thảo luận
* Tiếng Anh: [MXNet](https://discuss.d2l.ai/t/18), [PyTorch](https://discuss.d2l.ai/t/20), [TensorFlow](https://discuss.d2l.ai/t/186)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l), [Github Discussions](https://github.com/mlbvn/d2l-vn/discussions)

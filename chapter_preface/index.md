# Lời nói đầu

Chỉ vài năm trước, không có quân đoàn của các nhà khoa học học sâu phát triển các sản phẩm và dịch vụ thông minh tại các công ty lớn và công ty khởi nghiệp. Khi chúng tôi bước vào lĩnh vực này, học máy đã không ra lệnh tiêu đề trên các tờ báo hàng ngày. Cha mẹ chúng ta không biết học máy là gì, chứ đừng nói đến lý do tại sao chúng ta có thể thích nó hơn một nghề nghiệp trong y học hoặc luật pháp. Học máy là một kỷ luật học thuật bầu trời xanh có ý nghĩa công nghiệp bị giới hạn trong một tập hợp hẹp các ứng dụng trong thế giới thực, bao gồm nhận dạng giọng nói và tầm nhìn máy tính. Hơn nữa, nhiều ứng dụng trong số này đòi hỏi rất nhiều kiến thức về miền đến mức chúng thường được coi là các lĩnh vực hoàn toàn riêng biệt mà machine learning là một thành phần nhỏ. Vào thời điểm đó, mạng thần kinh - những người tiền nhiệm của các phương pháp học sâu mà chúng ta tập trung vào trong cuốn sách này - thường được coi là lỗi thời. 

Chỉ trong năm năm qua, deep learning đã gây bất ngờ cho thế giới, thúc đẩy tiến bộ nhanh chóng trong các lĩnh vực đa dạng như thị giác máy tính, xử lý ngôn ngữ tự nhiên, nhận dạng giọng nói tự động, học củng cố và tin học y sinh. Hơn nữa, sự thành công của học sâu về rất nhiều nhiệm vụ quan tâm thực tế thậm chí đã xúc tác những phát triển trong học máy lý thuyết và thống kê. Với những tiến bộ này trong tay, giờ đây chúng ta có thể chế tạo những chiếc xe tự lái với quyền tự chủ hơn bao giờ hết (và ít tự chủ hơn một số công ty có thể có bạn tin), hệ thống trả lời thông minh tự động soạn thảo các email trần tục nhất, giúp mọi người đào thoát khỏi các hộp thư đến lớn áp bức và phần mềm các đặc vụ thống trị con người giỏi nhất thế giới tại các trò chơi trên bàn như Go, một kỳ công từng được cho là cách xa hàng thập kỷ. Đã có, những công cụ này tạo ra những tác động ngày càng rộng hơn đến ngành công nghiệp và xã hội, thay đổi cách tạo ra phim ảnh, chẩn đoán bệnh tật và đóng vai trò ngày càng tăng trong các khoa học cơ bản — từ vật lý thiên văn đến sinh học. 

## Về cuốn sách này

Cuốn sách này đại diện cho nỗ lực của chúng tôi để làm cho việc học sâu dễ tiếp cận, dạy cho bạn các khái niệm *, *context*, và *mã*. 

### Một phương tiện kết hợp mã, toán, và HTML

Để bất kỳ công nghệ điện toán nào đạt được tác động đầy đủ của nó, nó phải được hiểu rõ, tài liệu tốt và được hỗ trợ bởi các công cụ trưởng thành, được duy trì tốt. Những ý tưởng quan trọng cần được chưng cất rõ ràng, giảm thiểu thời gian lên máy bay cần đưa các học viên mới cập nhật. Thư viện trưởng thành nên tự động hóa các tác vụ phổ biến, và mã mẫu sẽ giúp các học viên dễ dàng sửa đổi, áp dụng và mở rộng các ứng dụng phổ biến cho phù hợp với nhu cầu của họ. Lấy các ứng dụng web động làm ví dụ. Mặc dù có một số lượng lớn các công ty, như Amazon, phát triển các ứng dụng web dựa trên cơ sở dữ liệu thành công trong những năm 1990, tiềm năng của công nghệ này để hỗ trợ các doanh nhân sáng tạo đã được nhận ra ở mức độ lớn hơn nhiều trong mười năm qua, do một phần phát triển mạnh mẽ, có tài liệu tốt khuôn khổ. 

Kiểm tra tiềm năng của deep learning đưa ra những thách thức độc đáo bởi vì bất kỳ ứng dụng duy nhất nào tập hợp các ngành khác nhau. Áp dụng học sâu đòi hỏi sự hiểu biết đồng thời (i) động lực để đúc một vấn đề theo một cách cụ thể; (ii) hình thức toán học của một mô hình nhất định; (iii) các thuật toán tối ưu hóa để phù hợp với các mô hình với dữ liệu; (iv) các nguyên tắc thống kê cho chúng tôi biết khi nào chúng ta nên mong đợi các mô hình của chúng tôi để khái quát hóa dữ liệu vô hình và các phương pháp thực tế để xác nhận rằng chúng có, trên thực tế, khái quát hóa; và (v) các kỹ thuật kỹ thuật cần thiết để đào tạo mô hình một cách hiệu quả, điều hướng những cạm bẫy của máy tính số và tận dụng tối đa phần cứng có sẵn. Dạy cả kỹ năng tư duy phê phán cần thiết để xây dựng các vấn đề, toán học để giải quyết chúng và các công cụ phần mềm để thực hiện các giải pháp đó ở một nơi đều thể hiện những thách thức đáng gờm. Mục tiêu của chúng tôi trong cuốn sách này là trình bày một nguồn tài nguyên thống nhất để mang lại cho các học viên sẽ được tăng tốc. 

Khi chúng tôi bắt đầu dự án sách này, không có tài nguyên nào đồng thời (i) được cập nhật; (ii) bao phủ toàn bộ bề rộng của máy học hiện đại với chiều sâu kỹ thuật đáng kể; và (iii) trình bày xen kẽ về chất lượng mà người ta mong đợi từ một cuốn sách giáo khoa hấp dẫn với mã runnable sạch mà một hy vọng sẽ tìm thấy trong hướng dẫn thực hành. Chúng tôi tìm thấy rất nhiều ví dụ về cách sử dụng một khuôn khổ học sâu nhất định (ví dụ: cách thực hiện điện toán số cơ bản với ma trận trong TensorFlow) hoặc để thực hiện các kỹ thuật cụ thể (ví dụ: đoạn mã cho LeNet, AlexNet, ResNet, v.v.) nằm rải rác trên các bài đăng blog khác nhau và kho GitHub. Tuy nhiên, những ví dụ này thường tập trung vào
*làm thế nào* để thực hiện một cách tiếp cận nhất định,
nhưng để lại cuộc thảo luận về 
*tại sao các quyết định thuật toán nhất định được đưa ra.
Mặc dù một số tài nguyên tương tác đã xuất hiện lẻ tẻ để giải quyết một chủ đề cụ thể, ví dụ, các bài đăng trên blog hấp dẫn được xuất bản trên trang web [Distill](http://distill.pub) hoặc blog cá nhân, chúng chỉ đề cập đến các chủ đề được chọn trong học sâu và thường thiếu mã liên quan. Mặt khác, trong khi một số sách giáo khoa học sâu đã xuất hiện — ví dụ, :cite:`Goodfellow.Bengio.Courville.2016`, cung cấp một cuộc khảo sát toàn diện về những điều cơ bản về học sâu — những tài nguyên này không kết hôn với các mô tả để nhận ra các khái niệm trong mã, đôi khi khiến độc giả không biết gì về cách thực hiện chúng. Hơn nữa, quá nhiều tài nguyên được ẩn đằng sau các bức tường trả tiền của các nhà cung cấp khóa học thương mại. 

Chúng tôi đặt ra để tạo ra một tài nguyên có thể (i) được cung cấp miễn phí cho tất cả mọi người; (ii) cung cấp đủ chiều sâu kỹ thuật để cung cấp một điểm khởi đầu trên con đường để thực sự trở thành một nhà khoa học máy học ứng dụng; (iii) bao gồm mã runnable, hiển thị độc giả
*làm thế nào* để giải quyết các vấn đề trong thực tế;
(iv) cho phép cập nhật nhanh chóng, bởi cả chúng tôi và cả cộng đồng nói chung; và (v) được bổ sung bởi một [forum](http://discuss.d2l.ai) để thảo luận tương tác về chi tiết kỹ thuật và trả lời các câu hỏi. 

Những mục tiêu này thường bị xung đột. Phương trình, định lý và trích dẫn được quản lý tốt nhất và đặt ra trong LateX. Mã được mô tả tốt nhất trong Python. Và các trang web có nguồn gốc trong HTML và javascript. Hơn nữa, chúng tôi muốn nội dung có thể truy cập cả dưới dạng mã thực thi, dưới dạng sách vật lý, dưới dạng PDF có thể tải xuống và trên Internet dưới dạng trang web. Hiện tại không có công cụ và không có quy trình làm việc hoàn toàn phù hợp với những nhu cầu này, vì vậy chúng tôi phải lắp ráp riêng của chúng tôi. Chúng tôi mô tả cách tiếp cận của chúng tôi một cách chi tiết trong :numref:`sec_how_to_contribute`. Chúng tôi định cư trên GitHub để chia sẻ nguồn và tạo điều kiện cho cộng đồng đóng góp, máy tính xách tay Jupyter để trộn mã, phương trình và văn bản, Sphinx như một công cụ kết xuất để tạo ra nhiều đầu ra và Discourse cho diễn đàn. Mặc dù hệ thống của chúng tôi chưa hoàn hảo, nhưng những lựa chọn này cung cấp một sự thỏa hiệp tốt giữa các mối quan tâm cạnh tranh. Chúng tôi tin rằng đây có thể là cuốn sách đầu tiên được xuất bản bằng cách sử dụng quy trình làm việc tích hợp như vậy. 

### Học bằng cách làm

Nhiều sách giáo khoa trình bày các khái niệm liên tiếp, bao gồm từng chi tiết đầy đủ. Ví dụ, sách giáo khoa xuất sắc của Chris Bishop :cite:`Bishop.2006`, dạy từng chủ đề kỹ lưỡng đến mức nhận được chương về hồi quy tuyến tính đòi hỏi một lượng công việc không tầm thường. Trong khi các chuyên gia yêu thích cuốn sách này chính xác vì sự triệt để của nó, đối với những người mới bắt đầu thực sự, tài sản này hạn chế tính hữu ích của nó như một văn bản giới thiệu. 

Trong cuốn sách này, chúng tôi sẽ dạy hầu hết các khái niệm * chỉ trong thời gian*. Nói cách khác, bạn sẽ học các khái niệm vào lúc này rằng chúng cần thiết để hoàn thành một số kết thúc thực tế. Trong khi chúng tôi mất một thời gian ngay từ đầu để dạy các sơ bộ cơ bản, như đại số tuyến tính và xác suất, chúng tôi muốn bạn nếm thử sự hài lòng khi đào tạo mô hình đầu tiên của mình trước khi lo lắng về phân phối xác suất bí truyền hơn. 

Bên cạnh một vài sổ ghi chép sơ bộ cung cấp một khóa học sụp đổ trong nền toán học cơ bản, mỗi chương tiếp theo giới thiệu cả một số khái niệm mới hợp lý và cung cấp các ví dụ làm việc khép kín duy nhất—sử dụng bộ dữ liệu thực. Điều này trình bày một thách thức tổ chức. Một số mô hình có thể được nhóm lại với nhau một cách hợp lý trong một sổ ghi chép duy nhất. Và một số ý tưởng có thể được dạy tốt nhất bằng cách thực hiện một số mô hình liên tiếp. Mặt khác, có một lợi thế lớn để tuân thủ chính sách của * một ví dụ làm việc, một máy tính xách túc*: Điều này giúp bạn dễ dàng bắt đầu các dự án nghiên cứu của riêng mình bằng cách tận dụng mã của chúng tôi. Chỉ cần sao chép một sổ ghi chép và bắt đầu sửa đổi nó. 

Chúng tôi sẽ interleave mã runnable với tài liệu nền khi cần thiết. Nói chung, chúng ta thường sẽ sai về phía làm cho các công cụ có sẵn trước khi giải thích chúng đầy đủ (và chúng tôi sẽ theo dõi bằng cách giải thích nền sau). Ví dụ: chúng ta có thể sử dụng *stochastic gradient descent* trước khi giải thích đầy đủ lý do tại sao nó hữu ích hoặc tại sao nó hoạt động. Điều này giúp cho các học viên những đạn dược cần thiết để giải quyết vấn đề một cách nhanh chóng, với chi phí yêu cầu người đọc tin tưởng chúng tôi với một số quyết định giám tuyển. 

Cuốn sách này sẽ dạy các khái niệm học sâu từ đầu. Đôi khi, chúng tôi muốn đi sâu vào các chi tiết tốt về các mô hình thường sẽ bị ẩn khỏi người dùng bằng các trừu tượng nâng cao của khung học sâu. Điều này đặc biệt xuất hiện trong các hướng dẫn cơ bản, nơi chúng tôi muốn bạn hiểu mọi thứ xảy ra trong một lớp hoặc trình tối ưu hóa nhất định. Trong những trường hợp này, chúng tôi thường sẽ trình bày hai phiên bản của ví dụ: một trong đó chúng tôi thực hiện mọi thứ từ đầu, chỉ dựa vào chức năng giống như NumPy và sự khác biệt tự động, và một ví dụ khác, thực tế hơn, nơi chúng tôi viết mã ngắn gọn bằng cách sử dụng API cấp cao của các khuôn khổ học sâu. Khi chúng tôi đã dạy bạn cách thức hoạt động của một số component, chúng ta chỉ có thể sử dụng các API cấp cao trong các hướng dẫn tiếp theo. 

### Nội dung và cấu trúc

Cuốn sách có thể được chia thành ba phần, tập trung vào sơ bộ, kỹ thuật học sâu, và các chủ đề nâng cao tập trung vào các hệ thống và ứng dụng thực tế (:numref:`fig_book_org`). 

![Book structure](../img/book-org.svg)
:label:`fig_book_org`

* Phần đầu tiên bao gồm những điều cơ bản và sơ bộ.
:numref:`chap_introduction` cung cấp một giới thiệu về học sâu. Sau đó, vào năm :numref:`chap_preliminaries`, chúng tôi nhanh chóng đưa bạn tăng tốc các điều kiện tiên quyết cần thiết cho việc học sâu thực hành, chẳng hạn như cách lưu trữ và thao tác dữ liệu và cách áp dụng các phép toán số khác nhau dựa trên các khái niệm cơ bản từ đại số tuyến tính, tính toán và xác suất. :numref:`chap_linear` và :numref:`chap_perceptrons` bao gồm nhiều nhất các khái niệm và kỹ thuật cơ bản trong học sâu, bao gồm hồi quy và phân loại; mô hình tuyến tính và nhận thức đa lớp; và overfitting và regarization. 

* Năm chương tiếp theo tập trung vào các kỹ thuật học sâu hiện đại.
:numref:`chap_computation` mô tả các thành phần tính toán chính của các hệ thống học sâu và đặt nền tảng cho việc triển khai các mô hình phức tạp hơn tiếp theo của chúng tôi. Tiếp theo, :numref:`chap_cnn` và :numref:`chap_modern_cnn`, giới thiệu các mạng thần kinh phức tạp (CNN), các công cụ mạnh mẽ tạo thành xương sống của hầu hết các hệ thống thị giác máy tính hiện đại. Tương tự, :numref:`chap_rnn` và :numref:`chap_modern_rnn` giới thiệu các mạng nơ-ron tái phát (RNNs), các mô hình khai thác cấu trúc tuần tự (ví dụ, tạm thời) trong dữ liệu và thường được sử dụng để xử lý ngôn ngữ tự nhiên và dự đoán chuỗi thời gian. Năm :numref:`chap_attention`, chúng tôi giới thiệu một lớp mô hình tương đối mới dựa trên cái gọi là cơ chế chú ý đã thay thế RNN là kiến trúc thống trị cho hầu hết các nhiệm vụ xử lý ngôn ngữ tự nhiên. Những phần này sẽ mang lại cho bạn tốc độ trên các công cụ mạnh mẽ và chung nhất được sử dụng rộng rãi bởi các học viên học sâu. 

* Phần ba thảo luận về khả năng mở rộng, hiệu quả và ứng dụng.
Đầu tiên, vào năm :numref:`chap_optimization`, chúng tôi thảo luận về một số thuật toán tối ưu hóa phổ biến được sử dụng để đào tạo các mô hình học sâu. Chương tiếp theo, :numref:`chap_performance`, xem xét một số yếu tố chính ảnh hưởng đến hiệu suất tính toán của mã học sâu của bạn. Trong :numref:`chap_cv`, chúng tôi minh họa các ứng dụng chính của deep learning trong tầm nhìn máy tính. Trong :numref:`chap_nlp_pretrain` và :numref:`chap_nlp_app`, chúng tôi chỉ ra cách chuẩn bị các mô hình biểu diễn ngôn ngữ và áp dụng chúng vào các nhiệm vụ xử lý ngôn ngữ tự nhiên. 

### Mã
:label:`sec_code`

Hầu hết các phần của cuốn sách này đều có mã thực thi. Chúng tôi tin rằng một số trực giác được phát triển tốt nhất thông qua thử và sai, tinh chỉnh mã theo những cách nhỏ và quan sát kết quả. Lý tưởng nhất, một lý thuyết toán học thanh lịch có thể cho chúng ta biết chính xác làm thế nào để tinh chỉnh mã của chúng tôi để đạt được kết quả mong muốn. Tuy nhiên, ngày nay các học viên học sâu ngày nay phải thường bước đi nơi không có lý thuyết cogent nào có thể cung cấp hướng dẫn vững chắc. Bất chấp những nỗ lực tốt nhất của chúng tôi, những lời giải thích chính thức về hiệu quả của các kỹ thuật khác nhau vẫn còn thiếu, cả vì toán học để mô tả các mô hình này có thể rất khó khăn và cũng bởi vì cuộc điều tra nghiêm túc về các chủ đề này chỉ mới bắt đầu vào thiết bị cao. Chúng tôi hy vọng rằng khi lý thuyết học sâu tiến triển, các phiên bản trong tương lai của cuốn sách này có thể cung cấp những hiểu biết mà nhật thực những người hiện có. 

Để tránh sự lặp lại không cần thiết, chúng tôi đóng gói một số hàm và lớp được nhập và tham chiếu thường xuyên nhất của chúng tôi trong gói `d2l`. Để chỉ ra một khối mã, chẳng hạn như hàm, lớp hoặc tập hợp các câu lệnh import, sau đó sẽ được truy cập thông qua gói `d2l`, chúng ta sẽ đánh dấu nó bằng `# @save `. Chúng tôi cung cấp một cái nhìn tổng quan chi tiết về các chức năng và lớp trong :numref:`sec_d2l`. Gói `d2l` có trọng lượng nhẹ và chỉ yêu cầu các phụ thuộc sau:

```{.python .input}
#@tab all
#@save
import collections
from collections import defaultdict
from IPython import display
import math
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
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

:begin_tab:`mxnet`
Hầu hết các mã trong cuốn sách này dựa trên Apache MXNet, một khuôn khổ mã nguồn mở cho deep learning là sự lựa chọn ưa thích của AWS (Amazon Web Services), cũng như nhiều trường cao đẳng và công ty. Tất cả các mã trong cuốn sách này đã vượt qua các bài kiểm tra dưới phiên bản MXNet mới nhất. Tuy nhiên, do sự phát triển nhanh chóng của deep learning, một số mã*trong phiên bản in* có thể không hoạt động đúng trong các phiên bản MXNet trong tương lai. Chúng tôi dự định cập nhật phiên bản trực tuyến. Trong trường hợp bạn gặp bất kỳ vấn đề nào, vui lòng tham khảo :ref:`chap_installation` để cập nhật mã và môi trường thời gian chạy của bạn. 

Dưới đây là cách chúng tôi nhập các mô-đun từ MXNet.
:end_tab:

:begin_tab:`pytorch`
Hầu hết các mã trong cuốn sách này dựa trên PyTorch, một khuôn khổ mã nguồn mở cực kỳ phổ biến đã được cộng đồng nghiên cứu sâu đón nhận nhiệt tình. Tất cả các mã trong cuốn sách này đã vượt qua các bài kiểm tra theo phiên bản ổn định mới nhất của PyTorch. Tuy nhiên, do sự phát triển nhanh chóng của deep learning, một số mã*trong phiên bản in* có thể không hoạt động đúng trong các phiên bản PyTorch trong tương lai. Chúng tôi dự định cập nhật phiên bản trực tuyến. Trong trường hợp bạn gặp bất kỳ vấn đề nào, vui lòng tham khảo :ref:`chap_installation` để cập nhật mã và môi trường thời gian chạy của bạn. 

Dưới đây là cách chúng tôi nhập các mô-đun từ PyTorch.
:end_tab:

:begin_tab:`tensorflow`
Hầu hết các mã trong cuốn sách này dựa trên TensorFlow, một khuôn khổ mã nguồn mở cho deep learning được áp dụng rộng rãi trong ngành công nghiệp và phổ biến trong các dự trữ. Tất cả các mã trong cuốn sách này đã vượt qua các bài kiểm tra dưới phiên bản ổn định mới nhất TensorFlow. Tuy nhiên, do sự phát triển nhanh chóng của deep learning, một số code * trong phiên bản in* có thể không hoạt động đúng trong các phiên bản TensorFlow trong tương lai. Chúng tôi dự định cập nhật phiên bản trực tuyến. Trong trường hợp bạn gặp bất kỳ vấn đề nào, vui lòng tham khảo :ref:`chap_installation` để cập nhật mã và môi trường thời gian chạy của bạn. 

Dưới đây là cách chúng tôi nhập các mô-đun từ TensorFlow.
:end_tab:

```{.python .input}
#@save
from mxnet import autograd, context, gluon, image, init, np, npx
from mxnet.gluon import nn, rnn
```

```{.python .input}
#@tab pytorch
#@save
import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
```

```{.python .input}
#@tab tensorflow
#@save
import numpy as np
import tensorflow as tf
```

### Đối tượng mục tiêu

Cuốn sách này dành cho sinh viên (đại học hoặc sau đại học), kỹ sư và nhà nghiên cứu, những người tìm kiếm một nắm vững chắc về các kỹ thuật thực tế của học sâu. Bởi vì chúng tôi giải thích mọi khái niệm từ đầu, không có nền tảng trước đó trong học sâu hoặc học máy là bắt buộc. Giải thích đầy đủ các phương pháp học sâu đòi hỏi một số toán học và lập trình, nhưng chúng tôi sẽ chỉ cho rằng bạn đến với một số điều cơ bản, bao gồm một lượng khiêm tốn của đại số tuyến tính, giải tích, xác suất, và lập trình Python. Chỉ trong trường hợp bạn quên những điều cơ bản, Phụ lục cung cấp một bồi dưỡng về hầu hết các toán học bạn sẽ tìm thấy trong cuốn sách này. Hầu hết thời gian, chúng ta sẽ ưu tiên trực giác và ý tưởng hơn sự nghiêm ngặt về toán học. Nếu bạn muốn mở rộng các nền tảng này ngoài các điều kiện tiên quyết để hiểu cuốn sách của chúng tôi, chúng tôi vui vẻ đề xuất một số tài nguyên tuyệt vời khác: Phân tích tuyến tính của Bela Bollobas :cite:`Bollobas.1999` bao gồm đại số tuyến tính và phân tích chức năng ở độ sâu lớn. Tất cả các thống kê :cite:`Wasserman.2013` cung cấp một giới thiệu tuyệt vời về thống kê. [books](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1138369918) của Joe Blitzstein và [courses](https://projects.iq.harvard.edu/stat110/home) về xác suất và suy luận là đá quý sư phạm. Và nếu bạn chưa sử dụng Python trước đây, bạn có thể muốn xem xét [Python tutorial](http://learnpython.org/) này. 

### Diễn đàn

Liên quan đến cuốn sách này, chúng tôi đã đưa ra một diễn đàn thảo luận, đặt tại [discuss.d2l.ai](https://discuss.d2l.ai/). Khi bạn có câu hỏi về bất kỳ phần nào của cuốn sách, bạn có thể tìm thấy một liên kết đến trang thảo luận liên quan ở cuối mỗi sổ ghi chép. 

## Lời ghi nhận

Chúng tôi đang mắc nợ hàng trăm người đóng góp cho cả bản nháp tiếng Anh và tiếng Trung. Họ đã giúp cải thiện nội dung và cung cấp phản hồi có giá trị. Cụ thể, chúng tôi cảm ơn mọi người đóng góp của dự thảo tiếng Anh này đã làm cho nó tốt hơn cho tất cả mọi người. ID hoặc tên GitHub của họ là (không theo thứ tự cụ thể): alxnorden, avinashingit, bowen0701, brettkoonce, Chaitanya Prakash Bapat, cryptonaut, Davide Fiocco, edgarroman, gkutiel, John Mitro, Liang Pu, Rahul Agarwal, Mohamed Ali Stoui, Michael (u) Stewart, Mike Müller, NRauschmayr, Prakhar Srivastav, buồn, sfermigier, Sheng Zha, sundeepteki, topecongiro, tpdi, bún, Vishaal Kapoor, Vishwesh Ravi Shrimali, YaYaB, Yuhong Chen, Evgeniy Smirnov, lgov, Simon Corston-Oliver, Igor Dzreyev, Hà Nguyên, Lupmuens, Andrei venko, senorcinco, vfdev-5, dsweet, Mohammad Mahdi Rahimi, Abhishek Gupta, uwsd, DomKM, Lisa Oakley, Bowen Li, Aarush Ahuja, Prasanth Buddareddygari, brianhendee, mani2106, mtn, lkevinzc, caojilin, Lakshya, Fiete Lüer, Surbhi Vijayvargeeya, Muhyun Kim, dennismalmgren, adursun, Anirqudh Dagar, LiAnh, Pedro Larroy, lgov, và ozgur, Jun Wu, Matthias Blume, Lin Yuan, geogunow, Josh Gardner, Maximilian Böther, Hồi giáo Rakib, Leonard Lausen, Abhinav Upadhyay, rongruosong, Steve Sedlmeyer, Ruslan Baratov, Rafael Schlatter, liusy182, Giannis Pappas, ati-ozgur, qbaza, dchoi77, Adam Gerson, Phúc Lê, Mark Atwood, christabella, vn09, Haibin Lin, jjangga0214, RichyChen, Noelo, hansent, Giel Dops, dvincent1337, WhiteD3vil , Peter Kulit, codypenta, joseppinilla, ahmaurya, karolszk, heytitle, Peter Goetz, rigtorp, Tiep Vu, sfilip, mlxd, Kale-ab Tessera, Sanjar Adilov, MatteoFerrara, hsneto, Katarzyna Biesialska, Gregory Bruss, Duy—Thanh Đoàn, paulaurel, graytowne, Đức Phạm, sl7423, Jaedong Hwang, Yida Wang, cys4, cys4, cysm, Jean Kaddour , austinmw, trebeljahr, tbaums, Cường V Nguyễn, pavelkomarov, vzlamal, NotAnotherSystem, J-Arun-Mani, jancio, eldarkurtic, the great-shazbot, doctorcolossus, gducharme, cclauss, Daniel-Mietchen, hoonose, avagiom, abhinagiom sp0730, jonathanhrandall, ysraell, Nodar Okroshiashvili, UgurKap, Jiyang Kang, StevenJokes, Tomer Kaftan, liweiwp, netyster, ypandya, NishantTharani, heiligerl, SportsTHU, Hoa Nguyen, manuel-arno-korfmann-webentwicklung, aterzis-personal, nxby, Xiaoting He, Josiah Yoder, mathresearch, mzz2017, jroberayalas, iluu, ejc, BSharmi, vkramdev, simonwardjones, LakshKD, TalNeoran, Djliden, Nikhil95, Oren Barkan, guoweis, haozhu233, pratikhack, Yue Ying, tayfununal, steinsag, charleybeller, Andrew Lumsdaine, Jiekui Zhang, Deepak Pathak, Florian Donhauser, Tim Gates, Adriaan Tijsseling, Ron Medina, Gaurav Saha, Murat Semerci, Lei Mao, Levi McClam enny, Joshua Broyde, jake221, jonbally, zyhazwraith, Brian Pulfer, Nick Tomasino, Lefan Zhang, Hongshen Yang, Vinney Cavallo, Yuntai, Yuanxiang Zhu, Amarazov, Pasricha, Ben Greenawald, Shivam Upadhyay, Quanshangze Du, Biswajit Sahoo, Parthe Pandit, Ishan Kumar, HomunculusK, Lane Schwartz, Varadgunjal, Jason Wiener, Armin Gholampoor, Shreshtha13, Eigen-arnav, Hyeonggyu Kim, EmilyOng, Bálint Mucsányi, Chase DuBois. 

Chúng tôi cảm ơn Amazon Web Services, đặc biệt là Swami Sivasubramanian, Peter DeSantis, Adam Selipsky và Andrew Jassy vì sự hỗ trợ hào phóng của họ trong việc viết cuốn sách này. Nếu không có thời gian có sẵn, nguồn lực, thảo luận với đồng nghiệp và khuyến khích liên tục, cuốn sách này sẽ không xảy ra. 

## Tóm tắt

* Deep learning đã cách mạng hóa nhận dạng mẫu, giới thiệu công nghệ hiện nay cung cấp năng lượng cho một loạt các công nghệ, bao gồm tầm nhìn máy tính, xử lý ngôn ngữ tự nhiên, nhận dạng giọng nói tự động.
* Để áp dụng thành công deep learning, bạn phải hiểu làm thế nào để giải quyết một vấn đề, toán học của mô hình hóa, các thuật toán để phù hợp với mô hình của bạn với dữ liệu, và các kỹ thuật kỹ thuật để thực hiện tất cả.
* Cuốn sách này trình bày một nguồn tài nguyên toàn diện, bao gồm văn xuôi, số liệu, toán học và mã, tất cả ở một nơi.
* Để trả lời các câu hỏi liên quan đến cuốn sách này, hãy truy cập diễn đàn của chúng tôi tại https://discuss.d2l.ai/.
* Tất cả các máy tính xách tay đều có sẵn để tải xuống trên GitHub.

## Bài tập

1. Đăng ký một tài khoản trên diễn đàn thảo luận của cuốn sách này [discuss.d2l.ai](https://discuss.d2l.ai/).
1. Cài đặt Python trên máy tính của bạn.
1. Thực hiện theo các liên kết ở dưới cùng của phần đến diễn đàn, nơi bạn sẽ có thể tìm kiếm sự giúp đỡ và thảo luận về cuốn sách và tìm câu trả lời cho câu hỏi của bạn bằng cách thu hút các tác giả và cộng đồng rộng hơn.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/18)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/20)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/186)
:end_tab:

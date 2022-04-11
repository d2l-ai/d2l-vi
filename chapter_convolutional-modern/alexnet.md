# Mạng thần kinh phức tạp sâu (AlexNet)
:label:`sec_alexnet`

Mặc dù CNN nổi tiếng trong thị giác máy tính và cộng đồng học máy sau khi giới thiệu LeNet, họ đã không thống trị ngay lập tức lĩnh vực này. Mặc dù LeNet đạt được kết quả tốt trên các tập dữ liệu nhỏ ban đầu, nhưng hiệu suất và tính khả thi của việc đào tạo CNN trên các tập dữ liệu lớn hơn, thực tế hơn vẫn chưa được thiết lập. Trên thực tế, trong phần lớn thời gian can thiệp giữa đầu thập niên 1990 và kết quả đầu nguồn của năm 2012, các mạng thần kinh thường bị vượt qua bởi các phương pháp học máy khác, chẳng hạn như các máy vector hỗ trợ. 

Đối với tầm nhìn máy tính, so sánh này có lẽ không công bằng. Đó là mặc dù các đầu vào cho các mạng tích hợp bao gồm các giá trị pixel thô hoặc được xử lý nhẹ (ví dụ: bằng cách định tâm), các học viên sẽ không bao giờ cung cấp các pixel thô vào các mô hình truyền thống. Thay vào đó, đường ống tầm nhìn máy tính điển hình bao gồm các đường ống trích xuất tính năng kỹ thuật thủ công. Thay vì * tìm hiểu các tính năng*, các tính năng là * crafted*. Hầu hết các tiến bộ đến từ việc có nhiều ý tưởng thông minh hơn cho các tính năng, và thuật toán học tập thường được chuyển xuống một suy nghĩ sau. 

Mặc dù một số máy gia tốc mạng thần kinh đã có sẵn trong những năm 1990, nhưng chúng chưa đủ mạnh để tạo ra các CNN đa kênh sâu, đa lớp với một số lượng lớn các tham số. Hơn nữa, các bộ dữ liệu vẫn còn tương đối nhỏ. Thêm vào những trở ngại này, các thủ thuật chính để đào tạo các mạng thần kinh bao gồm heuristics khởi tạo tham số, các biến thể thông minh của gốc gradient ngẫu nhiên, chức năng kích hoạt không squashing, và các kỹ thuật chính quy hóa hiệu quả vẫn còn thiếu. 

Do đó, thay vì đào tạo hệ thống * end-to-end* (pixel để phân loại), các đường ống cổ điển trông như thế này hơn: 

1. Lấy một bộ dữ liệu thú vị. Trong những ngày đầu, các bộ dữ liệu này yêu cầu các cảm biến đắt tiền (vào thời điểm đó, hình ảnh 1 megapixel là hiện đại).
2. Xử lý trước bộ dữ liệu với các tính năng thủ công dựa trên một số kiến thức về quang học, hình học, các công cụ phân tích khác và thỉnh thoảng trên những khám phá ngẫu nhiên của sinh viên tốt nghiệp may mắn.
3. Cung cấp dữ liệu thông qua một bộ trích xuất tính năng tiêu chuẩn như SIFT (biến đổi tính năng bất biến quy mô) :cite:`Lowe.2004`, SURF (tăng tốc các tính năng mạnh mẽ) :cite:`Bay.Tuytelaars.Van-Gool.2006` hoặc bất kỳ số lượng đường ống điều chỉnh bằng tay nào khác.
4. Dump các biểu diễn kết quả vào phân loại yêu thích của bạn, có thể là một mô hình tuyến tính hoặc phương pháp hạt nhân, để đào tạo một phân loại.

Nếu bạn nói chuyện với các nhà nghiên cứu machine learning, họ tin rằng machine learning vừa quan trọng vừa đẹp. Các lý thuyết thanh lịch đã chứng minh các tính chất của các nhà phân loại khác nhau. Lĩnh vực máy học rất phát triển mạnh, nghiêm ngặt và hữu ích. Tuy nhiên, nếu bạn nói chuyện với một nhà nghiên cứu thị giác máy tính, bạn sẽ nghe một câu chuyện rất khác. Sự thật bẩn thỉu của nhận dạng hình ảnh, họ sẽ nói với bạn, đó là các tính năng, không phải học thuật toán, đã thúc đẩy tiến bộ. Các nhà nghiên cứu thị giác máy tính tin một cách hợp lý rằng một bộ dữ liệu lớn hơn hoặc sạch hơn một chút hoặc một đường ống trích xuất tính năng được cải thiện một chút quan trọng hơn nhiều so với độ chính xác cuối cùng so với bất kỳ thuật toán học tập nào. 

## Đại diện học tập

Một cách khác để đúc tình trạng của các vấn đề là phần quan trọng nhất của đường ống là đại diện. Và cho đến năm 2012, đại diện đã được tính toán bằng máy móc. Trong thực tế, kỹ thuật một bộ mới của chức năng tính năng, cải thiện kết quả, và viết lên phương pháp là một thể loại giấy nổi bật. SIFT :cite:`Lowe.2004`, SURF :cite:`Bay.Tuytelaars.Van-Gool.2006`, HOG (biểu đồ của gradient định hướng) :cite:`Dalal.Triggs.2005`, [bags of visual words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision) và các máy chiết xuất tính năng tương tự cai trị roost. 

Một nhóm các nhà nghiên cứu khác, bao gồm Yann LeCun, Geoff Hinton, Yoshua Bengio, Andrew Ng, Shun-ichi Amari, và Juergen Schmidhuber, có những kế hoạch khác nhau. Họ tin rằng bản thân các tính năng nên được học. Hơn nữa, họ tin rằng để phức tạp một cách hợp lý, các tính năng phải được cấu tạo theo thứ bậc với nhiều lớp học chung, mỗi lớp có các tham số có thể học được. Trong trường hợp của một hình ảnh, các lớp thấp nhất có thể đến để phát hiện các cạnh, màu sắc và kết cấu. Thật vậy, Alex Krizhevsky, Ilya Sutkever và Geoff Hinton đã đề xuất một biến thể mới của CNN,
*AlexNet*,
đạt được hiệu suất xuất sắc trong thử thách ImageNet 2012. AlexNet được đặt theo tên Alex Krizhevsky, tác giả đầu tiên của bài phân loại ImageNet đột phá :cite:`Krizhevsky.Sutskever.Hinton.2012`. 

Điều thú vị là ở các lớp thấp nhất của mạng, mô hình đã học được tính năng trích xuất giống như một số bộ lọc truyền thống. :numref:`fig_filters` được sao chép từ giấy AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012` và mô tả mô tả hình ảnh cấp thấp hơn. 

![Image filters learned by the first layer of AlexNet.](../img/filters.png)
:width:`400px`
:label:`fig_filters`

Các lớp cao hơn trong mạng có thể xây dựng dựa trên các biểu diễn này để đại diện cho các cấu trúc lớn hơn, như mắt, mũi, lưỡi cỏ, v.v. Thậm chí các lớp cao hơn có thể đại diện cho toàn bộ các đối tượng như người, máy bay, chó, hoặc frisbees. Cuối cùng, trạng thái ẩn cuối cùng học được một biểu diễn nhỏ gọn của hình ảnh tóm tắt nội dung của nó sao cho dữ liệu thuộc các danh mục khác nhau có thể dễ dàng tách ra. 

Trong khi bước đột phá cuối cùng cho các CNN nhiều lớp đến vào năm 2012, một nhóm các nhà nghiên cứu cốt lõi đã cống hiến hết mình cho ý tưởng này, cố gắng tìm hiểu các đại diện phân cấp của dữ liệu trực quan trong nhiều năm. Bước đột phá cuối cùng vào năm 2012 có thể được quy cho hai yếu tố chính. 

### Thiếu thành phần: Dữ liệu

Các mô hình sâu với nhiều lớp đòi hỏi một lượng lớn dữ liệu để vào chế độ mà chúng vượt trội hơn đáng kể các phương pháp truyền thống dựa trên tối ưu hóa lồi (ví dụ, phương pháp tuyến tính và hạt nhân). Tuy nhiên, với khả năng lưu trữ hạn chế của máy tính, chi phí tương đối của các cảm biến, và ngân sách nghiên cứu tương đối chặt chẽ hơn trong những năm 1990, hầu hết các nghiên cứu đều dựa vào các tập dữ liệu nhỏ. Nhiều giấy tờ đề cập đến bộ sưu tập dữ liệu UCI, nhiều trong số đó chỉ chứa hàng trăm hoặc (một vài) hàng ngàn hình ảnh được chụp trong các cài đặt không tự nhiên với độ phân giải thấp. 

Năm 2009, tập dữ liệu ImageNet được phát hành, thách thức các nhà nghiên cứu tìm hiểu các mô hình từ 1 triệu ví dụ, 1000 mỗi loại từ 1000 loại đối tượng riêng biệt. Các nhà nghiên cứu, dẫn đầu bởi Fei-Fei Li, người đã giới thiệu tập dữ liệu này đã tận dụng Google Image Search để lọc trước các bộ ứng cử viên lớn cho từng danh mục và sử dụng đường ống cộng đồng Amazon Mechanical Turk để xác nhận cho mỗi hình ảnh xem nó có thuộc về danh mục liên quan hay không. Quy mô này là chưa từng có. Cuộc cạnh tranh liên quan, được mệnh danh là ImageNet Challenge đã đẩy tầm nhìn máy tính và nghiên cứu máy học tiến lên phía trước, thách thức các nhà nghiên cứu để xác định mô hình nào hoạt động tốt nhất ở quy mô lớn hơn so với các học giả trước đây đã xem xét. 

### Thiếu thành phần: Phần cứng

Mô hình học sâu là những người tiêu dùng phàm ăn của chu kỳ tính toán. Đào tạo có thể mất hàng trăm kỷ nguyên, và mỗi lần lặp lại yêu cầu truyền dữ liệu qua nhiều lớp hoạt động đại số tuyến tính đắt tiền tính toán. Đây là một trong những lý do chính khiến vào những năm 1990 và đầu những năm 2000, các thuật toán đơn giản dựa trên các mục tiêu lồi được tối ưu hóa hiệu quả hơn được ưu tiên. 

*Các đơn vị xử lý đồ học* (GPU) được chứng minh là một người thay đổi trò chơi
in making chế tạo deep sâu learning học tập feasible khả thi. Những chip này từ lâu đã được phát triển để tăng tốc xử lý đồ họa để mang lại lợi ích cho các trò chơi máy tính. Đặc biệt, chúng đã được tối ưu hóa cho các sản phẩm ma thuật-vector $4 \times 4$ thông lượng cao, cần thiết cho nhiều tác vụ đồ họa máy tính. May mắn thay, toán học này rất giống với điều cần thiết để tính toán các lớp phức tạp. Khoảng thời gian đó, NVIDIA và ATI đã bắt đầu tối ưu hóa GPU cho các hoạt động điện toán chung, đi xa đến mức tiếp thị chúng như * GPU đa năng (GPGPU). 

Để cung cấp một số trực giác, hãy xem xét các lõi của bộ vi xử lý hiện đại (CPU). Mỗi lõi là khá mạnh mẽ chạy ở tần số đồng hồ cao và thể thao bộ nhớ cache lớn (lên đến vài megabyte L3). Mỗi lõi rất phù hợp để thực hiện một loạt các hướng dẫn, với các dự báo nhánh, một đường ống sâu, và các chuông và còi khác cho phép nó chạy một loạt các chương trình. Tuy nhiên, sức mạnh rõ ràng này cũng là gót chân Achilles của nó: lõi mục đích chung rất tốn kém để xây dựng. Chúng yêu cầu nhiều khu vực chip, cấu trúc hỗ trợ tinh vi (giao diện bộ nhớ, logic bộ nhớ đệm giữa các lõi, kết nối tốc độ cao, v.v.) và chúng tương đối xấu ở bất kỳ tác vụ duy nhất nào. Máy tính xách tay hiện đại có tới 4 lõi và thậm chí các máy chủ cao cấp hiếm khi vượt quá 64 lõi, đơn giản vì nó không hiệu quả về chi phí. 

Bằng cách so sánh, GPU bao gồm $100 \sim 1000$ các yếu tố xử lý nhỏ (các chi tiết khác nhau phần nào giữa NVIDIA, ATI, ARM và các nhà cung cấp chip khác), thường được nhóm thành các nhóm lớn hơn (NVIDIA gọi chúng là cong vênh). Trong khi mỗi lõi tương đối yếu, đôi khi thậm chí chạy ở tần số xung nhịp dưới 1GHz, đó là tổng số lõi như vậy làm cho các đơn đặt hàng GPU có độ lớn nhanh hơn CPU. Ví dụ, thế hệ Volta gần đây của NVIDIA cung cấp lên đến 120 TFlops mỗi chip cho các hướng dẫn chuyên ngành (và lên đến 24 TFlops cho các mục đích chung hơn), trong khi hiệu suất điểm nổi của CPU không vượt quá 1 TFlop cho đến nay. Lý do tại sao điều này là có thể thực sự khá đơn giản: đầu tiên, tiêu thụ điện năng có xu hướng tăng * quadratically* với tần số xung nhịp. Do đó, đối với ngân sách năng lượng của lõi CPU chạy nhanh hơn 4 lần (một số điển hình), bạn có thể sử dụng 16 lõi GPU ở tốc độ $1/4$, mang lại hiệu suất $16 \times 1/4 = 4$ lần. Hơn nữa, các lõi GPU đơn giản hơn nhiều (trên thực tế, trong một thời gian dài, chúng thậm chí không thể * để thực thi mã mục đích chung), điều này làm cho chúng tiết kiệm năng lượng hơn. Cuối cùng, nhiều hoạt động trong deep learning yêu cầu băng thông bộ nhớ cao. Một lần nữa, GPU tỏa sáng ở đây với các xe buýt có độ rộng ít nhất 10 lần so với nhiều CPU. 

Quay lại năm 2012. Một bước đột phá lớn đến khi Alex Krizhevsky và Ilya Sutkever triển khai một CNN sâu có thể chạy trên phần cứng GPU. Họ nhận ra rằng các tắc nghẽn tính toán trong CNN s, sự phức tạp và nhân ma trận, là tất cả các hoạt động có thể được song song trong phần cứng. Sử dụng hai NVIDIA GTX 580s với 3GB bộ nhớ, họ đã thực hiện các kết hợp nhanh. Mã [cuda-convnet](https://code.google.com/archive/p/cuda-convnet/) đủ tốt để trong nhiều năm, đó là tiêu chuẩn ngành và cung cấp cho vài năm đầu tiên của sự bùng nổ học tập sâu. 

## AlexNet

AlexNet, sử dụng CNN 8 lớp, đã giành chiến thắng trong ImageNet Large Scale Visual Recognition Challenge 2012 bởi một biên độ lớn phi thường. Mạng này lần đầu tiên cho thấy rằng các tính năng thu được bằng cách học tập có thể vượt qua các tính năng được thiết kế thủ công, phá vỡ mô hình trước đó trong tầm nhìn máy tính. 

Các kiến trúc của AlexNet và LeNet rất giống nhau, như :numref:`fig_alexnet` minh họa. Lưu ý rằng chúng tôi cung cấp một phiên bản hơi hợp lý của AlexNet loại bỏ một số điều kỳ lạ thiết kế cần thiết trong năm 2012 để làm cho mô hình phù hợp với hai GPU nhỏ. 

![From LeNet (left) to AlexNet (right).](../img/alexnet.svg)
:label:`fig_alexnet`

Các triết lý thiết kế của AlexNet và LeNet rất giống nhau, nhưng cũng có sự khác biệt đáng kể. Đầu tiên, AlexNet sâu hơn nhiều so với LeNet5 tương đối nhỏ. AlexNet bao gồm tám lớp: năm lớp phức tạp, hai lớp ẩn được kết nối hoàn toàn và một lớp đầu ra được kết nối hoàn toàn. Thứ hai, AlexNet sử dụng ReLU thay vì sigmoid làm chức năng kích hoạt của nó. Hãy để chúng tôi đi sâu vào các chi tiết dưới đây. 

### Architecture

Trong lớp đầu tiên của AlexNet, hình dạng cửa sổ phức tạp là $11\times11$. Vì hầu hết các hình ảnh trong ImageNet cao hơn mười lần và rộng hơn hình ảnh MNIST, các đối tượng trong dữ liệu ImageNet có xu hướng chiếm nhiều pixel hơn. Do đó, một cửa sổ phức tạp lớn hơn là cần thiết để chụp đối tượng. Hình dạng cửa sổ phức tạp trong lớp thứ hai được giảm xuống còn $5\times5$, tiếp theo là $3\times3$. Ngoài ra, sau các lớp phức tạp thứ nhất, thứ hai và thứ năm, mạng thêm các lớp tổng hợp tối đa với hình dạng cửa sổ là $3\times3$ và một sải chân là 2. Hơn nữa, AlexNet có kênh phức tạp gấp mười lần so với LeNet. 

Sau lớp phức tạp cuối cùng có hai lớp kết nối hoàn toàn với 4096 đầu ra. Hai lớp được kết nối hoàn toàn khổng lồ này tạo ra các thông số mô hình gần 1 GB. Do bộ nhớ hạn chế trong các GPU đầu tiên, AlexNet ban đầu đã sử dụng một thiết kế luồng dữ liệu kép, để mỗi trong hai GPU của họ có thể chịu trách nhiệm lưu trữ và tính toán chỉ một nửa mô hình của nó. May mắn thay, bộ nhớ GPU hiện nay tương đối phong phú, vì vậy chúng ta hiếm khi cần chia tay các mô hình trên GPU trong những ngày này (phiên bản của mô hình AlexNet của chúng tôi lệch khỏi giấy gốc ở khía cạnh này). 

### Chức năng kích hoạt

Bên cạnh đó, AlexNet đã thay đổi chức năng kích hoạt sigmoid thành chức năng kích hoạt ReLU đơn giản hơn. Một mặt, việc tính toán chức năng kích hoạt ReLU đơn giản hơn. Ví dụ, nó không có hoạt động số mũ được tìm thấy trong chức năng kích hoạt sigmoid. Mặt khác, chức năng kích hoạt ReLU giúp đào tạo mô hình dễ dàng hơn khi sử dụng các phương pháp khởi tạo tham số khác nhau. Điều này là do, khi đầu ra của chức năng kích hoạt sigmoid rất gần 0 hoặc 1, gradient của các vùng này gần như là 0, do đó sự lan truyền ngược không thể tiếp tục cập nhật một số tham số mô hình. Ngược lại, gradient của hàm kích hoạt ReLU trong khoảng dương luôn là 1. Do đó, nếu các tham số mô hình không được khởi tạo đúng cách, hàm sigmoid có thể thu được một gradient gần 0 trong khoảng dương, do đó mô hình không thể được đào tạo hiệu quả. 

### Kiểm soát công suất và tiền xử lý

AlexNet kiểm soát độ phức tạp mô hình của lớp kết nối hoàn toàn bằng cách bỏ học (:numref:`sec_dropout`), trong khi LeNet chỉ sử dụng trọng lượng phân rã. Để tăng cường dữ liệu hơn nữa, vòng đào tạo của AlexNet đã thêm rất nhiều nâng hình ảnh, chẳng hạn như lật, cắt và thay đổi màu sắc. Điều này làm cho mô hình mạnh mẽ hơn và kích thước mẫu lớn hơn làm giảm hiệu quả overfitting. Chúng tôi sẽ thảo luận về việc tăng dữ liệu chi tiết hơn trong :numref:`sec_image_augmentation`.

```{.python .input}
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn
npx.set_np()

net = nn.Sequential()
# Here, we use a larger 11 x 11 window to capture objects. At the same time,
# we use a stride of 4 to greatly reduce the height and width of the output.
# Here, the number of output channels is much larger than that in LeNet
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

net = nn.Sequential(
    # Here, we use a larger 11 x 11 window to capture objects. At the same
    # time, we use a stride of 4 to greatly reduce the height and width of the
    # output. Here, the number of output channels is much larger than that in
    # LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Make the convolution window smaller, set padding to 2 for consistent
    # height and width across the input and output, and increase the number of
    # output channels
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    # Use three successive convolutional layers and a smaller convolution
    # window. Except for the final convolutional layer, the number of output
    # channels is further increased. Pooling layers are not used to reduce the
    # height and width of input after the first two convolutional layers
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(),
    # Here, the number of outputs of the fully-connected layer is several
    # times larger than that in LeNet. Use the dropout layer to mitigate
    # overfitting
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    # Output layer. Since we are using Fashion-MNIST, the number of classes is
    # 10, instead of 1000 as in the paper
    nn.Linear(4096, 10))
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        # Here, we use a larger 11 x 11 window to capture objects. At the same
        # time, we use a stride of 4 to greatly reduce the height and width of
        # the output. Here, the number of output channels is much larger than
        # that in LeNet
        tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4,
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Make the convolution window smaller, set padding to 2 for consistent
        # height and width across the input and output, and increase the
        # number of output channels
        tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        # Use three successive convolutional layers and a smaller convolution
        # window. Except for the final convolutional layer, the number of
        # output channels is further increased. Pooling layers are not used to
        # reduce the height and width of input after the first two
        # convolutional layers
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same',
                               activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Flatten(),
        # Here, the number of outputs of the fully-connected layer is several
        # times larger than that in LeNet. Use the dropout layer to mitigate
        # overfitting
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        # Output layer. Since we are using Fashion-MNIST, the number of
        # classes is 10, instead of 1000 as in the paper
        tf.keras.layers.Dense(10)
    ])
```

Chúng tôi [** xây dựng một ví dụ dữ liệu một kênh**] với cả chiều cao và chiều rộng là 224 (** để quan sát hình dạng đầu ra của mỗi lớp**). Nó phù hợp với kiến trúc AlexNet vào năm :numref:`fig_alexnet`.

```{.python .input}
X = np.random.uniform(size=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)
```

```{.python .input}
#@tab pytorch
X = torch.randn(1, 1, 224, 224)
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output shape:\t',X.shape)
```

```{.python .input}
#@tab tensorflow
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```

## Đọc tập dữ liệu

Mặc dù AlexNet được đào tạo trên ImageNet trong giấy, chúng tôi sử dụng Fashion-MNIST ở đây vì đào tạo mô hình ImageNet để hội tụ có thể mất hàng giờ hoặc vài ngày ngay cả trên GPU hiện đại. Một trong những vấn đề với việc áp dụng AlexNet trực tiếp trên [**Fashion-MNIST**] là nó (** hình ảnh có độ phân giải thấp hơn**) ($28 \times 28$ pixel) (** hơn ImageNet images.**) Để làm cho mọi thứ hoạt động, (**chúng tôi upsample chúng lên $224 \times 224$**) (nói chung không phải là một thực hành thông minh, nhưng chúng tôi làm điều đó ở đây để trung thành với AlexNet kiến trúc). Chúng tôi thực hiện thay đổi kích thước này với đối số `resize` trong hàm `d2l.load_data_fashion_mnist`.

```{.python .input}
#@tab all
batch_size = 128
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
```

## Đào tạo

Bây giờ, chúng ta có thể [** bắt đầu đào tạo AlexNet.**] So với LeNet năm :numref:`sec_lenet`, thay đổi chính ở đây là việc sử dụng tốc độ học tập nhỏ hơn và đào tạo chậm hơn nhiều do mạng sâu hơn và rộng hơn, độ phân giải hình ảnh cao hơn và các phức tạp tốn kém hơn.

```{.python .input}
#@tab all
lr, num_epochs = 0.01, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Tóm tắt

* AlexNet có cấu trúc tương tự như của LeNet, nhưng sử dụng nhiều lớp phức tạp hơn và một không gian tham số lớn hơn để phù hợp với tập dữ liệu ImageNet quy mô lớn.
* Ngày nay AlexNet đã bị vượt qua bởi các kiến trúc hiệu quả hơn nhiều nhưng đó là một bước quan trọng từ nông đến mạng sâu được sử dụng ngày nay.
* Mặc dù có vẻ như chỉ có một vài dòng nữa trong việc thực hiện AlexNet so với LeNet, cộng đồng học thuật phải mất nhiều năm để nắm lấy sự thay đổi khái niệm này và tận dụng kết quả thử nghiệm tuyệt vời của nó. Điều này cũng là do thiếu các công cụ tính toán hiệu quả.
* Bỏ học, ReLU và tiền xử lý là những bước quan trọng khác trong việc đạt được hiệu suất tuyệt vời trong các tác vụ thị giác máy tính.

## Bài tập

1. Hãy thử tăng số lượng thời đại. So với LeNet, kết quả khác nhau như thế nào? Tại sao?
1. AlexNet có thể quá phức tạp đối với tập dữ liệu Fashion-MNIST.
    1. Hãy thử đơn giản hóa mô hình để đào tạo nhanh hơn, đồng thời đảm bảo rằng độ chính xác không giảm đáng kể.
    1. Thiết kế một mô hình tốt hơn hoạt động trực tiếp trên $28 \times 28$ hình ảnh.
1. Sửa đổi kích thước lô, và quan sát những thay đổi về độ chính xác và bộ nhớ GPU.
1. Phân tích hiệu suất tính toán của AlexNet.
    1. Phần chiếm ưu thế cho dấu chân bộ nhớ của AlexNet là gì?
    1. Phần chi phối cho tính toán trong AlexNet là gì?
    1. Làm thế nào về băng thông bộ nhớ khi tính toán kết quả?
1. Áp dụng dropout và ReLU cho LeNet-5. Nó có cải thiện không? Làm thế nào về tiền xử lý?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/75)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/76)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/276)
:end_tab:

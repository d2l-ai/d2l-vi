# Từ các lớp được kết nối hoàn toàn đến sự phức tạp
:label:`sec_why-conv`

Cho đến ngày nay, các mô hình mà chúng tôi đã thảo luận cho đến nay vẫn là các tùy chọn phù hợp khi chúng tôi đang xử lý dữ liệu dạng bảng. Theo bảng, chúng tôi có nghĩa là dữ liệu bao gồm các hàng tương ứng với các ví dụ và cột tương ứng với các tính năng. Với dữ liệu dạng bảng, chúng tôi có thể dự đoán rằng các mẫu chúng tôi tìm kiếm có thể liên quan đến tương tác giữa các tính năng, nhưng chúng tôi không giả định bất kỳ cấu trúc nào * a priori* liên quan đến cách các tính năng tương tác. 

Đôi khi, chúng tôi thực sự thiếu kiến thức để hướng dẫn việc xây dựng các kiến trúc craftier. Trong những trường hợp này, MLP có thể là tốt nhất mà chúng ta có thể làm. Tuy nhiên, đối với dữ liệu nhận thức chiều cao, các mạng không có cấu trúc như vậy có thể phát triển không tiện dụng. 

Ví dụ, chúng ta hãy quay lại ví dụ chạy của chúng tôi về việc phân biệt mèo với chó. Giả sử rằng chúng tôi thực hiện một công việc kỹ lưỡng trong việc thu thập dữ liệu, thu thập một bộ dữ liệu chú thích của các bức ảnh một megapixel. Điều này có nghĩa là mỗi đầu vào vào mạng có một triệu kích thước. Theo các cuộc thảo luận của chúng tôi về chi phí tham số hóa của các lớp được kết nối hoàn toàn trong :numref:`subsec_parameterization-cost-fc-layers`, ngay cả việc giảm tích cực xuống một nghìn kích thước ẩn sẽ đòi hỏi một lớp được kết nối hoàn toàn đặc trưng bởi các tham số $10^6 \times 10^3 = 10^9$. Trừ khi chúng ta có rất nhiều GPU, một tài năng để tối ưu hóa phân tán và một lượng kiên nhẫn phi thường, việc học các thông số của mạng này có thể trở nên không khả thi. 

Một người đọc cẩn thận có thể phản đối lập luận này trên cơ sở rằng độ phân giải một megapixel có thể không cần thiết. Tuy nhiên, trong khi chúng ta có thể thoát khỏi một trăm nghìn pixel, lớp ẩn có kích thước 1000 của chúng tôi đánh giá thấp số lượng đơn vị ẩn cần thiết để tìm hiểu các biểu diễn hình ảnh tốt, vì vậy một hệ thống thực tế vẫn sẽ yêu cầu hàng tỷ tham số. Hơn nữa, học một bộ phân loại bằng cách lắp rất nhiều tham số có thể yêu cầu thu thập một bộ dữ liệu khổng lồ. Nhưng ngày nay cả con người và máy tính đều có thể phân biệt mèo với chó khá tốt, dường như mâu thuẫn với những trực giác này. Đó là do hình ảnh thể hiện cấu trúc phong phú có thể được con người và các mô hình học máy khai thác như nhau. Mạng thần kinh phức tạp (CNN) là một cách sáng tạo mà machine learning đã chấp nhận để khai thác một số cấu trúc đã biết trong hình ảnh tự nhiên. 

## Bất biến

Hãy tưởng tượng rằng bạn muốn phát hiện một đối tượng trong một hình ảnh. Có vẻ hợp lý rằng bất cứ phương pháp nào chúng ta sử dụng để nhận ra các đối tượng không nên quá quan tâm đến vị trí chính xác của đối tượng trong ảnh. Lý tưởng nhất, hệ thống của chúng ta nên khai thác kiến thức này. Lợn thường không bay và máy bay thường không bơi. Tuy nhiên, chúng ta vẫn nên nhận ra một con lợn là một xuất hiện ở trên cùng của hình ảnh. Chúng ta có thể rút ra một số cảm hứng ở đây từ trò chơi dành cho trẻ em “Where's Waldo” (được mô tả trong :numref:`img_waldo`). Trò chơi bao gồm một số cảnh hỗn loạn bùng nổ với các hoạt động. Waldo xuất hiện ở đâu đó trong mỗi, thường ẩn nấp ở một số vị trí khó xảy ra. Mục tiêu của người đọc là xác định vị trí anh ta. Mặc dù trang phục đặc trưng của mình, điều này có thể khó khăn đáng ngạc nhiên, do số lượng lớn các phiền nhiễu. Tuy nhiên, * Waldo trông như vật* không phụ thuộc vào * nơi Waldo được định trong*. Chúng ta có thể quét hình ảnh bằng máy dò Waldo có thể gán điểm cho mỗi bản vá, cho biết khả năng bản vá có chứa Waldo. CNN hệ thống hóa ý tưởng này về * bất biến không gian*, khai thác nó để tìm hiểu các biểu diễn hữu ích với ít tham số hơn. 

![An image of the "Where's Waldo" game.](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`

Bây giờ chúng ta có thể làm cho những trực giác này trở nên cụ thể hơn bằng cách liệt kê một vài desiderata để hướng dẫn thiết kế kiến trúc mạng thần kinh phù hợp với tầm nhìn máy tính: 

1. Trong các lớp sớm nhất, mạng của chúng tôi sẽ phản hồi tương tự như cùng một bản vá, bất kể nó xuất hiện ở đâu trong hình ảnh. Nguyên tắc này được gọi là *translation invariance*.
1. Các lớp sớm nhất của mạng nên tập trung vào các vùng địa phương, mà không quan tâm đến nội dung của hình ảnh ở các vùng xa xôi. Đây là nguyên tắc *locality*. Cuối cùng, các đại diện địa phương này có thể được tổng hợp để đưa ra dự đoán ở cấp độ toàn bộ hình ảnh.

Chúng ta hãy xem điều này chuyển thành toán học như thế nào. 

## Hạn chế MLP

Để bắt đầu, chúng ta có thể xem xét một MLP với hình ảnh hai chiều $\mathbf{X}$ là đầu vào và các đại diện ẩn ngay lập tức của chúng $\mathbf{H}$ tương tự được biểu diễn dưới dạng ma trận trong toán học và là hàng chục hai chiều trong mã, trong đó cả $\mathbf{X}$ và $\mathbf{H}$ đều có hình dạng giống nhau. Hãy để cái đó chìm vào. Bây giờ chúng ta quan niệm không chỉ các đầu vào mà còn là các đại diện ẩn như sở hữu cấu trúc không gian. 

Hãy để $[\mathbf{X}]_{i, j}$ và $[\mathbf{H}]_{i, j}$ biểu thị điểm ảnh tại vị trí ($i$, $j$) trong hình ảnh đầu vào và biểu diễn ẩn, tương ứng. Do đó, để mỗi đơn vị ẩn nhận được đầu vào từ mỗi pixel đầu vào, chúng tôi sẽ chuyển từ sử dụng ma trận trọng lượng (như chúng tôi đã làm trước đây trong MLP s) để biểu thị các tham số của chúng tôi dưới dạng trọng lượng thứ tư $\mathsf{W}$. Giả sử rằng $\mathbf{U}$ chứa các thành kiến, chúng ta có thể chính thức thể hiện lớp được kết nối hoàn toàn như 

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned},$$

trong đó việc chuyển đổi từ $\mathsf{W}$ sang $\mathsf{V}$ là hoàn toàn mỹ phẩm cho bây giờ vì có sự tương ứng một-một giữa các hệ số trong cả hai hàng chục bậc bốn. Chúng tôi chỉ cần lập chỉ mục lại các bản đăng ký $(k, l)$ sao cho $k = i+a$ và $l = j+b$. Nói cách khác, chúng tôi đặt $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$. Các chỉ số $a$ và $b$ chạy trên cả độ lệch tích cực và âm, bao phủ toàn bộ hình ảnh. Đối với bất kỳ vị trí nhất định nào ($i$, $j$) trong biểu diễn ẩn $[\mathbf{H}]_{i, j}$, chúng tôi tính giá trị của nó bằng cách tổng hợp các pixel trong $x$, tập trung vào khoảng $(i, j)$ và có trọng số $[\mathsf{V}]_{i, j, a, b}$. 

### Dịch Bất biến

Bây giờ chúng ta hãy gọi nguyên tắc đầu tiên được thiết lập ở trên: bất biến dịch. Điều này ngụ ý rằng một sự thay đổi trong đầu vào $\mathbf{X}$ chỉ đơn giản là dẫn đến một sự thay đổi trong đại diện ẩn $\mathbf{H}$. Điều này chỉ có thể xảy ra nếu $\mathsf{V}$ và $\mathbf{U}$ không thực sự phụ thuộc vào $(i, j)$, tức là, chúng tôi có $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ và $\mathbf{U}$ là một hằng số, nói $u$. Kết quả là, chúng ta có thể đơn giản hóa định nghĩa cho $\mathbf{H}$: 

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$

Đây là một * convolution*! Chúng tôi có hiệu quả trọng điểm ảnh tại $(i+a, j+b)$ trong vùng lân cận của vị trí $(i, j)$ với hệ số $[\mathbf{V}]_{a, b}$ để có được giá trị $[\mathbf{H}]_{i, j}$. Lưu ý rằng $[\mathbf{V}]_{a, b}$ cần nhiều hệ số ít hơn $[\mathsf{V}]_{i, j, a, b}$ vì nó không còn phụ thuộc vào vị trí trong ảnh. Chúng tôi đã đạt được những tiến bộ đáng kể! 

###  Địa phương

Bây giờ chúng ta hãy gọi nguyên tắc thứ hai: địa phương. Như động lực ở trên, chúng tôi tin rằng chúng tôi không nên phải nhìn rất xa vị trí $(i, j)$ để thu thập thông tin liên quan để đánh giá những gì đang xảy ra tại $[\mathbf{H}]_{i, j}$. Điều này có nghĩa là bên ngoài một số phạm vi $|a|> \Delta$ hoặc $|b| > \Delta$, chúng ta nên đặt $[\mathbf{V}]_{a, b} = 0$. Tương đương, chúng ta có thể viết lại $[\mathbf{H}]_{i, j}$ như 

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

Lưu ý rằng :eqref:`eq_conv-layer`, tóm lại, là một lớp* phức tạp *.
*Mạng thần kinh hội lượng* (CNN)
là một họ đặc biệt của các mạng thần kinh có chứa các lớp phức tạp. Trong cộng đồng nghiên cứu deep learning, $\mathbf{V}$ được gọi là một hạt nhân * convolution*, một * filter*, hoặc đơn giản là trọng lượng* của lớp thường là các tham số có thể học được. Khi khu vực địa phương nhỏ, sự khác biệt so với mạng được kết nối hoàn toàn có thể rất ấn tượng. Trong khi trước đây, chúng ta có thể đã yêu cầu hàng tỷ tham số để đại diện cho chỉ một lớp duy nhất trong một mạng xử lý hình ảnh, bây giờ chúng ta thường chỉ cần một vài trăm, mà không làm thay đổi chiều chiều của một trong hai đầu vào hoặc biểu diễn ẩn. Giá phải trả cho việc giảm mạnh mẽ này trong các thông số là các tính năng của chúng tôi hiện đang dịch bất biến và lớp của chúng tôi chỉ có thể kết hợp thông tin cục bộ, khi xác định giá trị của mỗi kích hoạt ẩn. Tất cả việc học phụ thuộc vào sự thiên vị quy nạp áp đặt. Khi sự thiên vị đó đồng ý với thực tế, chúng tôi nhận được các mô hình hiệu quả mẫu mà khái quát hóa tốt với dữ liệu vô hình. Nhưng tất nhiên, nếu những thành kiến đó không đồng ý với thực tế, ví dụ, nếu hình ảnh hóa ra không phải là bất biến dịch thuật, các mô hình của chúng tôi có thể phải vật lộn ngay cả để phù hợp với dữ liệu đào tạo của chúng tôi. 

## Sự phức tạp

Trước khi đi xa hơn, chúng ta nên xem xét ngắn gọn lý do tại sao hoạt động trên được gọi là một sự phức tạp. Trong toán học, * convolution* giữa hai hàm, giả sử $f, g: \mathbb{R}^d \to \mathbb{R}$ được định nghĩa là 

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Đó là, chúng tôi đo sự chồng chéo giữa $f$ và $g$ khi một hàm được “lật” và dịch chuyển bởi $\mathbf{x}$. Bất cứ khi nào chúng ta có các đối tượng rời rạc, tích phân biến thành một tổng. Ví dụ, đối với vectơ từ tập hợp các vectơ chiều vô hạn có thể tổng hợp vuông với chỉ số chạy trên $\mathbb{Z}$ chúng ta có được định nghĩa sau: 

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

Đối với hàng chục hai chiều, chúng tôi có một tổng tương ứng với các chỉ số $(a, b)$ cho $f$ và $(i-a, j-b)$ cho $g$, tương ứng: 

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$
:eqlabel:`eq_2d-conv-discrete`

Điều này trông tương tự như :eqref:`eq_conv-layer`, với một sự khác biệt lớn. Thay vì sử dụng $(i+a, j+b)$, chúng tôi đang sử dụng sự khác biệt thay thế. Tuy nhiên, lưu ý rằng sự khác biệt này chủ yếu là mỹ phẩm vì chúng ta luôn có thể phù hợp với ký hiệu giữa :eqref:`eq_conv-layer` và :eqref:`eq_2d-conv-discrete`. Định nghĩa ban đầu của chúng tôi trong :eqref:`eq_conv-layer` mô tả đúng hơn một * tương quan chéo*. Chúng tôi sẽ trở lại với điều này trong phần sau. 

## “Where's Waldo” Revisited

Quay trở lại máy dò Waldo của chúng tôi, hãy để chúng tôi xem điều này trông như thế nào. Lớp phức tạp chọn các cửa sổ có kích thước nhất định và nặng cường độ theo bộ lọc $\mathsf{V}$, như được chứng minh trong :numref:`fig_waldo_mask`. Chúng ta có thể hướng đến việc tìm hiểu một mô hình để bất cứ nơi nào “waldoness” là cao nhất, chúng ta nên tìm một đỉnh trong các biểu diễn lớp ẩn. 

![Detect Waldo.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

### Kênh
:label:`subsec_why-conv-channels`

Chỉ có một vấn đề với cách tiếp cận này. Cho đến nay, chúng tôi hạnh phúc bỏ qua rằng hình ảnh bao gồm 3 kênh: đỏ, xanh lá cây và xanh dương. Trong thực tế, hình ảnh không phải là đối tượng hai chiều mà là hàng chục bậc ba, đặc trưng bởi chiều cao, chiều rộng và kênh, ví dụ, với hình dạng $1024 \times 1024 \times 3$ pixel. Trong khi hai trục đầu tiên trong số các trục này liên quan đến các mối quan hệ không gian, trục thứ ba có thể được coi là gán một biểu diễn đa chiều cho mỗi vị trí pixel. Do đó, chúng tôi chỉ số $\mathsf{X}$ là $[\mathsf{X}]_{i, j, k}$. Bộ lọc phức tạp phải thích ứng cho phù hợp. Thay vì $[\mathbf{V}]_{a,b}$, bây giờ chúng ta có $[\mathsf{V}]_{a,b,c}$. 

Hơn nữa, cũng giống như đầu vào của chúng tôi bao gồm tensor bậc ba, hóa ra là một ý tưởng tốt để xây dựng tương tự các đại diện ẩn của chúng tôi như là hàng chục bậc ba $\mathsf{H}$. Nói cách khác, thay vì chỉ có một biểu diễn ẩn duy nhất tương ứng với từng vị trí không gian, chúng ta muốn có toàn bộ vectơ biểu diễn ẩn tương ứng với từng vị trí không gian. Chúng ta có thể nghĩ về các biểu diễn ẩn như bao gồm một số lưới hai chiều xếp chồng lên nhau. Như trong các đầu vào, đôi khi chúng được gọi là * kênh*. Đôi khi chúng cũng được gọi là *feature maps*, vì mỗi người cung cấp một tập hợp các đối tượng đã học được không gian cho lớp tiếp theo. Trực giác, bạn có thể tưởng tượng rằng ở các lớp thấp hơn gần với đầu vào hơn, một số kênh có thể trở nên chuyên biệt để nhận ra các cạnh trong khi những người khác có thể nhận ra kết cấu. 

Để hỗ trợ nhiều kênh trong cả hai đầu vào ($\mathsf{X}$) và biểu diễn ẩn ($\mathsf{H}$), chúng ta có thể thêm tọa độ thứ tư vào $\mathsf{V}$:$[\mathsf{V}]_{a, b, c, d}$. Đặt mọi thứ lại với nhau chúng ta có: 

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$
:eqlabel:`eq_conv-layer-channels`

trong đó $d$ lập chỉ mục các kênh đầu ra trong các đại diện ẩn $\mathsf{H}$. Lớp phức tạp tiếp theo sẽ tiếp tục lấy tensor bậc ba, $\mathsf{H}$, làm đầu vào. Nói chung hơn, :eqref:`eq_conv-layer-channels` là định nghĩa của một lớp phức tạp cho nhiều kênh, trong đó $\mathsf{V}$ là một hạt nhân hoặc bộ lọc của lớp. 

Vẫn còn nhiều hoạt động mà chúng ta cần giải quyết. Ví dụ, chúng ta cần tìm ra cách kết hợp tất cả các biểu diễn ẩn với một đầu ra duy nhất, ví dụ, liệu có Waldo * bất cứ nơi nào* trong hình ảnh hay không. Chúng ta cũng cần quyết định cách tính toán mọi thứ một cách hiệu quả, cách kết hợp nhiều lớp, chức năng kích hoạt phù hợp và cách đưa ra các lựa chọn thiết kế hợp lý để mang lại các mạng có hiệu quả trong thực tế. Chúng tôi chuyển sang những vấn đề này trong phần còn lại của chương. 

## Tóm tắt

* Sự bất biến dịch trong hình ảnh ngụ ý rằng tất cả các bản vá của một hình ảnh sẽ được xử lý theo cách tương tự.
* Địa phương có nghĩa là chỉ một khu phố nhỏ của pixel sẽ được sử dụng để tính toán các biểu diễn ẩn tương ứng.
* Trong xử lý hình ảnh, các lớp tích hợp thường yêu cầu ít tham số hơn nhiều so với các lớp được kết nối hoàn toàn.
* CNNS là một họ đặc biệt của các mạng thần kinh có chứa các lớp phức tạp.
* Các kênh trên đầu vào và đầu ra cho phép mô hình của chúng tôi chụp nhiều khía cạnh của một hình ảnh tại mỗi vị trí không gian.

## Bài tập

1. Giả sử rằng kích thước của hạt nhân phức tạp là $\Delta = 0$. Cho thấy rằng trong trường hợp này hạt nhân phức tạp thực hiện một MLP độc lập cho mỗi tập hợp các kênh.
1. Tại sao có thể dịch bất biến không phải là một ý tưởng tốt sau khi tất cả?
1. Chúng ta phải giải quyết vấn đề gì khi quyết định cách xử lý các biểu diễn ẩn tương ứng với vị trí pixel ở ranh giới của hình ảnh?
1. Mô tả một lớp phức tạp tương tự cho âm thanh.
1. Bạn có nghĩ rằng các lớp phức tạp cũng có thể được áp dụng cho dữ liệu văn bản? Tại sao hoặc tại sao không?
1. Chứng minh rằng $f * g = g * f$.

[Discussions](https://discuss.d2l.ai/t/64)

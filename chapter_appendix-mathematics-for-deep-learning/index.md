# Phụ lục: Toán học cho Deep Learning
:label:`chap_appendix_math`

** Brent Werness** (* Amazon*), **Rachel Hu** (* Amazon*) và tác giả của cuốn sách này

Một trong những phần tuyệt vời của học sâu hiện đại là thực tế là phần lớn nó có thể được hiểu và sử dụng mà không có sự hiểu biết đầy đủ về toán học bên dưới nó. Đây là một dấu hiệu cho thấy lĩnh vực này đang trưởng thành. Cũng giống như hầu hết các nhà phát triển phần mềm không còn cần phải lo lắng về lý thuyết về các chức năng tính toán, các học viên học sâu cũng không cần phải lo lắng về nền tảng lý thuyết của khả năng học tối đa. 

Nhưng, chúng tôi chưa hoàn toàn ở đó. 

Trong thực tế, đôi khi bạn sẽ cần phải hiểu làm thế nào các lựa chọn kiến trúc ảnh hưởng đến dòng chảy gradient, hoặc các giả định ngầm mà bạn thực hiện bằng cách đào tạo với một chức năng mất mát nhất định. Bạn có thể cần phải biết những gì trong các biện pháp entropy thế giới, và làm thế nào nó có thể giúp bạn hiểu chính xác ý nghĩa của bitmỗi ký tự trong mô hình của bạn. Tất cả đều đòi hỏi sự hiểu biết toán học sâu hơn. 

Phụ lục này nhằm mục đích cung cấp cho bạn nền toán học bạn cần để hiểu lý thuyết cốt lõi của học sâu hiện đại, nhưng nó không đầy đủ. Chúng tôi sẽ bắt đầu với việc kiểm tra đại số tuyến tính ở độ sâu lớn hơn. Chúng tôi phát triển một sự hiểu biết hình học của tất cả các đối tượng đại số tuyến tính phổ biến và các hoạt động sẽ cho phép chúng tôi hình dung các hiệu ứng của các biến đổi khác nhau trên dữ liệu của chúng tôi. Một yếu tố chính là sự phát triển của những điều cơ bản của sự phân hủy eigen-. 

Tiếp theo chúng ta phát triển lý thuyết về phép tính vi phân đến mức chúng ta có thể hiểu đầy đủ tại sao gradient là hướng của dòng dõi dốc nhất, và tại sao sự lan truyền ngược lại có dạng nó. Tích phân sau đó được thảo luận với mức độ cần thiết để hỗ trợ chủ đề tiếp theo của chúng tôi, lý thuyết xác suất. 

Các vấn đề gặp phải trong thực tế thường xuyên là không chắc chắn, và do đó chúng ta cần một ngôn ngữ để nói về những điều không chắc chắn. Chúng tôi xem xét lý thuyết về các biến ngẫu nhiên và các bản phân phối thường gặp nhất để chúng tôi có thể thảo luận về các mô hình xác suất. Điều này cung cấp nền tảng cho phân loại Bayes ngây thơ, một kỹ thuật phân loại xác suất. 

Liên quan chặt chẽ đến lý thuyết xác suất là nghiên cứu về thống kê. Mặc dù số liệu thống kê là một lĩnh vực quá lớn để thực hiện công lý trong một phần ngắn, chúng tôi sẽ giới thiệu các khái niệm cơ bản mà tất cả các học viên máy học cần phải nhận thức, đặc biệt là: đánh giá và so sánh các ước lượng, tiến hành các bài kiểm tra giả thuyết và xây dựng khoảng thời gian tự tin. 

Cuối cùng, chúng ta chuyển sang chủ đề lý thuyết thông tin, đó là nghiên cứu toán học về lưu trữ và truyền thông tin. Điều này cung cấp ngôn ngữ cốt lõi mà theo đó chúng ta có thể thảo luận về số lượng thông tin mà một mô hình nắm giữ trên một miền diễn ngôn. 

Kết hợp với nhau, những hình thành cốt lõi của các khái niệm toán học cần thiết để bắt đầu con đường hướng tới một sự hiểu biết sâu sắc về học sâu.

```toc
:maxdepth: 2

geometry-linear-algebraic-ops
eigendecomposition
single-variable-calculus
multivariable-calculus
integral-calculus
random-variables
maximum-likelihood
distributions
naive-bayes
statistics
information-theory
```

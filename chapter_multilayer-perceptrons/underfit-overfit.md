# Lựa chọn mô hình, Underfitting, và Overfitting
:label:`sec_model_selection`

Là các nhà khoa học máy học, mục tiêu của chúng tôi là khám phá * mẫu*. Nhưng làm thế nào chúng ta có thể chắc chắn rằng chúng tôi đã thực sự phát hiện ra một mô hình * general* và không chỉ đơn giản là ghi nhớ dữ liệu của chúng tôi? Ví dụ, hãy tưởng tượng rằng chúng ta muốn săn lùng các mô hình giữa các dấu hiệu di truyền liên kết bệnh nhân với tình trạng sa sút trí tuệ của họ, nơi các nhãn được rút ra từ bộ $\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$. Bởi vì gen của mỗi người xác định chúng một cách duy nhất (bỏ qua các anh chị em giống hệt nhau), có thể ghi nhớ toàn bộ tập dữ liệu. 

Chúng tôi không muốn mô hình của chúng tôi nói
*“Đó là Bob! Tôi nhớ anh ấy! Anh ấy bị mất trí nhớ!” *
Lý do tại sao đơn giản. Khi chúng tôi triển khai mô hình trong tương lai, chúng tôi sẽ gặp những bệnh nhân mà mô hình chưa từng thấy trước đây. Dự đoán của chúng tôi sẽ chỉ hữu ích nếu mô hình của chúng tôi đã thực sự phát hiện ra một mô hình * general*. 

Để tóm lại chính thức hơn, mục tiêu của chúng tôi là khám phá các mô hình nắm bắt được sự đều đặn trong dân số cơ bản mà từ đó bộ đào tạo của chúng tôi được rút ra. Nếu chúng ta thành công trong nỗ lực này, thì chúng ta có thể đánh giá thành công rủi ro ngay cả đối với những cá nhân mà chúng ta chưa bao giờ gặp phải trước đây. Vấn đề này—làm thế nào để khám phá các mẫu mà * tổng hợp* — là vấn đề cơ bản của học máy. 

Nguy hiểm là khi chúng tôi đào tạo các mô hình, chúng tôi chỉ truy cập một mẫu dữ liệu nhỏ. Các tập dữ liệu hình ảnh công cộng lớn nhất chứa khoảng một triệu hình ảnh. Thường xuyên hơn, chúng ta phải học hỏi từ chỉ hàng ngàn hoặc hàng chục ngàn ví dụ dữ liệu. Trong một hệ thống bệnh viện lớn, chúng tôi có thể truy cập hàng trăm ngàn hồ sơ y tế. Khi làm việc với các mẫu hữu hạn, chúng tôi gặp rủi ro rằng chúng tôi có thể phát hiện ra các hiệp hội rõ ràng hóa ra không giữ khi chúng tôi thu thập nhiều dữ liệu hơn. 

Hiện tượng phù hợp với dữ liệu đào tạo của chúng tôi chặt chẽ hơn chúng tôi phù hợp với phân phối cơ bản được gọi là * overfitting*, và các kỹ thuật được sử dụng để chống lại quá mức được gọi là *regarization*. Trong các phần trước, bạn có thể đã quan sát thấy hiệu ứng này trong khi thử nghiệm với bộ dữ liệu Fashion-MNIST. Nếu bạn thay đổi cấu trúc mô hình hoặc các siêu tham số trong quá trình thử nghiệm, bạn có thể nhận thấy rằng với đủ tế bào thần kinh, lớp và kỷ nguyên đào tạo, mô hình cuối cùng có thể đạt được độ chính xác hoàn hảo trên bộ đào tạo, ngay cả khi độ chính xác trên dữ liệu thử nghiệm xấu đi. 

## Lỗi đào tạo và lỗi tổng quát

Để thảo luận về hiện tượng này một cách chính thức hơn, chúng ta cần phân biệt giữa lỗi đào tạo và lỗi tổng quát hóa. Lỗi*training lỗi* là lỗi của mô hình của chúng tôi như được tính trên tập dữ liệu đào tạo, trong khi lỗi tổng quát * là kỳ vọng về lỗi mô hình của chúng tôi là chúng tôi áp dụng nó vào một dòng vô hạn các ví dụ dữ liệu bổ sung được rút ra từ cùng một phân phối dữ liệu cơ bản như mẫu ban đầu của chúng tôi. 

Vấn đề, chúng ta không bao giờ có thể tính toán chính xác lỗi tổng quát hóa. Đó là bởi vì dòng dữ liệu vô hạn là một đối tượng tưởng tượng. Trong thực tế, chúng ta phải * ước tính* lỗi tổng quát bằng cách áp dụng mô hình của chúng tôi vào một tập kiểm tra độc lập cấu thành một lựa chọn ngẫu nhiên các ví dụ dữ liệu đã được giữ lại từ bộ đào tạo của chúng tôi. 

Ba thí nghiệm suy nghĩ sau đây sẽ giúp minh họa tình huống này tốt hơn. Hãy xem xét một sinh viên đại học đang cố gắng chuẩn bị cho kỳ thi cuối cùng của mình. Một sinh viên siêng năng sẽ cố gắng luyện tập tốt và kiểm tra khả năng của mình bằng cách sử dụng các kỳ thi từ những năm trước. Tuy nhiên, làm tốt trong các kỳ thi trong quá khứ là không có gì đảm bảo rằng anh ấy sẽ xuất sắc khi nó quan trọng. Ví dụ, học sinh có thể cố gắng chuẩn bị bằng cách rote học câu trả lời cho các câu hỏi thi. Điều này đòi hỏi học sinh phải ghi nhớ nhiều thứ. Cô ấy thậm chí có thể nhớ câu trả lời cho các kỳ thi trong quá khứ một cách hoàn hảo. Một sinh viên khác có thể chuẩn bị bằng cách cố gắng hiểu lý do để đưa ra câu trả lời nhất định. Trong hầu hết các trường hợp, học sinh sau này sẽ làm tốt hơn nhiều. 

Tương tự như vậy, hãy xem xét một mô hình chỉ đơn giản là sử dụng bảng tra cứu để trả lời các câu hỏi. Nếu tập hợp các đầu vào cho phép là rời rạc và hợp lý nhỏ, thì có lẽ sau khi xem * nhiều ví dụ đào tạo, cách tiếp cận này sẽ hoạt động tốt. Tuy nhiên, mô hình này không có khả năng làm tốt hơn so với đoán ngẫu nhiên khi đối mặt với các ví dụ mà nó chưa từng thấy trước đây. Trong thực tế, không gian đầu vào quá lớn để ghi nhớ các câu trả lời tương ứng với mọi đầu vào có thể tưởng tượng được. Ví dụ: hãy xem xét các hình ảnh $28\times28$ đen và trắng. Nếu mỗi pixel có thể lấy một trong số $256$ giá trị thang màu xám, thì có $256^{784}$ hình ảnh có thể. Điều đó có nghĩa là có những hình ảnh có kích thước thu nhỏ có độ phân giải thấp hơn nhiều so với các nguyên tử trong vũ trụ. Ngay cả khi chúng ta có thể gặp phải dữ liệu như vậy, chúng ta không bao giờ có thể đủ khả năng để lưu trữ bảng tra cứu. 

Cuối cùng, hãy xem xét vấn đề cố gắng phân loại kết quả của việc ném tiền xu (lớp 0: đầu, lớp 1: đuôi) dựa trên một số tính năng theo ngữ cảnh có thể có sẵn. Giả sử rằng đồng xu là công bằng. Không có vấn đề gì thuật toán chúng tôi đưa ra, lỗi tổng quát sẽ luôn là $\frac{1}{2}$. Tuy nhiên, đối với hầu hết các thuật toán, chúng ta nên mong đợi lỗi đào tạo của chúng ta thấp hơn đáng kể, tùy thuộc vào may mắn của trận hòa, ngay cả khi chúng ta không có bất kỳ tính năng nào! Xem xét tập dữ liệu {0, 1, 1, 1, 0, 1}. Thuật toán ít tính năng của chúng tôi sẽ phải rơi trở lại luôn dự đoán các lớp *đa số*, xuất hiện từ mẫu giới hạn của chúng tôi là *1*. Trong trường hợp này, mô hình luôn dự đoán lớp 1 sẽ phát sinh lỗi $\frac{1}{3}$, tốt hơn đáng kể so với lỗi tổng quát của chúng tôi. Khi chúng ta tăng lượng dữ liệu, xác suất phần đầu sẽ đi chệch đáng kể so với $\frac{1}{2}$ giảm và lỗi đào tạo của chúng tôi sẽ phù hợp với lỗi tổng quát. 

### Lý thuyết học thống kê

Vì khái quát hóa là vấn đề cơ bản trong học máy, bạn có thể không ngạc nhiên khi biết rằng nhiều nhà toán học và nhà lý luận đã cống hiến cuộc sống của họ để phát triển các lý thuyết chính thức để mô tả hiện tượng này. Trong [định lý cùng tên] của họ (https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem), Glivenko và Cantelli bắt nguồn tốc độ mà tại đó lỗi đào tạo hội tụ đến lỗi tổng quát hóa. Trong một loạt các bài báo seminal, [Vapnik và Chervonenkis](https://en.wikipedia.org/wiki/Vapnik%E2%80%93Chervonenkis_theory) đã mở rộng lý thuyết này sang các lớp hàm tổng quát hơn. Công trình này đặt nền tảng của lý thuyết học thống kê. 

Trong cài đặt học tập được giám sát tiêu chuẩn, mà chúng tôi đã giải quyết cho đến bây giờ và sẽ gắn bó với trong suốt hầu hết cuốn sách này, chúng tôi giả định rằng cả dữ liệu đào tạo và dữ liệu thử nghiệm đều được rút ra* độc lập* từ các bản phân phối * giống hệt nhau*. Điều này thường được gọi là giả định *i.d., có nghĩa là quá trình lấy mẫu dữ liệu của chúng tôi không có bộ nhớ. Nói cách khác, ví dụ thứ hai được vẽ và vẽ thứ ba không tương quan hơn mẫu thứ hai và hai triệu được vẽ. 

Trở thành một nhà khoa học máy học giỏi đòi hỏi phải suy nghĩ nghiêm túc, và bạn đã nên chọc lỗ hổng trong giả định này, đưa ra những trường hợp phổ biến mà giả định thất bại. Điều gì sẽ xảy ra nếu chúng ta đào tạo một dự báo rủi ro tử vong trên dữ liệu thu thập từ bệnh nhân tại Trung tâm Y tế UCSF và áp dụng nó cho bệnh nhân tại Bệnh viện Đa khoa Massachusetts? Những bản phân phối này chỉ đơn giản là không giống hệt nhau. Hơn nữa, rút thăm có thể tương quan trong thời gian. Điều gì sẽ xảy ra nếu chúng ta phân loại các chủ đề của Tweets? Chu kỳ tin tức sẽ tạo ra sự phụ thuộc thời gian trong các chủ đề đang được thảo luận, vi phạm bất kỳ giả định nào về sự độc lập. 

Đôi khi chúng ta có thể thoát khỏi những vi phạm nhỏ của giả định i.i.d. và các mô hình của chúng tôi sẽ tiếp tục hoạt động tốt đáng kể. Rốt cuộc, gần như mọi ứng dụng trong thế giới thực đều liên quan đến ít nhất một số vi phạm nhỏ đối với giả định i.i..d., nhưng chúng tôi có nhiều công cụ hữu ích cho các ứng dụng khác nhau như nhận dạng khuôn mặt, nhận dạng giọng nói và dịch ngôn ngữ. 

Các vi phạm khác chắc chắn sẽ gây rắc rối. Hãy tưởng tượng, ví dụ, nếu chúng ta cố gắng đào tạo một hệ thống nhận dạng khuôn mặt bằng cách đào tạo nó độc quyền trên sinh viên đại học và sau đó muốn triển khai nó như một công cụ để theo dõi lão khoa trong một dân số viện dưỡng lão. Điều này không có khả năng làm việc tốt vì sinh viên đại học có xu hướng trông khác biệt đáng kể so với người cao tuổi. 

Trong các chương tiếp theo, chúng tôi sẽ thảo luận về các vấn đề phát sinh từ vi phạm giả định i.i.d. Hiện tại, thậm chí lấy giả định i.i.d. cho phép, hiểu khái quát hóa là một vấn đề ghê gớm. Hơn nữa, làm sáng tỏ các nền tảng lý thuyết chính xác có thể giải thích tại sao các mạng thần kinh sâu khái quát hóa cũng như chúng tiếp tục làm suy nghĩ vĩ đại nhất trong lý thuyết học tập. 

Khi chúng tôi đào tạo các mô hình của mình, chúng tôi cố gắng tìm kiếm một chức năng phù hợp với dữ liệu đào tạo cũng như có thể. Nếu chức năng linh hoạt đến mức nó có thể bắt kịp các mẫu giả dễ dàng như các liên kết thực sự, thì nó có thể thực hiện * quá tốt* mà không tạo ra một mô hình khái quát hóa tốt với dữ liệu không nhìn thấy. Đây chính xác là những gì chúng ta muốn tránh hoặc ít nhất là kiểm soát. Nhiều kỹ thuật trong học sâu là heuristics và thủ thuật nhằm bảo vệ chống lại quá mức. 

### Độ phức tạp của mô hình

Khi chúng tôi có các mô hình đơn giản và dữ liệu phong phú, chúng tôi hy vọng lỗi tổng quát hóa giống với lỗi đào tạo. Khi chúng tôi làm việc với các mô hình phức tạp hơn và ít ví dụ hơn, chúng tôi hy vọng lỗi đào tạo sẽ đi xuống nhưng khoảng cách tổng quát sẽ phát triển. Điều chính xác cấu thành sự phức tạp của mô hình là một vấn đề phức tạp. Nhiều yếu tố chi phối liệu một mô hình sẽ khái quát hóa tốt hay không. Ví dụ, một mô hình có nhiều tham số hơn có thể được coi là phức tạp hơn. Một mô hình có tham số có thể lấy một phạm vi giá trị rộng hơn có thể phức tạp hơn. Thông thường với các mạng thần kinh, chúng tôi nghĩ về một mô hình có nhiều lần lặp đào tạo phức tạp hơn và một đối tượng để * stopping sớm* (ít lặp lại đào tạo hơn) là ít phức tạp hơn. 

Có thể khó so sánh sự phức tạp giữa các thành viên của các lớp mô hình khác nhau đáng kể (ví dụ, cây quyết định so với mạng thần kinh). Hiện tại, một quy tắc đơn giản của ngón tay cái là khá hữu ích: một mô hình có thể dễ dàng giải thích các sự kiện tùy ý là những gì các nhà thống kê xem là phức tạp, trong khi một trong đó chỉ có một sức mạnh biểu cảm hạn chế nhưng vẫn quản lý để giải thích tốt dữ liệu có lẽ là gần gũi hơn với sự thật. Trong triết học, điều này có liên quan chặt chẽ đến tiêu chí Popper về tính giả mạo của một lý thuyết khoa học: một lý thuyết là tốt nếu nó phù hợp với dữ liệu và nếu có những thử nghiệm cụ thể có thể được sử dụng để bác bỏ nó. Điều này rất quan trọng vì tất cả các ước tính thống kê là
*bài hoc*,
tức là, chúng tôi ước tính sau khi chúng tôi quan sát các sự kiện, do đó dễ bị tổn thương bởi sự sai lầm liên quan. Hiện tại, chúng tôi sẽ đặt triết lý sang một bên và dính vào các vấn đề hữu hình hơn. 

Trong phần này, để cung cấp cho bạn một số trực giác, chúng ta sẽ tập trung vào một vài yếu tố có xu hướng ảnh hưởng đến tính tổng quát của một lớp model: 

1. Số lượng các thông số có thể điều chỉnh. Khi số lượng các thông số có thể điều chỉnh, đôi khi được gọi là * độ tự do*, lớn, các mô hình có xu hướng dễ bị quá mức hơn.
1. Các giá trị được thực hiện bởi các tham số. Khi trọng lượng có thể mất một phạm vi rộng hơn các giá trị, các mô hình có thể dễ bị quá mức hơn.
1. Số lượng ví dụ đào tạo. Nó rất dễ dàng để overfit một tập dữ liệu chỉ chứa một hoặc hai ví dụ ngay cả khi mô hình của bạn là đơn giản. Nhưng quá nhiều tập dữ liệu với hàng triệu ví dụ đòi hỏi một mô hình cực kỳ linh hoạt.

## Lựa chọn mô hình

Trong học máy, chúng tôi thường chọn mô hình cuối cùng của mình sau khi đánh giá một số mô hình ứng cử viên. Quá trình này được gọi là *lựa chọn mô hình*. Đôi khi các mô hình có thể so sánh về cơ bản khác nhau về bản chất (nói, cây quyết định so với mô hình tuyến tính). Vào những thời điểm khác, chúng tôi đang so sánh các thành viên của cùng một lớp mô hình đã được đào tạo với các cài đặt siêu tham số khác nhau. 

Ví dụ, với MLP, chúng ta có thể muốn so sánh các mô hình với các số lớp ẩn khác nhau, số lượng đơn vị ẩn khác nhau và các lựa chọn khác nhau của các chức năng kích hoạt được áp dụng cho mỗi lớp ẩn. Để xác định tốt nhất trong số các mô hình ứng viên của chúng tôi, chúng tôi thường sẽ sử dụng một tập dữ liệu xác thực. 

### Validation Dataset

Về nguyên tắc, chúng ta không nên chạm vào bộ thử nghiệm của mình cho đến khi chúng tôi đã chọn tất cả các siêu tham số của mình. Nếu chúng tôi sử dụng dữ liệu thử nghiệm trong quá trình lựa chọn mô hình, có một rủi ro rằng chúng tôi có thể overfit dữ liệu thử nghiệm. Sau đó, chúng tôi sẽ gặp rắc rối nghiêm trọng. Nếu chúng tôi vượt quá dữ liệu đào tạo của mình, luôn có đánh giá về dữ liệu thử nghiệm để giữ cho chúng tôi trung thực. Nhưng nếu chúng ta vượt quá dữ liệu thử nghiệm, làm thế nào chúng ta sẽ biết? 

Do đó, chúng ta không bao giờ nên dựa vào dữ liệu thử nghiệm để lựa chọn mô hình. Tuy nhiên, chúng tôi không thể chỉ dựa vào dữ liệu đào tạo để lựa chọn mô hình vì chúng tôi không thể ước tính lỗi tổng quát hóa trên dữ liệu mà chúng tôi sử dụng để đào tạo mô hình. 

Trong các ứng dụng thực tế, hình ảnh trở nên muddier. Mặc dù lý tưởng nhất là chúng tôi sẽ chỉ chạm vào dữ liệu thử nghiệm một lần, để đánh giá mô hình tốt nhất hoặc so sánh một số lượng nhỏ các mô hình với nhau, dữ liệu thử nghiệm trong thế giới thực hiếm khi bị loại bỏ chỉ sau một lần sử dụng. Chúng ta hiếm khi có thể đủ khả năng một bộ thử nghiệm mới cho mỗi vòng thí nghiệm. 

Thực tiễn phổ biến để giải quyết vấn đề này là chia dữ liệu của chúng tôi ba cách, kết hợp một tập dữ liệu xác thức* (hoặc * bộ xác thực *) ngoài các tập dữ liệu đào tạo và kiểm tra. Kết quả là một thực hành âm u trong đó ranh giới giữa xác nhận và dữ liệu thử nghiệm đáng lo ngại một cách mơ hồ. Trừ khi được nêu rõ ràng khác, trong các thí nghiệm trong cuốn sách này, chúng tôi thực sự đang làm việc với những gì nên được gọi là dữ liệu đào tạo và dữ liệu xác nhận, không có bộ kiểm tra thực sự. Do đó, độ chính xác được báo cáo trong mỗi thí nghiệm của cuốn sách thực sự là độ chính xác xác thực chứ không phải là độ chính xác của bộ thử nghiệm thực sự. 

### $K$-Xác định chéo Fold

Khi dữ liệu đào tạo khan hiếm, chúng ta thậm chí có thể không đủ khả năng để giữ đủ dữ liệu để tạo thành một bộ xác nhận thích hợp. Một giải pháp phổ biến cho vấn đề này là sử dụng $K$* -fold cross-valation*. Tại đây, dữ liệu đào tạo ban đầu được chia thành $K$ các tập con không chồng chéo. Sau đó đào tạo mô hình và xác nhận được thực hiện $K$ lần, mỗi lần đào tạo trên $K-1$ tập con và xác nhận trên một tập con khác (tập hợp không được sử dụng để đào tạo trong vòng đó). Cuối cùng, các lỗi đào tạo và xác nhận được ước tính bằng cách trung bình trên các kết quả từ các thí nghiệm $K$. 

## Underfitting hoặc Overfitting?

Khi chúng tôi so sánh các lỗi đào tạo và xác nhận, chúng tôi muốn lưu ý đến hai tình huống phổ biến. Đầu tiên, chúng tôi muốn xem ra cho các trường hợp khi lỗi đào tạo và lỗi xác nhận của chúng tôi đều đáng kể nhưng có một khoảng cách nhỏ giữa chúng. Nếu mô hình không thể giảm lỗi đào tạo, điều đó có thể có nghĩa là mô hình của chúng tôi quá đơn giản (tức là không đủ biểu cảm) để nắm bắt mô hình mà chúng tôi đang cố gắng mô hình hóa. Hơn nữa, vì khoảng cách tổng quát *giữa các lỗi đào tạo và xác nhận của chúng tôi là nhỏ, chúng tôi có lý do để tin rằng chúng tôi có thể thoát khỏi một mô hình phức tạp hơn. Hiện tượng này được gọi là * underfitting*. 

Mặt khác, như chúng tôi đã thảo luận ở trên, chúng tôi muốn chú ý các trường hợp khi lỗi đào tạo của chúng tôi thấp hơn đáng kể so với lỗi xác thực của chúng tôi, cho thấy nghiêm trọng * overfitting*. Lưu ý rằng overfitting không phải lúc nào cũng là một điều xấu. Đặc biệt với việc học sâu, người ta biết rằng các mô hình dự đoán tốt nhất thường hoạt động tốt hơn nhiều về dữ liệu đào tạo so với dữ liệu lưu trữ. Cuối cùng, chúng tôi thường quan tâm nhiều hơn đến lỗi xác nhận hơn là khoảng cách giữa các lỗi đào tạo và xác nhận. 

Cho dù chúng ta overfit hay underfit có thể phụ thuộc cả vào sự phức tạp của mô hình của chúng tôi và kích thước của các tập dữ liệu đào tạo có sẵn, hai chủ đề mà chúng tôi thảo luận dưới đây. 

### Độ phức tạp của mô hình

Để minh họa một số trực giác cổ điển về độ phức tạp overfitting và mô hình, chúng tôi đưa ra một ví dụ bằng cách sử dụng đa thức. Với dữ liệu đào tạo bao gồm một tính năng duy nhất $x$ và một nhãn có giá trị thực tương ứng $y$, chúng tôi cố gắng tìm đa thức của độ $d$ 

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

to estimate ước tính the labels nhãn $y$. Đây chỉ là một vấn đề hồi quy tuyến tính trong đó các tính năng của chúng tôi được đưa ra bởi sức mạnh của $x$, trọng lượng của mô hình được đưa ra bởi $w_i$ và sự thiên vị được đưa ra bởi $w_0$ kể từ $x^0 = 1$ cho tất cả $x$. Vì đây chỉ là một bài toán hồi quy tuyến tính, chúng ta có thể sử dụng lỗi bình phương làm hàm mất mát của chúng ta. 

Một hàm đa thức bậc cao hơn phức tạp hơn một hàm đa thức bậc thấp hơn, vì đa thức bậc cao hơn có nhiều tham số hơn và phạm vi lựa chọn của hàm mô hình rộng hơn. Sửa tập dữ liệu đào tạo, hàm đa thức bậc cao hơn phải luôn đạt được sai số đào tạo thấp hơn (tồi tệ nhất, bằng nhau) so với đa thức mức độ thấp hơn. Trên thực tế, bất cứ khi nào các ví dụ dữ liệu mỗi ví dụ có giá trị riêng biệt là $x$, một hàm đa thức có mức độ bằng với số ví dụ dữ liệu có thể phù hợp với bộ đào tạo một cách hoàn hảo. Chúng tôi hình dung mối quan hệ giữa mức độ đa thức và underfitting so với overfitting trong :numref:`fig_capacity_vs_error`. 

![Influence of model complexity on underfitting and overfitting](../img/capacity-vs-error.svg)
:label:`fig_capacity_vs_error`

### Kích tập dữ liệu

Sự cân nhắc lớn khác cần lưu ý là kích thước tập dữ liệu. Sửa mô hình của chúng tôi, chúng ta càng có ít mẫu trong tập dữ liệu đào tạo, chúng ta càng gặp phải nhiều khả năng (và nghiêm trọng hơn). Khi chúng tôi tăng lượng dữ liệu đào tạo, lỗi tổng quát thường giảm. Hơn nữa, nói chung, nhiều dữ liệu hơn không bao giờ bị tổn thương. Đối với một tác vụ cố định và phân phối dữ liệu, thường có mối quan hệ giữa độ phức tạp của mô hình và kích thước tập dữ liệu. Với nhiều dữ liệu hơn, chúng tôi có thể có lợi nhuận cố gắng để phù hợp với một mô hình phức tạp hơn. Vắng mặt đủ dữ liệu, các mô hình đơn giản hơn có thể khó đánh bại hơn. Đối với nhiều nhiệm vụ, deep learning chỉ vượt trội hơn các mô hình tuyến tính khi có hàng ngàn ví dụ đào tạo. Một phần, sự thành công hiện tại của deep learning nợ sự phong phú hiện tại của các tập dữ liệu khổng lồ do các công ty Internet, lưu trữ giá rẻ, thiết bị kết nối và số hóa rộng rãi của nền kinh tế. 

## Hồi quy đa thức

Bây giờ chúng ta có thể (** khám phá các khái niệm này tương tác bằng cách lắp đa thức với dữ liệu**)

```{.python .input}
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import numpy as np
import math
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import numpy as np
import math
```

### Tạo tập dữ liệu

Đầu tiên chúng ta cần dữ liệu. Cho $x$, chúng tôi sẽ [** sử dụng đa thức khối sau đây để tạo nhãn**] về dữ liệu đào tạo và thử nghiệm: 

(**$$y = 5 + 1.2x - 3.4\ frac {x^2} {2!} + 5.6\ frac {x^3} {3!} +\ epsilon\ text {where}\ epsilon\ sim\ mathcal {N} (0, 0.1^2) .$$**) 

Thuật ngữ tiếng ồn $\epsilon$ tuân theo sự phân bố bình thường với trung bình 0 và độ lệch chuẩn là 0,1. Để tối ưu hóa, chúng tôi thường muốn tránh các giá trị rất lớn của gradient hoặc tổn thất. Đây là lý do tại sao các tính năng* được thay đổi lại từ $x^i$ thành $\ frac {x^i} {i!} $. Nó cho phép chúng ta tránh các giá trị rất lớn cho số mũ lớn $i$. Chúng tôi sẽ tổng hợp 100 mẫu mỗi mẫu cho bộ đào tạo và bộ kiểm tra.

```{.python .input}
#@tab all
max_degree = 20  # Maximum degree of the polynomial
n_train, n_test = 100, 100  # Training and test dataset sizes
true_w = np.zeros(max_degree)  # Allocate lots of empty space
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # `gamma(n)` = (n-1)!
# Shape of `labels`: (`n_train` + `n_test`,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
```

Một lần nữa, các monomials được lưu trữ trong `poly_features` được rescaled bởi hàm gamma, trong đó $\ Gamma (n) = (n-1)! $. [**Hãy xem 2 mẫu đầu tiên**] từ tập dữ liệu được tạo ra. Giá trị 1 về mặt kỹ thuật là một tính năng, cụ thể là tính năng không đổi tương ứng với sự thiên vị.

```{.python .input}
#@tab pytorch, tensorflow
# Convert from NumPy ndarrays to tensors
true_w, features, poly_features, labels = [d2l.tensor(x, dtype=
    d2l.float32) for x in [true_w, features, poly_features, labels]]
```

```{.python .input}
#@tab all
features[:2], poly_features[:2, :], labels[:2]
```

### Đào tạo và kiểm tra mô hình

Đầu tiên chúng ta hãy [**thực hiện một hàm để đánh giá sự mất mát trên một tập dữ liệu nhất định**].

```{.python .input}
#@tab mxnet, tensorflow
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        l = loss(net(X), y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

```{.python .input}
#@tab pytorch
def evaluate_loss(net, data_iter, loss):  #@save
    """Evaluate the loss of a model on the given dataset."""
    metric = d2l.Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = d2l.reshape(y, out.shape)
        l = loss(out, y)
        metric.add(d2l.reduce_sum(l), d2l.size(l))
    return metric[0] / metric[1]
```

Bây giờ [** xác định chức năng đào tạo**].

```{.python .input}
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
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
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data().asnumpy())
```

```{.python .input}
#@tab pytorch
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1,1)),
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1,1)),
                               batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```

```{.python .input}
#@tab tensorflow
def train(train_features, test_features, train_labels, test_labels,
          num_epochs=400):
    loss = tf.losses.MeanSquaredError()
    input_shape = train_features.shape[-1]
    # Switch off the bias since we already catered for it in the polynomial
    # features
    net = tf.keras.Sequential()
    net.add(tf.keras.layers.Dense(1, use_bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    test_iter = d2l.load_array((test_features, test_labels), batch_size,
                               is_train=False)
    trainer = tf.keras.optimizers.SGD(learning_rate=.01)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        d2l.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net.get_weights()[0].T)
```

### [** Phụ kiện đa thức đa thức thứ ba (Bình thường) **]

Chúng ta sẽ bắt đầu bằng cách sử dụng hàm đa thức bậc ba, có cùng thứ tự như hàm tạo dữ liệu. Kết quả cho thấy việc đào tạo và tổn thất thử nghiệm của mô hình này có thể được giảm một cách hiệu quả. Các tham số mô hình đã học cũng gần với các giá trị thực sự $w = [5, 1.2, -3.4, 5.6]$.

```{.python .input}
#@tab all
# Pick the first four dimensions, i.e., 1, x, x^2/2!, x^3/3! from the
# polynomial features
train(poly_features[:n_train, :4], poly_features[n_train:, :4],
      labels[:n_train], labels[n_train:])
```

### [** Phụ kiện chức năng tuyến tính (Underfitting) **]

Chúng ta hãy xem xét một chức năng tuyến tính phù hợp. Sau sự suy giảm trong thời đại đầu, việc giảm thêm sự mất mát huấn luyện của mô hình này trở nên khó khăn. Sau khi lần lặp kỷ nguyên cuối cùng đã được hoàn thành, tổn thất đào tạo vẫn còn cao. Khi được sử dụng để phù hợp với các mẫu phi tuyến (như hàm đa thức bậc ba ở đây) các mô hình tuyến tính có thể chịu trách nhiệm không phù hợp.

```{.python .input}
#@tab all
# Pick the first two dimensions, i.e., 1, x, from the polynomial features
train(poly_features[:n_train, :2], poly_features[n_train:, :2],
      labels[:n_train], labels[n_train:])
```

### [** Khớp nối đa thức đa thức bậc cao hơn (Overfitting) **]

Bây giờ chúng ta hãy cố gắng đào tạo mô hình bằng cách sử dụng một đa thức của mức độ quá cao. Ở đây, không có đủ dữ liệu để biết rằng các hệ số mức độ cao hơn nên có giá trị gần bằng không. Do đó, mô hình quá phức tạp của chúng tôi rất dễ bị ảnh hưởng bởi tiếng ồn trong dữ liệu đào tạo. Mặc dù tổn thất đào tạo có thể được giảm hiệu quả, nhưng tổn thất thử nghiệm vẫn cao hơn nhiều. Nó cho thấy mô hình phức tạp vượt quá dữ liệu.

```{.python .input}
#@tab all
# Pick all the dimensions from the polynomial features
train(poly_features[:n_train, :], poly_features[n_train:, :],
      labels[:n_train], labels[n_train:], num_epochs=1500)
```

Trong các phần tiếp theo, chúng tôi sẽ tiếp tục thảo luận về các vấn đề và phương pháp quá mức để đối phó với chúng, chẳng hạn như phân rã trọng lượng và bỏ học. 

## Tóm tắt

* Vì lỗi tổng quát hóa không thể được ước tính dựa trên lỗi đào tạo, chỉ cần giảm thiểu lỗi đào tạo sẽ không nhất thiết có nghĩa là giảm lỗi tổng quát hóa. Các mô hình học máy cần phải cẩn thận để bảo vệ chống quá mức để giảm thiểu lỗi tổng quát hóa.
* Một bộ xác thực có thể được sử dụng để lựa chọn mô hình, với điều kiện là nó không được sử dụng quá tự do.
* Underfitting có nghĩa là một mô hình không thể giảm lỗi đào tạo. Khi lỗi đào tạo thấp hơn nhiều so với lỗi xác nhận, có overfitting.
* Chúng ta nên chọn một mô hình phức tạp thích hợp và tránh sử dụng các mẫu đào tạo không đủ.

## Bài tập

1. Bạn có thể giải quyết vấn đề hồi quy đa thức chính xác? Gợi ý: sử dụng đại số tuyến tính.
1. Xem xét lựa chọn mô hình cho đa thức:
    1. Vẽ sự mất mát đào tạo so với độ phức tạp mô hình (mức độ đa thức). Bạn quan sát điều gì? Bạn cần mức độ đa thức nào để giảm tổn thất đào tạo xuống 0?
    1. Vẽ tổn thất thử nghiệm trong trường hợp này.
    1. Tạo ra âm mưu tương tự như một hàm của lượng dữ liệu.
1. Điều gì xảy ra nếu bạn thả bình thường hóa ($1/i! $) of the polynomial features $x^ i $? Bạn có thể khắc phục điều này theo một cách khác không?
1. Bạn có thể bao giờ mong đợi để thấy lỗi tổng quát không?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:

# Chuyển tiếp tuyên truyền, tuyên truyền ngược, và đồ thị tính toán
:label:`sec_backprop`

Cho đến nay, chúng tôi đã đào tạo các mô hình của chúng tôi với dòng dốc ngẫu nhiên minibatch. Tuy nhiên, khi chúng tôi thực hiện thuật toán, chúng tôi chỉ lo lắng về các tính toán liên quan đến *forward propagation* thông qua mô hình. Khi đến lúc tính toán gradient, chúng ta chỉ gọi hàm backpropagation được cung cấp bởi khung học sâu. 

Việc tính toán tự động độ dốc (phân biệt tự động) đơn giản hóa sâu sắc việc thực hiện các thuật toán học sâu. Trước khi phân biệt tự động, ngay cả những thay đổi nhỏ đối với các mô hình phức tạp đòi hỏi phải tính toán lại các dẫn xuất phức tạp bằng tay. Đáng ngạc nhiên thường xuyên, các bài báo học thuật đã phải phân bổ nhiều trang để đưa ra các quy tắc cập nhật. Mặc dù chúng ta phải tiếp tục dựa vào sự khác biệt tự động để chúng ta có thể tập trung vào các phần thú vị, bạn nên biết làm thế nào các gradient được tính toán dưới mui xe nếu bạn muốn vượt ra ngoài một sự hiểu biết nông cạn về học sâu. 

Trong phần này, chúng tôi đi sâu vào các chi tiết của * tuyên truyền ngược * (thường được gọi là *backpropagation*). Để truyền tải một số cái nhìn sâu sắc cho cả kỹ thuật và triển khai của chúng, chúng tôi dựa vào một số toán học cơ bản và đồ thị tính toán. Để bắt đầu, chúng tôi tập trung triển lãm của mình vào MLP một lớp ẩn với sự phân rã trọng lượng ($L_2$ đều đặn). 

## Chuyển tiếp tuyên truyền

*Tuyên truyền chuyển tiếp* (hoặc * chuyển tiếp*) đề cập đến việc tính toán và lưu trữ
của các biến trung gian (bao gồm cả đầu ra) cho một mạng thần kinh theo thứ tự từ lớp đầu vào đến lớp đầu ra. Bây giờ chúng tôi làm việc từng bước thông qua cơ chế của một mạng thần kinh với một lớp ẩn. Điều này có vẻ tẻ nhạt nhưng theo lời vĩnh cửu của nghệ sĩ funk James Brown, bạn phải “trả chi phí để trở thành ông chủ”. 

Vì lợi ích của sự đơn giản, chúng ta hãy giả định rằng ví dụ đầu vào là $\mathbf{x}\in \mathbb{R}^d$ và lớp ẩn của chúng ta không bao gồm một thuật ngữ thiên vị. Ở đây biến trung gian là: 

$$\mathbf{z}= \mathbf{W}^{(1)} \mathbf{x},$$

trong đó $\mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ là tham số trọng lượng của lớp ẩn. Sau khi chạy biến trung gian $\mathbf{z}\in \mathbb{R}^h$ thông qua chức năng kích hoạt $\phi$, chúng tôi có được vector kích hoạt ẩn của chúng tôi có chiều dài $h$, 

$$\mathbf{h}= \phi (\mathbf{z}).$$

Biến ẩn $\mathbf{h}$ cũng là một biến trung gian. Giả sử rằng các tham số của lớp đầu ra chỉ sở hữu trọng lượng $\mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$, ta có thể thu được một biến lớp đầu ra với một vectơ có chiều dài $q$: 

$$\mathbf{o}= \mathbf{W}^{(2)} \mathbf{h}.$$

Giả sử hàm mất là $l$ và nhãn ví dụ là $y$, sau đó chúng ta có thể tính toán thuật ngữ mất mát cho một ví dụ dữ liệu duy nhất, 

$$L = l(\mathbf{o}, y).$$

Theo định nghĩa của $L_2$ chính quy hóa, với siêu tham số $\lambda$, thuật ngữ chính quy hóa là 

$$s = \frac{\lambda}{2} \left(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2\right),$$
:eqlabel:`eq_forward-s`

trong đó định mức Frobenius của ma trận chỉ đơn giản là định mức $L_2$ được áp dụng sau khi làm phẳng ma trận thành một vectơ. Cuối cùng, tổn thất thường xuyên của mô hình trên một ví dụ dữ liệu nhất định là: 

$$J = L + s.$$

Chúng tôi đề cập đến $J$ là chức năng mục tiêu *trong cuộc thảo luận sau đây. 

## Đồ thị tính toán của chuyển tiếp tuyên truyền

Vẽ đồ thị tính toán * giúp chúng ta hình dung các phụ thuộc của toán tử và biến trong phép tính. :numref:`fig_forward` chứa đồ thị liên kết với mạng đơn giản được mô tả ở trên, trong đó các ô vuông biểu thị các biến và vòng tròn biểu thị các toán tử. Góc dưới bên trái biểu thị đầu vào và góc trên bên phải là đầu ra. Lưu ý rằng các hướng của các mũi tên (minh họa luồng dữ liệu) chủ yếu là bên phải và hướng lên. 

![Computational graph of forward propagation.](../img/forward.svg)
:label:`fig_forward`

## Backpropagation

*Backpropagation* đề cập đến phương pháp tính toán
độ dốc của các tham số mạng thần kinh. Nói tóm lại, phương pháp đi qua mạng theo thứ tự ngược lại, từ đầu ra đến lớp đầu vào, theo quy tắc chuỗi *từ tính toán. Thuật toán lưu trữ bất kỳ biến trung gian nào (dẫn xuất từng phần) cần thiết trong khi tính toán gradient đối với một số tham số. Giả sử rằng chúng ta có chức năng $\mathsf{Y}=f(\mathsf{X})$ và $\mathsf{Z}=g(\mathsf{Y})$, trong đó đầu vào và đầu ra $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ là hàng chục hình dạng tùy ý. Bằng cách sử dụng quy tắc chuỗi, chúng ta có thể tính toán đạo hàm của $\mathsf{Z}$ đối với $\mathsf{X}$ qua 

$$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right).$$

Ở đây chúng tôi sử dụng toán tử $\text{prod}$ để nhân các đối số của nó sau các hoạt động cần thiết, chẳng hạn như chuyển vị và trao đổi vị trí đầu vào, đã được thực hiện. Đối với vectơ, điều này rất đơn giản: nó chỉ đơn giản là phép nhân ma trận ma trận. Đối với hàng chục chiều cao hơn, chúng tôi sử dụng đối tác thích hợp. Các nhà điều hành $\text{prod}$ ẩn tất cả các ký hiệu trên cao. 

Nhớ lại rằng các tham số của mạng đơn giản với một lớp ẩn, có biểu đồ tính toán là trong :numref:`fig_forward`, là $\mathbf{W}^{(1)}$ và $\mathbf{W}^{(2)}$. Mục tiêu của sự lan truyền ngược là tính toán độ dốc $\partial J/\partial \mathbf{W}^{(1)}$ và $\partial J/\partial \mathbf{W}^{(2)}$. Để thực hiện điều này, chúng ta áp dụng quy tắc chuỗi và tính toán, lần lượt, gradient của mỗi biến và tham số trung gian. Thứ tự các tính toán được đảo ngược so với các tính toán được thực hiện trong quá trình truyền chuyển tiếp, vì chúng ta cần bắt đầu với kết quả của biểu đồ tính toán và làm việc theo cách của chúng tôi hướng tới các tham số. Bước đầu tiên là tính toán độ dốc của hàm khách quan $J=L+s$ liên quan đến thuật ngữ mất $L$ và thuật ngữ chính quy hóa $s$. 

$$\frac{\partial J}{\partial L} = 1 \; \text{and} \; \frac{\partial J}{\partial s} = 1.$$

Tiếp theo, chúng ta tính toán gradient của hàm mục tiêu đối với biến của lớp đầu ra $\mathbf{o}$ theo quy tắc chuỗi: 

$$
\frac{\partial J}{\partial \mathbf{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \mathbf{o}}\right)
= \frac{\partial L}{\partial \mathbf{o}}
\in \mathbb{R}^q.
$$

Tiếp theo, chúng ta tính toán độ dốc của thuật ngữ chính quy hóa đối với cả hai tham số: 

$$\frac{\partial s}{\partial \mathbf{W}^{(1)}} = \lambda \mathbf{W}^{(1)}
\; \text{and} \;
\frac{\partial s}{\partial \mathbf{W}^{(2)}} = \lambda \mathbf{W}^{(2)}.$$

Bây giờ chúng ta có thể tính toán gradient $\partial J/\partial \mathbf{W}^{(2)} \in \mathbb{R}^{q \times h}$ của các tham số mô hình gần nhất với lớp đầu ra. Sử dụng sản lượng quy tắc chuỗi: 

$$\frac{\partial J}{\partial \mathbf{W}^{(2)}}= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(2)}}\right)= \frac{\partial J}{\partial \mathbf{o}} \mathbf{h}^\top + \lambda \mathbf{W}^{(2)}.$$
:eqlabel:`eq_backprop-J-h`

Để có được gradient đối với $\mathbf{W}^{(1)}$, chúng ta cần tiếp tục lan truyền ngược dọc theo lớp đầu ra đến lớp ẩn. Gradient liên quan đến đầu ra của lớp ẩn $\partial J/\partial \mathbf{h} \in \mathbb{R}^h$ được đưa ra bởi 

$$
\frac{\partial J}{\partial \mathbf{h}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{o}}, \frac{\partial \mathbf{o}}{\partial \mathbf{h}}\right)
= {\mathbf{W}^{(2)}}^\top \frac{\partial J}{\partial \mathbf{o}}.
$$

Vì hàm kích hoạt $\phi$ áp dụng elementwise, tính gradient $\partial J/\partial \mathbf{z} \in \mathbb{R}^h$ của biến trung gian $\mathbf{z}$ yêu cầu chúng ta sử dụng toán tử nhân elementwise, mà chúng ta biểu thị bằng $\odot$: 

$$
\frac{\partial J}{\partial \mathbf{z}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{h}}, \frac{\partial \mathbf{h}}{\partial \mathbf{z}}\right)
= \frac{\partial J}{\partial \mathbf{h}} \odot \phi'\left(\mathbf{z}\right).
$$

Cuối cùng, chúng ta có thể lấy gradient $\partial J/\partial \mathbf{W}^{(1)} \in \mathbb{R}^{h \times d}$ của các tham số mô hình gần nhất với lớp đầu vào. Theo quy tắc chuỗi, chúng tôi nhận được 

$$
\frac{\partial J}{\partial \mathbf{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \mathbf{z}}, \frac{\partial \mathbf{z}}{\partial \mathbf{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \mathbf{W}^{(1)}}\right)
= \frac{\partial J}{\partial \mathbf{z}} \mathbf{x}^\top + \lambda \mathbf{W}^{(1)}.
$$

## Đào tạo mạng nơ-ron

Khi đào tạo mạng thần kinh, sự lan truyền về phía trước và lạc hậu phụ thuộc vào nhau. Đặc biệt, để tuyên truyền chuyển tiếp, chúng ta đi qua đồ thị tính toán theo hướng phụ thuộc và tính toán tất cả các biến trên đường đi của nó. Chúng sau đó được sử dụng để truyền ngược nơi thứ tự tính toán trên đồ thị bị đảo ngược. 

Lấy mạng đơn giản đã nói ở trên làm ví dụ để minh họa. Một mặt, tính toán thuật ngữ chính quy hóa :eqref:`eq_forward-s` trong quá trình truyền chuyển tiếp phụ thuộc vào các giá trị hiện tại của các tham số mô hình $\mathbf{W}^{(1)}$ và $\mathbf{W}^{(2)}$. Chúng được đưa ra bởi thuật toán tối ưu hóa theo backpropagation trong lần lặp mới nhất. Mặt khác, phép tính gradient cho tham số :eqref:`eq_backprop-J-h` trong quá trình truyền ngược phụ thuộc vào giá trị hiện tại của biến ẩn $\mathbf{h}$, được đưa ra bằng cách lan truyền chuyển tiếp. 

Do đó khi đào tạo mạng nơ-ron, sau khi các tham số mô hình được khởi tạo, chúng ta thay thế lan truyền chuyển tiếp với truyền ngược, cập nhật các tham số mô hình bằng cách sử dụng gradient được đưa ra bởi backpropagation. Lưu ý rằng backpropagation sử dụng lại các giá trị trung gian được lưu trữ từ chuyển tiếp tuyên truyền để tránh tính toán trùng lặp. Một trong những hậu quả là chúng ta cần giữ lại các giá trị trung gian cho đến khi truyền ngược hoàn tất. Đây cũng là một trong những lý do tại sao đào tạo đòi hỏi nhiều trí nhớ hơn đáng kể so với dự đoán đơn giản. Bên cạnh đó, kích thước của các giá trị trung gian như vậy là gần tỷ lệ thuận với số lượng lớp mạng và kích thước lô. Do đó, đào tạo các mạng sâu hơn bằng cách sử dụng kích thước lô lớn hơn dễ dàng dẫn đến * ra khỏi bộ nhớ* lỗi. 

## Tóm tắt

* Chuyển tiếp tuyên truyền tuần tự tính toán và lưu trữ các biến trung gian trong đồ thị tính toán được xác định bởi mạng nơ-ron. Nó tiến hành từ đầu vào đến lớp đầu ra.
* Backpropagation tuần tự tính toán và lưu trữ các gradient của các biến và tham số trung gian trong mạng nơ-ron theo thứ tự đảo ngược.
* Khi đào tạo các mô hình học sâu, sự lan truyền về phía trước và lan truyền trở lại phụ thuộc lẫn nhau.
* Đào tạo đòi hỏi nhiều trí nhớ hơn đáng kể so với dự đoán.

## Bài tập

1. Giả sử rằng các đầu vào $\mathbf{X}$ đến một số hàm vô hướng $f$ là ma trận $n \times m$. Chiều của gradient của $f$ đối với $\mathbf{X}$ là gì?
1. Thêm một thiên vị vào lớp ẩn của mô hình được mô tả trong phần này (bạn không cần phải bao gồm thiên vị trong thuật ngữ chính quy hóa).
    1. Vẽ đồ thị tính toán tương ứng.
    1. Lấy được các phương trình tuyên truyền tiến và lạc hậu.
1. Tính toán dấu chân bộ nhớ để đào tạo và dự đoán trong mô hình được mô tả trong phần này.
1. Giả sử rằng bạn muốn tính toán các dẫn xuất thứ hai. Điều gì xảy ra với đồ thị tính toán? Bạn mong đợi tính toán mất bao lâu?
1. Giả sử rằng đồ thị tính toán quá lớn đối với GPU của bạn.
    1. Bạn có thể phân vùng nó trên nhiều GPU không?
    1. Những ưu điểm và nhược điểm so với đào tạo trên một minibatch nhỏ hơn là gì?

[Discussions](https://discuss.d2l.ai/t/102)

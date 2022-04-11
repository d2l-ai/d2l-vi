# Chọn máy chủ và GPU
:label:`sec_buy_gpu`

Đào tạo học sâu thường đòi hỏi một lượng lớn tính toán. Hiện tại GPU là bộ tăng tốc phần cứng hiệu quả nhất cho việc học sâu. Đặc biệt, so với CPU, GPU rẻ hơn và cung cấp hiệu suất cao hơn, thường bằng một thứ tự cường độ. Hơn nữa, một máy chủ duy nhất có thể hỗ trợ nhiều GPU, lên đến 8 cho các máy chủ cao cấp. Các con số điển hình hơn là lên đến 4 GPU cho một máy trạm kỹ thuật, vì yêu cầu về nhiệt, làm mát và năng lượng leo thang nhanh chóng vượt quá những gì một tòa nhà văn phòng có thể hỗ trợ. Đối với các triển khai lớn hơn điện toán đám mây, chẳng hạn như các phiên bản [P3](https://aws.amazon.com/ec2/instance-types/p3/) và [G4](https://aws.amazon.com/blogs/aws/in-the-works-ec2-instances-g4-with-nvidia-t4-gpus/) của Amazon là một giải pháp thiết thực hơn nhiều. 

## Chọn máy chủ

Thường không cần phải mua CPU cao cấp với nhiều luồng vì phần lớn tính toán xảy ra trên GPU. Điều đó nói rằng, do Global Interpreter Lock (GIL) trong hiệu suất một luồng Python của CPU có thể quan trọng trong các tình huống mà chúng ta có 4-8 GPU. Tất cả mọi thứ bằng nhau điều này cho thấy rằng CPU có số lượng lõi nhỏ hơn nhưng tần số đồng hồ cao hơn có thể là một lựa chọn kinh tế hơn. Ví dụ, khi lựa chọn giữa CPU 6 nhân 4 GHz và 8 nhân 3, 5 GHz, trước đây là thích hợp hơn nhiều, mặc dù tốc độ tổng hợp của nó ít hơn. Một cân nhắc quan trọng là GPU sử dụng nhiều năng lượng và do đó tiêu tan rất nhiều nhiệt. Điều này đòi hỏi phải làm mát rất tốt và khung gầm đủ lớn để sử dụng GPU. Thực hiện theo các hướng dẫn dưới đây nếu có thể: 

1. ** Cung cấp điện**. GPU sử dụng một lượng công suất đáng kể. Ngân sách với tối đa 350W cho mỗi thiết bị (kiểm tra * nhu cầu cao nhất* của card đồ họa chứ không phải là nhu cầu điển hình, vì mã hiệu quả có thể sử dụng nhiều năng lượng). Nếu nguồn điện của bạn không theo yêu cầu, bạn sẽ thấy rằng hệ thống của bạn trở nên không ổn định.
1. ** Kích thước khung gầm **. GPU lớn và các đầu nối nguồn phụ thường cần thêm không gian. Ngoài ra, khung gầm lớn dễ dàng hơn để làm mát.
1. ** Làm mát GPU **. Nếu bạn có số lượng lớn GPU, bạn có thể muốn đầu tư vào việc làm mát bằng nước. Ngoài ra, nhằm mục đích thiết kế tham khảo* ngay cả khi chúng có ít quạt hơn, vì chúng đủ mỏng để cho phép hút khí giữa các thiết bị. Nếu bạn mua GPU nhiều quạt, nó có thể quá dày để có đủ không khí khi cài đặt nhiều GPU và bạn sẽ chạy vào điều tiết nhiệt.
1. ** Khe cắm PCIe**. Di chuyển dữ liệu đến và đi từ GPU (và trao đổi nó giữa GPU) đòi hỏi nhiều băng thông. Chúng tôi khuyên bạn nên PCIe 3.0 khe cắm với 16 làn xe. Nếu bạn gắn nhiều GPU, hãy chắc chắn đọc kỹ mô tả bo mạch chủ để đảm bảo rằng băng thông 16x vẫn khả dụng khi nhiều GPU được sử dụng cùng một lúc và bạn đang nhận được PCIe 3.0 trái ngược với PCIe 2.0 cho các khe cắm bổ sung. Một số bo mạch chủ hạ cấp xuống băng thông 8x hoặc thậm chí 4x với nhiều GPU được cài đặt. Điều này một phần là do số lượng làn PCIe mà CPU cung cấp.

Nói tóm lại, dưới đây là một số khuyến nghị để xây dựng một máy chủ học sâu: 

* **Người mới bắt đầu**. Mua GPU cấp thấp với mức tiêu thụ điện năng thấp (GPU chơi game giá rẻ thích hợp cho việc học sâu sử dụng 150-200W). Nếu bạn may mắn máy tính hiện tại của bạn sẽ hỗ trợ nó.
* **1 GPU**. CPU cấp thấp với 4 lõi sẽ đủ và hầu hết các bo mạch chủ đều đủ. Nhắm đến ít nhất 32 GB DRAM và đầu tư vào SSD để truy cập dữ liệu cục bộ. Một nguồn cung cấp điện với 600W nên là đủ. Mua GPU với nhiều người hâm mộ.
* ** 2 GPU **. Một CPU cấp thấp với 4-6 lõi sẽ đủ. Nhắm đến 64 GB DRAM và đầu tư vào ổ SSD. Bạn sẽ cần theo thứ tự 1000W cho hai GPU cao cấp. Về bo mạch chủ, hãy đảm bảo rằng chúng có * hai* PCIe 3.0 x16 khe cắm. Nếu bạn có thể, hãy lấy một bo mạch chủ có hai không gian trống (khoảng cách 60mm) giữa các khe PCIe 3.0 x16 để có thêm không khí. Trong trường hợp này, hãy mua hai GPU với nhiều người hâm mộ.
* **4 GPU **. Hãy chắc chắn rằng bạn mua một CPU với tốc độ đơn luồng tương đối nhanh (tức là tần số đồng hồ cao). Bạn có thể sẽ cần một CPU với số lượng làn PCIe lớn hơn, chẳng hạn như AMD Threadripper. Bạn có thể sẽ cần các bo mạch chủ tương đối đắt tiền để có được 4 khe cắm PCIe 3.0 x16 vì chúng có thể cần một PLX để ghép kênh các làn PCIe. Mua GPU với thiết kế tham chiếu hẹp và để không khí vào giữa các GPU. Bạn cần nguồn điện 1600-2000W và ổ cắm trong văn phòng của bạn có thể không hỗ trợ điều đó. Máy chủ này có thể sẽ chạy * to và nóng*. Bạn không muốn nó dưới bàn làm việc của bạn. 128 GB DRAM được khuyến khích. Nhận SSD (NVMe 1-2 TB) để lưu trữ cục bộ và một loạt các đĩa cứng trong cấu hình RAID để lưu trữ dữ liệu của bạn.
* **8 GPU **. Bạn cần mua khung máy chủ đa GPU chuyên dụng với nhiều bộ nguồn dự phòng (ví dụ: 2+1 cho 1600W cho mỗi nguồn điện). Điều này sẽ yêu cầu CPU máy chủ ổ cắm kép, 256 GB EC DRAM, card mạng nhanh (khuyến nghị 10 GBE) và bạn sẽ cần kiểm tra xem các máy chủ có hỗ trợ yếu tố hình thức vật lý* của GPU hay không. Luồng không khí và vị trí nối dây khác nhau đáng kể giữa GPU tiêu dùng và máy chủ (ví dụ: RTX 2080 so với Tesla V100). Điều này có nghĩa là bạn có thể không thể cài đặt GPU tiêu dùng trong máy chủ do không đủ giải phóng mặt bằng cho cáp nguồn hoặc thiếu dây nịt dây phù hợp (như một trong những đồng tác giả bị phát hiện đau đớn).

## Chọn GPU

Hiện tại, AMD và NVIDIA là hai nhà sản xuất GPU chuyên dụng chính. NVIDIA là người đầu tiên bước vào lĩnh vực học sâu và hỗ trợ tốt hơn cho các khuôn khổ học sâu thông qua CIDA. Do đó, hầu hết người mua chọn GPU NVIDIA. 

NVIDIA cung cấp hai loại GPU, nhắm mục tiêu đến người dùng cá nhân (ví dụ: thông qua dòng GTX và RTX) và người dùng doanh nghiệp (thông qua dòng Tesla của nó). Hai loại GPU cung cấp sức mạnh tính toán tương đương. Tuy nhiên, GPU người dùng doanh nghiệp thường sử dụng làm mát cưỡng bức (thụ động), nhiều bộ nhớ hơn và bộ nhớ EC (sửa lỗi). Các GPU này phù hợp hơn cho các trung tâm dữ liệu và thường tốn gấp mười lần GPU tiêu dùng. 

Nếu bạn là một công ty lớn với hơn 100 máy chủ, bạn nên xem xét dòng NVIDIA Tesla hoặc sử dụng các máy chủ GPU trong đám mây. Đối với một phòng thí nghiệm hoặc một công ty vừa và nhỏ với hơn 10 máy chủ, dòng NVIDIA RTX có thể hiệu quả nhất về chi phí. Bạn có thể mua các máy chủ được cấu hình sẵn với khung Supermicro hoặc Asus chứa 4-8 GPU hiệu quả. 

Các nhà cung cấp GPU thường phát hành một thế hệ mới cứ sau 1-2 năm, chẳng hạn như dòng GTX 1000 (Pascal) được phát hành vào năm 2017 và dòng RTX 2000 (Turing) được phát hành vào năm 2019. Mỗi loạt cung cấp một số mô hình khác nhau cung cấp các mức hiệu suất khác nhau. Hiệu suất GPU chủ yếu là sự kết hợp của ba tham số sau: 

1. ** Điện**. Nói chung chúng ta tìm kiếm công suất tính toán điểm nổi 32 bit. Đào tạo điểm nổi 16 bit (FP16) cũng đang đi vào dòng chính. Nếu bạn chỉ quan tâm đến dự đoán, bạn cũng có thể sử dụng số nguyên 8 bit. Thế hệ GPU Turing mới nhất cung cấp khả năng tăng tốc 4 bit. Thật không may, hiện nay các thuật toán để đào tạo các mạng có độ chính xác thấp vẫn chưa phổ biến.
1. ** Kích thước bộ nhớ**. Khi các mô hình của bạn trở nên lớn hơn hoặc các lô được sử dụng trong quá trình đào tạo phát triển lớn hơn, bạn sẽ cần nhiều bộ nhớ GPU hơn. Kiểm tra bộ nhớ HBM2 (High Bandwidth Memory) so với GDDR6 (Graphics DDR). HBM2 nhanh hơn nhưng đắt hơn nhiều.
1. ** Băng thông bộ nhớ**. Bạn chỉ có thể tận dụng tối đa sức mạnh tính toán của mình khi bạn có đủ băng thông bộ nhớ. Tìm xe buýt bộ nhớ rộng nếu sử dụng GDDR6.

Đối với hầu hết người dùng, nó là đủ để xem xét sức mạnh tính toán. Lưu ý rằng nhiều GPU cung cấp các loại tăng tốc khác nhau. Ví dụ, TensorCores của NVIDIA tăng tốc một tập hợp con của các nhà khai thác bằng 5x. Đảm bảo rằng thư viện của bạn hỗ trợ điều này. Bộ nhớ GPU không được dưới 4 GB (8 GB tốt hơn nhiều). Cố gắng tránh sử dụng GPU cũng để hiển thị GUI (sử dụng đồ họa tích hợp thay thế). Nếu bạn không thể tránh nó, hãy thêm 2 GB RAM để đảm bảo an toàn. 

:numref:`fig_flopsvsprice` so sánh sức mạnh tính toán điểm nổi 32 bit và giá của các mẫu GTX 900, GTX 1000 và RTX 2000 khác nhau. Giá là giá đề xuất được tìm thấy trên Wikipedia. 

![Floating-point compute power and price comparison. ](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`

Chúng ta có thể thấy một số điều: 

1. Trong mỗi loạt, giá cả và hiệu suất gần như tỷ lệ thuận. Các mô hình Titan chỉ huy một khoản phí bảo hiểm đáng kể vì lợi ích của lượng bộ nhớ GPU lớn hơn. Tuy nhiên, các mô hình mới hơn mang lại hiệu quả chi phí tốt hơn, như có thể thấy bằng cách so sánh 980 Ti và 1080 Ti. Giá dường như không cải thiện nhiều cho dòng RTX 2000. Tuy nhiên, điều này là do thực tế là chúng cung cấp hiệu suất chính xác thấp vượt trội hơn nhiều (FP16, INT8 và INT4).
2. Tỷ lệ hiệu suất trên chi phí của dòng GTX 1000 lớn hơn khoảng hai lần so với dòng 900.
3. Đối với dòng RTX 2000, giá là chức năng *affine* của giá.

![Floating-point compute power and energy consumption. ](../img/wattvsprice.svg)
:label:`fig_wattvsprice`

:numref:`fig_wattvsprice` cho thấy mức tiêu thụ năng lượng quy mô chủ yếu là tuyến tính như thế nào với lượng tính toán. Thứ hai, các thế hệ sau này hiệu quả hơn. Điều này dường như bị mâu thuẫn bởi đồ thị tương ứng với dòng RTX 2000. Tuy nhiên, đây là hậu quả của TensorCores thu hút nhiều năng lượng không cân xứng. 

## Tóm tắt

* Coi chừng nguồn điện, làn xe buýt PCIe, tốc độ luồng đơn CPU và làm mát khi xây dựng máy chủ.
* Bạn nên mua thế hệ GPU mới nhất nếu có thể.
* Sử dụng đám mây để triển khai lớn.
* Máy chủ mật độ cao có thể không tương thích với tất cả các GPU. Kiểm tra các thông số kỹ thuật cơ khí và làm mát trước khi bạn mua.
* Sử dụng FP16 hoặc độ chính xác thấp hơn cho hiệu quả cao.

[Discussions](https://discuss.d2l.ai/t/425)

<!--
# Selecting Servers and GPUs
-->

# Lựa chọn Máy chủ & GPU
:label:`sec_buy_gpu`


<!--
Deep learning training generally requires large amounts of computation.
At present GPUs are the most cost-effective hardware accelerators for deep learning.
In particular, compared with CPUs, GPUs are cheaper and offer higher performance, often by over an order of magnitude.
Furthermore, a single server can support multiple GPUs, up to 8 for high end servers.
More typical numbers are up to 4 GPUs for an engineering workstation,
since heat, cooling and power requirements escalate quickly beyond what an office building can support.
For larger deployments cloud computing, such as Amazon's [P3](https://aws.amazon.com/ec2/instance-types/p3/) 
and [G4](https://aws.amazon.com/blogs/aws/in-the-works-ec2-instances-g4-with-nvidia-t4-gpus/) instances are a much more practical solution.
-->


Việc huấn luyện học sâu thông thường đòi hỏi một lượng lớn tài nguyên tính toán.
Ở thời điểm hiện tại, GPU là công cụ tăng tốc phần cứng tiết kiệm chi phí nhất cho việc học sâu.
Cụ thể, so với CPU, GPU rẻ hơn và thường cung cấp hiệu suất cao hơn hàng chục lần.
Hơn nữa, một máy chủ có thể hỗ trợ đa GPU, tới 8 GPU với các máy chủ cao cấp.
Số GPU điển hình là 4 cho một máy trạm kỹ thuật,
vì vấn đề tỏa nhiệt, làm mát và lượng điện tiêu thụ sẽ tăng vọt, vượt quá khả năng một văn phòng có thể cung cấp.
Để triển khai trên số lượng lớn hơn, điện toán đám mây, chẳng hạn như các máy ảo [P3](https://aws.amazon.com/ec2/instance-types/p3/) và [G4](https://aws.amazon.com/blogs/aws/in-the-works-ec2-instances-g4-with-nvidia-t4-gpus/) của Amazon là một giải pháp thực tế hơn nhiều.


<!--
## Selecting Servers
-->

## Lựa chọn Máy chủ


<!--
There is typically no need to purchase high-end CPUs with many threads since much of the computation occurs on the GPUs.
That said, due to the Global Interpreter Lock (GIL) in Python single-thread performance of a CPU can matter in situations where we have 4-8 GPUs.
All things equal this suggests that CPUs with a smaller number of cores but a higher clock frequency might be a more economical choice.
E.g., when choosing between a 6-core 4 GHz and an 8-core 3.5 GHz CPU, the former is much preferable, even though its aggregate speed is less.
An important consideration is that GPUs use lots of power and thus dissipate lots of heat.
This requires very good cooling and a large enough chassis to use the GPUs.
Follow the guidelines below if possible:
-->

Thông thường không cần mua các dòng CPU cao cấp với nhiều luồng vì phần lớn việc tính toán diễn ra trên GPU.
Đồng nghĩa với việc, với Khóa trình thông dịch toàn cục (GIL) trong Python, hiệu suất đơn luồng của CPU có thể là vấn đề trong các tình huống mà chúng ta có 4-8 GPU.
Điều này cho thấy rằng các CPU với số lượng nhân nhỏ hơn nhưng có xung nhịp cao hơn có thể là sự lựa chọn ít tốn kém.
Chẳng hạn khi lựa chọn giữa CPU 6-nhân 4 GHz và 8-nhân 3.5 GHz, thì lựa chọn thứ nhất sẽ được ưu tiên hơn, mặc dù tốc độ kết hợp lại có thể thấp hơn.
Một lưu ý quan trọng là các GPU sử dụng rất nhiều năng lượng và do đó tỏa nhiệt rất nhiều, đòi hỏi khả năng làm mát tốt và một khung máy đủ lớn để sử dụng các GPU đó.
Bạn đọc nên theo sát các nguyên tắc bên dưới đây nếu có thể khi thiết kế máy trạm cho học sâu:


<!--
1. **Power Supply**. GPUs use significant amounts of power.
Budget with up to 350W per device (check for the *peak demand* of the graphics card rather than typical demand, 
since efficient code can use lots of energy).
If your power supply is not up to the demand you will find that your system becomes unstable.
1. **Chassis Size**. GPUs are large and the auxiliary power connectors often need extra space.
Also, large chassis are easier to cool.
1. **GPU Cooling**. If you have large numbers of GPUs you might want to invest in water cooling.
Also, aim for *reference designs* even if they have fewer fans, since they are thin enough to allow for air intake between the devices.
If you buy a multi-fan GPU it might be too thick to get enough air when installing multiple GPUs and you will run into thermal throttling.
1. **PCIe Slots**. Moving data to and from the GPU (and exchanging it between GPUs) requires lots of bandwidth.
We recommend PCIe 3.0 slots with 16 lanes. If you mount multiple GPUs, be sure to carefully read the motherboard description to ensure 
that 16x bandwidth is still available when multiple GPUs are used at the same time and that you are getting PCIe 3.0 as opposed to PCIe 2.0 for the additional slots.
Some motherboards downgrade to 8x or even 4x bandwidth with multiple GPUs installed.
This is partly due to the number of PCIe lanes that the CPU offers.
-->


1. **Bộ Nguồn Cấp Điện**. GPU sử dụng một lượng điện năng đáng kể.
Mỗi GPU có thể cần nguồn cấp lên đến 350W (kiểm tra *công suất đỉnh* của card đồ họa thay vì công suất trung bình, 
vì mã nguồn được tối ưu có thể ngốn nhiều năng lượng).
Nếu nguồn điện của bạn không đáp ứng được nhu cầu, hệ thống sẽ trở nên không ổn định.
2. **Kích thước khung chứa**. GPU có kích thước lớn và các đầu nối nguồn phụ trợ thường cần thêm không gian.
Thêm nữa, khung máy lớn giúp dễ làm mát hơn. 
3. **Làm mát GPU**. Nếu bạn có số lượng lớn GPU, bạn có thể muốn đầu tư hệ thống tản nhiệt nước.
Ngoài ra, có thể sử dụng các *thiết kế tham khảo* ngay cả khi chúng có số quạt làm mát ít hơn, vì chúng đủ mỏng để cho phép thông gió giữa các thiết bị.
Nếu bạn mua một GPU có nhiều quạt, nó có thể quá dày để nhận đủ không khí khi lắp đặt nhiều GPU và bạn sẽ gặp phải tình trạng khó tản nhiệt.
4. **Khe cắm PCIe**. Việc chuyển dữ liệu đến và đi từ GPU (và trao đổi giữa các GPU) đòi hỏi nhiều băng thông.
Chúng tôi đề xuất khe cắm PCIe 3.0 với 16 làn. Nếu bạn lắp nhiều GPU, hãy đảm bảo là bạn đọc kỹ mô tả bo mạch chủ để chắc chắn
băng thông 16x đó vẫn khả dụng khi nhiều GPU được sử dụng cùng lúc và tốc độ PCIe là 3.0 thay vì PCIe cho các khe cắm bổ sung.
Một số bo mạch chủ sẽ hạ xuống băng thông 8x hoặc thậm chí 4x khi nhiều GPU được cài đặt. 
Điều này một phần là do số lượng làn PCIe mà CPU đó cung cấp.


<!--
In short, here are some recommendations for building a deep learning server:
-->

Tóm lại, dưới đây là một số khuyến nghị để bạn xây dựng một máy chủ học sâu: 

<!--
* **Beginner**. Buy a low end GPU with low power consumption (cheap gaming GPUs suitable for deep learning use 150-200W).
If you are lucky your current computer will support it.
* **1 GPU**. A low-end CPU with 4 cores will be plenty sufficient and most motherboards suffice.
 Aim for at least 32 GB DRAM and invest into an SSD for local data access.
 A power supply with 600W should be sufficient. Buy a GPU with lots of fans.
* **2 GPUs**. A low-end CPU with 4-6 cores will suffice. Aim for 64 GB DRAM and invest into an SSD.
You will need in the order of 1000W for two high-end GPUs. In terms of mainboards, make sure that they have *two* PCIe 3.0 x16 slots.
If you can, get a mainboard that has two free spaces (60mm spacing) between the PCIe 3.0 x16 slots for extra air.
In this case, buy two GPUs with lots of fans.
* **4 GPUs**. Make sure that you buy a CPU with relatively fast single-thread speed (i.e., high clock frequency).
You will probably need a CPU with a larger number of PCIe lanes, such as an AMD Threadripper.
You will likely need relatively expensive mainboards to get 4 PCIe 3.0 x16 slots since they probably need a PLX to multiplex the PCIe lanes.
Buy GPUs with reference design that are narrow and let air in between the GPUs.
You need a 1600-2000W power supply and the outlet in your office might not support that.
This server will probably run *loud and hot*. You do not want it under your desk.
128 GB of DRAM is recommended. Get an SSD (1-2 TB NVMe) for local storage and a bunch of hard disks in RAID configuration to store your data.
* **8 GPUs**. You need to buy a dedicated multi-GPU server chassis with multiple redundant power supplies (e.g., 2+1 for 1600W per power supply).
This will require dual socket server CPUs, 256 GB ECC DRAM, a fast network card (10 GBE recommended),
and you will need to check whether the servers support the *physical form factor* of the GPUs.
Airflow and wiring placement differ significantly between consumer and server GPUs (e.g., RTX 2080 vs. Tesla V100).
This means that you might not be able to install the consumer GPU in a server due to insufficient clearance for the power cable 
or lack of a suitable wiring harness (as one of the coauthors painfully discovered).
-->


* **Người mới bắt đầu**. Một GPU cấp thấp với mức tiêu thụ điện năng thấp (GPU dành cho chơi game giá rẻ phù hợp cho việc sử dụng học sâu 150-200W).
Nếu may mắn, máy tính hiện tại của bạn đã có sẵn một GPU như trên.
* **1 GPU**. Một CPU cấp thấp với 4 nhân sẽ là quá đủ và hầu hết các bo mạch chủ đều đáp ứng được.
 Hãy nhắm đến ít nhất 32 GB DRAM và đầu tư một ổ SSD để truy cập dữ liệu cục bộ. 
 Nên sử dụng nguồn cung cấp 600W là đủ. Mua GPU có nhiều quạt.
* **2 GPU**. Một CPU cấp thấp với 4-6 nhân là đủ. Hãy nhắm đến 64 GB DRAM và đầu tư vào một ổ SSD.
Bạn sẽ cần nguồn tầm 1000W cho hai GPU cao cấp. Đối với bo mạch chủ, hãy đảm bảo rằng chúng có *hai* khe cắm PCIe 3.0 x16.
Nếu có thể, hãy mua một bo mạch chủ có hai khoảng trống (khoảng cách 60mm) giữa các khe PCIe 3.0 x16 để có thêm không khí. 
Trong trường hợp này, hãy mua hai GPU có nhiều quạt.
* **4 GPU**. Đảm bảo rằng bạn mua một CPU có tốc độ luồng đơn tương đối nhanh (cụ thể là tần số xung nhịp cao).
Bạn có thể sẽ cần một CPU có số lượng làn PCIe lớn hơn, chẳng hạn như một chip AMD Threadripper.
Bạn có thể sẽ cần bo mạch chủ tương đối đắt tiền để có 4 khe cắm PCIe 3.0 x16 vì chúng có thể cần PLX để ghép kênh các làn PCIe.
Hãy mua GPU có thiết kế tham khảo gốc vì nó hẹp hơn và cho phép không khí lưu thông giữa các GPU.
Bạn cần nguồn điện tầm 1600-2000W và ổ cắm trong văn phòng của bạn có thể không hỗ trợ điều đó.
Máy chủ này có thể sẽ *gây tiếng ồn và tỏa nhiệt* nhiều. Bạn hẳn là không muốn đặt nó dưới bàn làm việc của bạn.
Khuyến nghị sử dụng 128 GB DRAM. Mua một ổ SSD (1-2 TB NVMe) để lưu trữ cục bộ và một số ổ cứng theo cấu hình RAID để lưu trữ dữ liệu của bạn.
* **8 GPU**. Bạn cần mua khung máy chủ đa GPU chuyên dụng với nhiều nguồn điện dự phòng (chẳng hạn, 2 + 1 cho 1600W với mỗi bộ nguồn).
Điều này sẽ yêu cầu CPU máy chủ có khe cắm kép, 256 GB ECC DRAM, một cạc mạng nhanh (khuyến nghị 10 GBE),
và bạn sẽ cần kiểm tra liệu máy chủ có hỗ trợ *hình dạng kích thước vật lý* của GPU hay không.
Luồng không khí và bố trí đi dây có sự khác biệt đáng kể giữa GPU tiêu dùng và GPU máy chủ (cụ thể ở đây là RTX 2080 so với Tesla V100).
Điều này có nghĩa là bạn có thể không lắp đặt được GPU tiêu dùng vào máy chủ do không đủ khoảng trống cho cáp nguồn
hoặc thiếu dây nối phù hợp (như một trong các đồng tác giả đã đau khổ khi phát hiện ra).


<!--
## Selecting GPUs
-->

## Lựa chọn GPU


<!--
At present, AMD and NVIDIA are the two main manufacturers of dedicated GPUs.
NVIDIA was the first to enter the deep learning field and provides better support for deep learning frameworks via CUDA.
Therefore, most buyers choose NVIDIA GPUs.
-->

Hiện nay, AMD và NVIDIA là hai nhà sản xuất GPU chính.
NVIDIA là tiên phong trong tham gia lĩnh vực học sâu và cung cấp hỗ trợ tốt hơn cho các framework học sâu thông qua CUDA. 
Do đó, phần lớn người mua chọn GPU của NVIDIA.


<!--
NVIDIA provides two types of GPUs, targeting individual users (e.g., via the GTX and RTX series) and enterprise users (via its Tesla series).
The two types of GPUs provide comparable compute power.
However, the enterprise user GPUs generally use (passive) forced cooling, more memory, and ECC (error correcting) memory.
These GPUs are more suitable for data centers and usually cost ten times more than consumer GPUs.
-->

NVIDIA cung cấp hai loại GPU, nhắm tới người dùng cá nhân (ví dụ như dòng GTX và RTX) và người dùng doanh nghiệp (thông qua dòng Tesla của họ).
Hai loại GPU cung cấp khả năng tính toán tương đương nhau. 
Tuy nhiên, GPU dành cho người dùng doanh nghiệp thường sử dụng tản nhiệt cưỡng ép (thụ động), nhiều bộ nhớ, và bộ nhớ ECC (sửa sai - *error correcting*).
Những GPU này phù hợp hơn cho trung tâm dữ liệu và thường có giá cao hơn 10 lần so với GPU tiêu dùng.


<!--
If you are a large company with 100+ servers you should consider the NVIDIA Tesla series or alternatively use GPU servers in the cloud.
For a lab or a small to medium company with 10+ servers the NVIDIA RTX series is likely most cost effective.
You can buy preconfigured servers with Supermicro or Asus chassis that hold 4-8 GPUs efficiently.
-->

Nếu bạn là một công ty lớn với 100+ máy chủ, bạn nên cân nhắc dòng NVIDIA Tesla hoặc thay thế bằng cách sử dụng máy chủ GPU trên đám mây.
Với các phòng nghiên cứu hay một công ty tầm trung với 10+ máy chủ, dòng NVIDIA RTX có lẽ sẽ có hiệu quả chi phí tốt nhất. 
Bạn có thể mua máy chủ cấu hình sẵn với khung chứa Supermicro hay Asus, có thể chứa hiệu quả 4-8 GPU.


<!--
GPU vendors typically release a new generation every 1-2 years,
such as the GTX 1000 (Pascal) series released in 2017 and the RTX 2000 (Turing) series released in 2019.
Each series offers several different models that provide different performance levels.
GPU performance is primarily a combination of the following three parameters:
-->

Nhà cung cấp GPU thường ra mắt thế hệ mới mỗi 1-2 năm, 
ví dụ như dòng GTX 1000 (Pascal) ra mắt vào 2017 và dòng RTX 2000 (Turing) ra mắt vào 2019.
Mỗi dòng gồm có nhiều mẫu khác nhau cung cấp mức hiệu năng khác nhau. 
Hiệu năng GPU chủ yếu là sự kết hợp của ba thông số sau:


<!--
1. **Compute power**. Generally we look for 32-bit floating-point compute power.
16-bit floating point training (FP16) is also entering the mainstream.
If you are only interested in prediction, you can also use 8-bit integer.
The latest generation of Turing GPUs offers 4-bit acceleration.
Unfortunately at present the algorithms to train low-precision networks are not widespread yet.
1. **Memory size**. As your models become larger or the batches used during training grow bigger, you will need more GPU memory.
Check for HBM2 (High Bandwidth Memory) vs. GDDR6 (Graphics DDR) memory. HBM2 is faster but much more expensive.
1. **Memory bandwidth**. You can only get the most out of your compute power when you have sufficient memory bandwidth.
Look for wide memory buses if using GDDR6.
-->

1. **Khả năng tính toán**. Thông thường ta quan tâm đến khả năng tính toán dấu phẩy động 32-bit (*32-bit floating-point*).
Huấn luyện mô hình sử dụng dấu phẩy động 16-bit (FP16 - *16-bit floating point*) cũng đang dần phổ biến.
Nếu chỉ quan tâm đến tác vụ dự đoán, bạn cũng có thể sử dụng số nguyên 8-bit (*8-bit integer*).
Thế hệ mới nhất của GPU Turing còn cung cấp chế độ tăng tốc 4-bit (*4-bit acceleration*).
Không may là hiện nay, các thuật toán huấn luyện với số thực độ chính xác thấp vẫn chưa được phổ biến.
1. **Kích thước bộ nhớ**. Khi các mô hình của bạn trở nên lớn hơn hay khi tăng kích thước batch khi huấn luyện, bạn sẽ cần nhiều bộ nhớ GPU hơn.
Hãy kiểm tra HBM2 (Bộ nhớ Băng thông cao - *High Bandwidth Memory*) và GDDR6 (DDR Đồ hoạ - *Graphics DDR*). HBM2 nhanh hơn nhưng đắt hơn nhiều.
3. **Băng thông bộ nhớ**. Bạn chỉ có thể tận dụng tối đa khả năng tính toán nếu bạn có đủ băng thông bộ nhớ.
Hãy chọn bus bộ nhớ rộng nếu sử dụng GDDR6.


<!--
For most users, it is enough to look at compute power.
Note that many GPUs offer different types of acceleration.
E.g., NVIDIA's TensorCores accelerate a subset of operators by 5x.
Ensure that your libraries support this. The GPU memory should be no less than 4 GB (8 GB is much better).
Try to avoid using the GPU also for displaying a GUI (use the built-in graphics instead).
If you cannot avoid it, add an extra 2 GB of RAM for safety.
-->

Với phần lớn người dùng, tập trung vào khả năng tính toán là đủ.
Chú ý rằng các GPU khác nhau cung cấp các cách tăng tốc khác nhau, 
ví dụ như TensorCores của NVIDIA tăng tốc một tập con các toán tử lên tới gấp 5 lần.
Vậy nên hãy đảm bảo rằng thư viện của bạn hỗ trợ việc này. Bộ nhớ GPU thì không nên ít hơn 4 GB (8 GB thì hơn). 
Hãy cố gắng tránh sử dụng GPU để hiện thị giao diện đồ họa người dùng (GUI), thay vào đó nếu cần hãy sử dụng card đồ hoạ tích hợp sẵn trong máy.
Nếu bắt buộc phải dùng GPU để hiển thị GUI, hãy thêm vào 2 GB RAM cho an toàn.


<!--
:numref:`fig_flopsvsprice` compares the 32-bit floating-point compute power and price of the various GTX 900, GTX 1000 and RTX 2000 series models.
The prices are the suggested prices found on Wikipedia.
-->

:numref:`fig_flopsvsprice` so sánh khả năng tính toán dấu phẩy động 32-bit và giá của các mẫu khác nhau của các dòng GTX 900, GTX 1000 và RTX 2000.
Đây là bảng giá đề xuất có thể được tìm thấy trên Wikipedia.


<!--
![Floating-point compute power and price comparison.](../img/flopsvsprice.svg)
-->

![So sánh khả năng tính toán dấu phẩy động và giá.](../img/flopsvsprice.svg)
:label:`fig_flopsvsprice`


<!--
We can see a number of things:
-->

Bạn có thể thấy một số điểm sau: 


<!--
1. Within each series, price and performance are roughly proportional.
Titan models command a significant premium for the benefit of larger amounts of GPU memory.
However, the newer models offer better cost effectiveness, as can be seen by comparing the 980 Ti and 1080 Ti.
The price does not appear to improve much for the RTX 2000 series.
However, this is due to the fact that they offer far superior low precision performance (FP16, INT8 and INT4).
2. The performance-to-cost ratio of the GTX 1000 series is about two times greater than the 900 series.
3. For the RTX 2000 series the price is an *affine* function of the price.
-->

1. Trong cùng một dòng, giá và hiệu năng gần như tỷ lệ với nhau.
Mẫu Titan yêu cầu một khoản tiền đáng kể để đổi lấy lợi ích của lượng lớn bộ nhớ GPU. 
Tuy nhiên, những mẫu mới hơn cung cấp hiệu quả chi phí tốt hơn, như có thể thấy qua so sánh giữa 980 Ti và 1080 Ti.
Giá dường như không cải thiện nhiều đối với dòng RTX 2000. 
Tuy nhiên, việc này là do chúng cung cấp hiệu năng hoàn toàn vượt trội đối với các giá trị có độ chính xác thấp (FP16, INT8 và INT4).
2. tỷ lệ hiệu năng trên giá của dòng GTX 1000 lớn hơn khoảng 2 lần so với dòng 900. 
3. Với dòng RTX 2000, giá là một hàm *affine* của hiệu năng.


<!--
![Floating-point compute power and energy consumption.](../img/wattvsprice.svg)
-->

![Khả năng tính toán dấu phẩy động và năng lượng tiêu hao.](../img/wattvsprice.svg)
:label:`fig_wattvsprice`


<!--
:numref:`fig_wattvsprice` shows how energy consumption scales mostly linearly with the amount of computation.
Second, later generations are more efficient.
This seems to be contradicted by the graph corresponding to the RTX 2000 series.
However, this is a consequence of the TensorCores which draw disproportionately much energy.
-->

:numref:`fig_wattvsprice` chỉ ra lượng năng lượng tiêu hao chủ yếu tỷ lệ tuyến tính với khối lượng tính toán.
Thứ hai, các thế hệ sau có hiệu quả tốt hơn.
Đồ thị của dòng RTX 2000 có vẻ như mâu thuẫn với điều này.
Tuy nhiên, đây là hệ quả của TensorCore yêu cầu năng lượng rất lớn.


## Tóm tắt

<!--
* Watch out for power, PCIe bus lanes, CPU single thread speed and cooling when building a server.
* You should purchase the latest GPU generation if possible.
* Use the cloud for large deployments.
* High density servers may not be compatible with all GPUs.
Check the mechanical and cooling specifications before you buy.
* Use FP16 or lower precision for high efficiency.
-->

* Chú ý nguồn, luồng bus PCIe, tốc độ CPU đơn luồng và tản nhiệt khi xây dựng máy chủ.
* Bạn nên mua thế hệ GPU mới nhất nếu có thể.
* Sử dụng đám mây để triển khai các dự án lớn. 
* Máy chủ chạy nhiều ứng dụng có thể sẽ không tương thích với tất cả các GPU.
Kiểm tra các thông số cơ học và tản nhiệt trước khi mua. 
* Sử dụng FP16 hoặc độ chính xác thấp hơn để có được hiệu năng tốt hơn.


## Thảo luận
* Tiếng Anh: [Main Forum](https://discuss.d2l.ai/t/425)
* Tiếng Việt: [Diễn đàn Machine Learning Cơ Bản](https://forum.machinelearningcoban.com/c/d2l)


## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Đỗ Trường Giang
* Nguyễn Văn Cường
* Lê Khắc Hồng Phúc
* Phạm Hồng Vinh
* Phạm Minh Đức
* Nguyễn Văn Quang
* Nguyễn Mai Hoàng Long

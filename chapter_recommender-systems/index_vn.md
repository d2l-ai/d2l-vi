<!--
# Recommender Systems
-->

# Hệ thống Đề xuất
:label:`chap_recsys`


**Shuai Zhang** (*Amazon*), **Aston Zhang** (*Amazon*), và **Yi Tay** (*Google*).

<!--
Recommender systems are widely employed in industry and are ubiquitous in our daily lives. 
These systems are utilized in a number of areas such as online shopping sites (e.g., amazon.com), 
music/movie services site (e.g., Netflix and Spotify), mobile application stores (e.g., IOS app store and google play), 
online advertising, just to name a few.
-->

Hệ thống đề xuất được sử dụng một cách rộng rãi trong kinh doanh và luôn hiện diện trong cuộc sống hàng ngày của chúng ta.
Những hệ thống này được tận dụng trong nhiều lĩnh vực như thương mại điện tử (như amazon.com), các dịch vụ âm nhạc / điện ảnh (như Netflix và Spotify), cửa hàng ứng dụng di động (như App Store và Google Play), quảng cáo trực tuyến, v.v.


<!--
The major goal of recommender systems is to help users discover relevant items such as movies to watch, 
text to read or products to buy, so as to create a delightful user experience. 
Moreover, recommender systems are among the most powerful machine learning systems that online retailers implement in order to drive incremental revenue. 
Recommender systems are replacements of search engines by reducing the efforts in proactive searches and surprising users with offers they never searched for. 
Many companies managed to position themselves ahead of their competitors with the help of more effective recommender systems. 
As such, recommender systems are central to not only our everyday lives but also highly indispensable in some industries.
-->

Mục đích chính của các hệ thống đề xuất là giúp người dùng tìm ra những sản phẩm liên quan như phim để xem,
văn bản để đọc hay hàng hóa để mua, nhằm tạo nên một trải nghiệm thú vị cho người dùng.
Hơn nữa, hệ thống đề xuất là một trong những hệ thống máy học mạnh mẽ nhất mà các công ty bán lẻ áp dụng với mục đích tăng doanh thu.
Hệ thống đề xuất là công cụ thay thế cho các công cụ tìm kiếm bằng cách giảm nỗ lực tìm kiếm chủ động và tăng cơ hội tiếp cận của người dùng với những đề xuất mà họ không bao giờ tìm đến.
Rất nhiều công ty đã vượt lên trên các đối thủ nhờ có hệ thống đề xuất hiệu quả hơn.
Do đó, hệ thống đề xuất đã trở thành trung tâm không chỉ trong cuộc sống hàng ngày của chúng ta mà còn có vai trò quan trọng trong một số lĩnh vực kinh doanh.


<!--
In this chapter, we will cover the fundamentals and advancements of recommender systems, 
along with exploring some common fundamental techniques for building recommender systems with different data sources available and their implementations. 
Specifically, you will learn how to predict the rating a user might give to a prospective item, 
how to generate a recommendation list of items and how to predict the click-through rate from abundant features. 
These tasks are commonplace in real-world applications. 
By studying this chapter, you will get hands-on experience pertaining to solving real world 
recommendation problems with not only classical methods but the more advanced deep learning based models as well.
-->

Trong chương này, chúng tôi sẽ giới thiệu những nội dung cơ bản và những tiến bộ của hệ thống đề xuất,
cùng với việc khám phá một số kỹ thuật cơ bản để xây dựng hệ thống đề xuất với các nguồn dữ liệu có sẵn khác nhau và cách lập trình những kỹ thuật này.
Cụ thể, bạn sẽ học được cách để dự đoán mức đánh giá mà một người dùng sẽ đánh giá một sản phẩm,
cách để tạo ra danh sách các sản phẩm đề xuất và cách dự đoán tỷ lệ nhấp chuột (*click-through rate*) từ một lượng lớn đặc trưng.
Những tác vụ này vô cùng phổ biến trong các ứng dụng thực tế.
Thông qua việc học chương này, bạn sẽ có được trải nghiệm thực tiễn để giải các bài toán đề xuất thực tế
không chỉ với những phương pháp cổ điển mà còn là những mô hình tiên tiến hơn dựa trên học sâu.


```toc
:maxdepth: 2

recsys-intro_vn
movielens_vn
mf_vn
autorec_vn
ranking_vn
neumf_vn
seqrec_vn
ctr_vn
fm_vn
deepfm_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Lê Quang Nhật
* Đỗ Trường Giang
* Nguyễn Văn Cường

*Cập nhật lần cuối: 26/09/2020. (Cập nhật lần cuối từ nội dung gốc: 25/04/2020)*

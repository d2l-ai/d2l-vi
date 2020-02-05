<!-- ===================== Bắt đầu dịch Phần 1 ==================== -->

<!--
# Computational Performance
-->

# Hiệu năng Tính toán
:label:`chap_performance`

<!--
In deep learning, datasets are usually large and model computation is complex. Therefore, we are always very concerned about computing performance. This chapter will focus on the important factors that affect computing performance: imperative programming, symbolic programming, asynchronous programing, automatic parallel computation, and multi-GPU computation. By studying this chapter, you should be able to further improve the computing performance of the models that have been implemented in the previous chapters, for example, by reducing the model training time without affecting the accuracy of the model.
-->

Trong học sâu, các tập dữ liệu thường rất lớn và mô hình tính toán rất phức tạp.
Do đó, ta luôn cần quan tâm tới vấn đề hiệu năng tính toán.
Chương này sẽ tập trung vào các yếu tố quan trọng ảnh hưởng tới hiệu năng tính toán: lập trình mệnh lệnh, lập trình ký hiệu, lập trình bất đồng bộ, tính toán song song tự động và tính toán đa GPU.
Qua chương này, bạn đọc có thể cải thiện hơn nữa hiệu năng tính toán của mô hình đã được triển khai trong các chương trước, ví dụ, giảm thời gian huấn luyện mô hình mà không ảnh hưởng tới độ chính xác.

```toc
:maxdepth: 2

hybridize
async-computation
auto-parallelism
hardware
multiple-gpus
multiple-gpus-gluon
parameterserver
```

<!-- ===================== Kết thúc dịch Phần 1 ==================== -->

### Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.
-->

* Đoàn Võ Duy Thanh
<!-- Phần 1 -->
* Nguyễn Văn Tâm

<!--
# Computational Performance
-->

# Hiệu năng Tính toán
:label:`chap_performance`

<!--
In deep learning, datasets are usually large and model computation is complex.
Therefore, we are always very concerned about computing performance.
This chapter will focus on the important factors that affect computing performance: 
imperative programming, symbolic programming, asynchronous programing, automatic parallel computation, and multi-GPU computation.
By studying this chapter, you should be able to further improve the computing performance of the models that have been implemented in the previous chapters, for example, by reducing the model training time without affecting the accuracy of the model.
-->

Trong học sâu, các tập dữ liệu thường rất lớn và mô hình tính toán rất phức tạp.
Vì vậy, ta luôn cần quan tâm tới vấn đề hiệu năng tính toán.
Trong chương này, ta sẽ tập trung vào các yếu tố then chốt ảnh hưởng đến hiệu năng tính toán: lập trình mệnh lệnh, lập trình ký hiệu, lập trình bất đồng bộ, tính toán song song tự động và tính toán đa GPU.
Qua đó, độc giả có thể cải thiện nhiều hơn về hiệu năng tính toán của mô hình đã được triển khai trong các chương trước, như là giảm thời gian huấn luyện mà không ảnh hưởng tới độ chính xác của mô hình.

```toc
:maxdepth: 2

hybridize_vn
async-computation_vn
auto-parallelism_vn
hardware_vn
multiple-gpus_vn
multiple-gpus-concise_vn
parameterserver_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* Nguyễn Văn Tâm

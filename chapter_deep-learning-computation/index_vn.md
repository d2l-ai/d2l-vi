<!-- ===================== Bắt đầu dịch ==================== -->

<!--
# Deep Learning Computation
-->

# Tính toán Học sâu
:label:`chap_computation`

<!--
Alongside giant datasets and powerful hardware, great software tools have played an indispensable role in the rapid progress of deep learning.
Starting with the pathbreaking Theano library released in 2007, flexible open-source tools have enabled researchers to rapidly prototype models 
avoiding repetitive work when recycling standard components while still maintaining the ability to make low-level modifications.
Over time, deep learning's libraries have evolved to offer increasingly coarse abstractions.
Just as semiconductor designers went from specifying transistors to logical circuits to writing code, 
neural networks researchers have moved from thinking about the behavior of individual artificial neurons to conceiving of networks in terms of whole layers, 
and now often design architectures with far coarser *blocks* in mind.
-->

*dịch đoạn phía trên*


<!--
So far, we have introduced some basic machine learning concepts, ramping up to fully-functional deep learning models.
In the last chapter, we implemented each component of a multilayer perceptron from scratch and even showed how to leverage MXNet's Gluon library to roll out the same models effortlessly.
To get you that far that fast, we *called upon* the libraries, but skipped over more advanced details about *how they work*.
In this chapter, we will peel back the curtain, digging deeper into the key components of deep learning computation, 
namely model construction, parameter access and initialization, designing custom layers and blocks, reading and writing models to disk, and leveraging GPUs to achieve dramatic speedups.
These insights will move you from *end user* to *power user*, giving you the tools needed to combine the reap the benefits of a mature deep learning library, 
while retaining the flexibility to implement more complex models, including those you invent yourself!
While this chapter does not introduce any new models or datasets, the advanced modeling chapters that follow rely heavily on these techniques.
-->

*dịch đoạn phía trên*

```toc
:maxdepth: 2

model-construction_vn
parameters_vn
deferred-init_vn
custom-layer_vn
read-write_vn
use-gpu_vn
```

<!-- ===================== Kết thúc dịch ==================== -->

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:
<!--
Tác giả của mỗi Pull Request điền tên mình và tên những người review mà bạn thấy
hữu ích vào từng phần tương ứng. Mỗi dòng một tên, bắt đầu bằng dấu `*`.

Lưu ý:
* Nếu reviewer không cung cấp tên, bạn có thể dùng tên tài khoản GitHub của họ
với dấu `@` ở đầu. Ví dụ: @aivivn.

* Tên đầy đủ của các reviewer có thể được tìm thấy tại https://github.com/aivivn/d2l-vn/blob/master/docs/contributors_info.md
-->

* Đoàn Võ Duy Thanh
* 
<!--
# Optimization Algorithms
-->

# Thuật toán Tối ưu
:label:`chap_optimization`

<!--
If you read the book in sequence up to this point you already used a number of advanced optimization algorithms to train deep learning models.
They were the tools that allowed us to continue updating model parameters and to minimize the value of the loss function, as evaluated on the training set.
Indeed, anyone content with treating optimization as a black box device to minimize objective functions in a simple setting might well content oneself 
with the knowledge that there exists an array of incantations of such a procedure (with names such as "Adam", "NAG", or "SGD").
-->

*dịch đoạn phía trên*

<!--
To do well, however, some deeper knowledge is required.
Optimization algorithms are important for deep learning.
On one hand, training a complex deep learning model can take hours, days, or even weeks.
The performance of the optimization algorithm directly affects the model's training efficiency.
On the other hand, understanding the principles of different optimization algorithms and the role of their parameters will enable us 
to tune the hyperparameters in a targeted manner to improve the performance of deep learning models.
-->

*dịch đoạn phía trên*

<!--
In this chapter, we explore common deep learning optimization algorithms in depth.
Almost all optimization problems arising in deep learning are *nonconvex*.
Nonetheless, the design and analysis of algorithms in the context of convex problems has proven to be very instructive.
It is for that reason that this section includes a primer on convex optimization and the proof for a very simple stochastic gradient descent algorithm on a convex objective function.
-->

*dịch đoạn phía trên*

```toc
:maxdepth: 2

optimization-intro_vn
convexity_vn
gd_vn
sgd_vn
minibatch-sgd_vn
momentum_vn
adagrad_vn
rmsprop_vn
adadelta_vn
adam_vn
lr-scheduler_vn
```

## Những người thực hiện
Bản dịch trong trang này được thực hiện bởi:

* Đoàn Võ Duy Thanh
* 
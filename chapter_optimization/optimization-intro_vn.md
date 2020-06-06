<!-- ===================== Bắt đầu dịch Phần  ==================== -->
<!-- ========================================= REVISE PHẦN  - BẮT ĐẦU =================================== -->

<!--
# Optimization and Deep Learning
-->

# *dịch tiêu đề phía trên*

<!--
In this section, we will discuss the relationship between optimization and deep learning as well as the challenges of using optimization in deep learning. For a deep learning problem, we will usually define a loss function first. Once we have the loss function, we can use an optimization algorithm in attempt to minimize the loss. In optimization, a loss function is often referred to as the objective function of the optimization problem. By tradition and convention most optimization algorithms are concerned with *minimization*. If we ever need to maximize an objective there is a simple solution: just flip the sign on the objective.
-->

*dịch đoạn phía trên*

<!--
## Optimization and Estimation
-->

## *dịch tiêu đề phía trên*

<!--
Although optimization provides a way to minimize the loss function for deep
learning, in essence, the goals of optimization and deep learning are
fundamentally different. The former is primarily concerned with minimizing an
objective whereas the latter is concerned with finding a suitable model, given a
finite amount of data.  In :numref:`sec_model_selection`,
we discussed the difference between these two goals in detail. For instance,
training error and generalization error generally differ: since the objective
function of the optimization algorithm is usually a loss function based on the
training dataset, the goal of optimization is to reduce the training error.
However, the goal of statistical inference (and thus of deep learning) is to
reduce the generalization error.  To accomplish the latter we need to pay
attention to overfitting in addition to using the optimization algorithm to
reduce the training error. We begin by importing a few libraries with a function to annotate in a figure.
-->

*dịch đoạn phía trên*

```{.python .input  n=1}
%matplotlib inline
from d2l import mxnet as d2l
from mpl_toolkits import mplot3d
from mxnet import np, npx
npx.set_np()


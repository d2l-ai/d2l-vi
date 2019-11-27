# Optimization Algorithms
<<<<<<< HEAD

If you have read this book in order to this point, then you have already used optimization algorithms to train deep learning models. Specifically, when training models, we use optimization algorithms to continue updating the model parameters to reduce the value of the model loss function. When iteration ends, model training ends along with it. The model parameters we get here are the parameters that the model learned through training.
=======
:label:`chap_optimization`

If you read the book in sequence up to this point you already used a number of advanced optimization algorithms to train deep learning models. They were the tools that allowed us to continue updating model parameters and to minimize the value of the loss function, as evaluated on the training set. Indeed, anyone content with treating optimization as a black box device to minimize objective functions in a simple setting might well content oneself with the knowledge that there exists an array of incantations of such a procedure (with names such as "Adam", "NAG", or "SGD").
>>>>>>> 1ec5c63... copy from d2l-en (#16)

Optimization algorithms are important for deep learning. On the one hand, training a complex deep learning model can take hours, days, or even weeks. The performance of the optimization algorithm directly affects the model's training efficiency. On the other hand, understanding the principles of different optimization algorithms and the meanings of their hyperparameters will enable us to tune the hyperparameters in a targeted manner to improve the performance of deep learning models.

In this chapter, we explore common deep learning optimization algorithms in depth.

```eval_rst

<<<<<<< HEAD
.. toctree::
   :maxdepth: 2

   optimization-intro
   gd-sgd
   minibatch-sgd
   momentum
   adagrad
   rmsprop
   adadelta
   adam
=======
optimization-intro
convexity
gd
sgd
minibatch-sgd
momentum
adagrad
rmsprop
adadelta
adam
lr-scheduler
>>>>>>> 1ec5c63... copy from d2l-en (#16)
```


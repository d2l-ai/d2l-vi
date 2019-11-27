# Deep Learning Computation
<<<<<<< HEAD

The previous chapter introduced the principles and implementation for a simple deep learning model, including multi-layer perceptrons. In this chapter we will cover various key components of deep learning computation, such as model construction, parameter access and initialization, custom layers, and reading, storing, and using GPUs. Throughout this chapter, you will gain important insights into model implementation and computation details, which gives readers a solid foundation for implementing more complex models in the following chapters. 

```eval_rst
=======
:label:`chap_computation`

Alongside giant datasets and powerful hardware,
great software tools have played an indispensable role
in the rapid progress of deep learning.
Starting with the pathbreaking Theano library released in 2007,
flexible open-source tools have enabled researchers
to rapidly prototype models avoiding repetitive work
when recycling standard components
while still maintaining the ability to make low-level modifications.
Over time, deep learning's libraries have evolved
to offer increasingly coarse abstractions.
Just as semiconductor designers went from specifying transistors
to logical circuits to writing code,
neural networks researchers have moved from thinking about
the behavior of individual artificial neurons
to conceiving of networks in terms of whole layers,
and now often design architectures with far coarser *blocks* in mind.
>>>>>>> 1ec5c63... copy from d2l-en (#16)

.. toctree::
   :maxdepth: 2

<<<<<<< HEAD
   model-construction
   parameters
   deferred-init
   custom-layer
   read-write
   use-gpu
=======
So far, we have introduced some basic machine learning concepts,
ramping up to fully-functional deep learning models.
In the last chapter, we implemented each component of a multilayer perceptron from scratch and even showed how to leverage MXNet's Gluon library
to roll out the same models effortlessly.
To get you that far that fast, we *called upon* the libraries,
but skipped over more advanced details about *how they work*.
In this chapter, we will peel back the curtain,
digging deeper into the key components of deep learning computation,
namely model construction, parameter access and initialization,
designing custom layers and blocks, reading and writing models to disk,
and leveraging GPUs to achieve dramatic speedups.
These insights will move you from *end user* to *power user*,
giving you the tools needed to combine the reap the benefits
of a mature deep learning library, while retaining the flexibility
to implement more complex models, including those you invent yourself!
While this chapter does not introduce any new models or datasets,
the advanced modeling chapters that follow rely heavily on these techniques.
>>>>>>> 1ec5c63... copy from d2l-en (#16)

```

# Deep Recurrent Neural Networks

<<<<<<< HEAD:chapter_recurrent-neural-networks/deep-rnn.md
Up to now, we have only discussed recurrent neural networks with a single unidirectional hidden layer. In deep learning applications, we generally use recurrent neural networks that contain multiple hidden layers. These are also called deep recurrent neural networks. Figure 6.11 demonstrates a deep recurrent neural network with $L$ hidden layers. Each hidden state is continuously passed to the next time step of the current layer and the next layer of the current time step.
=======
:label:`sec_deep_rnn`

Up to now, we only discussed recurrent neural networks with a single unidirectional hidden layer. In it the specific functional form of how latent variables and observations interact was rather arbitrary. This is not a big problem as long as we have enough flexibility to model different types of interactions. With a single layer, however, this can be quite challenging. In the case of the perceptron, we fixed this problem by adding more layers. Within RNNs this is a bit more tricky, since we first need to decide how and where to add extra nonlinearity. Our discussion below focuses primarily on LSTMs, but it applies to other sequence models, too.

* We could add extra nonlinearity to the gating mechanisms. That is, instead of using a single perceptron we could use multiple layers. This leaves the *mechanism* of the LSTM unchanged. Instead it makes it more sophisticated. This would make sense if we were led to believe that the LSTM mechanism describes some form of universal truth of how latent variable autoregressive models work.
* We could stack multiple layers of LSTMs on top of each other. This results in a mechanism that is more flexible, due to the combination of several simple layers. In particular, data might be relevant at different levels of the stack. For instance, we might want to keep high-level data about financial market conditions (bear or bull market) available, whereas at a lower level we only record shorter-term temporal dynamics.

Beyond all this abstract discussion it is probably easiest to understand the family of models we are interested in by reviewing :numref:`fig_deep_rnn`. It describes a deep recurrent neural network with $L$ hidden layers. Each hidden state is continuously passed to both the next timestep of the current layer and the current timestep of the next layer.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_modern_recurrent-networks/deep-rnn.md

![ Architecture of a deep recurrent neural network. ](../img/deep-rnn.svg)
:label:`fig_deep_rnn`

<<<<<<< HEAD:chapter_recurrent-neural-networks/deep-rnn.md

In time step $t$, we assume the mini-batch input is given as $\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$ (number of examples: $n$, number of inputs: $d$). The hidden state of hidden layer $\ell$ ($\ell=1,\ldots,T$) is $\boldsymbol{H}_t^{(\ell)}  \in \mathbb{R}^{n \times h}$ (number of hidden units: $h$), the output layer variable is $\boldsymbol{O}_t \in \mathbb{R}^{n \times q}$ (number of outputs: $q$), and the hidden layer activation function is $\phi$. The hidden state of hidden layer 1 is calculated in the same way as before:

$$\boldsymbol{H}_t^{(1)} = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh}^{(1)} + \boldsymbol{H}_{t-1}^{(1)} \boldsymbol{W}_{hh}^{(1)}  + \boldsymbol{b}_h^{(1)}),$$


Here, the weight parameters $\boldsymbol{W}_{xh}^{(1)} \in \mathbb{R}^{d \times h} and \boldsymbol{W}_{hh}^{(1)} \in \mathbb{R}^{h \times h}$ and bias parameter $\boldsymbol{b}_h^{(1)} \in \mathbb{R}^{1 \times h}$ are the model parameters of hidden layer 1.
=======
## Functional Dependencies

At timestep $t$ we assume that we have a minibatch $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ (number of examples: $n$, number of inputs: $d$). The hidden state of hidden layer $\ell$ ($\ell=1,\ldots, T$) is $\mathbf{H}_t^{(\ell)}  \in \mathbb{R}^{n \times h}$ (number of hidden units: $h$), the output layer variable is $\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (number of outputs: $q$) and a hidden layer activation function $f_l$ for layer $l$. We compute the hidden state of layer $1$ as before, using $\mathbf{X}_t$ as input. For all subsequent layers, the hidden state of the previous layer is used in its place.

$$\begin{aligned}
\mathbf{H}_t^{(1)} & = f_1\left(\mathbf{X}_t, \mathbf{H}_{t-1}^{(1)}\right), \\
\mathbf{H}_t^{(l)} & = f_l\left(\mathbf{H}_t^{(l-1)}, \mathbf{H}_{t-1}^{(l)}\right).
\end{aligned}$$

Finally, the output layer is only based on the hidden state of hidden layer $L$. We use the output function $g$ to address this:

$$\mathbf{O}_t = g \left(\mathbf{H}_t^{(L)}\right).$$

Just as with multilayer perceptrons, the number of hidden layers $L$ and number of hidden units $h$ are hyper parameters. In particular, we can pick a regular RNN, a GRU, or an LSTM to implement the model.
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_modern_recurrent-networks/deep-rnn.md

When $1 < \ell \leq L$, the hidden state of hidden layer $\ell$ is expressed as follows:

<<<<<<< HEAD:chapter_recurrent-neural-networks/deep-rnn.md
$$\boldsymbol{H}_t^{(\ell)} = \phi(\boldsymbol{H}_t^{(\ell-1)} \boldsymbol{W}_{xh}^{(\ell)} + \boldsymbol{H}_{t-1}^{(\ell)} \boldsymbol{W}_{hh}^{(\ell)}  + \boldsymbol{b}_h^{(\ell)}),$$

=======
Fortunately many of the logistical details required to implement multiple layers of an RNN are readily available in Gluon. To keep things simple we only illustrate the implementation using such built-in functionality. The code is very similar to the one we used previously for LSTMs. In fact, the only difference is that we specify the number of layers explicitly rather than picking the default of a single layer. Let's begin by importing the appropriate modules and loading data.

```{.python .input  n=17}
import d2l
from mxnet import npx
from mxnet.gluon import rnn
npx.set_np()
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_modern_recurrent-networks/deep-rnn.md

Here, the weight parameters $\boldsymbol{W}_{xh}^{(\ell)} \in \mathbb{R}^{h \times h} and \boldsymbol{W}_{hh}^{(\ell)} \in \mathbb{R}^{h \times h}$ and bias parameter $\boldsymbol{b}_h^{(\ell)} \in \mathbb{R}^{1 \times h}$ are the model parameters of hidden layer $\ell$.

<<<<<<< HEAD:chapter_recurrent-neural-networks/deep-rnn.md
Finally, the output of the output layer is only based on the hidden state of hidden layer $L$:

$$\boldsymbol{O}_t = \boldsymbol{H}_t^{(L)} \boldsymbol{W}_{hq} + \boldsymbol{b}_q,$$

Here, the weight parameter $\boldsymbol{W}_{hq} \in \mathbb{R}^{h \times q}$ and bias parameter $\boldsymbol{b}_q \in \mathbb{R}^{1 \times q}$ are the model parameters of the output layer.
=======
The architectural decisions (such as choosing parameters) are very similar to those of previous sections. We pick the same number of inputs and outputs as we have distinct tokens, i.e., `vocab_size`. The number of hidden units is still 256. The only difference is that we now select a nontrivial number of layers `num_layers = 2`.

```{.python .input  n=22}
vocab_size, num_hiddens, num_layers, ctx = len(vocab), 256, 2, d2l.try_gpu()
lstm_layer = rnn.LSTM(num_hiddens, num_layers)
model = d2l.RNNModel(lstm_layer, len(vocab))
```
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_modern_recurrent-networks/deep-rnn.md

Just as with multilayer perceptrons, the number of hidden layers $L$ and number of hidden units $h$ are hyper parameters. In addition, we can create a deep gated recurrent neural network by replacing hidden state computation with GRU or LSTM computation.

## Summary

<<<<<<< HEAD:chapter_recurrent-neural-networks/deep-rnn.md
* In deep recurrent neural networks, hidden state information is continuously passed to the next time step of the current layer and the next layer of the current time step.


## Exercises

* Alter the model in the ["Implementation of a Recurrent Neural Network from Scratch"](rnn-scratch.md) section to create a recurrent neural network with two hidden layers. Observe and analyze the experimental phenomena.
=======
* In deep recurrent neural networks, hidden state information is passed to the next timestep of the current layer and the current timestep of the next layer.
* There exist many different flavors of deep RNNs, such as LSTMs, GRUs, or regular RNNs. Conveniently these models are all available as parts of the `rnn` module in Gluon.
* Initialization of the models requires care. Overall, deep RNNs require considerable amount of work (such as learning rate and clipping) to ensure proper convergence.

## Exercises

1. Try to implement a two-layer RNN from scratch using the single layer implementation we discussed in :numref:`sec_rnn_scratch`.
2. Replace the LSTM by a GRU and compare the accuracy.
3. Increase the training data to include multiple books. How low can you go on the perplexity scale?
4. Would you want to combine sources of different authors when modeling text? Why is this a good idea? What could go wrong?
>>>>>>> 1ec5c63... copy from d2l-en (#16):chapter_modern_recurrent-networks/deep-rnn.md

## [Discussions](https://discuss.mxnet.io/t/2369)

![](../img/qr_deep-rnn.svg)

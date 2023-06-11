# _Physics-Informed_ Neural Networks (PINNs) #1: 1D Harmonic Oscillator
In this code, we implement a neural network architecture based in constrained learning (physics-informed) to fit the soultion of the under-damped harmonic oscillator (DHO).

<div align="center">
   <img src="https://github.com/idiegoalvarado/pinn-dho/assets/87738807/feaaafd2-c4e8-4757-899e-1bfb13676c0f" width="800">
</div>

## Problem Description

A _harmonic oscillator_ is a fundamental concept in physics that describes a system where an object or a physical quantity oscillates back and forth around an equilibrium position. The differential equation (DE) for a simple oscillator is given by:

$$
    m \frac{\mathrm{d}^2 x}{\mathrm{d} t ^2} + \mu \frac{\mathrm{d} x}{\mathrm{d} t} + kx = 0
    ;
    \quad
    x(0) = x_0, \, \frac{\mathrm{d} x}{\mathrm{d} t} \bigg|_ {t=0} = v_0,
$$

where $x_0$, $v_0$ $\in \mathbb{R}$ and $m$, $\mu$ and $k$ > 0.

It's very important to the DE solution to define values for coefficients $m$, $\mu$ and $k$. Let's define $\delta$ and $\omega_0$

$$
  \begin{cases}
    \delta < \omega _ 0 & \text{(under-damped state) Exact analytical solution} \\
    \delta = \omega _ 0 & \text{(critical-damped state) } \\
    \delta > \omega _ 0 & \text{(over-damped state) No analytical solution}
  \end{cases}; \quad
  \delta = \frac{\mu}{2m}, \quad \omega _ 0 = \sqrt{ \frac{k}{m}}
$$

for the **under-damped state** with initial conditions { $x(0) = x_0, \, \dot{x}(0) = v_0$ } we have the following analytical solution:

$$
    x(t) = e^{-\delta t} \left( 2 A \cos(\phi + \omega t) \right), 
$$

where

$$
  \omega=\sqrt{\omega_0^2 - \delta^2}; \quad \phi = \arctan \left(-\frac{\delta}{\omega} - \frac{v_0}{x_0 \omega} \right), \quad A = \frac{x_0}{2} \sec \phi
$$


## Neural Network (NN): Standard _fully-connected_ neural network

In this case, we are going to use an standard **fully-connected neural network** (FCNN), also known as a dense neural network. In this kind of NN each neuron in one layer is connected to every neuron in the subsequent layer. In other words, all nodes in one layer are _fully connected_ to all nodes in the next layer.

In a fully connected neural network, the layers are typically organized into an **input layer**, one or more **hidden layers**, and an **output layer**: 
* The **input layer** receives the initial data, and each neuron in this layer corresponds to a feature or attribute of the input data. 
* The **hidden layers** are intermediary layers that perform computations and extract relevant features from the input data. 
* The **output layer** produces the final predictions or outputs of the network.

> Neural network unit:

$$
    H_{W, b} (\nu) = \phi (W \nu + b)
$$

> Multilayer neural network with linear output

$$
    u_{\theta} (\nu) = H_{W, b} (\nu) = W H_{\theta_{L-1}} \circ H_{\theta_{L-2}} \circ \cdots \circ H_{\theta_{L-1}} (\nu) + b
$$

### Machine Learning problem: Minimize _Loss_ function

A loss function is a mathematical function that measures the discrepancy or error between the predicted outputs of a machine learning model and the expected outputs. Most of the times, a machine learning problem implies to optimize this loss function.

$$
    u^{* } = \mathrm{argmin} _ {\mathbf{u}} \frac{1}{N} \sum _ {i=1} ^ {N} L (\nu_i, y_i, \mathbf{u})
$$

For a Mean Squared Error (MSE) loss function we have:

$$
    \theta^{* } = \mathrm{argmin} _ {\boldsymbol{\theta}} \frac{1}{N} \sum _ {i=1} ^ {N} (u_{\boldsymbol{\theta}}(t_i, x_i) - y_i)^2
$$

For the DHO we have
$$
    \theta^{* } = \mathrm{argmin} _ {\boldsymbol{\theta}} \frac{1}{N} \sum _ {i=1} ^ {N} \left(x_{\boldsymbol{\theta}}(t_i) - x_i \right)^2
$$
where $x_{\boldsymbol{\theta}}(t_i)$ are the values to be optimized, $x_i$ are the input values or training data points and $\boldsymbol{\theta}$ is the tensor of parameters under which $x$ must be optimized.


## Physics-Informed Neural Network (PINN)

PINN are a type of neural network architecture that combines deep learning techniques with principles of physics to solve partial differential equations (PDEs) or other physical problems. PINNs are particularly useful when only limited or noisy data is available to describe the underlying physics.

The main idea behind PINNs is to incorporate the governing equations of a physical system as additional constraints during the training of a neural network, this is a _learn under constraint_ problem.

In this frame, we can induce the constraint in the loss function optimization problem, this is

$$
\begin{align*}
    \theta^{* } &=  \mathrm{argmin} _ {\boldsymbol{\theta}} \frac{1}{N} \sum _ {i=1} ^ {N} \left(x_{\boldsymbol{\theta}}(t_i) - x_i \right)^2
    \\
    & \quad\quad \text{s. t. } \quad
    m \frac{\mathrm{d}^2}{\mathrm{d} t ^2} x_{\boldsymbol{\theta}}(t) + \mu \frac{\mathrm{d}}{\mathrm{d} t} x_{\boldsymbol{\theta}}(t) + kx_{\boldsymbol{\theta}}(t) = 0, \,\, t \in [0,T)
\end{align*}
$$

this is equivalent to

$$
\begin{align*}
    \theta^{* } &= \mathrm{argmin} _ {\boldsymbol{\theta}} \frac{1}{N} \sum _ {i=1} ^ {N} \left(x_{\boldsymbol{\theta}}(t_i) - x_i \right)^2
    \\
    &  \quad\quad \text{s. t. } \quad
    \int _ {[0,T)} \left( m \frac{\mathrm{d}^2}{\mathrm{d} t ^2} x_{\boldsymbol{\theta}}(t) + \mu \frac{\mathrm{d}}{\mathrm{d} t} x_{\boldsymbol{\theta}}(t) + kx_{\boldsymbol{\theta}}(t) \right) ^2 \mathrm{d}t 
\end{align*}
$$

Using Lagrange multiplier we have

$$
    \theta^{* } = \mathrm{argmin} _ {\boldsymbol{\theta}} \frac{1}{N} \sum _ {i=1} ^ {N} \left(x_{\boldsymbol{\theta}}(t_i) - x_i \right)^2
    + \lambda \int _ {[0,T)} \left( m \frac{\mathrm{d}^2}{\mathrm{d} t ^2} x_{\boldsymbol{\theta}}(t) + \mu \frac{\mathrm{d}}{\mathrm{d} t} x_{\boldsymbol{\theta}}(t) + kx_{\boldsymbol{\theta}}(t) \right) ^2 \mathrm{d}t 
$$

Finally we can use Monte Carlo fundations to state the following equation

$$
    \theta^{* } = \mathrm{argmin} _ {\boldsymbol{\theta}} \frac{1}{N} \sum _ {j=1} ^ {N} \left(x_{\boldsymbol{\theta}}(t_i) - x_i \right)^2
    + \lambda \frac{1}{N_{\text{phy}}} \sum _ {j=1}^{N_{\text{phy}}} \left( m \frac{\mathrm{d}^2}{\mathrm{d} t ^2} x_{\boldsymbol{\theta}}(t_j^p) + \mu \frac{\mathrm{d}}{\mathrm{d} t} x_{\boldsymbol{\theta}}(t_j^p) + kx_{\boldsymbol{\theta}}(t_j ^p ) \right) ^2
$$

for more details check https://doi.org/10.1016/j.jcp.2018.10.045.

> Introducing _Physics Loss_ to the machine learning model

The physics loss aims to ensure that the learned solution is consistent with the underlying differential equation. This is done by penalising the residual of the differential equation over a set of locations sampled from the domain.

Here we evaluate the physics loss at $N_{\text{phy}} =$ `n_phy` $=$ 50 points uniformly spaced over the time domain $[0,T)$. We can calculate the derivatives of the network solution with respect to its input variable at these points using `pytorch`'s autodifferentiation features, and can then easily compute the residual of the differential equation using these quantities.

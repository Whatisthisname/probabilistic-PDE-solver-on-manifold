# Notation
Here will be listed some of the less common conventions used this thesis. Most are introduced as convenience to reduce visual clutter in formulas, but it is of importance to make explicit the dependece between terms for the new reader:

# Differential equations and operators
In existing literature, if a $u$ is a function of one independent variable $t$, the derivative can be written in a number of ways: $\frac{\text{d}}{\text{d}t}u = u^{(1)} = u' = u_t$. In *this* article, we will prefer the explicit $\frac{\text{d}}{\text{d}t}u$ and $u^{(1)}$, because $u'$ might be confused with $u$, and $u_t$ looks like indexing a vector. With this choice, the $n$'th derivative is then written as $\frac{\text{d}^n}{\text{d}t^n}u = u^{(n)}$, which also suggests the equality $u = u^{(0)}$. 

For a function $v$ of multiple variables, we use partial derivatives to specify the variable vith respect to which we are taking the derivative. If $v$ takes arguments $x$ and $t$, one might find notation $\frac{\partial}{\partial t}v = v_t$ and $\frac{\partial}{\partial x}v = v_x$ and $\frac{\partial^2}{\partial t\partial x}v = v_{tx}$. We will stick to the $\partial$'s and avoid the subscript for the aforementioned reason.

If $w:\reals\to\reals^d$ is a vector-valued function with scalar input $t$ then $\frac{\partial}{\partial t}w : \reals\to\reals^d$. $\frac{\partial}{\partial t}q(t)$ tells us how an infinitesimal positive nudge in the input scalar $t$ will affect the vector valued output.

Had $w$ been a scalar-valued function with vector domain $\reals^d$, then $\nabla w : \reals^d\to\reals$. $\nabla w$ tells us how a infinitesimal positive nudge in the full input space will affect the scalar output dimensions.

When both domain and image are multi-dimensional,  $q:\reals^n\to\reals^d$, we use the Jacobian:

$J(w) = \begin{bmatrix} \frac{\partial}{\partial x_1}q & \cdots & \frac{\partial}{\partial x_n}q \end{bmatrix} =
  \begin{bmatrix} (\nabla q_1)^\intercal \\ \vdots \\ (\nabla q_d)^\intercal \end{bmatrix} = \begin{bmatrix} \frac{\partial q_1}{\partial x_1} & \cdots & \frac{\partial q_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial q_d}{\partial x_1} & \cdots & \frac{\partial q_d}{\partial x_n} \ \end{bmatrix} \in \reals^{d\times n}$

The jacobian has the property that it is the best linear approximation to $w$ at a point $x$, in the sense that $w(x+h) = w(x) + J(w)(x)h + o(h)$ for small $h$. This is the first-order Taylor expansion of $w$ around $x$.

$\text{div}(\phi)$ denotes the divergence of a vector field $\phi$, and is defined as $\text{div}(\phi) = \sum_{i=1}^n \frac{\partial}{\partial x_i}\phi_i$  We will not use it explicitly in any PDEs in this article, but it appears in the navier stokes' equations describing motion fluids, but more importantly in the definition of the Laplace operator.

$\Delta$ is used to denote the Laplace operator, $\Delta = \text{div}(\nabla) = \nabla\cdot\nabla = \sum_{i=1}^n \frac{\partial^2}{\partial x_i^2}$. It has applications in partial differential equations such as the heat and poisson equations $\frac{\partial}{\partial t}u(x, t) = -c \Delta u(x,t)$ and $\Delta u(x) = f(x)$ or the wave equation $\frac{\partial^2}{\partial t^2}u(x, t) = -c \Delta u(x,t)$. In one input dimension, $\Delta u (x) = u^{(2)}(x)$ is often interpreted as the curvature of a function, or the amount with which it exceeds its neigboring points. This intuition can be transferred to higher dimensions.

The Laplace-Beltrami operator $\Delta_g$ is the generalisation of the Laplace operator to riemannian manifolds. The explicit form is

$$\Delta_g f (x) = \frac{1}{\sqrt{|\text{det}(g(x))|}} \sum_{i=1}^n \frac{\partial}{\partial x_i}\Big(\sqrt{|\text{det}(g(x))|} g(x)^{-1} \nabla f(x)\Big)$$

# Notation
In this work, the Laplacian operator $\Delta$ is understood to act only on the spatial coordinates x. We use $\Delta$ both for the euclidean laplacian and for the Laplace-Beltrami operator

Capital variable names denote random variables, their lowercase variant deterministic, known values.

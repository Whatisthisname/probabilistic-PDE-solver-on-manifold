# Thesis
Theo Rüter Würtzen

## What is a manifold?

A manifold is hypersurface in an ambient space, for example, the surface of the earth is a 2-dimensional surface embedded in ambient 3-dimensional space.

A riemannian manifold is the same, but it requires the manifold to be smooth, and defines a way to measure areas and distances on the surface through an inner product at the tangent space at each point.

Each point in a riemannian manifold has an associated Tangent Space $\mathcal{TM}_p$, where the $p$ denotes a dependence on the point. The tangent space is a local "linearization" of the manifold. On a sphere like the earth, the tangent space of the north pole is the unique tangent plane (two dimensional subspace of the ambient space) that that brushes only the north pole. Moving from $p$ along a vector from the tangent space will in general take one away from the manifold.
While the points on the manifold do not form a vector space, the tangent space does when equipped with an inner product, which we denote $g_p: \mathcal{TM}_p \times \mathcal{TM}_p \to \reals$. $g_p$ takes two tangent vectors $t_1, t_2$, and returns something like their inner product, but "stretched". This connection gives rise to the notation $g_p(t, t) = \langle t,t \rangle _p = ||t||^2_p$ - a length measurement. 
If the tangent space is two-dimensional, (which implies that the surface is two-dimensional), then a two-dimensional local coordinate system will suffice to describe points and tangent vectors. In this case, $g_p$ can be expressed as a vector-matrix-vector product, $g_p(t_1, t_2) = t_1^\top g_p t_2$, where $g_p$ has been overloaded to also be a symmetric positive definite 2x2 matrix. If $g$ is the 2x2 identity, we recover the standard euclidean inner product. This matrix is called the metric tensor.

A RM is a set of points that locally look like a section of the euclidean plane, meaning that each open neighborhood of each point can be mapped bijectively to an open sphere in the euclidean plane. 

This mapping from manifold M to coordinate system R n is called a chart - the chart assigns a set of local coordinates to the manifold. They are local because they are based around the neighborhood of a point.
The coordinates of the tangent plane in conjunction with projection between manifold and tangent plane form a valid chart, as long as the projection is diffeomorphic. A diffeomorphism is when an invertible function itself and its inverse are continuously differentiable. 
If we use the north pole tangent plane as our map and project back and forth, there will be a problem for every point below the equatorial line.
The chart is an ordered tuple of $(U ,\phi)$, where U is an open subset of the manifold (example: nothern hemisphere without the equator) and $\phi$ is diffeomorphic (example: the aforementioned projection operation).

Something akin to the opposite of the chart is an embedding. An embedding is a function mapping from the coordinate-space to a the manifold expressed in a higher-dimensional euclidean coordinate system. Returning to the sphere, it has a 2D surface but lives in a 3D ambient space. If we define our manifold as the image of an embedding, we also get a metric tensor for this manifold.
If we want to measure something on a manifold, we never actually work with that manifold. The manifold is an abstract object relating points to each other, but it has no intrinsic coordinates. To talk about the manifold and do computations, we choose a local coordinate system. We can then measure lengths of paths and areas of patches on the manifold by measuring these quantities on our local coordinate system. We do however need to adjust our measurements, as the chart will in general distort lengths and angles. This is where the metric tensor, defined at each point on the manifold (and therefore at each point in the local coordinate system). 

Measurements on the chart will involve integrals - when we measure the length of a path, we integrate length of it's tangent vector. 

If we want to measure the length of a line $P: (0, T) \to \reals^2$ and $||P|| = \int_P ds$ where $ds$ is the length element, a differential 1-form. By the pythagorean theorem, the length is $ds = \sqrt{dx^2 + dy^2}$, and $dx$ (and $dy$) are $\frac{\partial P}{\partial x}$.
$dx$ is a differential 1-form, and can be thought of as a vector field defined on the line $P$, pointing in direction $x$ with magnitude equal to the partial derivative strength. 
$\int_P \sqrt{dx^2 + dy^2} = \int_0^T ||dP|| = \int_0^T ||\frac{\partial P}{\partial x} \text{d}x + \frac{\partial P}{\partial y}\text{d}y|| = \int_0^T ||\frac{\partial P}{\partial x} \text{d}x + \frac{\partial P}{\partial y}\text{d}y||$ 

To measure the length of a parameterized curve $P: (0, T) \to \reals^2$, we traditionally express it like this: $$||P|| = \int_0^T \sqrt{\frac{\partial P}{\partial x}^2 + \frac{\partial P}{\partial y}^2} \text{d}t$$ 
This example of a parametrized curve can be analyzed through the lenses of differential geometry. The image of the map $P$, $\mathcal{M} \subset \reals^2$, is our one-dimensional manifold embedded in $\reals^2$. 
If $P$ is diffeomorphic (invertible and both $P$ and $P^{-1}$ are continuously differentiable), then $(P^{-1}, \mathcal{M})$ is a chart of $\mathcal{M}$.
The length expression above can be broken down in the following way: The manifold is intrinsically one-dimensional. This is why our integral is on the real euclidean line from $0$ to $T$. The integrand $\sqrt{\frac{\partial P}{\partial x}^2 + \frac{\partial P}{\partial y}^2}\text{d}t$ is a differential 1-form. It becomes a differential 1-form because it contains the term $\text{d}t$, and $\sqrt{\frac{\partial P}{\partial x}^2 + \frac{\partial P}{\partial y}^2}$ is its magnitude. 

Had we taken just the expression $\int_0^T 1\cdot \text{d}t = T$, where '$1$' can be thought of as the positively oriented tangent vector of $\reals$, we would simply be measuring directly in the local coordinate system, not accounting for the stretching of the embedded manifold. We would like to keep measuring in our local coordinate system, but will have to adjust our measurement at each point to account for the stretching between the embedded manifold and the local coordinate system. The quantity $q := \sqrt{\frac{\partial P}{\partial x}^2 + \frac{\partial P}{\partial y}^2}$ is the magnitude of the line element on our manifold, which tells us how much the space is stretched or compressed relative to the local coordinate system. 
This quantity is the inner product of tangent vectors on the embedded manifold, which can be obtained by linearizing $P$ at each point using the Jacobian, $\sqrt{\nabla P^\top \nabla P} = ||\nabla P|| = q$, the length of the tangent vector on the manifold in $\reals^2$. 

As mentioned, we can use the metric tensor to measure lengths of tangent vectors. On $\reals$, the tangent 'vector' is the scalar '$1$'. This gives us a distance measurement of $\sqrt{g_p(1, 1)} = \sqrt{1^\top q^2 1} = q$, telling us the the metric tensor is $q^2$, the inner product of the tangent vectors in the ambient space.  

The metric tensor tells us how we should adjust our local coordinate measurements if we want to measure w.r.t. how the manifold 'looks' like in the embedded euclidean space.

Now, on to **gradients**!

Say we have a scalar function defined on the local coordinate system, call it $f: [0, T] \to \reals$. We want to get the gradient of this vector, we can take the classic gradient, but will have to transform the vector-field based on the underlying metric. 
$\nabla _g f = \sqrt{|g^{-1}|} \nabla f$

If we have a 2D manifold, the gradient will also be a 2-vector, and the tensor a 2x2 matrix.
$\nabla _g f_1 = \sqrt{|g^{-1}|}\cdot(\nabla f_1 g_{11} + \nabla f_2 g_{12}) = \sqrt{|g^{-1}|} g_{1}\top \nabla f$


**How does the laplace beltrami operator work?**

The laplacian is the divergence of the gradient. Just accept this as a fact, it works in euclidean space at least: $\Delta f = \nabla \cdot \nabla f = \text{div} (\nabla f)$.

If our space is not euclidean but has a metric tensor for each point, we need to incorporate that into our calculation. For intuition: If a function is seems to be changing rapidly (in local coordinates), but the space deformation is compressing a lot of space together in the point, then, after decompressing into an isometric map, that rapid change will have "spread out", and is actually not as rapid. So, stretching space will affect our laplacian and gradients.

To define the gradient on a manifold, first, we need a scalar function defined on the manifold. For us, the manifold is just the $\reals^2$ vector space with a metric tensor at each point that changes how we measure things - this is also known as a local coordinate system, but it is nice to work in, as it is a vector space and we have the classic cartesian basis vectors $e^1, e^2$. Let's denote this domain as $\mathcal{C} \subset \reals^2$, for "chart" or "cartesian".

We will define our scalar function on $\mathcal{C}$, our space with non-euclidean metric tensor. Let's agree to call it $\phi: \mathcal{C} \to \reals$. We are interested in calculating $\nabla_g \phi$, which is the gradient that respects the stretching from the metric tensor. We already know how to compute $\nabla \phi$, which is just the classic calculus jacobian from euclidean space. Alright - let me show you the formula first: 
$$\nabla_g \phi = \sqrt{|g^{-1}|}g\nabla \phi$$
Let's break it down.
$$\underbrace{\nabla_g \phi}_{\substack{\text{manifold} \\ \text{gradient} \\ \left(\substack{\text{tangent} \\ \text{basis}}\right)}} = \overbrace{\sqrt{|g^{-1}|}}^{\substack{\text{scalar} \\ \left(\substack{\text{normalization} \\ \text{factor}} \right)}} \underbrace{g}_{\substack{ \text{metric tensor} \\ \left(\substack{\text{change of basis to} \\ \text{scaled tangent plane}} \right)}} \overbrace{\nabla \phi}^{\substack{\text{euclidean}\\ \text{gradient} \\ \left(\substack{\text{local}\\ \text{basis}}\right)}}$$


## Papers to read:

https://arxiv.org/pdf/2006.10160 - Matérn Gaussian Processes on Manifolds

https://proceedings.mlr.press/v151/nikitin22a/nikitin22a.pdf - Non-separable Spatio-temporal Graph Kernels via SPDEs

https://users.aalto.fi/~asolin/sde-book/sde-book.pdf Applied Stochastic Differential Equations - Chapter 10 and 12, maybe chapter 1

# Notation
In this work, the Laplacian operator Δ is understood to act only on the spatial coordinates x.

# Abstract

In this thesis we discuss details of implementing a probabilistic numerical solver for partial differential equations whose spatial domain is not a vector space. We cover topics including the concept of the metric tensor, embeddings of manifolds, mesh representations and refinement, discrete differential operators, ODEs, PDEs, and SDEs, numerical linear algebra, state space models, the (extended) Kálmán-filter and -smoother, and numerical stability. We conclude with experiments that test the efficiency of physically informed prior solutions in non-euclidean domains.

We have structured the thesis as a guide on how to implement and follow the results, and we try to motivate the steps to the best of our efforts.

On a high level, these are the steps that will be explained thoroughly:

1. Choose a riemannian 2-manifold $\mathcal{M}$, either through a metric tensor and coordinate chart, or through an already discretized representation as a triangle mesh.
2. Choose a time-evolving PDE such as the linear heat equation $\frac{\partial}{\partial t}u(t,\mathbf{x}) = -\Delta u(t,\mathbf{x})$ or perhaps a nonlinear wave-like equation $\frac{\partial^2}{\partial t^2}u(t,\mathbf{x}) = -\Delta u(t,\mathbf{x}) - 3\Delta \tanh(u(\mathbf{x},t))$. Choose a time-window in which the PDE is to be simulated, $\mathcal{T} = [0, T_{final}]$, an initial condition $u(\mathbf{x}, 0)$ and potentially boundary conditions for all times $t\in \mathcal{T}$.
3. We will convert the PDE in to a system of ODEs using the Method Of Lines, by first discretizing the spatial domain into finite sets of points $V$. We denote the spatial restriction as $u_V(t) := \left . u(x,t)\right|_{x=V} \in \reals^{|V|}$. The discretized domain will require a discrete version of $\Delta$, for which the cotan-Laplacian $L \in \reals^{|V|\times |V|}$, a matrix, presents itself. Depending on the representation of the manifold, there are different ways to compute it. As an example, the linear heat equation then becomes $\frac{\partial}{\partial t}u_V(t) = -Lu_V(t)$ and is a system of ODEs.
4. Based on the PDE, select a Gauss-Markov prior process distribution $\mathcal{GP}(t)$ that jointly models the solution $u_V(t)$ and its first $q\geq 1$ time-derivatives restricted to spatial points $V$. $q$ must be chosen to be at least as high as the order of the PDE. Specifically, this means we have a single-input (time) multi-output (scalar value and derivatives at spatial points) gaussian process: $\begin{bmatrix} U_V(t) & \frac{\partial}{\partial t} U_V(t) & \cdots & \frac{\partial^q}{\partial t^q} U_V(t)\end{bmatrix}^{\intercal} = \vec{U}_V(t)\sim \mathcal{GP(t)}$.
5. Define a residual function $R(\vec{u}_v(t)) \in \reals^{|V|}$. Taking again the previous linear heat equation as an example, we choose $R(\vec{u}_V(t)) = u_V(t) + L u_V(t)$, because $R(\vec{u}_V(t)) = \mathbf{0} \iff u_V(t) = -Lu_V(t)$. A solution to the PDE would have residual $\mathbf{0}$ everywhere. The residual function relates the derivatives and solution to each other.
6. Further discretize the problem by considering $\mathcal{GP(t)}$ only at finite times $\mathbb{T} \subset [0, T_{final}]$. By our choice of a Gauss-Markov prior, the distributions at two different times will be multivariate gaussian and independent given an intermediate times between the pair. This time-space discretized multivariate joint distribution is denoted $\mathbb{GP}_V(t)$ for $t\in \mathbb{T}$. Using the conditional independence structure of the prior, factor the joint distribution into the product of a gaussian prior $\vec{U}_V(0) \sim \mathcal{N}(\mathbf{v}_0, \Sigma_0)$ and multiple 1-step recurrence relation of the form $\vec{U}(\mathbf{v},t+1) \;|\; \vec{u}(\mathbf{v}, t) \sim \mathcal{N}(A\mathbf{v}, \Sigma)$. This is known as a linear gaussian model for some given parameters $A, \Sigma$.

7. Define the posterior distribution over the solution to the PDE as $\mathbb{GP}_V(t) \;| \; \{R(\vec{U}_V(t)) = 0 \; \forall t \in \mathbb{T}\} $ 

8. Implement a Kalman-filter and -smoother to compute the posterior distribution $\mathbb{GP}(\mathbf{v},t) \; | \; \mathbf{v}_0, $


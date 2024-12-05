
# Abstract

In this thesis we discuss details of implementing a probabilistic numerical solver for partial differential equations whose spatial domain is not a vector space. We cover topics including the concept of the metric tensor, embeddings of manifolds, mesh representations and refinement, discrete differential operators, ODEs, PDEs, and SDEs, numerical linear algebra, state space models, the (extended) Kálmán-filter and -smoother, and numerical stability. We conclude with experiments that test the efficiency of physically informed prior solutions in non-euclidean domains.

We have structured the thesis as a guide on how to implement and follow the results, and we try to motivate the steps to the best of our efforts.

# Building a probabilistic numerical solver on a 2-manifold

On a very high level, these are the steps necessary to build a probabilistic numerical solver on a 2-manifold

1. Choose a riemannian 2-manifold $\mathcal{M}$, either through a metric tensor and coordinate chart, or through an already discretized representation as a triangle mesh.
2. Choose a time-evolving PDE such as the linear heat equation $$\frac{\partial}{\partial t}u(t,\mathbf{x}) = -\Delta u(t,\mathbf{x})$$. Choose a time-window in which the PDE is to be simulated, $\mathcal{T} = [0, T]$, an initial condition $u(\mathbf{x}, 0)$ and potentially boundary conditions for all times $t\in \mathcal{T}$.
3. We will convert the PDE in to a system of ODEs using the Method Of Lines, by first discretizing the spatial domain into finite sets of points $V$. We denote the spatial restriction as $$u_V(t) := \left . u(x,t)\right|_{x=V} \in \reals^{|V|}$$
4. The discretized domain will require a discrete version of the Laplacian $\Delta$, for which we will use the cotan-Laplacian $L \in \reals^{|V|\times |V|}$, a matrix. Depending on the representation of the manifold, there are different ways to compute it.  If a metric tensor and coordinate chart are provided, compute and intrinsic triangulation of the domain from points $V$. If a mesh with vertices $V$ is provided, consider it an extrinsic triangulation. Given a triangulation, we can compute $L$. As an example, with $L$, the linear heat equation then becomes $$\frac{\text{d}}{\text{d} t}u_V(t) = -Lu_V(t)$$ which is now a system of ODEs, because there are no partial derivatives.
5. Based on the PDE, select a prior Gauss-Markov process distribution $\mathcal{GP}$ that jointly models the solution function $u_V$ and its first $q\geq 1$ time-derivatives restricted to spatial points $V$. $q$ must be chosen to be at least as high as the order of the PDE. Specifically, this means we have a distribution over single-input (time) multi-output (scalar value and derivatives at spatial points) function distribution: $$\vec{U}_V :=\begin{bmatrix} U_V & \frac{\text{d}}{\text{d} t} U_V & \cdots & \frac{\text{d}^q}{\text{d} t^q} U_V\end{bmatrix}^{\intercal} \sim \mathcal{GP}$$
6. Based on the PDE, define the residual operator $R(\vec{u}_V)(t) \in \reals^{|V|}$. Taking as an example  again the previous linear heat equation, we define $$R(\vec{u}_V)(t) = \frac{\text{d}}{\text{d} t}u_V(t) + L u_V(t)$$ because of the relation $$R(\vec{u}_V)(t) = \mathbf{0} \iff u_V(t) = -Lu_V(t)$$ An exact solution to the PDE would yield $\mathbf{0}$ for all timesteps. 
7. Further discretize the problem by considering the function distribution $\mathcal{GP}$ only at finite equispaced timesteps $\mathbb{T} = \{0, h, 2h, 3h, \dots\, T\}$. By our choice of a Gauss-Markov prior, the marginal distributions at two different times will be multivariate gaussian and independent given an intermediate times between the pair. Using the conditional independence structure of the prior, factor the joint distribution into the product of a gaussian prior $$\vec{U}_V(0) \sim \mathcal{N}(\vec{u}_V(0), \Sigma_0)$$ and multiple 1-step recurrence relation of the form $$\vec{U}_V(t+1) \;|\; \vec{u}_V(t) \sim \mathcal{N}(A\vec{u}_V(t), \Sigma)$$ 
resulting in factorized joint distribution across all times $$P(\vec{u}_V(0), \vec{u}_V(h), \dots, \vec{u}_V(T)) = \mathcal{N}(\vec{u}_V(0), \Sigma _0)\prod_{t\in \mathbb{T}} \mathcal{N}(\vec{u}_V(t), \Sigma)$$
This is known as a Linear Gaussian Model, given computable parameters $A, \Sigma$.

8. Define and compute the posterior joint distribution over the solution to the PDE as $$P(\vec{u}_V(0), \vec{u}_V(h), \dots, \vec{u}_V(T)) \;\Big| \; (u_V(0), \Sigma_0) \; \cup \; \{R(\vec{U}_V)(t) = 0 \; \forall t \in \mathbb{T}\}$$ This can be computed in the Kalman-filter and -smoother in time $O(|\mathbb{T}||V|^3(1+q)^3)$

9. From the posterior joint distribution, extract quantities of interest (mean, covariance, derivatives)through simple indexing.


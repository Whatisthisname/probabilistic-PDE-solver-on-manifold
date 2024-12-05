# Thesis
Theo Rüter Würtzen



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


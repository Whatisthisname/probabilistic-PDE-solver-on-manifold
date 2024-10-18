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
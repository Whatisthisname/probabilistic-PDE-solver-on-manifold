# Riemannian Manifolds and Differential Geometry

This section covers relevant knowledge of Riemannian Manifold by motivating and explaining key concepts through definitions and examples. It is intended to give intuition and working knowledge.
### Coordinate Chart

For millenia society thought that earth was a flat disc, because that's what it looks like up close. This is becayse the surface of a unit-ball, the unit-sphere, looks locally flat. Mathematically, we say that each point is locally homeomorphic to $\reals^2$.

Although the surface of the earth is 3-dimensional, we humans have found a way to parametrize it in terms of just two numbers, latitude and longitude. This is known as a local representation. The most common representation is the Mercator projection, which is an explicit way to assign meaning to these local coordinates. 

---
#### Definition: Coordinate Chart
The 2D coordinate chart is a diffeomorphism that maps open subset of a 2-manifold $\mathcal{M}$ to an open subset in the Euclidean plane $\mathcal{C} \subseteq \reals^2$. 

---

---
#### Example: Coordinate Chart
We will use the surface of the earth as $\mathcal{M}$ and consruct a coordinate chart. We will map the open upper hemisphere $\subset \mathcal{M}$ to the open unit-disc $\mathcal{C}$ by projecting onto the unique plane that cuts through the equator. 

A bijection from the 2D-unit-disc to the half-sphere is given by $$\text{disc\_to\_hemisphere}(\begin{bmatrix}x & y\end{bmatrix}^\intercal) = \begin{bmatrix}x & y & \sqrt{1 - x^2 - y^2}\end{bmatrix}^\intercal $$ The inverse of this map corresponds to projecting the hemisphere onto the equatorial plane.
[PICTURE OF UNIT DISC AND DOME WITH ARROW MAPPING THEM].

If we carry over the positions of the landmasses with the projection, we end up with our map that is centered around the geographic north pole. [FIGURE WITH EQUIDISTANT LONGITUDAL LINES]. As can be seen, the longitudal lines get compressed towards the edge.

---

### Metric Tensor

Good maps are characterized by being amenable to direct measurements. A 'deficit' of the Mercator projection, that is often pointed out is that the landmasses at the poles, (say, Greenland), are disproportionately big when compared to countries at the equator, (say, Kenya). This difference in scale invalidates direct measurements of lengths using a conventional ruler. The Metric Tensor quantifies the distortions induced by the map, and is a vital tool for measurements on coordinate charts.

Before giving a definition, we will further motivate the Metric Tensor by pointing out what one takes for granted in Euclidean space (specifically, $\reals^2$). $\reals^2$ is a vector space, meaning that linear combinations of the basis-vectors $e_1$ and $e_2$ are still members of $\reals^2$. In a vector space we can additionally define a norm and an inner product. The Euclidean norm is defined as $||(x,y)||_2 = \sqrt{x^2 + y^2}$ and is used to define the distance $d(\vec{a}, \vec{b}) = ||\vec{a}-\vec{b}||_2$. The Euclidean inner product $\langle \vec{a},\vec{b} \rangle = a_1b_1 +a_2b_2$ relates to the angle $\theta$ (radians) between vectors through $\frac{\langle \vec{a},\vec{b} \rangle}{||\vec{a}||||\vec{b}||} = \cos(\theta)$. As we can see, measuring distances and angles is relatively straightforward here.

---
### FOOTNOTE: 🦶
$\reals^2$ is technically the set of all ordered pairs of real numbers, but in the context of differential geometry, it is also the Euclidean plane.

---

Conversely, the surface of the earth is not a vector space. There can be no set of basis vectors, because there is no defined notion of addition or scaling of coordinates (we do not add pairs ($lat$., $long$.), and we do not scale them). Even though we have coordinates, we still don't get distances or angles, because we don't have the inner product or norm. This is the problem the metric tensor solves. 

---
#### Definition: Metric Tensor

A 2D metric tensor is a symmetric, positive-definite 2×2 matrix $g$ defined at a point of a two-dimensional surface in specified in local coordinates. It provides a way to measure lengths, angles, and areas by describing how the infinitesimal distance $ds$ of a local offset in tangent space $\begin{bmatrix} dx & dy\end{bmatrix}^\intercal$ is computed as:

$$ds^2 = g_{11}dx^2 + 2g_{12}dxdy + 2g_{22}dy^2 = \begin{bmatrix} dx & dy\end{bmatrix} \begin{bmatrix} dx \\ dy\end{bmatrix} $$

---
#### Example: Metric Tensor

$ds$ and $dx$ and $dy$ have the same meaning as the $dx$ in $\int_0^1 f(x) \;dx$. Intuitively, if we were to relate the integral back to the Riemann sum, $dx$ is the size of the interval in our partition of $[0, 1] \subset \reals$. By having $dx$ depend on the local coordinates

Positive-definiteness in the definition ensures that all squared infinitesimal distances $ds^2$ will remain positive, no matter the entries in $g$. $g$ can be thought of as encoding a local version of the pythagorean theorem. If $g = \begin{bmatrix} 1 & 0 \\ 0 & 1\end{bmatrix}$, then the formula for infinitesimal distance reduces to $$ds^2 = dx^2 + dy^2$$ which looks very similar to $c^2 = a^2 + b^2$. 

If we have $g = \begin{bmatrix} \xi^2 & 0 \\ 0 & 0\end{bmatrix}$, we get $$ds^2=\xi^2dx^2 \iff ds = \xi dx$$ which states that the distance between two points

---

#### Definition: Riemannian Metric

A 2D Riemannian metric is a mathematical function that defines the infinitesimal distance on a two-dimensional surface. It assigns to each point a symmetric, positive-definite bilinear form that determines how lengths, angles, and areas are measured locally.

#### Example: Riemannian Metric
---
### FOOTNOTE: 🦶
A diffeomorphism is a bijection between two differentiable manifolds such that both it and its inverse are differentiable.

---

### Metric Tensor


Not all interesting spaces are vector spaces, consider the sphere.
        - Consider problems on the surface of the earth. How would a wave travel through some material? What places will be reached at the same time? What strength will the wave have when it hits this place?


Tissot Indicatrix



## What is a riemannian manifold?

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


In this thesis, we consider two representations of Riemannian 2-Manifolds
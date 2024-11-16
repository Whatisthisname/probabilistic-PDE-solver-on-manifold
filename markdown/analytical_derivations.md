## example one: $f(x, y) = [x, y, 2x + 2y]^\top$

$f(x, y) = [x, y, 2x + 2y]^\top$

$\nabla_p f \in \reals^{3\times2} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 2 & 2\end{bmatrix}$

The jacobian is itself the matrix of mapped orthogonal basis vectors (which are also tangent vectors in the original space).

Now we want to take the inner product between the mapped tangent vectors, which would be $ g^M = (\nabla_p f)^\top (\nabla_p f)$. This gives a 2x2 matrix, which is the "metric".
In this case, the matrix is $g^M = \begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix}$

$\nabla_p f$ maps tangent vectors in $\reals^2$ to tangent vectors on the manifold $\in \reals^3$.

Inner product of two vectors $u$ and $v$ turns out to be:
$(\nabla_p f u)^\top \nabla_p f v = u (\nabla_p f^\top \nabla_p f) v = u^\top g^M_p v =: g_p(u, v)$

Let's see what happens when we, at some point $p \in \reals^2$ consider the inner product of the two mapped basis vectors $e^1$ and $e^2$:

$$\begin{align*} 
g(e^1, e^2)=\begin{bmatrix} 1 & 0 \end{bmatrix} \Big(\begin{bmatrix} 5 & 4 \\ 4 & 5 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix}\Big) = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \end{bmatrix} = 4
\\

\end{align*}$$

Would've been zero normally, so this is interesting. Now, how are areas distorted? This would be the determinant of $g^M_p$, which tells us something about the magnitued of the volume enclosed by the transformed tangent vectors. $det(g^M_p) = 5\cdot 5-4\cdot 4 = 9$ (the a c - b d formula for 2x2 matrix determinant).

If we want to integrate the area of patch $ P =[0, 1] \times [0, 1]$ when mapped to the surface, we use the area form $dA = \text{det}(g^M_p) \text{d}x \wedge \text{d}y$. So:

$$\begin{align*} 
&\;\;\;\; \int_P dA = \int_P \sqrt{\text{det}(g^M_p)} \text{d}x \wedge \text{d}y = \sqrt{9} \int_{[0, 1]^2} \text{d}x \text{d}y = 3.
\end{align*}$$

## example two: $f(x, y) = [x, y, ax + by]^\top$

jacobian = $\begin{bmatrix} 1 & 0 \\ 0 & 1 \\ a & b \end{bmatrix}$, metric tensor = $\begin{bmatrix} 1+a^2 & ab \\ ab & 1+b^2 \end{bmatrix}$, determinant = $(1 +a^2 + b^2 + a^2b^2) - a^2b^2 = 1 + a^2 +b^2$.

So, let's get back to the integration example:

$$\begin{align*} 
&\;\;\;\; \int_P dA 
\\&= 
\int_P \sqrt{\text{det}(g^M_p)} \text{d}x \wedge \text{d}y  
\\&= 
\sqrt{1 + a^2 + b^2} \int_{[0, 1]^2} \text{d}x \text{d}y 
\\&= \sqrt{1 + a^2 + b^2}.
\end{align*}$$

Interestingly, because the determinant does not depend on the coordinates (yet), it can be pulled out of the integral and we get the determinant itself. 
Set $b=0$ and $a = 1$, and we get the area equal to $\sqrt{2}$ - this the right answer.

## Keenan asks to show the volume form derivation:

$$
\begin{align*}
&\:\;\;\;\;||df(u) \times df(v)|| 
\\&= ||\nabla f u \times \nabla f v || 
\\&\overset{def}{=} ||\nabla f u ||\cdot ||\nabla f v|| \cdot \underbrace{\sin(\theta)}_{=1}{}
\\&= ||\nabla f u ||\cdot ||\nabla f v||
\end{align*}
$$


Let $v = v_ie^i$ similarly for $v$. Let $\nabla f = J$.

$$
\begin{align*}
&= ||J u || \cdot ||J v||
\\ &= Ju =
\end{align*}
$$

$ ||Ju|| = g(a, u) = a^\top J^\top J u \implies a^\top J^\top = (J a)^\top = I_3 \iff$

$ ||Ju|| = g(a, u) = a^\top g^M u \implies a^\top J^\top = (J a)^\top = I_3 \iff$

$$
\begin{bmatrix}
df(u)_1& df(v)_1 \\
df(u)_2& df(v)_2 \\
df(u)_3& df(v)_3
\end{bmatrix}
$$

The determinant of this is the volume (area) spanned by the column vectors. Whoops. Not true. It doesn't have a determinant. It needs to be square. What I can do is take the following matrix:

$$
\begin{bmatrix}
df(u)^\top df(u) & df(u)^\top df(v) \\
df(u)^\top df(v) & df(v)^\top df(v)
\end{bmatrix} = 
\begin{bmatrix}
g(u, u) & g(u, v) \\
g(u, v) & g(v, v)
\end{bmatrix}
$$

To compute the spanned area, remove the projection of one on the other to keep the normal component:

We have u and v, can multiply lengths of u with (v - (u.v)u):

$$\text{area} = ||u||\cdot ||v - \frac{u}{||u||}^\top \frac{v}{||v||} v||$$
$$= ||v\cdot ||u|| - u^\top v \frac{v}{||v||}||$$
$$= ||v\cdot ||u|| - u^\top v \frac{v}{||v||}||$$

The local coordinate system of the embedded tangent space is:

We project each of the spanning bases onto the plane. We now have two new column vectors for the plane. We can

$$
\begin{bmatrix}
df(u)^\top df(u) & df(u)^\top df(v) \\
df(u)^\top df(v) & df(v)^\top df(v)
\end{bmatrix} = 
\begin{bmatrix}
df(u)^\top df(u) / ||df(u)|| & df(u)^\top df(v) \\
df(u)^\top df(v) & df(v)^\top df(v)
\end{bmatrix} = 
$$

$g(u,u) = u^\top g^M u$

---
Say we have transformation f(x,y) = [2x 2y]

$jac = \begin{bmatrix}2 &0 \\ 0& 2 \end{bmatrix} \implies g^M = \begin{bmatrix}4 &0 \\ 0& 4 \end{bmatrix}$

Areas here will be four times as big when mapped. The determinant is 16, when we take the root we get 4 again. So we need the root.

----
Now, say we have transformation f(x,y,z) = [2x 2y 2z]

$jac = \begin{bmatrix}2 &0 & 0 \\ 0& 2 & 0 \\ 0 & 0 & 2 \end{bmatrix} \implies g^M = \begin{bmatrix}4 &0 & 0 \\ 0& 4 & 0 \\ 0 & 0 & 4 \end{bmatrix}$. 

Volume will be 8 times larger. The determinant is 64.


---

$$\phi(x) = ix^3$$
Jacobian: $J(x) = 3ix^2$
$g(x)(a,b) = J(x)J(x) = 9i^2x^4$

$$ \Delta_\phi f(x)= \frac{1}{\sqrt{|det(g)|}} div(\sqrt{|det(g)|} g^{-1} \nabla f)$$

$$ \Delta_\phi f (x) = \frac{1}{\sqrt{|det(g(x))|}} \sum_{i\in axes} \frac{\partial}{\partial x_i}\Big(\sqrt{|det(g(x))|} g(x)^{-1} \nabla f(x)\Big)$$

### Laplace Beltrami of simple functions 

$$\begin{matrix}
(u, v, w) \in \mathcal{U} \subset \reals^n & (x, y, z) \in \mathcal{M}
\\ 
\phi: \mathcal{U} \to \mathcal{M} & f: \mathcal{M} \to \reals
\end{matrix}
$$

$$x:=\phi(u) = iu$$
$$f(x) = x$$

$$\Delta\phi(u) = i \implies g = i^2$$

$$(\Delta_\phi f) (x) = \frac{1}{\sqrt{|i^2|}}\frac{\partial}{\partial x}(\sqrt{|i^2|}i^{-2} \frac{\partial}{\partial x} f) $$

$$= \frac{1}{i^2}\frac{\partial}{\partial x}\frac{\partial}{\partial x} f \implies (\Delta_\phi f) (x) = \Delta f /i^2$$

#### what if it's the inverse jacobian:

$$x:=\phi(u) = iu$$
$$x:=\phi^-1(u) = \frac{u}{i}$$

$$\Delta\phi^-1(u) = 1/i \implies g = 1/i^2$$

$$(\Delta_\phi f) (x) = \frac{1}{\sqrt{|1/i^2|}}\frac{\partial}{\partial x}(\sqrt{|1/i^2|}i^2 \frac{\partial}{\partial x} f) $$

$$= i^2\frac{\partial}{\partial x}\frac{\partial}{\partial x} f \implies (\Delta_\phi f) (x) = i^2\Delta f $$

### Next example:

$$x:=\phi(u) = iu^3$$
$$f(x) = x$$

$$\nabla\phi(u) = 3iu^2 \implies g = 9i^2u^4$$

$$(\Delta_\phi f) (x) = \frac{1}{\sqrt{|9i^2u^4|}}\frac{\partial}{\partial x}(\sqrt{|i^2|}i^{-2} \frac{\partial}{\partial x} f) $$

$$= \frac{1}{i^2}\frac{\partial}{\partial x}\frac{\partial}{\partial x} f \implies (\Delta_\phi f) (x) = \Delta f /i^2$$

### What the fuck is a metric tensor:

Let's say we have map $f(u, v) = [2u,2v] = [x,y]$

So $f^{-1}(x, y) = [0.5x, 0.5y] = [u,v]$

jac is $I_2 \cdot 0.5$
Metric tensor bcomes $jac^\top jac = I_2 \cdot 0.25$.


### 1D:

$f(x) = x^2$ for $x\in (0,1)$

Say I want to measure distance from [0, 1] on my manifold which is the image of $f$.

$P = f(x), x \in (0, 1)$

$T(x) = 2x$

$\int_P ||T(s)|| ds$

$\int_{(0,1)} ||T(x)|| * sqrt(|det(J*J)|)  dx$

$\int_{(0,1)} ||T(x)|| * sqrt(|4x^2|)  dx$

$\int_{(0,1)} ||T(x)|| * 2x  dx$

$\int_{(0,1)} 2x * 2x  dx$

$\int_{(0,1)} 4x*2 dx$

$\frac{4}{3}x^3 |_{0,1} = \frac{4}{3}$

---
What else could I have done? I can make f into $f(x) = (x, x^2)$ and get this arc length. Will it give the same?

$T(x) = [1, 2x]^\top$

$\int_P ||T(s)|| ds$

$\int_{(0,1)} ||[1, 2x]^\top|| * sqrt(|det(J*J)|) dx$

$J^TJ = 1 + 4x^2$

$\int_{(0,1)} ||[1, 2x]^\top|| * sqrt(1 + 4x^2) dx$

$\int_{(0,1)} 1 + 4x^2 dx = x +\$


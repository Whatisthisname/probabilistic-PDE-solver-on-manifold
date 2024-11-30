How does one numerically solve a PDE on a 2-manifold?

In this thesis, we cover how to go through the required different steps for the presented approach.

- The PDE must be converted to a system of ODEs, using the method of lines.
    - For this, we need to resort to deriving a solution only in a finite amount of spatial points. The 2-manifold must be discretized into a simplicial 2-complex. The discrete exterior calculus (DEC) formalizes this.
    - If the laplacian is used (as in the wave or heat equation), one can use the cotan-laplacian formula to get a laplace matrix. DEC motivates this formula.
    The cotan-formula requires a mesh

- We need then specify a residual term: If a solution is found, the residual should be zero. For a function u' = f(u), the residual would be R(u)(t) = f(u(t)) - u'(t). If R(u)(t) = 0 for all times t, the solution is found.

- We then choose a prior distribution over functions and their time derivative. In the simplest case, we are modeling the joint distribution of the function and its time derivative. This distribution should be informed by the problem we are solving.

- We then choose a finite amount of points t_i in time, for which we will want our solution  R(u)(t_i) = 0. 

- We then take the joint distribution of the function and its time derivative and condition it on R(u)(t_i) = 0 for all t_i. The resulting distribution of functions is the posterior distribution.

We are essentially building the joint distribution over u, u' and R(u). We are choosing a finite amount of t_i, so we can build a joint distribution. We can build the bayesian network that models this:

If we choose our prior distribution over functions carefully, it will have the markov property, namely, that the function at some time i will be independent of previous times given the function at time i-1. This is the markov property.

There are certain choices for prior distributions that will be covered.

We would also like tractability. The bayesian update requires an integral to normalize the posterior distribution. It can rarely be computed in closed form, but if we choose gaussian priors and linear gaussian likelihoods, we can compute the posterior in closed form. An algorithm that computes this in linear time in the number of time points is the Kalman filter (and smoother).

Explain Kalman filter and smoother.

With our choice of a gaussian prior state, gauss-markov prior distribution of functions, we achieve tractability and exactness. 

After filtering and smoothing, we have the posterior distribution of the state.





\frac{\text{d}}{\text{d}t}

# Ordinary Differential Equation

An ordinary differential equation is a differential equation where the solution function $u$ is a function only of one independent variable (which can usually be interpreted as a time $t$ or distance $x$ along some axis). It generally takes the form $(\frac{\text{d}}{\text{d}t}u)(t) = f(u(t))$, where f is any function. 


This notation is often abbreviated to $\frac{\text{d}}{\text{d}t}u = f(u)$, omitting the dependence of $u$ on $t$. It is implied that this relation must hold for all times $t$, usually within some interval $[0, T]$

---

$note$: This is abbreviation is similar to how one might write the equation $h(a) = f(a) + g(a)$ as $h(a) = (f+g)(a)$ or $h = f+g$.

---

The **order** of an ODE is the degree of the highest derivative-term of the solution function $u$ involved in the equation. In the example above with $f$, the order is 1. $n$'th order ODE's can be written as $\frac{\text{d}}{\text{d}t}u^{(n-1)} = u^{(n)} = f(u, u^{(0)}, \dots, u^{(n-1)})$

The regularity of $u$ satisfying the above equation will depend on the regularity of $f$ - if $f(u)=u'$ is discontinuous at some point, then $\frac{\text{d}}{\text{d}t}u$ will be so too, meaning $u \in \mathcal{C}^1((0, T))$. In this article we will however not further discuss or consider this relationship between $f$ and the regularity of $u$, as it is a statement about exact solutions, and we will only deal with the numerical approximations of these. With a probabilistic solver, regularity of $u$ will be something we can control, independent of $f$.

If $f$ is a linear function, we have in our hands a *linear ODE*, which is class of ODE's with a number of desirable properties.

### Expressing a higher-order linear ODE as a 'bigger' first-order linear ODE

A useful "trick" with linear ODEs is the following: We can always rewrite any linear $n$'th order ODE into a system of $n$ first-order ODEs. We will give a classic example of how to do this. Suppose we have second order ODE with constants $k>0$ and $l>0$
$$\frac{\text{d}^2}{\text{d}t^2}u(t) = -ku(t) - l\frac{\text{d}}{\text{d}t}u(t)\hspace{20px} \text{ or, equivalently } \hspace{20px} \frac{\text{d}}{\text{d}t}u^{(1)} = -ku^{(0)} - lu^{(1)}$$ 

---

$note$: Readers with a background in physics, control or computer graphics will immediately recognize this is the damped harmonic oscillator (or a one-dimensional spring model in layman's terms). $k$ represents the spring stiffness constant and it is the same constant that is described in Hooke's law. $l$ is a damping coefficient that will oppose changes in motion, bringing the system to a halt over time. If the reader prefers so, they can mentally rename $u=u^{(0)}$, $u^{(1)}$ and $u^{(2)}$ respectively to the more descriptive names $p(t)$ (position), $v(t)$ (velocity), and $a(t)$ (acceleration).

There are a number of other ways we could have chosen to write this equation too, such as $$\frac{\text{d}}{\text{d}t}u^{(1)} + ku + lu^{(1)}= 0 \hspace{20px} \text{ or } \hspace{20px} u = - \frac{ku^{(1)}}{l} - \frac{\frac{\text{d}}{\text{d}t}u^{(1)}}{l}$$ but in ODE literature it is most common to isolate the highest derivative term by itself on the left side. 

---

The next step is to remember the relationship $\frac{\text{d}}{\text{d}t}u^{(0)} = u^{(1)}$, which can by itself be considered a linear ODE. If we add this constraints to our equation, we get the following system of first order ODEs: 
$$\begin{align*}
\frac{\text{d}}{\text{d}t}u^{(0)} &= u^{(1)}
\\ 
\frac{\text{d}}{\text{d}t}u^{(1)} &= -ku^{(0)} - lu^{(1)}
\end{align*}$$

We are still working with two derivatives of $u$ here, but we have gotten rid of the $\frac{\text{d}^2}{\text{d}t^2}$ operator. It is convenient however to work with a single order-1 ODE rather than multiple order-1 ODEs, so we can define the vector valued function $\vec{u}(t) =\begin{pmatrix}u^{(0)}(t) \\ u^{(1)}(t)\end{pmatrix} \iff \vec{u} = \begin{pmatrix}u^{(0)} \\ u^{(1)}\end{pmatrix}$ and instead write

$$\frac{\text{d}}{\text{d}t}\begin{pmatrix}u^{(0)} \\ u^{(1)}\end{pmatrix} = \begin{pmatrix}u^{(1)} \\ -ku^{(0)} - lu^{(1)}\end{pmatrix}$$

$$ \iff \frac{\text{d}}{\text{d}t}\vec{u} = \begin{bmatrix} 0 & 1 \\ -k & -l \end{bmatrix} \vec{u}=A\vec{u}$$


The matrix $A$ captures the entire system. Note the inconspicuous '$1$' in the off-diagonal - you will get to see it many times over in the later sections.

The solution of an ODE evaluated at time $t$ is often referred to as its "state" at that time. If we include time-derivatives of $u$ into the state, the state-space has a technical name: the *phase-space*. In the above example, we converted our second order ODE with scalar state-space into a first order ODE with $\reals^2$-valued state, the phase-space containing the first time-derivative. If one is only interested in the solution itself, one can discard the derivatives after solving the ODE.

### Analytical solutions of Linear ODEs
The reader will surely before have encountered the scalar linear ODE:
$$\frac{\text{d}}{\text{d}t}u(t) = au(t) \hspace{20px} \text{ with solution: } \hspace{20px}u(t) = \text{e}^{at}u(0)$$

If we let our intuition speak for a bit, we might be inclined to extrapolate the scalar solution to the vector-valued linear ODE with solution $\vec{u}$ and coefficient matrix $A$:
$$\frac{\text{d}}{\text{d}t}\vec{u}(t) = A\vec{u}(t) \hspace{20px} \text{ with solution: } \hspace{20px}\vec{u}(t) = \text{e}^{At}\vec{u}(0)$$

This turns out to indeed be the correct analytical solution, but fewer readers will be familiar with the meaning of $\text{e}^{At}$. It is known as the *matrix exponential*, is defined only for square matrices, and can be defined as an infinite series of scaled matrix powers: $\text{e}^{At} = \sum_{n=0}^{\infty} \frac{t^n}{n!}A^n$.




# Solving problems with differential equations
One class of problems involving ODE's are initial value problems. Given the state at time $t=0$, $\vec{u}(0)$, also known as the initial state, the task is to evolve the state according to a given ordinary differential equation $\frac{\text{d}}{\text{d}t}\vec{u}(t) = f(\vec{u}(t))$ up until some final time $T>0$. The object of interest might then be $\vec{u}(T)$, the final state, or the full continuous state $\vec{u}(t)\; \forall t\in [0, T]$. This kind of problem is called an *Initial Value Problem*. 

We can however usually not solve this problem analytically, hence the need for numerical methods.


# Partial Differential Equation

# Linear Stochastic Differential Equations 

- Linear Stochastic Differential Equations (LSDE)
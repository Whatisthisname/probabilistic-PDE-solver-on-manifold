Probablistic Numerical Solvers for Differential Equations on Manifolds
Context

Probabilistic Numerics views computation as statistical inference. We can pose "numerically solving an ODE" as a question of a posterior belief updated on observations of the ODE being "solved". 
The non-vector-space property of Riemannian Manifolds makes it difficult to apply traditional methods

What?

How?


- Differential Equations Summary

    - Context: Why are they useful models? What do they model?
    - Ordinary Differential Equation
        - Linear ODEs and exact solutions
            - Matrix exponential definition
        - Examples and solutions
            - Reducing to system of first order
                - Spring example having time derivatives in state
            - Numerical solvers
                - Forward Euler
                - Backward Euler
                - Runge-Kutta
    
    - Partial Differential Equation
        - Context: Spatial-and-time relations
        - Convert a PDE to a system of ODEs using the method of lines.
            - Finite differences
                - for Gradient
                - for Laplacian
            - 1-D examples
            - This introduces approximation error: cite PNMOL
        
        - Examples and solutions
            - Heat Equation
            - Wave Equation
            - Various nonlinearities added


- Riemannian Manifolds and Differential Geometry summary
    - Context: Not all interesting spaces are vector spaces, consider the sphere. Why do we need to consider manifolds?
        - Consider problems on the surface of the earth. How would a wave travel through some material? What places will be reached at the same time? What strength will the wave have when it hits this place?

    - What is a manifold? Definition
        - Context: Just because the surface is not a vector space, does not mean that we cannot have coordinates. But these coordinates will merely be identifiers of points in space. We call these coordinates "charts" TODO: verify.
        - Example:  
            - Map of projected bell with tissot indicatrix showing how space is stretched
        - Working with projections / charts
        - Map of the earth example with tissot indicatrix
    
    - Measuring Quantities: The metric tensor
        - A smoothly varying matrix defined at each coordinate. TODO: Must be smoothly varying?
        - Measuring lengths of paths
        - Defining Geodesic
        Example:
            - Showing distance fields

    - Discussing Embeddings
        - Context: In the previous examples, the non-euclidean metric on the chart can be interpreted as stemming from a projection into R^2. Given just a metric tensor and a chart, it is not clear how to find the embedding. 
        - Intrinsic properties: lengths, angles, curvature <-> angle defect
        - Extrinsic properties: "how it sits in space", normal vector
        - Self intersections?
        


- Discrete Exterior Calculus
    - Context: An exact version of exterior calculus that tries to capture many of the properties. It is not an approximation, it is its own theory. "Exactness Hypothesis".
    Simplicial Complex, "Manifold triangle mesh"
        - where are the vertices?
    - Discrete versions of exterior derivative, star
    - Discuss barycentric dual vs circumcentric dual
    - Mainly do a very rough summary to then show what the matrices look like
    - Show how to build the laplace operator: Contrast with graph laplacian.
    - Observe: Does not need to know the vertex positions.s


    - Extrinsic Triangulation
        - Delaunay Triangulation (no other triangles in encompassing vertices)
    - Drawback: Need to know embedding.
    - Solution: Work with intrinsic representation - edge-parameterization. Has no "position" and is therefore invariant to all rigid transformations that would not result in changed distances in neighborhoods.
    - Hyperbolic Plane - How to embed? Show Shape from Metric screenshot. Can be done! But we are only interested in pairwise distances along edges. Assuming that edges are the shortest paths.
    - Intrinsic Triangluation
        - Constructing it: Starting with a reference mesh and evaluate metric tensor at points and then assume linearity and then compute distances. Will result in illegal triangles.
        - "Solution"?:
            Further subdivide illegal triangles.


- Probabilistic Numerical Solver

    - Context: We want to support a prior distribution, into which we can include information about the PDE being satisfied, to get a posterior distribution that is consistent with the PDE.

    - Linear Stochastic Differential Equations
        - Notation
            - What is a solution?
        - Examples of Priors and their properties
            - Brownian Motion
            - Integrated Wiener Process
            - Matérn Process
            - Ornstein-Uhlenbeck Process

        - Spring with white noise force example

    - Convert to State-space model if has markov process




